"""
frequency_gate.py — FrequencyGate (v3) with ridge-aware band thresholds

Analyzes spatial-frequency content of token grids and produces a 3-band
energy vector (sub-ridge / ridge / super-ridge) per token. Used as
gating signal for the Frequency-Gated MoE-FFN.

v3 rationale (TASK_05):
    v2 used band thresholds of max_dim/4 and max_dim/2, which don't
    align with fingerprint ridge frequency. At typical 500 DPI after
    resize to 224x224 and stride-8/16 downsampling:

        Stage s2 (28x28 grid): ridge frequency around radial index 11-12
        Stage s3 (14x14 grid): ridge frequency around radial index 5-6

    v2's "low" band (< max_dim/4) cut off ridge energy entirely; "mid"
    and "high" both contained ridge harmonics mixed with pores. The
    MoE gate learned nothing from the resulting feature mess — visible
    as a flat balance_loss and uniform expert routing.

v3 bands:
    low  (sub-ridge):   dist <= 0.6 * ridge_freq    -> global structure
    mid  (ridge):       0.6 * ridge < dist <= 1.6 * ridge   -> ridge pattern
    high (super-ridge): dist > 1.6 * ridge          -> pores, spoof texture

Adapts to different spatial resolutions. Defaults calibrated on NIST
SD302 — re-measure (see scripts/calibrate_ridge_freq.py) for different
datasets.

I/O
---
    Input:  tokens (B, N, D), spatial_h, spatial_w
    Output: gate_input (B, N, 3) -- normalized [low, mid, high] energies
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


# Default ridge frequencies (measured on NIST SD302 at 500 DPI, 224x224)
# keyed by grid size. If your preprocessing differs, re-measure.
DEFAULT_RIDGE_FREQ: Dict[int, float] = {
    28: 11.5,   # stage s2 grid
    14:  5.8,   # stage s3 grid
    56: 22.5,   # stage s1 grid (in case something hooks here)
    7:   3.0,   # stage s4 grid (in case something hooks here)
}


def _make_ridge_band_masks(
    h: int, w: int,
    ridge_freq: float,
    low_mult: float = 0.6,
    high_mult: float = 1.6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build radial frequency-band masks anchored on fingerprint ridge frequency.

    Args:
        h, w:        spatial grid size.
        ridge_freq:  expected radial index where ridge energy peaks.
        low_mult:    upper bound of "low" band as fraction of ridge_freq.
        high_mult:   lower bound of "high" band as fraction of ridge_freq.

    Returns:
        (low_mask, mid_mask, high_mask), each float tensor of shape (h, w).
    """
    fy = torch.fft.fftfreq(h) * h
    fx = torch.fft.fftfreq(w) * w
    dist = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()

    low_cut = low_mult * ridge_freq
    high_cut = high_mult * ridge_freq

    low_mask  = dist <= low_cut
    mid_mask  = (dist > low_cut) & (dist <= high_cut)
    high_mask = dist > high_cut

    return low_mask.float(), mid_mask.float(), high_mask.float()


class FrequencyGate(nn.Module):
    """Spatial-frequency band-energy gate with ridge-aware thresholds.

    Reshapes (B, N, D) tokens to a spatial grid, applies per-band bandpass
    via FFT/iFFT, and returns per-token magnitude-squared energy pooled
    across the embedding dimension.

    No learnable weights on the filter side; all learning is in the
    downstream ``gate_proj`` of FreqGatedMoEFFN. A final LayerNorm over
    the 3-band vector makes the magnitude comparable across images.

    Args:
        ridge_freq:  optional dict mapping grid size to measured ridge
                     frequency. Defaults to DEFAULT_RIDGE_FREQ.
        low_mult:    band-edge multiplier for sub-ridge / ridge boundary.
        high_mult:   band-edge multiplier for ridge / super-ridge boundary.
    """

    def __init__(
        self,
        ridge_freq: Dict[int, float] | None = None,
        low_mult: float = 0.6,
        high_mult: float = 1.6,
    ) -> None:
        super().__init__()
        self.ridge_freq = dict(DEFAULT_RIDGE_FREQ)
        if ridge_freq is not None:
            self.ridge_freq.update(ridge_freq)
        self.low_mult = float(low_mult)
        self.high_mult = float(high_mult)

        self.norm = nn.LayerNorm(3)
        self._mask_cache: Dict[Tuple[int, int, torch.device], Tuple] = {}

    def _get_ridge_freq(self, h: int, w: int) -> float:
        """Look up ridge frequency for this grid. Fall back to interpolation
        from the nearest known size if not in the table.
        """
        key = max(h, w)
        if key in self.ridge_freq:
            return self.ridge_freq[key]
        # Linear interpolation by grid size — ridge frequency scales with
        # grid resolution.
        known = sorted(self.ridge_freq.keys())
        if key < known[0]:
            return self.ridge_freq[known[0]] * key / known[0]
        if key > known[-1]:
            return self.ridge_freq[known[-1]] * key / known[-1]
        # key is between two known sizes
        for lo, hi in zip(known, known[1:]):
            if lo <= key <= hi:
                t = (key - lo) / (hi - lo)
                return (1 - t) * self.ridge_freq[lo] + t * self.ridge_freq[hi]
        return self.ridge_freq[known[-1]]   # unreachable, but keeps mypy happy

    def _get_masks(
        self, h: int, w: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (h, w, str(device))
        cached = self._mask_cache.get(key)
        if cached is None:
            ridge = self._get_ridge_freq(h, w)
            lo, mid, hi = _make_ridge_band_masks(
                h, w, ridge_freq=ridge,
                low_mult=self.low_mult, high_mult=self.high_mult,
            )
            cached = (lo.to(device), mid.to(device), hi.to(device))
            self._mask_cache[key] = cached
        return cached

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_h: int,
        spatial_w: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens:    (B, N, D) where N = spatial_h * spatial_w.
            spatial_h: height of the spatial grid.
            spatial_w: width of the spatial grid.

        Returns:
            gate_input: (B, N, 3) LayerNorm-normalized band energies.
        """
        B, N, D = tokens.shape

        # (B, N, D) -> (B, D, H, W)
        spatial = tokens.reshape(B, spatial_h, spatial_w, D).permute(0, 3, 1, 2)

        # 2-D FFT (complex). Use orthonormal norm so mag^2 preserves
        # Parseval's identity at a meaningful scale.
        fft_out = torch.fft.fft2(spatial, norm="ortho")  # (B, D, H, W) complex

        low_mask, mid_mask, high_mask = self._get_masks(
            spatial_h, spatial_w, tokens.device,
        )

        # Bandpass + iFFT to recover per-token local band energy. Take
        # mag^2 and pool across the embedding dim. This gives a gating
        # signal localized in space AND in frequency — each token knows
        # whether ITS neighborhood has sub-ridge / ridge / super-ridge
        # content.
        low_spatial  = torch.fft.ifft2(fft_out * low_mask,  norm="ortho").abs() ** 2
        mid_spatial  = torch.fft.ifft2(fft_out * mid_mask,  norm="ortho").abs() ** 2
        high_spatial = torch.fft.ifft2(fft_out * high_mask, norm="ortho").abs() ** 2

        low_e  = low_spatial.mean(dim=1).flatten(1)   # (B, N)
        mid_e  = mid_spatial.mean(dim=1).flatten(1)
        high_e = high_spatial.mean(dim=1).flatten(1)

        gate_input = torch.stack([low_e, mid_e, high_e], dim=-1)  # (B, N, 3)
        gate_input = self.norm(gate_input)
        return gate_input

    def extra_repr(self) -> str:
        return (
            f"ridge_freq={self.ridge_freq}, "
            f"low_mult={self.low_mult}, high_mult={self.high_mult}"
        )
