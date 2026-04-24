"""
frequency_gate.py — FrequencyGate with ridge-aware spatial bands (v2)

Analyses spatial-frequency content of token grids and produces a 3-band
energy vector (sub-ridge / ridge / super-ridge) per token. Used as the
gating signal for the Frequency-Gated MoE-FFN.

Rationale (TASK_05)
-------------------
v1 used generic grid-size-relative thresholds (`max_dim/4`, `max_dim/2`)
which mapped badly onto fingerprint physics: the actual ridge peak at
28×28 sits around radius ~11-12 and at 14×14 around ~5-6, so the ridge
energy was split across the "mid" and "high" bands. The MoE gate_proj
could not learn meaningful routing from that.

v2 pegs the bands to the MEASURED ridge frequency at each resolution
and splits into three semantically distinct concepts:

    Band 0  sub-ridge      dist < 0.6 * ridge
                           (global structure, finger boundary, illumination)
    Band 1  ridge          0.6 * ridge <= dist <= 1.6 * ridge
                           (the actual ridge-valley pattern)
    Band 2  super-ridge    dist > 1.6 * ridge
                           (minutiae discontinuities, pores, sensor noise,
                            spoof-material micro-texture)

Defaults were calibrated on NIST SD302 at 500 DPI resized to 224×224:

    ridge_freq_28 = 11.5
    ridge_freq_14 =  5.8

If the upstream resize pipeline differs (different input DPI or crop
policy), run `scripts/calibrate_ridge_freq.py` and override via the
`ridge_freq` dict.

I/O
---
    Input:  tokens (B, N, D), spatial_h, spatial_w
    Output: gate_input (B, N, 3)  LayerNorm-normalised band energies
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


DEFAULT_RIDGE_FREQ: Dict[int, float] = {
    28: 11.5,
    14: 5.8,
}


def _make_band_masks(
    h: int,
    w: int,
    ridge_freq: float,
    low_frac: float = 0.6,
    high_frac: float = 1.6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build radial frequency-band masks for an (h x w) 2-D FFT spectrum.

    Uses `torch.fft.fftfreq(n) * n` so distances are in integer cycles
    per grid. Band edges are scaled to the measured ridge frequency so
    the ridge always falls inside the middle band regardless of grid
    size.
    """
    fy = torch.fft.fftfreq(h) * h
    fx = torch.fft.fftfreq(w) * w
    dist = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()

    low_thresh = low_frac * ridge_freq
    high_thresh = high_frac * ridge_freq

    low_mask = dist < low_thresh
    mid_mask = (dist >= low_thresh) & (dist <= high_thresh)
    high_mask = dist > high_thresh
    return low_mask.float(), mid_mask.float(), high_mask.float()


class FrequencyGate(nn.Module):
    """
    Ridge-aware spatial-frequency band-energy gate.

    Reshapes (B, N, D) tokens to a spatial grid, applies 2-D FFT, and
    accumulates per-token power within three concentric rings whose
    boundaries are scaled to the measured ridge frequency for that
    resolution.

    No learnable parameters aside from a final LayerNorm — the MoE gate
    projection downstream learns how to interpret these energy features.

    Band masks are cached per (h, w, device). Across a forward pass this
    is at most 2 entries (stages s2=28 and s3=14).

    Args:
        ridge_freq:  optional {grid_size: ridge_frequency_in_cycles_per_grid}.
                     Defaults to DEFAULT_RIDGE_FREQ (NIST-SD302 calibration).
        low_frac:    band 0 upper edge, relative to ridge (default 0.6).
        high_frac:   band 1 upper edge, relative to ridge (default 1.6).
    """

    def __init__(
        self,
        ridge_freq: Optional[Dict[int, float]] = None,
        low_frac: float = 0.6,
        high_frac: float = 1.6,
    ) -> None:
        super().__init__()
        self.ridge_freq: Dict[int, float] = dict(DEFAULT_RIDGE_FREQ)
        if ridge_freq:
            self.ridge_freq.update({int(k): float(v) for k, v in ridge_freq.items()})
        self.low_frac = float(low_frac)
        self.high_frac = float(high_frac)

        self.norm = nn.LayerNorm(3)
        self._mask_cache: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _resolve_ridge_freq(self, h: int, w: int) -> float:
        """Pick the calibrated ridge frequency for this resolution.

        Exact match preferred; otherwise fall back to the closest known
        grid size and scale linearly (ridge frequency is proportional to
        grid resolution under uniform downsampling).
        """
        grid = max(h, w)
        if grid in self.ridge_freq:
            return self.ridge_freq[grid]
        nearest = min(self.ridge_freq.keys(), key=lambda g: abs(g - grid))
        return self.ridge_freq[nearest] * (grid / nearest)

    def _get_masks(
        self, h: int, w: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (h, w, device)
        if key not in self._mask_cache:
            ridge = self._resolve_ridge_freq(h, w)
            lo, mid, hi = _make_band_masks(h, w, ridge, self.low_frac, self.high_frac)
            self._mask_cache[key] = (lo.to(device), mid.to(device), hi.to(device))
        return self._mask_cache[key]

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_h: int,
        spatial_w: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens:    (B, N, D) where N = spatial_h * spatial_w
            spatial_h: height of the spatial grid
            spatial_w: width of the spatial grid

        Returns:
            gate_input: (B, N, 3)  LayerNorm-normalised band energies
        """
        B, N, D = tokens.shape
        in_dtype = tokens.dtype

        # cuFFT in fp16 requires power-of-two signal sizes. FastViT stage
        # grids are 28 and 14 (non-POT), so under AMP (16-mixed) we must
        # promote to fp32 for the FFT path and cast the output back.
        spatial = (
            tokens.reshape(B, spatial_h, spatial_w, D)
            .permute(0, 3, 1, 2)
            .float()
        )

        fft_out = torch.fft.fft2(spatial, norm="ortho")  # (B, D, H, W) complex

        low_mask, mid_mask, high_mask = self._get_masks(
            spatial_h, spatial_w, tokens.device,
        )

        low_spatial  = torch.fft.ifft2(fft_out * low_mask,  norm="ortho").abs() ** 2
        mid_spatial  = torch.fft.ifft2(fft_out * mid_mask,  norm="ortho").abs() ** 2
        high_spatial = torch.fft.ifft2(fft_out * high_mask, norm="ortho").abs() ** 2

        low_e  = low_spatial.mean(dim=1).flatten(1)   # (B, N)
        mid_e  = mid_spatial.mean(dim=1).flatten(1)
        high_e = high_spatial.mean(dim=1).flatten(1)

        gate_input = torch.stack([low_e, mid_e, high_e], dim=-1).to(in_dtype)
        gate_input = self.norm(gate_input)
        return gate_input
