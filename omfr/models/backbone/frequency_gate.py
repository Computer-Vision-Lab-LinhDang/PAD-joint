"""
frequency_gate.py — FrequencyGate with variable spatial resolution

Analyses spatial-frequency content of token grids and produces a 3-band
energy vector (low / mid / high) per token.  Used as gating signal for
the Frequency-Gated MoE-FFN.

Adapts to different spatial resolutions:
    Stage 2: 28×28 = 784 tokens
    Stage 3: 14×14 = 196 tokens

I/O
---
    Input:  tokens (B, N, D), spatial_h, spatial_w
    Output: gate_input (B, N, 3) — normalized [low, mid, high] energies
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_band_masks(
    h: int, w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build radial frequency-band masks for an (h × w) 2-D FFT spectrum.

    Band definitions (radial distance from DC):
        low  : dist < max_dim / 4   (ridge flow, global structure)
        mid  : max_dim/4 ≤ dist < max_dim/2  (minutiae, incipient ridges)
        high : dist ≥ max_dim / 2   (pores, micro-texture, spoof artifacts)
    """
    fy = torch.fft.fftfreq(h) * h
    fx = torch.fft.fftfreq(w) * w
    dist = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()

    max_dim = max(h, w)
    low_thresh = max_dim / 4.0
    high_thresh = max_dim / 2.0

    low_mask = dist < low_thresh
    mid_mask = (dist >= low_thresh) & (dist < high_thresh)
    high_mask = dist >= high_thresh
    return low_mask.float(), mid_mask.float(), high_mask.float()


class FrequencyGate(nn.Module):
    """
    Spatial-frequency band-energy gate.

    Reshapes (B, N, D) tokens to a spatial grid, applies 2-D FFT,
    and accumulates power within three concentric frequency rings.

    No learnable parameters — the MoE gate projection learns to
    interpret these energy features.

    Caches band masks per (h, w) resolution for efficiency.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(3)
        self._mask_cache: dict[tuple[int, int], tuple] = {}

    def _get_masks(
        self, h: int, w: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (h, w)
        if key not in self._mask_cache or self._mask_cache[key][0].device != device:
            lo, mid, hi = _make_band_masks(h, w)
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
            tokens:    (B, N, D) where N = spatial_h × spatial_w
            spatial_h: height of the spatial grid
            spatial_w: width of the spatial grid

        Returns:
            gate_input: (B, N, 3)  LayerNorm-normalised band energies
        """
        B, N, D = tokens.shape

        # Reshape to spatial grid
        spatial = tokens.reshape(B, spatial_h, spatial_w, D).permute(0, 3, 1, 2)

        # 2-D FFT (kept in frequency space for band filtering)
        fft_out = torch.fft.fft2(spatial, norm="ortho")  # (B, D, H, W) complex

        # Band masks
        low_mask, mid_mask, high_mask = self._get_masks(
            spatial_h, spatial_w, tokens.device,
        )

        # Per-token local band energy: mask in freq-domain, iFFT back,
        # take magnitude², then pool across embedding dim. This preserves
        # spatial locality so each token gets its OWN gating signal based
        # on the frequency content at its grid position.
        low_spatial  = torch.fft.ifft2(fft_out * low_mask,  norm="ortho").abs() ** 2
        mid_spatial  = torch.fft.ifft2(fft_out * mid_mask,  norm="ortho").abs() ** 2
        high_spatial = torch.fft.ifft2(fft_out * high_mask, norm="ortho").abs() ** 2

        low_e  = low_spatial.mean(dim=1).flatten(1)   # (B, N)
        mid_e  = mid_spatial.mean(dim=1).flatten(1)
        high_e = high_spatial.mean(dim=1).flatten(1)

        gate_input = torch.stack([low_e, mid_e, high_e], dim=-1)  # (B, N, 3)
        gate_input = self.norm(gate_input)
        return gate_input
