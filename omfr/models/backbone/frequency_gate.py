"""
frequency_gate.py — FrequencyGate

Analyses the spatial-frequency content of ViT token grids and produces
a 3-band energy vector (low / mid / high) per token.  This vector is
used as the gating signal for the Frequency-Gated MoE-FFN.

I/O
---
    Input:  tokens  (B, 196, 192)
    Output: gate_input (B, 196, 3)   — [low_energy, mid_energy, high_energy]
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_band_masks(grid: int = 14) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build binary masks for low / mid / high frequency bands on a (grid x grid)
    2-D FFT magnitude spectrum.

    Band definitions (distance from DC component):
        low  : freq_dist <  grid // 4          (~coarse ridge structure)
        mid  : grid // 4 <= freq_dist < grid//2  (minutiae-scale)
        high : freq_dist >= grid // 2           (pores, micro-texture -> PAD-critical)

    Returns
    -------
    low_mask, mid_mask, high_mask : BoolTensor each (grid, grid)
    """
    fy = torch.fft.fftfreq(grid) * grid
    fx = torch.fft.fftfreq(grid) * grid
    dist = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()   # (grid, grid)

    low_thresh  = grid / 4.0
    high_thresh = grid / 2.0

    low_mask  = dist <  low_thresh
    mid_mask  = (dist >= low_thresh) & (dist < high_thresh)
    high_mask = dist >= high_thresh
    return low_mask, mid_mask, high_mask


class FrequencyGate(nn.Module):
    """
    Spatial-frequency band-energy gate.

    Reshapes the (B, 196, 192) token tensor back to a (B, 192, 14, 14)
    spatial grid, applies a 2-D FFT over the spatial dimensions, and
    accumulates power within three concentric frequency rings:

        low  (< 3.5 cycles) — coarse structure
        mid  (3.5-7 cycles) — minutiae scale
        high (>= 7 cycles)  — pores / micro-texture (most PAD-discriminative)

    The three per-token energies are stacked and passed through a
    LayerNorm to produce the routing gate signal consumed by
    FreqGatedMoEFFN.

    I/O
    ---
        Input:  tokens     (B, 196, 192)
        Output: gate_input (B, 196, 3)    normalised [low, mid, high] energies

    Band definitions (on 14x14 frequency grid, distance from DC):
        low  : freq_dist < 14//4  = 3.5 cycles  (~coarse structure)
        mid  : 3.5 <= freq_dist < 7  (minutiae-scale)
        high : freq_dist >= 7         (pores, micro-texture -> PAD-critical)

    Forward:
        spatial = tokens.reshape(B, 14, 14, 192).permute(0, 3, 1, 2)  # (B, 192, 14, 14)
        power   = fft2(spatial, norm='ortho').abs() ** 2
        [low_energy, mid_energy, high_energy] = sum over D dimension per band mask
        gate_input = LayerNorm(stack([low, mid, high], dim=-1))        # (B, 196, 3)
    """

    def __init__(self, grid: int = 14, embed_dim: int = 192) -> None:
        """
        Parameters
        ----------
        grid : int
            Spatial grid size; 196 = grid x grid tokens.  Default 14.
        embed_dim : int
            Token embedding dimension.  Default 192.
        """
        super().__init__()
        self.grid = grid
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(3)

        low_mask, mid_mask, high_mask = _make_band_masks(grid)
        self.register_buffer("low_mask",  low_mask.float())
        self.register_buffer("mid_mask",  mid_mask.float())
        self.register_buffer("high_mask", high_mask.float())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : Tensor (B, 196, 192)

        Returns
        -------
        gate_input : Tensor (B, 196, 3)
            LayerNorm-normalised [low_energy, mid_energy, high_energy] per token.
        """
        B, N, D = tokens.shape  # N = 196, D = 192
        g = self.grid           # 14

        # Reshape to spatial grid: (B, D, g, g)
        spatial = tokens.reshape(B, g, g, D).permute(0, 3, 1, 2)  # (B, D, g, g)

        # 2-D FFT over spatial dims
        fft_out = torch.fft.fft2(spatial, norm="ortho")
        power   = fft_out.abs() ** 2                                # (B, D, g, g)

        # Accumulate band energies; masks broadcast over (B, D, g, g)
        low_energy  = (power * self.low_mask).sum(dim=(-2, -1))    # (B, D)
        mid_energy  = (power * self.mid_mask).sum(dim=(-2, -1))    # (B, D)
        high_energy = (power * self.high_mask).sum(dim=(-2, -1))   # (B, D)

        # Sum across embedding dim; expand back so every token shares the gate
        low_e  = low_energy.sum(dim=1, keepdim=True).expand(B, N)   # (B, 196)
        mid_e  = mid_energy.sum(dim=1, keepdim=True).expand(B, N)
        high_e = high_energy.sum(dim=1, keepdim=True).expand(B, N)

        gate_input = torch.stack([low_e, mid_e, high_e], dim=-1)   # (B, 196, 3)
        gate_input = self.layer_norm(gate_input)                    # (B, 196, 3)
        return gate_input
