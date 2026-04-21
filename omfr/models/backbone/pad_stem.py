"""
pad_stem.py — Dedicated PAD feature extractor.

A parallel CNN that reads the 8-channel Gabor response (from a PAD-
specific Gabor bank) and produces a 256-D vector consumed by PADHead.
Lives OUTSIDE the TinyViT backbone so the PAD objective has a trainable
path of its own; the backbone stays shaped exclusively by the identity
objective (see OMFRModule._run_pad).

Pipeline (B, 8, 224, 224)
    -> stride-2 DS block  (8   ->  32, 224 -> 112)
    -> stride-2 DS block  (32  ->  64, 112 ->  56)
    -> stride-2 DS block  (64  -> 128,  56 ->  28)
    -> stride-2 DS block  (128 -> 256,  28 ->  14)
    -> GAP -> (B, 256)

Each block is (stride-2 3x3 conv) -> GN -> GELU
             -> depthwise 3x3 refinement -> GN -> GELU.
Depthwise refinement lets each channel learn its own spatial pattern at
the new scale without adding another cross-channel projection. GroupNorm
(not BatchNorm) is used for the same reason as in PADHead: cross-sensor
batches cause BN running-stat drift between train and test.

v2 (4 blocks, 256-D out): v1 was capacity-constrained — 3 blocks gave a
receptive field of ~12 px and a 128-D bottleneck, too narrow for global
spoof cues like silicone sheen or print grain patterns that span larger
regions. Adding a 4th stride-2 block doubles the effective RF to ~24 px
and widens the output to 256-D, roughly doubling pad_stem params
(170K -> ~400K) while still being far smaller than the backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PADStem(nn.Module):
    """Stride-2 DS-conv stem producing a 256-D liveness feature vector."""

    def __init__(
        self,
        in_ch: int = 8,
        dims: tuple[int, int, int, int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = dims
        self.stem = nn.Sequential(
            self._ds_block(in_ch, c1),   # 224 -> 112
            self._ds_block(c1, c2),      # 112 -> 56
            self._ds_block(c2, c3),      # 56  -> 28
            self._ds_block(c3, c4),      # 28  -> 14
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = c4

    @staticmethod
    def _ds_block(in_ch: int, out_ch: int) -> nn.Sequential:
        gn = PADStem._pick_groups(out_ch)
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.GroupNorm(gn, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,
                      groups=out_ch, bias=False),
            nn.GroupNorm(gn, out_ch),
            nn.GELU(),
        )

    @staticmethod
    def _pick_groups(channels: int) -> int:
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 8, 224, 224) — PAD Gabor-enhanced input.
        Returns:
            (B, 256) — liveness feature vector.
        """
        x = self.stem(x)            # (B, 256, 14, 14)
        return self.gap(x).flatten(1)
