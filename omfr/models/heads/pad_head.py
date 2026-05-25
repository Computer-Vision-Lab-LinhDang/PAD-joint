"""
pad_head.py - MS-TAH PAD Branch Head.

MS-TAH (Multi-Scale Texture Aggregation Head) is intentionally narrower in
feature sources than the previous PAD head: it consumes only early spatial
backbone stages and does not fuse PADStem vectors or MoE routing statistics.
That removes the most obvious shortcut path for sensor/year/material routing
patterns while preserving local texture structure until the last projection.

Input:
    stage1_feat: (B, 64, 56, 56)
    stage2_feat: (B, 128, 28, 28)

Output:
    pad_features:  (B, 192) for sensor adversarial / PAD supervision
    pad_embedding: (B, 64)  L2-normalized, for orthogonality / retrieval
    pad_logit:     (B, 1)   raw logit for BCE/focal PAD loss
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_groups(channels: int, preferred: int = 8) -> int:
    """Pick a GroupNorm group count that divides the channel count."""
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TextureBlock(nn.Module):
    """
    Texture-sensitive feature transform.

    Depthwise large-kernel convolution captures local texture statistics;
    pointwise convolution mixes channels. Stage 1 uses stride 2 to align
    56x56 features to Stage 2's 28x28 grid.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dw_kernel: int = 7,
    ) -> None:
        super().__init__()
        padding = dw_kernel // 2
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            dw_kernel,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(_pick_groups(out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return self.act(x)


class FusionBlock(nn.Module):
    """
    Spatial mixer for concatenated Stage 1 + Stage 2 texture features.

    A 1x1 reduction controls channel count, then a 3x3 convolution mixes local
    spatial context before aggregation.
    """

    def __init__(self, in_ch: int = 256, out_ch: int = 192) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm1 = nn.GroupNorm(_pick_groups(out_ch), out_ch)
        self.act1 = nn.GELU()
        self.spatial = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_pick_groups(out_ch), out_ch)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.reduce(x)))
        x = self.act2(self.norm2(self.spatial(x)))
        return x


class MultiScaleAggregator(nn.Module):
    """
    Aggregate texture maps at global, mid, and local spatial granularities.

    Global: GAP -> contextual texture statistics
    Mid:    28x28 -> 14x14 -> GAP
    Local:  28x28 -> 7x7  -> GAP

    Output is a concatenation of three embed_dim vectors.
    """

    def __init__(self, in_ch: int = 192, embed_dim: int = 192) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, embed_dim, 1, bias=False),
            nn.GroupNorm(_pick_groups(embed_dim), embed_dim),
            nn.GELU(),
        )
        self.mid_pool = nn.AdaptiveAvgPool2d(1)

        self.local_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False),
            nn.GroupNorm(_pick_groups(in_ch), in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, embed_dim, 1, bias=False),
            nn.GroupNorm(_pick_groups(embed_dim), embed_dim),
            nn.GELU(),
        )
        self.local_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        global_vec = self.global_pool(x).view(batch, -1)
        mid_vec = self.mid_pool(self.mid_conv(x)).view(batch, -1)
        local_vec = self.local_pool(self.local_conv(x)).view(batch, -1)
        return torch.cat([global_vec, mid_vec, local_vec], dim=-1)


class PADHead(nn.Module):
    """
    MS-TAH: Multi-Scale Texture Aggregation Head for fingerprint PAD.

    The constructor keeps legacy argument names used by OMFRModule/configs, but
    the head architecture only depends on stage1/stage2 spatial features.
    """

    PAD_FEATURES_DIM = 192
    PAD_EMBEDDING_DIM = 64
    USES_PAD_STEM = False
    USES_ROUTING_STATS = False

    def __init__(
        self,
        stage1_dim: int = 64,
        stage2_dim: int = 128,
        pad_stem_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 1,
        proj_dropout: float | None = None,
        align_ch: int = 128,
        fusion_ch: int | None = None,
        embedding_dim: int | None = None,
        stage1_dw_kernel: int = 9,
        stage2_dw_kernel: int = 5,
        pad_features_dim: int | None = None,
        pad_embedding_dim: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        del pad_stem_dim  # Kept only for compatibility with older configs.
        if int(num_classes) != 1:
            raise ValueError(
                "MS-TAH PADHead is a binary PAD head and must emit one logit; "
                f"got num_classes={num_classes}."
            )

        if fusion_ch is None:
            fusion_ch = pad_features_dim or self.PAD_FEATURES_DIM
        if embedding_dim is None:
            embedding_dim = pad_embedding_dim or self.PAD_EMBEDDING_DIM
        if proj_dropout is None:
            proj_dropout = dropout

        self.PAD_FEATURES_DIM = int(fusion_ch)
        self.PAD_EMBEDDING_DIM = int(embedding_dim)

        self.tb1 = TextureBlock(
            in_ch=int(stage1_dim),
            out_ch=int(align_ch),
            stride=2,
            dw_kernel=int(stage1_dw_kernel),
        )
        self.tb2 = TextureBlock(
            in_ch=int(stage2_dim),
            out_ch=int(align_ch),
            stride=1,
            dw_kernel=int(stage2_dw_kernel),
        )

        self.fusion = FusionBlock(
            in_ch=int(align_ch) * 2,
            out_ch=self.PAD_FEATURES_DIM,
        )
        self.aggregator = MultiScaleAggregator(
            in_ch=self.PAD_FEATURES_DIM,
            embed_dim=self.PAD_FEATURES_DIM,
        )

        agg_dim = self.PAD_FEATURES_DIM * 3
        self.proj = nn.Sequential(
            nn.Linear(agg_dim, self.PAD_FEATURES_DIM),
            nn.LayerNorm(self.PAD_FEATURES_DIM),
            nn.GELU(),
        )

        self.embedding_branch = nn.Sequential(
            nn.Linear(self.PAD_FEATURES_DIM, self.PAD_FEATURES_DIM // 2),
            nn.GELU(),
            nn.Linear(self.PAD_FEATURES_DIM // 2, self.PAD_EMBEDDING_DIM),
        )

        self.logit_branch = nn.Sequential(
            nn.Linear(self.PAD_FEATURES_DIM, self.PAD_FEATURES_DIM // 2),
            nn.GELU(),
            nn.Dropout(float(proj_dropout)),
            nn.Linear(self.PAD_FEATURES_DIM // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s1 = features["stage1_feat"]
        s2 = features["stage2_feat"]
        self._validate_stage_shapes(s1, s2)

        s1_aligned = self.tb1(s1)
        s2_aligned = self.tb2(s2)

        fused = torch.cat([s1_aligned, s2_aligned], dim=1)
        mixed = self.fusion(fused)
        agg = self.aggregator(mixed)
        pad_features = self.proj(agg)

        embedding = self.embedding_branch(pad_features)
        pad_embedding = F.normalize(embedding, p=2, dim=-1)
        pad_logit = self.logit_branch(pad_features)

        return {
            "pad_features": pad_features,
            "pad_embedding": pad_embedding,
            "pad_logit": pad_logit,
        }

    @staticmethod
    def _validate_stage_shapes(s1: torch.Tensor, s2: torch.Tensor) -> None:
        if s1.ndim != 4 or s2.ndim != 4:
            raise ValueError(
                "PADHead expects 4D stage tensors: "
                f"stage1_feat={tuple(s1.shape)}, stage2_feat={tuple(s2.shape)}"
            )
        if s1.shape[0] != s2.shape[0]:
            raise ValueError(
                "PADHead stage tensors must have same batch size: "
                f"stage1={s1.shape[0]}, stage2={s2.shape[0]}"
            )
        if s1.shape[-2] != 2 * s2.shape[-2] or s1.shape[-1] != 2 * s2.shape[-1]:
            raise ValueError(
                "Expected stage1 spatial size to be 2x stage2 spatial size, "
                f"got stage1={tuple(s1.shape[-2:])}, "
                f"stage2={tuple(s2.shape[-2:])}."
            )
