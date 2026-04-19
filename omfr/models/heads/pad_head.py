"""
pad_head.py — PAD Branch Head (v3: Depthwise-Separable Conv + GAP)

Input: {
    'stage1_feat':       (B, 64, 56, 56),
    'stage2_feat':       (B, 128, 28, 28),
    'routing_stats_s2':  {'expert_weights': (B, 784, 4), 'token_entropy': (B, 784)},
    'routing_stats_s3a': {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
    'routing_stats_s3b': {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
}

Output: {
    'pad_embedding': (B, 32),    # L2-normalized, for SupCon + orthogonality
    'pad_logit':     (B, 1),     # Raw logit for BCE
    'pad_features':  (B, 128),   # Pre-projection features for SupCon
}

v3 rationale:
    The previous (v2) feature path used AdaptiveAvgPool2d(7) + Flatten +
    Linear(3136/6272 -> 256) per stage. Two failure modes:

      1. Param explosion (~2.4M in proj_s1 + proj_s2 alone) driving severe
         overfit — BCE tracks ~0.03 on train but EER stays ~50% on held-out
         sensors.
      2. The 8x8 spatial averaging destroys the very micro-texture that
         separates live vs. spoof (print noise, pores, silicon grain).

    v3 replaces both Linear projections with depthwise-separable conv
    blocks (DW3x3 + PW1x1 + DW3x3) that KEEP full spatial resolution
    (56x56 / 28x28) and rely on Global Average Pooling at the very END
    to produce the per-stage 128-D vector. This preserves translation
    invariance and high-frequency detail while cutting the feature-path
    param budget by ~40x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PADConvBlock(nn.Module):
    """
    Depthwise-separable conv block preserving spatial resolution.

    Flow: DW3x3 -> GN -> GELU -> PW1x1 -> GN -> GELU -> DW3x3 -> GN -> GELU

    Uses GroupNorm instead of BatchNorm: PAD data spans multiple sensors
    and datasets, so BN running stats drift between train and test
    distributions. GN is batch-size and distribution independent.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        gn_in = self._pick_groups(in_ch)
        gn_out = self._pick_groups(out_ch)

        self.block = nn.Sequential(
            # DW 3x3 on input channels
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                      groups=in_ch, bias=False),
            nn.GroupNorm(gn_in, in_ch),
            nn.GELU(),
            # PW 1x1 to change channel count
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(gn_out, out_ch),
            nn.GELU(),
            # DW 3x3 to refine on new channel basis
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,
                      groups=out_ch, bias=False),
            nn.GroupNorm(gn_out, out_ch),
            nn.GELU(),
        )

    @staticmethod
    def _pick_groups(channels: int) -> int:
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PADHead(nn.Module):
    """
    Multi-scale feature aggregation + MoE routing stats fusion.

    Architecture (v3):
        -- Feature path (spatial-preserving) --
        stage1_feat (B,64,56,56)  -> PADConvBlock(64->128)  -> GAP -> (B,128)
        stage2_feat (B,128,28,28) -> PADConvBlock(128->128) -> GAP -> (B,128)
        -> concat -> (B, 256)

        -- Routing path (8 features per MoE layer x 3 layers = 24) --
        Per MoE layer:
            expert_weights: mean over tokens -> (B, 4)
            token_entropy:  [mean, std, max, min] -> (B, 4)
        -> concat all 3 layers -> (B, 24)

        -- Fusion --
        concat(feature_256, routing_24) -> (B, 280)
        -> LayerNorm -> MLP -> pad_features (B, 128)

        -- Two PARALLEL output branches --
        pad_features -> Linear(128, 32) -> L2Norm -> pad_embedding
        pad_features -> Linear(128, 1) -> pad_logit
    """

    FEAT_CH_PER_STAGE = 128
    ROUTE_DIM_PER_LAYER = 8   # 4 expert mean + 4 entropy stats
    NUM_MOE_LAYERS = 3
    ROUTE_DIM = ROUTE_DIM_PER_LAYER * NUM_MOE_LAYERS   # 24
    PAD_FEATURES_DIM = 128
    PAD_EMBEDDING_DIM = 32

    def __init__(
        self,
        stage1_dim: int = 64,
        stage2_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Feature path: depthwise-separable blocks keep spatial layout
        # intact; GAP is deferred to the very end of each stage.
        self.block_s1 = PADConvBlock(stage1_dim, self.FEAT_CH_PER_STAGE)
        self.block_s2 = PADConvBlock(stage2_dim, self.FEAT_CH_PER_STAGE)
        self.gap = nn.AdaptiveAvgPool2d(1)

        feat_dim = self.FEAT_CH_PER_STAGE * 2              # 256
        fusion_in = feat_dim + self.ROUTE_DIM              # 280

        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.PAD_FEATURES_DIM),
        )

        # Parallel output branches from pad_features
        self.embedding_proj = nn.Linear(self.PAD_FEATURES_DIM, self.PAD_EMBEDDING_DIM)
        self.logit_head = nn.Linear(self.PAD_FEATURES_DIM, 1)

    def _extract_routing_features(self, stats: Dict) -> torch.Tensor:
        """
        Extract 8-D feature vector from a single MoE layer's routing stats.

        Args:
            stats: {'expert_weights': (B, N, 4), 'token_entropy': (B, N)}

        Returns:
            features: (B, 8)
        """
        ew = stats['expert_weights']   # (B, N, 4)
        ent = stats['token_entropy']   # (B, N)

        # Expert load: mean across tokens -> (B, 4)
        load_mean = ew.mean(dim=1)

        # Entropy statistics: [mean, std, max, min] -> (B, 4)
        ent_stats = torch.stack([
            ent.mean(dim=1),
            ent.std(dim=1),
            ent.max(dim=1).values,
            ent.min(dim=1).values,
        ], dim=-1)

        return torch.cat([load_mean, ent_stats], dim=-1)  # (B, 8)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: {
                'stage1_feat':       (B, 64, 56, 56),
                'stage2_feat':       (B, 128, 28, 28),
                'routing_stats_s2':  {'expert_weights': ..., 'token_entropy': ...},
                'routing_stats_s3a': {'expert_weights': ..., 'token_entropy': ...},
                'routing_stats_s3b': {'expert_weights': ..., 'token_entropy': ...},
            }

        Returns: {
            'pad_features':  (B, 128),
            'pad_embedding': (B, 32),   # L2-normalized
            'pad_logit':     (B, 1),
        }
        """
        s1 = inputs['stage1_feat']    # (B, 64, 56, 56)
        s2 = inputs['stage2_feat']    # (B, 128, 28, 28)

        # -- Feature path: conv blocks at full resolution, GAP at the end --
        f1 = self.block_s1(s1)                 # (B, 128, 56, 56)
        f2 = self.block_s2(s2)                 # (B, 128, 28, 28)
        f1 = self.gap(f1).flatten(1)           # (B, 128)
        f2 = self.gap(f2).flatten(1)           # (B, 128)
        feat_combined = torch.cat([f1, f2], dim=-1)  # (B, 256)

        # -- Routing path --
        r_s2 = self._extract_routing_features(inputs['routing_stats_s2'])    # (B, 8)
        r_s3a = self._extract_routing_features(inputs['routing_stats_s3a'])  # (B, 8)
        r_s3b = self._extract_routing_features(inputs['routing_stats_s3b'])  # (B, 8)
        route_combined = torch.cat([r_s2, r_s3a, r_s3b], dim=-1)  # (B, 24)

        # -- Fusion --
        all_features = torch.cat([feat_combined, route_combined], dim=-1)  # (B, 280)
        pad_features = self.fusion_mlp(all_features)  # (B, 128)

        # -- Parallel output branches --
        pad_embedding = F.normalize(
            self.embedding_proj(pad_features), p=2, dim=-1
        )  # (B, 32)
        pad_logit = self.logit_head(pad_features)  # (B, 1)

        return {
            'pad_features': pad_features,
            'pad_embedding': pad_embedding,
            'pad_logit': pad_logit,
        }
