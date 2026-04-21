"""
pad_head.py — PAD Branch Head (v4: Dedicated Stem + DS-Conv Aux + Routing)

Input: {
    'pad_stem_feat':     (B, 128),              # from PADStem (TRAINABLE)
    'stage1_feat':       (B, 64, 56, 56),       # detached backbone feature
    'stage2_feat':       (B, 128, 28, 28),      # detached backbone feature
    'routing_stats_s2':  {'expert_weights': (B, 784, 4), 'token_entropy': (B, 784)},
    'routing_stats_s3a': {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
    'routing_stats_s3b': {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
}

Output: {
    'pad_embedding': (B, 32),    # L2-normalized, for SupCon + orthogonality
    'pad_logit':     (B, 1),     # Raw logit for BCE
    'pad_features':  (B, 128),   # Pre-projection features for SupCon
}

v4 rationale:
    v3 introduced depthwise-separable conv blocks on stage1/stage2 and
    killed the 2.4M Flatten+Linear explosion. On top of that, v4 adds a
    dedicated PAD stem (see backbone/pad_stem.py) that reads the Gabor
    output directly and produces a 128-D liveness vector. This is the
    ONLY PAD feature path that remains connected to the autograd graph:
    stage1/stage2 feature maps (and MoE routing stats) are detached in
    OMFRModule._run_pad, so the TinyViT backbone is shaped exclusively
    by L_identity. Without this parallel stem, PAD had to latch onto
    identity-shaped features alone and collapsed on held-out sensors
    (BPCER ~80%). With the stem, the PAD objective owns a small (~200K)
    trainable extractor of its own while the detached backbone features
    act as structural context.

    v3 feature-path rationale (still applies):
      * DS conv blocks preserve spatial structure (no premature 8x8
        averaging) so micro-texture cues (print noise, pores, silicon
        grain) survive into fusion.
      * Linear projection of 6272-D per stage would mean ~2.4M params
        driving severe overfit. DS conv + deferred GAP cuts that to
        ~30K per stage.
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
    Multi-scale feature aggregation + dedicated PAD stem + MoE routing.

    Architecture (v4):
        -- PAD stem path (TRAINABLE, runs upstream in OMFRModule) --
        pad_stem_feat (B, 128)

        -- Aux feature path on detached backbone stages --
        stage1_feat (B,64,56,56)  -> PADConvBlock(64->128)  -> GAP -> (B,128)
        stage2_feat (B,128,28,28) -> PADConvBlock(128->128) -> GAP -> (B,128)
        -> concat -> (B, 256)

        -- Routing path (8 features per MoE layer x 3 layers = 24) --
        Per MoE layer:
            expert_weights: mean over tokens -> (B, 4)
            token_entropy:  [mean, std, max, min] -> (B, 4)
        -> concat all 3 layers -> (B, 24)

        -- Fusion --
        concat(pad_stem_128, feature_256, routing_24) -> (B, 408)
        -> LayerNorm -> MLP -> pad_features (B, 128)

        -- Two PARALLEL output branches --
        pad_features -> Linear(128, 32) -> L2Norm -> pad_embedding
        pad_features -> Linear(128, 1) -> pad_logit
    """

    FEAT_CH_PER_STAGE = 128
    PAD_STEM_DIM = 256
    ROUTE_DIM_PER_LAYER = 8   # 4 expert mean + 4 entropy stats
    NUM_MOE_LAYERS = 3
    ROUTE_DIM = ROUTE_DIM_PER_LAYER * NUM_MOE_LAYERS   # 24
    PAD_FEATURES_DIM = 128
    PAD_EMBEDDING_DIM = 32

    def __init__(
        self,
        stage1_dim: int = 64,
        stage2_dim: int = 128,
        pad_stem_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pad_stem_dim = pad_stem_dim

        # Aux feature path on detached backbone stages — DS blocks preserve
        # spatial layout, GAP is deferred to the very end of each stage.
        self.block_s1 = PADConvBlock(stage1_dim, self.FEAT_CH_PER_STAGE)
        self.block_s2 = PADConvBlock(stage2_dim, self.FEAT_CH_PER_STAGE)
        self.gap = nn.AdaptiveAvgPool2d(1)

        feat_dim = self.FEAT_CH_PER_STAGE * 2              # 256
        fusion_in = pad_stem_dim + feat_dim + self.ROUTE_DIM   # 408

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
                'pad_stem_feat':     (B, 128),
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
        pad_stem = inputs['pad_stem_feat']    # (B, 128)  — trainable signal
        s1 = inputs['stage1_feat']            # (B, 64, 56, 56)  — detached
        s2 = inputs['stage2_feat']            # (B, 128, 28, 28) — detached

        # -- Aux feature path: conv blocks at full resolution, GAP at end --
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

        # -- Fusion: stem is the primary (trainable) signal, aux + routing join --
        all_features = torch.cat(
            [pad_stem, feat_combined, route_combined], dim=-1,
        )  # (B, 408)
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
