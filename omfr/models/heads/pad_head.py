"""
pad_head.py — PAD Branch Head (v2: Multi-Scale TinyViT Input)

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

Key novelty: PAD branch exploits both multi-scale features AND MoE routing
statistics. Live images show diverse routing (rich micro-texture), spoofs
show uniform routing (synthetic/print texture).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PADHead(nn.Module):
    """
    Multi-scale feature aggregation + MoE routing stats fusion.

    Architecture:
        -- Feature path --
        stage1_feat (B,64,56,56)  -> AdaptiveAvgPool(7,7) -> flatten -> Linear(64*49, 256)
        stage2_feat (B,128,28,28) -> AdaptiveAvgPool(7,7) -> flatten -> Linear(128*49, 256)
        -> concat -> (B, 512)

        -- Routing path (8 features per MoE layer x 3 layers = 24) --
        Per MoE layer:
            expert_weights: mean over tokens -> (B, 4)
            token_entropy:  [mean, std, max, min] -> (B, 4)
        -> concat all 3 layers -> (B, 24)

        -- Fusion --
        concat(feature_512, routing_24) -> (B, 536)
        -> LayerNorm -> Linear(536, 256) -> GELU -> Dropout
        -> Linear(256, 128) -> pad_features

        -- Two PARALLEL output branches --
        pad_features -> Linear(128, 32) -> L2Norm -> pad_embedding (SupCon + L_orth)
        pad_features -> Linear(128, 1) -> pad_logit (BCE)
    """

    POOL_SIZE = 7
    FEAT_PROJ_DIM = 256
    ROUTE_DIM_PER_LAYER = 8  # 4 expert mean + 4 entropy stats
    NUM_MOE_LAYERS = 3
    ROUTE_DIM = ROUTE_DIM_PER_LAYER * NUM_MOE_LAYERS  # 24
    PAD_FEATURES_DIM = 128
    PAD_EMBEDDING_DIM = 32

    def __init__(
        self,
        stage1_dim: int = 64,
        stage2_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Feature path: spatial pooling + projection per stage
        self.pool = nn.AdaptiveAvgPool2d(self.POOL_SIZE)

        flat1 = stage1_dim * self.POOL_SIZE * self.POOL_SIZE  # 64*49 = 3136
        flat2 = stage2_dim * self.POOL_SIZE * self.POOL_SIZE  # 128*49 = 6272
        self.proj_s1 = nn.Linear(flat1, self.FEAT_PROJ_DIM)
        self.proj_s2 = nn.Linear(flat2, self.FEAT_PROJ_DIM)

        feat_dim = self.FEAT_PROJ_DIM * 2  # 512
        fusion_in = feat_dim + self.ROUTE_DIM  # 536

        # Fusion MLP: (B, 536) -> (B, 128). Deeper than the original 2-layer
        # projection so the head can learn richer interactions between the
        # multi-scale features and the MoE routing statistics.
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

        # -- Feature path --
        f1 = self.pool(s1).flatten(1)  # (B, 64*49)
        f2 = self.pool(s2).flatten(1)  # (B, 128*49)
        f1 = self.proj_s1(f1)          # (B, 256)
        f2 = self.proj_s2(f2)          # (B, 256)
        feat_combined = torch.cat([f1, f2], dim=-1)  # (B, 512)

        # -- Routing path --
        r_s2 = self._extract_routing_features(inputs['routing_stats_s2'])    # (B, 8)
        r_s3a = self._extract_routing_features(inputs['routing_stats_s3a'])  # (B, 8)
        r_s3b = self._extract_routing_features(inputs['routing_stats_s3b'])  # (B, 8)
        route_combined = torch.cat([r_s2, r_s3a, r_s3b], dim=-1)  # (B, 24)

        # -- Fusion --
        all_features = torch.cat([feat_combined, route_combined], dim=-1)  # (B, 536)
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
