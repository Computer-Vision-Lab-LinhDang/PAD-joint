"""
pad_head.py — PAD Branch Head

Input: {
    'layer3_tokens':   (B, 196, 192),
    'layer7_tokens':   (B, 196, 192),
    'routing_stats_3': {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
    'routing_stats_7': {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
}

Output: Dict {
    'pad_embedding': (B, 32),    # L2-normalized, 32-D (for SupCon)
    'pad_logit':     (B, 1),     # Raw logit for BCE
    'pad_features':  (B, 128),   # Pre-projection features (for SupCon)
}

Key novelty: PAD branch exploits both token features AND MoE routing statistics.
Spoof images → uniform routing (synthetic/print texture → same experts for all tokens).
Live images  → diverse routing (rich micro-texture → diverse expert routing).
Routing entropy = direct PAD signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TokenAttentionPool(nn.Module):
    """
    Simple single-head attention pooling: learns an attention query to
    produce a weighted-sum of N tokens → single vector.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, in_dim))
        self.scale = in_dim ** -0.5
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, in_dim)

        Returns:
            pooled: (B, out_dim)
        """
        B, N, C = tokens.shape
        q = self.query.expand(B, -1, -1)              # (B, 1, in_dim)
        attn = (q @ tokens.transpose(-2, -1)) * self.scale  # (B, 1, N)
        attn = F.softmax(attn, dim=-1)
        pooled = (attn @ tokens).squeeze(1)            # (B, in_dim)
        return self.proj(pooled)                        # (B, out_dim)


class PADHead(nn.Module):
    """
    PAD feature aggregation from layer 3 + 7 tokens, fused with routing patterns.

    ★ Novelty: PAD branch exploits both features AND routing statistics.
       Insight: spoof images have uniform routing (uniform texture →
       same experts for every token), live images have diverse routing
       (rich micro-texture → diverse expert routing).
       Routing entropy = direct PAD signal without extra supervision.

    Architecture:
        ── Feature path ──
        layer3_tokens (B, 196, 192) → AttentionPool → (B, 256)
        layer7_tokens (B, 196, 192) → AttentionPool → (B, 256)
        feat_combined = Concat(256+256) → (B, 512)

        ── Routing path (12 features per layer × 2 layers = 24) ──
        routing_stats_3['expert_weights']: mean(dim=1) → (B, 4), std(dim=1) → (B, 4)
        routing_stats_7['expert_weights']: mean(dim=1) → (B, 4), std(dim=1) → (B, 4)
        routing_stats_3['token_entropy']: [mean, std, max, min] → (B, 4)
        routing_stats_7['token_entropy']: [mean, std, max, min] → (B, 4)
        route_combined = Concat → (B, 24)

        ── Fusion ──
        all_features = Concat(feat_combined, route_combined) → (B, 536)
        pad_features = fusion_mlp(all_features) → (B, 128)
            [LayerNorm → Linear(536, 256) → GELU → Dropout → Linear(256, 128)]

        pad_embedding = Linear(128, 32) → L2Norm → (B, 32)   [for SupCon / orthogonality]
        pad_logit     = Linear(128, 1)  → (B, 1)              [for BCE]

    Layer rationale:
        Layer 3: early routing captures pixel-level texture (pores, ridges)
        Layer 7: mid routing captures higher-order texture patterns
        Late layers (10+) attend globally → wash out local spoof artifacts
    """

    FEAT_POOL_DIM = 256   # per-layer attention pool output
    ROUTE_DIM = 24        # per layer: 4 expert_mean + 4 expert_std + 4 entropy_stats = 12 × 2 layers
    PAD_FEATURES_DIM = 128
    PAD_EMBEDDING_DIM = 32

    def __init__(
        self,
        token_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()

        feat_dim = self.FEAT_POOL_DIM * 2  # 512 — concat of two layers
        fusion_in = feat_dim + self.ROUTE_DIM  # 536

        # Feature path: attentive pooling per layer
        self.pool_layer3 = TokenAttentionPool(token_dim, self.FEAT_POOL_DIM)
        self.pool_layer7 = TokenAttentionPool(token_dim, self.FEAT_POOL_DIM)

        # Fusion MLP: (B, 536) → (B, 128)
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.PAD_FEATURES_DIM),
        )

        # Projection to 32-D embedding (for SupCon + orthogonality loss)
        self.embedding_proj = nn.Linear(self.PAD_FEATURES_DIM, self.PAD_EMBEDDING_DIM)

        # Binary liveness logit (for BCE)
        self.logit_head = nn.Linear(self.PAD_FEATURES_DIM, 1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: {
                'layer3_tokens':   (B, 196, 192),
                'layer7_tokens':   (B, 196, 192),
                'routing_stats_3': {
                    'expert_weights': (B, 196, 4),
                    'token_entropy':  (B, 196),
                },
                'routing_stats_7': {
                    'expert_weights': (B, 196, 4),
                    'token_entropy':  (B, 196),
                },
            }

        Returns: {
            'pad_features':  (B, 128),
            'pad_embedding': (B, 32),   # L2-normalized
            'pad_logit':     (B, 1),
        }
        """
        layer3 = inputs['layer3_tokens']    # (B, 196, 192)
        layer7 = inputs['layer7_tokens']    # (B, 196, 192)
        rs3    = inputs['routing_stats_3']
        rs7    = inputs['routing_stats_7']

        # ── Feature path ──
        f3 = self.pool_layer3(layer3)  # (B, 256)
        f7 = self.pool_layer7(layer7)  # (B, 256)
        feat_combined = torch.cat([f3, f7], dim=-1)  # (B, 512)

        # ── Routing path ──
        # Expert load statistics: mean + std across 196 tokens → (B, 4) each
        load3_mean = rs3['expert_weights'].mean(dim=1)   # (B, 4)
        load3_std  = rs3['expert_weights'].std(dim=1)    # (B, 4)
        load7_mean = rs7['expert_weights'].mean(dim=1)   # (B, 4)
        load7_std  = rs7['expert_weights'].std(dim=1)    # (B, 4)
        # Entropy statistics: [mean, std, max, min] over 196 tokens → (B, 4) per layer
        ent3 = rs3['token_entropy']  # (B, 196)
        ent3_stats = torch.stack([
            ent3.mean(dim=1), ent3.std(dim=1),
            ent3.max(dim=1).values, ent3.min(dim=1).values,
        ], dim=-1)  # (B, 4)
        ent7 = rs7['token_entropy']  # (B, 196)
        ent7_stats = torch.stack([
            ent7.mean(dim=1), ent7.std(dim=1),
            ent7.max(dim=1).values, ent7.min(dim=1).values,
        ], dim=-1)  # (B, 4)
        route_combined = torch.cat([
            load3_mean, load3_std, load7_mean, load7_std,
            ent3_stats, ent7_stats,
        ], dim=-1)  # (B, 24)

        # ── Fusion ──
        all_features = torch.cat([feat_combined, route_combined], dim=-1)  # (B, 536)
        pad_features = self.fusion_mlp(all_features)  # (B, 128)

        # Outputs
        pad_embedding = F.normalize(
            self.embedding_proj(pad_features), p=2, dim=-1
        )  # (B, 32)
        pad_logit = self.logit_head(pad_features)  # (B, 1)

        return {
            'pad_features':  pad_features,
            'pad_embedding': pad_embedding,
            'pad_logit':     pad_logit,
        }
