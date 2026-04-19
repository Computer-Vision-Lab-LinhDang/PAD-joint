"""
identity_head.py — MRL Identity Head (v2: Stage 3+4 Input, 7x7 RPE)

Input: {
    'stage3_feat': (B, 160, 14, 14),
    'stage4_feat': (B, 320, 7, 7),
}

Output: {
    'identity_embedding': (B, 256),     # L2-normalized, full 256-D
    'mrl_embeddings': {
        64:  (B, 64),
        128: (B, 128),
        256: (B, 256),
    },
}

ArcFace classifiers are NOT in this head — they live in ArcFaceLoss.
This avoids weight duplication and simplifies ONNX export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class RelativePositionEncoding(nn.Module):
    """
    Learned RPE for 7x7 spatial grid.

    Table: (num_heads, 13, 13) indexed by (dx+6, dy+6)
    where dx, dy in [-6, +6].
    """

    def __init__(self, grid_size: int = 7, num_heads: int = 8):
        super().__init__()
        self.grid_size = grid_size
        self.num_heads = num_heads
        table_size = 2 * grid_size - 1  # 13
        self.rpe_table = nn.Parameter(torch.zeros(num_heads, table_size, table_size))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

        # Precompute token coordinates for grid
        coords = torch.arange(grid_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer(
            'token_coords',
            torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1),  # (49, 2)
        )

    def get_bias(self) -> torch.Tensor:
        """
        Returns:
            bias: (num_heads, 49, 49) — RPE bias for all token pairs
        """
        coords = self.token_coords  # (49, 2)
        delta = coords.unsqueeze(0) - coords.unsqueeze(1)  # (49, 49, 2)
        dy = delta[..., 0] + self.grid_size - 1  # shift to [0, 12]
        dx = delta[..., 1] + self.grid_size - 1
        return self.rpe_table[:, dy, dx]  # (num_heads, 49, 49)


class StructuralAttentionBlock(nn.Module):
    """
    Multi-head self-attention with RPE on 7x7 token grid.

    Single layer: Pre-LN attention + FFN with residual connections.
    8 heads, head_dim = 256/8 = 32.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, grid_size: int = 7):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.rpe = RelativePositionEncoding(
            grid_size=grid_size,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 49, 256)
        Returns:
            x: (B, 49, 256)
        """
        B, N, C = x.shape

        # Self-attention with RPE
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn + self.rpe.get_bias().unsqueeze(0)  # add RPE
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = residual + x

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class AttentivePooling(nn.Module):
    """
    Multi-query attentive pooling with M=4 learned query seeds.

    4 query vectors each attend over all 49 tokens -> (B, 4, 256)
    -> flatten -> Linear(1024, 256) -> (B, 256)
    """

    def __init__(self, embed_dim: int = 256, num_queries: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5
        self.out_proj = nn.Linear(num_queries * embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, C) — N=49 spatial tokens
        Returns:
            pooled: (B, C) — single vector
        """
        B, N, C = tokens.shape
        q = self.queries.expand(B, -1, -1)     # (B, 4, C)
        k = self.key_proj(tokens)               # (B, N, C)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, 4, N)
        attn = F.softmax(attn, dim=-1)
        pooled = attn @ tokens                  # (B, 4, C)
        pooled = pooled.flatten(1)              # (B, 4*C)
        return self.out_proj(pooled)            # (B, C)


class IdentityHead(nn.Module):
    """
    Structural Attention on 49 tokens (7x7) + MRL embedding.

    Architecture:
        stage3_feat (B,160,14,14) -> AdaptiveAvgPool(7,7) -> reshape -> (B,49,160)
        stage4_feat (B,320,7,7) -> reshape -> (B,49,320)
        -> concat -> (B, 49, 480)
        -> Linear(480, 256) -> (B, 49, 256)
        -> StructuralAttentionBlock (8 heads, RPE 7x7)
        -> AttentivePooling (4 query seeds) -> (B, 256)
        -> LayerNorm -> L2Norm -> identity_embedding (B, 256)
        -> MRL: slice at {64, 128, 256}, re-normalize each

    No ArcFace classifiers here — those live in ArcFaceLoss to avoid
    weight duplication.

    Args:
        embed_dim: identity embedding dimension (256)
        num_heads: structural attention heads (8)
        grid_size: spatial grid size for RPE (7)
        num_queries: attentive pooling query seeds (4)
        stage3_dim: stage 3 feature dimension (160)
        stage4_dim: stage 4 feature dimension (320)
    """

    MRL_DIMS = [64, 128, 256]

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        grid_size: int = 7,
        num_queries: int = 4,
        stage3_dim: int = 160,
        stage4_dim: int = 320,
        num_decoder_blocks: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # Spatial pooling for stage3 (14x14 -> 7x7)
        self.pool_s3 = nn.AdaptiveAvgPool2d(grid_size)

        # Project concatenated features to embed_dim
        concat_dim = stage3_dim + stage4_dim  # 480
        self.input_proj = nn.Linear(concat_dim, embed_dim)

        # Stack of structural attention blocks with shared RPE depth.
        # Deeper decoder (3 blocks) gives the head enough capacity to learn
        # discriminative identity representations on par with face-recognition
        # baselines (ArcFace/AdaFace use ≥2 FC + conv decoders).
        self.struct_attn = nn.ModuleList([
            StructuralAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                grid_size=grid_size,
            )
            for _ in range(num_decoder_blocks)
        ])

        # Attentive pooling
        self.attn_pool = AttentivePooling(
            embed_dim=embed_dim,
            num_queries=num_queries,
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: {
                'stage3_feat': (B, 160, 14, 14),
                'stage4_feat': (B, 320, 7, 7),
            }

        Returns: {
            'identity_embedding': (B, 256),
            'mrl_embeddings': {64: (B,64), 128: (B,128), 256: (B,256)},
        }
        """
        s3 = inputs['stage3_feat']  # (B, 160, 14, 14)
        s4 = inputs['stage4_feat']  # (B, 320, 7, 7)
        B = s3.shape[0]
        G = self.grid_size  # 7

        # Pool stage3 to 7x7, reshape to token sequence
        s3_pooled = self.pool_s3(s3)                              # (B, 160, 7, 7)
        s3_tokens = s3_pooled.flatten(2).transpose(1, 2)          # (B, 49, 160)

        # Stage4 is already 7x7
        s4_tokens = s4.flatten(2).transpose(1, 2)                 # (B, 49, 320)

        # Concat features per spatial position
        tokens = torch.cat([s3_tokens, s4_tokens], dim=-1)        # (B, 49, 480)

        # Project to embed_dim
        tokens = self.input_proj(tokens)                          # (B, 49, 256)

        # Structural attention with RPE — stacked decoder blocks
        for block in self.struct_attn:
            tokens = block(tokens)                                # (B, 49, 256)

        # Attentive pooling -> single vector
        pooled = self.attn_pool(tokens)                           # (B, 256)
        pooled = self.final_norm(pooled)

        # L2 normalize for identity embedding
        identity_embedding = F.normalize(pooled, p=2, dim=-1)     # (B, 256)

        # MRL: slice prefix, re-normalize
        mrl_embeddings = {}
        for d in self.MRL_DIMS:
            z = identity_embedding[:, :d]
            z = F.normalize(z, p=2, dim=-1)
            mrl_embeddings[d] = z

        return {
            'identity_embedding': identity_embedding,
            'mrl_embeddings': mrl_embeddings,
        }

    def forward_embedding_only(
        self,
        inputs: Dict[str, torch.Tensor],
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning only embedding (no ArcFace logits).

        Used for: Phase 2 PAD batch (L_orth only), inference, ONNX export.

        Args:
            inputs: same as forward()
            dim: if provided, return MRL prefix at this dimension

        Returns:
            embedding: (B, dim) or (B, 256) — L2-normalized
        """
        out = self.forward(inputs)
        if dim is not None:
            return out['mrl_embeddings'][dim]
        return out['identity_embedding']
