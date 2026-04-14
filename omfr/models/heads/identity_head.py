"""
identity_head.py — MRL-ArcFace Identity Head

Input:
    layer12_tokens: (B, 196, 192)  — Final ViT layer output
    cls_token:      (B, 192)       — CLS token (global summary)

Output: Dict {
    'identity_embedding': (B, 256),      # L2-normalized, full 256-D
    'mrl_embeddings': {
        64:  (B, 64),
        128: (B, 128),
        256: (B, 256),
    },
    'mrl_logits': {
        64:  (B, num_classes),
        128: (B, num_classes),
        256: (B, num_classes),
    },
}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class RelativePositionEncoding(nn.Module):
    """
    Relative Position Encoding for fingerprint tokens.

    Encodes (Δx, Δy) spatial offsets between token positions on the 14×14 grid.
    Fingerprint-aware: also encodes angular offset Δθ and frequency-band affinity Δf.

    For simplicity, Δθ and Δf are set to learned biases (trainable offsets added
    to the (Δx, Δy) table). Full Δθ/Δf computation would require ridge orientation
    maps; this approximation uses learned per-head position biases instead.
    """

    def __init__(self, grid_size: int = 14, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.num_heads = num_heads
        # Learnable table: (2*grid_size-1) × (2*grid_size-1) × num_heads
        # Indexed by (Δx + grid_size - 1, Δy + grid_size - 1)
        table_size = 2 * grid_size - 1
        self.rpe_table = nn.Parameter(
            torch.zeros(num_heads, table_size, table_size)
        )
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

        # Precompute token position indices for 14×14 grid
        coords = torch.arange(grid_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        # (196, 2) — (row, col) for each token
        self.register_buffer(
            'token_coords',
            torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        )

    def get_bias(self) -> torch.Tensor:
        """
        Compute RPE bias matrix for all token pairs.

        Returns:
            bias: (num_heads, 196+1, 196+1) — includes CLS token at index 0.
                  CLS-to-token and token-to-CLS biases are zero.
        """
        N = self.grid_size ** 2  # 196
        coords = self.token_coords  # (196, 2)

        # Δy = row_i - row_j, Δx = col_i - col_j
        delta = coords.unsqueeze(0) - coords.unsqueeze(1)  # (196, 196, 2)
        dy = delta[..., 0] + self.grid_size - 1  # shift to [0, 2*G-2]
        dx = delta[..., 1] + self.grid_size - 1

        # Gather from table: (num_heads, 196, 196)
        bias = self.rpe_table[:, dy, dx]

        # Prepend zero row/col for CLS token → (num_heads, 197, 197)
        bias_padded = F.pad(bias, (1, 0, 1, 0), value=0.0)
        return bias_padded


class StructuralAttentionBlock(nn.Module):
    """
    Single multi-head self-attention block with RPE for fingerprint tokens.

    Takes (B, 197, 256) tokens (CLS prepended) and outputs same shape.
    Uses RPE bias from RelativePositionEncoding.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, grid_size: int = 14):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.rpe = RelativePositionEncoding(
            grid_size=grid_size,
            num_heads=num_heads,
            head_dim=self.head_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 197, embed_dim) — CLS token at index 0

        Returns:
            x: (B, 197, embed_dim)
        """
        B, N, C = x.shape

        # Self-attention with RPE
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # Add RPE bias
        rpe_bias = self.rpe.get_bias()  # (heads, N, N)
        attn = attn + rpe_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = residual + x

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class AttentivePooling(nn.Module):
    """
    Attentive pooling: learns a query vector that attends over all tokens
    to produce a single weighted-sum representation.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, embed_dim)

        Returns:
            pooled: (B, embed_dim)
        """
        B, N, C = tokens.shape
        q = self.query.expand(B, -1, -1)  # (B, 1, C)
        k = self.key(tokens)               # (B, N, C)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, 1, N)
        attn = F.softmax(attn, dim=-1)
        pooled = (attn @ tokens).squeeze(1)  # (B, C)
        return pooled


class ArcFaceClassifier(nn.Module):
    """
    ArcFace classifier head: normalized linear projection used as the
    class weight matrix in ArcFaceLoss.

    Stores the weight matrix W ∈ R^(num_classes × embed_dim).
    During forward, returns cosine similarity logits (before margin).
    The margin is applied in ArcFaceLoss (arcface.py) during training.
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, embed_dim) — L2-normalized

        Returns:
            logits: (B, num_classes) — cosine similarity × scale (scale applied in loss)
        """
        # Normalize weight matrix
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity (embeddings already normalized)
        return F.linear(embeddings, w_norm)


class IdentityHead(nn.Module):
    """
    MRL-ArcFace head with structural attention on layer-12 tokens.

    Architecture:
        1. Prepend CLS token to layer12_tokens → (B, 197, 192)
        2. Project 192 → 256
        3. StructuralAttentionBlock with RPE(Δx, Δy)
        4. AttentivePooling → (B, 256)
        5. L2Norm → identity_embedding (B, 256)
        6. MRL: slice prefix at {64, 128, 256}, L2Norm each
        7. Per-dimension ArcFaceClassifier (weights NOT shared)

    Args:
        num_classes: int   — number of training identities
        embed_dim: int = 256
        num_heads: int = 4 — attention heads in structural block
        mrl_dims: list     — MRL nesting dimensions
    """

    MRL_DIMS = [64, 128, 256]

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        grid_size: int = 14,
        token_dim: int = 192,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Project backbone tokens to head embed_dim
        self.input_proj = nn.Linear(token_dim, embed_dim)

        # Structural attention block
        self.struct_attn = StructuralAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            grid_size=grid_size,
        )

        # Attentive pooling to produce single vector
        self.attn_pool = AttentivePooling(embed_dim=embed_dim)

        # Final LayerNorm before L2Norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # Per-dimension ArcFace classifiers (NOT shared weights)
        self.classifiers = nn.ModuleDict({
            str(d): ArcFaceClassifier(d, num_classes)
            for d in self.MRL_DIMS
        })

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: {
                'layer12_tokens': (B, 196, 192),
                'cls_token':      (B, 192),
            }

        Returns: {
            'identity_embedding': (B, 256),
            'mrl_embeddings': {64: (B,64), 128: (B,128), 256: (B,256)},
            'mrl_logits':     {64: (B,C),  128: (B,C),   256: (B,C)},
        }
        """
        layer12 = inputs['layer12_tokens']  # (B, 196, 192)
        cls     = inputs['cls_token']       # (B, 192)

        B = layer12.shape[0]

        # Prepend CLS → (B, 197, 192)
        tokens = torch.cat([cls.unsqueeze(1), layer12], dim=1)

        # Project to embed_dim
        tokens = self.input_proj(tokens)  # (B, 197, 256)

        # Structural attention with RPE
        tokens = self.struct_attn(tokens)  # (B, 197, 256)

        # Attentive pooling → single vector
        pooled = self.attn_pool(tokens)  # (B, 256)
        pooled = self.final_norm(pooled)

        # L2-normalize for identity embedding
        identity_embedding = F.normalize(pooled, p=2, dim=-1)  # (B, 256)

        # MRL: slice prefix, L2-normalize, compute cosine logits
        mrl_embeddings = {}
        mrl_logits = {}
        for d in self.MRL_DIMS:
            z = identity_embedding[:, :d]             # (B, d)
            z = F.normalize(z, p=2, dim=-1)           # re-normalize prefix
            mrl_embeddings[d] = z
            mrl_logits[d] = self.classifiers[str(d)](z)  # (B, num_classes)

        return {
            'identity_embedding': identity_embedding,
            'mrl_embeddings':     mrl_embeddings,
            'mrl_logits':         mrl_logits,
        }

    def forward_embedding_only(
        self,
        inputs: Dict[str, torch.Tensor],
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning embedding without ArcFace logit computation.

        Use cases:
          - Phase 2 PAD batch: compute identity embedding for L_orth without
            activating ArcFace classifiers (called with detached layer12/CLS).
          - Inference / deployment: no classifier weights needed at runtime.
          - ONNX export: excludes large ArcFace weight matrices from the graph.

        Args:
            inputs: same as forward() — {'layer12_tokens': ..., 'cls_token': ...}
            dim: if provided, return the MRL prefix at this dimension; else full 256-D.

        Returns:
            embedding: (B, dim) or (B, 256) — L2-normalized
        """
        out = self.forward(inputs)
        if dim is not None:
            return out['mrl_embeddings'][dim]
        return out['identity_embedding']
