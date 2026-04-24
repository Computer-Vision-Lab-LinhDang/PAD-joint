"""
vit_tiny.py — ViT-Tiny backbone with Frequency-Gated MoE-FFN

Input:  (B, 3, 224, 224) — 3-channel enhanced fingerprint (from LearnableGaborStem)
Output: Dict {
    'layer3_tokens':   (B, 196, 192),
    'layer7_tokens':   (B, 196, 192),
    'layer10_tokens':  (B, 196, 192),
    'layer12_tokens':  (B, 196, 192),
    'cls_token':       (B, 192),
    'routing_stats': {
        3:  {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
        7:  {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
        10: {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
    },
    'balance_losses': [scalar, scalar, scalar],   # one per MoE layer
}
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_ffn import FreqGatedMoEFFN


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Image → patch token sequence.

    Args:
        img_size    (int): Input image size. Default 224.
        patch_size  (int): Patch size. Default 16.
        in_chans    (int): Number of input channels. Default 3.
        embed_dim   (int): Embedding dimension. Default 192.

    I/O:
        Input:  (B, in_chans, H, W)
        Output: (B, num_patches, embed_dim)   num_patches = (H/P)*(W/P)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 196
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)                  # (B, D, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, D)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention.

    I/O:
        Input:  (B, N, D)
        Output: (B, N, D)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    Standard 2-layer MLP FFN used in non-MoE transformer blocks.

    I/O:
        Input:  (B, N, D)
        Output: (B, N, D)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard ViT transformer block (Pre-LN).

    Uses FreqGatedMoEFFN when is_moe=True, else standard Mlp FFN.

    I/O:
        Input:  (B, N+1, D)   — includes CLS token
        Output: (B, N+1, D), routing_stats | None
    """

    def __init__(
        self,
        dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        is_moe: bool = False,
    ) -> None:
        super().__init__()
        self.is_moe = is_moe

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)

        if is_moe:
            ffn_hidden = int(dim * mlp_ratio)
            self.ffn = FreqGatedMoEFFN(embed_dim=dim, ffn_hidden=ffn_hidden, drop=drop)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N+1, D)  — full sequence including CLS token at position 0

        Returns:
            x:             (B, N+1, D)
            routing_stats: dict | None  (only non-None for MoE blocks)
        """
        # Self-attention (Pre-LN)
        x = x + self.attn(self.norm1(x))

        # FFN
        normed = self.norm2(x)

        if self.is_moe:
            # MoE FFN operates only on spatial tokens (excluding CLS at pos 0)
            cls_tok = normed[:, :1, :]      # (B, 1, D)
            spatial_tok = normed[:, 1:, :]  # (B, 196, D)
            
            # [NEW] Tính toán kích thước không gian (spatial_h, spatial_w)
            # Vì tổng số patch là 196, nên chiều dài/rộng là căn bậc 2 của 196 = 14
            spatial_len = spatial_tok.shape[1]
            spatial_h = int(math.sqrt(spatial_len))
            spatial_w = spatial_h

            # [NEW] Truyền đầy đủ 3 tham số vào cho ffn
            ffn_out, routing_stats = self.ffn(spatial_tok, spatial_h, spatial_w)

            # Reassemble: CLS goes through identity (no MoE for CLS)
            cls_residual = x[:, :1, :]  # original CLS (before norm)
            # Note: standard residual — cls unchanged, spatial gets MoE output
            x = torch.cat([
                cls_residual,
                x[:, 1:, :] + ffn_out,
            ], dim=1)

            return x, routing_stats


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------

class ViTTinyBackbone(nn.Module):
    """
    ViT-Tiny (flat, depth=12) with Frequency-Gated MoE-FFN at layers {3, 7, 10}.

    All 196 spatial tokens maintain the same 14×14 resolution throughout all
    12 layers, giving global attention at every stage and clean MoE placement.

    Config:
        embed_dim:   192
        depth:       12
        num_heads:   3
        mlp_ratio:   4.0   → FFN hidden dim = 768
        patch_size:  16    → 224/16 = 14×14 = 196 spatial tokens
        MoE layers:  {3, 7, 10}  (1-indexed)
        ~8.35M params (backbone only, excluding heads)

    Why ViT-Tiny over TinyViT:
        - Flat architecture: every token attends to all others at every layer
        - Uniform FFN structure → MoE drop-in replacement is trivial
        - MoE routing is learned, not hard-coded like hierarchical stages
        - Routing statistics (expert load / token entropy) serve as a novel
          PAD signal: live images have diverse routing, spoofs have uniform routing

    I/O:
        Input:  (B, 3, 224, 224)
        Output: dict — see module docstring header for full schema
    """

    MOE_LAYERS = {3, 7, 10}  # 1-indexed

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196

        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Build transformer blocks (1-indexed layer numbers)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate + dpr[i],
                attn_drop=attn_drop_rate,
                is_moe=(i + 1) in self.MOE_LAYERS,  # i is 0-indexed, layers are 1-indexed
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Args:
            x: (B, 3, 224, 224) — 3-channel enhanced fingerprint image

        Returns:
            dict with keys:
                'layer3_tokens':  (B, 196, 192)
                'layer7_tokens':  (B, 196, 192)
                'layer10_tokens': (B, 196, 192)
                'layer12_tokens': (B, 196, 192)
                'cls_token':      (B, 192)
                'routing_stats':  {3: {...}, 7: {...}, 10: {...}}
                'balance_losses': [scalar, scalar, scalar]
        """
        B = x.shape[0]

        # Patch embedding
        tokens = self.patch_embed(x)  # (B, 196, D)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 197, D)

        # Position embedding
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # Collect outputs
        layer_outputs: Dict = {}
        routing_stats: Dict = {}
        balance_losses: List = []

        # Pass through transformer blocks (1-indexed layer numbers)
        for i, block in enumerate(self.blocks):
            layer_idx = i + 1  # 1-indexed
            tokens, r_stats = block(tokens)

            if r_stats is not None:
                # MoE layer — strip CLS for tapped tokens
                routing_stats[layer_idx] = r_stats
                balance_losses.append(r_stats["balance_loss"])

            if layer_idx in {3, 7, 10, 12}:
                # Tap spatial tokens (exclude CLS at pos 0)
                spatial = tokens[:, 1:, :]  # (B, 196, D)
                layer_outputs[layer_idx] = spatial

        # Final LayerNorm
        tokens = self.norm(tokens)

        # Overwrite layer12 with normed version for identity head
        layer_outputs[12] = tokens[:, 1:, :]  # (B, 196, D)
        cls_out = tokens[:, 0, :]              # (B, D)

        return {
            "layer3_tokens": layer_outputs[3],
            "layer7_tokens": layer_outputs[7],
            "layer10_tokens": layer_outputs[10],
            "layer12_tokens": layer_outputs[12],
            "cls_token": cls_out,
            "routing_stats": routing_stats,
            "balance_losses": balance_losses,
        }

    # -----------------------------------------------------------------------
    # Utility methods for differential LR configuration
    # -----------------------------------------------------------------------

    def get_layer_params(self, layers: List[int]) -> Iterator[nn.Parameter]:
        """
        Yield parameters belonging to specific (1-indexed) transformer layers.

        Args:
            layers: list of 1-indexed layer numbers, e.g. [1, 2, 3, 4, 5, 6]

        Yields:
            nn.Parameter instances for those layers
        """
        for layer_idx in layers:
            i = layer_idx - 1  # 0-indexed block index
            if 0 <= i < len(self.blocks):
                yield from self.blocks[i].parameters()

    def get_moe_params(self) -> Iterator[nn.Parameter]:
        """
        Yield parameters belonging to all MoE-FFN blocks (experts + gate projections).

        These typically require a higher learning rate (2× base LR) since they
        are newly initialized and not loaded from ImageNet pretrained weights.

        Yields:
            nn.Parameter instances for all MoE blocks
        """
        for layer_idx in self.MOE_LAYERS:
            i = layer_idx - 1
            block = self.blocks[i]
            if block.is_moe:
                yield from block.ffn.parameters()

    def get_stem_and_embed_params(self) -> Iterator[nn.Parameter]:
        """
        Yield patch embedding and positional embedding parameters.

        Yields:
            nn.Parameter instances for patch embed, cls_token, pos_embed
        """
        yield from self.patch_embed.parameters()
        yield self.cls_token
        yield self.pos_embed
