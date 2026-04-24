"""
tiny_vit.py — TinyViT-5M Backbone Wrapper with Pretrained Loading + MoE Insertion

Loads TinyViT-5M from timm (pretrained on ImageNet-22k, fine-tuned on 1k).
Adapts patch_embed for 8-channel Gabor input and inserts Frequency-Gated
MoE-FFN at 3 strategic locations.

TinyViT-5M config:
    embed_dims  = [64, 128, 160, 320]
    depths      = [2, 2, 6, 2]
    num_heads   = [2, 4, 5, 10]
    window_sizes = [7, 7, 14, 7]

MoE placement:
    Stage 1 (index 1): block 1 (0-indexed) — 28×28 tokens, 128-ch
    Stage 2 (index 2): block 2, block 5 (0-indexed) — 14×14 tokens, 160-ch

I/O:
    Input:  (B, 8, 224, 224) — 8-channel Gabor-enhanced fingerprint
    Output: dict {
        'stage1_feat': (B, 64, 56, 56),
        'stage2_feat': (B, 128, 28, 28),
        'stage3_feat': (B, 160, 14, 14),
        'stage4_feat': (B, 320, 7, 7),
        'routing_stats': {'s2': {...}, 's3a': {...}, 's3b': {...}},
        'balance_losses': [scalar, scalar, scalar],
    }
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import torch
import torch.nn as nn

import timm

from .moe_ffn import FreqGatedMoEFFN


class MoEMlpWrapper(nn.Module):
    """
    Drop-in replacement for timm's NormMlp inside TinyVitBlock.

    Keeps the pretrained LayerNorm from NormMlp, replaces the FFN body
    with FreqGatedMoEFFN. Stores last routing stats as an attribute
    so the backbone can collect them after the forward pass.

    The TinyVitBlock calls `self.mlp(x)` where x is (B, N, C).
    This wrapper matches that interface.
    """

    def __init__(
        self,
        norm: nn.LayerNorm,
        embed_dim: int,
        ffn_hidden: int,
        spatial_h: int,
        spatial_w: int,
        num_experts: int = 4,
        top_k: int = 2,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = norm  # reuse pretrained LayerNorm
        self.moe_ffn = FreqGatedMoEFFN(
            embed_dim=embed_dim,
            ffn_hidden=ffn_hidden,
            num_experts=num_experts,
            top_k=top_k,
            drop=drop,
        )
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Routing stats stored here after each forward — collected by backbone
        self.last_routing_stats: dict | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) — token sequence (N = spatial_h * spatial_w)
        Returns:
            out: (B, N, C)
        """
        x = self.norm(x)
        out, routing_stats = self.moe_ffn(x, self.spatial_h, self.spatial_w)
        self.last_routing_stats = routing_stats
        return out

    @property
    def temperature(self) -> float:
        return self.moe_ffn.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.moe_ffn.temperature = value


class TinyViTBackbone(nn.Module):
    """
    TinyViT-5M backbone with pretrained loading and MoE insertion.

    Modifications from pretrained model:
        1. patch_embed.conv1: Conv2d(3,32,...) → Conv2d(8,32,...) for 8-ch Gabor
        2. Stage 1 block 1: NormMlp → MoEMlpWrapper (28×28, 128-ch)
        3. Stage 2 block 2: NormMlp → MoEMlpWrapper (14×14, 160-ch)
        4. Stage 2 block 5: NormMlp → MoEMlpWrapper (14×14, 160-ch)
        5. Classification head removed — returns intermediate stage features

    Args:
        pretrained: load ImageNet pretrained weights
        in_chans: input channels (8 for Gabor stem)
        num_experts: MoE experts per layer
        top_k: active experts per token
        drop: dropout in MoE FFN
    """

    # MoE locations: (stage_idx, block_idx, spatial_h, spatial_w)
    MOE_LOCATIONS = [
        (1, 1, 28, 28),  # Stage 1 (idx 1), block 1 — 's2'
        (2, 2, 14, 14),  # Stage 2 (idx 2), block 2 — 's3a'
        (2, 5, 14, 14),  # Stage 2 (idx 2), block 5 — 's3b'
    ]
    MOE_KEYS = ['s2', 's3a', 's3b']

    # Hidden dim ratios per stage (from timm: mlp_ratio=4)
    STAGE_DIMS = {
        1: (128, 512),   # embed_dim, ffn_hidden
        2: (160, 640),
    }

    def __init__(
        self,
        pretrained: bool = True,
        in_chans: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
        drop: float = 0.0,
        use_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint

        # Load pretrained TinyViT-5M
        base_model = timm.create_model(
            'tiny_vit_5m_224',
            pretrained=pretrained,
            num_classes=0,  # remove classifier head
        )

        # Extract components
        self.patch_embed = base_model.patch_embed
        self.stages = base_model.stages

        # 1. Adapt patch_embed for 8-channel input
        self._adapt_patch_embed(in_chans, pretrained)

        # 2. Insert MoE-FFN at specified locations
        self._moe_wrappers: List[MoEMlpWrapper] = []
        for (stage_idx, block_idx, sh, sw) in self.MOE_LOCATIONS:
            embed_dim, ffn_hidden = self.STAGE_DIMS[stage_idx]
            block = self.stages[stage_idx].blocks[block_idx]

            # Extract the pretrained LayerNorm from NormMlp
            old_mlp = block.mlp
            pretrained_norm = old_mlp.norm

            wrapper = MoEMlpWrapper(
                norm=pretrained_norm,
                embed_dim=embed_dim,
                ffn_hidden=ffn_hidden,
                spatial_h=sh,
                spatial_w=sw,
                num_experts=num_experts,
                top_k=top_k,
                drop=drop,
            )
            block.mlp = wrapper
            self._moe_wrappers.append(wrapper)

        # Remove the classification head (not needed)
        # base_model.head is already excluded since we set num_classes=0

    def _adapt_patch_embed(self, in_chans: int, pretrained: bool) -> None:
        """Replace first conv from 3-ch to in_chans-ch, preserving pretrained weights."""
        old_conv = self.patch_embed.conv1.conv  # Conv2d(3, 32, 3, 2, 1)
        new_conv = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                # Copy pretrained weights for first 3 channels
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # Extra channels: mean of pretrained + small Gaussian noise so
                # each channel can specialize (identical init → redundant filters).
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                std = old_conv.weight.std().item() * 0.1
                for c in range(3, in_chans):
                    noise = torch.randn_like(mean_weight) * std
                    new_conv.weight[:, c:c+1, :, :] = mean_weight + noise
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

        self.patch_embed.conv1.conv = new_conv

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            x: (B, 8, 224, 224) — 8-channel Gabor-enhanced fingerprint

        Returns:
            dict with keys:
                'stage1_feat':    (B, 64, 56, 56)
                'stage2_feat':    (B, 128, 28, 28)
                'stage3_feat':    (B, 160, 14, 14)
                'stage4_feat':    (B, 320, 7, 7)
                'routing_stats':  {'s2': {...}, 's3a': {...}, 's3b': {...}}
                'balance_losses': [scalar, scalar, scalar]
        """
        # Patch embedding: (B, 8, 224, 224) → (B, 64, 56, 56)
        x = self.patch_embed(x)

        stage_feats = {}

        # Run through 4 stages, capturing intermediate feature maps.
        # Optional gradient checkpointing trades ~30% extra compute for
        # roughly a 2× reduction in activation memory — the primary knob
        # for fitting Phase 2 (two-head forward + orth loss) on a single
        # GPU. Non-reentrant checkpointing is required so the MoE side-
        # channel (wrapper.last_routing_stats) behaves correctly through
        # the re-forward that happens during backward.
        use_ckpt = self.use_grad_checkpoint and self.training and x.requires_grad
        for i, stage in enumerate(self.stages):
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    stage, x, use_reentrant=False,
                )
            else:
                x = stage(x)
            # x is (B, C, H, W) after each stage
            stage_feats[i] = x

        # Collect routing stats from MoE wrappers
        routing_stats = {}
        balance_losses = []
        for key, wrapper in zip(self.MOE_KEYS, self._moe_wrappers):
            stats = wrapper.last_routing_stats
            if stats is not None:
                routing_stats[key] = {
                    'expert_weights': stats['expert_weights'],
                    'token_entropy': stats['token_entropy'],
                    # Gate-only copies: grad stops at gate_proj params.
                    # Consumed by PADHead when we want to shape the
                    # gate without leaking PAD grad back into the
                    # backbone (see OMFRModule._run_pad).
                    'expert_weights_gateonly': stats['expert_weights_gateonly'],
                    'token_entropy_gateonly':  stats['token_entropy_gateonly'],
                    # Raw 3-band frequency energy (detached in moe_ffn).
                    # Bypasses the Temperature-collapse problem: at T=0.5
                    # token_entropy saturates near 0, so PAD needs a
                    # pre-softmax signal. Shape: (B, N, 3).
                    'gate_input_gateonly': stats['gate_input_gateonly'],
                }
                balance_losses.append(stats['balance_loss'])
            else:
                # Should not happen during forward, but safety fallback
                balance_losses.append(torch.tensor(0.0, device=x.device))

        return {
            'stage1_feat': stage_feats[0],   # (B, 64, 56, 56)
            'stage2_feat': stage_feats[1],   # (B, 128, 28, 28)
            'stage3_feat': stage_feats[2],   # (B, 160, 14, 14)
            'stage4_feat': stage_feats[3],   # (B, 320, 7, 7)
            'routing_stats': routing_stats,
            'balance_losses': balance_losses,
        }

    # -------------------------------------------------------------------
    # Utility methods for differential LR / parameter groups
    # -------------------------------------------------------------------

    def get_stage_params(self, stage_indices: List[int]) -> Iterator[nn.Parameter]:
        """
        Yield parameters for specific stages (0-indexed).
        Excludes MoE parameters (use get_moe_params() for those).
        """
        moe_ids = {id(p) for p in self.get_moe_params()}
        for idx in stage_indices:
            for p in self.stages[idx].parameters():
                if id(p) not in moe_ids:
                    yield p

    def get_moe_params(self) -> Iterator[nn.Parameter]:
        """Yield parameters for all MoE-FFN layers (experts + gating)."""
        for wrapper in self._moe_wrappers:
            yield from wrapper.moe_ffn.parameters()

    def get_embed_params(self) -> Iterator[nn.Parameter]:
        """Yield patch_embed parameters."""
        yield from self.patch_embed.parameters()

    def set_moe_temperature(self, temperature: float) -> None:
        """Set temperature for all MoE routing layers."""
        for wrapper in self._moe_wrappers:
            wrapper.temperature = temperature

    def freeze_stages(self, stage_indices: List[int]) -> None:
        """Freeze specific stages (0-indexed)."""
        for idx in stage_indices:
            for p in self.stages[idx].parameters():
                p.requires_grad = False

    def unfreeze_stages(self, stage_indices: List[int]) -> None:
        """Unfreeze specific stages (0-indexed)."""
        for idx in stage_indices:
            for p in self.stages[idx].parameters():
                p.requires_grad = True
