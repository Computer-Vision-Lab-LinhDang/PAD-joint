"""
fastvit.py — FastViT-SA12 backbone wrapper with MoE insertion.

Replaces the TinyViT-5M backbone with a stronger FastViT-SA12 (~10.5M
params): RepMixer stages 0-2 for efficient local mixing and genuine
self-attention in stage 3. Gives OMFR global context for identity
matching without blowing the compute budget.

Stage dims & resolutions (224² input):
    stage0: (B,  64, 56, 56)    2 x RepMixerBlock
    stage1: (B, 128, 28, 28)    2 x RepMixerBlock
    stage2: (B, 256, 14, 14)    6 x RepMixerBlock
    stage3: (B, 512,  7,  7)    2 x AttentionBlock

MoE placement:
    stage1.blocks[1] — 128ch, 28×28 — key  's2'
    stage2.blocks[2] — 256ch, 14×14 — key  's3a'
    stage2.blocks[5] — 256ch, 14×14 — key  's3b'

In all FastVit blocks the FFN is a ``ConvMlp``:

    (B,C,H,W) → dw7×7 conv (+BN) → 1×1 fc1 → GELU → 1×1 fc2 → (B,C,H,W)

We keep the depthwise conv (important local mixer, pretrained) and only
replace the fc1/GELU/fc2 expansion with FreqGatedMoEFFN. Tokens are
flattened (B,N,C) across H·W for the MoE expert dispatch and reshaped
back to (B,C,H,W) afterwards.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import timm
import torch
import torch.nn as nn

from .moe_ffn import FreqGatedMoEFFN


class ConvMoEMlpWrapper(nn.Module):
    """Drop-in replacement for FastVit's ``ConvMlp`` that keeps the
    depthwise-7×7 spatial mixer from the pretrained checkpoint and
    swaps the point-wise expansion for FreqGatedMoEFFN.
    """

    def __init__(
        self,
        conv: nn.Module,             # pretrained depth-wise 7×7 ConvNormAct
        embed_dim: int,
        ffn_hidden: int,
        spatial_h: int,
        spatial_w: int,
        num_experts: int = 4,
        top_k: int = 2,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.moe_ffn = FreqGatedMoEFFN(
            embed_dim=embed_dim,
            ffn_hidden=ffn_hidden,
            num_experts=num_experts,
            top_k=top_k,
            drop=drop,
        )
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.last_routing_stats: dict | None = None
        self.route_mode = "identity"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local mixer first (pretrained) — retains FastVit inductive bias.
        x = self.conv(x)                                    # (B,C,H,W)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B,N,C)
        out, routing_stats = self.moe_ffn(
            tokens, H, W, route_mode=self.route_mode,
        )
        self.last_routing_stats = routing_stats
        return out.transpose(1, 2).reshape(B, C, H, W)

    @property
    def temperature(self) -> float:
        return self.moe_ffn.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.moe_ffn.temperature = value


class FastViTBackbone(nn.Module):
    """FastViT-SA12 backbone with 8-channel Gabor input and MoE-FFN
    inserted at 3 stages. Output keys match the TinyViT backbone so
    the rest of the pipeline (PAD head, identity head) is unchanged
    except for two stage dims (stage3 160→256, stage4 320→512).
    """

    # (stage_idx, block_idx, H, W, key)
    MOE_LOCATIONS = [
        (1, 1, 28, 28, 's2'),
        (2, 2, 14, 14, 's3a'),
        (2, 5, 14, 14, 's3b'),
    ]
    MOE_KEYS = ['s2', 's3a', 's3b']

    # MLP ratio = 4 (FastVit default)
    STAGE_DIMS = {
        1: (128, 512),
        2: (256, 1024),
    }

    OUT_DIMS = (64, 128, 256, 512)

    def __init__(
        self,
        pretrained: bool = True,
        in_chans: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
        drop: float = 0.0,
        use_grad_checkpoint: bool = False,
        model_name: str = 'fastvit_sa12',
    ) -> None:
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint
        self.model_name = model_name

        base = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.stem = base.stem
        self.stages = base.stages

        self._adapt_stem(in_chans, pretrained)

        self._moe_wrappers: List[ConvMoEMlpWrapper] = []
        for (stage_idx, block_idx, sh, sw, _key) in self.MOE_LOCATIONS:
            embed_dim, ffn_hidden = self.STAGE_DIMS[stage_idx]
            block = self.stages[stage_idx].blocks[block_idx]
            old_mlp = block.mlp                               # ConvMlp
            wrapper = ConvMoEMlpWrapper(
                conv=old_mlp.conv,                            # reuse pretrained dw7×7
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

    # ------------------------------------------------------------------
    # stem channel adaptation
    # ------------------------------------------------------------------

    def _adapt_stem(self, in_chans: int, pretrained: bool) -> None:
        """Replace the first Conv2d(3, 64, ...) of stem[0] with an
        in_chans→64 conv, copying pretrained weights for the first 3
        channels and seeding the rest from the mean + Gaussian noise so
        each extra Gabor channel starts different.
        """
        mb = self.stem[0]                                     # MobileOneBlock
        old_conv = mb.conv_kxk[0].conv                        # Conv2d(3, 64, 3, 2, 1)
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
                new_conv.weight[:, :3] = old_conv.weight
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                std = old_conv.weight.std().item() * 0.1
                for c in range(3, in_chans):
                    new_conv.weight[:, c:c+1] = mean_w + torch.randn_like(mean_w) * std
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        mb.conv_kxk[0].conv = new_conv

        # Parallel 1×1 shortcut path (conv_scale) also reads 3 input chs.
        if getattr(mb, 'conv_scale', None) is not None:
            old_s = mb.conv_scale.conv
            new_s = nn.Conv2d(
                in_chans,
                old_s.out_channels,
                kernel_size=old_s.kernel_size,
                stride=old_s.stride,
                padding=old_s.padding,
                bias=old_s.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    new_s.weight[:, :3] = old_s.weight
                    mean_s = old_s.weight.mean(dim=1, keepdim=True)
                    std_s = old_s.weight.std().item() * 0.1
                    for c in range(3, in_chans):
                        new_s.weight[:, c:c+1] = mean_s + torch.randn_like(mean_s) * std_s
            else:
                nn.init.kaiming_normal_(new_s.weight, mode='fan_out', nonlinearity='relu')
            mb.conv_scale.conv = new_s

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        route_mode: str = "identity",
    ) -> Dict[str, Any]:
        """
        Args:
            x: (B, in_chans, 224, 224)

        Returns dict:
            'stage1_feat':   (B,  64, 56, 56)   — stage 0 output
            'stage2_feat':   (B, 128, 28, 28)   — stage 1 output
            'stage3_feat':   (B, 256, 14, 14)   — stage 2 output
            'stage4_feat':   (B, 512,  7,  7)   — stage 3 output
            'routing_stats': {'s2':..., 's3a':..., 's3b':...}
            'balance_losses': [scalar, scalar, scalar]
        """
        x = self.stem(x)

        feats = {}
        use_ckpt = self.use_grad_checkpoint and self.training and x.requires_grad
        for i, stage in enumerate(self.stages):
            if use_ckpt:
                def _stage_forward(inp: torch.Tensor, stage_module: nn.Module = stage) -> torch.Tensor:
                    for wrapper in self._moe_wrappers:
                        wrapper.route_mode = route_mode
                    return stage_module(inp)

                x = torch.utils.checkpoint.checkpoint(
                    _stage_forward, x, use_reentrant=False,
                )
            else:
                for wrapper in self._moe_wrappers:
                    wrapper.route_mode = route_mode
                x = stage(x)
            feats[i] = x

        routing_stats: Dict[str, Any] = {}
        balance_losses: List[torch.Tensor] = []
        for key, wrapper in zip(self.MOE_KEYS, self._moe_wrappers):
            stats = wrapper.last_routing_stats
            if stats is not None:
                routing_stats[key] = {
                    'expert_weights':          stats['expert_weights'],
                    'token_entropy':           stats['token_entropy'],
                    'router_mode':             stats['router_mode'],
                    'expert_weights_gateonly': stats['expert_weights_gateonly'],
                    'token_entropy_gateonly':  stats['token_entropy_gateonly'],
                    'gate_input_gateonly':     stats['gate_input_gateonly'],
                }
                balance_losses.append(stats['balance_loss'])
            else:
                balance_losses.append(torch.tensor(0.0, device=x.device))

        return {
            'stage1_feat':    feats[0],
            'stage2_feat':    feats[1],
            'stage3_feat':    feats[2],
            'stage4_feat':    feats[3],
            'routing_stats':  routing_stats,
            'balance_losses': balance_losses,
        }

    # ------------------------------------------------------------------
    # param-group helpers (mirror TinyViTBackbone API)
    # ------------------------------------------------------------------

    def get_stage_params(self, stage_indices: List[int]) -> Iterator[nn.Parameter]:
        moe_ids = {id(p) for p in self.get_moe_params()}
        for idx in stage_indices:
            for p in self.stages[idx].parameters():
                if id(p) not in moe_ids:
                    yield p

    def get_moe_params(self) -> Iterator[nn.Parameter]:
        for wrapper in self._moe_wrappers:
            yield from wrapper.moe_ffn.parameters()

    def get_embed_params(self) -> Iterator[nn.Parameter]:
        yield from self.stem.parameters()

    def set_moe_temperature(self, temperature: float) -> None:
        for wrapper in self._moe_wrappers:
            wrapper.temperature = temperature

    def refresh_gateonly_stats(
        self,
        routing_stats: Dict[str, Any],
        route_mode: str,
    ) -> None:
        for key, wrapper in zip(self.MOE_KEYS, self._moe_wrappers):
            stats = routing_stats.get(key)
            if not stats or "gate_input_gateonly" not in stats:
                continue
            reproj = wrapper.moe_ffn.project_gateonly(
                stats["gate_input_gateonly"], route_mode,
            )
            stats.update(reproj)
            stats["router_mode"] = route_mode

    def freeze_stages(self, stage_indices: List[int]) -> None:
        for idx in stage_indices:
            for p in self.stages[idx].parameters():
                p.requires_grad = False

    def unfreeze_stages(self, stage_indices: List[int]) -> None:
        for idx in stage_indices:
            for p in self.stages[idx].parameters():
                p.requires_grad = True
