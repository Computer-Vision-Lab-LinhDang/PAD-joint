"""
dinov2.py - DINOv2 ViT backbone adapter.

The rest of OMFR expects a hierarchical backbone API with stage1..stage4
feature maps plus MoE-style routing stats. DINOv2 is a plain ViT, so this
adapter exposes compatible feature maps from intermediate patch-token maps and
returns neutral routing stats for the PAD head's existing auxiliary route path.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import (
    TaskConditionalAdaDoRALinear,
    collect_orthogonality_loss,
    count_trainable,
    freeze_non_lora,
    inject_adadora,
    lora_parameters,
    set_lora_task,
)
from .pad_shallow_moe import PADShallowMoE


class DINOv2Backbone(nn.Module):
    """DINOv2 patch-14 ViT adapter for OMFR.

    Output keys match TinyViT/FastViT:
        stage1_feat: (B,  64, 56, 56)
        stage2_feat: (B, 128, 28, 28)
        stage3_feat: (B,   C, 16, 16) for 224 input and patch14
        stage4_feat: (B,   C,  8,  8) downsampled late tokens

    The stage3/stage4 channel count depends on the DINOv2 model size
    (384 for vit_small_patch14_dinov2, 768 for base, etc.).
    """

    MOE_KEYS = ["s2", "s3a", "s3b"]

    def __init__(
        self,
        pretrained: bool = True,
        in_chans: int = 8,
        model_name: str = "vit_small_patch14_dinov2",
        img_size: int = 224,
        use_grad_checkpoint: bool = False,
        stage1_dim: int = 64,
        stage2_dim: int = 128,
        intermediate_indices: Sequence[int] | None = None,
        lora_cfg: Optional[Dict[str, Any]] = None,
        use_pad_shallow_moe: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        if hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(use_grad_checkpoint)

        self.embed_dim = int(getattr(self.model, "embed_dim"))
        self.out_dims = (stage1_dim, stage2_dim, self.embed_dim, self.embed_dim)

        self._adapt_patch_embed(in_chans=in_chans, pretrained=pretrained)

        num_blocks = len(self.model.blocks)
        if intermediate_indices is None:
            # Four roughly even snapshots through the transformer.
            intermediate_indices = (
                max(num_blocks // 4 - 1, 0),
                max(num_blocks // 2 - 1, 0),
                max((3 * num_blocks) // 4 - 1, 0),
                num_blocks - 1,
            )
        self.intermediate_indices = tuple(int(i) for i in intermediate_indices)
        self._stage_slices = self._build_stage_slices(num_blocks)

        self.stage1_proj = nn.Conv2d(self.embed_dim, stage1_dim, kernel_size=1)
        self.stage2_proj = nn.Conv2d(self.embed_dim, stage2_dim, kernel_size=1)

        # ----- LoRA (DoRA + AdaLoRA-flavor) injection -----
        # Replaces nn.Linear inside each ViT block with task-conditional
        # adapters and freezes the rest of the timm trunk so only the
        # adapters and the OMFR-side projections train.
        self.lora_cfg = dict(lora_cfg or {})
        self.use_lora: bool = bool(self.lora_cfg.get("enabled", False))
        self.lora_tasks: List[str] = list(self.lora_cfg.get("tasks", ["identity", "pad"]))
        if self.use_lora:
            target_modules = list(
                self.lora_cfg.get(
                    "target_modules", ["qkv", "proj", "fc1", "fc2"]
                )
            )
            rank = int(self.lora_cfg.get("rank", 16))
            alpha = float(self.lora_cfg.get("alpha", 32.0))
            dropout = float(self.lora_cfg.get("dropout", 0.0))
            n_replaced = inject_adadora(
                self.model.blocks,
                target_leaf_names=target_modules,
                rank=rank,
                alpha=alpha,
                tasks=self.lora_tasks,
                dropout=dropout,
            )
            # Freeze the timm trunk (blocks + norm + cls/pos tokens). The
            # adapted patch_embed remains trainable so the 8-ch Gabor input
            # can be absorbed.
            freeze_non_lora(self.model.blocks)
            for p in self.model.norm.parameters():
                p.requires_grad = False
            for name in ("cls_token", "pos_embed", "reg_token"):
                tok = getattr(self.model, name, None)
                if isinstance(tok, nn.Parameter):
                    tok.requires_grad = False
            self._lora_stats = {
                "replaced": n_replaced,
                "rank": rank,
                "alpha": alpha,
                "tasks": tuple(self.lora_tasks),
            }

        # PAD-specific shallow MoE: replaces dummy routing_stats for route_mode="pad"
        self.pad_shallow_moe: Optional[PADShallowMoE] = None
        if use_pad_shallow_moe:
            self.pad_shallow_moe = PADShallowMoE(
                stage2_dim=stage2_dim,
                stage3_dim=self.embed_dim,
            )

    @staticmethod
    def _build_stage_slices(num_blocks: int) -> List[range]:
        cuts = [
            0,
            num_blocks // 4,
            num_blocks // 2,
            (3 * num_blocks) // 4,
            num_blocks,
        ]
        return [range(cuts[i], cuts[i + 1]) for i in range(4)]

    def _adapt_patch_embed(self, in_chans: int, pretrained: bool) -> None:
        old_conv = self.model.patch_embed.proj
        if old_conv.in_channels == in_chans:
            return

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
                copy_ch = min(old_conv.in_channels, in_chans)
                new_conv.weight[:, :copy_ch] = old_conv.weight[:, :copy_ch]
                if in_chans > old_conv.in_channels:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    std = old_conv.weight.std().item() * 0.1
                    for c in range(old_conv.in_channels, in_chans):
                        new_conv.weight[:, c : c + 1] = (
                            mean_w + torch.randn_like(mean_w) * std
                        )
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)
        self.model.patch_embed.proj = new_conv

    @staticmethod
    def _dummy_routing_stats(ref: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        bsz = ref.shape[0]
        expert_weights = ref.new_full((bsz, num_tokens, 4), 0.25)
        token_entropy = ref.new_zeros((bsz, num_tokens))
        gate_input = ref.new_zeros((bsz, num_tokens, 3))
        return {
            "expert_weights": expert_weights,
            "token_entropy": token_entropy,
            "router_mode": "shared",
            "expert_weights_gateonly": expert_weights,
            "token_entropy_gateonly": token_entropy,
            "gate_input_gateonly": gate_input,
        }

    def forward(self, x: torch.Tensor, route_mode: str = "identity") -> Dict[str, Any]:
        # When LoRA is active route_mode picks the task-conditional adapter
        # set; otherwise DINOv2 is a shared trunk and the flag is a no-op.
        if self.use_lora:
            task = route_mode if route_mode in self.lora_tasks else self.lora_tasks[0]
            set_lora_task(self.model.blocks, task)

        feats = self.model.forward_intermediates(
            x,
            indices=self.intermediate_indices,
            intermediates_only=True,
        )
        early_a, early_b, mid, late = feats

        stage1_feat = F.interpolate(
            self.stage1_proj(early_a),
            size=(56, 56),
            mode="bilinear",
            align_corners=False,
        )
        stage2_feat = F.interpolate(
            self.stage2_proj(early_b),
            size=(28, 28),
            mode="bilinear",
            align_corners=False,
        )
        stage3_feat = mid
        stage4_feat = F.adaptive_avg_pool2d(
            late,
            output_size=(
                max(late.shape[-2] // 2, 1),
                max(late.shape[-1] // 2, 1),
            ),
        )

        zero = stage3_feat.new_zeros(())
        if route_mode == "pad" and self.pad_shallow_moe is not None:
            routing_stats, balance_losses = self.pad_shallow_moe(stage2_feat, stage3_feat)
        else:
            s2_tokens = stage2_feat.shape[-2] * stage2_feat.shape[-1]
            s3_tokens = stage3_feat.shape[-2] * stage3_feat.shape[-1]
            routing_stats = {
                "s2":  self._dummy_routing_stats(stage2_feat, s2_tokens),
                "s3a": self._dummy_routing_stats(stage3_feat, s3_tokens),
                "s3b": self._dummy_routing_stats(stage3_feat, s3_tokens),
            }
            balance_losses = [zero, zero, zero]

        return {
            "stage1_feat": stage1_feat,
            "stage2_feat": stage2_feat,
            "stage3_feat": stage3_feat,
            "stage4_feat": stage4_feat,
            "routing_stats": routing_stats,
            "balance_losses": balance_losses,
        }

    def get_stage_params(self, stage_indices: List[int]) -> Iterator[nn.Parameter]:
        for idx in stage_indices:
            if idx == 0:
                yield from self.stage1_proj.parameters()
            elif idx == 1:
                yield from self.stage2_proj.parameters()

            if 0 <= idx < len(self._stage_slices):
                for block_idx in self._stage_slices[idx]:
                    yield from self.model.blocks[block_idx].parameters()
            if idx == 3:
                yield from self.model.norm.parameters()

    def get_moe_params(self) -> Iterator[nn.Parameter]:
        if self.pad_shallow_moe is not None:
            yield from self.pad_shallow_moe.parameters()

    # ------------------------------------------------------------------
    # LoRA helpers
    # ------------------------------------------------------------------
    def get_lora_params(
        self, task: Optional[str] = None, include_magnitude: bool = True
    ) -> Iterator[nn.Parameter]:
        """Yield trainable LoRA params under the timm trunk.

        Args:
            task: optional task id ("identity" / "pad"); ``None`` yields params
                for all registered tasks.
            include_magnitude: if True, include the DoRA magnitude vectors.
        """
        if not self.use_lora:
            return iter(())
        return lora_parameters(
            self.model.blocks, task=task, include_magnitude=include_magnitude
        )

    def get_lora_orthogonality_loss(self) -> torch.Tensor:
        """AdaLoRA-style P/Q orthogonality regularizer (active task only)."""
        if not self.use_lora:
            return torch.zeros((), device=self.stage1_proj.weight.device)
        return collect_orthogonality_loss(self.model.blocks)

    def lora_param_summary(self) -> Dict[str, Any]:
        if not self.use_lora:
            return {"enabled": False}
        trainable, total = count_trainable(self.model)
        return {"enabled": True, "trainable": trainable, "total": total, **self._lora_stats}

    def get_embed_params(self) -> Iterator[nn.Parameter]:
        yield from self.model.patch_embed.parameters()
        for name in ("cls_token", "pos_embed", "reg_token"):
            param = getattr(self.model, name, None)
            if isinstance(param, nn.Parameter):
                yield param

    def set_moe_temperature(self, temperature: float) -> None:
        if self.pad_shallow_moe is not None:
            self.pad_shallow_moe.set_temperature(temperature)

    def refresh_gateonly_stats(
        self,
        routing_stats: Dict[str, Any],
        route_mode: str,
    ) -> None:
        if route_mode != "pad" or self.pad_shallow_moe is None:
            return
        result = self.pad_shallow_moe.refresh_gateonly()
        if result is None:
            return
        go_s2, go_s3 = result
        routing_stats["s2"].update(go_s2)
        routing_stats["s3a"].update(go_s3)
        routing_stats["s3b"].update(go_s3)

    def freeze_stages(self, stage_indices: List[int]) -> None:
        for param in self.get_stage_params(stage_indices):
            param.requires_grad = False

    def unfreeze_stages(self, stage_indices: List[int]) -> None:
        for param in self.get_stage_params(stage_indices):
            param.requires_grad = True
