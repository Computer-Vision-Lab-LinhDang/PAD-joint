"""
pad_shallow_moe.py — PAD-specific shallow MoE for DINOv2 backbone.

DINOv2 is a plain ViT without MoE layers, so the routing statistics that
PADHead._extract_routing_features() expects are placeholders (dummy).  This
module adds two FreqGatedMoEFFN layers that run on projected stage2 and stage3
tokens when route_mode="pad", producing real routing diversity signal.

Live fingerprints: diverse expert routing (varied micro-texture).
Spoof fingerprints: uniform routing (synthetic / replicated texture).

The two layers map to the three routing-stat slots that PADHead expects:
    s2  → moe_s2  (stage2, 28×28 tokens, embed_dim=stage2_dim)
    s3a → moe_s3  (stage3, 16×16 tokens, embed_dim=stage3_dim)
    s3b → aliased to s3a  (no third layer; same routing signal)

Gradient path in Phase 2 (backbone_no_grad=True):
    forward() runs under torch.no_grad — gate_inputs are cached as .detach().
    refresh_gateonly() is called outside no_grad and re-runs only the gate
    projections so grad flows: loss → routing features → gate_proj weights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .moe_ffn import FreqGatedMoEFFN


class PADShallowMoE(nn.Module):
    """Two-layer PAD-specific FreqGatedMoEFFN stack for DINOv2.

    Args:
        stage2_dim:  channel count of stage2_feat (default 128).
        stage3_dim:  channel count of stage3_feat = DINOv2 embed_dim (default 384).
        ffn_ratio:   hidden-dim multiplier for expert FFNs (default 2.0).
        num_experts: number of MoE experts (default 4).
        top_k:       active experts per token (default 2).
    """

    def __init__(
        self,
        stage2_dim: int = 128,
        stage3_dim: int = 384,
        ffn_ratio: float = 2.0,
        num_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.moe_s2 = FreqGatedMoEFFN(
            embed_dim=stage2_dim,
            ffn_hidden=int(stage2_dim * ffn_ratio),
            num_experts=num_experts,
            top_k=top_k,
        )
        self.moe_s3 = FreqGatedMoEFFN(
            embed_dim=stage3_dim,
            ffn_hidden=int(stage3_dim * ffn_ratio),
            num_experts=num_experts,
            top_k=top_k,
        )
        # Cached detached gate_inputs from the most recent forward().
        # refresh_gateonly() re-runs gate projections outside no_grad
        # to give gate_proj weights a live gradient path.
        self._cached_gate_input_s2: Optional[torch.Tensor] = None
        self._cached_gate_input_s3: Optional[torch.Tensor] = None

    def forward(
        self,
        stage2_feat: torch.Tensor,
        stage3_feat: torch.Tensor,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[torch.Tensor]]:
        """
        Args:
            stage2_feat: (B, C2, H2, W2) — projected stage2 feature map
            stage3_feat: (B, C3, H3, W3) — stage3 patch-token map

        Returns:
            routing_stats:  dict with keys "s2", "s3a", "s3b"
            balance_losses: list of 3 scalar tensors [s2, s3, zero]
        """
        zero = stage2_feat.new_zeros(())

        # stage2: (B, C2, H2, W2) → (B, H2*W2, C2) tokens
        s2_h, s2_w = stage2_feat.shape[-2:]
        s2_tokens = stage2_feat.flatten(2).permute(0, 2, 1).contiguous()
        _, stats_s2 = self.moe_s2(s2_tokens, s2_h, s2_w, route_mode="pad")
        # FreqGatedMoEFFN stores gate_input as .detach() under this key
        self._cached_gate_input_s2 = stats_s2["gate_input_gateonly"]

        # stage3: (B, C3, H3, W3) → (B, H3*W3, C3) tokens
        s3_h, s3_w = stage3_feat.shape[-2:]
        s3_tokens = stage3_feat.flatten(2).permute(0, 2, 1).contiguous()
        _, stats_s3 = self.moe_s3(s3_tokens, s3_h, s3_w, route_mode="pad")
        self._cached_gate_input_s3 = stats_s3["gate_input_gateonly"]

        routing_stats: Dict[str, Dict[str, Any]] = {
            "s2":  stats_s2,
            "s3a": stats_s3,
            "s3b": stats_s3,   # aliased — PADHead reads s3a/s3b identically
        }
        # s3b aliased to s3a: only count s3 balance loss once
        balance_losses: List[torch.Tensor] = [
            stats_s2["balance_loss"],
            stats_s3["balance_loss"],
            zero,
        ]
        return routing_stats, balance_losses

    def refresh_gateonly(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Re-run gate projections outside no_grad using cached gate_inputs.

        Called by DINOv2Backbone.refresh_gateonly_stats() after a
        backbone_no_grad forward pass. Returns (stats_go_s2, stats_go_s3)
        with live-grad expert_weights_gateonly / token_entropy_gateonly,
        or None if no cached inputs are available (first call, eval mode).
        """
        if self._cached_gate_input_s2 is None or self._cached_gate_input_s3 is None:
            return None
        stats_go_s2 = self.moe_s2.project_gateonly(self._cached_gate_input_s2, "pad")
        stats_go_s3 = self.moe_s3.project_gateonly(self._cached_gate_input_s3, "pad")
        return stats_go_s2, stats_go_s3

    def set_temperature(self, temperature: float) -> None:
        self.moe_s2.temperature = temperature
        self.moe_s3.temperature = temperature
