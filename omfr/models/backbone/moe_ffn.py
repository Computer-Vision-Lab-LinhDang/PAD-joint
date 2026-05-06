"""
moe_ffn.py — Frequency-Gated Mixture-of-Experts FFN Block

Drop-in replacement for standard FFN in TinyViT blocks.
Adapts to variable embed_dim and spatial resolution per stage.

Config:
    num_experts: 4
    top_k:       2 (sparse routing)
    temperature: controlled by PhaseScheduler (2.0 → 1.0 → 0.5)

Balance loss uses Switch Transformer's f·P formulation for proper
gradient flow through soft routing probabilities.

I/O
---
    Input:  tokens (B, N, D), spatial_h, spatial_w
    Output: (output (B, N, D), routing_stats dict)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency_gate import FrequencyGate


class FreqGatedMoEFFN(nn.Module):
    """
    Frequency-Gated MoE-FFN with temperature-controlled routing.

    Routing pipeline:
        1. gate_input = FrequencyGate(tokens, H, W)    → (B, N, 3)
        2. gate_logits = Linear(3, E)(gate_input)       → (B, N, E)
        3. expert_weights = softmax(logits / τ)         → (B, N, E)
        4. top2 selection + renormalization
        5. output = Σ weight_k × Expert_k(tokens)
        6. balance_loss = E · Σ(f_e · P_e)  (Switch Transformer)

    Args:
        embed_dim:   token embedding dimension (128 for Stage 2, 160 for Stage 3)
        ffn_hidden:  FFN hidden dimension (embed_dim × mlp_ratio)
        num_experts: number of experts (default 4)
        top_k:       experts active per token (default 2)
        drop:        dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_hidden: int,
        num_experts: int = 4,
        top_k: int = 2,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_hidden = ffn_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = 1.0  # Controlled by PhaseScheduler

        # Frequency-based gating — 2-layer router so the gate can learn a
        # richer mapping from the 3-band energy features to E experts.
        self.freq_gate = FrequencyGate()
        self.gate_proj = self._build_router()
        self.pad_gate_proj = self._build_router()
        # PAD starts close to the identity routing policy and only learns
        # its own deviations when PAD supervision arrives.
        self.pad_gate_proj.load_state_dict(self.gate_proj.state_dict())
        self.active_router_mode = "identity"
        self.last_router_mode = "identity"
        # Noisy exploration is useful on identity routing, but it is too
        # unstable for the PAD router while we are still constraining PAD
        # gradients to the router path only.
        self.pad_noisy_gate_std = 0.0
        # PAD router gain starts small so the first PAD updates do not
        # immediately diverge from the identity routing template.
        self.pad_router_gain = nn.Parameter(torch.tensor(-2.2))
        # Blend raw PAD logits with the shared identity router. A small,
        # learnable delta is more stable than a fully independent PAD
        # router from step 1.
        self.pad_router_blend = nn.Parameter(torch.tensor(-2.2))
        # Restrict the PAD router to a low-rank correction on top of the
        # shared gate features. This keeps the experts shared while giving
        # PAD a task-specific routing degree of freedom.
        self.pad_router_delta = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, num_experts, bias=True),
        )
        # Noisy top-k exploration (Shazeer et al., 2017) — prevents early
        # monopoly by one or two experts. Disabled in eval.
        self.noisy_gate_std = 0.3

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(ffn_hidden, embed_dim),
                nn.Dropout(drop),
            )
            for _ in range(num_experts)
        ])

    def _build_router(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, self.num_experts, bias=True),
        )

    def _compute_gate_logits(
        self,
        gate_input: torch.Tensor,
        route_mode: str,
    ) -> torch.Tensor:
        if route_mode not in {"identity", "pad"}:
            raise ValueError(f"Unsupported route_mode={route_mode!r}")

        identity_logits = self.gate_proj(gate_input)
        if route_mode == "identity":
            return identity_logits

        pad_delta = self.pad_router_delta(gate_input)
        pad_gain = torch.sigmoid(self.pad_router_gain)
        pad_logits_raw = self.pad_gate_proj(gate_input) + pad_gain * pad_delta
        blend = torch.sigmoid(self.pad_router_blend)
        identity_anchor = identity_logits.detach()
        return identity_anchor + blend * (pad_logits_raw - identity_anchor)

    def _balance_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer balance loss: L = E · Σ(f_e · P_e).

        f_e = fraction of tokens dispatched to expert e (hard, non-differentiable)
        P_e = mean routing probability for expert e (soft, differentiable)

        Gradient flows through P_e, enabling the gate to learn balanced routing.
        Minimized when each expert handles ~25% of tokens.
        """
        B, N, E = expert_weights.shape

        # f: hard dispatch fraction per expert
        _, top_indices = expert_weights.topk(self.top_k, dim=-1)
        indicator = torch.zeros_like(expert_weights)
        indicator.scatter_(-1, top_indices, 1.0)
        f = indicator.float().mean(dim=(0, 1))  # (E,)

        # P: soft routing probability per expert (differentiable)
        P = expert_weights.mean(dim=(0, 1))  # (E,)

        return E * (f * P).sum()

    def project_gateonly(
        self,
        gate_input: torch.Tensor,
        route_mode: str,
    ) -> dict[str, torch.Tensor]:
        gate_logits = self._compute_gate_logits(gate_input.detach(), route_mode)
        expert_weights = F.softmax(
            gate_logits.float() / self.temperature, dim=-1,
        ).to(gate_logits.dtype)
        token_entropy = -(
            expert_weights * (expert_weights + 1e-8).log()
        ).sum(dim=-1)
        return {
            "expert_weights_gateonly": expert_weights,
            "token_entropy_gateonly": token_entropy,
            "gate_input_gateonly": gate_input.detach(),
        }

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_h: int,
        spatial_w: int,
        route_mode: str = "identity",
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            tokens:    (B, N, D) — input tokens
            spatial_h: spatial grid height
            spatial_w: spatial grid width

        Returns:
            output:        (B, N, D) — MoE FFN output
            routing_stats: dict with expert_weights, token_entropy, balance_loss
        """
        B, N, D = tokens.shape

        # Gating
        self.active_router_mode = route_mode
        gate_input = self.freq_gate(tokens, spatial_h, spatial_w)  # (B, N, 3)
        gate_logits = self._compute_gate_logits(gate_input, route_mode)

        # Noisy top-k: add Gaussian noise in training for exploration.
        noise_std = self.noisy_gate_std if route_mode == "identity" else self.pad_noisy_gate_std
        if self.training and noise_std > 0.0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * noise_std

        # Promote to fp32 for softmax/renorm stability under AMP fp16.
        expert_weights = F.softmax(
            gate_logits.float() / self.temperature, dim=-1,
        ).to(gate_logits.dtype)

        # Top-k selection + renormalization. Use a larger epsilon to stay
        # well above fp16's smallest positive (~6e-8).
        top_weights, top_indices = expert_weights.topk(self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-4)

        # Expert computation (sparse dispatch) — writes go through
        # index_add_ rather than in-place slice add to stay AMP-safe.
        tokens_flat = tokens.reshape(B * N, D)
        top_indices_flat = top_indices.reshape(B * N, self.top_k)
        top_weights_flat = top_weights.reshape(B * N, self.top_k)

        output_flat = torch.zeros_like(tokens_flat)

        for k_idx in range(self.top_k):
            expert_idx = top_indices_flat[:, k_idx]
            weight_k = top_weights_flat[:, k_idx]

            for e_idx in range(self.num_experts):
                mask = expert_idx == e_idx
                if not mask.any():
                    continue
                idx = mask.nonzero(as_tuple=False).squeeze(-1)
                expert_out = self.experts[e_idx](tokens_flat.index_select(0, idx))
                weighted = weight_k.index_select(0, idx).unsqueeze(-1) * expert_out
                # index_add requires matching scalar types. Under AMP, the
                # Linear experts can emit fp32 while output_flat tracks the
                # input dtype (fp16), so cast back to keep them aligned.
                output_flat = output_flat.index_add(0, idx, weighted.to(output_flat.dtype))

        output = output_flat.reshape(B, N, D)

        # Routing statistics
        token_entropy = -(
            expert_weights * (expert_weights + 1e-8).log()
        ).sum(dim=-1)  # (B, N)
        balance_loss = self._balance_loss(expert_weights)

        # Gate-only routing copy: re-project from a DETACHED gate_input
        # so downstream consumers (PAD head) can backprop into gate_proj
        # weights without the gradient escaping upstream into freq_gate
        # → tokens → backbone. Only needed during training — in eval
        # we alias the already-computed expert_weights (detached by the
        # caller anyway) to avoid an extra Linear pass and the
        # activation memory that comes with it.
        if self.training:
            gate_logits_go = self._compute_gate_logits(gate_input.detach(), route_mode)
            expert_weights_go = F.softmax(
                gate_logits_go.float() / self.temperature, dim=-1,
            ).to(gate_logits_go.dtype)
            token_entropy_go = -(
                expert_weights_go * (expert_weights_go + 1e-8).log()
            ).sum(dim=-1)
        else:
            expert_weights_go = expert_weights
            token_entropy_go = token_entropy

        self.last_router_mode = route_mode
        routing_stats = {
            "expert_weights": expert_weights,  # (B, N, E)
            "token_entropy": token_entropy,    # (B, N)
            "balance_loss": balance_loss,      # scalar
            "router_mode": route_mode,
            # Gate-only views — grad stops at gate_proj params.
            "expert_weights_gateonly": expert_weights_go,
            "token_entropy_gateonly":  token_entropy_go,
            # Thêm tín hiệu gốc (3 dải tần) để cứu nhánh PAD khỏi bẫy Temperature
            "gate_input_gateonly": gate_input.detach(), # <--- THÊM DÒNG NÀY, 
        }

        return output, routing_stats
