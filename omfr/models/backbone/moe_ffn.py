"""
moe_ffn.py — Frequency-Gated Mixture-of-Experts FFN Block

Input:  tokens (B, 196, 192) — attention output
Output: tokens (B, 196, 192) — same shape (residual added outside in ViT block)
Side:   routing_stats dict {
            'expert_weights': (B, 196, 4),
            'token_entropy':  (B, 196),
            'balance_loss':   scalar,
        }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency_gate import FrequencyGate


class FreqGatedMoEFFN(nn.Module):
    """
    Frequency-Gated MoE-FFN — replaces standard FFN at ViT layers {3, 7, 10}.

    Config:
        num_experts:  4
        top_k:        2   (sparse routing — top-2 of 4 experts active per token)
        embed_dim:    192
        ffn_hidden:   768 (mlp_ratio = 4.0)

    Parameter budget (per MoE layer):
        Standard FFN: 192×768×2 ≈ 295K params
        MoE-FFN:      4 × 295K + gate Linear(3→4) ≈ 1.18M params

    FLOPs per token: ~2× standard FFN (top-2 of 4 experts active)

    Routing pipeline:
        1. gate_input = FrequencyGate(tokens)               (B, 196, 3)
        2. gate_logits = Linear(3→4)(gate_input)            (B, 196, 4)
        3. expert_weights = Softmax(gate_logits)            (B, 196, 4)
        4. top2_indices, top2_weights = TopK(expert_weights, k=2)
        5. output = Σ_{k∈top2} weight_k × Expert_k(tokens) (B, 196, 192)
        6. token_entropy = -Σ p_k · log(p_k + ε)          (B, 196)
        7. balance_loss = CV²(expert_load)                  scalar

    Expert load: fraction of tokens routed to each expert (target ≈ 25%).
    L_balance penalizes coefficient of variation → uniform expert utilization.

    I/O:
        Input:  tokens (B, 196, 192)
        Output: (output_tokens, routing_stats)
            output_tokens  — (B, 196, 192)
            routing_stats  — dict with 'expert_weights', 'token_entropy', 'balance_loss'
    """

    def __init__(
        self,
        embed_dim: int = 192,
        ffn_hidden: int = 768,
        num_experts: int = 4,
        top_k: int = 2,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_hidden = ffn_hidden
        self.num_experts = num_experts
        self.top_k = top_k

        # Frequency-based gating network
        self.freq_gate = FrequencyGate(embed_dim=embed_dim)

        # Gate projection: band energies (3) → expert logits (num_experts)
        self.gate_proj = nn.Linear(3, num_experts, bias=False)

        # Expert FFNs: Linear(D, 4D) -> GELU -> Linear(4D, D)
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

    def _balance_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute load-balancing loss = CV²(expert_load).

        expert_load[e] = fraction of (B*N) tokens where expert e is in top-k.

        Args:
            expert_weights: (B, N, E) — full softmax distribution over experts

        Returns:
            scalar balance loss
        """
        B, N, E = expert_weights.shape
        # Top-k indicator: (B, N, E) binary — 1 if expert is selected
        _, top_indices = expert_weights.topk(self.top_k, dim=-1)  # (B, N, top_k)
        indicator = torch.zeros_like(expert_weights)
        indicator.scatter_(-1, top_indices, 1.0)  # (B, N, E)

        # Expert load = mean over all tokens
        expert_load = indicator.mean(dim=(0, 1))  # (E,)

        # Coefficient of variation squared: Var / Mean²
        mean_load = expert_load.mean()
        var_load = expert_load.var()
        cv2 = var_load / (mean_load ** 2 + 1e-8)
        return cv2

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: (B, 196, 192) — input token sequence (after self-attention)

        Returns:
            output:        (B, 196, 192) — MoE FFN output (residual added in ViT block)
            routing_stats: dict {
                'expert_weights': (B, 196, 4),
                'token_entropy':  (B, 196),
                'balance_loss':   scalar tensor,
            }
        """
        B, N, D = tokens.shape

        # --- Gating ---
        gate_input = self.freq_gate(tokens)              # (B, 196, 3)
        gate_logits = self.gate_proj(gate_input)         # (B, 196, 4)
        expert_weights = F.softmax(gate_logits, dim=-1)  # (B, 196, 4)

        # Top-k selection
        top_weights, top_indices = expert_weights.topk(self.top_k, dim=-1)
        # Renormalize top-k weights so they sum to 1
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # --- Expert computation (token-by-token sparse dispatch) ---
        # Flatten tokens for easier indexing
        tokens_flat = tokens.reshape(B * N, D)         # (B*N, D)
        top_indices_flat = top_indices.reshape(B * N, self.top_k)  # (B*N, top_k)
        top_weights_flat = top_weights.reshape(B * N, self.top_k)  # (B*N, top_k)

        output_flat = torch.zeros_like(tokens_flat)  # (B*N, D)

        for k_idx in range(self.top_k):
            expert_idx = top_indices_flat[:, k_idx]   # (B*N,)
            weight_k = top_weights_flat[:, k_idx]     # (B*N,)

            for e_idx in range(self.num_experts):
                # Mask tokens routed to expert e
                mask = (expert_idx == e_idx)  # (B*N,) bool
                if not mask.any():
                    continue
                tokens_e = tokens_flat[mask]                     # (m, D)
                expert_out = self.experts[e_idx](tokens_e)       # (m, D)
                weight_e = weight_k[mask].unsqueeze(-1)          # (m, 1)
                output_flat[mask] += weight_e * expert_out

        output = output_flat.reshape(B, N, D)  # (B, 196, 192)

        # --- Routing statistics ---
        # Token entropy: H(p) = -Σ p_k · log(p_k)
        token_entropy = -(expert_weights * (expert_weights + 1e-8).log()).sum(dim=-1)  # (B, 196)

        # Balance loss
        balance_loss = self._balance_loss(expert_weights)

        routing_stats = {
            "expert_weights": expert_weights,  # (B, 196, 4)
            "token_entropy": token_entropy,    # (B, 196)
            "balance_loss": balance_loss,      # scalar
        }

        return output, routing_stats
