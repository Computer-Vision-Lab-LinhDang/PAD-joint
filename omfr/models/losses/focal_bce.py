"""
focal_bce.py — Focal Binary Cross-Entropy for PAD.

Replaces SupCon(τ=0.07)+BCE for the PAD branch. Reason: SupCon on a
2-class problem with the small τ is degenerate — observed plateau at
~4.5 (TASK_01). Plain BCE underweights hard silicone / high-quality
spoof samples; focal loss recovers that signal by reweighting with
(1 - p_t)^gamma.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """
    Focal binary cross-entropy.

    L = - alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = p          if y == 1
            = 1 - p      if y == 0
        alpha_t = alpha     if y == 1
                = 1 - alpha if y == 0
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B,)  raw logits (not sigmoid)
            targets: (B,)  float 0/1 targets

            sample_weight: optional per-sample weights.

        Returns:
            scalar weighted mean focal BCE loss.
        """
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        loss = focal_weight * bce
        if sample_weight is not None:
            weight = sample_weight.to(device=loss.device, dtype=loss.dtype).reshape_as(loss)
            return (loss * weight).sum() / weight.sum().clamp_min(1.0)
        return loss.mean()
