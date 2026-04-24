"""
focal_bce.py — Focal BCE Loss for PAD

Replaces SupCon(τ=0.07) + vanilla BCE for the liveness classification
branch. Focal loss weights hard examples up and easy examples down:

    FL(p_t) = -alpha_t · (1 - p_t)^gamma · log(p_t)

where p_t = sigmoid(logit) for positives, 1 - sigmoid(logit) for negatives,
and alpha_t balances positive/negative prior.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

Why this fits fingerprint PAD:
- Binary classification, so 2-class SupCon geometry is trivially degenerate.
- LivDet has a wide spectrum of "how fake" spoofs are: easy (visible latex
  edge) to hard (high-res silicone on optical sensor). Focal loss keeps
  the gradient concentrated on the hard samples that actually distinguish
  sensors at test time.
- alpha_t = 0.5 keeps the balanced-batch prior neutral (since
  BalancedPADSampler already enforces 50/50 at sample level).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Focal binary cross-entropy with logits.

    Args:
        gamma: focusing parameter. 2.0 is the RetinaNet default; larger
            values put even more weight on hard examples but can destabilize
            training on small datasets. Range [0.0, 5.0] is sensible.
        alpha: class-balance weight for positives in [0, 1]. 0.5 means
            live and spoof contribute equally. Use 0.25 if positives are
            rare in your sampler.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be mean/sum/none, got {reduction}")

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B,) raw logits (before sigmoid)
            targets: (B,) float 0.0 / 1.0

        Returns:
            scalar loss (unless reduction='none')
        """
        # Compute per-element BCE without reduction. Use the logits form
        # for numerical stability (never materialize sigmoid outputs for
        # the log).
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none",
        )  # (B,)

        # p_t = sigmoid(logits) for positives, 1 - sigmoid(logits) for negatives.
        # Computed from exp(-bce) for numerical stability:
        # if y=1: bce = -log(p)    →  p = exp(-bce)    → p_t = p
        # if y=0: bce = -log(1-p)  →  (1-p) = exp(-bce) → p_t = 1-p = exp(-bce)
        p_t = torch.exp(-bce)

        focal_weight = (1.0 - p_t).pow(self.gamma)

        # alpha_t = alpha for positive, (1 - alpha) for negative
        alpha_t = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, alpha={self.alpha}, reduction={self.reduction!r}"
