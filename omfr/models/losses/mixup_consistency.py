"""
mixup_consistency.py — MixUp manifold-consistency loss for PAD.

Enforces f(lam * x1 + (1-lam) * x2) ~= lam * f(x1) + (1-lam) * f(x2)
on pad_features. Acts as a smoothness regularizer on the PAD manifold,
much better suited to binary liveness than SupCon(τ=0.07).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class MixUpConsistency(nn.Module):
    """
    MixUp feature-consistency loss.

    Samples lam ~ Beta(alpha, alpha) per batch, builds a permutation
    and computes L = || f(lam x1 + (1-lam) x2) - (lam f(x1) + (1-lam) f(x2)) ||_2^2,
    averaged over the batch.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images:     (B, C, H, W)
            labels:     (B,) — unused but kept so callers can log together
            forward_fn: callable producing (B, D) pad_features for a given
                        image batch. Must include ALL trainable layers that
                        should be regularized (Gabor_pad -> PADStem -> fusion).
        """
        B = images.shape[0]
        if B < 2:
            return images.new_zeros(())

        lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample().item())
        # Avoid lam extremes — no signal at lam ~ 0 or 1
        lam = max(min(lam, 0.9), 0.1)

        perm = torch.randperm(B, device=images.device)
        mixed_images = lam * images + (1.0 - lam) * images[perm]

        feat_a_raw = forward_fn(images)
        feat_mixed_raw = forward_fn(mixed_images)

        # Promote to fp32 *before* any arithmetic: in 16-mixed the raw
        # pad_features can already hold values whose squares exceed the
        # fp16 max (~65504), so subtraction + .pow(2) overflows to inf.
        feat_a = feat_a_raw.float()
        feat_mixed = feat_mixed_raw.float()
        feat_b = feat_a[perm]
        target = lam * feat_a + (1.0 - lam) * feat_b

        loss = (feat_mixed - target).pow(2).sum(dim=-1).mean()
        return loss.to(feat_a_raw.dtype)
