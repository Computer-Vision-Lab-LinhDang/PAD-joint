"""
mixup_consistency.py — MixUp-based manifold consistency for PAD features

Enforces that the PAD feature extractor is approximately linear on the
image interpolation manifold:

    f(lam*x_a + (1-lam)*x_b) ≈ lam*f(x_a) + (1-lam)*f(x_b)

This is a much better-suited regularizer for binary PAD than SupCon:

- It doesn't require a notion of "positives" in feature space (which is
  trivially degenerate with 2 classes).
- It adds a smoothness prior exactly along the direction where sensor
  shortcuts tend to be sharp (a model that memorizes sensor signature
  has very non-linear decision boundaries; mixup penalizes this).
- The strength is easy to control with a single `alpha` parameter of
  the Beta mixing distribution.

Reference:
    Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
    Verma et al., "Manifold Mixup", ICML 2019 (feature-space variant).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixUpConsistency(nn.Module):
    """Consistency loss between mixed-input features and mixed features.

    For each batch of images `x` in R^(B, C, H, W) and labels (unused here):

        1. Draw lam ~ Beta(alpha, alpha) and a permutation `perm` of B.
        2. Compute x_mix = lam * x + (1 - lam) * x[perm].
        3. Compute f(x), f(x_mix), f(x[perm]) via `forward_fn`.
        4. Target = lam * f(x) + (1 - lam) * f(x[perm]).
        5. Loss = ||f(x_mix) - Target||_2^2  (per-sample, then mean).

    Because the loss is on the feature space (not logits), the gradient
    flows back through every layer of `forward_fn`. The alpha parameter
    controls how aggressive the mixing is:

        alpha=0.2 → most lam values near 0 or 1 (mild)
        alpha=0.4 → moderate mixing (recommended)
        alpha=1.0 → uniform in [0, 1] (aggressive)

    Args:
        alpha: Beta distribution concentration. Higher = more mixing.
        normalize: if True, L2-normalize features before computing MSE.
            Useful when features already have unit norm (e.g., the loss
            becomes a cosine-distance penalty).
    """

    def __init__(self, alpha: float = 0.4, normalize: bool = False) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.alpha = alpha
        self.normalize = normalize

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images:     (B, C, H, W) — input images (live/spoof mixed batch).
            labels:     (B,)         — liveness labels; unused for loss value
                                        but the signature matches the
                                        training-step convention.
            forward_fn: (B, C, H, W) → (B, D) function producing PAD features.

        Returns:
            scalar loss.
        """
        B = images.shape[0]
        if B < 2:
            return images.new_zeros(())

        device = images.device
        # Sample lam from Beta(alpha, alpha). Scalar lam (same for the whole
        # batch) keeps things simple and behaves well in practice; per-sample
        # lam can be added later if needed.
        lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample().item())
        # Clamp away from 0/1 to avoid trivial zero loss when lam is exactly
        # 0 or 1 (x_mix = x or x_mix = x_perm).
        lam = max(min(lam, 1.0 - 1e-3), 1e-3)

        perm = torch.randperm(B, device=device)

        # Mixed input
        x_mix = lam * images + (1.0 - lam) * images[perm]

        # Three forward passes. We avoid concatenating into a big tensor so
        # GroupNorm statistics stay per-sub-batch identical (relevant because
        # PADStem uses GN, but just good practice).
        f_a = forward_fn(images)                 # (B, D)
        f_b = f_a.index_select(0, perm)          # free — same forward output
        f_mix = forward_fn(x_mix)                # (B, D)

        if self.normalize:
            f_a   = F.normalize(f_a,   p=2, dim=-1)
            f_b   = F.normalize(f_b,   p=2, dim=-1)
            f_mix = F.normalize(f_mix, p=2, dim=-1)

        # Target: linear interpolation in feature space.
        target = lam * f_a + (1.0 - lam) * f_b

        # Per-sample squared error, mean across batch and feature dim.
        return F.mse_loss(f_mix, target)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, normalize={self.normalize}"
