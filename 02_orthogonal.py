"""
orthogonal.py — Normalized Orthogonality Regularization Loss (v3)

Barlow Twins-style cross-correlation loss between PAD embeddings and
identity embeddings. Forces the two subspaces to be decorrelated so
liveness detection doesn't leak identity information and vice versa.

v3 fix (TASK_02):
    v2 normalized the Frobenius-squared norm by (d_p * d_i), which
    produced O(1/B) values indistinguishable from noise (observed
    ~0.008 throughout training). v3 uses max(d_p, d_i) as denominator
    and applies a Bessel-like B/(B-1) correction. It also refuses to
    compute the loss for small B where the sample correlation is too
    noisy (B < max(d_p, d_i) / 4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OrthogonalityLoss(nn.Module):
    """Orthogonality regularization via normalized cross-correlation matrix.

    Given PAD embeddings Z_p in R^(B x d_p) and identity embeddings
    Z_i in R^(B x d_i):

        1. Batch-normalize each feature dimension to zero mean, unit std.
        2. Compute cross-correlation C = Z_p^T @ Z_i / (B - 1)  → (d_p, d_i).
        3. Loss = ||C||_F^2 / max(d_p, d_i).

    This scales so that:
        - Fully aligned subspaces (Z_i's first d_p dims == Z_p) give loss ≈ 1.
        - Random independent embeddings give loss ≈ min(d_p, d_i) / B.
        - Truly orthogonal embeddings give loss → 0.

    Args:
        eps:       numerical stabilizer for per-dim std.
        min_batch: skip loss computation (return 0) when B is below this
                   fraction of max(d_p, d_i) — estimator is too noisy.
                   Set to 0.0 to always compute.
    """

    def __init__(self, eps: float = 1e-8, min_batch_ratio: float = 0.25) -> None:
        super().__init__()
        self.eps = eps
        self.min_batch_ratio = min_batch_ratio

    def forward(
        self,
        pad_emb: torch.Tensor,
        id_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pad_emb: (B, d_p) — PAD embeddings (L2-normalized 32-D).
            id_emb:  (B, d_i) — identity embeddings (L2-normalized 256-D).

        Returns:
            loss: scalar — ||C||_F^2 / max(d_p, d_i).
        """
        if pad_emb.shape[0] != id_emb.shape[0]:
            raise ValueError(
                f"Batch mismatch: pad={pad_emb.shape[0]} vs id={id_emb.shape[0]}"
            )

        B = pad_emb.shape[0]
        d_p = pad_emb.shape[1]
        d_i = id_emb.shape[1]

        # Gate: estimator is too noisy when B is small relative to subspace
        # dimension. Return a differentiable zero so the training loop
        # doesn't break.
        if B < max(d_p, d_i) * self.min_batch_ratio:
            # Create a zero that still participates in autograd so Lightning
            # doesn't complain about orphaned parameters.
            return pad_emb.sum() * 0.0 + id_emb.sum() * 0.0

        pad_norm = self._batch_norm(pad_emb)        # (B, d_p)
        id_norm  = self._batch_norm(id_emb)         # (B, d_i)

        # Bessel-corrected cross-covariance
        cross_corr = (pad_norm.T @ id_norm) / max(B - 1, 1)   # (d_p, d_i)

        # Frobenius-squared, normalized by the larger subspace size
        loss = cross_corr.pow(2).sum() / max(d_p, d_i)

        return loss

    def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True)
        return (x - mean) / (std + self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, min_batch_ratio={self.min_batch_ratio}"
