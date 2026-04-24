"""
orthogonal.py — Normalized Orthogonality Regularization Loss (v2)

Barlow Twins-style cross-correlation loss between PAD embeddings and
identity embeddings. Forces the two subspaces to be decorrelated so
liveness detection doesn't leak identity information and vice versa.

v2 fix (TASK_02):
    - OLD normalization `/ (d_p * d_i)` (= 1/8192 for 32x256) crushed the
      signal — observed train/orth_loss ~ 0.008, indistinguishable from
      finite-batch noise.
    - NEW normalization `/ max(d_p, d_i)` keeps loss O(1) but preserves
      the actual subspace-overlap signal.
    - Bessel correction B / (B - 1) debiases small-batch estimates.
    - Batch-size gate: B >= max(d_p, d_i) / 4 — otherwise the correlation
      estimator is too noisy and we return 0 instead of a biased reading.
"""

import torch
import torch.nn as nn


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality regularization via normalized cross-correlation matrix.

    Pipeline:
        1. Batch-normalize each feature dimension to zero mean, unit std.
        2. C = Z_p^T @ Z_i / (B - 1)          shape (d_p, d_i)
        3. Loss = sum(C**2) / max(d_p, d_i)

    With the new normalizer:
        * perfectly aligned subspace ~ loss approx 1
        * random embeddings          ~ loss 0.1-0.4
        * truly orthogonal           ~ loss < 0.1
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pad_emb: torch.Tensor,
        id_emb: torch.Tensor,
    ) -> torch.Tensor:
        B = pad_emb.shape[0]
        d_p = pad_emb.shape[1]
        d_i = id_emb.shape[1]

        # Gate on undersampled batches: correlation estimate variance is
        # too high and we'd just be punishing noise. Return 0 so beta
        # times 0 contributes nothing.
        if B < max(d_p, d_i) // 4:
            return pad_emb.new_zeros(())

        # Promote to fp32 before the batch-norm division. In 16-mixed,
        # sub-normal per-dim std + tiny eps used to produce inf -> NaN
        # in cross_corr.pow(2).sum().
        orig_dtype = pad_emb.dtype
        pad_norm = self._batch_norm(pad_emb.float())  # (B, d_p)
        id_norm  = self._batch_norm(id_emb.float())   # (B, d_i)

        # Bessel-style denominator (B - 1) debiases the sample correlation
        cross_corr = (pad_norm.T @ id_norm) / max(B - 1, 1)  # (d_p, d_i)

        loss = cross_corr.pow(2).sum() / max(d_p, d_i)
        return loss.to(orig_dtype)

    def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + self.eps)
