"""
orthogonal.py — Normalized Orthogonality Regularization Loss

Barlow Twins-style cross-correlation loss between PAD embeddings and
identity embeddings.  Forces the two subspaces to be decorrelated so
liveness detection doesn't leak identity information and vice versa.

Fix from v1: loss is normalized by (d_p × d_i) so the scale is ~O(1)
regardless of embedding dimensions.
"""

import torch
import torch.nn as nn


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality regularization via normalized cross-correlation matrix.

    Given PAD embeddings Z_p ∈ R^(B×d_p) and identity embeddings Z_i ∈ R^(B×d_i):
        1. Batch-normalize each feature dimension to zero mean, unit std
        2. Compute cross-correlation C = Z_p^T @ Z_i / B    → (d_p, d_i)
        3. Loss = Σ C_ij² / (d_p × d_i)

    Normalization by (d_p × d_i) keeps loss ~O(1) for any embedding dims.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pad_emb: torch.Tensor,
        id_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pad_emb: (B, d_p) — PAD embeddings (L2-normalized 32-D)
            id_emb:  (B, d_i) — identity embeddings (L2-normalized 256-D)

        Returns:
            loss: scalar — normalized sum of squared cross-correlation entries
        """
        B = pad_emb.shape[0]
        d_p = pad_emb.shape[1]
        d_i = id_emb.shape[1]

        # Batch-normalize each dimension
        pad_norm = self._batch_norm(pad_emb)  # (B, d_p)
        id_norm = self._batch_norm(id_emb)    # (B, d_i)

        # Cross-correlation matrix
        cross_corr = (pad_norm.T @ id_norm) / B  # (d_p, d_i)

        # Normalized loss
        loss = cross_corr.pow(2).sum() / (d_p * d_i)

        return loss

    def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + self.eps)
