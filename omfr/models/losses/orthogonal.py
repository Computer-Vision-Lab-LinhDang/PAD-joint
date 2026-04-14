"""
orthogonal.py — Orthogonality Regularization Loss

Barlow Twins-style cross-correlation loss between PAD embeddings and
identity embeddings. Forces the two subspaces to be decorrelated so
liveness detection doesn't leak identity information and vice versa.

Reference: Zbontar et al., "Barlow Twins: Self-Supervised Learning via
Redundancy Reduction", ICML 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality regularization via cross-correlation matrix.

    Given PAD embeddings Z_p ∈ R^(B×d_p) and identity embeddings Z_i ∈ R^(B×d_i),
    compute the normalized cross-correlation matrix C ∈ R^(d_p × d_i):

        C_kl = Σ_b z_p_bk · z_i_bl / (||z_p_k|| · ||z_i_l||)

    Loss = Σ_k Σ_l C_kl²

    All entries should be driven to zero → no linear dependence between
    any PAD dimension and any identity dimension.

    Unlike Barlow Twins (which also has an on-diagonal term for invariance),
    we only care about the cross-correlation going to zero — we want the two
    representations to be in orthogonal subspaces.

    Args:
        eps: float — small constant for numerical stability in normalization
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pad_emb: torch.Tensor,
        id_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-correlation orthogonality loss.

        Args:
            pad_emb: (B, d_p) — PAD embeddings (L2-normalized 32-D)
            id_emb:  (B, d_i) — identity embeddings (L2-normalized 256-D)

        Returns:
            loss: scalar — sum of squared cross-correlation entries
        """
        B = pad_emb.shape[0]

        # Batch-normalize each dimension to zero mean, unit variance
        # This is the "normalized" cross-correlation in Barlow Twins style
        pad_norm = self._batch_norm(pad_emb)  # (B, d_p)
        id_norm  = self._batch_norm(id_emb)   # (B, d_i)

        # Cross-correlation matrix: (d_p, d_i)
        cross_corr = (pad_norm.T @ id_norm) / B  # (d_p, d_i)

        # All entries should be zero → minimize sum of squares
        loss = cross_corr.pow(2).sum()

        return loss

    def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize each feature dimension to zero mean, unit std across batch.

        Args:
            x: (B, D)

        Returns:
            x_norm: (B, D)
        """
        mean = x.mean(dim=0, keepdim=True)   # (1, D)
        std  = x.std(dim=0, keepdim=True)    # (1, D)
        return (x - mean) / (std + self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
