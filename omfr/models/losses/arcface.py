"""
arcface.py — ArcFace Loss with MRL support

ArcFace (Additive Angular Margin) loss for metric learning.
Used independently per MRL dimension {64, 128, 256}.

Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
Face Recognition", CVPR 2019.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss: adds an additive angular margin m to the target class angle
    before computing softmax cross-entropy.

    L = -log( exp(s·cos(θ_yi + m)) / (exp(s·cos(θ_yi + m)) + Σ_{j≠yi} exp(s·cos(θ_j))) )

    Used per MRL dimension — weights are NOT shared across dimensions.

    Args:
        embedding_dim: int   — dimension of L2-normalized input embeddings
        num_classes:   int   — number of training identities
        s: float = 64.0      — feature scale (ArcFace logit scale)
        margin: float = 0.5  — additive angular margin (radians)
        easy_margin: bool    — if True, use piecewise cos/sin (more stable for
                               large margins or small scale)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.margin = margin
        self.easy_margin = easy_margin

        # Weight matrix: each row is a class center (normalized during forward)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)  # for easy_margin=False
        self.mm = math.sin(math.pi - margin) * margin  # for easy_margin=False

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: (B, embedding_dim) — L2-normalized feature vectors
            labels:     (B,)               — LongTensor, class indices in [0, num_classes)

        Returns:
            loss: scalar
        """
        # Normalize weight matrix
        w_norm = F.normalize(self.weight, p=2, dim=1)  # (num_classes, embedding_dim)

        # Cosine similarity: embeddings assumed already L2-normalized
        cosine = F.linear(embeddings, w_norm)  # (B, num_classes)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # sin(θ) = sqrt(1 - cos²(θ))
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(θ + m) = cos(θ)·cos(m) - sin(θ)·sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # Use cos(θ + m) only when θ + m < π (i.e., cos(θ) > 0)
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard: use cos(θ + m) when cos(θ) > threshold, else linear fallback
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # One-hot encode labels and build logit tensor
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # For target class: use phi; for non-target: use cosine
        output = one_hot * phi + (1.0 - one_hot) * cosine

        # Scale and compute cross-entropy
        output = output * self.s
        return self.ce(output, labels)

    def set_scale(self, s: float) -> None:
        """Update the scale parameter (called by PhaseSchedulerCallback during ramp)."""
        self.s = s

    def set_margin(self, margin: float) -> None:
        """Update the margin and recompute derived constants (called during warmup)."""
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, num_classes={self.num_classes}, "
            f"s={self.s}, margin={self.margin}"
        )
