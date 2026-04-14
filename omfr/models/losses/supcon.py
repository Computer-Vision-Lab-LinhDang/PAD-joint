"""
supcon.py — Supervised Contrastive Loss

Used for PAD branch: pulls together same-class (live/live or spoof/spoof) pairs
and pushes apart cross-class pairs in the 128-D PAD feature space.

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    For each anchor i:
        L_i = -1/|P(i)| · Σ_{p∈P(i)} log(
            exp(z_i · z_p / τ) /
            Σ_{a∈A(i)} exp(z_i · z_a / τ)
        )

    where:
        P(i) = set of positive samples (same class as i, excluding i itself)
        A(i) = all other samples in the batch
        τ    = temperature

    Args:
        temperature: float = 0.07  — contrastive temperature
        contrast_mode: str         — 'all' uses all views as anchors,
                                     'one' uses only the first view
        base_temperature: float    — for loss scaling (default = temperature)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = 'all',
        base_temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Supervised Contrastive loss.

        Args:
            features: (B, dim) or (B, n_views, dim) — L2-normalized feature vectors.
                      If 2D, treated as single-view (B, 1, dim) internally.
            labels:   (B,) — LongTensor class labels (e.g., 0=spoof, 1=live)

        Returns:
            loss: scalar
        """
        device = features.device

        if features.ndim == 2:
            # Single-view: (B, dim) → (B, 1, dim)
            features = features.unsqueeze(1)

        B, n_views, dim = features.shape

        # L2-normalize along feature dimension
        features = F.normalize(features, p=2, dim=-1)

        # Build contrast features and labels
        if self.contrast_mode == 'one':
            anchor_features = features[:, 0]          # (B, dim) — first view only
            anchor_count = 1
        else:  # 'all'
            anchor_features = features.reshape(B * n_views, dim)  # (B*V, dim)
            anchor_count = n_views

        contrast_features = features.reshape(B * n_views, dim)   # (B*V, dim)
        contrast_labels   = labels.repeat(n_views)               # (B*V,)

        # Expand anchor labels
        anchor_labels = labels.repeat(anchor_count)  # (B*V,) or (B,)

        # Similarity matrix: (anchor_count*B, B*n_views)
        anchor_dot_contrast = torch.div(
            anchor_features @ contrast_features.T,
            self.temperature,
        )

        # For numerical stability: subtract max per row
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask: same class = positive pair, self = excluded
        anchor_n   = anchor_labels.shape[0]
        contrast_n = contrast_labels.shape[0]

        # (anchor_n, contrast_n) — 1 where same class
        label_eq_mask = anchor_labels.unsqueeze(1).eq(
            contrast_labels.unsqueeze(0)
        ).float()

        # Remove self-contrast entries (diagonal when anchor == contrast set)
        if anchor_count == n_views and B * n_views == contrast_n:
            # Full all-views case: self-contrast is the main diagonal
            self_mask = torch.eye(anchor_n, device=device)
            label_eq_mask = label_eq_mask * (1 - self_mask)

        # Denominator mask: all pairs except self
        if anchor_count == n_views and anchor_n == contrast_n:
            denom_mask = 1 - torch.eye(anchor_n, device=device)
        else:
            denom_mask = torch.ones(anchor_n, contrast_n, device=device)

        # Log-softmax over valid negatives + positives
        exp_logits = torch.exp(logits) * denom_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Positive count per anchor (avoid division by zero)
        positive_count = label_eq_mask.sum(dim=1)  # (anchor_n,)
        valid_anchor = positive_count > 0

        if not valid_anchor.any():
            # No valid positives in batch (e.g., all same class or batch size 1)
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Mean log-prob over positives, then mean over anchors
        mean_log_prob_pos = (label_eq_mask * log_prob).sum(dim=1)
        mean_log_prob_pos = mean_log_prob_pos[valid_anchor] / positive_count[valid_anchor]

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}, contrast_mode='{self.contrast_mode}'"
