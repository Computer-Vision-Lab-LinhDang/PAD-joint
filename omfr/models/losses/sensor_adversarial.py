"""
sensor_adversarial.py — Gradient-Reversal Sensor Classifier (DANN).

LivDet APCER >> BPCER means the PAD head is solving a sensor-ID shortcut
instead of learning liveness. This module plugs a small classifier on
pad_features via a Gradient Reversal Layer (Ganin et al., 2015) so the
features become sensor-invariant.

Forward path:
    pad_features --GRL(lam)--> MLP --> sensor_logits

Backward path:
    grad wrt pad_features is flipped and scaled by lam, so pad_features
    learns to be UN-predictive of sensor while the head tries to predict
    normally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


class SensorAdversarialHead(nn.Module):
    """
    Small MLP that predicts sensor ID from pad_features through a GRL.

    Kept shallow (one hidden) so the adversary isn't overpowering;
    DANN works best when the adversary is just strong enough to keep
    the feature extractor honest.
    """

    def __init__(
        self,
        in_features: int = 128,
        num_sensors: int = 9,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_sensors),
        )

    def forward(self, pad_features: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        reversed_feat = grad_reverse(pad_features, lam)
        return self.net(reversed_feat)


class SensorAdversarialLoss(nn.Module):
    """Cross-entropy on sensor logits with optional per-sample masking."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        sensor_logits: torch.Tensor,
        sensor_labels: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        labels = sensor_labels.to(device=sensor_logits.device).long().reshape(-1)
        if sensor_logits.shape[0] != labels.numel():
            raise ValueError(
                "SensorAdversarialLoss labels must match batch size: "
                f"got {labels.numel()} labels for {sensor_logits.shape[0]} logits"
            )
        logits = sensor_logits.reshape(sensor_logits.shape[0], -1)
        per_sample = F.cross_entropy(logits, labels, reduction="none")

        if sample_weight is None:
            return per_sample.mean()

        weights = sample_weight.to(device=sensor_logits.device, dtype=per_sample.dtype).reshape(-1)
        if weights.numel() != per_sample.numel():
            raise ValueError(
                "SensorAdversarialLoss sample_weight must match batch size: "
                f"got {weights.numel()} weights for {per_sample.numel()} samples"
            )

        weight_sum = weights.sum()
        if weight_sum <= 0:
            return per_sample.sum() * 0.0
        return (per_sample * weights).sum() / weight_sum
