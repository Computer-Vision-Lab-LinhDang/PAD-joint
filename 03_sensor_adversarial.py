"""
sensor_adversarial.py — Gradient-Reversal Sensor Classifier for PAD

Implements DANN-style domain adversarial training (Ganin et al., 2015)
where the "domain" is the capture sensor (Biometrika, DigitalPersona,
GreenBit, HiScan, ...).

Pipeline:

    pad_features (B, 128) -> GRL(lam) -> sensor_head -> sensor_logits -> CE

The sensor classifier learns normally (gradient from CE pushes it to
predict the correct sensor). The Gradient Reversal Layer flips the sign
of gradients flowing *back* into pad_features, so the feature extractor
is pushed to produce features from which the sensor *cannot* be predicted.

Recommended schedule (see TASK_03):

    lam_adv:   0 -> 1.0  over Phase 2 (DANN sigmoid ramp)
    alpha_adv: 0 -> 0.1  (overall weight of L_sensor in total loss)

Using lam_adv alone (without alpha_adv) is sufficient for simple cases,
but keeping them separate lets us independently tune "how hard to fool
the adversary" (lam_adv) vs "how much of total loss budget" (alpha_adv).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradientReversalFn(torch.autograd.Function):
    """Forward: identity. Backward: multiply grad by -lambda."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Flip sign and scale by lambda; pass None for lam since it's not a tensor.
        return -ctx.lam * grad_output, None


def gradient_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal on `x` with scalar coefficient `lam`."""
    return _GradientReversalFn.apply(x, lam)


# ---------------------------------------------------------------------------
# Sensor classification head
# ---------------------------------------------------------------------------

class SensorAdversarialHead(nn.Module):
    """Small MLP sensor classifier consumed through a GRL.

    Architecture:
        Linear(in_features, hidden) -> GELU -> Dropout -> Linear(hidden, num_sensors)

    Args:
        in_features: dim of pad_features (128 for current PADHead).
        num_sensors: number of sensor classes (9 for LivDet convention in
            pad_dataset.py — 8 known + 1 Unknown).
        hidden_dim:  MLP hidden dim.
        dropout:     dropout rate.
    """

    def __init__(
        self,
        in_features: int = 128,
        num_sensors: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sensors),
        )

    def forward(self, features: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """
        Args:
            features: (B, in_features) — pad_features.
            lam:      scalar GRL coefficient. 0 -> no adversarial pressure
                      on features. 1.0 -> full DANN.

        Returns:
            logits: (B, num_sensors).
        """
        reversed_features = gradient_reverse(features, lam)
        return self.mlp(reversed_features)


# ---------------------------------------------------------------------------
# Loss wrapper
# ---------------------------------------------------------------------------

class SensorAdversarialLoss(nn.Module):
    """Cross-entropy over sensor logits with label smoothing.

    Label smoothing helps when the sensor count is small (e.g., 4-5
    actually-used sensors out of 9 registered) — prevents the head
    from becoming over-confident in the easy cases and keeps the
    adversarial gradient flowing.
    """

    def __init__(self, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        sensor_logits: torch.Tensor,
        sensor_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sensor_logits: (B, num_sensors).
            sensor_labels: (B,) LongTensor.
        """
        return self.ce(sensor_logits, sensor_labels)

    def extra_repr(self) -> str:
        return f"label_smoothing={self.ce.label_smoothing}"


# ---------------------------------------------------------------------------
# Schedule helper — optional
# ---------------------------------------------------------------------------

def dann_lambda_schedule(progress: float) -> float:
    """DANN's original lambda schedule.

    Args:
        progress: float in [0, 1] — training progress through the
                  adversarial-active phase.
    Returns:
        lam: float in [0, 1] following 2 / (1 + exp(-10p)) - 1.
    """
    progress = max(0.0, min(1.0, progress))
    import math
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
