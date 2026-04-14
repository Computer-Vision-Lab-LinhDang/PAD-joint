"""
gradient_monitor.py — GradientMonitor Callback

Logs L2 gradient norms per optimizer param group after each backward pass.
Helps detect gradient conflicts and vanishing/exploding gradients across
different component groups (gabor, backbone, moe_experts, heads, arcface).

Logged metrics:
    grad_norm/<group_name>  — L2 norm of all gradients in that param group
    grad_norm/total         — L2 norm across all parameters
"""

from __future__ import annotations

import math
from typing import Any, Optional

import lightning as L
import torch


class GradientMonitor(L.Callback):
    """
    Logs gradient norms per optimizer param group each training step.

    Args:
        log_every_n_steps: how often to compute and log norms (default 50).
                           Computing norms every step is expensive for large models.
        norm_type:         p-norm type (default 2.0 for L2 norm).
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        norm_type: float = 2.0,
    ) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.norm_type = norm_type

    def on_after_backward(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        optimizer = trainer.optimizers
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        metrics: dict[str, float] = {}
        total_sq = 0.0
        total_count = 0

        for group in optimizer.param_groups:
            group_name = group.get("name", f"group_{id(group)}")
            group_sq = 0.0
            has_grad = False

            for param in group["params"]:
                if param.grad is None:
                    continue
                has_grad = True
                grad_data = param.grad.detach().float()
                group_sq += grad_data.pow(self.norm_type).sum().item()
                total_sq += grad_data.pow(self.norm_type).sum().item()
                total_count += 1

            if has_grad:
                group_norm = group_sq ** (1.0 / self.norm_type)
                metrics[f"grad_norm/{group_name}"] = group_norm

        if total_count > 0:
            metrics["grad_norm/total"] = total_sq ** (1.0 / self.norm_type)

        pl_module.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False)
