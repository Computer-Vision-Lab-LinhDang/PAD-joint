"""
phase_scheduler.py — PhaseSchedulerCallback

Manages transitions between OMFR training phases and ramps loss weights.

Phase schedule (default):
    Phase 1: epochs 0-19  (identity foundation)
    Phase 2: epochs 20-39 (PAD integration, alternating batches)
    Phase 3: epochs 40-59 (joint refinement)

Within Phase 2, linearly ramps over the first `warmup_epochs` epochs:
    alpha:       0.0 → 1.0   (PAD loss weight)
    beta:        0.0 → 0.1   (orthogonality loss weight)
    ArcFace s:   32.0 → 64.0 (scale in all ArcFace losses)
"""

from __future__ import annotations

import math
from typing import Any

import lightning as L


class PhaseSchedulerCallback(L.Callback):
    """
    Controls phase transitions and loss-weight ramps for OMFRModule.

    Args:
        phase1_epochs:         Number of epochs for Phase 1 (default 20).
        phase2_epochs:         Number of epochs for Phase 2 (default 20).
        phase3_epochs:         Number of epochs for Phase 3 (default 20).
        warmup_epochs:         Phase 2 warmup length for alpha/beta/scale ramps (default 5).
        alpha_target:          Target alpha at end of warmup (default 1.0).
        beta_target:           Target beta at end of warmup (default 0.1).
        arcface_scale_start:   ArcFace scale at Phase 1 / Phase 2 start (default 32.0).
        arcface_scale_end:     ArcFace scale at end of Phase 2 warmup (default 64.0).
    """

    def __init__(
        self,
        phase1_epochs: int = 20,
        phase2_epochs: int = 20,
        phase3_epochs: int = 20,
        warmup_epochs: int = 5,
        alpha_target: float = 1.0,
        beta_target: float = 0.1,
        arcface_scale_start: float = 32.0,
        arcface_scale_end: float = 64.0,
    ) -> None:
        super().__init__()
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.phase3_epochs = phase3_epochs
        self.warmup_epochs = warmup_epochs
        self.alpha_target = alpha_target
        self.beta_target = beta_target
        self.arcface_scale_start = arcface_scale_start
        self.arcface_scale_end = arcface_scale_end

        self._phase2_start = phase1_epochs
        self._phase3_start = phase1_epochs + phase2_epochs

    # ------------------------------------------------------------------
    # Lightning callback hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: Any,
    ) -> None:
        epoch = trainer.current_epoch

        # ── Phase transitions ──
        if epoch == self._phase2_start:
            self._transition_to_phase2(pl_module, trainer)
        elif epoch == self._phase3_start:
            self._transition_to_phase3(pl_module, trainer)

        # ── Ramp scheduling within Phase 2 ──
        if pl_module.current_phase == 2:
            phase2_epoch = epoch - self._phase2_start
            progress = min(phase2_epoch / max(self.warmup_epochs, 1), 1.0)

            pl_module.alpha = progress * self.alpha_target
            pl_module.beta = progress * self.beta_target

            new_scale = self.arcface_scale_start + progress * (
                self.arcface_scale_end - self.arcface_scale_start
            )
            for af_loss in pl_module.arcface_losses.values():
                af_loss.set_scale(new_scale)

            trainer.logger.log_metrics(
                {
                    "phase/alpha": pl_module.alpha,
                    "phase/beta": pl_module.beta,
                    "phase/arcface_scale": new_scale,
                    "phase/current": float(pl_module.current_phase),
                },
                step=trainer.global_step,
            )

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _transition_to_phase2(self, pl_module: Any, trainer: L.Trainer) -> None:
        """Phase 1 → Phase 2: introduce PAD head, reset alpha/beta to 0."""
        pl_module.current_phase = 2
        pl_module.alpha = 0.0
        pl_module.beta = 0.0

        # Reinitialize PAD head weights for a clean Phase 2 start
        pl_module.pad_head.apply(self._xavier_init)

        # Set ArcFace scale to start value for the ramp
        for af_loss in pl_module.arcface_losses.values():
            af_loss.set_scale(self.arcface_scale_start)

        # Notify datamodule so it can reconfigure the dataloader
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            trainer.datamodule.current_phase = 2

        print(
            f"\n{'═'*60}\n"
            f"  PHASE 2 START (epoch {trainer.current_epoch})\n"
            f"  Introducing PAD objective — alternating ID/PAD batches\n"
            f"  ArcFace scale ramp: {self.arcface_scale_start} → {self.arcface_scale_end}\n"
            f"{'═'*60}\n"
        )

    def _transition_to_phase3(self, pl_module: Any, trainer: L.Trainer) -> None:
        """Phase 2 → Phase 3: lock alpha/beta at final values, enable joint batches."""
        pl_module.current_phase = 3
        pl_module.alpha = self.alpha_target
        pl_module.beta = self.beta_target

        for af_loss in pl_module.arcface_losses.values():
            af_loss.set_scale(self.arcface_scale_end)

        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            trainer.datamodule.current_phase = 3

        print(
            f"\n{'═'*60}\n"
            f"  PHASE 3 START (epoch {trainer.current_epoch})\n"
            f"  Joint refinement — all three dataset types active\n"
            f"  alpha={pl_module.alpha:.2f}, beta={pl_module.beta:.2f}\n"
            f"{'═'*60}\n"
        )

    @staticmethod
    def _xavier_init(module: Any) -> None:
        """Xavier uniform init for Linear layers, zero init for biases."""
        import torch.nn as nn

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
