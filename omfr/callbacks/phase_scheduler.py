"""
phase_scheduler.py — PhaseSchedulerCallback

Manages transitions between OMFR training phases and ramps loss weights.

Phase schedule (default):
    Phase 1: epochs 0-19  (identity foundation)
    Phase 2: epochs 20-39 (PAD integration, alternating batches)
    Phase 3: epochs 40-59 (joint refinement)

Phase 1 warmup (CRITICAL for stability):
    ArcFace scale:  1.0 → 32.0   (prevents gradient explosion with many classes)
    ArcFace margin: 0.0 → 0.5    (gradual angular margin introduction)
    Warmup over first `phase1_warmup_epochs` epochs.

Phase 2 ramps:
    alpha:       0.0 → 1.0   (PAD loss weight)
    beta:        0.0 → 0.1   (orthogonality loss weight)
    ArcFace s:   32.0 → 64.0

MoE temperature schedule:
    Phase 1: 2.0 (soft routing, exploration)
    Phase 2: 2.0 → 1.0 (gradual sharpening)
    Phase 3: 1.0 → 0.5 (sharp routing, specialization)

Dataloader reload:
    Calls trainer.reset_train_dataloader() at phase transitions to switch
    from Phase 1 identity-only loader to Phase 2/3 combined loaders.
"""

from __future__ import annotations

import math
from typing import Any

import lightning as L


class PhaseSchedulerCallback(L.Callback):
    """
    Controls phase transitions, loss-weight ramps, and dataloader switching.
    """

    def __init__(
        self,
        phase1_epochs: int = 20,
        phase2_epochs: int = 20,
        phase3_epochs: int = 20,
        warmup_epochs: int = 5,
        phase1_warmup_epochs: int = 5,
        phase1_warmup_delay: int = 5,
        alpha_target: float = 1.0,
        beta_target: float = 0.1,
        arcface_scale_init: float = 1.0,
        arcface_scale_start: float = 32.0,
        arcface_scale_end: float = 64.0,
        arcface_margin_init: float = 0.0,
        arcface_margin_target: float = 0.5,
        moe_temp_phase1: float = 2.0,
        moe_temp_phase2_end: float = 1.0,
        moe_temp_phase3_end: float = 0.5,
    ) -> None:
        super().__init__()
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.phase3_epochs = phase3_epochs
        self.warmup_epochs = warmup_epochs
        self.phase1_warmup_epochs = phase1_warmup_epochs
        self.phase1_warmup_delay = phase1_warmup_delay
        self.alpha_target = alpha_target
        self.beta_target = beta_target
        self.arcface_scale_init = arcface_scale_init
        self.arcface_scale_start = arcface_scale_start
        self.arcface_scale_end = arcface_scale_end
        self.arcface_margin_init = arcface_margin_init
        self.arcface_margin_target = arcface_margin_target
        self.moe_temp_phase1 = moe_temp_phase1
        self.moe_temp_phase2_end = moe_temp_phase2_end
        self.moe_temp_phase3_end = moe_temp_phase3_end

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

        # ── Phase 1: ArcFace scale/margin warmup + MoE temp ──
        # Delayed ArcFace warmup: hold s=1, m=0 until LR warmup completes,
        # then ramp linearly over `phase1_warmup_epochs` epochs. This prevents
        # gradient explosion when LR is still tiny and embeddings are random.
        if pl_module.current_phase == 1:
            delay = max(self.phase1_warmup_delay, 0)
            effective_epoch = max(epoch - delay, 0)
            p1_progress = min(effective_epoch / max(self.phase1_warmup_epochs, 1), 1.0)

            # Scale: 1.0 → 32.0 over warmup (after delay)
            scale = self.arcface_scale_init + p1_progress * (
                self.arcface_scale_start - self.arcface_scale_init
            )
            # Margin: 0.0 → 0.5 over warmup (after delay)
            margin = self.arcface_margin_init + p1_progress * (
                self.arcface_margin_target - self.arcface_margin_init
            )
            for af_loss in pl_module.arcface_losses.values():
                af_loss.set_scale(scale)
                af_loss.set_margin(margin)

            self._set_moe_temperature(pl_module, self.moe_temp_phase1)

            trainer.logger.log_metrics(
                {
                    "phase/arcface_scale": scale,
                    "phase/arcface_margin": margin,
                    "phase/moe_temperature": self.moe_temp_phase1,
                    "phase/current": 1.0,
                },
                step=trainer.global_step,
            )

        # ── Ramp scheduling within Phase 2 ──
        if pl_module.current_phase == 2:
            phase2_epoch = epoch - self._phase2_start
            progress = min(phase2_epoch / max(self.warmup_epochs, 1), 1.0)
            phase2_progress = min(phase2_epoch / max(self.phase2_epochs - 1, 1), 1.0)

            pl_module.alpha = progress * self.alpha_target
            pl_module.beta = progress * self.beta_target

            new_scale = self.arcface_scale_start + progress * (
                self.arcface_scale_end - self.arcface_scale_start
            )
            for af_loss in pl_module.arcface_losses.values():
                af_loss.set_scale(new_scale)

            # MoE temperature: 2.0 → 1.0 over Phase 2
            moe_temp = self.moe_temp_phase1 + phase2_progress * (
                self.moe_temp_phase2_end - self.moe_temp_phase1
            )
            self._set_moe_temperature(pl_module, moe_temp)

            trainer.logger.log_metrics(
                {
                    "phase/alpha": pl_module.alpha,
                    "phase/beta": pl_module.beta,
                    "phase/arcface_scale": new_scale,
                    "phase/moe_temperature": moe_temp,
                    "phase/current": 2.0,
                },
                step=trainer.global_step,
            )

        # ── Ramp scheduling within Phase 3 ──
        if pl_module.current_phase == 3:
            phase3_epoch = epoch - self._phase3_start
            phase3_progress = min(phase3_epoch / max(self.phase3_epochs - 1, 1), 1.0)

            # MoE temperature: 1.0 → 0.5 over Phase 3
            moe_temp = self.moe_temp_phase2_end + phase3_progress * (
                self.moe_temp_phase3_end - self.moe_temp_phase2_end
            )
            self._set_moe_temperature(pl_module, moe_temp)

            trainer.logger.log_metrics(
                {
                    "phase/moe_temperature": moe_temp,
                    "phase/current": 3.0,
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

        # Do NOT reinitialize pad_head here — it keeps its Phase 1 (identity-only)
        # starting state. Re-xavier-init destroyed warmup progress in prior runs.

        # Set ArcFace scale to Phase 2 start, margin locked at target
        for af_loss in pl_module.arcface_losses.values():
            af_loss.set_scale(self.arcface_scale_start)
            af_loss.set_margin(self.arcface_margin_target)

        # MoE temperature starts at Phase 1 value, will ramp down
        self._set_moe_temperature(pl_module, self.moe_temp_phase1)

        # CRITICAL: Reload dataloaders to switch from identity-only to combined
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            trainer.datamodule.current_phase = 2
        # Force Lightning 2.x to re-call train_dataloader()
        # setup_data() already ran in on_advance_start before this callback,
        # so we must clear AND rebuild immediately.
        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.setup_data()

        print(
            f"\n{'═'*60}\n"
            f"  PHASE 2 START (epoch {trainer.current_epoch})\n"
            f"  Introducing PAD objective — alternating ID/PAD batches\n"
            f"  Dataloader RELOADED for combined ID+PAD batches\n"
            f"  ArcFace scale ramp: {self.arcface_scale_start} → {self.arcface_scale_end}\n"
            f"  MoE temperature ramp: {self.moe_temp_phase1} → {self.moe_temp_phase2_end}\n"
            f"{'═'*60}\n"
        )

    def _transition_to_phase3(self, pl_module: Any, trainer: L.Trainer) -> None:
        """Phase 2 → Phase 3: lock alpha/beta at final values, enable joint batches."""
        pl_module.current_phase = 3
        pl_module.alpha = self.alpha_target
        pl_module.beta = self.beta_target

        for af_loss in pl_module.arcface_losses.values():
            af_loss.set_scale(self.arcface_scale_end)

        # MoE temperature starts at Phase 2 end value, will ramp down
        self._set_moe_temperature(pl_module, self.moe_temp_phase2_end)

        # CRITICAL: Reload dataloaders for Phase 3 combined loader
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            trainer.datamodule.current_phase = 3
        # Force Lightning 2.x to re-call train_dataloader()
        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.setup_data()

        print(
            f"\n{'═'*60}\n"
            f"  PHASE 3 START (epoch {trainer.current_epoch})\n"
            f"  Joint refinement — all dataset types active\n"
            f"  Dataloader RELOADED for joint batches\n"
            f"  alpha={pl_module.alpha:.2f}, beta={pl_module.beta:.2f}\n"
            f"  MoE temperature ramp: {self.moe_temp_phase2_end} → {self.moe_temp_phase3_end}\n"
            f"{'═'*60}\n"
        )

    @staticmethod
    def _set_moe_temperature(pl_module: Any, temperature: float) -> None:
        """Set MoE routing temperature on the backbone."""
        if hasattr(pl_module, 'backbone') and hasattr(pl_module.backbone, 'set_moe_temperature'):
            pl_module.backbone.set_moe_temperature(temperature)

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
