"""
phase_scheduler.py — PhaseSchedulerCallback (DYNAMIC STATE MACHINE)

Two-phase dynamic schedule (no fixed Phase-1 length, no Phase 3):

    Phase 1  — identity foundation by default, or PAD foundation when
               phases.phase1_task=pad. Runs until `phase1_max_epochs` unless
               plateau transition is explicitly enabled.

    Phase 2  — identity integration. Lasts `phase2_epochs`. On the transition
               epoch (a Lightning epoch-start hook, OUTSIDE the autograd
               region) we, in order:
                 1. swap identity-only loader -> combined id+pad loader,
                 2. hard re-scale the optimizer LR groups
                    (entire identity side -> 0.0, PAD unleashed),
                 3. start the alpha/beta/arcface ramps anchored at the
                    dynamic transition epoch.

Phase 1 ArcFace warmup is unchanged (scale init -> start after a delay).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import lightning as L


class PhaseSchedulerCallback(L.Callback):
    """Dynamic phase-transition + loss-weight ramps + loader switching."""

    def __init__(
        self,
        phase1_max_epochs: int = 80,
        phase2_epochs: int = 40,
        plateau_patience: int = 5,
        plateau_monitor: str = "val/cascaded_IM",
        plateau_mode: str = "max",
        plateau_min_delta: float = 1.0e-4,
        warmup_epochs: int = 5,
        phase1_warmup_epochs: int = 5,
        phase1_warmup_delay: int = 5,
        alpha_target: float = 1.0,
        beta_target: float = 0.02,
        gamma_target: float = 1.0,
        alpha_adv_target: float = 0.0,
        arcface_scale_init: float = 1.0,
        arcface_scale_start: float = 32.0,
        arcface_scale_end: float = 48.0,
        arcface_margin_init: float = 0.0,
        arcface_margin_target: float = 0.5,
        moe_temp_phase1: float = 2.0,
        moe_temp_phase2_end: float = 1.0,
        phase1_task: str = "identity",
        transition_on_plateau: bool = True,
        # legacy/back-compat kwargs (ignored by the dynamic machine)
        phase1_epochs: Optional[int] = None,
        phase3_epochs: Optional[int] = None,
        moe_temp_phase3_end: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.phase1_max_epochs = int(phase1_max_epochs)
        self.phase2_epochs = int(phase2_epochs)
        self.plateau_patience = int(plateau_patience)
        self.plateau_monitor = str(plateau_monitor)
        self.plateau_mode = str(plateau_mode)
        self.plateau_min_delta = float(plateau_min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.phase1_warmup_epochs = int(phase1_warmup_epochs)
        self.phase1_warmup_delay = int(phase1_warmup_delay)
        self.alpha_target = alpha_target
        self.beta_target = beta_target
        self.gamma_target = gamma_target
        self.alpha_adv_target = alpha_adv_target
        self.arcface_scale_init = arcface_scale_init
        self.arcface_scale_start = arcface_scale_start
        self.arcface_scale_end = arcface_scale_end
        self.arcface_margin_init = arcface_margin_init
        self.arcface_margin_target = arcface_margin_target
        self.moe_temp_phase1 = moe_temp_phase1
        self.moe_temp_phase2_end = moe_temp_phase2_end
        self.phase1_task = str(phase1_task).lower()
        self.transition_on_plateau = bool(transition_on_plateau)

        # Dynamic state
        self._best: Optional[float] = None
        self._no_improve: int = 0
        self._pending_phase2: bool = False
        self._phase2_started: bool = False
        self._phase2_start: int = self.phase1_max_epochs   # dynamic anchor
        self._total_phase_epochs = self.phase1_max_epochs + self.phase2_epochs

    # ------------------------------------------------------------------
    # Ramp helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_ramp(
        epoch: int, start: int, end: int, target: float, min_frac: float = 0.01,
    ) -> float:
        if epoch < start:
            return 0.0
        if epoch >= end:
            return target
        p = (epoch - start) / max(end - start, 1)
        ramp = 0.5 * (1.0 - math.cos(math.pi * p))
        return target * (min_frac + (1.0 - min_frac) * ramp)

    @staticmethod
    def _dann_lambda(epoch: int, start: int, end: int) -> float:
        if epoch < start:
            return 0.0
        if epoch >= end:
            return 1.0
        p = (epoch - start) / max(end - start, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    def _arcface_margin_for_epoch(self, epoch: int) -> float:
        p = min(max(epoch, 0) / max(self._total_phase_epochs - 1, 1), 1.0)
        return self.arcface_margin_init + p * (
            self.arcface_margin_target - self.arcface_margin_init
        )

    @staticmethod
    def _set_arcface_margin(pl_module: Any, margin: float) -> None:
        for af_loss in pl_module.arcface_losses.values():
            af_loss.set_margin(margin)

    @staticmethod
    def _set_moe_temperature(pl_module: Any, temperature: float) -> None:
        if hasattr(pl_module, "backbone") and hasattr(
            pl_module.backbone, "set_moe_temperature"
        ):
            pl_module.backbone.set_moe_temperature(temperature)

    # ------------------------------------------------------------------
    # Plateau tracking (Phase 1 only)
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        if self._best is None:
            return True
        if self.plateau_mode == "max":
            return value > self._best + self.plateau_min_delta
        return value < self._best - self.plateau_min_delta

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: Any) -> None:
        # Skip the pre-train sanity validation and anything past Phase 1.
        if trainer.sanity_checking:
            return
        if not self.transition_on_plateau:
            return
        if getattr(pl_module, "current_phase", 1) != 1 or self._phase2_started:
            return

        metric = trainer.callback_metrics.get(self.plateau_monitor)
        if metric is None:
            return
        value = float(metric)

        if self._is_improvement(value):
            self._best = value
            self._no_improve = 0
        else:
            self._no_improve += 1

        epoch = trainer.current_epoch
        plateaued = self._no_improve >= self.plateau_patience
        capped = (epoch + 1) >= self.phase1_max_epochs
        if plateaued or capped:
            self._pending_phase2 = True
            reason = "plateau" if plateaued else "max-epochs cap"
            print(
                f"\n[phase-sm] Phase-2 ARMED at epoch {epoch} "
                f"({reason}); best {self.plateau_monitor}={self._best:.4f}, "
                f"no_improve={self._no_improve}/{self.plateau_patience}"
            )

        trainer.logger.log_metrics(
            {
                "phase/p1_best_metric": self._best if self._best is not None else 0.0,
                "phase/p1_no_improve": float(self._no_improve),
            },
            step=trainer.global_step,
        )

    # ------------------------------------------------------------------
    # Epoch-start: transitions + ramps
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: Any) -> None:
        epoch = trainer.current_epoch

        # Fire the dynamic transition (idempotent).
        if (
            getattr(pl_module, "current_phase", 1) == 1
            and not self._phase2_started
            and (self._pending_phase2 or epoch >= self.phase1_max_epochs)
        ):
            self._transition_to_phase2(pl_module, trainer)

        arcface_margin = self._arcface_margin_for_epoch(epoch)

        if pl_module.current_phase == 1:
            if self.phase1_task in {"pad", "pad_foundation"}:
                scale = self.arcface_scale_start
                pl_module.alpha = self.alpha_target
            else:
                delay = max(self.phase1_warmup_delay, 0)
                effective_epoch = max(epoch - delay, 0)
                p1 = min(effective_epoch / max(self.phase1_warmup_epochs, 1), 1.0)
                scale = self.arcface_scale_init + p1 * (
                    self.arcface_scale_start - self.arcface_scale_init
                )
                pl_module.alpha = 0.0
            for af_loss in pl_module.arcface_losses.values():
                af_loss.set_scale(scale)
            self._set_arcface_margin(pl_module, arcface_margin)

            pl_module.beta = 0.0
            if hasattr(pl_module, "identity_weight"):
                pl_module.identity_weight = 0.0
            pl_module.alpha_adv = 0.0
            pl_module.lam_adv = 0.0
            self._set_moe_temperature(pl_module, self.moe_temp_phase1)

            trainer.logger.log_metrics(
                {
                    "phase/arcface_scale": scale,
                    "phase/arcface_margin": arcface_margin,
                    "phase/moe_temperature": self.moe_temp_phase1,
                    "phase/alpha": pl_module.alpha,
                    "phase/gamma_identity": getattr(pl_module, "identity_weight", 0.0),
                    "phase/beta": pl_module.beta,
                    "phase/current": 1.0,
                },
                step=trainer.global_step,
            )

        elif pl_module.current_phase == 2:
            p2_start = self._phase2_start
            p2_end = self._phase2_start + self.phase2_epochs
            ramp_end_short = min(p2_start + self.warmup_epochs, p2_end)

            pl_module.alpha = self.alpha_target
            pl_module.beta = self._cosine_ramp(
                epoch, p2_start, ramp_end_short, self.beta_target, min_frac=0.0,
            )
            if hasattr(pl_module, "identity_weight"):
                pl_module.identity_weight = self._cosine_ramp(
                    epoch, p2_start, ramp_end_short, self.gamma_target, min_frac=0.0,
                )
            pl_module.alpha_adv = self._cosine_ramp(
                epoch, p2_start, p2_end, self.alpha_adv_target,
            )
            pl_module.lam_adv = self._dann_lambda(epoch, p2_start, p2_end)

            scale_progress = self._cosine_ramp(
                epoch, p2_start, ramp_end_short, 1.0, min_frac=0.0,
            )
            new_scale = self.arcface_scale_start + scale_progress * (
                self.arcface_scale_end - self.arcface_scale_start
            )
            for af_loss in pl_module.arcface_losses.values():
                af_loss.set_scale(new_scale)
            self._set_arcface_margin(pl_module, arcface_margin)

            p2p = min(
                (epoch - p2_start) / max(self.phase2_epochs - 1, 1), 1.0,
            )
            moe_temp = self.moe_temp_phase1 + p2p * (
                self.moe_temp_phase2_end - self.moe_temp_phase1
            )
            self._set_moe_temperature(pl_module, moe_temp)

            trainer.logger.log_metrics(
                {
                    "phase/alpha": pl_module.alpha,
                    "phase/beta": pl_module.beta,
                    "phase/gamma_identity": getattr(pl_module, "identity_weight", 0.0),
                    "phase/alpha_adv": pl_module.alpha_adv,
                    "phase/lam_adv": pl_module.lam_adv,
                    "phase/arcface_scale": new_scale,
                    "phase/arcface_margin": arcface_margin,
                    "phase/moe_temperature": moe_temp,
                    "phase/current": 2.0,
                },
                step=trainer.global_step,
            )

    # ------------------------------------------------------------------
    # Transition Phase 1 -> Phase 2
    # ------------------------------------------------------------------

    def _transition_to_phase2(self, pl_module: Any, trainer: L.Trainer) -> None:
        epoch = trainer.current_epoch
        pl_module.current_phase = 2
        self._phase2_started = True
        self._phase2_start = epoch          # dynamic ramp anchor

        pl_module.alpha = 0.0
        pl_module.beta = 0.0
        if hasattr(pl_module, "identity_weight"):
            pl_module.identity_weight = 0.0
        pl_module.alpha_adv = 0.0
        pl_module.lam_adv = 0.0

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for af_loss in pl_module.arcface_losses.values():
            af_loss.set_scale(self.arcface_scale_start)
        self._set_arcface_margin(
            pl_module, self._arcface_margin_for_epoch(epoch),
        )
        self._set_moe_temperature(pl_module, self.moe_temp_phase1)

        # (1) swap identity-only loader -> combined id+pad loader
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            trainer.datamodule.current_phase = 2
        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.setup_data()

        # (2) phase-2 LR surgery: absolute-freeze configs brake the whole
        #     identity side; Level-1 configs keep only early shared texture
        #     groups on a tiny LR and leave late identity layers frozen.
        if hasattr(pl_module, "apply_phase2_lr_multipliers"):
            pl_module.apply_phase2_lr_multipliers(trainer)

        phase2_mode = "IDENTITY INTEGRATION"
        print(
            f"\n{'='*64}\n"
            f"  PHASE 2 START (epoch {epoch}) — IDENTITY INTEGRATION\n"
            f"  reason: {'plateau/armed' if self._pending_phase2 else 'fixed phase1 cap'}\n"
            f"  mode: {phase2_mode}\n"
            f"  alpha={self.alpha_target}, gamma->{self.gamma_target}, "
            f"beta->{self.beta_target}, "
            f"phase2 length={self.phase2_epochs} ep\n"
            f"{'='*64}\n"
        )
