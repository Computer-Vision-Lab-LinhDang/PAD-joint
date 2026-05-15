"""
phase_scheduler.py — PhaseSchedulerCallback

Manages transitions between OMFR training phases and ramps loss weights.

Phase schedule (default):
    Phase 1: epochs 0-19  (identity foundation)
    Phase 2: epochs 20-39 (PAD integration, joint batches via TASK_04)
    Phase 3: epochs 40-59 (joint refinement)

Phase 1 warmup (CRITICAL for stability):
    ArcFace scale: configured init -> start value (prevents gradient explosion)
    Warmup over first `phase1_warmup_epochs` epochs, after optional delay.

Phase 2 ramps (TASK_04 — cosine soft-start):
    alpha:       0.01 * target -> target over first 5 epochs of Phase 2
    beta:        0.01 * target -> target (target = 0.05 after TASK_02 re-norm)
    alpha_adv:   0 -> target over the FULL Phase 2 (DANN-style slow ramp)
    lam_adv:     sigmoid schedule 2/(1+exp(-10p)) - 1 over full Phase 2
    ArcFace s:   configured start -> configured end

ArcFace margin:
    Continuous schedule over all phases: 0.2 -> 0.45 by final epoch.

MoE temperature:
    Phase 1: 2.0 (soft routing, exploration)
    Phase 2: 2.0 -> 1.0 (gradual sharpening)
    Phase 3: 1.0 -> 0.8 (conservative sharpening)
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
        beta_target: float = 0.05,
        alpha_adv_target: float = 0.1,
        arcface_scale_init: float = 1.0,
        arcface_scale_start: float = 32.0,
        arcface_scale_end: float = 64.0,
        arcface_margin_init: float = 0.2,
        arcface_margin_target: float = 0.45,
        moe_temp_phase1: float = 2.0,
        moe_temp_phase2_end: float = 1.0,
        moe_temp_phase3_end: float = 0.8,
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
        self.alpha_adv_target = alpha_adv_target
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
        self._total_phase_epochs = phase1_epochs + phase2_epochs + phase3_epochs

    # ------------------------------------------------------------------
    # Ramp helper
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_ramp(
        epoch: int,
        start: int,
        end: int,
        target: float,
        min_frac: float = 0.01,
    ) -> float:
        """Cosine ramp from min_frac*target (epoch=start) to target (epoch>=end)."""
        if epoch < start:
            return 0.0
        if epoch >= end:
            return target
        p = (epoch - start) / max(end - start, 1)
        ramp = 0.5 * (1.0 - math.cos(math.pi * p))
        return target * (min_frac + (1.0 - min_frac) * ramp)

    @staticmethod
    def _dann_lambda(epoch: int, start: int, end: int) -> float:
        """DANN schedule lam(p) = 2/(1+exp(-10p)) - 1, p in [0, 1]."""
        if epoch < start:
            return 0.0
        if epoch >= end:
            return 1.0
        p = (epoch - start) / max(end - start, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    def _arcface_margin_for_epoch(self, epoch: int) -> float:
        """Linear margin schedule across the full three-phase run."""
        p = min(max(epoch, 0) / max(self._total_phase_epochs - 1, 1), 1.0)
        return self.arcface_margin_init + p * (
            self.arcface_margin_target - self.arcface_margin_init
        )

    @staticmethod
    def _set_arcface_margin(pl_module: Any, margin: float) -> None:
        for af_loss in pl_module.arcface_losses.values():
            af_loss.set_margin(margin)

    # ------------------------------------------------------------------
    # Lightning callback hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: Any,
    ) -> None:
        epoch = trainer.current_epoch
        arcface_margin = self._arcface_margin_for_epoch(epoch)

        # Phase transitions
        if epoch == self._phase2_start:
            self._transition_to_phase2(pl_module, trainer)
        elif epoch == self._phase3_start:
            self._transition_to_phase3(pl_module, trainer)

        # Phase 1: ArcFace warmup + MoE temp
        if pl_module.current_phase == 1:
            delay = max(self.phase1_warmup_delay, 0)
            effective_epoch = max(epoch - delay, 0)
            p1_progress = min(effective_epoch / max(self.phase1_warmup_epochs, 1), 1.0)

            scale = self.arcface_scale_init + p1_progress * (
                self.arcface_scale_start - self.arcface_scale_init
            )
            for af_loss in pl_module.arcface_losses.values():
                af_loss.set_scale(scale)
            self._set_arcface_margin(pl_module, arcface_margin)

            pl_module.alpha = 0.0
            pl_module.beta = 0.0
            pl_module.alpha_adv = 0.0
            pl_module.lam_adv = 0.0

            self._set_moe_temperature(pl_module, self.moe_temp_phase1)

            trainer.logger.log_metrics(
                {
                    "phase/arcface_scale": scale,
                    "phase/arcface_margin": arcface_margin,
                    "phase/moe_temperature": self.moe_temp_phase1,
                    "phase/current": 1.0,
                },
                step=trainer.global_step,
            )

        # Phase 2: cosine soft-start on alpha/beta (5-epoch ramp),
        # sigmoid DANN ramp on lam_adv over full Phase 2, cosine ramp
        # on alpha_adv over full Phase 2.
        if pl_module.current_phase == 2:
            phase2_start = self._phase2_start
            phase2_end   = self._phase2_start + self.phase2_epochs
            ramp_end_short = min(phase2_start + 5, phase2_end)

            pl_module.alpha = self._cosine_ramp(
                epoch, phase2_start, ramp_end_short, self.alpha_target,
            )
            pl_module.beta = self._cosine_ramp(
                epoch, phase2_start, ramp_end_short, self.beta_target,
            )
            pl_module.alpha_adv = self._cosine_ramp(
                epoch, phase2_start, phase2_end, self.alpha_adv_target,
            )
            pl_module.lam_adv = self._dann_lambda(epoch, phase2_start, phase2_end)

            # ArcFace scale: 32 -> 64 over the short warmup too (tied to
            # identity stability). Safer than stretching it over full P2.
            scale_progress = self._cosine_ramp(
                epoch, phase2_start, ramp_end_short, 1.0, min_frac=0.0,
            )
            new_scale = self.arcface_scale_start + scale_progress * (
                self.arcface_scale_end - self.arcface_scale_start
            )
            for af_loss in pl_module.arcface_losses.values():
                af_loss.set_scale(new_scale)
            self._set_arcface_margin(pl_module, arcface_margin)

            # MoE temperature: linear 2.0 -> 1.0 across Phase 2
            p2_progress = min(
                (epoch - phase2_start) / max(self.phase2_epochs - 1, 1),
                1.0,
            )
            moe_temp = self.moe_temp_phase1 + p2_progress * (
                self.moe_temp_phase2_end - self.moe_temp_phase1
            )
            self._set_moe_temperature(pl_module, moe_temp)

            trainer.logger.log_metrics(
                {
                    "phase/alpha": pl_module.alpha,
                    "phase/beta": pl_module.beta,
                    "phase/alpha_adv": pl_module.alpha_adv,
                    "phase/lam_adv": pl_module.lam_adv,
                    "phase/arcface_scale": new_scale,
                    "phase/arcface_margin": arcface_margin,
                    "phase/moe_temperature": moe_temp,
                    "phase/current": 2.0,
                },
                step=trainer.global_step,
            )

        # Phase 3: everything at target, only MoE temp keeps decaying.
        if pl_module.current_phase == 3:
            phase3_epoch = epoch - self._phase3_start
            phase3_progress = min(phase3_epoch / max(self.phase3_epochs - 1, 1), 1.0)

            pl_module.alpha = self.alpha_target
            pl_module.beta = self.beta_target
            pl_module.alpha_adv = self.alpha_adv_target
            pl_module.lam_adv = 1.0

            moe_temp = self.moe_temp_phase2_end + phase3_progress * (
                self.moe_temp_phase3_end - self.moe_temp_phase2_end
            )
            self._set_moe_temperature(pl_module, moe_temp)
            self._set_arcface_margin(pl_module, arcface_margin)

            trainer.logger.log_metrics(
                {
                    "phase/alpha": pl_module.alpha,
                    "phase/beta": pl_module.beta,
                    "phase/alpha_adv": pl_module.alpha_adv,
                    "phase/lam_adv": pl_module.lam_adv,
                    "phase/arcface_margin": arcface_margin,
                    "phase/moe_temperature": moe_temp,
                    "phase/balancing_loss_weight": getattr(
                        pl_module,
                        "phase3_balancing_loss_weight",
                        0.0,
                    ),
                    "phase/current": 3.0,
                },
                step=trainer.global_step,
            )

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _transition_to_phase2(self, pl_module: Any, trainer: L.Trainer) -> None:
        """Phase 1 -> Phase 2: swap loader, reset ramped weights to near-zero."""
        pl_module.current_phase = 2
        pl_module.alpha = 0.0
        pl_module.beta = 0.0
        pl_module.alpha_adv = 0.0
        pl_module.lam_adv = 0.0

        # Phase 2 activation-memory pattern differs from Phase 1. Free
        # fragmented Phase-1 blocks before the new pattern settles.
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for af_loss in pl_module.arcface_losses.values():
            af_loss.set_scale(self.arcface_scale_start)
        self._set_arcface_margin(
            pl_module,
            self._arcface_margin_for_epoch(trainer.current_epoch),
        )

        self._set_moe_temperature(pl_module, self.moe_temp_phase1)

        # Reload dataloaders to switch from identity-only to combined
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            trainer.datamodule.current_phase = 2
        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.setup_data()

        print(
            f"\n{'='*60}\n"
            f"  PHASE 2 START (epoch {trainer.current_epoch})\n"
            f"  Joint ID+PAD batches (TASK_04)\n"
            f"  alpha target: {self.alpha_target}, beta target: {self.beta_target}\n"
            f"  alpha_adv target: {self.alpha_adv_target}\n"
            f"  Cosine soft-start over first 5 epochs; DANN ramp over full P2\n"
            f"{'='*60}\n"
        )

    def _transition_to_phase3(self, pl_module: Any, trainer: L.Trainer) -> None:
        pl_module.current_phase = 3
        pl_module.alpha = self.alpha_target
        pl_module.beta = self.beta_target
        pl_module.alpha_adv = self.alpha_adv_target
        pl_module.lam_adv = 1.0

        for af_loss in pl_module.arcface_losses.values():
            af_loss.set_scale(self.arcface_scale_end)
        self._set_arcface_margin(
            pl_module,
            self._arcface_margin_for_epoch(trainer.current_epoch),
        )
        if hasattr(pl_module, "phase3_balancing_loss_weight"):
            pl_module.phase3_balancing_loss_weight = 0.0

        self._set_moe_temperature(pl_module, self.moe_temp_phase2_end)

        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            trainer.datamodule.current_phase = 3
        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.setup_data()

        print(
            f"\n{'='*60}\n"
            f"  PHASE 3 START (epoch {trainer.current_epoch})\n"
            f"  Joint refinement — all losses at target\n"
            f"  alpha={pl_module.alpha}, beta={pl_module.beta}, "
            f"alpha_adv={pl_module.alpha_adv}\n"
            f"{'='*60}\n"
        )

    @staticmethod
    def _set_moe_temperature(pl_module: Any, temperature: float) -> None:
        if hasattr(pl_module, 'backbone') and hasattr(pl_module.backbone, 'set_moe_temperature'):
            pl_module.backbone.set_moe_temperature(temperature)
