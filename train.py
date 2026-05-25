"""
train.py — OMFR Training Entry Point

Usage:
    python train.py                                   # uses configs/base.yaml
    python train.py --config configs/base.yaml
    python train.py --config configs/base.yaml --lr 5e-5 --total_epochs 80

The script:
    1. Loads YAML config (with optional CLI key overrides)
    2. Instantiates OMFRModule + OMFRDataModule
    3. Configures Trainer with callbacks (PhaseScheduler, GradientMonitor,
       ModelCheckpoint, LearningRateMonitor)
    4. Calls trainer.fit()
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from omfr.models.omfr import OMFRModule
from omfr.data.datamodule import OMFRDataModule
from omfr.callbacks.phase_scheduler import PhaseSchedulerCallback
from omfr.callbacks.gradient_monitor import GradientMonitor


class PhaseAwareEarlyStopping(EarlyStopping):
    """EarlyStopping that only starts checking once Phase 3 is active."""

    def __init__(self, min_phase: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_phase = int(min_phase)

    def _run_early_stopping_check(self, trainer: L.Trainer) -> None:
        pl_module = getattr(trainer, "lightning_module", None)
        current_phase = int(getattr(pl_module, "current_phase", 1))
        if current_phase < self.min_phase:
            return
        super()._run_early_stopping_check(trainer)


class PhaseGatedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that only tracks/saves during ONE training phase.

    Fixes the "tricked checkpoint" bug: a single checkpoint monitoring
    `val/cascaded_IM` kept a Phase-1 epoch (PAD untrained) as global
    best, so eval EER was ~50%. Two gated instances give a clean
    `best_phase1` (identity teacher source) and `best_phase2` (the
    deployable joint model) — each best-tracked only within its phase.
    """

    def __init__(
        self,
        phase_tag: int,
        identity_guard: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.omfr_phase_tag = int(phase_tag)
        self.identity_guard = identity_guard or {}
        # Not persisted in state_dict — guarantees a one-time reset the
        # first time THIS process sees its phase active, even if state
        # was restored from a prior run's checkpoint.
        self._reset_done_this_run = False

    @property
    def state_key(self) -> str:
        # Two instances coexist (phase 1 & 2). Lightning requires a
        # unique state_key per stateful callback, otherwise it raises
        # "Found more than one stateful callback of type ...". Tag the
        # base ModelCheckpoint state_key with the phase so checkpoint
        # callback state round-trips correctly on resume.
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            omfr_phase_tag=self.omfr_phase_tag,
        )

    def _in_phase(self, trainer: L.Trainer) -> bool:
        pl = getattr(trainer, "lightning_module", None)
        return int(getattr(pl, "current_phase", 1)) == self.omfr_phase_tag

    @staticmethod
    def _metric_float(trainer: L.Trainer, name: str) -> Optional[float]:
        value = trainer.callback_metrics.get(name)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu())
        return float(value)

    def _passes_identity_guard(self, trainer: L.Trainer, pl_module: Any) -> bool:
        guard = self.identity_guard
        if not guard.get("enabled", False):
            return True
        if self.omfr_phase_tag < int(guard.get("min_phase", 2)):
            return True

        failures = []
        checks = (
            ("val/identity_rank1", "min_rank1", "max_rank1_drop"),
            ("val/tar_at_far", "min_tar_at_far", "max_tar_drop"),
        )
        baseline = dict(getattr(pl_module, "_identity_guard_baseline", {}) or {})
        if guard.get("baseline_rank1") is not None:
            baseline["val/identity_rank1"] = float(guard["baseline_rank1"])
        if guard.get("baseline_tar_at_far") is not None:
            baseline["val/tar_at_far"] = float(guard["baseline_tar_at_far"])

        for metric_name, min_key, drop_key in checks:
            current = self._metric_float(trainer, metric_name)
            if current is None:
                if guard.get("require_metrics", True):
                    failures.append(f"{metric_name}=missing")
                continue

            min_value = guard.get(min_key)
            if min_value is not None and current < float(min_value):
                failures.append(f"{metric_name}={current:.4f} < {float(min_value):.4f}")

            max_drop = guard.get(drop_key)
            base_value = baseline.get(metric_name)
            if max_drop is not None and base_value is not None:
                floor = float(base_value) - float(max_drop)
                if current < floor:
                    failures.append(
                        f"{metric_name}={current:.4f} < baseline-drop {floor:.4f}"
                    )

        passed = not failures
        trainer.logger.log_metrics(
            {"checkpoint/identity_guard_pass": 1.0 if passed else 0.0},
            step=trainer.global_step,
        )
        if not passed:
            print(
                "[identity-guard] reject checkpoint: "
                + " | ".join(failures)
            )
        return passed

    def on_validation_end(self, trainer: L.Trainer, pl_module: Any) -> None:
        # Skip entirely (incl. monitor/best bookkeeping) outside the
        # target phase, so best is chosen only among in-phase epochs.
        if not self._in_phase(trainer):
            return
        if not self._passes_identity_guard(trainer, pl_module):
            return
        if not self._reset_done_this_run:
            # First time this RUN enters our phase. Drop any best_score
            # restored from a previous run's state_dict — a Phase-1
            # `val/cascaded_IM` peak must not block Phase-2 saves once
            # PAD's BPCER filter trims genuine accept rate.
            self.best_model_score = None
            self.best_k_models = {}
            self.kth_best_model_path = ""
            self.best_model_path = ""
            self.current_score = None
            self._reset_done_this_run = True
        super().on_validation_end(trainer, pl_module)


class FreshStartSWA(StochasticWeightAveraging):
    """
    Resume the trainer/optimizer state from a checkpoint, but start SWA fresh.

    Old Phase-3 checkpoints may contain SWA callback state created with a much
    larger SWA LR. Reusing the same state key lets Lightning route that old
    state here, then this callback intentionally discards it so the configured
    `swa_lrs` is the one that takes effect after resume.
    """

    @property
    def state_key(self) -> str:
        return "StochasticWeightAveraging"

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.n_averaged = None
        self._swa_scheduler = None
        self._initialized = False
        self._init_n_averaged = 0
        self._latest_update_epoch = -1
        self._scheduler_state = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return _unwrap_wandb_values(yaml.safe_load(f))


def _unwrap_wandb_values(value: Any) -> Any:
    """Convert W&B exported config blocks from {'value': x} to x."""
    if isinstance(value, dict):
        if set(value.keys()) == {"value"}:
            return _unwrap_wandb_values(value["value"])
        return {k: _unwrap_wandb_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap_wandb_values(v) for v in value]
    return value


def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Apply flat key=value overrides onto a nested config dict."""
    for key, value in overrides.items():
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value


def _deep_get(d: Dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


def _prepare_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = dict(config)
    data_cfg = config.get("data", {})
    losses_cfg = config.get("losses", {})
    optimizer_cfg = config.get("optimizer", {})
    phases_cfg = config.get("phases", {})
    identity_head_cfg = config.get("identity_head", {})

    runtime["identity_root"] = data_cfg.get(
        "identity_data_root", config.get("identity_root", "")
    )
    runtime["pad_root"] = data_cfg.get(
        "pad_data_root", config.get("pad_root", "")
    )
    runtime["joint_root"] = data_cfg.get(
        "joint_data_root", config.get("joint_root", "")
    )
    runtime["identity_datasets"] = data_cfg.get(
        "identity_datasets", config.get("identity_datasets", [])
    )
    runtime["pad_datasets"] = data_cfg.get(
        "pad_datasets", config.get("pad_datasets", [])
    )
    runtime["joint_datasets"] = data_cfg.get(
        "joint_datasets", config.get("joint_datasets", [])
    )
    runtime["num_workers"] = data_cfg.get("num_workers", config.get("num_workers", 8))
    runtime["pk_P"] = data_cfg.get("pk_p", config.get("pk_P", 32))
    runtime["pk_K"] = data_cfg.get("pk_k", config.get("pk_K", 4))
    runtime["pad_batch_size"] = data_cfg.get(
        "pad_batch_size", config.get("pad_batch_size", 128)
    )
    runtime["val_batch_size"] = data_cfg.get(
        "val_batch_size", config.get("val_batch_size", 64)
    )
    runtime["pin_memory"] = data_cfg.get("pin_memory", config.get("pin_memory", True))

    runtime["num_classes"] = identity_head_cfg.get(
        "num_classes", config.get("num_classes", 0)
    )
    runtime["lr"] = optimizer_cfg.get("lr", config.get("lr", 1e-4))
    runtime["weight_decay"] = optimizer_cfg.get(
        "weight_decay", config.get("weight_decay", 0.05)
    )
    runtime["total_epochs"] = phases_cfg.get(
        "total_epochs", config.get("total_epochs", 60)
    )
    runtime["gamma"] = losses_cfg.get("gamma", config.get("gamma", 0.01))
    runtime["balancing_loss_weight"] = losses_cfg.get(
        "balancing_loss_weight",
        config.get("balancing_loss_weight", runtime["gamma"]),
    )
    runtime["phase2_identity_loss_weight"] = losses_cfg.get(
        "phase2_identity_loss_weight",
        config.get("phase2_identity_loss_weight", 0.0),
    )
    runtime["phase2_balance_loss_weight"] = losses_cfg.get(
        "phase2_balance_loss_weight",
        config.get("phase2_balance_loss_weight", 0.0),
    )
    runtime["pad_supcon_weight"] = losses_cfg.get(
        "pad_supcon_weight", config.get("pad_supcon_weight", 1.0)
    )
    runtime["pad_bce_phase_weight"] = losses_cfg.get(
        "pad_bce_phase_weight", config.get("pad_bce_phase_weight", 1.0)
    )
    runtime["identity_supcon_weight"] = losses_cfg.get(
        "identity_supcon_weight", config.get("identity_supcon_weight", 0.7)
    )
    runtime["identity_arcface_weight"] = losses_cfg.get(
        "identity_arcface_weight", config.get("identity_arcface_weight", 0.1)
    )
    runtime["warmup_epochs"] = phases_cfg.get("warmup_epochs", 5)
    runtime["phase2_freeze_identity"] = phases_cfg.get(
        "phase2_freeze_identity",
        config.get("phase2_freeze_identity", True),
    )
    backbone_cfg = config.get("backbone", {})
    runtime["pretrained"] = backbone_cfg.get("pretrained", True)
    runtime["grad_checkpoint"] = backbone_cfg.get(
        "grad_checkpoint", config.get("grad_checkpoint", False)
    )
    return runtime


def _infer_num_classes(datamodule: OMFRDataModule) -> int:
    counts = []
    if datamodule.identity_ds is not None:
        counts.append(int(datamodule.identity_ds.num_classes))
    if datamodule.joint_ds is not None:
        counts.append(int(datamodule.joint_ds.num_classes))
    return max(counts, default=0)


# ---------------------------------------------------------------------------
# Build objects from config
# ---------------------------------------------------------------------------

def build_module(config: Dict[str, Any]) -> OMFRModule:
    return OMFRModule(config=config)


def build_datamodule(config: Dict[str, Any]) -> OMFRDataModule:
    return OMFRDataModule(config=config)


def build_callbacks(config: Dict[str, Any]) -> list:
    phase_cfg = config.get("phases", {})
    ckpt_cfg  = config.get("checkpoint", {})
    tr_cfg    = config.get("trainer", {})
    cb_cfg    = config.get("callbacks", {})

    plateau_cfg = (
        cb_cfg.get("dynamic_phase", {}) if isinstance(cb_cfg, dict) else {}
    )
    phase_scheduler = PhaseSchedulerCallback(
        phase1_max_epochs=phase_cfg.get(
            "phase1_max_epochs", phase_cfg.get("phase1_epochs", 80)
        ),
        phase2_epochs=phase_cfg.get("phase2_epochs", 40),
        plateau_patience=plateau_cfg.get(
            "patience", phase_cfg.get("patience", 5)
        ),
        plateau_monitor=plateau_cfg.get("monitor", "val/cascaded_IM"),
        plateau_mode=plateau_cfg.get("mode", "max"),
        plateau_min_delta=float(plateau_cfg.get("min_delta", 1.0e-4)),
        warmup_epochs=phase_cfg.get("warmup_epochs", 5),
        phase1_warmup_epochs=phase_cfg.get("phase1_warmup_epochs", 5),
        phase1_warmup_delay=phase_cfg.get("phase1_warmup_delay", 5),
        alpha_target=phase_cfg.get("alpha_target", 1.0),
        beta_target=phase_cfg.get("beta_target", 0.02),
        gamma_target=phase_cfg.get("gamma_target", 1.0),
        alpha_adv_target=phase_cfg.get("alpha_adv_target", 0.0),
        arcface_scale_init=phase_cfg.get("arcface_scale_init", 1.0),
        arcface_scale_start=phase_cfg.get("arcface_scale_start", 32.0),
        arcface_scale_end=phase_cfg.get("arcface_scale_end", 48.0),
        arcface_margin_init=phase_cfg.get("arcface_margin_init", 0.0),
        arcface_margin_target=phase_cfg.get("arcface_margin_target", 0.5),
        moe_temp_phase1=phase_cfg.get("moe_temp_phase1", 2.0),
        moe_temp_phase2_end=phase_cfg.get("moe_temp_phase2_end", 1.0),
        phase1_task=phase_cfg.get("phase1_task", "identity"),
        transition_on_plateau=phase_cfg.get("transition_on_plateau", True),
    )

    gradient_monitor = GradientMonitor(log_every_n_steps=50)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_cfg = cb_cfg.get("early_stopping", {}) if isinstance(cb_cfg, dict) else {}
    early_enabled = early_cfg.get("enabled", True)
    if early_enabled:
        early_stop = PhaseAwareEarlyStopping(
            monitor=early_cfg.get("monitor", "val/cascaded_IM"),
            mode=early_cfg.get("mode", "max"),
            patience=int(early_cfg.get("patience", 7)),
            min_phase=int(early_cfg.get("min_phase", 3)),
        )
    else:
        early_stop = None
    progress   = RichProgressBar()

    callbacks = [phase_scheduler, gradient_monitor, lr_monitor, progress]
    if early_stop is not None:
        callbacks.insert(2, early_stop)
    if tr_cfg.get("enable_checkpointing", True):
        dirpath = ckpt_cfg.get("dirpath", "checkpoints/")
        monitor = ckpt_cfg.get("monitor", "val/cascaded_IM")
        mode    = ckpt_cfg.get("mode", "max")
        # Phase-2 uses a PAD-native metric: cascaded_IM is unfair across
        # phases (BPCER trims genuine acceptance once PAD turns on), so
        # the deployable joint model is selected by PAD quality directly.
        phase2_monitor = ckpt_cfg.get("phase2_monitor", "val/pad_accuracy")
        phase2_mode    = ckpt_cfg.get("phase2_mode", "max")
        # Phase-1 best — fixed filename so the teacher path is stable.
        ckpt_phase1 = PhaseGatedModelCheckpoint(
            phase_tag=1,
            dirpath=dirpath,
            filename="best_phase1",
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_last=False,
            enable_version_counter=False,
        )
        # Phase-2 best — the deployable joint model (PAD trained).
        ckpt_phase2 = PhaseGatedModelCheckpoint(
            phase_tag=2,
            identity_guard=ckpt_cfg.get("identity_guard", {}),
            dirpath=dirpath,
            filename="best_phase2",
            monitor=phase2_monitor,
            mode=phase2_mode,
            save_top_k=1,
            save_last=False,
            enable_version_counter=False,
        )
        # Phase-agnostic rolling `last.ckpt` for crash/resume safety.
        ckpt_last = ModelCheckpoint(
            dirpath=dirpath,
            filename="omfr-{epoch:03d}",
            save_top_k=0,
            save_last=True,
        )
        callbacks.insert(2, ckpt_phase1)
        callbacks.insert(3, ckpt_phase2)
        callbacks.insert(4, ckpt_last)

    swa_cfg = cb_cfg.get("swa", {}) if isinstance(cb_cfg, dict) else {}
    if swa_cfg.get("enabled", False):
        callbacks.append(FreshStartSWA(
            swa_lrs=float(swa_cfg.get("swa_lrs", 1.0e-5)),
            swa_epoch_start=int(swa_cfg.get("swa_epoch_start", 40)),
        ))

    return callbacks


def build_logger(config: Dict[str, Any]) -> list:
    log_cfg = config.get("logging", {})
    loggers = []

    try:
        loggers.append(TensorBoardLogger(
            save_dir=log_cfg.get("save_dir", "logs/"),
            name=log_cfg.get("name", "omfr"),
            version=log_cfg.get("version", None),
        ))
    except ModuleNotFoundError as exc:
        print(f"[logger] TensorBoard unavailable ({exc}); using CSVLogger.")
        loggers.append(CSVLogger(
            save_dir=log_cfg.get("save_dir", "logs/"),
            name=log_cfg.get("name", "omfr"),
            version=log_cfg.get("version", None),
        ))

    wandb_cfg = log_cfg.get("wandb", {})
    if wandb_cfg.get("enabled", True):
        try:
            loggers.append(WandbLogger(
                project=wandb_cfg.get("project", "omfr"),
                name=wandb_cfg.get("name", None),
                save_dir=log_cfg.get("save_dir", "logs/"),
                log_model=wandb_cfg.get("log_model", False),
                config=config,
            ))
        except ModuleNotFoundError as exc:
            print(f"[logger] W&B unavailable ({exc}); continuing without W&B.")

    return loggers


def build_trainer(
    config: Dict[str, Any],
    callbacks: list,
    logger: list,
) -> L.Trainer:
    tr_cfg = config.get("trainer", {})
    trainer_kwargs = dict(
        accelerator=tr_cfg.get("accelerator", "gpu"),
        devices=tr_cfg.get("devices", 1),
        precision=tr_cfg.get("precision", "16-mixed"),
        max_epochs=config.get("phases", {}).get("total_epochs", 60),
        gradient_clip_val=tr_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=tr_cfg.get("accumulate_grad_batches", 2),
        log_every_n_steps=tr_cfg.get("log_every_n_steps", 10),
        check_val_every_n_epoch=tr_cfg.get("check_val_every_n_epoch", 1),
        enable_progress_bar=tr_cfg.get("enable_progress_bar", True),
        deterministic=tr_cfg.get("deterministic", False),
        callbacks=callbacks,
        logger=logger,
    )
    optional_keys = (
        "fast_dev_run",
        "limit_train_batches",
        "limit_val_batches",
        "num_sanity_val_steps",
        "enable_checkpointing",
    )
    for key in optional_keys:
        if key in tr_cfg:
            trainer_kwargs[key] = tr_cfg[key]
    return L.Trainer(**trainer_kwargs)


def load_init_weights(
    model: OMFRModule,
    checkpoint_path: str,
    skip_prefixes: Optional[list[str]] = None,
) -> None:
    """Initialize model weights from a checkpoint without optimizer state.

    This is intentionally different from ``--resume``. It loads only matching
    tensors from the checkpoint state_dict, so a Phase-1 teacher trained with
    a different identity class count can still initialize the shared trunk and
    heads while ArcFace classifier weights are skipped.
    """
    skip_prefixes = skip_prefixes or []
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    own_state = model.state_dict()

    loaded: Dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped.append(f"{key}: skipped by prefix")
            continue
        if key not in own_state:
            skipped.append(f"{key}: missing in target")
            continue
        if tuple(value.shape) != tuple(own_state[key].shape):
            skipped.append(
                f"{key}: shape {tuple(value.shape)} != {tuple(own_state[key].shape)}"
            )
            continue
        loaded[key] = value

    missing, unexpected = model.load_state_dict(loaded, strict=False)
    print(
        f"[init-from] loaded {len(loaded)} tensors from {checkpoint_path}; "
        f"skipped {len(skipped)} tensors; missing {len(missing)}; "
        f"unexpected {len(unexpected)}"
    )
    preview = skipped[:20]
    if preview:
        print("[init-from] skipped preview:")
        for item in preview:
            print(f"  - {item}")
    if len(skipped) > len(preview):
        print(f"[init-from] ... {len(skipped) - len(preview)} more skipped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the OMFR (Orthogonal Matryoshka Fingerprint) model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help=(
            "Initialize model weights from a checkpoint without restoring "
            "optimizer/callback state. Shape-mismatched tensors are skipped."
        ),
    )
    parser.add_argument(
        "--init-skip-prefix",
        action="append",
        default=["arcface_losses."],
        help=(
            "State-dict prefix to skip when using --init-from. Can be passed "
            "multiple times. Defaults to skipping ArcFace classifiers."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    # Allow arbitrary config overrides as --key value (dot-separated for nested)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides as key=value pairs (e.g. optimizer.lr=1e-5).",
    )
    return parser.parse_args()


def _parse_overrides(override_list: list[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, raw_val = item.split("=", 1)
        try:
            val: Any = yaml.safe_load(raw_val)
        except yaml.YAMLError:
            val = raw_val
        result[key] = val
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.resume and args.init_from:
        raise ValueError("Use either --resume or --init-from, not both.")

    L.seed_everything(args.seed, workers=True)

    # ── Load config ──
    config = _load_config(args.config)
    if args.overrides:
        _apply_overrides(config, _parse_overrides(args.overrides))

    runtime_config = _prepare_runtime_config(config)

    # ── Build components ──
    datamodule = build_datamodule(runtime_config)
    datamodule.setup("fit")
    inferred_num_classes = _infer_num_classes(datamodule)
    if inferred_num_classes > 0:
        runtime_config["num_classes"] = inferred_num_classes
        config.setdefault("identity_head", {})["num_classes"] = inferred_num_classes

    model      = build_module(runtime_config)
    if args.init_from:
        load_init_weights(
            model,
            checkpoint_path=args.init_from,
            skip_prefixes=args.init_skip_prefix,
        )
    callbacks  = build_callbacks(config)
    logger     = build_logger(config)
    trainer    = build_trainer(config, callbacks, logger)

    # ── Train ──
    fit_kwargs = {"datamodule": datamodule, "ckpt_path": args.resume}
    if args.resume:
        # PyTorch 2.6 defaults torch.load(weights_only=True), but resuming
        # Lightning training needs optimizer/callback/loop state too.
        fit_kwargs["weights_only"] = False
    trainer.fit(model, **fit_kwargs)


if __name__ == "__main__":
    main()
