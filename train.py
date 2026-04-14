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
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from omfr.models.omfr import OMFRModule
from omfr.data.datamodule import OMFRDataModule
from omfr.callbacks.phase_scheduler import PhaseSchedulerCallback
from omfr.callbacks.gradient_monitor import GradientMonitor


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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

    phase_scheduler = PhaseSchedulerCallback(
        phase1_epochs=phase_cfg.get("phase1_epochs", 20),
        phase2_epochs=phase_cfg.get("phase2_epochs", 20),
        phase3_epochs=phase_cfg.get("phase3_epochs", 20),
        warmup_epochs=phase_cfg.get("warmup_epochs", 5),
        alpha_target=phase_cfg.get("alpha_target", 1.0),
        beta_target=phase_cfg.get("beta_target", 0.1),
        arcface_scale_start=phase_cfg.get("arcface_scale_start", 32.0),
        arcface_scale_end=phase_cfg.get("arcface_scale_end", 64.0),
    )

    gradient_monitor = GradientMonitor(log_every_n_steps=50)

    checkpoint = ModelCheckpoint(
        dirpath=ckpt_cfg.get("dirpath", "checkpoints/"),
        filename=ckpt_cfg.get(
            "filename", "omfr-{epoch:03d}-{val/cascaded_IM:.4f}"
        ),
        monitor=ckpt_cfg.get("monitor", "val/cascaded_IM"),
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=ckpt_cfg.get("save_top_k", 3),
        save_last=ckpt_cfg.get("save_last", True),
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress   = RichProgressBar()

    return [phase_scheduler, gradient_monitor, checkpoint, lr_monitor, progress]


def build_logger(config: Dict[str, Any]) -> TensorBoardLogger:
    log_cfg = config.get("logging", {})
    return TensorBoardLogger(
        save_dir=log_cfg.get("save_dir", "logs/"),
        name=log_cfg.get("name", "omfr"),
        version=log_cfg.get("version", None),
    )


def build_trainer(
    config: Dict[str, Any],
    callbacks: list,
    logger: Any,
) -> L.Trainer:
    tr_cfg = config.get("trainer", {})
    return L.Trainer(
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
        # Attempt numeric conversion
        try:
            val: Any = int(raw_val)
        except ValueError:
            try:
                val = float(raw_val)
            except ValueError:
                val = raw_val
        result[key] = val
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    L.seed_everything(args.seed, workers=True)

    # ── Load config ──
    config = _load_config(args.config)
    if args.overrides:
        _apply_overrides(config, _parse_overrides(args.overrides))

    # ── Build components ──
    model      = build_module(config)
    datamodule = build_datamodule(config)
    callbacks  = build_callbacks(config)
    logger     = build_logger(config)
    trainer    = build_trainer(config, callbacks, logger)

    # ── Train ──
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
