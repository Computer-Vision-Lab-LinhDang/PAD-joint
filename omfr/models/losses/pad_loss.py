"""
pad_loss.py - Hybrid PAD classification loss.

Combines BCE, focal BCE, hard spoof weighting, and OHEM. The config keys
mirror the April 26 runs:

    pad_bce_weight, pad_focal_alpha, pad_focal_gamma, pad_focal_weight,
    pad_hard_spoof_sensor_ids, pad_hard_spoof_material_ids,
    pad_hard_spoof_weight, pad_ohem_fraction, pad_ohem_weight
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from omfr.models.losses.focal_bce import FocalBCELoss


def _unwrap_wandb_value(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both normal YAML and W&B exported `key: {value: ...}` blocks."""
    value = cfg.get("value")
    if isinstance(value, dict):
        return value
    return cfg


class PADLoss(nn.Module):
    """PAD BCE/focal loss with optional hard spoof weighting and spoof OHEM."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.0,
        hard_spoof_sensor_ids: Optional[list[int]] = None,
        hard_spoof_material_ids: Optional[list[int]] = None,
        hard_spoof_weight: float = 1.0,
        ohem_fraction: float = 0.0,
        ohem_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.focal_weight = float(focal_weight)
        self.hard_spoof_weight = float(hard_spoof_weight)
        self.ohem_fraction = float(ohem_fraction)
        self.ohem_weight = float(ohem_weight)
        self.hard_spoof_sensor_ids = {int(x) for x in (hard_spoof_sensor_ids or [])}
        self.hard_spoof_material_ids = {int(x) for x in (hard_spoof_material_ids or [])}
        self.focal = FocalBCELoss(gamma=float(focal_gamma), alpha=float(focal_alpha))

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PADLoss":
        cfg = _unwrap_wandb_value(cfg or {})
        return cls(
            bce_weight=cfg.get("pad_bce_weight", 1.0),
            focal_alpha=cfg.get("pad_focal_alpha", 0.5),
            focal_gamma=cfg.get("pad_focal_gamma", 2.0),
            focal_weight=cfg.get("pad_focal_weight", 0.0),
            hard_spoof_sensor_ids=cfg.get("pad_hard_spoof_sensor_ids", []),
            hard_spoof_material_ids=cfg.get("pad_hard_spoof_material_ids", []),
            hard_spoof_weight=cfg.get("pad_hard_spoof_weight", 1.0),
            ohem_fraction=cfg.get("pad_ohem_fraction", 0.0),
            ohem_weight=cfg.get("pad_ohem_weight", 1.0),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sensor_labels: Optional[torch.Tensor] = None,
        material_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = logits.squeeze(-1).reshape(-1)
        targets = targets.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        sample_weight = self._sample_weights(
            logits,
            targets,
            sensor_labels=sensor_labels,
            material_labels=material_labels,
        )

        bce_per_sample = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        bce = self._weighted_mean(bce_per_sample, sample_weight)
        focal = self.focal(logits, targets, sample_weight=sample_weight)
        total = self.bce_weight * bce + self.focal_weight * focal

        return {
            "total": total,
            "bce": bce,
            "focal": focal,
            "sample_weight_mean": sample_weight.mean().detach(),
            "sample_weight_max": sample_weight.max().detach(),
        }

    def _sample_weights(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sensor_labels: Optional[torch.Tensor] = None,
        material_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weights = torch.ones_like(targets, dtype=logits.dtype)
        spoof_mask = targets == 0

        if self.hard_spoof_weight > 1.0 and spoof_mask.any():
            hard_group = spoof_mask.clone()

            if self.hard_spoof_sensor_ids:
                if sensor_labels is None:
                    hard_group = torch.zeros_like(hard_group)
                else:
                    sensor = sensor_labels.to(logits.device).long().reshape(-1)
                    sensor_ids = torch.tensor(
                        sorted(self.hard_spoof_sensor_ids),
                        device=logits.device,
                        dtype=torch.long,
                    )
                    hard_group = hard_group & torch.isin(sensor, sensor_ids)

            if self.hard_spoof_material_ids:
                if material_labels is None:
                    hard_group = torch.zeros_like(hard_group)
                else:
                    material = material_labels.to(logits.device).long().reshape(-1)
                    material_ids = torch.tensor(
                        sorted(self.hard_spoof_material_ids),
                        device=logits.device,
                        dtype=torch.long,
                    )
                    hard_group = hard_group & torch.isin(material, material_ids)

            weights = torch.where(
                hard_group,
                weights * self.hard_spoof_weight,
                weights,
            )

        if self.ohem_fraction > 0.0 and self.ohem_weight > 1.0 and spoof_mask.any():
            with torch.no_grad():
                losses = F.binary_cross_entropy_with_logits(
                    logits.detach(), targets, reduction="none",
                )
                spoof_indices = spoof_mask.nonzero(as_tuple=False).flatten()
                k = max(1, int(round(float(spoof_indices.numel()) * self.ohem_fraction)))
                k = min(k, int(spoof_indices.numel()))
                hard_rel = losses[spoof_indices].topk(k).indices
                hard_indices = spoof_indices[hard_rel]
            weights = weights.clone()
            weights[hard_indices] = weights[hard_indices] * self.ohem_weight

        return weights

    @staticmethod
    def _weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def extra_repr(self) -> str:
        return (
            f"bce_weight={self.bce_weight}, focal_weight={self.focal_weight}, "
            f"hard_spoof_weight={self.hard_spoof_weight}, "
            f"ohem_fraction={self.ohem_fraction}, ohem_weight={self.ohem_weight}"
        )
