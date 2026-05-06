"""
omfr.py — Main OMFR Lightning Module (v2: TinyViT-5M Backbone)

Orchestrates:
    LearnableGaborStem → TinyViTBackbone (MoE) → IdentityHead + PADHead

Phase-aware training:
    Phase 1 (epochs  0-19): Identity only — MRL-ArcFace + L_balance
    Phase 2 (epochs 20-39): Alternating ID/PAD — all losses, alpha/beta ramp
    Phase 3 (epochs 40-59): Joint refinement — all losses fixed, spoof-masked ArcFace

Loss total:
    L = L_Identity + alpha * L_PAD + beta * L_orth + gamma * L_balance

Config dict keys:
    num_classes:   int    — number of training identities
    lr:            float  — base LR (default 1e-4)
    weight_decay:  float  — AdamW weight decay (default 0.05)
    total_epochs:  int    — total training epochs (default 60)
    gamma:         float  — MoE balance loss weight (default 0.01)
    pretrained:    bool   — load pretrained TinyViT (default True)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from omfr.models.backbone.gabor_stem import LearnableGaborStem
from omfr.models.backbone.pad_stem import PADStem
from omfr.models.backbone.tiny_vit import TinyViTBackbone
from omfr.models.backbone.fastvit import FastViTBackbone
from omfr.models.backbone.dinov2 import DINOv2Backbone
from omfr.models.heads.identity_head import IdentityHead
from omfr.models.heads.pad_head import PADHead
from omfr.models.losses.arcface import ArcFaceLoss
from omfr.models.losses.supcon import SupConLoss
from omfr.models.losses.orthogonal import OrthogonalityLoss
from omfr.models.losses.focal_bce import FocalBCELoss
from omfr.models.losses.mixup_consistency import MixUpConsistency
from omfr.models.losses.sensor_adversarial import (
    SensorAdversarialHead,
    SensorAdversarialLoss,
)
from omfr.models.heads.pad_head import PADHead as _PADHeadCls  # for constants


class OMFRModule(L.LightningModule):
    """
    OMFR main Lightning module with TinyViT-5M backbone.

    Args:
        config: dict with training and model hyperparameters.
    """

    MRL_DIMS = [64, 128, 256]

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = config
        self.save_hyperparameters(config)

        num_classes: int = config["num_classes"]
        pretrained: bool = config.get("pretrained", True)
        grad_checkpoint: bool = bool(config.get("grad_checkpoint", False))

        # -- Model components --
        self.gabor = LearnableGaborStem()
        # Legacy PAD-specific modules are kept in the module/state_dict for
        # checkpoint compatibility. The unified PAD path below now consumes
        # the shared Gabor + shared backbone representation instead.
        self.gabor_pad = LearnableGaborStem(init_frequency=0.35)

        # Backbone dispatch. `backbone.name` selects the implementation:
        #   tiny_vit_5m_224          -> TinyViTBackbone
        #   fastvit_sa12             -> FastViTBackbone
        #   vit_*_patch14_*_dinov2   -> DINOv2Backbone
        # Stage dims differ between backbones and must flow into the heads.
        backbone_cfg = config.get("backbone", {}) or {}
        backbone_name = str(backbone_cfg.get("name", "tiny_vit_5m_224"))
        num_experts = int(backbone_cfg.get("num_experts", 4))
        top_k = int(backbone_cfg.get("top_k", 2))

        if "dinov2" in backbone_name:
            self.backbone = DINOv2Backbone(
                pretrained=pretrained,
                in_chans=int(backbone_cfg.get("in_chans", 8)),
                model_name=backbone_name,
                img_size=int(
                    backbone_cfg.get(
                        "img_size",
                        config.get("data", {}).get("image_size", 224),
                    )
                ),
                use_grad_checkpoint=grad_checkpoint,
                stage1_dim=int(backbone_cfg.get("stage1_dim", 64)),
                stage2_dim=int(backbone_cfg.get("stage2_dim", 128)),
                lora_cfg=config.get("lora") or backbone_cfg.get("lora"),
            )
            default_stage1, default_stage2, default_stage3, default_stage4 = (
                self.backbone.out_dims
            )
        elif backbone_name.startswith("fastvit"):
            self.backbone = FastViTBackbone(
                pretrained=pretrained,
                in_chans=8,
                num_experts=num_experts,
                top_k=top_k,
                use_grad_checkpoint=grad_checkpoint,
                model_name=backbone_name,
            )
            default_stage1, default_stage2, default_stage3, default_stage4 = (
                FastViTBackbone.OUT_DIMS
            )
        else:
            self.backbone = TinyViTBackbone(
                pretrained=pretrained,
                in_chans=8,
                num_experts=num_experts,
                top_k=top_k,
                use_grad_checkpoint=grad_checkpoint,
            )
            default_stage1, default_stage2, default_stage3, default_stage4 = (
                64, 128, 160, 320,
            )

        id_cfg = config.get("identity_head", {}) or {}
        pad_cfg = config.get("pad_head", {}) or {}

        self.pad_stem = PADStem()
        identity_embed_dim = int(id_cfg.get("embed_dim", 256))
        # PADHead shared_dim must match backbone stage3_feat channels (384 for DINOv2).
        # Falls back to identity_embed_dim for legacy FastViT checkpoints.
        backbone_stage3_dim = int(id_cfg.get("stage3_dim", default_stage3))
        self.pad_head = PADHead(
            stage1_dim=int(pad_cfg.get("stage1_dim", default_stage1)),
            stage2_dim=int(pad_cfg.get("stage2_dim", default_stage2)),
            shared_dim=int(pad_cfg.get("shared_dim", backbone_stage3_dim)),
            gabor_dim=int(pad_cfg.get("gabor_dim", 8)),
        )
        self.identity_head = IdentityHead(
            embed_dim=identity_embed_dim,
            num_heads=int(id_cfg.get("num_heads", 8)),
            grid_size=int(id_cfg.get("grid_size", 14)),
            num_queries=int(id_cfg.get("num_queries", 4)),
            stage3_dim=int(id_cfg.get("stage3_dim", default_stage3)),
            stage4_dim=int(id_cfg.get("stage4_dim", default_stage4)),
        )

        # -- Losses --
        losses_cfg = config.get("losses", {}) or {}
        self.identity_ignore_index: int = int(losses_cfg.get("identity_ignore_index", -1))
        # ArcFace: one per MRL dim, weights stored HERE (not in IdentityHead)
        # Init with s=1.0, margin=0.0 — PhaseSchedulerCallback warms up to s=32, m=0.5
        self.arcface_losses = nn.ModuleDict({
            "64":  ArcFaceLoss(
                64, num_classes=num_classes, s=1.0, margin=0.0,
                ignore_index=self.identity_ignore_index,
            ),
            "128": ArcFaceLoss(
                128, num_classes=num_classes, s=1.0, margin=0.0,
                ignore_index=self.identity_ignore_index,
            ),
            "256": ArcFaceLoss(
                256, num_classes=num_classes, s=1.0, margin=0.0,
                ignore_index=self.identity_ignore_index,
            ),
        })
        # PAD branch — hybrid BCE + focal on top of the shared latent
        # representation plus task-conditioned routing.
        self.pad_focal_weight = float(losses_cfg.get("pad_focal_weight", 1.0))
        self.pad_bce_weight = float(losses_cfg.get("pad_bce_weight", 0.5))
        self.pad_focal_loss = FocalBCELoss(
            gamma=float(losses_cfg.get("pad_focal_gamma", 1.0)),
            alpha=float(losses_cfg.get("pad_focal_alpha", 0.5)),
        )
        self.pad_hard_spoof_weight = float(
            losses_cfg.get("pad_hard_spoof_weight", 2.0)
        )
        self.pad_hard_spoof_sensor_ids = {
            int(x) for x in losses_cfg.get("pad_hard_spoof_sensor_ids", [6])
        }
        self.pad_hard_spoof_material_ids = {
            int(x) for x in losses_cfg.get("pad_hard_spoof_material_ids", [4, 6, 7])
        }
        self.pad_ohem_fraction = float(losses_cfg.get("pad_ohem_fraction", 0.25))
        self.pad_ohem_weight = float(losses_cfg.get("pad_ohem_weight", 1.5))
        self.pad_mixup_loss = MixUpConsistency(alpha=0.4)
        self.bce_loss = nn.BCEWithLogitsLoss()
        # Dedicated SupCon for identity embeddings — higher temperature so
        # the contrastive signal is smoother across many identities.
        self.supcon_identity_loss = SupConLoss(temperature=0.1)
        self.orth_loss   = OrthogonalityLoss()

        # -- Sensor adversarial (TASK_03): forces pad_features to be
        # sensor-invariant via a gradient-reversal layer. num_sensors
        # is read from PADDataset.SENSOR_NAMES (default 9) and can be
        # overridden in config.
        num_sensors = int(config.get("num_sensors", 9))
        self.sensor_adv_head = SensorAdversarialHead(
            in_features=_PADHeadCls.PAD_FEATURES_DIM,
            num_sensors=num_sensors,
            hidden_dim=128,
        )
        self.sensor_adv_loss = SensorAdversarialLoss()

        # -- Loss weights --
        self.alpha: float = 0.0   # PAD weight — ramped in Phase 2
        self.beta:  float = 0.0   # bridge-consistency weight — ramped in Phase 2
        self.alpha_adv: float = 0.0  # sensor-adv weight, ramped in Phase 2
        self.lam_adv:   float = 0.0  # GRL lambda, ramped in Phase 2
        self.gamma: float = float(config.get("gamma", 0.01))
        # Hybrid identity loss: weighted mix of SupCon (open-set) and
        # ArcFace (class discriminative). SupCon dominates so embeddings
        # generalize across unseen identities (FVC open-set scenario).
        self.identity_supcon_weight: float = float(
            config.get("identity_supcon_weight", 0.7)
        )
        self.identity_arcface_weight: float = float(
            config.get("identity_arcface_weight", 0.3)
        )
        self.bridge_mrl_dims: List[int] = [
            int(dim) for dim in config.get("bridge_mrl_dims", [64])
        ]

        # AdaLoRA-style orthogonality regularizer weight on (P^T P - I)
        # and (Q Q^T - I). Scaled per-layer so the magnitude is ~O(1).
        self.lora_orth_weight: float = float(
            (config.get("lora") or {}).get("orth_weight", 0.0)
        )
        # Barlow-Twins orthogonality between PAD and identity embedding subspaces.
        # Ramped in Phase 2 alongside alpha. Start at 0 — PhaseScheduler sets it.
        self.beta_orth: float = 0.0

        # -- Phase state --
        self.current_phase: int = 1

        # -- Validation accumulators --
        self._val_id_embeddings:   List[torch.Tensor] = []
        self._val_id_labels:       List[torch.Tensor] = []
        self._val_id_pad_logits:   List[torch.Tensor] = []  # PAD logits for identity samples (cascaded_IM)
        self._val_pad_logits:      List[torch.Tensor] = []
        self._val_liveness_labels: List[torch.Tensor] = []

    # -------------------------------------------------------------------------
    # Internal forward helpers
    # -------------------------------------------------------------------------

    def _run_backbone(
        self,
        images: torch.Tensor,
        backbone_no_grad: bool = False,
        route_mode: str = "identity",
        detach_backbone_outputs: bool = False,
    ) -> Dict:
        """Shared Gabor stem -> shared backbone with task-conditioned routing."""
        enhanced = self.gabor(images)        # (B, 8, 224, 224)
        if backbone_no_grad:
            with torch.no_grad():
                out = self.backbone(enhanced, route_mode=route_mode)
        else:
            out = self.backbone(enhanced, route_mode=route_mode)
        if backbone_no_grad and route_mode == "pad":
            self.backbone.refresh_gateonly_stats(out["routing_stats"], route_mode)
        if detach_backbone_outputs:
            out["stage1_feat"] = out["stage1_feat"].detach()
            out["stage2_feat"] = out["stage2_feat"].detach()
            out["stage3_feat"] = out["stage3_feat"].detach()
            out["stage4_feat"] = out["stage4_feat"].detach()
            out["balance_losses"] = [
                loss.detach() if torch.is_tensor(loss) else loss
                for loss in out["balance_losses"]
            ]
            routing_stats = {}
            for key, stats in out["routing_stats"].items():
                detached_stats = {}
                for name, value in stats.items():
                    if name in {
                        "expert_weights_gateonly",
                        "token_entropy_gateonly",
                        "gate_input_gateonly",
                        "router_mode",
                    }:
                        detached_stats[name] = value
                    elif torch.is_tensor(value):
                        detached_stats[name] = value.detach()
                    else:
                        detached_stats[name] = value
                routing_stats[key] = detached_stats
            out["routing_stats"] = routing_stats
        out["gabor_feat"]     = enhanced
        # Legacy alias preserved for scripts/checkpoints that still inspect it.
        out["gabor_pad_feat"] = enhanced
        return out

    def _run_identity(self, backbone_out: Dict) -> Dict:
        """Identity head using stage3 + stage4 features."""
        return self.identity_head({
            "stage3_feat": backbone_out["stage3_feat"],
            "stage4_feat": backbone_out["stage4_feat"],
        })

    def _run_pad(self, backbone_out: Dict) -> Dict:
        """PAD head using backbone stage3_feat directly (no identity head dependency)."""
        rs = backbone_out["routing_stats"]

        def _detach_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            res = {
                "expert_weights": stats["expert_weights"].detach(),
                "token_entropy":  stats["token_entropy"].detach(),
            }
            if "gate_input_gateonly" in stats:
                res["gate_input_gateonly"] = stats["gate_input_gateonly"].detach()
            return res

        def _gateonly_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            res = {
                "expert_weights": stats["expert_weights_gateonly"],
                "token_entropy":  stats["token_entropy_gateonly"],
            }
            if "gate_input_gateonly" in stats:
                res["gate_input_gateonly"] = stats["gate_input_gateonly"]
            return res

        if self.current_phase >= 2:
            rs_s2  = _gateonly_stats(rs["s2"])
            rs_s3a = _gateonly_stats(rs["s3a"])
            rs_s3b = _gateonly_stats(rs["s3b"])
        else:
            rs_s2  = _detach_stats(rs["s2"])
            rs_s3a = _detach_stats(rs["s3a"])
            rs_s3b = _detach_stats(rs["s3b"])

        return self.pad_head({
            "stage3_feat":       backbone_out["stage3_feat"],
            "gabor_feat":        backbone_out["gabor_feat"].detach(),
            "stage1_feat":       backbone_out["stage1_feat"].detach(),
            "stage2_feat":       backbone_out["stage2_feat"].detach(),
            "routing_stats_s2":  rs_s2,
            "routing_stats_s3a": rs_s3a,
            "routing_stats_s3b": rs_s3b,
        })

    def _arcface_loss(
        self,
        mrl_embeddings: Dict[int, torch.Tensor],
        identity_labels: torch.Tensor,
    ) -> torch.Tensor:
        """MRL-ArcFace: average loss across {64, 128, 256} dims.

        Per-dim values are logged so it is visible which prefix is the
        weakest (typically the smallest dim with many classes).
        """
        per_dim = {
            dim: self.arcface_losses[str(dim)](mrl_embeddings[dim], identity_labels)
            for dim in self.MRL_DIMS
        }
        if self.training:
            for dim, loss in per_dim.items():
                self.log(
                    f"train/id_arcface_d{dim}",
                    loss.detach(),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
        return sum(per_dim.values()) / len(per_dim)

    def _identity_loss(
        self,
        mrl_embeddings: Dict[int, torch.Tensor],
        identity_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Hybrid identity loss: SupCon (open-set) + ArcFace (closed-set).

        SupCon drives open-set generalization (important for FVC where test
        IDs are disjoint from train IDs). ArcFace adds class-discriminative
        structure without dominating.

        Returns a dict so each component can be logged separately.
        """
        valid_mask = self._valid_identity_mask(identity_labels)
        if not valid_mask.any():
            anchor = sum(emb.sum() for emb in mrl_embeddings.values()) * 0.0
            zero = anchor + next(iter(self.arcface_losses.values())).weight.sum() * 0.0
            return {"arcface": zero, "supcon": zero, "total": zero}

        if not bool(valid_mask.all()):
            mrl_embeddings = {
                dim: emb[valid_mask]
                for dim, emb in mrl_embeddings.items()
            }
            identity_labels = identity_labels[valid_mask]

        l_arcface = self._arcface_loss(mrl_embeddings, identity_labels)
        # SupCon is computed on the full 256-D embedding — the MRL prefixes
        # inherit the same geometry via truncation + re-normalization.
        l_supcon = self.supcon_identity_loss(
            mrl_embeddings[self.MRL_DIMS[-1]], identity_labels,
        )
        total = (
            self.identity_arcface_weight * l_arcface
            + self.identity_supcon_weight * l_supcon
        )
        return {"arcface": l_arcface, "supcon": l_supcon, "total": total}

    def _valid_identity_mask(self, identity_labels: torch.Tensor) -> torch.Tensor:
        """Return True for labels that belong to the global identity classifier."""
        num_classes = next(iter(self.arcface_losses.values())).num_classes
        labels = identity_labels.long()
        return (
            (labels != self.identity_ignore_index)
            & (labels >= 0)
            & (labels < num_classes)
        )

    def _pad_classification_loss(
        self,
        pad_logit: torch.Tensor,
        liveness_labels: torch.Tensor,
        sensor_labels: Optional[torch.Tensor] = None,
        material_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = pad_logit.squeeze(-1)
        targets = liveness_labels.float()
        sample_weight = self._pad_sample_weights(
            logits,
            liveness_labels,
            sensor_labels=sensor_labels,
            material_labels=material_labels,
        )
        l_focal = self.pad_focal_loss(logits, targets, sample_weight=sample_weight)
        bce_per_sample = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        l_bce = (bce_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
        total = self.pad_focal_weight * l_focal + self.pad_bce_weight * l_bce
        return {
            "focal": l_focal,
            "bce": l_bce,
            "total": total,
            "sample_weight_mean": sample_weight.mean().detach(),
            "sample_weight_max": sample_weight.max().detach(),
        }

    def _pad_sample_weights(
        self,
        logits: torch.Tensor,
        liveness_labels: torch.Tensor,
        sensor_labels: Optional[torch.Tensor] = None,
        material_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        targets = liveness_labels.to(logits.device).float().reshape(-1)
        weights = torch.ones_like(targets, dtype=logits.dtype)
        spoof_mask = targets == 0

        if (
            self.pad_hard_spoof_weight > 1.0
            and sensor_labels is not None
            and material_labels is not None
            and self.pad_hard_spoof_sensor_ids
            and self.pad_hard_spoof_material_ids
        ):
            sensor = sensor_labels.to(logits.device).long().reshape(-1)
            material = material_labels.to(logits.device).long().reshape(-1)
            sensor_ids = torch.tensor(
                sorted(self.pad_hard_spoof_sensor_ids),
                device=logits.device,
                dtype=torch.long,
            )
            material_ids = torch.tensor(
                sorted(self.pad_hard_spoof_material_ids),
                device=logits.device,
                dtype=torch.long,
            )
            hard_group = (
                spoof_mask
                & torch.isin(sensor, sensor_ids)
                & torch.isin(material, material_ids)
            )
            weights = torch.where(
                hard_group,
                weights * self.pad_hard_spoof_weight,
                weights,
            )

        if self.pad_ohem_fraction > 0.0 and self.pad_ohem_weight > 1.0 and spoof_mask.any():
            with torch.no_grad():
                losses = F.binary_cross_entropy_with_logits(
                    logits.detach().reshape(-1),
                    targets,
                    reduction="none",
                )
                spoof_indices = spoof_mask.nonzero(as_tuple=False).flatten()
                k = max(1, int(round(float(spoof_indices.numel()) * self.pad_ohem_fraction)))
                k = min(k, int(spoof_indices.numel()))
                hard_rel = losses[spoof_indices].topk(k).indices
                hard_indices = spoof_indices[hard_rel]
            weights = weights.clone()
            weights[hard_indices] = weights[hard_indices] * self.pad_ohem_weight

        return weights

    def _phase_aware_bridge_loss(
        self,
        pad_mrl_embeddings: Dict[int, torch.Tensor],
        identity_mrl_embeddings: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for dim in self.bridge_mrl_dims:
            pad_z = pad_mrl_embeddings[dim]
            id_z = identity_mrl_embeddings[dim]
            if self.current_phase == 2:
                id_z = id_z.detach()
            losses.append(1.0 - (pad_z * id_z).sum(dim=-1).mean())
        if not losses:
            anchor_dim = self.bridge_mrl_dims[0] if self.bridge_mrl_dims else 64
            return pad_mrl_embeddings[anchor_dim].new_zeros(())
        return torch.stack(losses).mean()

    # -------------------------------------------------------------------------
    # Phase-specific training steps
    # -------------------------------------------------------------------------

    def _phase1_step(self, batch: Any) -> torch.Tensor:
        """Phase 1 — identity only."""
        images, identity_labels = self._unpack_identity_batch(batch)

        backbone_out = self._run_backbone(images)
        id_out       = self._run_identity(backbone_out)

        id_parts  = self._identity_loss(id_out["mrl_embeddings"], identity_labels)
        l_balance = sum(backbone_out["balance_losses"])
        loss      = id_parts["total"] + self.gamma * l_balance

        self.log("train/identity_loss",  id_parts["total"], prog_bar=True, sync_dist=True)
        self.log("train/id_arcface",     id_parts["arcface"],                sync_dist=True)
        self.log("train/id_supcon",      id_parts["supcon"],                 sync_dist=True)
        self.log("train/balance_loss",   l_balance,                          sync_dist=True)
        self.log("train/total_loss",     loss,              prog_bar=True, sync_dist=True)
        return loss

    def _pad_features_from_images(self, images: torch.Tensor) -> torch.Tensor:
        """Legacy helper for MixUp; now runs through the shared PAD path."""
        backbone_out = self._run_backbone(images, route_mode="pad")
        pad_out = self._run_pad(backbone_out)
        return pad_out["pad_features"]

    def _log_pad_separation(
        self,
        pad_logit: torch.Tensor,
        pad_target: torch.Tensor,
        valid_pad_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Log train-time PAD separation stats to detect collapse early."""
        with torch.no_grad():
            prob = torch.sigmoid(pad_logit.float()).reshape(-1)
            y = pad_target.float().reshape(-1)

            if valid_pad_mask is None:
                mask = torch.ones_like(y, dtype=torch.bool)
            else:
                mask = valid_pad_mask.bool().reshape(-1)

            mask = mask & torch.isfinite(prob) & torch.isfinite(y)
            if not mask.any():
                return

            prob_m = prob[mask]
            y_m = y[mask]
            pos_mask = y_m == 1
            neg_mask = y_m == 0
            pred = (prob_m > 0.5).float()

            self.log("train/pad_valid", mask.sum().float(), on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/pad_pos_rate", y_m.mean(), on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/pad_prob_mean", prob_m.mean(), on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/pad_prob_std", prob_m.std(unbiased=False), on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/pad_acc", (pred == y_m).float().mean(), on_step=True, on_epoch=False, sync_dist=True)

            prob_y0 = None
            if neg_mask.any():
                prob_y0 = prob_m[neg_mask].mean()
                self.log("train/pad_prob_y0", prob_y0, on_step=True, on_epoch=False, sync_dist=True)

            prob_y1 = None
            if pos_mask.any():
                prob_y1 = prob_m[pos_mask].mean()
                self.log("train/pad_prob_y1", prob_y1, on_step=True, on_epoch=False, sync_dist=True)

            if prob_y0 is not None and prob_y1 is not None:
                self.log(
                    "train/pad_prob_gap",
                    prob_y1 - prob_y0,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )

            debug_every = int(self.cfg.get("pad_debug_print_every_n_steps", 50))
            if debug_every > 0 and (int(self.global_step) % debug_every == 0):
                prob_y0_str = "nan" if prob_y0 is None else f"{prob_y0.item():.4f}"
                prob_y1_str = "nan" if prob_y1 is None else f"{prob_y1.item():.4f}"
                self.print(
                    "[pad-debug] "
                    f"step={int(self.global_step)} "
                    f"valid={int(mask.sum().item())} "
                    f"pos_rate={y_m.mean().item():.4f} "
                    f"prob_mean={prob_m.mean().item():.4f} "
                    f"prob_std={prob_m.std(unbiased=False).item():.4f} "
                    f"prob_y0={prob_y0_str} "
                    f"prob_y1={prob_y1_str} "
                    f"acc={(pred == y_m).float().mean().item():.4f}"
                )

    def _phase2_identity_step(self, batch: Any) -> torch.Tensor:
        """Phase 2 — identity-only batch. Full gradients to backbone + identity head."""
        images, identity_labels = self._unpack_identity_batch(batch)

        id_backbone = self._run_backbone(images, route_mode="identity")
        id_out      = self._run_identity(id_backbone)

        id_parts  = self._identity_loss(id_out["mrl_embeddings"], identity_labels)
        l_balance = sum(id_backbone["balance_losses"])
        loss      = id_parts["total"] + self.gamma * l_balance

        self.log("train/identity_loss", id_parts["total"], prog_bar=True, sync_dist=True)
        self.log("train/id_arcface",    id_parts["arcface"],                sync_dist=True)
        self.log("train/id_supcon",     id_parts["supcon"],                 sync_dist=True)
        self.log("train/balance_loss",  l_balance,         sync_dist=True)
        self.log("train/total_loss",    loss,              prog_bar=True, sync_dist=True)
        return loss

    def _phase2_pad_step(self, batch: Any) -> torch.Tensor:
        """
        Phase 2 legacy PAD-only step — invoked only when CombinedLoader
        yields a PAD-only dict (e.g. identity loader exhausted). The
        primary Phase 2 path is _phase2_joint_step.
        """
        images, liveness_labels = self._unpack_pad_batch(batch)
        sensor_labels = self._unpack_sensor_labels(batch)
        material_labels = self._unpack_material_labels(batch)

        pad_backbone = self._run_backbone(
            images,
            backbone_no_grad=True,
            route_mode="pad",
            detach_backbone_outputs=True,
        )
        pad_out      = self._run_pad(pad_backbone)

        # Bridge loss requires an identity-route forward of the same images.
        # In this PAD-only fallback step we drop bridge to save a full ViT
        # forward — identity batches still drive the bridge alignment in
        # `_phase2_identity_step`, where both routes are already needed.
        pad_parts = self._pad_classification_loss(
            pad_out["pad_logit"],
            liveness_labels,
            sensor_labels=sensor_labels,
            material_labels=material_labels,
        )
        self._log_pad_separation(
            pad_out["pad_logit"].squeeze(-1),
            liveness_labels,
        )

        # MixUp disabled: destroys fingerprint micro-texture and caused NaN in smoke test.
        l_mixup = images.new_zeros(())

        if sensor_labels is not None and self.lam_adv > 0:
            sensor_logits = self.sensor_adv_head(
                pad_out["pad_features"], lam=self.lam_adv,
            )
            l_sensor = self.sensor_adv_loss(sensor_logits, sensor_labels)
        else:
            l_sensor = images.new_zeros(())
        l_bridge  = images.new_zeros(())
        l_balance = sum(pad_backbone["balance_losses"])

        loss = (self.alpha * (pad_parts["total"] + l_mixup)
                + self.alpha_adv * l_sensor
                + self.beta * l_bridge
                + self.gamma * l_balance)

        self.log("train/pad_focal_loss", pad_parts["focal"], sync_dist=True)
        self.log("train/pad_bce_loss",   pad_parts["bce"],   sync_dist=True)
        self.log("train/pad_cls_loss",   pad_parts["total"], sync_dist=True)
        self.log("train/pad_weight_mean", pad_parts["sample_weight_mean"], sync_dist=True)
        self.log("train/pad_weight_max",  pad_parts["sample_weight_max"],  sync_dist=True)
        self.log("train/pad_mixup_loss", l_mixup,   sync_dist=True)
        self.log("train/pad_sensor_adv", l_sensor,  sync_dist=True)
        self.log("train/bridge_loss",    l_bridge,  sync_dist=True)
        self.log("train/balance_loss",   l_balance, sync_dist=True)
        self.log("train/total_loss",     loss, prog_bar=True, sync_dist=True)
        return loss

    def _phase2_joint_step(
        self,
        id_batch: Any,
        pad_batch: Any,
    ) -> torch.Tensor:
        """Phase 2 unified step. Processes one identity and one PAD batch
        in the same optimizer step so the loss landscape is stable for
        Adam's second-moment tracking (TASK_04).

        Forward structure:
          1. Identity branch on id_batch — full grad to backbone + id head.
          2. PAD branch on pad_batch — shared trunk read-only in Phase 2;
             PAD adapts through the PAD router + shared latent head.
          3. Bridge loss aligns PAD-route and ID-route shared latents.
          4. Optional sensor-adversarial via GRL on pad_features.
        """
        # --- Identity branch ---
        id_images, id_labels = self._unpack_identity_batch(id_batch)
        id_backbone = self._run_backbone(id_images)
        id_out      = self._run_identity(id_backbone)
        id_parts    = self._identity_loss(id_out["mrl_embeddings"], id_labels)
        l_balance_id = sum(id_backbone["balance_losses"])

        # --- PAD branch ---
        pad_images, liveness_labels = self._unpack_pad_batch(pad_batch)
        sensor_labels = self._unpack_sensor_labels(pad_batch)
        material_labels = self._unpack_material_labels(pad_batch)
        # PAD branch gets its own router in the shared MoE backbone.
        # `detach_backbone_outputs=True` keeps only the PAD-router path
        # live while stage maps / raw routing tensors stay read-only.
        pad_backbone = self._run_backbone(
            pad_images,
            backbone_no_grad=True,
            route_mode="pad",
            detach_backbone_outputs=True,
        )
        pad_out      = self._run_pad(pad_backbone)
        # NOTE: a third backbone forward (identity-route on PAD images) used
        # to be run here only to feed bridge_loss. We drop it to halve the
        # Phase-2 PAD-side trunk cost — bridge alignment is still driven by
        # `_phase2_identity_step`, where both routes are needed anyway.
        pad_parts = self._pad_classification_loss(
            pad_out["pad_logit"],
            liveness_labels,
            sensor_labels=sensor_labels,
            material_labels=material_labels,
        )
        self._log_pad_separation(
            pad_out["pad_logit"].squeeze(-1),
            liveness_labels,
        )
        
        # MixUp costs an extra shared PAD forward. At
        # the Phase-2 boundary, alpha starts at ~0.01 (cosine soft-start),
        # so the contribution alpha * l_mixup is negligible while the
        # memory cost isn't. Skip under a small threshold.
        # MixUp disabled: destroys fingerprint micro-texture and caused NaN in smoke test.
        l_mixup = pad_images.new_zeros(())
        l_balance_pad = sum(pad_backbone["balance_losses"])

        if sensor_labels is not None and self.lam_adv > 0:
            sensor_logits = self.sensor_adv_head(
                pad_out["pad_features"], lam=self.lam_adv,
            )
            l_sensor = self.sensor_adv_loss(sensor_logits, sensor_labels)
        else:
            l_sensor = pad_images.new_zeros(())

        # Bridge alignment is driven by `_phase2_identity_step` (which already
        # has both routes in scope). On the joint step we only have the
        # PAD-route forward of pad_images, so an extra identity-route forward
        # would just be a memory tax. Zero out bridge here.
        l_bridge = pad_images.new_zeros(())

        l_balance = 0.5 * (l_balance_id + l_balance_pad)

        # Orthogonality: cross-correlate PAD ↔ identity embedding subspaces.
        # Id and PAD batches are different images so trim to the smaller B.
        _B_orth = min(id_out["identity_embedding"].shape[0], pad_out["pad_embedding"].shape[0])
        l_orth = self.orth_loss(
            pad_out["pad_embedding"][:_B_orth],
            id_out["identity_embedding"][:_B_orth],
        )

        loss = (id_parts["total"]
                + self.alpha * (pad_parts["total"] + l_mixup)
                + self.alpha_adv * l_sensor
                + self.beta * l_bridge
                + self.beta_orth * l_orth
                + self.gamma * l_balance)

        self.log("train/identity_loss",  id_parts["total"], prog_bar=True, sync_dist=True)
        self.log("train/id_arcface",     id_parts["arcface"],                sync_dist=True)
        self.log("train/id_supcon",      id_parts["supcon"],                 sync_dist=True)
        self.log("train/pad_focal_loss", pad_parts["focal"],                  sync_dist=True)
        self.log("train/pad_bce_loss",   pad_parts["bce"],                    sync_dist=True)
        self.log("train/pad_cls_loss",   pad_parts["total"],                  sync_dist=True)
        self.log("train/pad_weight_mean", pad_parts["sample_weight_mean"],     sync_dist=True)
        self.log("train/pad_weight_max",  pad_parts["sample_weight_max"],      sync_dist=True)
        self.log("train/pad_mixup_loss", l_mixup,                            sync_dist=True)
        self.log("train/pad_sensor_adv", l_sensor,                           sync_dist=True)
        self.log("train/bridge_loss",    l_bridge,                           sync_dist=True)
        self.log("train/orth_loss",      l_orth,                             sync_dist=True)
        self.log("train/balance_loss",   l_balance,                          sync_dist=True)
        self.log("train/total_loss",     loss,              prog_bar=True, sync_dist=True)
        self.log("train/alpha",          self.alpha,                         sync_dist=True)
        self.log("train/beta",           self.beta,                          sync_dist=True)
        self.log("train/beta_orth",      self.beta_orth,                     sync_dist=True)
        self.log("train/alpha_adv",      self.alpha_adv,                     sync_dist=True)
        self.log("train/lam_adv",        self.lam_adv,                       sync_dist=True)
        return loss

    def _phase3_step(self, batch: Any) -> torch.Tensor:
        """Phase 3 — single sub-batch loss for joint refinement.

        Only forwards the backbone routes that the sub-batch actually needs:
          * identity-only sub-batch -> identity route only
          * PAD-only sub-batch      -> PAD route only
          * true joint sub-batch (valid identity + liveness) -> both routes + bridge loss
          * LivDet pseudo-joint sub-batch -> PAD route only (`identity_loss_ignore`)

        Earlier this function ran *both* routes on every sub-batch, which (a)
        wasted a forward pass and (b) let the bridge loss reshape the identity
        backbone on PAD-only LivDet batches. The Phase-3 collapse from
        cascaded_IM=0.35 (epoch 56-58) down to 0.07 (epoch 80+) tracks back
        to that leakage combined with batch_idx alternation.

        This step computes the per-sub-batch loss only; the caller in
        `training_step` sums losses over all sub-batches in a single
        optimizer step (TASK_04 concat scheme), so Adam's second-moment
        sees a single stable loss landscape per step.
        """
        is_dict = isinstance(batch, dict)
        has_id  = is_dict and "identity_labels" in batch
        has_liv = is_dict and "liveness_labels" in batch

        images = batch["images"] if is_dict else batch[0]

        # Multi-view identity sub-batch arrives as (B, V, C, H, W).
        # Flatten before the Gabor stem (which expects (B, 1, H, W)).
        # PAD/joint sub-batches are already 4-D and pass through untouched.
        v_repeat = 1
        if images.ndim == 5:
            B, V, C, H, W = images.shape
            images = images.reshape(B * V, C, H, W)
            v_repeat = V

        def _repeat_if_multiview(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is not None and v_repeat > 1:
                return tensor.repeat_interleave(v_repeat)
            return tensor

        id_labels = None
        valid_id_mask = None
        has_valid_id = False
        if has_id:
            id_labels = _repeat_if_multiview(batch["identity_labels"])
            ignore_mask = _repeat_if_multiview(batch.get("identity_loss_ignore"))
            if ignore_mask is not None:
                id_labels = id_labels.clone()
                ignore_mask = ignore_mask.to(device=id_labels.device, dtype=torch.bool)
                id_labels[ignore_mask] = self.identity_ignore_index
            valid_id_mask = self._valid_identity_mask(id_labels)
            has_valid_id = bool(valid_id_mask.any().item())
            self.log(
                "train/id_ignored_samples",
                (~valid_id_mask).sum().float(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

        liveness = _repeat_if_multiview(batch["liveness_labels"]) if has_liv else None

        loss: torch.Tensor = images.new_zeros(())
        l_balance_total = images.new_zeros(())
        n_routes = 0

        id_out = None
        pad_out = None

        if has_valid_id:
            id_images = images[valid_id_mask]
            id_backbone = self._run_backbone(id_images, route_mode="identity")
            id_out = self._run_identity(id_backbone)
            l_balance_total = l_balance_total + sum(id_backbone["balance_losses"])
            n_routes += 1

        if has_liv:
            # Gradient isolation for Phase 3:
            # - No LoRA: detach backbone unless it's a true joint batch (both
            #   identity + liveness labels), keeping the Phase-2 guarantee that
            #   PAD loss cannot reshape frozen identity backbone weights.
            # - LoRA enabled: PAD adapter (task="pad") can safely receive grad
            #   even on PAD-only batches because the base weights stay frozen —
            #   only the PAD LoRA adapter updates. This is the Phase-3 "LoRA
            #   unlock" — adapters converge toward their per-task subspaces.
            use_lora = getattr(self.backbone, 'use_lora', False)
            pad_full_grad = bool(has_valid_id) or use_lora
            if pad_full_grad:
                pad_backbone = self._run_backbone(images, route_mode="pad")
            else:
                pad_backbone = self._run_backbone(
                    images,
                    backbone_no_grad=True,
                    route_mode="pad",
                    detach_backbone_outputs=True,
                )
            pad_out = self._run_pad(pad_backbone)
            l_balance_total = l_balance_total + sum(pad_backbone["balance_losses"])
            n_routes += 1

        # Identity loss: ignore LivDet pseudo-identities (-1) and preserve the
        # existing spoof mask for true joint batches.
        if has_valid_id and id_out is not None and id_labels is not None:
            id_loss_labels = id_labels[valid_id_mask]
            id_loss_embs = id_out["mrl_embeddings"]
            if has_liv and liveness is not None:
                live_mask = liveness[valid_id_mask] == 1
                if live_mask.any():
                    id_loss_embs = {
                        dim: emb[live_mask]
                        for dim, emb in id_loss_embs.items()
                    }
                    id_loss_labels = id_loss_labels[live_mask]
                else:
                    id_loss_labels = id_loss_labels[:0]
                    id_loss_embs = {
                        dim: emb[:0]
                        for dim, emb in id_loss_embs.items()
                    }

            if id_loss_labels.numel() > 0:
                id_parts = self._identity_loss(id_loss_embs, id_loss_labels)
                loss = loss + id_parts["total"]
                self.log("train/identity_loss", id_parts["total"], prog_bar=True, sync_dist=True)
                self.log("train/id_arcface",   id_parts["arcface"], sync_dist=True)
                self.log("train/id_supcon",    id_parts["supcon"],  sync_dist=True)

        # PAD loss — focal + BCE on shared representation
        if has_liv and pad_out is not None:
            sensor_labels = _repeat_if_multiview(self._unpack_sensor_labels(batch))
            material_labels = _repeat_if_multiview(self._unpack_material_labels(batch))
            pad_parts = self._pad_classification_loss(
                pad_out["pad_logit"],
                liveness,
                sensor_labels=sensor_labels,
                material_labels=material_labels,
            )
            self._log_pad_separation(pad_out["pad_logit"].squeeze(-1), liveness)

            # MixUp disabled: destroys fingerprint micro-texture and caused NaN in smoke test.
            l_mixup = images.new_zeros(())

            loss = loss + self.alpha * (pad_parts["total"] + l_mixup)
            self.log("train/pad_focal_loss", pad_parts["focal"], sync_dist=True)
            self.log("train/pad_bce_loss",   pad_parts["bce"],   sync_dist=True)
            self.log("train/pad_cls_loss",   pad_parts["total"], sync_dist=True)
            self.log("train/pad_weight_mean", pad_parts["sample_weight_mean"], sync_dist=True)
            self.log("train/pad_weight_max",  pad_parts["sample_weight_max"],  sync_dist=True)
            self.log("train/pad_mixup_loss",  l_mixup, sync_dist=True)

            if sensor_labels is not None and self.lam_adv > 0:
                sensor_logits = self.sensor_adv_head(
                    pad_out["pad_features"], lam=self.lam_adv,
                )
                l_sensor = self.sensor_adv_loss(sensor_logits, sensor_labels)
                loss = loss + self.alpha_adv * l_sensor
                self.log("train/pad_sensor_adv", l_sensor, sync_dist=True)

        # Bridge alignment is meaningful only when both routes saw the same
        # images (i.e. a joint slice with both label types). Computing it on
        # PAD-only or ID-only sub-batches reshapes the identity backbone on
        # spoof signals and was a key contributor to the Phase-3 collapse.
        if has_valid_id and has_liv and id_out is not None and pad_out is not None:
            # Orthogonality: same-image joint samples have both embeddings —
            # slice PAD embeddings to the valid-identity subset to match sizes.
            l_orth = self.orth_loss(
                pad_out["pad_embedding"][valid_id_mask],
                id_out["identity_embedding"],
            )
            loss = loss + self.beta_orth * l_orth
            self.log("train/orth_loss", l_orth, sync_dist=True)

        # Balance: average across the routes that actually ran this step,
        # matching `_phase2_joint_step` (which halves when both routes run).
        if n_routes > 1:
            l_balance_eff = l_balance_total / float(n_routes)
        else:
            l_balance_eff = l_balance_total
        loss = loss + self.gamma * l_balance_eff
        self.log("train/balance_loss", l_balance_eff, sync_dist=True)
        return loss

    # -------------------------------------------------------------------------
    # LightningModule interface
    # -------------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.current_phase == 1:
            loss = self._phase1_step(batch)

        elif self.current_phase == 2:
            # TASK_04: concatenate identity + PAD into one optimizer step so
            # Adam's second-moment tracks a stable loss landscape (was:
            # batch_idx % 2 alternation, which caused total_loss spikes).
            if isinstance(batch, dict) and "identity" in batch and "pad" in batch:
                loss = self._phase2_joint_step(batch["identity"], batch["pad"])
            elif isinstance(batch, dict) and "liveness_labels" in batch:
                loss = self._phase2_pad_step(batch)
            else:
                loss = self._phase2_identity_step(batch)

        else:  # phase 3
            # TASK_04 (Phase 3): concatenate identity + PAD (+ joint) sub-batches
            # into ONE optimizer step. Previously this branch dispatched a
            # single sub-batch per step via `batch_idx % N` alternation, which
            # is the same anti-pattern Phase 2 abandoned: total_loss alternates
            # between large (identity ~5.5) and small (PAD ~0.1) values, which
            # corrupts Adam's second-moment estimate and caused the Phase-3
            # collapse from cascaded_IM≈0.35 to ≈0.07.
            if isinstance(batch, dict) and any(
                k in batch for k in ("identity", "pad", "joint")
            ):
                loss = None
                for k in ("identity", "pad", "joint"):
                    sub = batch.get(k)
                    if sub is None:
                        continue
                    sub_loss = self._phase3_step(sub)
                    loss = sub_loss if loss is None else loss + sub_loss
                if loss is None:
                    anchor = next(p for p in self.parameters() if p.requires_grad)
                    loss = anchor.sum() * 0.0
            else:
                loss = self._phase3_step(batch)

            self.log("train/total_loss", loss, prog_bar=True, sync_dist=True)
            self.log("train/alpha",      self.alpha,     sync_dist=True)
            self.log("train/beta",       self.beta,      sync_dist=True)
            self.log("train/alpha_adv",  self.alpha_adv, sync_dist=True)
            self.log("train/lam_adv",    self.lam_adv,   sync_dist=True)

        # AdaLoRA-style orthogonality regularizer on every layer's (P, Q),
        # averaged across all registered tasks. Direct param-only gradient,
        # so works regardless of which route_mode forwarded this step.
        if self.lora_orth_weight > 0 and getattr(self.backbone, "use_lora", False):
            l_orth_lora = self.backbone.get_lora_orthogonality_loss()
            loss = loss + self.lora_orth_weight * l_orth_lora
            self.log(
                "train/lora_orth", l_orth_lora.detach(), sync_dist=True
            )

        # Non-finite loss would poison Adam's moments permanently. Replace
        # with a zero scalar tied to a live parameter so autograd has a
        # graph to traverse (Lightning calls .backward() unconditionally)
        # but every gradient ends up zero.
        if not torch.isfinite(loss):
            self.log("train/nonfinite_step", 1.0, prog_bar=True, sync_dist=True)
            anchor = next(p for p in self.parameters() if p.requires_grad)
            return anchor.sum() * 0.0
        return loss

    # -------------------------------------------------------------------------
    # Checkpoint persistence for phase-schedule state
    # -------------------------------------------------------------------------
    #
    # current_phase, ArcFace scale/margin, alpha/beta, and MoE temperature are
    # plain Python attributes mutated by PhaseSchedulerCallback. Without these
    # hooks they reset to __init__ defaults when the checkpoint is reloaded,
    # making resume-training restart at Phase 1 with s=1, m=0.

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        arc = {
            k: {"s": float(v.s), "margin": float(v.margin)}
            for k, v in self.arcface_losses.items()
        }
        moe_temps = []
        if hasattr(self.backbone, "_moe_wrappers"):
            moe_temps = [float(w.temperature) for w in self.backbone._moe_wrappers]
        checkpoint["omfr_phase_state"] = {
            "current_phase": int(self.current_phase),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "beta_orth": float(self.beta_orth),
            "alpha_adv": float(self.alpha_adv),
            "lam_adv": float(self.lam_adv),
            "arcface": arc,
            "moe_temperatures": moe_temps,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        st = checkpoint.get("omfr_phase_state")
        if not st:
            return
        self.current_phase = int(st.get("current_phase", self.current_phase))
        self.alpha     = float(st.get("alpha",     self.alpha))
        self.beta      = float(st.get("beta",      self.beta))
        self.beta_orth = float(st.get("beta_orth", self.beta_orth))
        self.alpha_adv = float(st.get("alpha_adv", self.alpha_adv))
        self.lam_adv   = float(st.get("lam_adv",   self.lam_adv))
        for k, params in st.get("arcface", {}).items():
            if k in self.arcface_losses:
                self.arcface_losses[k].set_scale(float(params["s"]))
                self.arcface_losses[k].set_margin(float(params["margin"]))
        moe_temps = st.get("moe_temperatures", [])
        if hasattr(self.backbone, "_moe_wrappers") and moe_temps:
            for w, t in zip(self.backbone._moe_wrappers, moe_temps):
                w.temperature = float(t)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        images = batch["images"] if isinstance(batch, dict) else batch[0]
        id_out = None
        pad_out = None

        if not isinstance(batch, dict) or "identity_labels" in batch:
            id_backbone = self._run_backbone(images, route_mode="identity")
            id_out = self._run_identity(id_backbone)

        # PAD logits are needed both for PAD validation and for the
        # cascaded IM metric on identity samples.
        if not isinstance(batch, dict) or any(
            k in batch for k in ("identity_labels", "liveness_labels")
        ):
            pad_backbone = self._run_backbone(images, route_mode="pad")
            pad_out = self._run_pad(pad_backbone)

        if isinstance(batch, dict):
            if "identity_labels" in batch and id_out is not None:
                self._val_id_embeddings.append(id_out["identity_embedding"].detach().cpu())
                self._val_id_labels.append(batch["identity_labels"].detach().cpu())
                # Store PAD logit for same identity samples → cascaded_IM
                if pad_out is not None:
                    self._val_id_pad_logits.append(pad_out["pad_logit"].detach().cpu())

            if "liveness_labels" in batch and pad_out is not None:
                self._val_pad_logits.append(pad_out["pad_logit"].detach().cpu())
                self._val_liveness_labels.append(batch["liveness_labels"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        # ── PAD accuracy (from PAD or joint val samples) ──
        if self._val_pad_logits and self._val_liveness_labels:
            logits = torch.cat(self._val_pad_logits,      dim=0).squeeze(-1)
            labels = torch.cat(self._val_liveness_labels, dim=0).float()
            preds  = (torch.sigmoid(logits) > 0.5).float()
            pad_acc = (preds == labels).float().mean()
            self.log("val/pad_accuracy", pad_acc, sync_dist=True, prog_bar=True)

            # Detailed PAD metrics
            live_mask  = labels == 1
            spoof_mask = labels == 0
            if live_mask.any():
                bpcer = 1.0 - (preds[live_mask] == labels[live_mask]).float().mean()
                self.log("val/bpcer", bpcer, sync_dist=True)
            if spoof_mask.any():
                apcer = 1.0 - (preds[spoof_mask] == labels[spoof_mask]).float().mean()
                self.log("val/apcer", apcer, sync_dist=True)

        # ── Identity rank-1 accuracy (256-D cosine) ──
        cascaded_im = torch.tensor(0.0)
        if self._val_id_embeddings and self._val_id_labels:
            embs   = torch.cat(self._val_id_embeddings, dim=0)
            labels = torch.cat(self._val_id_labels,     dim=0)
            embs_n = F.normalize(embs, p=2, dim=-1)

            sim = embs_n @ embs_n.T
            sim.fill_diagonal_(-float("inf"))
            rank1_acc = (labels[sim.argmax(dim=1)] == labels).float().mean()
            self.log("val/identity_rank1", rank1_acc, sync_dist=True, prog_bar=True)

            # ── Cascaded IM: PAD-filtered rank-1 on identity samples ──
            # Phase 1: PAD head not trained → use raw rank-1
            # Phase 2+: filter identity samples through PAD predictions
            cascaded_im = rank1_acc
            if self.current_phase >= 2 and self._val_id_pad_logits:
                id_pad_logits = torch.cat(self._val_id_pad_logits, dim=0).squeeze(-1)
                live_mask = torch.sigmoid(id_pad_logits) > 0.5
                if live_mask.sum() > 1:
                    live_embs   = embs_n[live_mask]
                    live_labels = labels[live_mask]
                    live_sim    = live_embs @ live_embs.T
                    live_sim.fill_diagonal_(-float("inf"))
                    cascaded_im = (live_labels[live_sim.argmax(dim=1)] == live_labels).float().mean()
                # Log PAD acceptance rate on identity samples (should be ~100% for live prints)
                accept_rate = live_mask.float().mean()
                self.log("val/pad_accept_rate", accept_rate, sync_dist=True)

        self.log("val/cascaded_IM", cascaded_im, prog_bar=True, sync_dist=True)

        self._val_id_embeddings.clear()
        self._val_id_labels.clear()
        self._val_id_pad_logits.clear()
        self._val_pad_logits.clear()
        self._val_liveness_labels.clear()

    def configure_optimizers(self):
        lr            = float(self.cfg.get("lr",           1e-4))
        weight_decay  = float(self.cfg.get("weight_decay", 0.05))
        total_epochs  = int(  self.cfg.get("total_epochs", 60))

        use_lora = bool(getattr(self.backbone, "use_lora", False))
        lora_cfg = self.cfg.get("lora", {}) or {}
        lora_lr_mult = float(lora_cfg.get("lr_mult", 5.0))
        magnitude_lr_mult = float(lora_cfg.get("magnitude_lr_mult", 1.0))

        opt_cfg = self.cfg.get("optimizer", {}) or {}
        lr_mults = opt_cfg.get("lr_multipliers", {}) or {}
        arcface_lr_mult = float(lr_mults.get("arcface", 0.1))

        # MoE params need separate group — exclude from backbone groups
        moe_ids = {id(p) for p in self.backbone.get_moe_params()}
        # LoRA params are owned by the backbone but live inside attn/MLP
        # blocks; exclude them from the generic stage groups so we can apply
        # a dedicated LR.
        lora_ids = (
            {id(p) for p in self.backbone.get_lora_params()}
            if use_lora
            else set()
        )

        def _trainable(it):
            return [p for p in it if p.requires_grad]

        # Stage 0+1 = early (PAD branch reads these)
        backbone_early = _trainable(
            p for p in self.backbone.get_stage_params([0, 1])
            if id(p) not in moe_ids and id(p) not in lora_ids
        )
        # Stage 2+3 = late (identity branch reads these)
        backbone_late = _trainable(
            p for p in self.backbone.get_stage_params([2, 3])
            if id(p) not in moe_ids and id(p) not in lora_ids
        )

        param_groups = [
            # Identity Gabor stem — low LR (only 16 params, stable)
            {
                "params": list(self.gabor.parameters()),
                "lr":     lr * 0.1,
                "name":   "gabor",
            },
            # Legacy PAD Gabor bank kept for checkpoint compatibility.
            {
                "params": list(self.gabor_pad.parameters()),
                "lr":     lr * 1.0,
                "name":   "gabor_pad",
            },
            # Patch embed — standard LR (kept trainable to absorb 8-ch input)
            {
                "params": _trainable(self.backbone.get_embed_params()),
                "lr":     lr,
                "name":   "backbone_embed",
            },
            # Backbone stages 0+1 — empty when LoRA freezes the trunk
            {
                "params": backbone_early,
                "lr":     lr,
                "name":   "backbone_early",
            },
            # Backbone stages 2+3 — empty when LoRA freezes the trunk
            {
                "params": backbone_late,
                "lr":     lr,
                "name":   "backbone_late",
            },
            # MoE experts + gates — 2x LR (newly initialized)
            {
                "params": list(self.backbone.get_moe_params()),
                "lr":     lr * 2.0,
                "name":   "moe_experts",
            },
            # Identity Head — standard LR
            {
                "params": list(self.identity_head.parameters()),
                "lr":     lr,
                "name":   "identity_head",
            },
            # Legacy PAD stem retained for checkpoint compatibility.
            {
                "params": list(self.pad_stem.parameters()),
                "lr":     lr * 2.0,
                "name":   "pad_stem",
            },
            # PAD Head — always 2x.
            {
                "params": list(self.pad_head.parameters()),
                "lr":     lr * 2.0,
                "name":   "pad_head",
            },
            # ArcFace classifiers — lr_mult from config (default 0.1).
            # Higher LRs blow up the classifier weights once scale reaches
            # ~32 with many classes; the yaml multiplier is now honored.
            {
                "params": [p for af in self.arcface_losses.values()
                           for p in af.parameters()],
                "lr":     lr * arcface_lr_mult,
                "name":   "arcface",
            },
            # Sensor adversarial head — standard LR; trained normally while
            # the GRL flips grad into pad_features.
            {
                "params": list(self.sensor_adv_head.parameters()),
                "lr":     lr,
                "name":   "sensor_adv_head",
            },
        ]

        if use_lora:
            # Identity adapter (P, E, Q) at lora_lr_mult; magnitude at base LR.
            id_lora = [p for p in self.backbone.get_lora_params(
                task="identity", include_magnitude=False
            )]
            id_mag = [p for p in self.backbone.get_lora_params(
                task="identity", include_magnitude=True
            ) if not any(p is q for q in id_lora)]
            pad_lora = [p for p in self.backbone.get_lora_params(
                task="pad", include_magnitude=False
            )]
            pad_mag = [p for p in self.backbone.get_lora_params(
                task="pad", include_magnitude=True
            ) if not any(p is q for q in pad_lora)]
            param_groups += [
                {"params": id_lora,  "lr": lr * lora_lr_mult,        "name": "lora_identity"},
                {"params": pad_lora, "lr": lr * lora_lr_mult,        "name": "lora_pad"},
                {"params": id_mag,   "lr": lr * magnitude_lr_mult,   "name": "dora_mag_identity"},
                {"params": pad_mag,  "lr": lr * magnitude_lr_mult,   "name": "dora_mag_pad"},
            ]

        # Drop empty param groups (PyTorch tolerates them but they pollute logs).
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        warmup_epochs = int(self.cfg.get("warmup_epochs", 5))
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,   # 1% of target LR at epoch 0
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
            },
        }

    # -------------------------------------------------------------------------
    # Batch unpacking helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _unpack_identity_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            images = batch["images"]
            labels = batch["identity_labels"]
        else:
            images, labels = batch[0], batch[1]

        # Multi-view (TASK_06): (B, V, C, H, W) -> (B*V, C, H, W) with
        # labels repeated V times so downstream SupCon / ArcFace see
        # each view as an independent sample of the same class.
        if images.ndim == 5:
            B, V, C, H, W = images.shape
            images = images.reshape(B * V, C, H, W)
            labels = labels.repeat_interleave(V)
        return images, labels

    @staticmethod
    def _unpack_pad_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch["images"], batch["liveness_labels"]
        return batch[0], batch[1]

    @staticmethod
    def _unpack_sensor_labels(batch: Any):
        """Returns sensor_labels tensor if available, else None."""
        if isinstance(batch, dict):
            return batch.get("sensor_labels")
        return None

    @staticmethod
    def _unpack_material_labels(batch: Any):
        """Returns material_labels tensor if available, else None."""
        if isinstance(batch, dict):
            return batch.get("material_labels")
        return None
