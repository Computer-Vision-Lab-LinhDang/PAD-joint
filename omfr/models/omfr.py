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
from omfr.models.heads.identity_head import IdentityHead
from omfr.models.heads.pad_head import PADHead
from omfr.models.losses.arcface import ArcFaceLoss
from omfr.models.losses.supcon import SupConLoss
from omfr.models.losses.orthogonal import OrthogonalityLoss
from omfr.models.losses.pad_loss import PADLoss
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
        # Separate Gabor bank for PAD — initialized at a higher base
        # frequency (~0.35 cycles/px) so its filter bank covers pore-
        # and micro-texture scales, while the identity Gabor stays on
        # ridge frequencies (~0.12 cycles/px). Each bank has its own
        # σ/γ params, so identity and PAD never interfere at the
        # preprocessing stage (each objective shapes its own bank).
        self.gabor_pad = LearnableGaborStem(init_frequency=0.35)

        # Backbone dispatch. `backbone.name` selects the implementation:
        #   tiny_vit_5m_224  -> TinyViTBackbone (5M, hierarchical, local window)
        #   fastvit_sa12     -> FastViTBackbone (~10.5M, RepMixer + self-attn @ s3)
        # Stage dims differ between backbones — must flow into IdentityHead.
        backbone_cfg = config.get("backbone", {}) or {}
        backbone_name = str(backbone_cfg.get("name", "tiny_vit_5m_224"))
        num_experts = int(backbone_cfg.get("num_experts", 4))
        top_k = int(backbone_cfg.get("top_k", 2))

        if backbone_name.startswith("fastvit"):
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
        self.pad_head = PADHead(
            stage1_dim=int(pad_cfg.get("stage1_dim", default_stage1)),
            stage2_dim=int(pad_cfg.get("stage2_dim", default_stage2)),
        )
        self.identity_head = IdentityHead(
            embed_dim=int(id_cfg.get("embed_dim", 256)),
            num_heads=int(id_cfg.get("num_heads", 8)),
            grid_size=int(id_cfg.get("grid_size", 14)),
            num_queries=int(id_cfg.get("num_queries", 4)),
            stage3_dim=int(id_cfg.get("stage3_dim", default_stage3)),
            stage4_dim=int(id_cfg.get("stage4_dim", default_stage4)),
        )

        # -- Losses --
        losses_cfg = config.get("losses", {}) or {}
        if isinstance(losses_cfg.get("value"), dict):
            losses_cfg = losses_cfg["value"]
        arcface_cfg = losses_cfg.get("arcface", {}) or {}
        label_smoothing = float(
            arcface_cfg.get(
                "label_smoothing",
                losses_cfg.get("identity_label_smoothing", 0.0),
            )
        )
        # ArcFace: one per MRL dim, weights stored HERE (not in IdentityHead)
        # Init with s=1.0, margin=0.0 — PhaseSchedulerCallback warms up to s=32, m=0.5
        self.arcface_losses = nn.ModuleDict({
            "64": ArcFaceLoss(
                64,
                num_classes=num_classes,
                s=1.0,
                margin=0.0,
                label_smoothing=label_smoothing,
            ),
            "128": ArcFaceLoss(
                128,
                num_classes=num_classes,
                s=1.0,
                margin=0.0,
                label_smoothing=label_smoothing,
            ),
            "256": ArcFaceLoss(
                256,
                num_classes=num_classes,
                s=1.0,
                margin=0.0,
                label_smoothing=label_smoothing,
            ),
        })
        # PAD branch — configurable BCE + focal + OHEM + hard-spoof weights.
        # Defaults are BCE-only so the original 24/4 config remains valid;
        # the 26/4 checkpoint config enables focal/OHEM/hard-spoof terms.
        self.pad_loss = PADLoss.from_config(losses_cfg)
        self.pad_mixup_loss = MixUpConsistency(alpha=0.4)
        # Legacy BCE / SupCon kept for back-compat (val only). Unused in
        # training paths after TASK_01.
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
        self.unknown_sensor_id = int(config.get("unknown_sensor_id", num_sensors - 1))
        self.sensor_adv_head = SensorAdversarialHead(
            in_features=_PADHeadCls.PAD_FEATURES_DIM,
            num_sensors=num_sensors,
            hidden_dim=128,
        )
        self.sensor_adv_loss = SensorAdversarialLoss()

        # -- Loss weights --
        self.alpha: float = 0.0   # PAD weight — ramped in Phase 2
        self.beta:  float = 0.0   # Orth weight — ramped in Phase 2
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
    ) -> Dict:
        """Gabor stem -> TinyViT backbone. Returns backbone output dict.

        Two Gabor responses are computed:
          * ``gabor_feat``     — identity Gabor (ridge-tuned), fed into
            the TinyViT backbone and used downstream by identity_head.
          * ``gabor_pad_feat`` — PAD Gabor (pore/micro-texture-tuned),
            consumed by pad_stem inside _run_pad. Separate banks mean
            identity grads never touch PAD σ/γ and vice-versa.

        ``backbone_no_grad=True`` wraps the expensive TinyViT forward in
        ``torch.no_grad()`` so none of its activations are retained for
        backward. The PAD-side Gabor bank stays under autograd because
        pad_stem must still learn through it. Used by Phase-2 joint step
        for the PAD branch, where stage1/2 feats are detached downstream
        and the only gradient we lose is a thin one to MoE gate_proj —
        already driven by the concurrent identity branch.
        """
        enhanced     = self.gabor(images)        # (B, 8, 224, 224)
        enhanced_pad = self.gabor_pad(images)    # (B, 8, 224, 224)
        if backbone_no_grad:
            with torch.no_grad():
                out = self.backbone(enhanced)
        else:
            out = self.backbone(enhanced)
        out["gabor_feat"]     = enhanced
        out["gabor_pad_feat"] = enhanced_pad
        return out

    def _run_identity(self, backbone_out: Dict) -> Dict:
        """Identity head using stage3 + stage4 features."""
        return self.identity_head({
            "stage3_feat": backbone_out["stage3_feat"],
            "stage4_feat": backbone_out["stage4_feat"],
        })

    def _run_pad(self, backbone_out: Dict) -> Dict:
        """PAD head reading stage1 + stage2 features + routing stats.

        ALL tensors coming out of the backbone are detached before they
        reach the PAD head. Rationale:

          * stage1/stage2 feature maps — without the detach, PAD
            BCE/SupCon back-propagates through the 5.4 M-param TinyViT
            backbone. LivDet datasets have strong sensor signatures, so
            the backbone quickly memorizes sensor -> class shortcuts.
            Observed symptoms: identity_loss jumps from ~5 to ~12 at
            the Phase 2 boundary (backbone features shift away from
            what identity head expects), val BPCER climbs to ~80% on
            held-out sensors (test prints look "alien", head defaults
            to spoof). The fix is to make the PAD head a pure read-only
            classifier on top of identity-shaped features.
          * routing_stats — same reason for the MoE gate: if PAD loss
            could shape routing, the gate would encode liveness as a
            sensor fingerprint and fail cross-split.

        Net effect: the backbone is shaped only by L_identity (Phase 1,
        2-identity, 3-joint) + balance loss. PAD head trains its own
        ~340K params from a frozen view of the backbone.
        """
        rs = backbone_out["routing_stats"]

        def _detach_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            res = {
                "expert_weights": stats["expert_weights"].detach(),
                "token_entropy":  stats["token_entropy"].detach(),
            }
            # [NEW] Đảm bảo vector tần số cũng được truyền qua an toàn
            if "gate_input_gateonly" in stats:
                res["gate_input_gateonly"] = stats["gate_input_gateonly"].detach()
            return res

        def _gateonly_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            # "Gate-only" routing copy — grad can reach gate_proj
            # weights but stops at gate_input (see moe_ffn.forward).
            res = {
                "expert_weights": stats["expert_weights_gateonly"],
                "token_entropy":  stats["token_entropy_gateonly"],
            }
            # [NEW] Bypass 3 dải tần số gốc để PAD trực tiếp phân tích bọt khí/nhiễu
            if "gate_input_gateonly" in stats:
                res["gate_input_gateonly"] = stats["gate_input_gateonly"]
            return res

        # Phase-gated routing un-detach: from Phase 2 on, let PAD
        # gradients flow back into the MoE gate projection only
        # (~200 params/layer × 3 layers ≈ 600 params). Expert bodies,
        # stage features, and the upstream tokens that feed FrequencyGate
        # stay isolated — PAD can bias routing toward spoof-texture
        # experts without reshaping the backbone.
        # Phase 1: keep everything detached — backbone is still
        # converging on identity and we don't want a noisy PAD head
        # steering routing before it has a useful signal.
        if self.current_phase >= 2:
            rs_s2  = _gateonly_stats(rs["s2"])
            rs_s3a = _gateonly_stats(rs["s3a"])
            rs_s3b = _gateonly_stats(rs["s3b"])
        else:
            rs_s2  = _detach_stats(rs["s2"])
            rs_s3a = _detach_stats(rs["s3a"])
            rs_s3b = _detach_stats(rs["s3b"])

        # Dedicated PAD stem runs on the PAD-specific Gabor bank
        # (self.gabor_pad). Because it's a separate filter bank from
        # the identity Gabor, PAD gradients never touch identity σ/γ —
        # no detach needed. The PAD bank is free to learn pore- and
        # texture-scale frequencies throughout all phases.
        pad_stem_feat = self.pad_stem(backbone_out["gabor_pad_feat"])

        return self.pad_head({
            "pad_stem_feat":     pad_stem_feat,
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
        """MRL-ArcFace: average loss across {64, 128, 256} dims."""
        total = sum(
            self.arcface_losses[str(dim)](emb, identity_labels)
            for dim, emb in mrl_embeddings.items()
        )
        return total / len(mrl_embeddings)

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

    def _pad_classification_loss(
        self,
        pad_logit: torch.Tensor,
        liveness_labels: torch.Tensor,
        sensor_labels: Optional[torch.Tensor] = None,
        material_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Configurable PAD loss used by all phase-2/3 PAD paths."""
        return self.pad_loss(
            pad_logit,
            liveness_labels,
            sensor_labels=sensor_labels,
            material_labels=material_labels,
        )

    def _sensor_adversarial_loss(
        self,
        pad_features: torch.Tensor,
        sensor_labels: torch.Tensor,
    ) -> torch.Tensor:
        sensor_logits = self.sensor_adv_head(pad_features, lam=self.lam_adv)
        sensor = sensor_labels.to(sensor_logits.device).long().reshape(-1)
        sample_weight = (sensor != self.unknown_sensor_id).to(sensor_logits.dtype)
        l_sensor = self.sensor_adv_loss(
            sensor_logits,
            sensor,
            sample_weight=sample_weight,
        )
        self.log(
            "train/pad_sensor_adv_unknown_frac",
            1.0 - sample_weight.mean().detach(),
            sync_dist=True,
        )
        return l_sensor

    def _log_pad_loss_parts(self, pad_parts: Dict[str, torch.Tensor]) -> None:
        self.log("train/pad_focal_loss", pad_parts["focal"], sync_dist=True)
        self.log("train/pad_bce_loss",   pad_parts["bce"],   sync_dist=True)
        self.log("train/pad_cls_loss",   pad_parts["total"], sync_dist=True)
        self.log(
            "train/pad_weight_mean",
            pad_parts["sample_weight_mean"],
            sync_dist=True,
        )
        self.log(
            "train/pad_weight_max",
            pad_parts["sample_weight_max"],
            sync_dist=True,
        )

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
        """Gabor_pad -> PADStem -> pad_head fusion, producing pad_features.

        Used by MixUpConsistency to get features on linearly-mixed images
        without paying for the full TinyViT backbone. We skip the backbone:
        PADStem is the only trainable pixel-path on PAD, and the routing /
        stage features are batch-dependent (can't be linearly mixed anyway),
        so feeding zeros for those lets the consistency loss target the
        pad_stem manifold.
        """
        gabor_pad = self.gabor_pad(images)
        pad_stem_feat = self.pad_stem(gabor_pad)
        B = images.shape[0]
        device = images.device
        zeros_s1 = torch.zeros(B, 64, 56, 56, device=device, dtype=pad_stem_feat.dtype)
        zeros_s2 = torch.zeros(B, 128, 28, 28, device=device, dtype=pad_stem_feat.dtype)
        zeros_rs = {
            "expert_weights": torch.zeros(B, 1, 4, device=device, dtype=pad_stem_feat.dtype),
            "token_entropy":  torch.zeros(B, 1, device=device, dtype=pad_stem_feat.dtype),
            # [NEW] Dummy tensor 3 chiều để thỏa mãn 444 chiều đầu vào của fusion_mlp
            "gate_input_gateonly": torch.zeros(B, 1, 3, device=device, dtype=pad_stem_feat.dtype),
        }
        out = self.pad_head({
            "pad_stem_feat":     pad_stem_feat,
            "stage1_feat":       zeros_s1,
            "stage2_feat":       zeros_s2,
            "routing_stats_s2":  zeros_rs,
            "routing_stats_s3a": zeros_rs,
            "routing_stats_s3b": zeros_rs,
        })
        return out["pad_features"]

    def _phase2_identity_step(self, batch: Any) -> torch.Tensor:
        """Phase 2 — identity batch. Full gradients to backbone + identity head."""
        images, identity_labels = self._unpack_identity_batch(batch)

        backbone_out = self._run_backbone(images)
        id_out       = self._run_identity(backbone_out)
        pad_out      = self._run_pad(backbone_out)

        id_parts  = self._identity_loss(id_out["mrl_embeddings"], identity_labels)
        l_orth    = self.orth_loss(pad_out["pad_embedding"], id_out["identity_embedding"])
        l_balance = sum(backbone_out["balance_losses"])
        loss      = id_parts["total"] + self.beta * l_orth + self.gamma * l_balance

        self.log("train/identity_loss", id_parts["total"], prog_bar=True, sync_dist=True)
        self.log("train/id_arcface",    id_parts["arcface"],                sync_dist=True)
        self.log("train/id_supcon",     id_parts["supcon"],                 sync_dist=True)
        self.log("train/orth_loss",     l_orth,            sync_dist=True)
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

        backbone_out = self._run_backbone(images)
        pad_out      = self._run_pad(backbone_out)

        id_out = self.identity_head({
            "stage3_feat": backbone_out["stage3_feat"].detach(),
            "stage4_feat": backbone_out["stage4_feat"].detach(),
        })

        pad_parts = self._pad_classification_loss(
            pad_out["pad_logit"],
            liveness_labels,
            sensor_labels=sensor_labels,
            material_labels=material_labels,
        )
        
        # MixUp disabled: destroys fingerprint micro-texture and caused NaN in smoke test.
        l_mixup = images.new_zeros(())
        
        if sensor_labels is not None and self.lam_adv > 0:
            l_sensor = self._sensor_adversarial_loss(
                pad_out["pad_features"],
                sensor_labels,
            )
        else:
            l_sensor = images.new_zeros(())
        l_orth    = self.orth_loss(pad_out["pad_embedding"], id_out["identity_embedding"])
        l_balance = sum(backbone_out["balance_losses"])

        loss = (self.alpha * (pad_parts["total"] + l_mixup)
                + self.alpha_adv * l_sensor
                + self.beta * l_orth
                + self.gamma * l_balance)

        self._log_pad_loss_parts(pad_parts)
        self.log("train/pad_mixup_loss", l_mixup,   sync_dist=True)
        self.log("train/pad_sensor_adv", l_sensor,  sync_dist=True)
        self.log("train/orth_loss",      l_orth,    sync_dist=True)
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
          2. PAD branch on pad_batch — grad into pad_stem + pad_head only
             (backbone stage1/2 still detached inside _run_pad).
          3. Orth loss on matched subset of the two embedding sets.
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
        # Wrap TinyViT forward for the PAD branch in no_grad: the only
        # gradient we would otherwise get here is a thin one into
        # MoE gate_proj via `expert_weights_gateonly`, which the ID
        # branch (above) is already driving. All other outputs are
        # detached inside `_run_pad` anyway. Releases ~40% of Phase-2
        # activation memory — the fix for the epoch-20 OOM jump.
        pad_backbone = self._run_backbone(pad_images, backbone_no_grad=True)
        pad_out      = self._run_pad(pad_backbone)

        pad_parts = self._pad_classification_loss(
            pad_out["pad_logit"],
            liveness_labels,
            sensor_labels=sensor_labels,
            material_labels=material_labels,
        )
        
        # MixUp costs two extra pad_stem forwards (original + mixed). At
        # the Phase-2 boundary, alpha starts at ~0.01 (cosine soft-start),
        # so the contribution alpha * l_mixup is negligible while the
        # memory cost isn't. Skip under a small threshold.
        # MixUp disabled: destroys fingerprint micro-texture and caused NaN in smoke test.
        l_mixup = pad_images.new_zeros(())
        l_balance_pad = sum(pad_backbone["balance_losses"])

        if sensor_labels is not None and self.lam_adv > 0:
            l_sensor = self._sensor_adversarial_loss(
                pad_out["pad_features"],
                sensor_labels,
            )
        else:
            l_sensor = pad_images.new_zeros(())

        # --- Orthogonality across the two branches ---
        Bmin = min(id_out["identity_embedding"].shape[0],
                   pad_out["pad_embedding"].shape[0])
        l_orth = self.orth_loss(
            pad_out["pad_embedding"][:Bmin],
            id_out["identity_embedding"][:Bmin],
        )

        l_balance = 0.5 * (l_balance_id + l_balance_pad)

        loss = (id_parts["total"]
                + self.alpha * (pad_parts["total"] + l_mixup)
                + self.alpha_adv * l_sensor
                + self.beta * l_orth
                + self.gamma * l_balance)

        self.log("train/identity_loss",  id_parts["total"], prog_bar=True, sync_dist=True)
        self.log("train/id_arcface",     id_parts["arcface"],                sync_dist=True)
        self.log("train/id_supcon",      id_parts["supcon"],                 sync_dist=True)
        self._log_pad_loss_parts(pad_parts)
        self.log("train/pad_mixup_loss", l_mixup,                            sync_dist=True)
        self.log("train/pad_sensor_adv", l_sensor,                           sync_dist=True)
        self.log("train/orth_loss",      l_orth,                             sync_dist=True)
        self.log("train/balance_loss",   l_balance,                          sync_dist=True)
        self.log("train/total_loss",     loss,              prog_bar=True, sync_dist=True)
        self.log("train/alpha",          self.alpha,                         sync_dist=True)
        self.log("train/beta",           self.beta,                          sync_dist=True)
        self.log("train/alpha_adv",      self.alpha_adv,                     sync_dist=True)
        self.log("train/lam_adv",        self.lam_adv,                       sync_dist=True)
        return loss

    def _phase3_step(self, batch: Any) -> torch.Tensor:
        """Phase 3 — joint refinement. Spoof-masked ArcFace."""
        images = batch["images"] if isinstance(batch, dict) else batch[0]

        # Multi-view identity sub-batch arrives as (B, V, C, H, W).
        # Flatten before the Gabor stem (which expects (B, 1, H, W)).
        # PAD/joint sub-batches are already 4-D and pass through
        # untouched. Labels must be replicated in lockstep so the
        # identity loss sees (B*V) samples with the correct class.
        v_repeat = 1
        if images.ndim == 5:
            B, V, C, H, W = images.shape
            images = images.reshape(B * V, C, H, W)
            v_repeat = V

        liveness_labels = None
        sensor_labels = None
        material_labels = None
        if isinstance(batch, dict) and "liveness_labels" in batch:
            liveness_labels = batch["liveness_labels"]
            sensor_labels = self._unpack_sensor_labels(batch)
            material_labels = self._unpack_material_labels(batch)
            if v_repeat > 1:
                liveness_labels = liveness_labels.repeat_interleave(v_repeat)
                if sensor_labels is not None:
                    sensor_labels = sensor_labels.repeat_interleave(v_repeat)
                if material_labels is not None:
                    material_labels = material_labels.repeat_interleave(v_repeat)

        backbone_out = self._run_backbone(images)
        id_out       = self._run_identity(backbone_out)
        pad_out      = self._run_pad(backbone_out)

        l_orth    = self.orth_loss(pad_out["pad_embedding"], id_out["identity_embedding"])
        l_balance = sum(backbone_out["balance_losses"])
        loss: torch.Tensor = self.beta * l_orth + self.gamma * l_balance

        # Identity loss — spoof-masked when liveness labels available
        if isinstance(batch, dict) and "identity_labels" in batch:
            id_labels = batch["identity_labels"]
            if v_repeat > 1:
                id_labels = id_labels.repeat_interleave(v_repeat)
            if liveness_labels is not None:
                live_mask = liveness_labels == 1
                if live_mask.any():
                    masked_embs = {
                        dim: emb[live_mask]
                        for dim, emb in id_out["mrl_embeddings"].items()
                    }
                    id_parts = self._identity_loss(masked_embs, id_labels[live_mask])
                    loss = loss + id_parts["total"]
                    self.log("train/id_arcface", id_parts["arcface"], sync_dist=True)
                    self.log("train/id_supcon",  id_parts["supcon"],  sync_dist=True)
            else:
                id_parts = self._identity_loss(id_out["mrl_embeddings"], id_labels)
                loss = loss + id_parts["total"]
                self.log("train/id_arcface", id_parts["arcface"], sync_dist=True)
                self.log("train/id_supcon",  id_parts["supcon"],  sync_dist=True)

        # PAD loss — focal + mixup (TASK_01). MixUp runs on the raw
        # images tensor; it computes features via _pad_features_from_images
        # so only pad_stem + pad_head are regularized (backbone not touched).
        if liveness_labels is not None:
            pad_parts = self._pad_classification_loss(
                pad_out["pad_logit"],
                liveness_labels,
                sensor_labels=sensor_labels,
                material_labels=material_labels,
            )
            
            # MixUp disabled: destroys fingerprint micro-texture and caused NaN in smoke test.
            l_mixup = images.new_zeros(())
            
            loss = loss + self.alpha * (pad_parts["total"] + l_mixup)
            self._log_pad_loss_parts(pad_parts)
            self.log("train/pad_mixup_loss", l_mixup, sync_dist=True)

            # Sensor adversarial if labels available (Phase 3 joint set
            # typically doesn't carry sensor labels — guard with None check).
            if sensor_labels is not None and self.lam_adv > 0:
                l_sensor = self._sensor_adversarial_loss(
                    pad_out["pad_features"],
                    sensor_labels,
                )
                loss = loss + self.alpha_adv * l_sensor
                self.log("train/pad_sensor_adv", l_sensor, sync_dist=True)

        self.log("train/orth_loss",    l_orth,    sync_dist=True)
        self.log("train/balance_loss", l_balance, sync_dist=True)
        self.log("train/total_loss",   loss, prog_bar=True, sync_dist=True)
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
            if isinstance(batch, dict) and any(
                k in batch for k in ("identity", "pad", "joint")
            ):
                key_order = [k for k in ("identity", "pad", "joint") if batch.get(k) is not None]
                key = key_order[batch_idx % len(key_order)]
                sub = batch.get(key)
                if sub is not None:
                    loss = self._phase3_step(sub)
                else:
                    for k in key_order:
                        if k in batch:
                            loss = self._phase3_step(batch[k])
                            break
            else:
                loss = self._phase3_step(batch)

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
        self.alpha = float(st.get("alpha", self.alpha))
        self.beta  = float(st.get("beta",  self.beta))
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

        backbone_out = self._run_backbone(images)
        id_out       = self._run_identity(backbone_out)
        pad_out      = self._run_pad(backbone_out)

        if isinstance(batch, dict):
            if "identity_labels" in batch:
                self._val_id_embeddings.append(id_out["identity_embedding"].detach().cpu())
                self._val_id_labels.append(batch["identity_labels"].detach().cpu())
                # Store PAD logit for same identity samples → cascaded_IM
                self._val_id_pad_logits.append(pad_out["pad_logit"].detach().cpu())

            if "liveness_labels" in batch:
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

        # MoE params need separate group — exclude from backbone groups
        moe_ids = {id(p) for p in self.backbone.get_moe_params()}

        # Stage 0+1 = early (PAD branch reads these)
        backbone_early = [
            p for p in self.backbone.get_stage_params([0, 1])
            if id(p) not in moe_ids
        ]
        # Stage 2+3 = late (identity branch reads these)
        backbone_late = [
            p for p in self.backbone.get_stage_params([2, 3])
            if id(p) not in moe_ids
        ]

        param_groups = [
            # Identity Gabor stem — low LR (only 16 params, stable)
            {
                "params": list(self.gabor.parameters()),
                "lr":     lr * 0.1,
                "name":   "gabor",
            },
            # PAD Gabor stem — starts from scratch (pore frequencies),
            # needs a higher LR than the identity bank to converge in
            # the same number of epochs. Only 16 params.
            {
                "params": list(self.gabor_pad.parameters()),
                "lr":     lr * 1.0,
                "name":   "gabor_pad",
            },
            # Patch embed — standard LR
            {
                "params": list(self.backbone.get_embed_params()),
                "lr":     lr,
                "name":   "backbone_embed",
            },
            # Backbone stages 0+1 (early, PAD-relevant) — standard LR
            {
                "params": backbone_early,
                "lr":     lr,
                "name":   "backbone_early",
            },
            # Backbone stages 2+3 (late, identity-relevant) — standard LR
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
            # Dedicated PAD stem — always 2x. The previous conditional
            # (2x only in Phase 2) was a bug: configure_optimizers runs
            # once at fit start when current_phase is still 1, so the
            # 2x multiplier never actually applied. Pinning to 2x keeps
            # the PAD extractor converging fast throughout training.
            {
                "params": list(self.pad_stem.parameters()),
                "lr":     lr * 2.0,
                "name":   "pad_stem",
            },
            # PAD Head — always 2x (same reasoning as pad_stem).
            {
                "params": list(self.pad_head.parameters()),
                "lr":     lr * 2.0,
                "name":   "pad_head",
            },
            # ArcFace classifiers — 1x LR (reduced from 10x to prevent gradient explosion)
            {
                "params": [p for af in self.arcface_losses.values()
                           for p in af.parameters()],
                "lr":     lr * 1.0,
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
