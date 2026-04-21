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

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from omfr.models.backbone.gabor_stem import LearnableGaborStem
from omfr.models.backbone.pad_stem import PADStem
from omfr.models.backbone.tiny_vit import TinyViTBackbone
from omfr.models.heads.identity_head import IdentityHead
from omfr.models.heads.pad_head import PADHead
from omfr.models.losses.arcface import ArcFaceLoss
from omfr.models.losses.supcon import SupConLoss
from omfr.models.losses.orthogonal import OrthogonalityLoss


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
        self.backbone = TinyViTBackbone(
            pretrained=pretrained,
            in_chans=8,
            num_experts=4,
            top_k=2,
            use_grad_checkpoint=grad_checkpoint,
        )
        self.pad_stem = PADStem()
        self.pad_head = PADHead()
        self.identity_head = IdentityHead()

        # -- Losses --
        # ArcFace: one per MRL dim, weights stored HERE (not in IdentityHead)
        # Init with s=1.0, margin=0.0 — PhaseSchedulerCallback warms up to s=32, m=0.5
        self.arcface_losses = nn.ModuleDict({
            "64":  ArcFaceLoss(64,  num_classes=num_classes, s=1.0, margin=0.0),
            "128": ArcFaceLoss(128, num_classes=num_classes, s=1.0, margin=0.0),
            "256": ArcFaceLoss(256, num_classes=num_classes, s=1.0, margin=0.0),
        })
        self.supcon_loss = SupConLoss(temperature=0.07)
        # Dedicated SupCon for identity embeddings — higher temperature so
        # the contrastive signal is smoother across many identities.
        self.supcon_identity_loss = SupConLoss(temperature=0.1)
        self.bce_loss    = nn.BCEWithLogitsLoss()
        self.orth_loss   = OrthogonalityLoss()

        # -- Loss weights --
        self.alpha: float = 0.0   # PAD weight — ramped in Phase 2
        self.beta:  float = 0.0   # Orth weight — ramped in Phase 2
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

    def _run_backbone(self, images: torch.Tensor) -> Dict:
        """Gabor stem -> TinyViT backbone. Returns backbone output dict.

        Two Gabor responses are computed:
          * ``gabor_feat``     — identity Gabor (ridge-tuned), fed into
            the TinyViT backbone and used downstream by identity_head.
          * ``gabor_pad_feat`` — PAD Gabor (pore/micro-texture-tuned),
            consumed by pad_stem inside _run_pad. Separate banks mean
            identity grads never touch PAD σ/γ and vice-versa.
        """
        enhanced     = self.gabor(images)        # (B, 8, 224, 224)
        enhanced_pad = self.gabor_pad(images)    # (B, 8, 224, 224)
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
            return {
                "expert_weights": stats["expert_weights"].detach(),
                "token_entropy":  stats["token_entropy"].detach(),
            }

        def _gateonly_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            # "Gate-only" routing copy — grad can reach gate_proj
            # weights but stops at gate_input (see moe_ffn.forward).
            return {
                "expert_weights": stats["expert_weights_gateonly"],
                "token_entropy":  stats["token_entropy_gateonly"],
            }

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
        Phase 2 — PAD batch.
        Gradient isolation: stage3/4 features detached so PAD loss doesn't
        disturb identity-critical layers.
        """
        images, liveness_labels = self._unpack_pad_batch(batch)

        backbone_out = self._run_backbone(images)
        pad_out      = self._run_pad(backbone_out)

        # Detach stage3/4: identity features learned in Phase 1 are preserved
        id_out = self.identity_head({
            "stage3_feat": backbone_out["stage3_feat"].detach(),
            "stage4_feat": backbone_out["stage4_feat"].detach(),
        })

        l_supcon  = self.supcon_loss(pad_out["pad_features"], liveness_labels)
        l_bce     = self.bce_loss(pad_out["pad_logit"].squeeze(-1), liveness_labels.float())
        l_orth    = self.orth_loss(pad_out["pad_embedding"], id_out["identity_embedding"])
        l_balance = sum(backbone_out["balance_losses"])

        loss = (self.alpha * (l_supcon + l_bce)
                + self.beta * l_orth
                + self.gamma * l_balance)

        self.log("train/pad_supcon_loss", l_supcon, sync_dist=True)
        self.log("train/pad_bce_loss",    l_bce,    sync_dist=True)
        self.log("train/orth_loss",       l_orth,   sync_dist=True)
        self.log("train/balance_loss",    l_balance, sync_dist=True)
        self.log("train/total_loss",      loss, prog_bar=True, sync_dist=True)
        return loss

    def _phase3_step(self, batch: Any) -> torch.Tensor:
        """Phase 3 — joint refinement. Spoof-masked ArcFace."""
        images = batch["images"] if isinstance(batch, dict) else batch[0]

        backbone_out = self._run_backbone(images)
        id_out       = self._run_identity(backbone_out)
        pad_out      = self._run_pad(backbone_out)

        l_orth    = self.orth_loss(pad_out["pad_embedding"], id_out["identity_embedding"])
        l_balance = sum(backbone_out["balance_losses"])
        loss: torch.Tensor = self.beta * l_orth + self.gamma * l_balance

        # Identity loss — spoof-masked when liveness labels available
        if isinstance(batch, dict) and "identity_labels" in batch:
            id_labels = batch["identity_labels"]
            if "liveness_labels" in batch:
                live_mask = batch["liveness_labels"] == 1
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

        # PAD loss
        if isinstance(batch, dict) and "liveness_labels" in batch:
            liveness = batch["liveness_labels"]
            l_supcon = self.supcon_loss(pad_out["pad_features"], liveness)
            l_bce    = self.bce_loss(pad_out["pad_logit"].squeeze(-1), liveness.float())
            loss = loss + self.alpha * (l_supcon + l_bce)
            self.log("train/pad_supcon_loss", l_supcon, sync_dist=True)
            self.log("train/pad_bce_loss",    l_bce,    sync_dist=True)

        self.log("train/orth_loss",    l_orth,    sync_dist=True)
        self.log("train/balance_loss", l_balance, sync_dist=True)
        self.log("train/total_loss",   loss, prog_bar=True, sync_dist=True)
        return loss

    # -------------------------------------------------------------------------
    # LightningModule interface
    # -------------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.current_phase == 1:
            return self._phase1_step(batch)

        elif self.current_phase == 2:
            if isinstance(batch, dict) and "identity" in batch and "pad" in batch:
                if batch_idx % 2 == 0:
                    return self._phase2_identity_step(batch["identity"])
                else:
                    return self._phase2_pad_step(batch["pad"])
            if isinstance(batch, dict) and "liveness_labels" in batch:
                return self._phase2_pad_step(batch)
            return self._phase2_identity_step(batch)

        else:  # phase 3
            if isinstance(batch, dict) and any(
                k in batch for k in ("identity", "pad", "joint")
            ):
                key_order = [k for k in ("identity", "pad", "joint") if batch.get(k) is not None]
                key = key_order[batch_idx % len(key_order)]
                sub = batch.get(key)
                if sub is not None:
                    return self._phase3_step(sub)
                for k in key_order:
                    if k in batch:
                        return self._phase3_step(batch[k])
            return self._phase3_step(batch)

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
            return batch["images"], batch["identity_labels"]
        return batch[0], batch[1]

    @staticmethod
    def _unpack_pad_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch["images"], batch["liveness_labels"]
        return batch[0], batch[1]
