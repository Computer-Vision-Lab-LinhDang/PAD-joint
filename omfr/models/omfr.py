"""
omfr.py — Main OMFR Lightning Module

Orchestrates:
    LearnableGaborStem → ViTTinyBackbone (MoE) → IdentityHead + PADHead

Phase-aware training:
    Phase 1 (epochs  1–20): Identity only — MRL-ArcFace + L_balance
    Phase 2 (epochs 20–40): Alternating ID/PAD — all losses, α/β ramp
    Phase 3 (epochs 40–60): Joint refinement — all losses fixed, spoof-masked ArcFace

Loss total:
    L = L_Identity + α·L_PAD + β·L_orth + γ·L_balance

Config dict keys:
    num_classes:   int    — number of training identities
    lr:            float  — base LR (default 1e-4)
    weight_decay:  float  — AdamW weight decay (default 0.05)
    total_epochs:  int    — total training epochs (default 60)
    gamma:         float  — MoE balance loss weight (default 0.01)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from omfr.models.backbone.gabor_stem import LearnableGaborStem
from omfr.models.backbone.vit_tiny import ViTTinyBackbone
from omfr.models.heads.identity_head import IdentityHead
from omfr.models.heads.pad_head import PADHead
from omfr.models.losses.arcface import ArcFaceLoss
from omfr.models.losses.supcon import SupConLoss
from omfr.models.losses.orthogonal import OrthogonalityLoss


class OMFRModule(L.LightningModule):
    """
    OMFR main Lightning module.

    Args:
        config: dict with training and model hyperparameters.
    """

    MRL_DIMS = [64, 128, 256]

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = config
        self.save_hyperparameters(config)

        num_classes: int = config["num_classes"]

        # ── Model components ──────────────────────────────────────────────
        self.gabor         = LearnableGaborStem()
        self.backbone      = ViTTinyBackbone()
        self.pad_head      = PADHead()
        self.identity_head = IdentityHead(num_classes=num_classes)

        # ── Losses ────────────────────────────────────────────────────────
        # ArcFace: one per MRL dim, initial scale=32 (gentle Phase 1)
        self.arcface_losses = nn.ModuleDict({
            "64":  ArcFaceLoss(64,  num_classes=num_classes, s=32.0, margin=0.5),
            "128": ArcFaceLoss(128, num_classes=num_classes, s=32.0, margin=0.5),
            "256": ArcFaceLoss(256, num_classes=num_classes, s=32.0, margin=0.5),
        })
        self.supcon_loss = SupConLoss(temperature=0.07)
        self.bce_loss    = nn.BCEWithLogitsLoss()
        self.orth_loss   = OrthogonalityLoss()

        # ── Loss weights ──────────────────────────────────────────────────
        self.alpha: float = 0.0   # PAD weight  — ramped in Phase 2, fixed 1.0 in Phase 3
        self.beta:  float = 0.0   # Orth weight — ramped in Phase 2, fixed 0.1 in Phase 3
        self.gamma: float = float(config.get("gamma", 0.01))   # balance (always active)

        # ── Phase state ───────────────────────────────────────────────────
        self.current_phase: int = 1

        # ── Validation accumulators (cleared each epoch) ──────────────────
        self._val_id_embeddings:   List[torch.Tensor] = []
        self._val_id_labels:       List[torch.Tensor] = []
        self._val_pad_logits:      List[torch.Tensor] = []
        self._val_liveness_labels: List[torch.Tensor] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Internal forward helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_backbone(self, images: torch.Tensor) -> Dict:
        """Gabor stem → ViT backbone. Returns backbone output dict."""
        enhanced = self.gabor(images)   # (B, 3, 224, 224)
        return self.backbone(enhanced)

    def _run_identity(self, backbone_out: Dict) -> Dict:
        return self.identity_head({
            "layer12_tokens": backbone_out["layer12_tokens"],
            "cls_token":      backbone_out["cls_token"],
        })

    def _run_pad(self, backbone_out: Dict) -> Dict:
        return self.pad_head({
            "layer3_tokens":   backbone_out["layer3_tokens"],
            "layer7_tokens":   backbone_out["layer7_tokens"],
            "routing_stats_3": backbone_out["routing_stats"][3],
            "routing_stats_7": backbone_out["routing_stats"][7],
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

    # ─────────────────────────────────────────────────────────────────────────
    # Phase-specific training steps
    # ─────────────────────────────────────────────────────────────────────────

    def _phase1_step(self, batch: Any) -> torch.Tensor:
        """
        Phase 1 — identity only.
        Gradients: Gabor + full backbone + Identity Head + ArcFace.
        Loss: (1/3)·Σ L_ArcFace_dim + γ·L_balance
        """
        images, identity_labels = self._unpack_identity_batch(batch)

        backbone_out = self._run_backbone(images)
        id_out       = self._run_identity(backbone_out)

        l_id      = self._arcface_loss(id_out["mrl_embeddings"], identity_labels)
        l_balance = sum(backbone_out["balance_losses"])
        loss      = l_id + self.gamma * l_balance

        self.log("train/identity_loss", l_id,      prog_bar=True, sync_dist=True)
        self.log("train/balance_loss",  l_balance,               sync_dist=True)
        self.log("train/total_loss",    loss,       prog_bar=True, sync_dist=True)
        return loss

    def _phase2_identity_step(self, batch: Any) -> torch.Tensor:
        """
        Phase 2 — identity batch.
        Gradients: full backbone + Identity Head (full) + PAD Head (orth only).
        Loss: L_id + β·L_orth + γ·L_balance
        """
        images, identity_labels = self._unpack_identity_batch(batch)

        backbone_out = self._run_backbone(images)
        id_out       = self._run_identity(backbone_out)
        pad_out      = self._run_pad(backbone_out)

        l_id      = self._arcface_loss(id_out["mrl_embeddings"], identity_labels)
        l_orth    = self.orth_loss(pad_out["pad_embedding"], id_out["identity_embedding"])
        l_balance = sum(backbone_out["balance_losses"])
        loss      = l_id + self.beta * l_orth + self.gamma * l_balance

        self.log("train/identity_loss", l_id,   sync_dist=True)
        self.log("train/orth_loss",     l_orth,  sync_dist=True)
        self.log("train/balance_loss",  l_balance, sync_dist=True)
        self.log("train/total_loss",    loss, prog_bar=True, sync_dist=True)
        return loss

    def _phase2_pad_step(self, batch: Any) -> torch.Tensor:
        """
        Phase 2 — PAD batch.
        Gradient isolation:
          • Layers 1-7 + PAD Head: full gradient (L_PAD + L_orth + L_balance)
          • Layers 8-12 + Identity Head: NO gradient from PAD batch
            (layer12/CLS detached — identity features from Phase 1 preserved)
        Loss: α·(L_SupCon + L_BCE) + β·L_orth + γ·L_balance
        """
        images, liveness_labels = self._unpack_pad_batch(batch)

        backbone_out = self._run_backbone(images)
        pad_out      = self._run_pad(backbone_out)

        # Detach layer12/CLS: prevents L_PAD and L_orth from reaching layers 8-12
        # and identity head. Identity features learned in Phase 1 are preserved;
        # layers 8-12 only receive gradients from identity batches.
        id_out = self.identity_head({
            "layer12_tokens": backbone_out["layer12_tokens"].detach(),
            "cls_token":      backbone_out["cls_token"].detach(),
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
        """
        Phase 3 — joint refinement.
        All modules receive full gradients.
        ArcFace masked: spoof samples excluded if liveness labels present.
        Loss: (opt) L_id + α·(L_SupCon + L_BCE) + β·L_orth + γ·L_balance
        """
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
                    loss = loss + self._arcface_loss(masked_embs, id_labels[live_mask])
            else:
                loss = loss + self._arcface_loss(id_out["mrl_embeddings"], id_labels)

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

    # ─────────────────────────────────────────────────────────────────────────
    # LightningModule interface
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.current_phase == 1:
            return self._phase1_step(batch)

        elif self.current_phase == 2:
            # CombinedLoader delivers {"identity": id_batch, "pad": pad_batch}
            # Alternate: even steps → identity, odd steps → PAD
            if isinstance(batch, dict) and "identity" in batch and "pad" in batch:
                if batch_idx % 2 == 0:
                    return self._phase2_identity_step(batch["identity"])
                else:
                    return self._phase2_pad_step(batch["pad"])
            # Single-loader fallback
            if isinstance(batch, dict) and "liveness_labels" in batch:
                return self._phase2_pad_step(batch)
            return self._phase2_identity_step(batch)

        else:  # phase 3
            # CombinedLoader delivers {"identity": ..., "pad": ..., "joint": ...}
            if isinstance(batch, dict) and any(
                k in batch for k in ("identity", "pad", "joint")
            ):
                # Round-robin: 0→identity, 1→pad, 2→joint
                key_order = ["identity", "pad", "joint"]
                key = key_order[batch_idx % 3]
                sub = batch.get(key)
                if sub is not None:
                    return self._phase3_step(sub)
                # Fallback to first available
                for k in key_order:
                    if k in batch:
                        return self._phase3_step(batch[k])
            return self._phase3_step(batch)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        images = batch["images"] if isinstance(batch, dict) else batch[0]

        backbone_out = self._run_backbone(images)
        id_out       = self._run_identity(backbone_out)
        pad_out      = self._run_pad(backbone_out)

        self._val_id_embeddings.append(id_out["identity_embedding"].detach().cpu())
        self._val_pad_logits.append(pad_out["pad_logit"].detach().cpu())

        if isinstance(batch, dict):
            if "liveness_labels" in batch:
                self._val_liveness_labels.append(batch["liveness_labels"].detach().cpu())
            if "identity_labels" in batch:
                self._val_id_labels.append(batch["identity_labels"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        # ── PAD accuracy ─────────────────────────────────────────────────
        if self._val_pad_logits and self._val_liveness_labels:
            logits = torch.cat(self._val_pad_logits,      dim=0).squeeze(-1)
            labels = torch.cat(self._val_liveness_labels, dim=0).float()
            preds  = (torch.sigmoid(logits) > 0.5).float()
            pad_acc = (preds == labels).float().mean()
            self.log("val/pad_accuracy", pad_acc, sync_dist=True, prog_bar=True)

        # ── Identity rank-1 accuracy (256-D cosine) ───────────────────────
        cascaded_im = torch.tensor(0.0)
        if self._val_id_embeddings and self._val_id_labels:
            embs   = torch.cat(self._val_id_embeddings, dim=0)  # (N, 256)
            labels = torch.cat(self._val_id_labels,     dim=0)  # (N,)
            embs_n = F.normalize(embs, p=2, dim=-1)

            sim = embs_n @ embs_n.T                              # (N, N)
            sim.fill_diagonal_(-float("inf"))
            rank1_acc = (labels[sim.argmax(dim=1)] == labels).float().mean()
            self.log("val/identity_rank1", rank1_acc, sync_dist=True)

            # Cascaded IM: rank-1 on PAD-accepted (predicted-live) samples
            if self._val_pad_logits:
                logits   = torch.cat(self._val_pad_logits, dim=0).squeeze(-1)
                live_mask = torch.sigmoid(logits) > 0.5
                if live_mask.sum() > 1:
                    live_embs   = embs_n[live_mask]
                    live_labels = labels[live_mask]
                    live_sim    = live_embs @ live_embs.T
                    live_sim.fill_diagonal_(-float("inf"))
                    live_rank1  = (live_labels[live_sim.argmax(dim=1)] == live_labels)
                    cascaded_im = live_rank1.float().mean()
                else:
                    cascaded_im = rank1_acc
            else:
                cascaded_im = rank1_acc

        self.log("val/cascaded_IM", cascaded_im, prog_bar=True, sync_dist=True)

        # Clear accumulators
        self._val_id_embeddings.clear()
        self._val_id_labels.clear()
        self._val_pad_logits.clear()
        self._val_liveness_labels.clear()

    def configure_optimizers(self):
        lr            = float(self.cfg.get("lr",           1e-4))
        weight_decay  = float(self.cfg.get("weight_decay", 0.05))
        total_epochs  = int(  self.cfg.get("total_epochs", 60))

        # Deduplicate: MoE params must not appear in backbone layer groups
        moe_ids = {id(p) for p in self.backbone.get_moe_params()}

        backbone_early = [
            p for p in self.backbone.get_layer_params(list(range(1, 7)))
            if id(p) not in moe_ids
        ]
        backbone_late = [
            p for p in self.backbone.get_layer_params(list(range(7, 13)))
            if id(p) not in moe_ids
        ]
        # Include final backbone LayerNorm (applied after all 12 blocks)
        backbone_late += list(self.backbone.norm.parameters())
        stem_embed = list(self.backbone.get_stem_and_embed_params())

        param_groups = [
            # Gabor stem — very low LR (stable learned filters)
            {
                "params": list(self.gabor.parameters()),
                "lr":     lr * 0.1,
                "name":   "gabor",
            },
            # ViT patch embed + pos embed + CLS token — standard LR
            {
                "params": stem_embed,
                "lr":     lr,
                "name":   "backbone_embed",
            },
            # ViT layers 1–6 (PAD tap: layer 3) — standard LR
            {
                "params": backbone_early,
                "lr":     lr,
                "name":   "backbone_early",
            },
            # ViT layers 7–12 (identity tap: layers 7, 10, 12) — standard LR
            {
                "params": backbone_late,
                "lr":     lr,
                "name":   "backbone_late",
            },
            # MoE experts + gates — higher LR (newly initialized)
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
            # PAD Head — 2× in Phase 2 (catch up to backbone), 1× otherwise
            {
                "params": list(self.pad_head.parameters()),
                "lr":     lr * (2.0 if self.current_phase == 2 else 1.0),
                "name":   "pad_head",
            },
            # ArcFace classifiers — 10× LR (large, sparse gradients)
            {
                "params": [p for af in self.arcface_losses.values()
                           for p in af.parameters()],
                "lr":     lr * 10.0,
                "name":   "arcface",
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Batch unpacking helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_identity_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (images, identity_labels) from various batch formats."""
        if isinstance(batch, dict):
            return batch["images"], batch["identity_labels"]
        # (images, labels) tuple
        return batch[0], batch[1]

    @staticmethod
    def _unpack_pad_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (images, liveness_labels) from various batch formats."""
        if isinstance(batch, dict):
            return batch["images"], batch["liveness_labels"]
        return batch[0], batch[1]
