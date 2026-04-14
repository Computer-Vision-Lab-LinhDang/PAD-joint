"""
datamodule.py — OMFR Lightning DataModule

Manages three datasets and switches DataLoader strategy per training phase:

    Phase 1: Identity dataset only (PKSampler — P=32 identities × K=4 samples)
    Phase 2: CombinedLoader — {"identity": ..., "pad": ...}  (interleaved steps)
    Phase 3: CombinedLoader — {"identity": ..., "pad": ..., "joint": ...}

Datasets:
    identity_ds  — FVC2004 + NIST SD302          (identity labels only)
    pad_ds       — LivDet 2015 + LivDet 2017     (liveness labels only)
    joint_ds     — MSU-FPAD v2.0                 (both identity + liveness)
    val_ds       — joint or identity test split  (validation)

Config dict keys:
    identity_root:   str  — root dir for identity datasets
    pad_root:        str  — root dir for PAD datasets
    joint_root:      str  — root dir for joint dataset
    num_workers:     int  — DataLoader workers (default 8)
    pk_P:            int  — P identities per identity batch (default 32)
    pk_K:            int  — K samples per identity (default 4)
    pad_batch_size:  int  — batch size for PAD and joint loaders (default 128)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L
from torch.utils.data import DataLoader

from omfr.data.samplers.pk_sampler import PKSampler
from omfr.data.samplers.balanced_pad_sampler import BalancedPADSampler


class OMFRDataModule(L.LightningDataModule):
    """
    Phase-aware DataModule for OMFR.

    Args:
        config: dict with all data configuration keys.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = config

        self.identity_ds = None
        self.pad_ds      = None
        self.joint_ds    = None
        self.val_ds      = None

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiate all datasets.

        Called by Lightning once before dataloaders are needed.
        Deferred imports keep startup fast and allow optional datasets
        (e.g., joint_root may be empty during Phase 1 testing).
        """
        from omfr.data.datasets.identity_dataset import IdentityDataset
        from omfr.data.datasets.pad_dataset import PADDataset
        from omfr.data.datasets.joint_dataset import JointDataset

        identity_root = self.cfg.get("identity_root", "")
        pad_root      = self.cfg.get("pad_root",      "")
        joint_root    = self.cfg.get("joint_root",    "")

        if stage in ("fit", None):
            if identity_root:
                self.identity_ds = IdentityDataset(root=identity_root, split="train")
            if pad_root:
                self.pad_ds = PADDataset(root=pad_root, split="train")
            if joint_root:
                self.joint_ds = JointDataset(root=joint_root, split="train")

        if stage in ("fit", "validate", None):
            # Prefer joint dataset for validation (has both label types)
            if joint_root:
                self.val_ds = JointDataset(root=joint_root, split="test")
            elif identity_root:
                self.val_ds = IdentityDataset(root=identity_root, split="test")

    # ─────────────────────────────────────────────────────────────────────────
    # DataLoaders
    # ─────────────────────────────────────────────────────────────────────────

    def train_dataloader(self) -> Any:
        """
        Returns a DataLoader (Phase 1) or a dict of DataLoaders (Phase 2/3).

        Lightning automatically wraps a dict return in CombinedLoader
        (default mode: 'min_size' — stops when shortest loader is exhausted).
        Each training_step receives a batch dict with the same keys.
        """
        phase       = self._current_phase()
        num_workers = int(self.cfg.get("num_workers",    8))
        P           = int(self.cfg.get("pk_P",           32))
        K           = int(self.cfg.get("pk_K",           4))
        pad_bs      = int(self.cfg.get("pad_batch_size", 128))

        if phase == 1:
            assert self.identity_ds is not None, \
                "identity_ds not loaded — check identity_root in config"
            sampler = PKSampler(
                labels=self.identity_ds.get_labels(),
                P=P, K=K,
            )
            return DataLoader(
                self.identity_ds,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )

        elif phase == 2:
            assert self.identity_ds is not None and self.pad_ds is not None, \
                "identity_ds and pad_ds required for Phase 2"
            id_sampler  = PKSampler(
                labels=self.identity_ds.get_labels(), P=P, K=K,
            )
            pad_sampler = BalancedPADSampler(
                liveness_labels=self.pad_ds.get_liveness_labels(),
                batch_size=pad_bs,
            )
            id_loader  = DataLoader(
                self.identity_ds,
                batch_sampler=id_sampler,
                num_workers=max(num_workers // 2, 1),
                pin_memory=True,
            )
            pad_loader = DataLoader(
                self.pad_ds,
                batch_sampler=pad_sampler,
                num_workers=max(num_workers // 2, 1),
                pin_memory=True,
            )
            return {"identity": id_loader, "pad": pad_loader}

        else:  # phase 3
            assert (
                self.identity_ds is not None
                and self.pad_ds is not None
                and self.joint_ds is not None
            ), "identity_ds, pad_ds, and joint_ds all required for Phase 3"

            id_sampler    = PKSampler(
                labels=self.identity_ds.get_labels(), P=P, K=K,
            )
            pad_sampler   = BalancedPADSampler(
                liveness_labels=self.pad_ds.get_liveness_labels(),
                batch_size=pad_bs,
            )
            joint_sampler = BalancedPADSampler(
                liveness_labels=self.joint_ds.get_liveness_labels(),
                batch_size=pad_bs,
            )
            workers_each = max(num_workers // 3, 1)
            return {
                "identity": DataLoader(
                    self.identity_ds,
                    batch_sampler=id_sampler,
                    num_workers=workers_each,
                    pin_memory=True,
                ),
                "pad": DataLoader(
                    self.pad_ds,
                    batch_sampler=pad_sampler,
                    num_workers=workers_each,
                    pin_memory=True,
                ),
                "joint": DataLoader(
                    self.joint_ds,
                    batch_sampler=joint_sampler,
                    num_workers=workers_each,
                    pin_memory=True,
                ),
            }

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=int(self.cfg.get("val_batch_size", 64)),
            shuffle=False,
            num_workers=int(self.cfg.get("num_workers", 8)),
            pin_memory=True,
            drop_last=False,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _current_phase(self) -> int:
        """Read current training phase from the Lightning module."""
        if self.trainer is not None and self.trainer.lightning_module is not None:
            return getattr(self.trainer.lightning_module, "current_phase", 1)
        return 1

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset size info (useful for logging / debugging)
    # ─────────────────────────────────────────────────────────────────────────

    def dataset_sizes(self) -> Dict[str, int]:
        return {
            "identity": len(self.identity_ds) if self.identity_ds is not None else 0,
            "pad":      len(self.pad_ds)      if self.pad_ds      is not None else 0,
            "joint":    len(self.joint_ds)    if self.joint_ds    is not None else 0,
            "val":      len(self.val_ds)      if self.val_ds      is not None else 0,
        }
