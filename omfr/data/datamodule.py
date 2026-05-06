"""
datamodule.py — OMFR Lightning DataModule

Manages three datasets and switches DataLoader strategy per training phase:

    Phase 1: Identity dataset only (PKSampler — P=32 identities × K=4 samples)
    Phase 2: CombinedLoader — {"identity": ..., "pad": ...}  (interleaved steps)
    Phase 3: CombinedLoader — {"identity": ..., "pad": ..., "joint": ...}

Datasets:
    identity_ds  — FVC2004 + NIST SD302          (identity labels only)
    pad_ds       — LivDet 2015 + LivDet 2017     (liveness labels only)
    joint_ds     — MSU-FPAD v2.0, or LivDetJoint fallback for Phase 3
    val_ds       — joint or identity test split  (validation)

Config dict keys:
    identity_root:   str  — root dir for identity datasets
    pad_root:        str  — root dir for PAD datasets
    joint_root:      str  — root dir for joint dataset
    livdet_joint_enabled: bool — synthesize Phase-3 joint fallback from LivDet
    num_workers:     int  — DataLoader workers (default 8)
    pk_P:            int  — P identities per identity batch (default 32)
    pk_K:            int  — K samples per identity (default 4)
    pad_batch_size:  int  — batch size for PAD and joint loaders (default 128)
"""

from __future__ import annotations

import re
from pathlib import Path
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
        self.current_phase = 1

        self.identity_ds = None
        self.pad_ds      = None
        self.joint_ds    = None
        self.val_ds      = None
        self.val_pad_ds  = None

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
        from omfr.data.datasets.pad_dataset import PADDataset, make_pad_train_transform
        from omfr.data.datasets.joint_dataset import JointDataset
        from omfr.data.datasets.livdet_joint_dataset import LivDetJointDataset
        from omfr.data.transforms import get_transforms

        identity_root = self._cfg_get("identity_root", "data.identity_data_root", default="")
        pad_root      = self._cfg_get("pad_root", "data.pad_data_root", default="")
        joint_root    = self._cfg_get("joint_root", "data.joint_data_root", default="")
        pad_datasets  = self._cfg_get("pad_datasets", "data.pad_datasets", default=[])
        pad_roots     = self._resolve_named_roots(pad_root, pad_datasets)
        livdet_joint_enabled = bool(self._cfg_get(
            "livdet_joint_enabled", "data.livdet_joint.enabled", default=False,
        ))
        livdet_joint_root = self._cfg_get(
            "livdet_joint_root", "data.livdet_joint.root", default="",
        )
        livdet_joint_datasets = self._cfg_get(
            "livdet_joint_datasets", "data.livdet_joint.datasets", default=pad_datasets,
        )
        livdet_joint_fallback = bool(self._cfg_get(
            "livdet_joint_fallback_to_pad_roots",
            "data.livdet_joint.fallback_to_pad_roots",
            default=True,
        ))
        livdet_joint_required = bool(self._cfg_get(
            "livdet_joint_required", "data.livdet_joint.required", default=False,
        ))
        livdet_joint_roots = []
        if livdet_joint_enabled:
            if livdet_joint_root:
                livdet_joint_roots = self._resolve_named_roots(
                    livdet_joint_root, livdet_joint_datasets,
                )
                if not livdet_joint_roots and Path(livdet_joint_root).exists():
                    livdet_joint_roots = [str(Path(livdet_joint_root))]
            if not livdet_joint_roots and livdet_joint_fallback:
                livdet_joint_roots = pad_roots
        identity_root = identity_root if identity_root and Path(identity_root).exists() else ""
        joint_root = joint_root if joint_root and Path(joint_root).exists() else ""

        group_by_subject = bool(self._cfg_get(
            "group_by_subject", "data.group_by_subject", default=False,
        ))
        identity_label_mode = self._cfg_get(
            "identity_label_mode", "data.identity_label_mode", default=None,
        )
        identity_num_views = int(self._cfg_get(
            "identity_num_views", "data.identity_num_views", default=2,
        ))
        identity_preset = str(self._cfg_get(
            "identity_preset", "data.identity_preset", default="fingerprint_hard",
        ))
        image_size = int(self._cfg_get(
            "image_size", "data.image_size", default=224,
        ))

        if stage in ("fit", None):
            if identity_root:
                self.identity_ds = IdentityDataset(
                    root=identity_root, split="train",
                    group_by_subject=group_by_subject,
                    identity_label_mode=identity_label_mode,
                    num_views=identity_num_views,
                    transform=get_transforms(
                        'train', output_size=image_size, preset=identity_preset,
                    ),
                    image_size=image_size,
                )
            if pad_roots:
                pad_image_size = int(self._cfg_get(
                    "image_size", "data.image_size", default=224,
                ))
                self.pad_ds = PADDataset(
                    root=pad_roots if len(pad_roots) > 1 else pad_roots[0],
                    split="train",
                    transform=make_pad_train_transform(image_size=pad_image_size),
                    image_size=pad_image_size,
                )
            if joint_root:
                self.joint_ds = JointDataset(root=joint_root, split="train")
            elif livdet_joint_roots:
                try:
                    self.joint_ds = LivDetJointDataset(
                        root=(
                            livdet_joint_roots
                            if len(livdet_joint_roots) > 1
                            else livdet_joint_roots[0]
                        ),
                        split="train",
                        transform=make_pad_train_transform(image_size=image_size),
                        image_size=image_size,
                    )
                except (FileNotFoundError, RuntimeError):
                    if livdet_joint_required:
                        raise
                    self.joint_ds = None

        if stage in ("fit", "validate", None):
            # Prefer joint dataset for validation (has both label types)
            if joint_root:
                self.val_ds = JointDataset(root=joint_root, split="test")
            elif identity_root:
                self.val_ds = IdentityDataset(
                    root=identity_root, split="val",
                    group_by_subject=group_by_subject,
                    identity_label_mode=identity_label_mode,
                )

            # PAD validation — needed when val_ds lacks liveness labels
            if not joint_root and pad_roots:
                try:
                    self.val_pad_ds = PADDataset(
                        root=pad_roots if len(pad_roots) > 1 else pad_roots[0],
                        split="test",
                    )
                except (FileNotFoundError, RuntimeError):
                    self.val_pad_ds = None

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
        num_workers = int(self._cfg_get("num_workers", "data.num_workers", default=8))
        P           = int(self._cfg_get("pk_P", "data.pk_p", default=32))
        K           = int(self._cfg_get("pk_K", "data.pk_k", default=4))
        pad_bs      = int(self._cfg_get("pad_batch_size", "data.pad_batch_size", default=128))
        pin_memory  = bool(self._cfg_get("pin_memory", "data.pin_memory", default=True))

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
                pin_memory=pin_memory,
            )

        elif phase == 2:
            assert self.identity_ds is not None and self.pad_ds is not None, \
                "identity_ds and pad_ds required for Phase 2"
            workers_each = 0 if num_workers == 0 else max(num_workers // 2, 1)
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
                num_workers=workers_each,
                pin_memory=pin_memory,
            )
            pad_loader = DataLoader(
                self.pad_ds,
                batch_sampler=pad_sampler,
                num_workers=workers_each,
                pin_memory=pin_memory,
            )
            return {"identity": id_loader, "pad": pad_loader}

        else:  # phase 3
            workers_each = 0 if num_workers == 0 else max(num_workers // 3, 1)

            loaders = {}

            if self.identity_ds is not None:
                id_sampler = PKSampler(
                    labels=self.identity_ds.get_labels(), P=P, K=K,
                )
                loaders["identity"] = DataLoader(
                    self.identity_ds,
                    batch_sampler=id_sampler,
                    num_workers=workers_each,
                    pin_memory=pin_memory,
                )

            if self.pad_ds is not None:
                pad_sampler = BalancedPADSampler(
                    liveness_labels=self.pad_ds.get_liveness_labels(),
                    batch_size=pad_bs,
                )
                loaders["pad"] = DataLoader(
                    self.pad_ds,
                    batch_sampler=pad_sampler,
                    num_workers=workers_each,
                    pin_memory=pin_memory,
                )

            if self.joint_ds is not None:
                joint_sampler = BalancedPADSampler(
                    liveness_labels=self.joint_ds.get_liveness_labels(),
                    batch_size=pad_bs,
                )
                loaders["joint"] = DataLoader(
                    self.joint_ds,
                    batch_sampler=joint_sampler,
                    num_workers=workers_each,
                    pin_memory=pin_memory,
                )

            if not loaders:
                raise ValueError("No datasets available for Phase 3.")

            if len(loaders) == 1:
                return next(iter(loaders.values()))

            return loaders

    def val_dataloader(self):
        if self.val_ds is None and self.val_pad_ds is None:
            return None

        bs  = int(self._cfg_get("val_batch_size", "data.val_batch_size", default=64))
        nw  = int(self._cfg_get("num_workers", "data.num_workers", default=8))
        pm  = bool(self._cfg_get("pin_memory", "data.pin_memory", default=True))
        kwargs = dict(batch_size=bs, shuffle=False, num_workers=nw,
                      pin_memory=pm, drop_last=False)

        loaders = []
        if self.val_ds is not None:
            loaders.append(DataLoader(self.val_ds, **kwargs))
        if self.val_pad_ds is not None:
            loaders.append(DataLoader(self.val_pad_ds, **kwargs))

        if len(loaders) == 1:
            return loaders[0]
        return loaders

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _current_phase(self) -> int:
        """Read current training phase from the Lightning module."""
        if self.trainer is not None and self.trainer.lightning_module is not None:
            return getattr(self.trainer.lightning_module, "current_phase", 1)
        return getattr(self, "current_phase", 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset size info (useful for logging / debugging)
    # ─────────────────────────────────────────────────────────────────────────

    def dataset_sizes(self) -> Dict[str, int]:
        return {
            "identity": len(self.identity_ds)  if self.identity_ds  is not None else 0,
            "pad":      len(self.pad_ds)       if self.pad_ds       is not None else 0,
            "joint":    len(self.joint_ds)     if self.joint_ds     is not None else 0,
            "val":      len(self.val_ds)       if self.val_ds       is not None else 0,
            "val_pad":  len(self.val_pad_ds)   if self.val_pad_ds   is not None else 0,
        }

    def _cfg_get(self, *paths: str, default: Any = None) -> Any:
        for path in paths:
            value = self.cfg
            for key in path.split("."):
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if value is not None:
                return value
        return default

    def _resolve_named_roots(self, base_root: str, dataset_names: Any) -> list[str]:
        if not base_root:
            return []

        base_path = Path(base_root)
        if not dataset_names or not isinstance(dataset_names, (list, tuple)):
            return [str(base_path)]

        if not base_path.exists() or not base_path.is_dir():
            return []

        subdirs = [item for item in base_path.iterdir() if item.is_dir() and not item.name.startswith('.')]
        resolved = []
        for dataset_name in dataset_names:
            normalized = self._normalize_name(str(dataset_name))
            match = next(
                (item for item in subdirs if self._normalize_name(item.name) == normalized),
                None,
            )
            if match is not None:
                resolved.append(str(match))

        if resolved:
            return resolved

        requested = {
            self._normalize_name(str(dataset_name))
            for dataset_name in dataset_names
        }
        if self._normalize_name(base_path.name) in requested:
            return [str(base_path)]

        return []

    @staticmethod
    def _normalize_name(value: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', value.lower())
