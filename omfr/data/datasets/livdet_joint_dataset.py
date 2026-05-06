"""
livdet_joint_dataset.py — Synthesize a joint (identity + liveness) dataset
from LivDet 2013/2015 by parsing the (subject, finger) pair out of filenames.

LivDet spoofs are physical casts of real fingers, so the same subject_id +
finger position appears in BOTH Live and Fake/<material>/. This gives us
sample pairs that share identity but disagree on liveness — exactly what
Phase-3 joint refinement needs to enforce orthogonality between identity
and PAD heads, and to fire `bridge_loss` (which only triggers when both
labels coexist on the same sample).

Filename patterns supported:
  LivDet2015  — every sensor:    "<subj>_<finger>_<sample>.<ext>"
  LivDet2013 CrossMatch / Swipe: "<subj>_<finger>_<sample>.<ext>" (subj 7-digit)
  LivDet2013 Italdata / Biometrika:
      "<subj>(.<n>)?T<wm>?<pos>(Itd|Bmk).<ext>" with pos in {Lidx,Lltl,Lmdl,
      Lrng,Lthb,Ridx,Rltl,Rmdl,Rrng,Rthb}.

Identity key:    f"{sensor}:{subject}:{finger}"
Liveness label:  1 (live) or 0 (spoof)
Sensor / material: indexed against PADDataset.SENSOR_NAMES / MATERIAL_NAMES.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from omfr.data.datasets.pad_dataset import PADDataset


# ─────────────────────────────────────────────────────────────────────────────
# Filename parsers
# ─────────────────────────────────────────────────────────────────────────────

# <subj>_<finger>_<sample> — covers LivDet2015 (all sensors), LivDet2013
# CrossMatch & Swipe. Finger may be plain digits ("5") or letter+digits
# ("R1", "L2"), and subject may be 3..8 digits.
_PAT_UNDERSCORE = re.compile(r"^(\d{1,8})_([LR]?\d{1,2})_\d{1,3}$")

# LivDet2013 Italdata/Biometrika — e.g. "031TamLidxItd", "011.1TwfRthbItd",
# "063TRltlBmk", "083TrrLthbItd". Suffix Itd|Bmk fixed by source dataset.
_PAT_L13_ITD_BMK = re.compile(
    r"^(\d{1,4}(?:\.\d+)?)T\w*?([LR](?:idx|ltl|mdl|rng|thb))(?:Itd|Bmk)$",
    re.IGNORECASE,
)


def _parse_subject_finger(stem: str) -> Optional[Tuple[str, str]]:
    """Return (subject_str, finger_str) parsed from filename stem, or None."""
    m = _PAT_UNDERSCORE.match(stem)
    if m:
        return m.group(1), m.group(2).upper()
    m = _PAT_L13_ITD_BMK.match(stem)
    if m:
        finger_raw = m.group(2)
        finger_norm = finger_raw[0].upper() + finger_raw[1:].lower()
        return m.group(1), finger_norm
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class LivDetJointDataset(Dataset):
    """
    Joint identity + liveness dataset synthesized from LivDet 2013/2015.

    Every sample carries:
        - identity_labels  (int in [0, num_classes))   — derived from filename
        - identity_loss_ignore (bool=True)              — pseudo-id is local
        - liveness_labels  (0/1)                        — Live vs Fake/Spoof dir
        - sensor_labels    (int)                        — same indexing as PAD
        - material_labels  (int)                        — Live=0 or material id

    Args:
        root:        single dir or list of dirs (LivDet2013 + LivDet2015 etc.)
        split:       "train" or "test"
        transform:   optional augmentation pipeline applied to (1, H, W) tensor
        image_size:  resize target (default 224)
    """

    LIVE_DIRS = PADDataset.LIVE_DIRS
    SPOOF_DIRS = PADDataset.SPOOF_DIRS
    IMG_EXTS = PADDataset.IMG_EXTS
    SPLIT_DIR_ALIASES = PADDataset.SPLIT_DIR_ALIASES
    SPLIT_FALLBACKS = PADDataset.SPLIT_FALLBACKS
    SENSOR_NAMES = PADDataset.SENSOR_NAMES
    MATERIAL_NAMES = PADDataset.MATERIAL_NAMES

    LABEL_LIVE = 1
    LABEL_SPOOF = 0
    # LivDet identities are parsed pseudo-labels local to the LivDet files.
    # They are useful for bookkeeping, but must not expand or train the
    # global ArcFace identity classifier used by the identity datasets.
    contributes_identity_classes = False

    def __init__(
        self,
        root: Union[str, Path, Sequence[Union[str, Path]]],
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if isinstance(root, (str, Path)):
            self.roots = [Path(root)]
        else:
            self.roots = [Path(r) for r in root]
        self.split = split
        self.transform = transform
        self.image_size = image_size

        # (path, identity_idx, liveness, sensor_id, material_id)
        self.samples: List[Tuple[Path, int, int, int, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self._unparsed: int = 0
        self._kept: int = 0

        self._load_samples()

    # ------------------------------------------------------------------ load

    def _load_samples(self) -> None:
        raw: List[Tuple[Path, str, int, int, int]] = []

        for root in self.roots:
            if not root.exists():
                raise FileNotFoundError(
                    f"LivDetJointDataset root does not exist: {root}"
                )
            scan_root = self._resolve_split_root(root)
            for path in self._iter_image_files(scan_root):
                liveness = self._infer_liveness(path, scan_root)
                if liveness is None:
                    continue
                parsed = _parse_subject_finger(path.stem)
                if parsed is None:
                    self._unparsed += 1
                    continue
                subject, finger = parsed
                sensor_id = self._infer_sensor_id(path, scan_root)
                material_id = self._infer_material_id(path, scan_root, liveness)
                sensor_name = self.SENSOR_NAMES[sensor_id]
                ident_key = f"{sensor_name}:{subject}:{finger}"
                raw.append((path, ident_key, liveness, sensor_id, material_id))

        if not raw:
            roots_str = ", ".join(str(r) for r in self.roots)
            raise RuntimeError(
                f"LivDetJointDataset found no parsable samples for split="
                f"{self.split!r} under: {roots_str}"
            )

        # Build class index in deterministic order
        unique_keys = sorted({k for _, k, _, _, _ in raw})
        self.class_to_idx = {k: i for i, k in enumerate(unique_keys)}

        for path, ident_key, liveness, sensor_id, material_id in raw:
            self.samples.append(
                (path, self.class_to_idx[ident_key], liveness, sensor_id, material_id)
            )
        self._kept = len(self.samples)

    # ------------------------------------------------------------- discovery

    def _iter_image_files(self, scan_root: Path):
        for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=True):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            cur = Path(dirpath)
            for fname in filenames:
                p = cur / fname
                if p.suffix.lower() in self.IMG_EXTS:
                    yield p

    def _infer_liveness(self, path: Path, scan_root: Path) -> Optional[int]:
        for ancestor in self._ancestors_until(path.parent, scan_root):
            if ancestor.name in self.LIVE_DIRS:
                return self.LABEL_LIVE
            if ancestor.name in self.SPOOF_DIRS:
                return self.LABEL_SPOOF
        return None

    def _resolve_split_root(self, root: Path) -> Path:
        for split_name in self._iter_split_candidates():
            if root.name.lower() == split_name.lower():
                return root
            candidate = root / split_name
            if candidate.is_dir():
                return candidate
        return root

    def _iter_split_candidates(self):
        requested = [self.split.lower(), *self.SPLIT_FALLBACKS.get(self.split.lower(), ())]
        seen: set = set()
        for key in requested:
            for alias in self.SPLIT_DIR_ALIASES.get(key, (key,)):
                a = alias.lower()
                if a in seen:
                    continue
                seen.add(a)
                yield alias

    # ------------------------------------------- sensor / material inference

    def _infer_sensor_id(self, path: Path, scan_root: Path) -> int:
        targets = {
            self._normalize_sensor_name(name): idx
            for idx, name in enumerate(self.SENSOR_NAMES)
        }
        unknown_idx = self.SENSOR_NAMES.index("Unknown")
        for ancestor in self._ancestors_until(path.parent, scan_root):
            idx = targets.get(self._normalize_sensor_name(ancestor.name))
            if idx is not None:
                return idx
        return unknown_idx

    def _infer_material_id(
        self, path: Path, scan_root: Path, liveness: int,
    ) -> int:
        if liveness == self.LABEL_LIVE:
            return self.MATERIAL_NAMES.index("Live")
        targets = {
            self._normalize_material_name(name): idx
            for idx, name in enumerate(self.MATERIAL_NAMES)
        }
        unknown_idx = self.MATERIAL_NAMES.index("Unknown")
        for ancestor in self._ancestors_until(path.parent, scan_root):
            idx = targets.get(self._normalize_material_name(ancestor.name))
            if idx is not None:
                return idx
        return unknown_idx

    @staticmethod
    def _ancestors_until(path: Path, stop_at: Path):
        current = path
        while True:
            yield current
            if current == stop_at or current.parent == current:
                break
            current = current.parent

    @staticmethod
    def _normalize_sensor_name(name: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "", name.lower())
        for suffix in ("train", "test"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
        return cleaned

    @staticmethod
    def _normalize_material_name(name: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "", name.lower())
        aliases = {
            "playdoh": "playdoh",
            "playdough": "playdoh",
            "gelatine": "gelatin",
            "gelatin": "gelatin",
            "bodydouble": "bodydouble",
            "woodglue": "woodglue",
        }
        return aliases.get(cleaned, cleaned)

    # ------------------------------------------------------------ Dataset IO

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        tensor = TF.to_tensor(img)
        tensor = TF.resize(tensor, self.image_size, antialias=True)
        tensor = TF.center_crop(tensor, self.image_size)
        return tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, ident, liveness, sensor_id, material_id = self.samples[idx]
        try:
            image = self._load_image(path)
        except (OSError, IOError):
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform is not None:
            image = self.transform(image)

        return {
            "images": image,
            "identity_labels": torch.tensor(ident, dtype=torch.long),
            "identity_loss_ignore": torch.tensor(True, dtype=torch.bool),
            "liveness_labels": torch.tensor(liveness, dtype=torch.long),
            "sensor_labels": torch.tensor(sensor_id, dtype=torch.long),
            "material_labels": torch.tensor(material_id, dtype=torch.long),
        }

    # -------------------------------------------------------------- helpers

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def get_labels(self) -> List[int]:
        return [s[1] for s in self.samples]

    def get_liveness_labels(self) -> List[int]:
        return [s[2] for s in self.samples]

    def summary(self) -> str:
        live = sum(1 for s in self.samples if s[2] == self.LABEL_LIVE)
        spoof = len(self.samples) - live
        return (
            f"LivDetJointDataset(split={self.split}, samples={len(self.samples)}, "
            f"identities={self.num_classes}, live={live}, spoof={spoof}, "
            f"unparsed_skipped={self._unparsed})"
        )
