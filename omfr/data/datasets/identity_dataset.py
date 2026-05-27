"""
Identity dataset for OMFR training.

Supports: FVC2000/2002/2004 (flat <subj>_<imp>.tif layout) and NIST
SD302/SD300/SD302a (per-folder layout, plus the `nist_person` variant
where one folder = one PERSON with multiple rolls).

Two construction modes:
    1. Legacy single-root:
            IdentityDataset(root="...", split="train", ...)
       Walks a directory or a split_train.txt file and treats each
       sub-folder as a class.

    2. Multi-source (recommended for FVC+NIST mix):
            IdentityDataset(sources=[
                {"type": "fvc",  "root": "/.../FVC2000", "dbs_glob": "Db*_a", "tag": "fvc2000"},
                {"type": "nist_person", "root": "/.../SD302a_ids_a/full",   "tag": "nist302a"},
                ...
            ])
       Each source has its own parser; class names are tag-qualified so
       collisions across years/DBs are impossible. group_by_subject is
       ignored in this mode (each source decides its own grouping).

Returns: {'images': (B, 1, 224, 224), 'identity_labels': (B,) LongTensor}
"""

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


_SUBJECT_RE = re.compile(r"(?:^|[_\-])subj[_\-](\d+)(?=[_\-]|$)", re.IGNORECASE)
_IMG_EXTS = {'.bmp', '.png', '.wsq', '.jpg', '.jpeg', '.tif', '.tiff'}


def _scan_fvc_source(
    root: Path,
    dbs_glob: str,
    tag: str,
) -> List[Tuple[Path, str, str]]:
    """Scan an FVC year root.

    Expected layout: <root>/Dbs/<DB>/<subject>_<impression>.tif
        e.g. /.../FVC2000/Dbs/Db1_a/100_3.tif

    Each (db, subject_local_id) becomes a distinct class name
    ``f"{tag}_{db}_{subj}"`` — so FVC2000/Db1_a/1 is NOT merged with
    FVC2002/Db1_a/1 (different captures of different fingers).

    Returns triples (path, class_name, group="fvc"); empty if the FVC
    root does not exist or has no matching DBs.
    """
    dbs_root = root / "Dbs"
    if not dbs_root.is_dir():
        return []

    out: List[Tuple[Path, str, str]] = []
    for db_dir in sorted(dbs_root.glob(dbs_glob)):
        if not db_dir.is_dir():
            continue
        db_name = db_dir.name
        for img in sorted(db_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in _IMG_EXTS:
                continue
            stem = img.stem
            # FVC naming: "<subj>_<impression>" — split on the FIRST
            # underscore so subj keeps any leading zeros.
            if "_" not in stem:
                continue
            subj = stem.split("_", 1)[0]
            class_name = f"{tag}_{db_name}_{subj}"
            out.append((img, class_name, "fvc"))
    return out


def _scan_nist_person_source(
    root: Path,
    tag: str,
) -> List[Tuple[Path, str, str]]:
    """Scan a NIST SD302a person-level root.

    Expected layout: <root>/<person_id>/<person_id>_A_roll_NN.png
        e.g. /.../SD302a_ids_a/full/00002303/00002303_A_roll_07.png

    One folder = one PERSON; the 10 rolls per folder are different
    fingers of the same person, all sharing the same class label.
    Class name = ``f"{tag}_{folder_name}"``.

    Mostly kept for back-compat; finger-level identity training should
    use ``dir_per_class`` on NIST-300-ds instead — see below.
    """
    if not root.is_dir():
        return []

    out: List[Tuple[Path, str, str]] = []
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir() or person_dir.name.startswith('.'):
            continue
        class_name = f"{tag}_{person_dir.name}"
        for img in sorted(person_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in _IMG_EXTS:
                out.append((img, class_name, "nist"))
    return out


def _scan_dir_per_class_source(
    root: Path,
    tag: str,
    group: str = "nist",
) -> List[Tuple[Path, str, str]]:
    """Generic ``<root>/<class_folder>/*.img`` scanner.

    Used for NIST-300-ds (``subj_XXXX_frgp_NN`` per finger), the legacy
    ``identity_stage2_nist302a`` finger-level split, and anything else
    that already lays out one class per folder.

    Class name = ``f"{tag}_{folder_name}"``; the ``group`` arg overrides
    the per-sample group tag (defaults to ``"nist"`` because every
    existing user of this scanner is NIST-derived).
    """
    if not root.is_dir():
        return []

    out: List[Tuple[Path, str, str]] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        class_name = f"{tag}_{class_dir.name}"
        for img in sorted(class_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in _IMG_EXTS:
                out.append((img, class_name, group))
    return out


_SOURCE_SCANNERS = {
    "fvc": _scan_fvc_source,
    "nist_person": _scan_nist_person_source,
    "dir_per_class": _scan_dir_per_class_source,
}


def _subject_id(folder_name: str) -> str:
    """
    Extract subject ID from NIST-style folder names.

    Examples:
        'a300_subj_00001000_frgp_01' -> 'subj_00001000'
        'subj_00001000'              -> 'subj_00001000'
        'subject_0001'               -> 'subject_0001' (unchanged, no delimiter after 'subj')
    """
    m = _SUBJECT_RE.search(folder_name)
    if m:
        return f"subj_{m.group(1)}"
    return folder_name


class IdentityDataset(Dataset):
    """
    Fingerprint identity dataset supporting multiple benchmarks.

    Directory layout expected (one folder per identity):
        root/
            subject_0001/
                f1.bmp
                f2.bmp
            subject_0002/
                ...

    Args:
        root:       Root directory of the dataset.
        split:      'train' or 'test'. If a split file exists at
                    root/split_train.txt (one relative path per line),
                    it is used; otherwise all images are used.
        transform:  Optional additional augmentation transform.
        image_size: Resize target (default 224).
        dataset_name: Informational label ('FVC2004', 'NIST_SD302',
                       'NIST_SD300', 'NIST_SD302a').
    """

    SUPPORTED = ('FVC2004', 'NIST_SD302', 'NIST_SD300', 'NIST_SD302a')
    IMG_EXTS = {'.bmp', '.png', '.wsq', '.jpg', '.jpeg', '.tif', '.tiff'}
    SPLIT_DIR_ALIASES = {
        'train': ('train', 'Train', 'Training'),
        'val': ('val', 'Val', 'Validation', 'valid', 'Valid'),
        'test': ('test', 'Test', 'Testing'),
    }
    SPLIT_FALLBACKS = {
        'test': ('val',),
        'val': ('test',),
    }

    def __init__(
        self,
        root: Optional[str] = None,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224,
        dataset_name: str = 'NIST_SD302',
        group_by_subject: bool = False,
        num_views: int = 1,
        sources: Optional[List[Dict[str, Any]]] = None,
        val_class_fraction: float = 0.0,
        val_seed: int = 42,
    ):
        super().__init__()
        if sources is None and not root:
            raise ValueError(
                "IdentityDataset requires either `root=` (legacy single-root "
                "mode) or `sources=` (multi-source mode)."
            )
        self.root = Path(root) if root else None
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.group_by_subject = group_by_subject
        self.num_views = max(int(num_views), 1)
        self.sources = sources
        # Open-set val holdout: when sources are used and
        # val_class_fraction > 0, classes are partitioned per group
        # (stratified). Same `sources` + `val_class_fraction` + `val_seed`
        # always produce the same partition, so train and val datasets
        # constructed with split='train' / split='val' are exact
        # complements with disjoint class sets.
        self.val_class_fraction = float(val_class_fraction)
        self.val_seed = int(val_seed)

        self.samples: List[Tuple[Path, int]] = []   # (image_path, class_idx)
        self.class_to_idx: Dict[str, int] = {}
        # Per-sample group tag ("fvc" / "nist" / ...). Populated by every
        # loader path so GroupBalancedPKSampler works in either mode.
        self._sample_groups: List[str] = []

        if sources is not None:
            self._load_from_sources(sources)
        else:
            self._load_samples()

    def _class_name_from_dir(self, dir_name: str) -> str:
        if self.group_by_subject:
            return _subject_id(dir_name)
        return dir_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self):
        if not self.root.exists():
            raise FileNotFoundError(
                f"IdentityDataset root does not exist: {self.root}"
            )

        split_file = self.root / f'split_{self.split}.txt'
        if split_file.exists():
            self._load_from_split_file(split_file)
        else:
            self._load_from_directory(self._resolve_split_root())

    def _load_from_sources(self, sources: List[Dict[str, Any]]) -> None:
        """Scan a list of typed sources (fvc / nist_person / ...).

        All sources share one ``class_to_idx`` so ArcFace sees one
        contiguous class space. Class names are tag-qualified per source
        scanner so cross-source collisions are impossible.

        Missing roots are skipped with a warning rather than raising —
        useful when a config is reused across machines with different
        data layouts.
        """
        triples: List[Tuple[Path, str, str]] = []
        for src in sources:
            stype = str(src.get("type", "")).lower()
            scanner = _SOURCE_SCANNERS.get(stype)
            if scanner is None:
                raise ValueError(
                    f"Unknown identity source type {stype!r}; "
                    f"expected one of {sorted(_SOURCE_SCANNERS)}."
                )
            tag = str(src.get("tag", stype))
            root = Path(src["root"])
            if not root.exists():
                print(f"[IdentityDataset] WARN: source root missing, skipped: {root}")
                continue
            if stype == "fvc":
                dbs_glob = str(src.get("dbs_glob", "Db*_a"))
                triples.extend(_scan_fvc_source(root, dbs_glob, tag))
            elif stype == "nist_person":
                triples.extend(_scan_nist_person_source(root, tag))
            elif stype == "dir_per_class":
                group = str(src.get("group", "nist"))
                triples.extend(_scan_dir_per_class_source(root, tag, group))

        if not triples:
            raise RuntimeError(
                "IdentityDataset(sources=...) produced no samples — "
                "check `root` / `dbs_glob` in your config."
            )

        # Group every class by source-group tag so the train/val split
        # can be stratified (FVC and NIST contribute proportionally to
        # the held-out set).
        class_paths: Dict[str, List[Path]] = defaultdict(list)
        class_group: Dict[str, str] = {}
        for path, class_name, group in triples:
            class_paths[class_name].append(path)
            class_group[class_name] = group

        chosen_classes = self._select_classes(class_paths, class_group)
        chosen_classes = sorted(chosen_classes)
        self.class_to_idx = {c: i for i, c in enumerate(chosen_classes)}
        for class_name in chosen_classes:
            for path in class_paths[class_name]:
                self.samples.append((path, self.class_to_idx[class_name]))
                self._sample_groups.append(class_group[class_name])

    def _select_classes(
        self,
        class_paths: Dict[str, List[Path]],
        class_group: Dict[str, str],
    ) -> List[str]:
        """Pick which classes belong in this split.

        With ``val_class_fraction <= 0`` (legacy / training-only setup)
        every class is included. Otherwise a deterministic shuffle per
        group selects ``round(N_g * frac)`` classes from each group for
        val, and the complement for train, so the two splits are exact
        complements over the same source list.
        """
        all_classes = list(class_paths.keys())
        if self.val_class_fraction <= 0.0:
            return all_classes

        classes_by_group: Dict[str, List[str]] = defaultdict(list)
        for cname in all_classes:
            classes_by_group[class_group[cname]].append(cname)

        val_classes: set = set()
        for group, classes in classes_by_group.items():
            # Deterministic per-group shuffle: same seed + same group
            # ordering produce the same partition on train and val sides.
            local = sorted(classes)
            rng = random.Random(hash((self.val_seed, group)) & 0x7fffffff)
            rng.shuffle(local)
            n_val = int(round(len(local) * self.val_class_fraction))
            n_val = max(0, min(n_val, len(local) - 1))   # keep ≥1 in train
            val_classes.update(local[:n_val])

        if self.split.lower() in {"val", "valid", "validation", "test"}:
            return sorted(val_classes)
        # Default = train: complement of val set.
        return [c for c in all_classes if c not in val_classes]

    def _resolve_split_root(self) -> Path:
        for split_name in self._iter_split_candidates():
            if self.root.name.lower() == split_name.lower():
                return self.root

            candidate = self.root / split_name
            if candidate.is_dir():
                return candidate

        return self.root

    def _iter_split_candidates(self):
        requested = [self.split.lower(), *self.SPLIT_FALLBACKS.get(self.split.lower(), ())]
        seen = set()
        for key in requested:
            for alias in self.SPLIT_DIR_ALIASES.get(key, (key,)):
                alias_lc = alias.lower()
                if alias_lc in seen:
                    continue
                seen.add(alias_lc)
                yield alias

    def _load_from_split_file(self, split_file: Path):
        lines = split_file.read_text().strip().splitlines()
        class_names: List[str] = []
        paths: List[Path] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            p = self.root / line
            class_name = self._class_name_from_dir(p.parent.name)
            class_names.append(class_name)
            paths.append(p)

        unique_classes = sorted(set(class_names))
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        for path, class_name in zip(paths, class_names):
            if path.suffix.lower() in self.IMG_EXTS:
                self.samples.append((path, self.class_to_idx[class_name]))
                self._sample_groups.append(
                    "fvc" if "fvc" in path.parent.name.lower() else "nist"
                )

    def _load_from_directory(self, scan_root: Path):
        class_dirs = sorted(
            d for d in scan_root.iterdir() if d.is_dir() and not d.name.startswith('.')
        )
        unique_classes = sorted({self._class_name_from_dir(d.name) for d in class_dirs})
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[self._class_name_from_dir(class_dir.name)]
            group = "fvc" if "fvc" in class_dir.name.lower() else "nist"
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((img_path, class_idx))
                    self._sample_groups.append(group)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('L')             # grayscale
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = TF.to_tensor(img)                      # (1, H, W), [0, 1]
        return tensor

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label = self.samples[idx]
        try:
            image = self._load_image(path)
        except (OSError, IOError):
            return self.__getitem__((idx + 1) % len(self.samples))

        # Single-view path: standard transform-or-passthrough.
        if self.num_views <= 1:
            if self.transform is not None:
                image = self.transform(image)
            return {
                'images': image,                        # (1, H, W)
                'identity_labels': torch.tensor(label, dtype=torch.long),
            }

        # Multi-view: stack V independently-augmented copies. Without a
        # stochastic transform the views are identical, which defeats
        # multi-view SupCon — fall back to single-view in that case.
        if self.transform is None:
            return {
                'images': image,
                'identity_labels': torch.tensor(label, dtype=torch.long),
            }

        views = torch.stack(
            [self.transform(image) for _ in range(self.num_views)],
            dim=0,
        )                                               # (V, 1, H, W)
        return {
            'images': views,
            'identity_labels': torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def get_labels(self) -> List[int]:
        return [label for _, label in self.samples]

    def get_groups(self) -> List[str]:
        """Per-sample dataset group tag (``"fvc"`` / ``"nist"`` / ...).

        Populated when samples are loaded (both source-mode and legacy
        single-root mode). Used by ``GroupBalancedPKSampler`` to keep
        every batch class-balanced across sources even when one source
        has many more classes than the other (FVC ~1,200 vs NIST 197).
        """
        return list(self._sample_groups)


def build_identity_dataset(
    root: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    image_size: int = 224,
    dataset_name: str = 'NIST_SD302',
    group_by_subject: bool = False,
    num_views: int = 1,
) -> IdentityDataset:
    """Convenience factory."""
    return IdentityDataset(
        root=root,
        split=split,
        transform=transform,
        image_size=image_size,
        dataset_name=dataset_name,
        group_by_subject=group_by_subject,
        num_views=num_views,
    )
