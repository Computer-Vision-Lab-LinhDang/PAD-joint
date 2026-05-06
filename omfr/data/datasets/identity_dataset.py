"""
Identity dataset for OMFR training.

Supports: FVC2004, NIST SD302, NIST SD300, NIST SD302a
Returns: {'images': (B, 1, 224, 224), 'identity_labels': (B,) LongTensor}
"""

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


_SUBJECT_RE = re.compile(r"(?:^|[_\-])subj[_\-](\d+)(?=[_\-]|$)", re.IGNORECASE)
_A300_DIR_RE = re.compile(r"^a300_subj_(\d+)_frgp_(\d+)$", re.IGNORECASE)
_R302_DIR_RE = re.compile(r"^r302_subj_(\d+)_frgp_(\d+)$", re.IGNORECASE)
_D302_DIR_RE = re.compile(r"^d302_(\d+)_fp(\d+)$", re.IGNORECASE)
_NIST_FRGP_DIR_RE = re.compile(r"^subj_(\d+)_frgp_(\d+)$", re.IGNORECASE)


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


def _canonical_finger_id(folder_name: str) -> str:
    """
    Return the matching identity key for NIST-style single-finger folders.

    For fingerprint identity, a class is a subject *and* a finger position.
    In the combined NIST layout, SD302 rolled (`r302`) and plain/flat (`d302`)
    views are different impressions of the same subject+finger, so they share
    the same `n302` namespace. SD300 gets its own namespace to avoid accidental
    subject-id collisions across databases.
    """
    m = _A300_DIR_RE.match(folder_name)
    if m:
        subject, frgp = m.groups()
        return f"a300:{subject}:{int(frgp):02d}"

    m = _R302_DIR_RE.match(folder_name)
    if m:
        subject, frgp = m.groups()
        return f"n302:{subject}:{int(frgp):02d}"

    m = _D302_DIR_RE.match(folder_name)
    if m:
        subject, frgp = m.groups()
        return f"n302:{subject}:{int(frgp):02d}"

    m = _NIST_FRGP_DIR_RE.match(folder_name)
    if m:
        subject, frgp = m.groups()
        return f"subj:{subject}:{int(frgp):02d}"

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
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224,
        dataset_name: str = 'NIST_SD302',
        group_by_subject: bool = False,
        identity_label_mode: Optional[str] = None,
        num_views: int = 1,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.group_by_subject = group_by_subject
        self.identity_label_mode = self._normalize_label_mode(identity_label_mode)
        self.num_views = max(int(num_views), 1)

        self.samples: List[Tuple[Path, int]] = []   # (image_path, class_idx)
        self.class_to_idx: Dict[str, int] = {}

        self._load_samples()

    def _class_name_from_dir(self, dir_name: str) -> str:
        if self.identity_label_mode == 'canonical_finger':
            return _canonical_finger_id(dir_name)
        if self.identity_label_mode == 'subject':
            return _subject_id(dir_name)
        return dir_name

    def _normalize_label_mode(self, mode: Optional[str]) -> str:
        if mode is None:
            return 'subject' if self.group_by_subject else 'folder'

        normalized = str(mode).strip().lower().replace('-', '_')
        aliases = {
            'dir': 'folder',
            'directory': 'folder',
            'class_folder': 'folder',
            'subject_id': 'subject',
            'group_by_subject': 'subject',
            'finger': 'canonical_finger',
            'subject_finger': 'canonical_finger',
            'subject_position': 'canonical_finger',
            'canonical': 'canonical_finger',
        }
        normalized = aliases.get(normalized, normalized)
        valid = {'folder', 'subject', 'canonical_finger'}
        if normalized not in valid:
            raise ValueError(
                f"Unsupported identity_label_mode={mode!r}; "
                f"expected one of {sorted(valid)}"
            )
        return normalized

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

    def _load_from_directory(self, scan_root: Path):
        class_dirs = sorted(
            d for d in scan_root.iterdir() if d.is_dir() and not d.name.startswith('.')
        )
        unique_classes = sorted({self._class_name_from_dir(d.name) for d in class_dirs})
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[self._class_name_from_dir(class_dir.name)]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((img_path, class_idx))

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


def build_identity_dataset(
    root: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    image_size: int = 224,
    dataset_name: str = 'NIST_SD302',
    group_by_subject: bool = False,
    identity_label_mode: Optional[str] = None,
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
        identity_label_mode=identity_label_mode,
        num_views=num_views,
    )
