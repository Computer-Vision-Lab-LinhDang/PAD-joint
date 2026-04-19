"""
FVC fingerprint dataset loader for evaluation.

Supports FVC2000, FVC2002, FVC2004.
Flat file layout: {subject_id}_{impression_id}.tif
    100 subjects × 8 impressions = 800 files per DB.

Returns: {'images': (1, 224, 224), 'identity_labels': LongTensor}
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class FVCDataset(Dataset):
    """
    FVC fingerprint identity dataset.

    Parses flat directories with naming convention: {subject}_{impression}.tif
    Subject IDs become identity labels (0-indexed).

    Args:
        root:        Path to a specific DB folder (e.g. FVC2004/Dbs/DB1_A)
        transform:   Optional augmentation.
        image_size:  Resize target (default 224).
        dataset_name: Informational label (e.g. 'FVC2004_DB1_A').
    """

    IMG_EXTS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        dataset_name: str = 'FVC',
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.image_size = image_size
        self.dataset_name = dataset_name

        self.samples: List[Tuple[Path, int]] = []
        self.subject_to_idx: Dict[str, int] = {}

        self._load_samples()

    def _load_samples(self):
        if not self.root.exists():
            raise FileNotFoundError(f"FVCDataset root does not exist: {self.root}")

        files = sorted(
            f for f in self.root.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMG_EXTS
        )

        subjects = set()
        for f in files:
            subject_id = f.stem.split('_')[0]
            subjects.add(subject_id)

        # Sort subjects numerically
        sorted_subjects = sorted(subjects, key=lambda x: int(x))
        self.subject_to_idx = {s: i for i, s in enumerate(sorted_subjects)}

        for f in files:
            subject_id = f.stem.split('_')[0]
            self.samples.append((f, self.subject_to_idx[subject_id]))

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return TF.to_tensor(img)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label = self.samples[idx]
        image = self._load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return {
            'images': image,
            'identity_labels': torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.subject_to_idx)

    def get_labels(self) -> List[int]:
        return [label for _, label in self.samples]


def discover_fvc_dbs(fvc_root: str) -> List[Tuple[str, str]]:
    """
    Auto-discover all FVC DB directories under a root.

    Scans for FVC2000/FVC2002/FVC2004 → Dbs → DB*_A directories.

    Returns:
        List of (dataset_name, db_path) tuples.
        e.g. [('FVC2000_Db1_a', '/path/FVC2000/Dbs/Db1_a'), ...]
    """
    root = Path(fvc_root)
    results = []

    for fvc_dir in sorted(root.iterdir()):
        if not fvc_dir.is_dir():
            continue
        name = fvc_dir.name
        if not name.upper().startswith('FVC'):
            continue

        dbs_dir = fvc_dir / 'Dbs'
        if not dbs_dir.is_dir():
            continue

        for db in sorted(dbs_dir.iterdir()):
            if not db.is_dir():
                continue
            # Only use set A (public set)
            if db.name.lower().endswith('_a'):
                ds_name = f"{name}_{db.name}"
                results.append((ds_name, str(db)))

    return results
