"""
Identity dataset for OMFR training.

Supports: FVC2004, NIST SD302, NIST SD300, NIST SD302a
Returns: {'images': (B, 1, 224, 224), 'identity_labels': (B,) LongTensor}
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


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

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224,
        dataset_name: str = 'NIST_SD302',
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.dataset_name = dataset_name

        self.samples: List[Tuple[Path, int]] = []   # (image_path, class_idx)
        self.class_to_idx: Dict[str, int] = {}

        self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self):
        split_file = self.root / f'split_{self.split}.txt'
        if split_file.exists():
            self._load_from_split_file(split_file)
        else:
            self._load_from_directory()

    def _load_from_split_file(self, split_file: Path):
        lines = split_file.read_text().strip().splitlines()
        class_names: List[str] = []
        paths: List[Path] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            p = self.root / line
            class_name = p.parent.name
            class_names.append(class_name)
            paths.append(p)

        unique_classes = sorted(set(class_names))
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        for path, class_name in zip(paths, class_names):
            if path.suffix.lower() in self.IMG_EXTS:
                self.samples.append((path, self.class_to_idx[class_name]))

    def _load_from_directory(self):
        class_dirs = sorted(
            d for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTS:
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
        image = self._load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return {
            'images': image,                            # (1, H, W)
            'identity_labels': torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


def build_identity_dataset(
    root: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    image_size: int = 224,
    dataset_name: str = 'NIST_SD302',
) -> IdentityDataset:
    """Convenience factory."""
    return IdentityDataset(
        root=root,
        split=split,
        transform=transform,
        image_size=image_size,
        dataset_name=dataset_name,
    )
