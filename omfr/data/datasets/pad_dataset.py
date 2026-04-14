"""
PAD (Presentation Attack Detection / Liveness) dataset for OMFR training.

Supports: LivDet 2013, LivDet 2015, LivDet 2017
Labels: 0 = spoof (fake), 1 = live

Returns: {'images': (1, 224, 224), 'liveness_labels': LongTensor scalar}
"""

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PADDataset(Dataset):
    """
    Fingerprint liveness dataset supporting LivDet benchmarks.

    Expected directory layout (LivDet convention):
        root/
            Train/
                Live/
                    *.bmp
                Fake/
                    *.bmp
            Test/
                Live/
                    *.bmp
                Fake/
                    *.bmp

    Alternatively, a labels CSV at root/labels_{split}.csv with columns:
        relative_path, label   (label: 'live'/'Live' = 1, else 0)

    Args:
        root:          Root directory.
        split:         'train' or 'test'.
        transform:     Optional augmentation.
        image_size:    Resize target (default 224).
        dataset_name:  'LivDet2013', 'LivDet2015', or 'LivDet2017'.
        sensor:        Optional sensor name filter (e.g. 'Digital_Persona').
    """

    LIVE_DIRS  = {'live', 'Live', 'LIVE', 'Alive'}
    SPOOF_DIRS = {'fake', 'Fake', 'FAKE', 'Spoof', 'spoof'}
    IMG_EXTS   = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    LABEL_LIVE  = 1
    LABEL_SPOOF = 0

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224,
        dataset_name: str = 'LivDet2015',
        sensor: Optional[str] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.sensor = sensor

        self.samples: List[Tuple[Path, int]] = []   # (path, label)
        self._load_samples()

    # ------------------------------------------------------------------

    def _load_samples(self):
        csv_file = self.root / f'labels_{self.split}.csv'
        if csv_file.exists():
            self._load_from_csv(csv_file)
        else:
            self._load_from_directory()

    def _load_from_csv(self, csv_file: Path):
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row.get('relative_path') or row.get('path', '')
                label_str = row.get('label', '0')
                path = self.root / rel_path
                label = self.LABEL_LIVE if label_str.lower() in {'live', '1', 'alive'} else self.LABEL_SPOOF
                if path.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((path, label))

    def _load_from_directory(self):
        split_dir_name = 'Train' if self.split == 'train' else 'Test'
        split_dir = self.root / split_dir_name

        if not split_dir.exists():
            # Flat layout fallback: root/Live/, root/Fake/
            split_dir = self.root

        # Optionally filter by sensor subdirectory
        search_dirs = [split_dir]
        if self.sensor:
            sensor_dir = split_dir / self.sensor
            if sensor_dir.exists():
                search_dirs = [sensor_dir]

        for base_dir in search_dirs:
            for subdir in sorted(base_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                if subdir.name in self.LIVE_DIRS:
                    label = self.LABEL_LIVE
                elif subdir.name in self.SPOOF_DIRS:
                    label = self.LABEL_SPOOF
                else:
                    continue
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in self.IMG_EXTS:
                        self.samples.append((img_path, label))

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return TF.to_tensor(img)                        # (1, H, W)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label = self.samples[idx]
        image = self._load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return {
            'images': image,
            'liveness_labels': torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_live(self) -> int:
        return sum(1 for _, l in self.samples if l == self.LABEL_LIVE)

    @property
    def num_spoof(self) -> int:
        return sum(1 for _, l in self.samples if l == self.LABEL_SPOOF)
