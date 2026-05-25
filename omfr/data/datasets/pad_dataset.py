"""
PAD (Presentation Attack Detection / Liveness) dataset for OMFR training.

Supports: LivDet 2013, LivDet 2015, LivDet 2017
Labels: 0 = spoof (fake), 1 = live

Returns: {
    'images': (1, 224, 224),
    'liveness_labels': LongTensor scalar,
    'sensor_labels': LongTensor scalar,
    'material_labels': LongTensor scalar,
}
"""

import csv
import os
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def make_pad_train_transform(image_size: int = 224) -> T.Compose:
    """
    Strong augmentation pipeline for PAD training.

    Combats cross-sensor overfit by simulating intra-class variability
    (sensor noise, brightness/contrast drift, mild geometric jitter) so
    the head cannot memorise per-image pixel statistics. Operates on
    1-channel tensors already loaded by PADDataset._load_image.

    Keep output on the same [0, 1] scale used by validation/evaluation and
    the identity path. A train-only Normalize here creates a PAD train/test
    distribution shift because eval PADDataset instances do not transform.
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(
            [T.RandomRotation(degrees=15, fill=0.0)],
            p=0.7,
        ),
        T.RandomResizedCrop(
            image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1),
            antialias=True,
        ),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
            p=0.3,
        ),
        T.RandomErasing(
            p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0,
        ),
    ])


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

    # Ordered sensor list — index in this tuple is the sensor_id integer
    # used by the adversarial head (TASK_03). Append to the END only so
    # indices stay stable across runs.
    SENSOR_NAMES = (
        'Biometrika',
        'CrossMatch',
        'DigitalPersona',
        'GreenBit',
        'HiScan',
        'Italdata',
        'Swipe',
        'Orcanthus',
        'Unknown',
    )
    MATERIAL_NAMES = (
        'Live',
        'BodyDouble',
        'Ecoflex',
        'Gelatin',
        'Latex',
        'Modasil',
        'PlayDoh',
        'WoodGlue',
        'Unknown',
    )

    LABEL_LIVE  = 1
    LABEL_SPOOF = 0
    SPLIT_DIR_ALIASES = {
        'train': ('Training', 'Train', 'train'),
        'val': ('Validation', 'Val', 'val', 'Testing', 'Test'),
        'test': ('Testing', 'Test', 'test', 'Val', 'val'),
    }
    SPLIT_FALLBACKS = {
        'test': ('val',),
        'val': ('test',),
    }

    def __init__(
        self,
        root: Union[str, Sequence[str]],
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224,
        dataset_name: str = 'LivDet2015',
        sensor: Optional[str] = None,
    ):
        super().__init__()
        if isinstance(root, (str, Path)):
            self.roots = [Path(root)]
        else:
            self.roots = [Path(item) for item in root]

        self.root = self.roots[0]
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.sensor = sensor

        # (path, label, sensor_id, material_id) — ids index the tuples above.
        self.samples: List[Tuple[Path, int, int, int]] = []
        self._load_samples()

    # ------------------------------------------------------------------

    def _load_samples(self):
        for root in self.roots:
            if not root.exists():
                raise FileNotFoundError(
                    f"PADDataset root does not exist: {root}"
                )

            csv_file = root / f'labels_{self.split}.csv'
            if csv_file.exists():
                self._load_from_csv(root, csv_file)
            else:
                self._load_from_directory(root)

        if not self.samples:
            roots = ", ".join(str(root) for root in self.roots)
            raise RuntimeError(
                f"PADDataset found no samples for split={self.split!r} under: {roots}"
            )

    def _load_from_csv(self, root: Path, csv_file: Path):
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row.get('relative_path') or row.get('path', '')
                label_str = row.get('label', '0')
                path = root / rel_path
                label = self.LABEL_LIVE if label_str.lower() in {'live', '1', 'alive'} else self.LABEL_SPOOF
                if path.is_file() and path.suffix.lower() in self.IMG_EXTS:
                    sensor_id = int(row.get('sensor_id', -1))
                    if sensor_id < 0:
                        sensor_id = self._infer_sensor_id(path, root)
                    material_id = int(row.get('material_id', -1))
                    if material_id < 0:
                        material_id = self._infer_material_id(path, root, label)
                    self.samples.append((path, label, sensor_id, material_id))

    def _load_from_directory(self, root: Path):
        scan_root = self._resolve_split_root(root)

        for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=True):
            dirnames[:] = [name for name in dirnames if not name.startswith('.')]
            current_dir = Path(dirpath)

            for filename in filenames:
                img_path = current_dir / filename
                if img_path.suffix.lower() not in self.IMG_EXTS:
                    continue

                label = self._infer_label(img_path, scan_root)
                if label is None:
                    continue

                if self.sensor and not self._matches_sensor(img_path, scan_root):
                    continue

                sensor_id = self._infer_sensor_id(img_path, scan_root)
                material_id = self._infer_material_id(img_path, scan_root, label)
                self.samples.append((img_path, label, sensor_id, material_id))

    def _infer_sensor_id(self, path: Path, scan_root: Path) -> int:
        """Walk ancestors until a directory matches a known sensor name.
        Falls back to the 'Unknown' index."""
        normalized_targets = {
            self._normalize_sensor_name(name): idx
            for idx, name in enumerate(self.SENSOR_NAMES)
        }
        unknown_idx = self.SENSOR_NAMES.index('Unknown')
        for ancestor in self._ancestors_until(path.parent, scan_root):
            idx = normalized_targets.get(self._normalize_sensor_name(ancestor.name))
            if idx is not None:
                return idx
        return unknown_idx

    def _infer_material_id(self, path: Path, scan_root: Path, label: int) -> int:
        """Infer spoof material from ancestors; live samples use material id 0."""
        if label == self.LABEL_LIVE:
            return self.MATERIAL_NAMES.index('Live')

        normalized_targets = {
            self._normalize_material_name(name): idx
            for idx, name in enumerate(self.MATERIAL_NAMES)
        }
        unknown_idx = self.MATERIAL_NAMES.index('Unknown')
        for ancestor in self._ancestors_until(path.parent, scan_root):
            idx = normalized_targets.get(self._normalize_material_name(ancestor.name))
            if idx is not None:
                return idx
        return unknown_idx

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
        seen = set()
        for key in requested:
            for alias in self.SPLIT_DIR_ALIASES.get(key, (key,)):
                alias_lc = alias.lower()
                if alias_lc in seen:
                    continue
                seen.add(alias_lc)
                yield alias

    def _infer_label(self, path: Path, scan_root: Path) -> Optional[int]:
        for ancestor in self._ancestors_until(path.parent, scan_root):
            if ancestor.name in self.LIVE_DIRS:
                return self.LABEL_LIVE
            if ancestor.name in self.SPOOF_DIRS:
                return self.LABEL_SPOOF
        return None

    def _matches_sensor(self, path: Path, scan_root: Path) -> bool:
        sensor = self._normalize_sensor_name(self.sensor or '')
        if not sensor:
            return True

        for ancestor in self._ancestors_until(path.parent, scan_root):
            if self._normalize_sensor_name(ancestor.name) == sensor:
                return True

        return False

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
        cleaned = re.sub(r'[^a-z0-9]+', '', name.lower())
        for suffix in ('train', 'test'):
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
        return cleaned

    @staticmethod
    def _normalize_material_name(name: str) -> str:
        cleaned = re.sub(r'[^a-z0-9]+', '', name.lower())
        aliases = {
            'bodydouble': 'bodydouble',
            'gelatin': 'gelatin',
            'gelatine': 'gelatin',
            'playdoh': 'playdoh',
            'playdough': 'playdoh',
            'woodglue': 'woodglue',
        }
        return aliases.get(cleaned, cleaned)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('L')
        tensor = TF.to_tensor(img)                      # (1, H, W)
        tensor = TF.resize(
            tensor, self.image_size, antialias=True,
        )                                                # resize short edge
        tensor = TF.center_crop(tensor, self.image_size)
        return tensor

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label, sensor_id, material_id = self.samples[idx]
        try:
            image = self._load_image(path)
        except (OSError, IOError):
            # Corrupt image — return a random valid sample instead
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform is not None:
            image = self.transform(image)

        return {
            'images': image,
            'liveness_labels': torch.tensor(label, dtype=torch.long),
            'sensor_labels': torch.tensor(sensor_id, dtype=torch.long),
            'material_labels': torch.tensor(material_id, dtype=torch.long),
        }

    @property
    def num_live(self) -> int:
        return sum(1 for _, l, _, _ in self.samples if l == self.LABEL_LIVE)

    @property
    def num_spoof(self) -> int:
        return sum(1 for _, l, _, _ in self.samples if l == self.LABEL_SPOOF)

    def get_liveness_labels(self) -> List[int]:
        return [label for _, label, _, _ in self.samples]

    def get_sensor_labels(self) -> List[int]:
        return [sensor for _, _, sensor, _ in self.samples]

    def get_material_labels(self) -> List[int]:
        return [material for _, _, _, material in self.samples]

    @property
    def num_sensors(self) -> int:
        return len(self.SENSOR_NAMES)
