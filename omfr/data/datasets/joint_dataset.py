"""
Joint dataset for samples with BOTH identity and liveness labels.

Supports: MSU-FPAD v2.0 (has per-finger identity + live/spoof label)

Returns: {
    'images':           (1, 224, 224),
    'identity_labels':  LongTensor scalar,
    'liveness_labels':  LongTensor scalar  (0=spoof, 1=live),
}
"""

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class JointDataset(Dataset):
    """
    Fingerprint dataset with simultaneous identity and liveness labels.

    Requires a labels CSV at root/labels_{split}.csv with columns:
        relative_path, identity_label, liveness_label
        (liveness_label: 1=live, 0=spoof)

    Alternatively, directory layout:
        root/
            subject_001/
                live/
                    *.bmp
                spoof/
                    *.bmp

    Args:
        root:         Dataset root.
        split:        'train' or 'test'.
        transform:    Optional augmentation.
        image_size:   224.
        dataset_name: e.g. 'MSU-FPAD'.
    """

    IMG_EXTS    = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    LIVE_DIRS   = {'live', 'Live', 'LIVE'}
    SPOOF_DIRS  = {'fake', 'Fake', 'spoof', 'Spoof'}

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224,
        dataset_name: str = 'MSU-FPAD',
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.dataset_name = dataset_name

        # (path, identity_idx, liveness_label)
        self.samples: List[Tuple[Path, int, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        self._load_samples()

    # ------------------------------------------------------------------

    def _load_samples(self):
        csv_file = self.root / f'labels_{self.split}.csv'
        if csv_file.exists():
            self._load_from_csv(csv_file)
        else:
            self._load_from_directory()

    def _load_from_csv(self, csv_file: Path):
        rows = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        # Build class_to_idx from all identity labels
        identity_names = sorted({r['identity_label'] for r in rows})
        self.class_to_idx = {n: i for i, n in enumerate(identity_names)}

        for row in rows:
            path = self.root / row['relative_path']
            if path.suffix.lower() not in self.IMG_EXTS:
                continue
            identity_idx = self.class_to_idx[row['identity_label']]
            liveness = int(row.get('liveness_label', 1))
            self.samples.append((path, identity_idx, liveness))

    def _load_from_directory(self):
        subject_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        self.class_to_idx = {d.name: i for i, d in enumerate(subject_dirs)}

        for subject_dir in subject_dirs:
            identity_idx = self.class_to_idx[subject_dir.name]
            for liveness_dir in sorted(subject_dir.iterdir()):
                if not liveness_dir.is_dir():
                    continue
                if liveness_dir.name in self.LIVE_DIRS:
                    liveness = 1
                elif liveness_dir.name in self.SPOOF_DIRS:
                    liveness = 0
                else:
                    continue
                for img_path in sorted(liveness_dir.iterdir()):
                    if img_path.suffix.lower() in self.IMG_EXTS:
                        self.samples.append((img_path, identity_idx, liveness))

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return TF.to_tensor(img)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, identity_idx, liveness = self.samples[idx]
        image = self._load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return {
            'images':           image,
            'identity_labels':  torch.tensor(identity_idx, dtype=torch.long),
            'liveness_labels':  torch.tensor(liveness, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)
