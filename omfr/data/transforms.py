"""
transforms.py — Fingerprint-Specific Augmentations

Augmentation pipeline designed for fingerprint images (grayscale, 224×224).
Applied during training to improve generalization across sensors, subjects,
and acquisition conditions.

Key considerations:
  - Fingerprints are roughly rotationally invariant (any finger angle is valid)
  - Elastic deformation simulates finger pressure/distortion variations
  - Brightness/contrast variation simulates sensor differences
  - Small crops simulate partial fingerprint captures
  - NEVER flip horizontally (chirality of ridge patterns is identity-preserving
    but cross-sensor flips are rare; we keep horizontal flip as optional)
"""

import random
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RandomRotation90:
    """Rotate by a random multiple of 90 degrees."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        k = random.randint(0, 3)
        return torch.rot90(img, k, dims=(-2, -1))


class ElasticDeformation:
    """
    Random elastic deformation to simulate finger pressure variations.

    Generates a random displacement field and applies it via grid_sample.
    Displacement magnitude controlled by alpha; smoothness by sigma.

    Args:
        alpha: float  — displacement magnitude (pixels)
        sigma: float  — Gaussian smoothing sigma for displacement field
        p: float      — probability of applying the transform
    """

    def __init__(self, alpha: float = 20.0, sigma: float = 5.0, p: float = 0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (C, H, W) or (1, H, W) tensor in [0, 1]

        Returns:
            deformed: same shape
        """
        if random.random() > self.p:
            return img

        C, H, W = img.shape

        # Random displacement field
        dx = torch.randn(1, 1, H, W) * self.alpha
        dy = torch.randn(1, 1, H, W) * self.alpha

        # Smooth with Gaussian (approximate via avg_pool repeated)
        kernel_size = max(3, int(self.sigma * 3) | 1)  # ensure odd
        padding = kernel_size // 2
        dx = F.avg_pool2d(dx, kernel_size, stride=1, padding=padding)
        dy = F.avg_pool2d(dy, kernel_size, stride=1, padding=padding)

        # Normalize to [-1, 1] grid coordinates
        norm_dx = dx / (W / 2)
        norm_dy = dy / (H / 2)

        # Build sampling grid
        base_grid = self._make_base_grid(H, W, img.device)  # (1, H, W, 2)
        grid = base_grid + torch.stack([norm_dx, norm_dy], dim=-1).squeeze(1)  # (1,H,W,2)
        grid = grid.clamp(-1.0, 1.0)

        deformed = F.grid_sample(
            img.unsqueeze(0).float(),
            grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=True,
        ).squeeze(0)

        return deformed.to(img.dtype)

    @staticmethod
    def _make_base_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)


class RandomBrightnessContrast:
    """
    Randomly adjust brightness and contrast to simulate sensor differences.

    Args:
        brightness: float  — max brightness delta in [-b, +b]
        contrast:   float  — contrast scale in [1-c, 1+c]
        p: float           — probability
    """

    def __init__(self, brightness: float = 0.3, contrast: float = 0.3, p: float = 0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        # Brightness: additive shift
        b_delta = random.uniform(-self.brightness, self.brightness)
        img = img + b_delta

        # Contrast: multiplicative around mean
        c_scale = random.uniform(1 - self.contrast, 1 + self.contrast)
        mean = img.mean()
        img = (img - mean) * c_scale + mean

        return img.clamp(0.0, 1.0)


class RandomCropPad:
    """
    Randomly crop and pad back to original size to simulate partial captures.

    Args:
        crop_ratio: float  — fraction of image to keep (e.g., 0.85 keeps 85%)
        output_size: int   — output spatial size (square)
        p: float           — probability
    """

    def __init__(
        self,
        crop_ratio: float = 0.85,
        output_size: int = 224,
        p: float = 0.5,
    ):
        self.crop_ratio = crop_ratio
        self.output_size = output_size
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (C, H, W)

        Returns:
            img: (C, output_size, output_size)
        """
        if random.random() > self.p:
            # Still resize to output_size
            return F.interpolate(
                img.unsqueeze(0), size=self.output_size, mode='bilinear',
                align_corners=False
            ).squeeze(0)

        C, H, W = img.shape
        crop_h = int(H * self.crop_ratio)
        crop_w = int(W * self.crop_ratio)

        top  = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        img = img[:, top:top + crop_h, left:left + crop_w]

        # Resize back
        img = F.interpolate(
            img.unsqueeze(0), size=self.output_size, mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return img


class RandomGaussianNoise:
    """Add Gaussian noise to simulate sensor noise."""

    def __init__(self, std: float = 0.02, p: float = 0.3):
        self.std = std
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        noise = torch.randn_like(img) * self.std
        return (img + noise).clamp(0.0, 1.0)


class FingerprintTrainTransform:
    """
    Full training augmentation pipeline for fingerprint images.

    Expected input: (1, H, W) grayscale tensor in [0, 1].
    Output:         (1, 224, 224) tensor in [0, 1].

    Pipeline:
        1. Random rotation (0/90/180/270°)
        2. Elastic deformation (p=0.5)
        3. Random crop + resize (p=0.5)
        4. Brightness/contrast jitter (p=0.8)
        5. Gaussian noise (p=0.3)
    """

    def __init__(self, output_size: int = 224):
        self.transforms = [
            RandomRotation90(),
            ElasticDeformation(alpha=20.0, sigma=5.0, p=0.5),
            RandomCropPad(crop_ratio=0.85, output_size=output_size, p=0.5),
            RandomBrightnessContrast(brightness=0.3, contrast=0.3, p=0.8),
            RandomGaussianNoise(std=0.02, p=0.3),
        ]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


class _RandomResizedPartial:
    """Random crop of image area in [scale_min, scale_max], resize to output_size.

    Simulates partial captures — FVC2004 frequently crops out the finger
    edge or centers only a fraction of the pad, so the model must match
    off-center ROIs to whole-finger gallery images.
    """

    def __init__(
        self,
        output_size: int = 224,
        scale: Tuple[float, float] = (0.6, 1.0),
        p: float = 0.7,
    ):
        self.output_size = output_size
        self.scale = scale
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return F.interpolate(
                img.unsqueeze(0), size=self.output_size,
                mode='bilinear', align_corners=False,
            ).squeeze(0)
        C, H, W = img.shape
        scale = random.uniform(*self.scale)
        h = max(int(H * (scale ** 0.5)), 1)
        w = max(int(W * (scale ** 0.5)), 1)
        top  = random.randint(0, max(H - h, 0))
        left = random.randint(0, max(W - w, 0))
        img = img[:, top:top + h, left:left + w]
        return F.interpolate(
            img.unsqueeze(0), size=self.output_size,
            mode='bilinear', align_corners=False,
        ).squeeze(0)


class _RandomErasingBig:
    """RandomErasing with up to `max_area` fraction of the image.

    Larger than the default torchvision erasing range — FVC captures can
    have latex occlusion or dirt smudges covering sizeable regions.
    """

    def __init__(self, p: float = 0.35, max_area: float = 0.30):
        self.p = p
        self.max_area = max_area

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        C, H, W = img.shape
        area = random.uniform(0.05, self.max_area) * H * W
        aspect = random.uniform(0.3, 3.3)
        h = int((area * aspect) ** 0.5)
        w = int((area / aspect) ** 0.5)
        if h <= 0 or w <= 0 or h >= H or w >= W:
            return img
        top  = random.randint(0, H - h)
        left = random.randint(0, W - w)
        img = img.clone()
        img[:, top:top + h, left:left + w] = 0.0
        return img


class FingerprintHardTransform:
    """FVC2004-mimicking augmentation for identity training.

    Stronger than FingerprintTrainTransform because the open-set gap on
    FVC2004 is dominated by intra-class distortion that NIST SD302
    training impressions do not reproduce.

      - Stronger elastic deformation (alpha=40, sigma=7): heavy finger
        pressure variation.
      - Partial capture: RandomResizedPartial with scale in [0.6, 1.0].
      - Wider brightness/contrast jitter (0.4 each): sensor difference.
      - Larger RandomErasing (up to 30% of area): latex/dirt occlusion.
    """

    def __init__(self, output_size: int = 224):
        self.transforms = [
            RandomRotation90(),
            ElasticDeformation(alpha=40.0, sigma=7.0, p=0.7),
            _RandomResizedPartial(output_size=output_size, scale=(0.6, 1.0), p=0.7),
            RandomBrightnessContrast(brightness=0.4, contrast=0.4, p=0.9),
            RandomGaussianNoise(std=0.03, p=0.4),
            _RandomErasingBig(p=0.35, max_area=0.30),
        ]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


class FingerprintValTransform:
    """
    Validation/inference transform: only resize, no augmentation.

    Expected input: (1, H, W) grayscale tensor in [0, 1].
    Output:         (1, 224, 224) tensor in [0, 1].
    """

    def __init__(self, output_size: int = 224):
        self.output_size = output_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-1] != self.output_size or img.shape[-2] != self.output_size:
            img = F.interpolate(
                img.unsqueeze(0), size=self.output_size,
                mode='bilinear', align_corners=False
            ).squeeze(0)
        return img


def get_transforms(
    split: str = 'train',
    output_size: int = 224,
    preset: str = 'fingerprint',
):
    """Factory function returning the appropriate transform for a given split.

    Args:
        split:       'train' | 'val' | 'test'
        output_size: spatial size (square)
        preset:      'fingerprint' (default, stock training aug) or
                     'fingerprint_hard' (FVC-mimicking, heavier distortion
                     + partial capture for open-set transfer).

    Only affects training; val/test always returns FingerprintValTransform.
    """
    if split == 'train':
        if preset == 'fingerprint_hard':
            return FingerprintHardTransform(output_size=output_size)
        return FingerprintTrainTransform(output_size=output_size)
    return FingerprintValTransform(output_size=output_size)
