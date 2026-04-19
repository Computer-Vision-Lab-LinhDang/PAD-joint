"""
gabor_stem.py — 8-Channel Learnable Gabor Preprocessing Stem

Input:  (B, 1, H, W) — grayscale fingerprint image, H=W=224
Output: (B, 8, H, W) — 8-channel Gabor responses (one per orientation)

Channels:
    ch 0: θ=0°     (horizontal ridges)
    ch 1: θ=22.5°
    ch 2: θ=45°    (diagonal ridges)
    ch 3: θ=67.5°
    ch 4: θ=90°    (vertical ridges)
    ch 5: θ=112.5°
    ch 6: θ=135°
    ch 7: θ=157.5°

Learnable per filter: frequency σ, bandwidth γ
Fixed per filter: orientation θ (evenly spaced), kernel_size=31
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaborStem(nn.Module):
    """
    8-channel Gabor filter bank for fingerprint ridge/valley enhancement.

    Fixed orientations at 8 evenly-spaced angles in [0, π).
    Learnable frequency (σ) and bandwidth (γ) per filter.
    Output is 8-channel — TinyViT patch_embed adapted for 8 input channels.
    """

    N_ORIENTATIONS = 8

    def __init__(
        self,
        kernel_size: int = 31,
        init_frequency: float = 0.12,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_filters = self.N_ORIENTATIONS

        # Fixed orientations: evenly spaced in [0, π)
        theta = torch.linspace(0, math.pi, self.N_ORIENTATIONS + 1)[:-1]
        self.register_buffer("theta", theta)

        # Learnable frequency (σ) per orientation — init ~5 cycles/mm at 500 DPI
        self.sigma = nn.Parameter(
            torch.full((self.N_ORIENTATIONS,), 1.0 / init_frequency)
        )

        # Learnable bandwidth (γ) per orientation
        self.gamma = nn.Parameter(
            torch.full((self.N_ORIENTATIONS,), 0.5)
        )

        # Fixed base frequency
        self.register_buffer(
            "frequency",
            torch.full((self.N_ORIENTATIONS,), init_frequency),
        )

        # Coordinate grid (built once)
        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", x.contiguous().clone())
        self.register_buffer("grid_y", y.contiguous().clone())

    def _build_kernels(self) -> torch.Tensor:
        """Build Gabor filter bank from current parameters.

        Returns:
            kernels: (8, 1, kernel_size, kernel_size)
        """
        theta = self.theta
        sigma = self.sigma.abs() + 1e-6
        gamma = self.gamma.abs() + 1e-6
        freq = self.frequency

        cos_t = theta.cos().view(-1, 1, 1)
        sin_t = theta.sin().view(-1, 1, 1)

        x = self.grid_x.unsqueeze(0)
        y = self.grid_y.unsqueeze(0)

        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t

        sigma = sigma.view(-1, 1, 1)
        gamma = gamma.view(-1, 1, 1)
        freq = freq.view(-1, 1, 1)

        # Gaussian envelope with anisotropic bandwidth
        gaussian = torch.exp(
            -0.5 * (x_rot ** 2 + (gamma ** 2) * y_rot ** 2) / sigma ** 2
        )
        # Sinusoidal carrier
        carrier = torch.cos(2 * math.pi * freq * x_rot)

        kernels = gaussian * carrier

        # Zero-mean, unit-norm normalization
        kernels = kernels - kernels.mean(dim=(-2, -1), keepdim=True)
        norms = kernels.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        kernels = kernels / norms

        return kernels.unsqueeze(1)  # (8, 1, ks, ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) — grayscale fingerprint

        Returns:
            out: (B, 8, H, W) — one Gabor response per orientation
        """
        assert x.shape[1] == 1, f"Expected 1-channel input, got {x.shape[1]}"

        kernels = self._build_kernels()  # (8, 1, ks, ks)
        pad = self.kernel_size // 2
        return F.conv2d(x, kernels, padding=pad)  # (B, 8, H, W)
