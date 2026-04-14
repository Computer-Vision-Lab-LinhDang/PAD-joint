"""
gabor_stem.py — Learnable Gabor Preprocessing Stem

Input:  (B, 1, H, W) — grayscale fingerprint image, H=W=224
Output: (B, 3, H, W) — 3-channel enhanced image [original, gabor_ridge, gabor_valley]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaborStem(nn.Module):
    """
    Learnable Gabor filter bank for ridge/valley enhancement.

    Outputs 3 channels: [original, gabor_ridge, gabor_valley] so that
    ViT-Tiny can receive 3-channel input compatible with ImageNet pretrained weights.

    Parameters:
        n_orientations (int): Number of Gabor orientations. Default: 8
        n_frequencies  (int): Number of Gabor frequencies. Default: 4
        kernel_size    (int): Spatial kernel size (odd). Default: 31
        learnable      (bool): Allow learning freq/orientation parameters. Default: True

    Learnable params: frequency (sigma), orientation (theta) per filter
    Fixed params: kernel_size, spatial envelope

    Forward:
        1. gabor_bank(image) → (B, n_ori * n_freq, H, W)
        2. channel_attention  → scores per response (B, n_ori * n_freq)
        3. select top-2 responses
        4. concat [original, top1, top2] → (B, 3, H, W)

    I/O:
        Input:  (B, 1, H, W)
        Output: (B, 3, H, W)
    """

    def __init__(
        self,
        n_orientations: int = 8,
        n_frequencies: int = 4,
        kernel_size: int = 31,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.n_orientations = n_orientations
        self.n_frequencies = n_frequencies
        self.kernel_size = kernel_size
        self.n_filters = n_orientations * n_frequencies

        # Learnable Gabor parameters
        # orientations: (n_orientations,)  — radians
        theta_init = torch.linspace(0, math.pi, n_orientations + 1)[:-1]
        # frequencies: (n_frequencies,)    — cycles per pixel
        freq_init = torch.linspace(0.05, 0.5, n_frequencies)

        if learnable:
            self.theta = nn.Parameter(theta_init.repeat(n_frequencies))
            self.sigma = nn.Parameter(
                (1.0 / freq_init).repeat_interleave(n_orientations)
            )
        else:
            self.register_buffer("theta", theta_init.repeat(n_frequencies))
            self.register_buffer(
                "sigma", (1.0 / freq_init).repeat_interleave(n_orientations)
            )

        # Frequency values, repeated for each orientation
        freq_repeated = freq_init.repeat_interleave(n_orientations)
        self.register_buffer("frequency", freq_repeated)

        # Channel attention: squeeze-and-excite to score each Gabor response
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # (B, n_filters, 1, 1)
            nn.Flatten(),                       # (B, n_filters)
            nn.Linear(self.n_filters, self.n_filters // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_filters // 2, self.n_filters),
            nn.Sigmoid(),
        )

        # Build coordinate grid once (used in forward)
        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", x)  # (ks, ks)
        self.register_buffer("grid_y", y)  # (ks, ks)

    def _build_gabor_kernels(self) -> torch.Tensor:
        """
        Construct Gabor kernels from current (possibly learned) parameters.

        Returns:
            kernels: (n_filters, 1, kernel_size, kernel_size)
        """
        theta = self.theta        # (n_filters,)
        sigma = self.sigma.abs() + 1e-6  # (n_filters,)
        freq = self.frequency     # (n_filters,)

        # Rotated coordinates per filter
        # cos_t, sin_t: (n_filters, 1, 1)
        cos_t = theta.cos().view(-1, 1, 1)
        sin_t = theta.sin().view(-1, 1, 1)

        x = self.grid_x.unsqueeze(0)  # (1, ks, ks)
        y = self.grid_y.unsqueeze(0)  # (1, ks, ks)

        x_rot = x * cos_t + y * sin_t   # (n_filters, ks, ks)
        y_rot = -x * sin_t + y * cos_t  # (n_filters, ks, ks)

        sigma = sigma.view(-1, 1, 1)
        freq = freq.view(-1, 1, 1)

        # Gaussian envelope
        gaussian = torch.exp(-0.5 * (x_rot ** 2 + y_rot ** 2) / sigma ** 2)
        # Sinusoidal carrier
        carrier = torch.cos(2 * math.pi * freq * x_rot)

        kernels = gaussian * carrier  # (n_filters, ks, ks)

        # Normalize each kernel to zero mean, unit norm
        kernels = kernels - kernels.mean(dim=(-2, -1), keepdim=True)
        norms = kernels.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        kernels = kernels / norms

        return kernels.unsqueeze(1)  # (n_filters, 1, ks, ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) — grayscale fingerprint

        Returns:
            out: (B, 3, H, W) — [original, top1_gabor, top2_gabor]
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected 1-channel input, got {C}"

        # Build kernels from current parameters
        kernels = self._build_gabor_kernels()  # (n_filters, 1, ks, ks)

        pad = self.kernel_size // 2
        # Apply all Gabor filters at once via grouped conv
        responses = F.conv2d(x, kernels, padding=pad)  # (B, n_filters, H, W)

        # Channel attention scores
        attn_scores = self.channel_attention(responses)  # (B, n_filters)

        # Select top-2 filters (by attention score) per sample
        top2_idx = attn_scores.topk(k=2, dim=1).indices  # (B, 2)

        # Gather top-2 responses: (B, 2, H, W)
        responses_flat = responses.view(B, self.n_filters, H * W)
        idx_exp = top2_idx.unsqueeze(-1).expand(-1, -1, H * W)  # (B, 2, H*W)
        top2_responses = responses_flat.gather(dim=1, index=idx_exp)
        top2_responses = top2_responses.view(B, 2, H, W)

        # Concatenate: [original, top1, top2]
        out = torch.cat([x, top2_responses], dim=1)  # (B, 3, H, W)
        return out
