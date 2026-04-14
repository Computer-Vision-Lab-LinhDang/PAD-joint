"""
balanced_pad_sampler.py — Balanced PAD Sampler (50/50 live/spoof)

Ensures each PAD batch contains exactly 50% live and 50% spoof samples.
Required because real-world PAD datasets are heavily imbalanced (typically
far more live than spoof samples). Imbalanced batches cause BCE loss to
degenerate toward predicting "live" for everything.

Usage:
    sampler = BalancedPADSampler(liveness_labels, batch_size=64)
    loader  = DataLoader(dataset, batch_sampler=sampler)
    # Each batch: 32 live + 32 spoof
"""

import random
from typing import Dict, Iterator, List, Optional

from torch.utils.data import Sampler


class BalancedPADSampler(Sampler):
    """
    Batch sampler that yields 50/50 live/spoof batches.

    Each batch contains batch_size // 2 live samples and
    batch_size // 2 spoof samples, drawn without replacement per epoch
    (with cycling when one class is exhausted before the other).

    Args:
        liveness_labels: List[int]  — 0=spoof, 1=live for each dataset sample
        batch_size:      int        — total batch size (must be even)
        drop_last:       bool       — drop last batch if incomplete
        shuffle:         bool       — shuffle indices each epoch
    """

    LIVE_LABEL = 1
    SPOOF_LABEL = 0

    def __init__(
        self,
        liveness_labels: List[int],
        batch_size: int = 64,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        super().__init__()

        if batch_size % 2 != 0:
            raise ValueError(
                f"BalancedPADSampler: batch_size must be even, got {batch_size}"
            )

        self.batch_size = batch_size
        self.half = batch_size // 2
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Split indices by class
        self.live_indices: List[int] = [
            i for i, l in enumerate(liveness_labels) if int(l) == self.LIVE_LABEL
        ]
        self.spoof_indices: List[int] = [
            i for i, l in enumerate(liveness_labels) if int(l) == self.SPOOF_LABEL
        ]

        if not self.live_indices:
            raise ValueError("BalancedPADSampler: no live samples found in dataset.")
        if not self.spoof_indices:
            raise ValueError("BalancedPADSampler: no spoof samples found in dataset.")

        # Number of batches per epoch: determined by the majority class
        # (minority class will be resampled with replacement if needed)
        n_majority = max(len(self.live_indices), len(self.spoof_indices))
        n_batches = n_majority // self.half
        if not drop_last and (n_majority % self.half) > 0:
            n_batches += 1
        self._len = n_batches

    def _sample_class(self, indices: List[int], n: int) -> List[int]:
        """
        Draw n samples from indices without replacement.
        If len(indices) < n, cycle through shuffled indices.
        """
        if n <= len(indices):
            return random.sample(indices, n)

        # Need more samples than available: cycle
        result: List[int] = []
        pool = list(indices)
        while len(result) < n:
            if self.shuffle:
                random.shuffle(pool)
            needed = n - len(result)
            result.extend(pool[:needed])
        return result

    def __iter__(self) -> Iterator[List[int]]:
        live_pool = list(self.live_indices)
        spoof_pool = list(self.spoof_indices)

        if self.shuffle:
            random.shuffle(live_pool)
            random.shuffle(spoof_pool)

        # Extend shorter class to match longer (cycling)
        n_majority = max(len(live_pool), len(spoof_pool))
        n_needed = ((n_majority + self.half - 1) // self.half) * self.half

        live_extended  = self._extend_pool(live_pool,  n_needed)
        spoof_extended = self._extend_pool(spoof_pool, n_needed)

        for start in range(0, n_needed, self.half):
            end = start + self.half

            live_batch  = live_extended[start:end]
            spoof_batch = spoof_extended[start:end]

            if len(live_batch) < self.half or len(spoof_batch) < self.half:
                if self.drop_last:
                    break
                # Pad incomplete batch
                live_batch  = live_batch  + random.choices(live_pool,  k=self.half - len(live_batch))
                spoof_batch = spoof_batch + random.choices(spoof_pool, k=self.half - len(spoof_batch))

            combined = live_batch + spoof_batch
            random.shuffle(combined)  # Shuffle so live/spoof aren't contiguous
            yield combined

    def _extend_pool(self, pool: List[int], target_len: int) -> List[int]:
        """Extend pool to target_len by cycling (with optional shuffle per cycle)."""
        result = list(pool)
        while len(result) < target_len:
            extra = list(pool)
            if self.shuffle:
                random.shuffle(extra)
            result.extend(extra)
        return result[:target_len]

    def __len__(self) -> int:
        return self._len

    @property
    def num_live(self) -> int:
        return len(self.live_indices)

    @property
    def num_spoof(self) -> int:
        return len(self.spoof_indices)

    @property
    def imbalance_ratio(self) -> float:
        """live / spoof ratio before balancing."""
        return self.num_live / max(self.num_spoof, 1)
