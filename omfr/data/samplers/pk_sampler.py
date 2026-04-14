"""
pk_sampler.py — P×K Sampler for Metric Learning

Samples exactly P identities × K samples per identity per batch.
Required for ArcFace and contrastive losses that need multiple samples
per identity to form positive pairs.

Usage:
    sampler = PKSampler(dataset, P=32, K=4)
    loader  = DataLoader(dataset, batch_sampler=sampler)
    # Each batch: 32 identities × 4 samples = 128 samples
"""

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    Batch sampler: yields batches of exactly P*K indices.

    Each batch contains:
        - P identities drawn without replacement from all available identities
        - K samples per identity drawn with replacement if fewer than K exist

    Ensures every batch has balanced identity representation — critical for
    ArcFace convergence and supervised contrastive learning.

    Args:
        labels:       List[int] — identity label for each dataset sample
        P:            int       — number of identities per batch (e.g., 32)
        K:            int       — samples per identity per batch (e.g., 4)
        drop_last:    bool      — if True, drop the last incomplete batch
        shuffle_ids:  bool      — if True, shuffle identity order each epoch
    """

    def __init__(
        self,
        labels: List[int],
        P: int = 32,
        K: int = 4,
        drop_last: bool = True,
        shuffle_ids: bool = True,
    ):
        super().__init__()
        self.P = P
        self.K = K
        self.drop_last = drop_last
        self.shuffle_ids = shuffle_ids

        # Build index: identity → list of sample indices
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[int(label)].append(idx)

        self.unique_labels: List[int] = sorted(self.label_to_indices.keys())
        self.num_identities = len(self.unique_labels)

        if self.num_identities < P:
            raise ValueError(
                f"PKSampler: dataset has {self.num_identities} identities "
                f"but P={P} was requested."
            )

        # Precompute number of batches
        n_batches = self.num_identities // P
        if not drop_last and (self.num_identities % P) > 0:
            n_batches += 1
        self._len = n_batches

    def __iter__(self) -> Iterator[List[int]]:
        ids = list(self.unique_labels)
        if self.shuffle_ids:
            random.shuffle(ids)

        batch_indices: List[int] = []

        for identity in ids:
            samples = self.label_to_indices[identity]

            if len(samples) >= self.K:
                chosen = random.sample(samples, self.K)
            else:
                # Sample with replacement if fewer than K available
                chosen = random.choices(samples, k=self.K)

            batch_indices.extend(chosen)

            if len(batch_indices) == self.P * self.K:
                yield batch_indices
                batch_indices = []

        # Handle remainder
        if batch_indices and not self.drop_last:
            yield batch_indices

    def __len__(self) -> int:
        return self._len

    @property
    def batch_size(self) -> int:
        return self.P * self.K
