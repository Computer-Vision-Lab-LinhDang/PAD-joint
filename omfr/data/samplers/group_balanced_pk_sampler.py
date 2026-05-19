"""
group_balanced_pk_sampler.py — Group-balanced P×K sampler.

Data-centric fix for the NIST/FVC class imbalance. The identity corpus
has ~70× more NIST classes than FVC classes, so a plain ``PKSampler``
fills almost every batch with NIST identities and the 24M backbone
overfits NIST while ignoring FVC.

``GroupBalancedPKSampler`` guarantees a fixed FVC:NIST *class* ratio in
EVERY batch (default 50/50) regardless of the global class counts:

    P identities/batch  ->  P_fvc = round(P * fvc_ratio) from the FVC
                            class pool, P_nist = P - P_fvc from NIST.
    K samples/identity  ->  same as PKSampler (with replacement if a
                            class has < K images).

Each group is drawn from an independently-shuffled cursor that reshuffles
when exhausted, so the minority group (FVC) is resampled with replacement
within an epoch while the majority (NIST) is covered roughly once. Epoch
length is tied to whichever group needs the most batches for one full
pass, so NIST volume still sets the training duration.

Falls back to plain P×K behaviour when only one group is present.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Sequence

from torch.utils.data import Sampler


class GroupBalancedPKSampler(Sampler):
    """Batch sampler yielding P*K indices with a fixed per-batch group ratio.

    Args:
        labels:     per-sample identity class index.
        groups:     per-sample group tag (parallel to ``labels``);
                    expected values ``"fvc"`` / ``"nist"`` but any two
                    tags work.
        P:          identities per batch.
        K:          samples per identity per batch.
        fvc_ratio:  fraction of the P identities drawn from the "fvc"
                    group (or, if no "fvc" tag exists, from the
                    lexicographically-first group). Default 0.5 (50/50).
        shuffle:    reshuffle identity order every epoch.
    """

    def __init__(
        self,
        labels: Sequence[int],
        groups: Sequence[str],
        P: int = 32,
        K: int = 2,
        fvc_ratio: float = 0.5,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        if len(labels) != len(groups):
            raise ValueError(
                f"labels ({len(labels)}) and groups ({len(groups)}) "
                f"must be the same length."
            )
        self.P = int(P)
        self.K = int(K)
        self.shuffle = shuffle

        # class idx -> sample indices
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        # group tag -> set of class indices in that group
        group_to_labels: Dict[str, set] = defaultdict(set)
        for idx, (label, grp) in enumerate(zip(labels, groups)):
            self.label_to_indices[int(label)].append(idx)
            group_to_labels[str(grp)].add(int(label))

        self.group_labels: Dict[str, List[int]] = {
            g: sorted(lbls) for g, lbls in group_to_labels.items()
        }
        self.present_groups = sorted(self.group_labels.keys())

        if len(self.present_groups) <= 1:
            # Degenerate: behave like a plain P×K sampler.
            self._single_group = True
            only = self.present_groups[0] if self.present_groups else None
            self._all_labels = (
                self.group_labels.get(only, [])
                if only is not None
                else sorted(self.label_to_indices.keys())
            )
            if len(self._all_labels) < self.P:
                raise ValueError(
                    f"GroupBalancedPKSampler: {len(self._all_labels)} "
                    f"classes < P={self.P}."
                )
            self._len = max(len(self._all_labels) // self.P, 1)
            return

        self._single_group = False
        # Primary group = "fvc" if present, else the first tag.
        primary = "fvc" if "fvc" in self.group_labels else self.present_groups[0]
        secondary = next(g for g in self.present_groups if g != primary)
        self.primary, self.secondary = primary, secondary

        self.P_primary = max(1, min(self.P - 1, round(self.P * float(fvc_ratio))))
        self.P_secondary = self.P - self.P_primary

        n_prim = len(self.group_labels[primary])
        n_sec = len(self.group_labels[secondary])
        if n_prim == 0 or n_sec == 0:
            raise ValueError("Both groups must contain at least one class.")

        # Epoch length: enough batches for the group that needs the most
        # full-coverage passes (NIST volume drives duration).
        batches_prim = -(-n_prim // self.P_primary)      # ceil div
        batches_sec = -(-n_sec // self.P_secondary)
        self._len = max(batches_prim, batches_sec, 1)

    # ------------------------------------------------------------------

    def _cursor(self, pool: List[int]) -> Iterator[int]:
        """Infinite class-id stream over ``pool``; reshuffles each pass."""
        while True:
            order = list(pool)
            if self.shuffle:
                random.shuffle(order)
            for cid in order:
                yield cid

    def _pick_samples(self, class_id: int) -> List[int]:
        pool = self.label_to_indices[class_id]
        if len(pool) >= self.K:
            return random.sample(pool, self.K)
        return random.choices(pool, k=self.K)

    def __iter__(self) -> Iterator[List[int]]:
        if self._single_group:
            ids = list(self._all_labels)
            if self.shuffle:
                random.shuffle(ids)
            batch: List[int] = []
            for cid in ids:
                batch.extend(self._pick_samples(cid))
                if len(batch) == self.P * self.K:
                    yield batch
                    batch = []
            return

        prim = self._cursor(self.group_labels[self.primary])
        sec = self._cursor(self.group_labels[self.secondary])
        for _ in range(self._len):
            batch: List[int] = []
            for _ in range(self.P_primary):
                batch.extend(self._pick_samples(next(prim)))
            for _ in range(self.P_secondary):
                batch.extend(self._pick_samples(next(sec)))
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._len

    @property
    def batch_size(self) -> int:
        return self.P * self.K
