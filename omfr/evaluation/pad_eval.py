"""
pad_eval.py — Presentation Attack Detection Evaluation Metrics

Provides:
    compute_apcer_bpcer_acer(scores, labels, threshold) → dict
        APCER: Attack Presentation Classification Error Rate
               = FP / (FP + TN)  — fraction of spoofs classified as live
        BPCER: Bonafide Presentation Classification Error Rate
               = FN / (FN + TP)  — fraction of live classified as spoof
        ACER:  (APCER + BPCER) / 2

    compute_eer(scores, labels) → (eer, threshold)
        Equal Error Rate: operating point where APCER == BPCER.

    PADMetrics
        Stateful accumulator for computing metrics across validation batches.

Convention: scores are P(live) ∈ [0, 1], labels are 1=live, 0=spoof.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def compute_apcer_bpcer_acer(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute APCER, BPCER, ACER at a given decision threshold.

    Args:
        scores:    (N,) — liveness probability P(live) in [0, 1]
        labels:    (N,) — ground-truth labels, 1=live, 0=spoof
        threshold: decision boundary (predict live if score ≥ threshold)

    Returns:
        dict with keys 'APCER', 'BPCER', 'ACER'
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    predictions = (scores >= threshold).astype(np.int32)

    live_mask  = labels == 1
    spoof_mask = labels == 0

    n_live  = live_mask.sum()
    n_spoof = spoof_mask.sum()

    # APCER: fraction of spoofs falsely accepted (predicted as live)
    apcer = float(predictions[spoof_mask].sum()) / max(n_spoof, 1)

    # BPCER: fraction of live falsely rejected (predicted as spoof)
    bpcer = float((1 - predictions[live_mask]).sum()) / max(n_live, 1)

    acer = (apcer + bpcer) / 2.0

    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer}


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.

    EER is the operating point where APCER ≈ BPCER.

    Args:
        scores: (N,) — liveness probability P(live) in [0, 1]
        labels: (N,) — ground-truth labels, 1=live, 0=spoof

    Returns:
        (eer, threshold): both floats
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    # Unique thresholds to sweep
    thresholds = np.unique(scores)

    best_eer = 1.0
    best_thresh = 0.5

    for t in thresholds:
        result = compute_apcer_bpcer_acer(scores, labels, threshold=float(t))
        apcer = result["APCER"]
        bpcer = result["BPCER"]
        gap = abs(apcer - bpcer)
        eer_candidate = (apcer + bpcer) / 2.0

        if gap < abs(best_eer - eer_candidate) or gap < 1e-4:
            if eer_candidate <= best_eer:
                best_eer = eer_candidate
                best_thresh = float(t)

    return best_eer, best_thresh


class PADMetrics:
    """
    Stateful accumulator for PAD metrics across validation batches.

    Usage::

        metrics = PADMetrics()
        for batch in val_loader:
            scores = model(batch)  # P(live)
            metrics.update(scores, labels)
        result = metrics.compute()  # {'APCER', 'BPCER', 'ACER', 'EER'}
        metrics.reset()
    """

    def __init__(self) -> None:
        self._scores: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Accumulate batch predictions.

        Args:
            scores: (B,) — P(live) in [0, 1], float tensor
            labels: (B,) — 1=live, 0=spoof, long or int tensor
        """
        self._scores.append(scores.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute APCER, BPCER, ACER, and EER over all accumulated data.

        Args:
            threshold: decision threshold for APCER/BPCER/ACER (default 0.5)

        Returns:
            dict with keys 'APCER', 'BPCER', 'ACER', 'EER'
        """
        if not self._scores:
            return {"APCER": 0.0, "BPCER": 0.0, "ACER": 0.0, "EER": 0.0}

        all_scores = torch.cat(self._scores).numpy()
        all_labels = torch.cat(self._labels).numpy()

        result = compute_apcer_bpcer_acer(all_scores, all_labels, threshold)
        eer, _ = compute_eer(all_scores, all_labels)
        result["EER"] = eer
        return result

    def reset(self) -> None:
        """Clear accumulated scores and labels."""
        self._scores.clear()
        self._labels.clear()
