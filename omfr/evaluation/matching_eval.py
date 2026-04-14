"""
matching_eval.py — Fingerprint Matching Evaluation Metrics

Provides:
    compute_tar_at_far(embeddings, labels, far) → float
        TAR (True Accept Rate) at a specified FAR (False Accept Rate).
        Used to evaluate matching at each MRL dimension.

    compute_cmc_curve(embeddings, labels, max_rank) → np.ndarray
        Cumulative Match Characteristic curve.

All pairwise similarity computations are done with cosine similarity
(embeddings assumed L2-normalized going in).
"""

from __future__ import annotations

import numpy as np
import torch


def _cosine_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """
    Compute all-pairs cosine similarity matrix.

    Args:
        embeddings: (N, D) — L2-normalized float tensor

    Returns:
        sim_matrix: (N, N) np.ndarray, values in [-1, 1]
    """
    # Embeddings should already be normalized; re-normalize defensively
    emb = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    sim = (emb @ emb.T).cpu().numpy()
    return sim


def compute_tar_at_far(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    far: float = 1e-4,
) -> float:
    """
    Compute TAR (True Accept Rate) at the given FAR (False Accept Rate).

    Procedure:
        1. Compute all genuine pairs (same identity, i ≠ j) → genuine scores.
        2. Compute all impostor pairs (different identity, upper-triangle) → impostor scores.
        3. Find threshold T such that FAR(T) ≤ target FAR.
        4. TAR = fraction of genuine pairs with score ≥ T.

    Args:
        embeddings: (N, D) — L2-normalized feature vectors
        labels:     (N,)  — identity class labels (LongTensor or ndarray)
        far:        target False Accept Rate (default 1e-4 = 0.01%)

    Returns:
        tar: float in [0, 1]
    """
    N = embeddings.shape[0]
    sim = _cosine_similarity_matrix(embeddings)

    if isinstance(labels, torch.Tensor):
        lbl = labels.cpu().numpy()
    else:
        lbl = np.asarray(labels)

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    for i in range(N):
        for j in range(i + 1, N):
            score = float(sim[i, j])
            if lbl[i] == lbl[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    if not impostor_scores or not genuine_scores:
        return 0.0

    genuine_arr = np.array(genuine_scores, dtype=np.float32)
    impostor_arr = np.array(impostor_scores, dtype=np.float32)

    # Sweep thresholds from high to low; find where FAR ≤ target
    thresholds = np.sort(impostor_arr)[::-1]  # descending

    best_tar = 0.0
    for t in thresholds:
        current_far = float((impostor_arr >= t).mean())
        if current_far <= far:
            best_tar = float((genuine_arr >= t).mean())
            break

    return best_tar


def compute_cmc_curve(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    max_rank: int = 20,
) -> np.ndarray:
    """
    Compute Cumulative Match Characteristic (CMC) curve.

    For each probe, the gallery is all other samples. The probe is
    considered a match at rank-k if the true mate appears in the
    top-k retrieved results.

    Args:
        embeddings: (N, D) — L2-normalized feature vectors
        labels:     (N,)  — identity labels
        max_rank:   maximum CMC rank to compute (default 20)

    Returns:
        cmc: np.ndarray of shape (max_rank,), cmc[k-1] = rank-k accuracy
    """
    N = embeddings.shape[0]
    sim = _cosine_similarity_matrix(embeddings)

    if isinstance(labels, torch.Tensor):
        lbl = labels.cpu().numpy()
    else:
        lbl = np.asarray(labels)

    np.fill_diagonal(sim, -2.0)  # exclude self from retrieval

    cmc = np.zeros(max_rank, dtype=np.float64)
    num_probes = 0

    for i in range(N):
        # Find if there is at least one genuine mate
        genuine_exists = np.any(lbl == lbl[i]) and (lbl == lbl[i]).sum() > 1
        if not genuine_exists:
            continue

        num_probes += 1
        ranked_indices = np.argsort(sim[i])[::-1]  # descending similarity

        for rank in range(min(max_rank, N - 1)):
            j = ranked_indices[rank]
            if lbl[j] == lbl[i]:
                cmc[rank:] += 1.0
                break

    if num_probes > 0:
        cmc /= num_probes

    return cmc
