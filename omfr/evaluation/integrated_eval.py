"""
integrated_eval.py — Integrated (Cascaded) Evaluation

Cascaded inference pipeline:
    1. Run PAD model → reject spoofs (score < pad_threshold → spoof)
    2. On the surviving live samples, run identity matching
    3. Report cascaded metrics

Provides:
    compute_cascaded_accuracy(pad_scores, liveness_labels,
                              embeddings, identity_labels,
                              pad_threshold) → dict

Primary checkpoint metric: val/cascaded_IM (maximize)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .matching_eval import compute_tar_at_far, compute_cmc_curve
from .pad_eval import compute_apcer_bpcer_acer, compute_eer


def compute_cascaded_accuracy(
    pad_scores: torch.Tensor,
    liveness_labels: torch.Tensor,
    embeddings: torch.Tensor,
    identity_labels: torch.Tensor,
    pad_threshold: float = 0.5,
    far: float = 1e-4,
    cmc_max_rank: int = 10,
) -> Dict[str, float]:
    """
    Compute cascaded (integrated) accuracy.

    Step 1 — PAD gate:
        Samples with pad_score < pad_threshold are flagged as spoof and rejected.
        Genuine live samples passing the gate → used for identity matching.

    Step 2 — Identity matching on accepted samples:
        TAR@FAR and CMC computed only on samples accepted by the PAD gate.

    Step 3 — Combined metric:
        cascaded_IM = TAR@FAR * (1 - ACER)
        This penalizes both identity matching errors and PAD errors jointly.

    Args:
        pad_scores:       (N,) — P(live) from PAD head, float in [0, 1]
        liveness_labels:  (N,) — 1=live, 0=spoof
        embeddings:       (N, D) — L2-normalized identity embeddings
        identity_labels:  (N,) — identity class labels
        pad_threshold:    PAD decision boundary (default 0.5)
        far:              FAR target for TAR@FAR computation (default 1e-4)
        cmc_max_rank:     maximum rank for CMC curve (default 10)

    Returns:
        dict with keys:
            'APCER', 'BPCER', 'ACER'       — PAD metrics (all samples)
            'EER'                           — PAD EER
            'n_accepted'                    — samples accepted by PAD gate
            'n_total'                       — total samples
            'accept_rate'                   — fraction accepted
            'TAR@FAR'                       — matching TAR@FAR on accepted live
            f'CMC@{k}' for k in 1..max_rank — CMC accuracy at each rank
            'cascaded_IM'                   — primary combined metric
    """
    if isinstance(pad_scores, torch.Tensor):
        pad_np = pad_scores.detach().cpu().numpy()
    else:
        pad_np = np.asarray(pad_scores, dtype=np.float32)

    if isinstance(liveness_labels, torch.Tensor):
        live_np = liveness_labels.detach().cpu().numpy()
    else:
        live_np = np.asarray(liveness_labels, dtype=np.int32)

    # ── Step 1: PAD metrics on all samples ──
    pad_result = compute_apcer_bpcer_acer(pad_np, live_np, threshold=pad_threshold)
    eer, _ = compute_eer(pad_np, live_np)
    pad_result["EER"] = eer

    # ── Step 2: Accept gate (PAD score ≥ threshold → accepted as live) ──
    accepted_mask = pad_np >= pad_threshold
    n_total = len(pad_np)
    n_accepted = int(accepted_mask.sum())

    result: Dict[str, float] = {
        **pad_result,
        "n_accepted": float(n_accepted),
        "n_total": float(n_total),
        "accept_rate": float(n_accepted) / max(n_total, 1),
    }

    # ── Step 3: Matching on accepted live samples ──
    if isinstance(embeddings, torch.Tensor):
        emb_tensor = embeddings.detach().cpu()
    else:
        emb_tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))

    if isinstance(identity_labels, torch.Tensor):
        id_lbl = identity_labels.detach().cpu()
    else:
        id_lbl = torch.from_numpy(np.asarray(identity_labels))

    # Filter to accepted AND truly live samples for matching
    truly_live = torch.from_numpy((live_np == 1) & accepted_mask)
    accepted_emb = emb_tensor[truly_live]
    accepted_ids = id_lbl[truly_live]

    tar = 0.0
    cmc = np.zeros(cmc_max_rank, dtype=np.float64)

    if accepted_emb.shape[0] >= 2:
        tar = compute_tar_at_far(accepted_emb, accepted_ids, far=far)
        cmc = compute_cmc_curve(accepted_emb, accepted_ids, max_rank=cmc_max_rank)

    result[f"TAR@FAR"] = tar
    for k in range(1, cmc_max_rank + 1):
        result[f"CMC@{k}"] = float(cmc[k - 1])

    # ── Combined metric ──
    # cascaded_IM: high matching TAR and low ACER → close to 1.0
    acer = pad_result["ACER"]
    result["cascaded_IM"] = tar * (1.0 - acer)

    return result
