"""
eval.py — OMFR Evaluation Script

Evaluates a trained OMFR checkpoint on:
    1. Identity matching: Rank-1 accuracy + TAR@FAR on FVC2000/2002/2004
    2. PAD: EER, ACER, APCER, BPCER on LivDet2013/2015
    3. Integrated: Cascaded metrics (PAD gate → identity matching)

Usage:
    python eval.py --checkpoint checkpoints/last.ckpt
    python eval.py --checkpoint checkpoints/last.ckpt --fvc_root /path/to/FVC_Dataset
    python eval.py --checkpoint checkpoints/last.ckpt --far 1e-3
    python eval.py --checkpoint checkpoints/last.ckpt --mrl_dims 64 128 256
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from omfr.models.omfr import OMFRModule
from omfr.data.datasets.fvc_dataset import FVCDataset, discover_fvc_dbs
from omfr.data.datasets.pad_dataset import PADDataset
from omfr.evaluation.matching_eval import compute_tar_at_far, compute_cmc_curve
from omfr.evaluation.pad_eval import compute_apcer_bpcer_acer, compute_eer


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_identity_embeddings(
    model: OMFRModule,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract L2-normalized identity embeddings for all samples.

    Returns:
        embeddings: (N, 256) float32 on CPU
        labels:     (N,) long on CPU
    """
    model.eval()
    all_embs = []
    all_labels = []

    for batch in dataloader:
        images = batch["images"].to(device)
        labels = batch["identity_labels"]

        backbone_out = model._run_backbone(images)
        id_out = model._run_identity(backbone_out)

        emb = id_out["identity_embedding"].cpu()  # (B, 256), already L2-normed
        all_embs.append(emb)
        all_labels.append(labels)

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def extract_pad_scores(
    model: OMFRModule,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract PAD liveness scores P(live) for all samples.

    Returns:
        scores: (N,) float32 numpy — P(live) in [0, 1]
        labels: (N,) int32 numpy — 1=live, 0=spoof
        meta:   optional arrays such as sensor/material ids
    """
    model.eval()
    all_scores = []
    all_labels = []
    all_sensors = []
    all_materials = []

    for batch in dataloader:
        images = batch["images"].to(device)
        labels = batch["liveness_labels"]

        backbone_out = model._run_backbone(images, route_mode="pad")
        pad_out = model._run_pad(backbone_out)

        scores = torch.sigmoid(pad_out["pad_logit"].squeeze(-1)).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels.numpy())
        if isinstance(batch, dict) and "sensor_labels" in batch:
            all_sensors.append(batch["sensor_labels"].numpy())
        if isinstance(batch, dict) and "material_labels" in batch:
            all_materials.append(batch["material_labels"].numpy())

    meta: Dict[str, np.ndarray] = {}
    if all_sensors:
        meta["sensor_ids"] = np.concatenate(all_sensors)
    if all_materials:
        meta["material_ids"] = np.concatenate(all_materials)
    return np.concatenate(all_scores), np.concatenate(all_labels), meta


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def eval_identity(
    model: OMFRModule,
    dataloader: DataLoader,
    device: torch.device,
    mrl_dims: List[int],
    far_targets: List[float],
    cmc_max_rank: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate identity matching: Rank-1..10, TAR@FAR at multiple thresholds.
    Evaluates at each MRL dimension.
    """
    embeddings_256, labels = extract_identity_embeddings(model, dataloader, device)

    results = {}
    for dim in mrl_dims:
        emb = embeddings_256[:, :dim]
        emb = F.normalize(emb, p=2, dim=-1)

        # CMC curve → Rank-1..max_rank
        cmc = compute_cmc_curve(emb, labels, max_rank=cmc_max_rank)

        dim_results = {}
        for k in range(1, min(cmc_max_rank + 1, 11)):
            dim_results[f"Rank-{k}"] = float(cmc[k - 1])

        # TAR@FAR at multiple operating points
        for far in far_targets:
            tar = compute_tar_at_far(emb, labels, far=far)
            dim_results[f"TAR@FAR={far}"] = tar

        results[f"{dim}-D"] = dim_results

    return results


def eval_pad(
    model: OMFRModule,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate PAD: EER, ACER, APCER, BPCER."""
    scores, labels, meta = extract_pad_scores(model, dataloader, device)

    eer, eer_thresh = compute_eer(scores, labels)

    # Report APCER/BPCER/ACER at both fixed threshold=0.5 and EER threshold
    metrics_fixed = compute_apcer_bpcer_acer(scores, labels, threshold=0.5)
    metrics_eer = compute_apcer_bpcer_acer(scores, labels, threshold=eer_thresh)

    result: Dict[str, Any] = {
        "EER": eer,
        "EER_threshold": eer_thresh,
        "ACER@0.5": metrics_fixed["ACER"],
        "APCER@0.5": metrics_fixed["APCER"],
        "BPCER@0.5": metrics_fixed["BPCER"],
        "ACER@EER": metrics_eer["ACER"],
        "APCER@EER": metrics_eer["APCER"],
        "BPCER@EER": metrics_eer["BPCER"],
        "num_live": int((labels == 1).sum()),
        "num_spoof": int((labels == 0).sum()),
    }
    if "sensor_ids" in meta:
        result["sensor_breakdown"] = compute_pad_group_breakdown(
            scores,
            labels,
            meta["sensor_ids"],
            list(PADDataset.SENSOR_NAMES),
        )
    if "material_ids" in meta:
        result["spoof_material_breakdown"] = compute_spoof_material_breakdown(
            scores,
            labels,
            meta["material_ids"],
            list(PADDataset.MATERIAL_NAMES),
        )
    if "sensor_ids" in meta and "material_ids" in meta:
        result["sensor_material_breakdown"] = compute_sensor_material_breakdown(
            scores,
            labels,
            meta["sensor_ids"],
            meta["material_ids"],
            list(PADDataset.SENSOR_NAMES),
            list(PADDataset.MATERIAL_NAMES),
        )
    return result


def compute_pad_group_breakdown(
    scores: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    group_names: List[str],
) -> List[Dict[str, Any]]:
    """Compute PAD metrics per group, e.g. per capture sensor."""
    rows: List[Dict[str, Any]] = []
    for group_id in sorted(np.unique(group_ids).astype(int).tolist()):
        mask = group_ids == group_id
        if not mask.any():
            continue
        group_scores = scores[mask]
        group_labels = labels[mask]
        fixed = compute_apcer_bpcer_acer(group_scores, group_labels, threshold=0.5)
        if (group_labels == 1).any() and (group_labels == 0).any():
            group_eer, group_thresh = compute_eer(group_scores, group_labels)
        else:
            group_eer, group_thresh = float("nan"), 0.5
        rows.append({
            "name": group_names[group_id] if 0 <= group_id < len(group_names) else str(group_id),
            "id": group_id,
            "n": int(mask.sum()),
            "num_live": int((group_labels == 1).sum()),
            "num_spoof": int((group_labels == 0).sum()),
            "EER": group_eer,
            "EER_threshold": group_thresh,
            "APCER@0.5": fixed["APCER"],
            "BPCER@0.5": fixed["BPCER"],
            "ACER@0.5": fixed["ACER"],
            "live_score_mean": float(group_scores[group_labels == 1].mean()) if (group_labels == 1).any() else float("nan"),
            "spoof_score_mean": float(group_scores[group_labels == 0].mean()) if (group_labels == 0).any() else float("nan"),
        })
    return rows


def compute_spoof_material_breakdown(
    scores: np.ndarray,
    labels: np.ndarray,
    material_ids: np.ndarray,
    material_names: List[str],
) -> List[Dict[str, Any]]:
    """Report APCER by spoof material at the fixed threshold."""
    rows: List[Dict[str, Any]] = []
    spoof_mask = labels == 0
    for material_id in sorted(np.unique(material_ids[spoof_mask]).astype(int).tolist()):
        mask = spoof_mask & (material_ids == material_id)
        if not mask.any():
            continue
        material_scores = scores[mask]
        rows.append({
            "name": material_names[material_id] if 0 <= material_id < len(material_names) else str(material_id),
            "id": material_id,
            "n": int(mask.sum()),
            "APCER@0.5": float((material_scores >= 0.5).mean()),
            "score_mean": float(material_scores.mean()),
            "score_p90": float(np.percentile(material_scores, 90)),
        })
    rows.sort(key=lambda row: row["APCER@0.5"], reverse=True)
    return rows


def compute_sensor_material_breakdown(
    scores: np.ndarray,
    labels: np.ndarray,
    sensor_ids: np.ndarray,
    material_ids: np.ndarray,
    sensor_names: List[str],
    material_names: List[str],
) -> List[Dict[str, Any]]:
    """Report APCER for each spoof material within each sensor."""
    rows: List[Dict[str, Any]] = []
    spoof_mask = labels == 0
    pairs = sorted({
        (int(sensor_id), int(material_id))
        for sensor_id, material_id in zip(sensor_ids[spoof_mask], material_ids[spoof_mask])
    })
    for sensor_id, material_id in pairs:
        mask = spoof_mask & (sensor_ids == sensor_id) & (material_ids == material_id)
        material_scores = scores[mask]
        rows.append({
            "sensor": sensor_names[sensor_id] if 0 <= sensor_id < len(sensor_names) else str(sensor_id),
            "sensor_id": sensor_id,
            "material": material_names[material_id] if 0 <= material_id < len(material_names) else str(material_id),
            "material_id": material_id,
            "n": int(mask.sum()),
            "APCER@0.5": float((material_scores >= 0.5).mean()),
            "score_mean": float(material_scores.mean()),
            "score_p90": float(np.percentile(material_scores, 90)),
        })
    rows.sort(key=lambda row: row["APCER@0.5"], reverse=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_identity_results(db_name: str, results: Dict[str, Dict]):
    print(f"\n  [{db_name}]")

    for dim_key, metrics in results.items():
        print(f"\n    {dim_key}:")
        # Rank accuracy
        rank_keys = [k for k in metrics if k.startswith("Rank-")]
        if rank_keys:
            rank_line = "    "
            for k in sorted(rank_keys, key=lambda x: int(x.split("-")[1])):
                rank_line += f"  {k}: {metrics[k]*100:.2f}%"
                if k == "Rank-5":
                    print(rank_line)
                    rank_line = "    "
            if rank_line.strip():
                print(rank_line)

        # TAR@FAR
        tar_keys = [k for k in metrics if k.startswith("TAR@FAR")]
        for k in tar_keys:
            print(f"      {k}: {metrics[k]*100:.2f}%")


def print_pad_results(ds_name: str, metrics: Dict[str, Any]):
    print(f"\n  [{ds_name}]  (live={metrics['num_live']}, spoof={metrics['num_spoof']})")
    print(f"    EER:        {metrics['EER']*100:.2f}%  (threshold={metrics['EER_threshold']:.4f})")
    print(f"    @threshold=0.5:")
    print(f"      APCER:  {metrics['APCER@0.5']*100:.2f}%")
    print(f"      BPCER:  {metrics['BPCER@0.5']*100:.2f}%")
    print(f"      ACER:   {metrics['ACER@0.5']*100:.2f}%")
    print(f"    @EER threshold:")
    print(f"      APCER:  {metrics['APCER@EER']*100:.2f}%")
    print(f"      BPCER:  {metrics['BPCER@EER']*100:.2f}%")
    print(f"      ACER:   {metrics['ACER@EER']*100:.2f}%")

    sensor_rows = metrics.get("sensor_breakdown") or []
    if sensor_rows:
        print(f"    By sensor @0.5:")
        for row in sensor_rows:
            print(
                f"      {row['name']:<15}"
                f" EER={row['EER']*100:5.2f}%"
                f" APCER={row['APCER@0.5']*100:5.2f}%"
                f" BPCER={row['BPCER@0.5']*100:5.2f}%"
                f" n={row['n']}"
            )

    material_rows = metrics.get("sensor_material_breakdown") or []
    if material_rows:
        print(f"    Worst spoof sensor/material @0.5:")
        for row in material_rows[:8]:
            print(
                f"      {row['sensor']:<15} {row['material']:<10}"
                f" APCER={row['APCER@0.5']*100:5.2f}%"
                f" mean={row['score_mean']:.4f}"
                f" p90={row['score_p90']:.4f}"
                f" n={row['n']}"
            )


def print_summary_table(
    identity_results: Dict[str, Dict],
    pad_results: Dict[str, Dict],
    mrl_dims: List[int],
):
    """Print a compact summary table across all datasets."""
    print_section("SUMMARY")

    # Identity summary: Rank-1 at 256-D for each DB
    print("\n  Identity Matching (Rank-1 Accuracy):")
    print(f"    {'Dataset':<25}", end="")
    for dim in mrl_dims:
        print(f"  {dim:>5}-D", end="")
    print()
    print(f"    {'-'*25}", end="")
    for _ in mrl_dims:
        print(f"  {'------':>7}", end="")
    print()

    for db_name, db_results in identity_results.items():
        print(f"    {db_name:<25}", end="")
        for dim in mrl_dims:
            dim_key = f"{dim}-D"
            rank1 = db_results.get(dim_key, {}).get("Rank-1", 0)
            print(f"  {rank1*100:6.2f}%", end="")
        print()

    # PAD summary
    if pad_results:
        print(f"\n  PAD Metrics:")
        print(f"    {'Dataset':<25}  {'EER':>7}  {'ACER':>7}  {'APCER':>7}  {'BPCER':>7}")
        print(f"    {'-'*25}  {'------':>7}  {'------':>7}  {'------':>7}  {'------':>7}")
        for ds_name, metrics in pad_results.items():
            print(
                f"    {ds_name:<25}"
                f"  {metrics['EER']*100:6.2f}%"
                f"  {metrics['ACER@EER']*100:6.2f}%"
                f"  {metrics['APCER@EER']*100:6.2f}%"
                f"  {metrics['BPCER@EER']*100:6.2f}%"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OMFR Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--fvc_root", type=str,
        default="/home/linhdang/workspace2/FVC_Dataset",
        help="Root directory containing FVC2000/FVC2002/FVC2004 folders",
    )
    parser.add_argument(
        "--livdet_roots", type=str, nargs="*",
        default=None,
        help="LivDet dataset roots. Auto-detected from project dir if not set.",
    )
    parser.add_argument(
        "--mrl_dims", type=int, nargs="+", default=[64, 128, 256],
        help="MRL dimensions to evaluate",
    )
    parser.add_argument(
        "--far", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4],
        help="FAR targets for TAR@FAR computation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detect if not set)",
    )
    parser.add_argument(
        "--cmc_max_rank", type=int, default=10,
        help="Max rank for CMC curve",
    )
    parser.add_argument(
        "--skip_identity", action="store_true",
        help="Skip identity evaluation",
    )
    parser.add_argument(
        "--skip_pad", action="store_true",
        help="Skip PAD evaluation",
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Input image size",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = OMFRModule.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        strict=False,
    )
    model = model.to(device)
    model.eval()
    print("Model loaded.\n")

    all_identity_results: Dict[str, Dict] = OrderedDict()
    all_pad_results: Dict[str, Dict] = OrderedDict()

    # ── Identity evaluation on FVC datasets ──────────────────────────────
    if not args.skip_identity:
        fvc_root = Path(args.fvc_root)
        if fvc_root.exists():
            fvc_dbs = discover_fvc_dbs(str(fvc_root))
            if fvc_dbs:
                print_section("IDENTITY MATCHING EVALUATION")
                print(f"  Found {len(fvc_dbs)} FVC DBs")

                for db_name, db_path in fvc_dbs:
                    print(f"\n  Evaluating {db_name} ({db_path})...")
                    dataset = FVCDataset(
                        root=db_path,
                        image_size=args.image_size,
                        dataset_name=db_name,
                    )
                    print(f"    Samples: {len(dataset)}, Subjects: {dataset.num_classes}")

                    loader = DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )

                    results = eval_identity(
                        model, loader, device,
                        mrl_dims=args.mrl_dims,
                        far_targets=args.far,
                        cmc_max_rank=args.cmc_max_rank,
                    )
                    all_identity_results[db_name] = results
                    print_identity_results(db_name, results)
            else:
                print(f"No FVC DBs found under {fvc_root}")
        else:
            print(f"FVC root not found: {fvc_root}")

    # ── PAD evaluation on LivDet datasets ────────────────────────────────
    if not args.skip_pad:
        # Auto-detect LivDet directories
        livdet_roots = args.livdet_roots
        if livdet_roots is None:
            project_dir = Path(__file__).parent
            livdet_roots = []
            for candidate in sorted(project_dir.iterdir()):
                if candidate.is_dir() and "livdet" in candidate.name.lower():
                    livdet_roots.append(str(candidate))

        if livdet_roots:
            print_section("PAD EVALUATION")
            print(f"  Found {len(livdet_roots)} LivDet datasets")

            for livdet_root in livdet_roots:
                ds_name = Path(livdet_root).name
                print(f"\n  Evaluating {ds_name} ({livdet_root})...")

                try:
                    dataset = PADDataset(
                        root=livdet_root,
                        split="test",
                        image_size=args.image_size,
                        dataset_name=ds_name,
                    )
                    print(f"    Samples: {len(dataset)} (live={dataset.num_live}, spoof={dataset.num_spoof})")

                    loader = DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )

                    metrics = eval_pad(model, loader, device)
                    all_pad_results[ds_name] = metrics
                    print_pad_results(ds_name, metrics)

                except Exception as e:
                    print(f"    ERROR: {e}")
        else:
            print("No LivDet datasets found.")

    # ── Summary table ────────────────────────────────────────────────────
    if all_identity_results or all_pad_results:
        print_summary_table(all_identity_results, all_pad_results, args.mrl_dims)

    print(f"\n{'=' * 70}")
    print("  Evaluation complete.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
