from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from omfr.data.datasets.pad_dataset import PADDataset
from omfr.evaluation.pad_eval import compute_apcer_bpcer_acer, compute_eer
from omfr.models.omfr import OMFRModule


BASE_PAD_POLICY = {
    "backbone_grad": True,
    "backbone_input": "pad_gabor",
    "detach_backbone_features": False,
    "detach_routing_stats": True,
}

VARIANTS = {
    "base": {},
    "backbone_identity": {"backbone_input": "identity_gabor"},
    "stem_identity": {"pad_stem_input": "identity_gabor"},
    "all_identity": {
        "backbone_input": "identity_gabor",
        "pad_stem_input": "identity_gabor",
    },
    "lowpass5": {"pad_gabor_blur_kernel": 5},
}


def balanced_subset(dataset: PADDataset, max_per_class: int, seed: int) -> Subset:
    labels = np.asarray(dataset.get_liveness_labels())
    rng = np.random.default_rng(seed)
    indices = []
    for label in (0, 1):
        label_indices = np.flatnonzero(labels == label)
        rng.shuffle(label_indices)
        indices.extend(label_indices[:max_per_class].tolist())
    rng.shuffle(indices)
    return Subset(dataset, indices)


@torch.no_grad()
def collect_scores(
    model: OMFRModule,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    scores = []
    labels = []
    gabor_energy = []

    model.eval()
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        label = batch["liveness_labels"].numpy().astype(np.int32)

        backbone_out, pad_out = model._run_pad_branch(
            images,
            use_phase_policy=True,
            backbone_no_grad=True,
        )
        score = torch.sigmoid(pad_out["pad_logit"].squeeze(-1))
        energy = backbone_out["gabor_pad_feat"].float().square().mean(dim=(1, 2, 3)).sqrt()

        scores.append(score.cpu().numpy())
        labels.append(label)
        gabor_energy.append(energy.cpu().numpy())

    return {
        "scores": np.concatenate(scores),
        "labels": np.concatenate(labels),
        "gabor_energy": np.concatenate(gabor_energy),
    }


def summarize_arrays(scores: np.ndarray, labels: np.ndarray, energy: np.ndarray) -> dict[str, Any]:
    eer, eer_threshold = compute_eer(scores, labels)
    fixed = compute_apcer_bpcer_acer(scores, labels, threshold=0.5)
    eer_metrics = compute_apcer_bpcer_acer(scores, labels, threshold=eer_threshold)

    live = labels == 1
    spoof = labels == 0
    live_reject = live & (scores < 0.5)
    spoof_accept = spoof & (scores >= 0.5)

    def stats(mask: np.ndarray, values: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        subset = values[mask]
        return {
            "mean": float(np.mean(subset)),
            "p10": float(np.quantile(subset, 0.10)),
            "p50": float(np.quantile(subset, 0.50)),
            "p90": float(np.quantile(subset, 0.90)),
        }

    return {
        "n_live": int(np.sum(live)),
        "n_spoof": int(np.sum(spoof)),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "acer_0p5": float(fixed["ACER"]),
        "apcer_0p5": float(fixed["APCER"]),
        "bpcer_0p5": float(fixed["BPCER"]),
        "acer_eer": float(eer_metrics["ACER"]),
        "apcer_eer": float(eer_metrics["APCER"]),
        "bpcer_eer": float(eer_metrics["BPCER"]),
        "score_live": stats(live, scores),
        "score_spoof": stats(spoof, scores),
        "energy_live": stats(live, energy),
        "energy_spoof": stats(spoof, energy),
        "energy_live_rejected_0p5": stats(live_reject, energy),
        "energy_spoof_accepted_0p5": stats(spoof_accept, energy),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--livdet-roots", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-per-class", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/pad_gabor_diagnostic.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = OMFRModule.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    ).to(device)
    model.current_phase = 2

    datasets: list[tuple[str, DataLoader]] = []
    for root in args.livdet_roots:
        root_path = Path(root)
        dataset = PADDataset(root=root, split="test", image_size=224, dataset_name=root_path.name)
        subset = balanced_subset(dataset, max_per_class=args.max_per_class, seed=args.seed)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        datasets.append((root_path.name, loader))

    results: dict[str, Any] = {}
    for variant_name, overrides in VARIANTS.items():
        print(f"\n=== {variant_name} ===", flush=True)
        model.pad_stage2_cfg = {**BASE_PAD_POLICY, **overrides}
        variant_results: dict[str, Any] = {}
        all_scores = []
        all_labels = []
        all_energy = []

        for dataset_name, loader in datasets:
            collected = collect_scores(model, loader, device)
            summary = summarize_arrays(
                collected["scores"],
                collected["labels"],
                collected["gabor_energy"],
            )
            variant_results[dataset_name] = summary
            all_scores.append(collected["scores"])
            all_labels.append(collected["labels"])
            all_energy.append(collected["gabor_energy"])
            print(
                f"{dataset_name}: EER={summary['eer']*100:.2f}% "
                f"thr={summary['eer_threshold']:.4f} "
                f"ACER@0.5={summary['acer_0p5']*100:.2f}% "
                f"APCER@0.5={summary['apcer_0p5']*100:.2f}% "
                f"BPCER@0.5={summary['bpcer_0p5']*100:.2f}%",
                flush=True,
            )

        combined = summarize_arrays(
            np.concatenate(all_scores),
            np.concatenate(all_labels),
            np.concatenate(all_energy),
        )
        variant_results["combined"] = combined
        results[variant_name] = variant_results
        print(
            f"combined: EER={combined['eer']*100:.2f}% "
            f"thr={combined['eer_threshold']:.4f} "
            f"ACER@0.5={combined['acer_0p5']*100:.2f}% "
            f"APCER@0.5={combined['apcer_0p5']*100:.2f}% "
            f"BPCER@0.5={combined['bpcer_0p5']*100:.2f}%",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
