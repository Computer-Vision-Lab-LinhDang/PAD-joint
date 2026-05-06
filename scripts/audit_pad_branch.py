#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omfr.data.datamodule import OMFRDataModule
from omfr.models.omfr import OMFRModule
from train import _prepare_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PAD/frequency-gate gradients and raw routing cues on one phase-2 batch.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/linhdang/workspace2/PAD-joint/configs/base.yaml",
    )
    parser.add_argument("--identity-root", type=str, default=None)
    parser.add_argument("--pad-root", type=str, default=None)
    parser.add_argument(
        "--pad-datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names, e.g. livdet2013,livdet2015",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pk-p", type=int, default=None)
    parser.add_argument("--pk-k", type=int, default=None)
    parser.add_argument("--pad-batch-size", type=int, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.setdefault("data", {})
    if args.identity_root:
        data_cfg["identity_data_root"] = args.identity_root
    if args.pad_root:
        data_cfg["pad_data_root"] = args.pad_root
    if args.pad_datasets:
        data_cfg["pad_datasets"] = [
            item.strip() for item in args.pad_datasets.split(",") if item.strip()
        ]
    data_cfg["num_workers"] = args.num_workers
    data_cfg["pin_memory"] = False
    if args.pk_p is not None:
        data_cfg["pk_p"] = args.pk_p
    if args.pk_k is not None:
        data_cfg["pk_k"] = args.pk_k
    if args.pad_batch_size is not None:
        data_cfg["pad_batch_size"] = args.pad_batch_size
    return cfg


def build_runtime_config(cfg: Dict, num_classes: int) -> Dict:
    runtime = _prepare_runtime_config(cfg)
    runtime["backbone"] = cfg.get("backbone", {})
    runtime["identity_head"] = cfg.get("identity_head", {})
    runtime["pad_head"] = cfg.get("pad_head", {})
    runtime["num_sensors"] = cfg.get("data", {}).get("num_sensors", 9)
    runtime["num_classes"] = int(num_classes)
    runtime["pad_debug_print_every_n_steps"] = 0
    return runtime


def _grad_norm(loss: torch.Tensor, params: Iterable[torch.nn.Parameter]) -> Dict[str, float]:
    params = list(params)
    if not loss.requires_grad:
        return {"used": 0, "total": len(params), "norm": 0.0}

    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq_norm = 0.0
    used = 0
    for grad in grads:
        if grad is None:
            continue
        sq_norm += float(grad.float().pow(2).sum().item())
        used += 1
    return {"used": used, "total": len(params), "norm": sq_norm ** 0.5}


def _mean_band_energy(gate_input: torch.Tensor) -> torch.Tensor:
    return gate_input.mean(dim=1)


def _pairwise_centroid_distance(features: torch.Tensor, labels: torch.Tensor) -> float:
    uniq = labels.unique(sorted=True)
    if uniq.numel() < 2:
        return 0.0
    centroids = []
    for label in uniq:
        mask = labels == label
        if mask.any():
            centroids.append(features[mask].mean(dim=0))
    if len(centroids) < 2:
        return 0.0
    stack = torch.stack(centroids, dim=0)
    dist = torch.cdist(stack, stack, p=2)
    triu = torch.triu_indices(dist.shape[0], dist.shape[1], offset=1)
    return float(dist[triu[0], triu[1]].mean().item())


def _route_summary(
    stats: Dict[str, torch.Tensor],
    liveness: torch.Tensor,
    sensor_labels: torch.Tensor | None,
) -> Dict[str, float]:
    gate_input = stats["gate_input_gateonly"]
    band_mean = _mean_band_energy(gate_input).float()
    live_mask = liveness == 1
    spoof_mask = liveness == 0

    result: Dict[str, float] = {
        "band_centroid_gap_live_spoof": 0.0,
        "band_sensor_centroid_gap": 0.0,
    }
    if live_mask.any() and spoof_mask.any():
        result["band_centroid_gap_live_spoof"] = float(
            torch.norm(
                band_mean[live_mask].mean(dim=0) - band_mean[spoof_mask].mean(dim=0),
                p=2,
            ).item()
        )
    if sensor_labels is not None:
        result["band_sensor_centroid_gap"] = _pairwise_centroid_distance(
            band_mean, sensor_labels,
        )
    for idx, name in enumerate(("low", "ridge", "high")):
        result[f"{name}_mean_all"] = float(band_mean[:, idx].mean().item())
        if live_mask.any():
            result[f"{name}_mean_live"] = float(band_mean[live_mask, idx].mean().item())
        if spoof_mask.any():
            result[f"{name}_mean_spoof"] = float(band_mean[spoof_mask, idx].mean().item())
    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    datamodule = OMFRDataModule(cfg)
    datamodule.current_phase = 2
    datamodule.setup("fit")
    if datamodule.identity_ds is None or datamodule.pad_ds is None:
        raise RuntimeError("Phase-2 audit requires both identity_ds and pad_ds.")

    loaders = datamodule.train_dataloader()
    id_batch = next(iter(loaders["identity"]))
    pad_batch = next(iter(loaders["pad"]))

    runtime = build_runtime_config(cfg, datamodule.identity_ds.num_classes)
    model = OMFRModule(runtime)
    model.train()
    model.current_phase = 2
    model.alpha = 1.0
    model.beta = float(cfg.get("phases", {}).get("beta_target", 0.02))
    model.gamma = float(runtime.get("gamma", 0.05))
    model.alpha_adv = float(cfg.get("phases", {}).get("alpha_adv_target", 0.0))
    model.lam_adv = 0.0

    id_images, id_labels = model._unpack_identity_batch(id_batch)
    id_backbone = model._run_backbone(id_images)
    id_out = model._run_identity(id_backbone)
    id_parts = model._identity_loss(id_out["mrl_embeddings"], id_labels)
    l_balance_id = sum(id_backbone["balance_losses"])

    pad_images, liveness = model._unpack_pad_batch(pad_batch)
    sensor_labels = model._unpack_sensor_labels(pad_batch)
    pad_backbone = model._run_backbone(
        pad_images,
        backbone_no_grad=True,
        route_mode="pad",
        detach_backbone_outputs=True,
    )
    pad_out = model._run_pad(pad_backbone)
    pad_parts = model._pad_classification_loss(pad_out["pad_logit"], liveness)
    l_balance_pad = sum(pad_backbone["balance_losses"])
    bmin = min(id_out["identity_embedding"].shape[0], pad_out["pad_embedding"].shape[0])
    l_orth = model._phase_aware_orth_loss(
        pad_out["pad_embedding"][:bmin],
        id_out["identity_embedding"][:bmin],
    )
    l_balance = 0.5 * (l_balance_id + l_balance_pad)

    losses = {
        "id_total": id_parts["total"],
        "focal": pad_parts["focal"],
        "bce": pad_parts["bce"],
        "pad_cls": pad_parts["total"],
        "orth": l_orth,
        "balance": l_balance,
        "weighted_focal": model.alpha * pad_parts["focal"],
        "weighted_bce": model.alpha * pad_parts["bce"],
        "weighted_pad_cls": model.alpha * pad_parts["total"],
        "weighted_orth": model.beta * l_orth,
        "weighted_balance": model.gamma * l_balance,
        "pad_weighted_total": (
            model.alpha * pad_parts["total"]
            + model.beta * l_orth
        ),
        "weighted_total": (
            id_parts["total"]
            + model.alpha * pad_parts["total"]
            + model.beta * l_orth
            + model.gamma * l_balance
        ),
    }

    param_groups = {
        "id_gate_proj": [
            p for n, p in model.named_parameters()
            if "gate_proj" in n and "pad_gate_proj" not in n
        ],
        "pad_gate_proj": [
            p for n, p in model.named_parameters()
            if "pad_gate_proj" in n or "pad_router_" in n
        ],
        "freq_norm": [p for n, p in model.named_parameters() if "freq_gate.norm" in n],
        "pad_stem": [p for n, p in model.named_parameters() if n.startswith("pad_stem.")],
        "pad_head": [p for n, p in model.named_parameters() if n.startswith("pad_head.")],
        "gabor_pad": [p for n, p in model.named_parameters() if n.startswith("gabor_pad.")],
        "identity_head": [
            p for n, p in model.named_parameters() if n.startswith("identity_head.")
        ],
    }

    route_summary = {
        name: {
            **_route_summary(stats, liveness, sensor_labels),
            "router_mode": stats.get("router_mode", "unknown"),
        }
        for name, stats in (
            ("s2", pad_backbone["routing_stats"]["s2"]),
            ("s3a", pad_backbone["routing_stats"]["s3a"]),
            ("s3b", pad_backbone["routing_stats"]["s3b"]),
        )
    }

    result = {
        "config": str(Path(args.config).resolve()),
        "batch_shapes": {
            "id_images": list(id_images.shape),
            "pad_images": list(pad_images.shape),
            "id_embedding": list(id_out["identity_embedding"].shape),
            "pad_embedding": list(pad_out["pad_embedding"].shape),
        },
        "weights": {
            "alpha": model.alpha,
            "beta": model.beta,
            "gamma": model.gamma,
            "alpha_adv": model.alpha_adv,
        },
        "loss_values": {name: float(loss.detach().item()) for name, loss in losses.items()},
        "loss_ratios": {
            "pad_to_id_weighted": float(
                (
                    losses["pad_weighted_total"].detach()
                    / losses["id_total"].detach().clamp_min(1e-8)
                ).item()
            ),
            "weighted_orth_to_weighted_focal": float(
                (
                    losses["weighted_orth"].detach()
                    / losses["weighted_focal"].detach().clamp_min(1e-8)
                ).item()
            ),
            "weighted_orth_to_weighted_pad_cls": float(
                (
                    losses["weighted_orth"].detach()
                    / losses["weighted_pad_cls"].detach().clamp_min(1e-8)
                ).item()
            ),
        },
        "grad_norms": {
            loss_name: {
                group_name: _grad_norm(loss, params)
                for group_name, params in param_groups.items()
            }
            for loss_name, loss in losses.items()
        },
        "route_summary": route_summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
