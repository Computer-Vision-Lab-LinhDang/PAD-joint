"""
export.py — ONNX Export for Trained OMFRModule

Exports the full inference pipeline (Gabor → ViT-Tiny+MoE → heads) to ONNX.
The exported graph accepts (B, 1, 224, 224) grayscale images and produces:
    - identity_embedding: (B, 256)
    - pad_logit:          (B, 1)     (apply sigmoid for P(live))

Usage:
    python export.py --checkpoint checkpoints/last.ckpt
    python export.py --checkpoint checkpoints/last.ckpt --output exports/omfr.onnx
    python export.py --checkpoint checkpoints/last.ckpt --dim 128  # export 128-D MRL
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from omfr.models.omfr import OMFRModule


# ---------------------------------------------------------------------------
# Thin wrapper for ONNX export
# ---------------------------------------------------------------------------

class OMFRExportWrapper(nn.Module):
    """
    Wraps OMFRModule for ONNX-compatible export.

    Input:   (B, 1, 224, 224) — grayscale fingerprint image
    Outputs: identity_embedding (B, dim), pad_logit (B, 1)

    MoE routing is handled transparently inside the backbone.
    Only the final outputs are returned (no balance_losses / routing_stats).
    """

    def __init__(self, module: OMFRModule, identity_dim: int = 256) -> None:
        super().__init__()
        self.gabor         = module.gabor
        self.backbone      = module.backbone
        self.identity_head = module.identity_head
        self.pad_head      = module.pad_head
        self.identity_dim  = identity_dim

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 1, 224, 224) — grayscale, float32, range [0, 1]

        Returns:
            identity_embedding: (B, identity_dim) — L2-normalized
            pad_logit:          (B, 1)            — raw logit; sigmoid → P(live)
        """
        enhanced     = self.gabor(images)            # (B, 3, 224, 224)
        backbone_out = self.backbone(enhanced)

        id_out = self.identity_head({
            "layer12_tokens": backbone_out["layer12_tokens"],
            "cls_token":      backbone_out["cls_token"],
        })

        pad_out = self.pad_head({
            "layer3_tokens":   backbone_out["layer3_tokens"],
            "layer7_tokens":   backbone_out["layer7_tokens"],
            "routing_stats_3": backbone_out["routing_stats"][3],
            "routing_stats_7": backbone_out["routing_stats"][7],
        })

        # Return MRL slice if dim < 256
        if self.identity_dim < 256:
            id_emb = id_out["mrl_embeddings"][self.identity_dim]
        else:
            id_emb = id_out["identity_embedding"]

        return id_emb, pad_out["pad_logit"]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _try_simplify(onnx_path: str) -> None:
    """Attempt to simplify the ONNX model (requires onnxsim)."""
    try:
        import onnx
        from onnxsim import simplify

        model = onnx.load(onnx_path)
        simplified, ok = simplify(model)
        if ok:
            onnx.save(simplified, onnx_path)
            print(f"  [onnxsim] Model simplified successfully.")
        else:
            print("  [onnxsim] Simplification failed — keeping original.")
    except ImportError:
        print("  [onnxsim] Not installed — skipping simplification.")
    except Exception as exc:
        print(f"  [onnxsim] Error: {exc} — keeping original.")


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    identity_dim: int = 256,
    opset_version: int = 17,
    simplify: bool = True,
    device: str = "cpu",
) -> None:
    """
    Load a trained OMFRModule from a checkpoint and export to ONNX.

    Args:
        checkpoint_path: path to .ckpt file
        output_path:     destination .onnx file
        identity_dim:    MRL embedding dimension to export (64, 128, or 256)
        opset_version:   ONNX opset (default 17)
        simplify:        run onnxsim simplifier after export (default True)
        device:          'cpu' or 'cuda' for tracing (default 'cpu')
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    module = OMFRModule.load_from_checkpoint(checkpoint_path, map_location=device)
    module.eval()
    module.to(device)

    wrapper = OMFRExportWrapper(module, identity_dim=identity_dim)
    wrapper.eval()

    # Dummy input: (1, 1, 224, 224) — grayscale, single sample
    dummy_input = torch.zeros(1, 1, 224, 224, dtype=torch.float32, device=device)

    # Warm-up pass to check shapes before export
    with torch.no_grad():
        id_emb, pad_logit = wrapper(dummy_input)
    print(
        f"  Verified output shapes — "
        f"identity: {tuple(id_emb.shape)}, pad_logit: {tuple(pad_logit.shape)}"
    )

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX (opset={opset_version}): {output_path}")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["images"],
        output_names=["identity_embedding", "pad_logit"],
        dynamic_axes={
            "images":              {0: "batch_size"},
            "identity_embedding":  {0: "batch_size"},
            "pad_logit":           {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    print(f"  ONNX model written to: {output_path}")

    if simplify:
        print("  Running onnxsim simplifier...")
        _try_simplify(output_path)

    # Verify the exported model loads cleanly
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("  ONNX model check passed.")
    except ImportError:
        print("  [onnx] Not installed — skipping model check.")
    except Exception as exc:
        print(f"  [onnx] Model check error: {exc}")

    print("Export complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained OMFRModule to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .ckpt Lightning checkpoint file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .onnx file path (default: exports/omfr_dim<dim>.onnx).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
        choices=[64, 128, 256],
        help="Identity embedding dimension to export via MRL (default: 256).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim simplification step.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for tracing (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output = args.output or f"exports/omfr_dim{args.dim}.onnx"

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=output,
        identity_dim=args.dim,
        opset_version=args.opset,
        simplify=not args.no_simplify,
        device=args.device,
    )


if __name__ == "__main__":
    main()
