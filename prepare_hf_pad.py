#!/usr/bin/env python3
"""
prepare_hf_pad.py — Download & lay out HuggingFace style-transferred PAD data.

Pulls two HF repos and writes their images into the LivDet-style directory
layout that ``omfr.data.datasets.pad_dataset.PADDataset`` understands:

    PAD-joint/StyleTransfer2013/
        Train/Live/*.png
        Train/Fake/*.png
        Test/Live/*.png
        Test/Fake/*.png
    PAD-joint/StyleTransfer2015/
        Train/Live/*.png
        ...

Default repo -> output-dir mapping:
    tourmii/style-transfered-2013 -> StyleTransfer2013
    tourmii/style-transfered-2015 -> StyleTransfer2015

The HF dataset schema is auto-detected:
  * image column   : the first feature of type ``datasets.Image`` (or a
                      column named in --image-keys).
  * label column   : first column named in --label-keys. ``ClassLabel``
                      features are decoded via ``int2str``; the resulting
                      string is matched against the live/spoof keyword
                      sets. Plain ints fall back to ``1 == live`` unless
                      ``--spoof-is-one`` is given.
  * split mapping  : any split whose name contains "train" -> Train,
                      anything containing "test"/"val"/"eval" -> Test.
                      Unknown split names default to Train (override with
                      --default-split).

Usage:
    python prepare_hf_pad.py                       # both repos, default dirs
    python prepare_hf_pad.py --only 2013           # just StyleTransfer2013
    python prepare_hf_pad.py --out-root /data/PAD  # different destination
    python prepare_hf_pad.py --dry-run             # inspect schema only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUT_ROOT = "/home/linhdang/workspace2/PAD-joint"

# repo id -> output sub-directory name (matched by the datamodule's
# _normalize_name, so "StyleTransfer2013" resolves the config entry
# `StyleTransfer2013`).
REPO_MAP: Dict[str, str] = {
    "tourmii/style-transfered-2013": "StyleTransfer2013",
    "tourmii/style-transfered-2015": "StyleTransfer2015",
}

IMAGE_KEYS = ("image", "img", "picture", "pixel_values", "fingerprint")
LABEL_KEYS = (
    "label", "labels", "liveness", "liveness_label", "is_live",
    "class", "target", "y", "spoof", "live",
)

LIVE_WORDS = {"live", "alive", "real", "genuine", "bonafide", "bona_fide", "1", "true"}
SPOOF_WORDS = {"fake", "spoof", "spoofed", "attack", "pa", "0", "false"}


# ---------------------------------------------------------------------------
# Schema detection helpers
# ---------------------------------------------------------------------------

def _pick_image_column(features: Any) -> str:
    """Return the column holding the image."""
    try:
        from datasets import Image as HFImage
    except Exception:  # pragma: no cover - datasets always importable here
        HFImage = None

    if HFImage is not None:
        for name, feat in features.items():
            if isinstance(feat, HFImage):
                return name
    for key in IMAGE_KEYS:
        if key in features:
            return key
    raise SystemExit(
        f"[prepare] Could not find an image column. Columns: {list(features)}"
    )


def _pick_label_column(features: Any) -> Optional[str]:
    for key in LABEL_KEYS:
        if key in features:
            return key
    return None


def _label_to_liveness(
    raw: Any,
    feature: Any,
    spoof_is_one: bool,
) -> Optional[int]:
    """Map a raw label value to 1 (live) / 0 (spoof), or None if unknown."""
    # ClassLabel -> decode int to its string name first.
    name = None
    try:
        from datasets import ClassLabel
        if isinstance(feature, ClassLabel) and isinstance(raw, int):
            name = feature.int2str(raw).strip().lower()
    except Exception:
        pass

    if name is None and isinstance(raw, str):
        name = raw.strip().lower()

    if name is not None:
        if any(w in name for w in ("live", "real", "genuine", "bonafide")):
            return 1
        if any(w in name for w in ("fake", "spoof", "attack")):
            return 0
        if name in LIVE_WORDS:
            return 1
        if name in SPOOF_WORDS:
            return 0

    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        v = int(raw)
        if v not in (0, 1):
            return None
        # Convention: 1 == live, 0 == spoof, unless overridden.
        return (1 - v) if spoof_is_one else v
    return None


def _split_kind(split_name: str, default_split: str) -> str:
    s = split_name.lower()
    if "train" in s:
        return "Train"
    if any(k in s for k in ("test", "val", "eval", "dev")):
        return "Test"
    return default_split


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------

def export_repo(
    repo_id: str,
    out_dir: Path,
    spoof_is_one: bool,
    default_split: str,
    dry_run: bool,
    image_format: str,
) -> None:
    from datasets import load_dataset
    from datasets import DatasetDict

    print(f"\n[prepare] === {repo_id} -> {out_dir} ===")
    ds = load_dataset(repo_id)
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    first_split = next(iter(ds.values()))
    feats = first_split.features
    img_col = _pick_image_column(feats)
    lbl_col = _pick_label_column(feats)
    print(f"[prepare] image column = {img_col!r}")
    print(f"[prepare] label column = {lbl_col!r}")
    print(f"[prepare] splits       = {list(ds.keys())}")
    if lbl_col is None:
        raise SystemExit(
            f"[prepare] No label column found in {list(feats)}. "
            f"Pass one via --label-keys."
        )
    lbl_feature = feats[lbl_col]

    if dry_run:
        sample = first_split[0]
        print(f"[prepare] sample label raw   = {sample.get(lbl_col)!r}")
        print(f"[prepare] sample label -> liveness = "
              f"{_label_to_liveness(sample.get(lbl_col), lbl_feature, spoof_is_one)}")
        print("[prepare] dry-run: no files written.")
        return

    counts: Dict[Tuple[str, str], int] = {}
    skipped = 0
    for split_name, split_ds in ds.items():
        kind = _split_kind(split_name, default_split)  # Train | Test
        for i, row in enumerate(split_ds):
            liveness = _label_to_liveness(row.get(lbl_col), lbl_feature, spoof_is_one)
            if liveness is None:
                skipped += 1
                continue
            sub = "Live" if liveness == 1 else "Fake"
            dst_dir = out_dir / kind / sub
            dst_dir.mkdir(parents=True, exist_ok=True)

            img = row[img_col]
            # `datasets.Image` decodes to a PIL.Image; some repos store
            # raw bytes/paths — normalize to a PIL image.
            if not hasattr(img, "save"):
                from PIL import Image as PILImage
                import io
                if isinstance(img, dict) and img.get("bytes"):
                    img = PILImage.open(io.BytesIO(img["bytes"]))
                elif isinstance(img, dict) and img.get("path"):
                    img = PILImage.open(img["path"])
                else:
                    skipped += 1
                    continue
            if img.mode not in ("L", "RGB"):
                img = img.convert("L")

            fname = f"{split_name}_{i:06d}.{image_format}"
            img.save(dst_dir / fname)
            counts[(kind, sub)] = counts.get((kind, sub), 0) + 1

        print(f"[prepare]   split {split_name!r} -> {kind} done")

    print(f"[prepare] {repo_id} written:")
    for (kind, sub), n in sorted(counts.items()):
        print(f"[prepare]   {kind}/{sub}: {n}")
    if skipped:
        print(f"[prepare]   WARNING: skipped {skipped} rows (unmapped label/image)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download HF style-transferred PAD data into LivDet layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-root", default=DEFAULT_OUT_ROOT,
        help="Destination root (StyleTransfer2013/2015 created under it).",
    )
    p.add_argument(
        "--only", choices=["2013", "2015"], default=None,
        help="Process only one year (default: both).",
    )
    p.add_argument(
        "--spoof-is-one", action="store_true",
        help="Numeric labels use 1=spoof,0=live instead of the default 1=live.",
    )
    p.add_argument(
        "--default-split", choices=["Train", "Test"], default="Train",
        help="Where to put splits whose name is neither train nor test.",
    )
    p.add_argument(
        "--image-format", default="png", choices=["png", "bmp", "jpg"],
        help="On-disk image format.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Detect schema and print mapping without writing files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)

    repos = dict(REPO_MAP)
    if args.only is not None:
        repos = {
            r: d for r, d in REPO_MAP.items() if args.only in r
        }
        if not repos:
            sys.exit(f"[prepare] --only {args.only} matched no repo.")

    for repo_id, sub in repos.items():
        export_repo(
            repo_id=repo_id,
            out_dir=out_root / sub,
            spoof_is_one=args.spoof_is_one,
            default_split=args.default_split,
            dry_run=args.dry_run,
            image_format=args.image_format,
        )

    print("\n[prepare] DONE. Add to config: "
          "pad_datasets: [LivDet2013, LivDet2015, "
          "StyleTransfer2013, StyleTransfer2015]")


if __name__ == "__main__":
    main()
