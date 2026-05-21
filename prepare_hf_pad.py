#!/usr/bin/env python3
"""
prepare_hf_pad.py — Download HuggingFace style-transferred PAD repos.

Each repo is fetched in a *single* call via ``snapshot_download`` and
materialized into its own folder, **keeping the original repo structure**
(per-material subfolders + README). No image decoding / relabeling is done
in the base mode.

Default repo -> output folder mapping:
    tourmii/style-transfered-2013 -> <out-root>/StyleTransfer2013
    tourmii/style-transfered-2015 -> <out-root>/StyleTransfer2015

Resulting layout (mirrors the HF repo exactly):
    PAD-joint/StyleTransfer2013/
        bio-ecoflex-2013/*.png
        bio-gelatine-2013/*.png
        ...
        README.md

--spoof-only  (PAD anti-spoof augmentation)
-------------------------------------------
These repos are 100% spoof (style-transferred attack materials; NO live).
With ``--spoof-only`` the downloaded tree is normalized into the
LivDet layout ``PADDataset`` understands, classified as Fake only:

    PAD-joint/StyleTransfer2013/Train/Fake/<Sensor>/<Material>/<src>__<img>

The ``<Sensor>/<Material>`` nesting (e.g. Biometrika/Ecoflex) is what
lets ``PADDataset._infer_sensor_id`` / ``_infer_material_id`` populate
the ids exactly — folder names are matched by *exact normalized name*,
so the raw ``bio-ecoflex-2013`` would otherwise resolve to Unknown and
the ``pad_hard_spoof_material_ids`` (Latex/WoodGlue ×2) boost would
never fire. ALL images go to ``Train`` — NO Test/Val split is created
for the HF data (EER is measured only on the original LivDet test set).
Originals are kept in place (PADDataset roots at the ``Train`` subdir,
so siblings are ignored); the copy is idempotent (skips existing).

Usage:
    python prepare_hf_pad.py                          # download both
    python prepare_hf_pad.py --only 2013              # one year
    python prepare_hf_pad.py --spoof-only             # download + normalize
    python prepare_hf_pad.py --spoof-only --skip-download   # normalize only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUT_ROOT = "/home/linhdang/workspace2/PAD-joint"

# repo id -> output sub-directory name (kept compatible with the datamodule's
# _normalize_name, so "StyleTransfer2013" resolves config `StyleTransfer2013`).
REPO_MAP: Dict[str, str] = {
    "tourmii/style-transfered-2013": "StyleTransfer2013",
    "tourmii/style-transfered-2015": "StyleTransfer2015",
}

IMG_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}

# Source dir-name token -> canonical name. Canonical names normalize
# EXACTLY to PADDataset.SENSOR_NAMES / MATERIAL_NAMES entries, so the
# nested layout makes sensor_id / material_id resolve correctly.
SENSOR_TOKENS = {
    "bio": "Biometrika", "biometrika": "Biometrika",
    "ita": "Italdata", "italdata": "Italdata",
    "cm": "CrossMatch", "crossmatch": "CrossMatch",
    "dp": "DigitalPersona", "digitalpersona": "DigitalPersona",
    "gb": "GreenBit", "greenbit": "GreenBit",
    "hi": "HiScan", "hiscan": "HiScan",
    "sw": "Swipe", "swipe": "Swipe",
    "orc": "Orcanthus", "orcanthus": "Orcanthus",
}
MATERIAL_KEYWORDS = [
    ("ecoflex", "Ecoflex"),
    ("gelatine", "Gelatin"), ("gelatin", "Gelatin"),
    ("latex", "Latex"), ("lat", "Latex"),
    ("modasil", "Modasil"),
    ("woodglue", "WoodGlue"), ("wood", "WoodGlue"),
    ("playdough", "PlayDoh"), ("playdoh", "PlayDoh"), ("playdo", "PlayDoh"),
    ("bodydouble", "BodyDouble"), ("body", "BodyDouble"),
]
SKIP_DIR_NAMES = {".cache", ".git", "train", "test", "__pycache__"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_repo(repo_id: str, dest: Path) -> None:
    """Fetch the *entire* dataset repo into ``dest`` in one call."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n[prepare] === {repo_id} -> {dest} ===")
    print(f"[prepare] snapshot_download({repo_id!r}) ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
    )
    n_files = sum(1 for p in dest.rglob("*") if p.is_file())
    top = sorted(p.name for p in dest.iterdir() if p.is_dir())
    print(f"[prepare] done: {n_files} files under {dest}")
    print(f"[prepare] top-level folders ({len(top)}): {top}")


# ---------------------------------------------------------------------------
# --spoof-only normalization
# ---------------------------------------------------------------------------

def _parse_sensor_material(dir_name: str) -> Tuple[str, str]:
    """``bio-ecoflex-2013`` -> ("Biometrika", "Ecoflex"). Unknown fallback."""
    tokens = [t for t in dir_name.lower().replace("_", "-").split("-") if t]
    sensor = "Unknown"
    for t in tokens:
        if t in SENSOR_TOKENS:
            sensor = SENSOR_TOKENS[t]
            break
    material = "Unknown"
    joined = "".join(tokens)
    for kw, canon in MATERIAL_KEYWORDS:
        if kw in joined:
            material = canon
            break
    return sensor, material


def _spoof_source_dirs(repo_dir: Path) -> List[Path]:
    """Top-level material dirs in the snapshot (skip cache/git/Train)."""
    out: List[Path] = []
    for d in sorted(repo_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name in SKIP_DIR_NAMES or d.name.startswith("."):
            continue
        out.append(d)
    return out


def normalize_spoof_only(repo_dir: Path) -> None:
    """Copy every image into Train/Fake/<Sensor>/<Material>/ (idempotent).

    100% Train — no Test/Val is ever created for the HF data.
    """
    if not repo_dir.is_dir():
        print(f"[prepare] spoof-only: {repo_dir} missing — skipped")
        return

    fake_root = repo_dir / "Train" / "Fake"
    src_dirs = _spoof_source_dirs(repo_dir)
    if not src_dirs:
        print(f"[prepare] spoof-only: no material dirs under {repo_dir}")
        return

    copied = skipped = 0
    summary: Dict[Tuple[str, str], int] = {}
    for sdir in src_dirs:
        sensor, material = _parse_sensor_material(sdir.name)
        dst_dir = fake_root / sensor / material
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in sorted(sdir.rglob("*")):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            # Prefix the source dir so files from different material
            # folders never collide and stay traceable.
            dst = dst_dir / f"{sdir.name}__{img.name}"
            if dst.exists():
                skipped += 1
                continue
            shutil.copy2(img, dst)
            copied += 1
            summary[(sensor, material)] = summary.get((sensor, material), 0) + 1
        print(f"[prepare]   {sdir.name} -> Train/Fake/{sensor}/{material}")

    print(f"[prepare] spoof-only {repo_dir.name}: copied {copied}, "
          f"skipped(existing) {skipped}")
    for (s, m), n in sorted(summary.items()):
        print(f"[prepare]   Train/Fake/{s}/{m}: {n}")
    unknown = sum(n for (s, m), n in summary.items()
                  if s == "Unknown" or m == "Unknown")
    if unknown:
        print(f"[prepare]   NOTE: {unknown} imgs have Unknown sensor/material "
              f"(name not parseable) — still valid spoof, no hard-weight boost.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download / normalize HF style-transferred PAD repos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                   help="Destination root (StyleTransfer2013/2015 under it).")
    p.add_argument("--only", choices=["2013", "2015"], default=None,
                   help="Process only one year (default: both).")
    p.add_argument("--spoof-only", action="store_true",
                   help="Normalize the repo into Train/Fake/<Sensor>/<Material>/ "
                        "(100%% spoof, NO test split).")
    p.add_argument("--skip-download", action="store_true",
                   help="Don't download; only run --spoof-only normalization "
                        "on already-present folders.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)

    repos = dict(REPO_MAP)
    if args.only is not None:
        repos = {r: d for r, d in REPO_MAP.items() if args.only in r}
        if not repos:
            sys.exit(f"[prepare] --only {args.only} matched no repo.")

    for repo_id, sub in repos.items():
        dest = out_root / sub
        if not args.skip_download:
            download_repo(repo_id=repo_id, dest=dest)
        if args.spoof_only:
            normalize_spoof_only(dest)

    if args.spoof_only:
        print("\n[prepare] DONE. Spoof-only layout: "
              "StyleTransfer20{13,15}/Train/Fake/<Sensor>/<Material>/ . "
              "config pad_datasets already lists StyleTransfer2013/2015.")
    else:
        print("\n[prepare] DONE. Repos downloaded with original structure intact. "
              "Re-run with --spoof-only to build the Train/Fake layout.")


if __name__ == "__main__":
    main()
