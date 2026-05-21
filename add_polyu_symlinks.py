#!/usr/bin/env python3
"""
add_polyu_symlinks.py — Graft PolyU contact-based identities into the
aggregated identity root, the SAME way NIST is already integrated.

The training identity root (``FVC_Processed_Train``) is a real directory
whose ``train/`` is an aggregate built from symlinks + a path-list file
(``split_train.txt``). NIST classes are 23 180 symlinks into
``NIST-300-ds``; FVC classes are real dirs. ``IdentityDataset`` is
split-file-driven and ``PIL.Image.open`` follows symlink chains, so we
extend the corpus by replicating exactly that mechanism.

This script adds the 336 PolyU **contact-based** finger classes
(``contact-based_fingerprints_split/train``) — TRAIN ONLY. The
validation set (``split_val.txt``) is deliberately left untouched so
Rank-1 stays an honest open-set measurement.

For each ``finger_XXX``:
  * symlink  FVC_Processed_Train/train/polyu_finger_XXX -> <abs PolyU dir>
  * append   "train/polyu_finger_XXX/<img>" lines to split_train.txt

``polyu_*`` class names contain no "fvc", so ``IdentityDataset.get_groups``
tags them as the "nist" group — they join the rest pool and the
GroupBalancedPKSampler keeps its 50% FVC share intact (Option a).

Idempotent: re-running rebuilds the polyu_ symlinks and *replaces* all
``train/polyu_`` lines in the split file (never duplicates). A one-time
``split_train.txt.bak`` backup is created.

Usage:
    python add_polyu_symlinks.py --dry-run     # preview, write nothing
    python add_polyu_symlinks.py               # apply
    python add_polyu_symlinks.py --force       # also fix mismatched links
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Mirror IdentityDataset.IMG_EXTS so enumerated files match what the
# dataset will actually load.
IMG_EXTS = {".bmp", ".png", ".wsq", ".jpg", ".jpeg", ".tif", ".tiff"}

DEFAULT_SRC = (
    "/home/linhdang/workspace2/dataset/PolyU/"
    "contact-based_fingerprints_split/train"
)
DEFAULT_DST_ROOT = "/home/linhdang/workspace2/FVC_Processed_Train"
PREFIX = "polyu_"


def _enumerate_images(class_dir: Path) -> List[str]:
    """Sorted image filenames (relative to class_dir), following symlinks."""
    names: List[str] = []
    for entry in sorted(class_dir.rglob("*")):
        # follow symlinks: resolve & confirm it's a real file
        try:
            if not entry.is_file():  # is_file() follows symlinks
                continue
        except OSError:
            continue
        if entry.suffix.lower() in IMG_EXTS:
            names.append(str(entry.relative_to(class_dir)))
    return names


def _plan(src: Path, prefix: str) -> List[Tuple[str, Path, List[str]]]:
    """Return [(link_name, abs_target_dir, [img_rel,...]), ...]."""
    if not src.is_dir():
        sys.exit(f"[polyu] source not found / not a dir: {src}")
    plan: List[Tuple[str, Path, List[str]]] = []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        imgs = _enumerate_images(d)
        if not imgs:
            print(f"[polyu] WARN: {d.name} has no images — skipped")
            continue
        plan.append((f"{prefix}{d.name}", d.resolve(), imgs))
    return plan


def _make_symlink(link: Path, target: Path, force: bool) -> str:
    """Create/repair link -> target. Returns one of created|ok|fixed|skip."""
    if link.is_symlink():
        cur = os.readlink(link)
        if cur == str(target):
            return "ok"
        if force:
            link.unlink()
            link.symlink_to(target, target_is_directory=True)
            return "fixed"
        print(f"[polyu] WARN: {link.name} -> {cur} (want {target}); "
              f"use --force to repair — skipped")
        return "skip"
    if link.exists():  # real dir/file with the same name
        print(f"[polyu] WARN: {link} exists and is not a symlink — skipped")
        return "skip"
    link.symlink_to(target, target_is_directory=True)
    return "created"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Graft PolyU contact-based classes into the ID train root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="PolyU contact-based_fingerprints_split/train dir.")
    ap.add_argument("--dst-root", default=DEFAULT_DST_ROOT,
                    help="Aggregated identity root (has train/ + split_train.txt).")
    ap.add_argument("--prefix", default=PREFIX,
                    help="Class-name prefix for the grafted PolyU dirs.")
    ap.add_argument("--force", action="store_true",
                    help="Repair symlinks whose target differs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan; create/modify nothing.")
    args = ap.parse_args()

    src = Path(args.src)
    dst_root = Path(args.dst_root)
    train_dir = dst_root / "train"
    split_file = dst_root / "split_train.txt"

    if not train_dir.is_dir():
        sys.exit(f"[polyu] train dir not found: {train_dir}")
    if not split_file.is_file():
        sys.exit(f"[polyu] split_train.txt not found: {split_file}")

    plan = _plan(src, args.prefix)
    n_imgs = sum(len(imgs) for _, _, imgs in plan)
    print(f"[polyu] source classes : {len(plan)}")
    print(f"[polyu] source images  : {n_imgs}")

    new_lines = [
        f"train/{link}/{img}"
        for link, _, imgs in plan
        for img in imgs
    ]

    # Idempotent split rewrite: drop every existing polyu_ line, keep the
    # rest verbatim, append the freshly-built block.
    existing = split_file.read_text().splitlines()
    kept = [ln for ln in existing if not ln.startswith(f"train/{args.prefix}")]
    dropped = len(existing) - len(kept)
    final_lines = kept + new_lines

    print(f"[polyu] split_train.txt: {len(existing)} lines "
          f"(drop {dropped} stale polyu, +{len(new_lines)} new) "
          f"-> {len(final_lines)}")

    if args.dry_run:
        print("\n[polyu] DRY-RUN — no changes. Sample:")
        for link, tgt, imgs in plan[:3]:
            print(f"  ln -s {tgt}  {train_dir}/{link}   ({len(imgs)} imgs)")
        for ln in new_lines[:3]:
            print(f"  + {ln}")
        return

    # 1) symlinks
    stats = {"created": 0, "ok": 0, "fixed": 0, "skip": 0}
    for link_name, target, _ in plan:
        stats[_make_symlink(train_dir / link_name, target, args.force)] += 1
    print(f"[polyu] symlinks: {stats}")

    # 2) split file (back up once, then atomic-ish replace)
    bak = split_file.with_suffix(split_file.suffix + ".bak")
    if not bak.exists():
        bak.write_text("\n".join(existing) + "\n")
        print(f"[polyu] backup written: {bak}")
    tmp = split_file.with_suffix(split_file.suffix + ".tmp")
    tmp.write_text("\n".join(final_lines) + "\n")
    os.replace(tmp, split_file)
    print(f"[polyu] split_train.txt updated ({len(final_lines)} lines).")
    print("[polyu] DONE. Val set untouched. Re-run is safe (idempotent).")


if __name__ == "__main__":
    main()
