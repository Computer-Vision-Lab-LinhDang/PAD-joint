#!/usr/bin/env python3
"""Prepare an IdentityDataset-compatible root for Stage-2 training.

The current IdentityDataset expects:

    root/
      train/<class_id>/<image>
      split_train.txt
      split_val.txt

NIST302a stores identity in the filename instead of the parent folder:

    00002481_A_roll_04.png

For fingerprint identity, the stable class is subject + finger position,
so the filename above becomes:

    nist302a_00002481_04

The script creates symlinks only; source datasets are not modified.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable


IMAGE_EXTS = {".bmp", ".png", ".wsq", ".jpg", ".jpeg", ".tif", ".tiff"}
NIST302A_RE = re.compile(r"^(?P<subject>\d+)_(?P<session>[^_]+)_roll_(?P<finger>\d+)$")


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _safe_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            try:
                if link_path.resolve() == target.resolve():
                    return
            except OSError:
                return
        raise FileExistsError(f"Refusing to overwrite existing file: {link_path}")
    os.symlink(str(target.resolve()), str(link_path))


def _write_split(root: Path, filename: str, rel_paths: Iterable[PurePosixPath]) -> None:
    lines = sorted(str(path) for path in rel_paths)
    (root / filename).write_text("\n".join(lines) + "\n")


def _split_subjects(subjects: list[str], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    subjects = sorted(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    if len(subjects) <= 1:
        return set(subjects), set()
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val = set(subjects[:n_val])
    train = set(subjects[n_val:])
    return train, val


def _prepare_nist302a(src_root: Path, out_root: Path, val_ratio: float, seed: int) -> tuple[int, int, int]:
    files = sorted(src_root.glob("images/challengers/*/roll/png/*"))
    files = [path for path in files if _is_image(path)]
    by_subject: dict[str, list[tuple[Path, str]]] = defaultdict(list)

    for path in files:
        match = NIST302A_RE.match(path.stem)
        if match is None:
            continue
        subject = match.group("subject")
        finger = match.group("finger")
        class_id = f"nist302a_{subject}_{finger}"
        by_subject[subject].append((path, class_id))

    train_subjects, val_subjects = _split_subjects(
        list(by_subject.keys()), val_ratio=val_ratio, seed=seed,
    )
    train_rel: list[PurePosixPath] = []
    val_rel: list[PurePosixPath] = []

    for subject, items in sorted(by_subject.items()):
        target_rel = train_rel if subject in train_subjects else val_rel
        for src, class_id in items:
            rel = PurePosixPath("train") / class_id / src.name
            _safe_symlink(src, out_root / rel)
            target_rel.append(rel)

    _write_split(out_root, "split_train.txt", train_rel)
    _write_split(out_root, "split_val.txt", val_rel)
    classes = {path.name for path in (out_root / "train").iterdir() if path.is_dir()}
    return len(train_rel), len(val_rel), len(classes)


def _fvc_class_id(year: str, db_name: str, filename: str) -> str | None:
    stem = Path(filename).stem
    subject = stem.split("_", 1)[0]
    if not subject.isdigit():
        return None
    return f"{year.lower()}_{db_name.lower()}_{int(subject):03d}"


def _iter_fvc_db_dirs(fvc_root: Path, scope: str) -> list[Path]:
    db_dirs = sorted(path for path in fvc_root.glob("FVC*/Dbs/*") if path.is_dir())
    if scope == "all":
        return db_dirs
    if scope == "all_a":
        return [path for path in db_dirs if path.name.lower().endswith("_a")]
    return [
        path for path in db_dirs
        if path.parent.parent.name in {"FVC2000", "FVC2002", "FVC2004"}
        and path.name.lower() in {"db1_a", "db1_a".lower()}
    ]


def _prepare_fvc(src_root: Path, out_root: Path, val_ratio: float, seed: int, scope: str) -> tuple[int, int, int]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for db_dir in _iter_fvc_db_dirs(src_root, scope):
        year = db_dir.parent.parent.name
        for path in sorted(db_dir.iterdir()):
            if not _is_image(path):
                continue
            class_id = _fvc_class_id(year, db_dir.name, path.name)
            if class_id is None:
                continue
            by_class[class_id].append(path)

    train_classes, val_classes = _split_subjects(
        list(by_class.keys()), val_ratio=val_ratio, seed=seed,
    )
    train_rel: list[PurePosixPath] = []
    val_rel: list[PurePosixPath] = []

    for class_id, files in sorted(by_class.items()):
        target_rel = train_rel if class_id in train_classes else val_rel
        for src in files:
            rel = PurePosixPath("train") / class_id / src.name
            _safe_symlink(src, out_root / rel)
            target_rel.append(rel)

    _write_split(out_root, "split_train.txt", train_rel)
    _write_split(out_root, "split_val.txt", val_rel)
    return len(train_rel), len(val_rel), len(by_class)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare identity root for OMFR.")
    parser.add_argument("--source", choices=("nist302a", "fvc"), default="nist302a")
    parser.add_argument("--nist302a-root", default="/home/aiserver/works/fingerprint/dataset/302a")
    parser.add_argument("--fvc-root", default="/home/aiserver/works/fingerprint/dataset/FVC_Dataset")
    parser.add_argument("--out-root", default="data/processed/identity_stage2_nist302a")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fvc-scope", choices=("db1_a", "all_a", "all"), default="db1_a")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.source == "nist302a":
        train_n, val_n, class_n = _prepare_nist302a(
            Path(args.nist302a_root), out_root, args.val_ratio, args.seed,
        )
    else:
        train_n, val_n, class_n = _prepare_fvc(
            Path(args.fvc_root), out_root, args.val_ratio, args.seed, args.fvc_scope,
        )

    print(f"identity_root={out_root}")
    print(f"source={args.source}")
    print(f"classes={class_n}")
    print(f"train_images={train_n}")
    print(f"val_images={val_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
