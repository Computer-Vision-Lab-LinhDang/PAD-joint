#!/usr/bin/env python3
"""Prepare FVC+NIST identity root with subject-disjoint splits and config.

Tasks:
  1) Normalize FVC flat files into subject folders using symlinks.
  2) Symlink NIST subject folders into the same root (no data copying).
  3) Create subject-disjoint split_train.txt / split_val.txt (80/20).
  4) Generate configs/base_grand_fusion.yaml from configs/base_golden.yaml.
  5) Emit run_final.sh and chmod +x.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml


IMAGE_EXTS = {".bmp", ".png", ".wsq", ".jpg", ".jpeg", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _safe_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            try:
                if link_path.resolve() == target.resolve():
                    return
            except OSError:
                return
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(target), str(link_path))


def _list_images_flat(directory: Path) -> List[Path]:
    return sorted(
        p for p in directory.iterdir() if _is_image(p)
    )


def _list_images_in_subject(subject_dir: Path) -> List[Path]:
    return sorted(
        p for p in subject_dir.iterdir() if _is_image(p)
    )


def _discover_nist_subject_dirs(nist_root: Path) -> List[Tuple[str, Path]]:
    if not nist_root.is_dir():
        raise FileNotFoundError(f"NIST root not found: {nist_root}")

    train_dir = nist_root / "train"
    search_dir = train_dir if train_dir.is_dir() else nist_root

    candidates: Dict[str, Path] = {}
    for entry in sorted(search_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        key = f"nist_{entry.name}"
        candidates[key] = entry

    return sorted(candidates.items(), key=lambda item: item[0])


def _subject_from_fvc_name(filename: str) -> int | None:
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def _discover_fvc_db(root: Path, name: str, db_folder: str) -> Tuple[str, Path]:
    db_path = root / name / "Dbs" / db_folder
    if db_path.is_dir():
        return name.lower(), db_path
    raise FileNotFoundError(f"Missing FVC DB folder: {db_path}")


def _subject_dir_name(dataset_tag: str, subject_id: int) -> str:
    return f"{dataset_tag}_{subject_id:03d}"


def _split_subjects(
    subject_names: Sequence[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    names = list(subject_names)
    rng = random.Random(seed)
    rng.shuffle(names)
    if not names:
        return [], []
    split_idx = int(len(names) * (1.0 - val_ratio))
    split_idx = max(1, split_idx) if len(names) > 1 else len(names)
    split_idx = min(split_idx, len(names) - 1) if len(names) > 1 else len(names)
    return names[:split_idx], names[split_idx:]


def _write_split_file(root: Path, rel_paths: Iterable[PurePosixPath], filename: str) -> None:
    out_path = root / filename
    lines = [str(p) for p in rel_paths]
    out_path.write_text("\n".join(lines) + "\n")


def _count_images_recursive(subject_dir: Path) -> int:
    count = 0
    for root, _, files in os.walk(subject_dir):
        for fname in files:
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                count += 1
    return count


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _generate_config(
    base_cfg_path: Path,
    output_cfg_path: Path,
    identity_root: Path,
) -> None:
    data = yaml.safe_load(base_cfg_path.read_text())

    data.setdefault("data", {})
    data["data"]["identity_data_root"] = str(identity_root)
    data["data"]["identity_datasets"] = [
        "nist_sd300",
        "fvc2000_db1_a",
        "fvc2002_db1_a",
        "fvc2004_db1_a",
    ]

    data.setdefault("losses", {})
    data["losses"].setdefault("arcface", {})
    data["losses"]["pad_mixup_alpha"] = 0.0
    data["losses"]["pad_mixup_enabled"] = False
    data["losses"]["identity_label_smoothing"] = 0.1
    data["losses"]["arcface"]["label_smoothing"] = 0.1

    data.setdefault("callbacks", {})
    data["callbacks"]["swa"] = {
        "enabled": True,
        "swa_lrs": 1.0e-4,
        "swa_epoch_start": 40,
    }
    data["callbacks"]["early_stopping"] = {
        "enabled": True,
        "monitor": "val/cascaded_IM",
        "mode": "max",
        "patience": 7,
        "min_phase": 3,
    }

    data.setdefault("phases", {})
    data["phases"]["total_epochs"] = 60
    data.setdefault("trainer", {})
    data["trainer"]["max_epochs"] = 60

    output_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    output_cfg_path.write_text(
        yaml.safe_dump(data, sort_keys=False)
    )


def _write_run_script(repo_root: Path, config_path: Path) -> None:
    run_path = repo_root / "run_final.sh"
    content = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"python train.py --config {config_path.as_posix()}",
            "",
        ]
    )
    run_path.write_text(content)
    run_path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare FVC + NIST training root.")
    parser.add_argument(
        "--fvc-root",
        default="/home/linhdang/workspace2/FVC_Dataset",
    )
    parser.add_argument(
        "--nist-root",
        default="/home/linhdang/workspace2/NIST-300-ds/combined_nist",
    )
    parser.add_argument(
        "--out-root",
        default="/home/linhdang/workspace2/FVC_Processed_Train",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-config",
        default="configs/base_golden.yaml",
    )
    parser.add_argument(
        "--out-config",
        default="configs/base_grand_fusion.yaml",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    fvc_root = Path(args.fvc_root)
    nist_root = Path(args.nist_root)
    out_root = Path(args.out_root)
    train_root = out_root / "train"

    if train_root.exists() and train_root.is_symlink():
        raise RuntimeError(
            f"Refusing to use symlinked train dir: {train_root}"
        )
    os.makedirs(train_root, exist_ok=True)

    fvc_dbs = [
        _discover_fvc_db(fvc_root, "FVC2000", "Db1_a"),
        _discover_fvc_db(fvc_root, "FVC2002", "Db1_a"),
        _discover_fvc_db(fvc_root, "FVC2004", "DB1_A"),
    ]

    subject_dirs: Dict[str, Path] = {}

    nist_subjects = _discover_nist_subject_dirs(nist_root)
    for subject_name, subject_path in nist_subjects:
        if _is_within_root(subject_path, out_root):
            continue
        link_path = train_root / subject_name
        _safe_symlink(subject_path, link_path)
        subject_dirs[subject_name] = link_path

    for dataset_tag, db_path in fvc_dbs:
        for img_path in _list_images_flat(db_path):
            if _is_within_root(img_path, out_root):
                continue
            subject_id = _subject_from_fvc_name(img_path.name)
            if subject_id is None:
                continue
            subject_name = _subject_dir_name(dataset_tag, subject_id)
            subject_dir = train_root / subject_name
            subject_dir.mkdir(parents=True, exist_ok=True)
            link_path = subject_dir / img_path.name
            _safe_symlink(img_path, link_path)
            subject_dirs[subject_name] = subject_dir

    subject_names = sorted(subject_dirs.keys())
    train_subjects, val_subjects = _split_subjects(subject_names, args.val_ratio, args.seed)

    train_rel_paths: List[PurePosixPath] = []
    val_rel_paths: List[PurePosixPath] = []
    empty_subjects: List[str] = []

    for subject in train_subjects:
        subject_dir = subject_dirs[subject]
        images = _list_images_in_subject(subject_dir)
        if not images:
            recursive_count = _count_images_recursive(subject_dir)
            if recursive_count > 0:
                empty_subjects.append(subject)
            continue
        for img_path in images:
            rel = PurePosixPath("train") / subject / img_path.name
            train_rel_paths.append(rel)

    for subject in val_subjects:
        subject_dir = subject_dirs[subject]
        images = _list_images_in_subject(subject_dir)
        if not images:
            recursive_count = _count_images_recursive(subject_dir)
            if recursive_count > 0:
                empty_subjects.append(subject)
            continue
        for img_path in images:
            rel = PurePosixPath("train") / subject / img_path.name
            val_rel_paths.append(rel)

    _write_split_file(out_root, sorted(train_rel_paths), "split_train.txt")
    _write_split_file(out_root, sorted(val_rel_paths), "split_val.txt")

    base_cfg = repo_root / args.base_config
    out_cfg = repo_root / args.out_config
    _generate_config(base_cfg, out_cfg, out_root)

    _write_run_script(repo_root, out_cfg)

    print(f"Prepared identity root: {out_root}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Train images: {len(train_rel_paths)}")
    print(f"Val images: {len(val_rel_paths)}")
    if empty_subjects:
        preview = ", ".join(empty_subjects[:10])
        print(
            "Warning: subjects with no direct images detected (nested only): "
            f"{preview}"
        )
    print(f"Config written: {out_cfg}")
    print(f"Run script: {repo_root / 'run_final.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
