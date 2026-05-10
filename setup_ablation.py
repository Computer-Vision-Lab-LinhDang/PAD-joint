"""
Generate the post-baseline ablation matrix from configs/base_golden.yaml.

Outputs:
  - configs/base_full_adv.yaml
  - configs/base_no_orth.yaml
  - run_ablation.sh
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "base_golden.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)


def require_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing or invalid section: {key}")
    return value


def write_run_script(config_paths: list[Path], path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for cfg in config_paths:
        lines.append(f"python train.py --config {cfg.as_posix()}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(path, path.stat().st_mode | 0o111)


def make_variant(
    base_config: dict[str, Any],
    *,
    beta_target: float,
    alpha_adv_target: float,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    phases = require_section(config, "phases")
    phases["beta_target"] = beta_target
    phases["alpha_adv_target"] = alpha_adv_target
    return config


def main() -> None:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_CONFIG}")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml(BASE_CONFIG)

    variants = [
        (
            CONFIG_DIR / "base_full_adv.yaml",
            make_variant(
                base_config,
                beta_target=0.02,
                alpha_adv_target=0.02,
            ),
        ),
        (
            CONFIG_DIR / "base_no_orth.yaml",
            make_variant(
                base_config,
                beta_target=0.0,
                alpha_adv_target=0.02,
            ),
        ),
    ]

    for path, config in variants:
        dump_yaml(config, path)

    run_script = ROOT / "run_ablation.sh"
    write_run_script([path.relative_to(ROOT) for path, _ in variants], run_script)

    print("Generated:")
    for path, _ in variants:
        print(f"  {path.relative_to(ROOT)}")
    print(f"  {run_script.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
