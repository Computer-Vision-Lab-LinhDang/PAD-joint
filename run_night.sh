#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/base_golden.yaml
python train.py --config configs/base_no_ohem.yaml
python train.py --config configs/base_margin_035.yaml
