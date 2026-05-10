#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/base_full_adv.yaml
python train.py --config configs/base_no_orth.yaml
