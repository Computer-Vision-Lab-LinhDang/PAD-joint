#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT_PATH="${CKPT_PATH:-/home/linhdang/workspace2/PAD-joint/checkpoints/last-v22.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/base.yaml}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-80}"
SEED="${SEED:-42}"
LOG_NAME="${LOG_NAME:-omfr-pad-debug}"
TB_VERSION="${TB_VERSION:-pad-debug-last-v22}"
WANDB_NAME="${WANDB_NAME:-pad-debug-last-v22}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
PAD_DEBUG_PRINT_EVERY_N_STEPS="${PAD_DEBUG_PRINT_EVERY_N_STEPS:-10}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints/pad-debug-last-v22}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

mkdir -p "$CKPT_DIR"
cd "$REPO_ROOT"

python - <<'PY' "$CKPT_PATH"
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu")
print(
    "[init-meta] "
    f"path={path} "
    f"epoch={ckpt.get('epoch')} "
    f"global_step={ckpt.get('global_step')}"
)
PY

python train.py \
  --config "$CONFIG_PATH" \
  --init-from "$CKPT_PATH" \
  --seed "$SEED" \
  "phases.total_epochs=${TOTAL_EPOCHS}" \
  "trainer.max_epochs=${TOTAL_EPOCHS}" \
  "trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}" \
  "checkpoint.dirpath=${CKPT_DIR}" \
  "logging.name=${LOG_NAME}" \
  "logging.version=${TB_VERSION}" \
  "logging.wandb.enabled=${WANDB_ENABLED}" \
  "logging.wandb.name=${WANDB_NAME}" \
  "pad_debug_print_every_n_steps=${PAD_DEBUG_PRINT_EVERY_N_STEPS}" \
  "$@"
