#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/linhdang/anaconda3/etc/profile.d/conda.sh
conda activate base

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/base.yaml}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-60}"
SEED="${SEED:-42}"
LOG_NAME="${LOG_NAME:-omfr-pad-v23}"
TB_VERSION="${TB_VERSION:-pad-v23}"
WANDB_NAME="${WANDB_NAME:-pad-v23}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
PAD_DEBUG_PRINT_EVERY_N_STEPS="${PAD_DEBUG_PRINT_EVERY_N_STEPS:-20}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints/pad-v23}"

mkdir -p "$CKPT_DIR"
cd "$REPO_ROOT"

python train.py \
  --config "$CONFIG_PATH" \
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
