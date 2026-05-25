#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/aiserver/works/fingerprint/PAD-joint"
cd "$REPO_DIR"

mkdir -p logs/pad_gabor_ablation
RUN_LOG="logs/pad_gabor_ablation/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

PYTHON_BIN="${PYTHON_BIN:-/home/aiserver/miniconda3/bin/python}"
GPU="${GPU:-0}"
CKPT="${CKPT:-checkpoints_sa24_stage2_l1_initfrom/best_phase1.ckpt}"

COMMON_ARGS=(
  --config configs/grand_fusion_sa24_padft.yaml
  --init-from "$CKPT"
  backbone.pretrained=false
  logging.wandb.enabled=false
  trainer.enable_checkpointing=false
  trainer.accumulate_grad_batches=1
  trainer.limit_train_batches=15
  trainer.limit_val_batches=20
  data.pk_p=32
  data.pk_k=2
  data.pad_batch_size=64
  phases.phase1_max_epochs=0
  phases.phase2_epochs=20
  phases.total_epochs=20
  trainer.max_epochs=20
)

run_variant() {
  local name="$1"
  shift || true

  echo
  echo "================================================================"
  echo "PAD Gabor ablation: ${name}"
  echo "Checkpoint: ${CKPT}"
  echo "Extra overrides: $*"
  echo "================================================================"

  CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. \
    "$PYTHON_BIN" train.py \
      "${COMMON_ARGS[@]}" \
      "logging.name=ablate_padgabor_${name}" \
      "$@"
}

run_variant base
run_variant backbone_identity \
  pad_stage2.backbone_input=identity_gabor
run_variant stem_identity \
  pad_stage2.pad_stem_input=identity_gabor
run_variant lowpass5 \
  pad_stage2.pad_gabor_blur_kernel=5

echo
echo "All PAD Gabor ablations completed."
