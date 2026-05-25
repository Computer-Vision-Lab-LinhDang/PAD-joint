# PAD-joint Codex Handoff

Last updated: 2026-05-25 10:45 +07.

This file is the practical handoff summary for Codex or another engineer working
on this repository. It focuses on the current problem, codebase structure,
training/evaluation pipeline, and known model state.

## 1. Project Goal

This repository trains an OMFR model: Orthogonal Matryoshka Fingerprint
Representation. The model tries to solve two fingerprint tasks in one system:

- Identity matching: extract an embedding for fingerprint matching.
- PAD/liveness: classify an input as live or spoof.

The current work has focused on improving PAD generalization on LivDet while not
destroying identity behavior. The recent issue was that the previous Phase-2
model still had high PAD error on external LivDet evaluation, especially APCER
at threshold 0.5. The latest fine-tune improved spoof rejection and EER, but it
also increased BPCER at the fixed 0.5 threshold, so threshold calibration is now
required.

## 2. Current Important State

Recent fine-tune config:

- Config: `configs/grand_fusion_sa24_padft.yaml`
- Log: `logs/omfr_sa24_padft/version_0/metrics.csv`
- Intended best checkpoint from that run: `checkpoints_sa24_stage2_padft/best_phase2.ckpt`

Important warning: after an interrupted cleanup, the directories
`checkpoints_sa24_stage2_padft/` and `checkpoints_sa24_stage2_split/` are not
present in the workspace anymore. The log still exists, but the best fine-tuned
checkpoint is not visible at its previous path. Before any more cleanup, check
for a backup, W&B artifact, remote copy, or filesystem snapshot.

Checkpoints currently visible from this workspace scan:

- `checkpoints/best_phase1-v1.ckpt`
- `checkpoints_sa24_stage2_l1_initfrom/best_phase1.ckpt`
- `checkpoints_sa24_stage2_l1_initfrom/best_phase2.ckpt`

These are older checkpoints and are not the same as the latest PAD fine-tuned
best model.

## 3. Latest Training Result

The fine-tune run `omfr_sa24_padft/version_0` completed 20 validation epochs.
Best internal validation by `val/acer`:

- Epoch: 13
- Step: 3821
- `val/pad_accuracy`: 86.87%
- `val/acer`: 13.23%
- `val/apcer`: 10.92%
- `val/bpcer`: 15.53%
- `val/identity_rank1`: 1.64%
- `val/cascaded_IM`: 2.35%

Final epoch 19 was slightly worse by ACER:

- `val/pad_accuracy`: 86.52%
- `val/acer`: 13.40%
- `val/apcer`: 15.40%
- `val/bpcer`: 11.41%
- `val/identity_rank1`: 1.79%
- `val/cascaded_IM`: 2.67%

Compared with the previous split Phase-2 best:

- PAD accuracy improved from 84.14% to 86.87%.
- ACER improved from 15.66% to 13.23%.
- APCER improved from 20.51% to 10.92%.
- BPCER worsened from 10.81% to 15.53%.

External LivDet EER comparison from the latest successful evaluation:

| Dataset | Previous EER | Fine-tuned EER |
| --- | ---: | ---: |
| LivDet2011 | 19.28% | 15.78% |
| LivDet2013 | 17.05% | 14.01% |
| LivDet2015 | 10.70% | 10.02% |

At fixed threshold 0.5, APCER improved strongly but BPCER worsened. This means
the model became stricter on live samples. The new operating threshold should be
calibrated; do not reuse the old threshold blindly. Observed EER thresholds were
approximately:

- LivDet2011: 0.4898
- LivDet2013: 0.3621
- LivDet2015: 0.3743

## 4. Repository Layout

Top-level entry points:

- `train.py`: Lightning training entrypoint.
- `eval.py`: checkpoint evaluation for identity, PAD, and integrated metrics.
- `eval_tta.py`: evaluation with test-time augmentation.
- `export.py`: ONNX/export path.
- `prepare_identity_root.py`: prepares identity dataset root/splits.
- `prep_grand_fusion.py`: data preparation utility for the grand fusion setup.
- `prepare_hf_pad.py`: PAD dataset preparation utility.
- `add_polyu_symlinks.py`: helper for dataset symlinks.

Main package:

- `omfr/models/omfr.py`: main LightningModule.
- `omfr/models/backbone/`: Gabor, PAD stem, FastViT/TinyViT, MoE/frequency
  routing modules.
- `omfr/models/heads/`: identity and PAD heads.
- `omfr/models/losses/`: ArcFace, SupCon, PAD BCE/focal/OHEM loss,
  orthogonality, sensor adversarial loss.
- `omfr/data/datamodule.py`: phase-aware data loading.
- `omfr/data/datasets/`: identity, PAD, joint datasets.
- `omfr/data/samplers/`: PK and balanced samplers.
- `omfr/callbacks/phase_scheduler.py`: dynamic phase state machine.
- `omfr/callbacks/gradient_monitor.py`: gradient logging.
- `omfr/evaluation/`: PAD, matching, and integrated metrics.

Configs:

- `configs/base.yaml`: older/base config.
- `configs/base_grand_fusion.yaml`: grand fusion base.
- `configs/grand_fusion_sa24_dynamic.yaml`: previous dynamic two-phase SA24
  config.
- `configs/grand_fusion_sa24_padft.yaml`: latest PAD fine-tune config.

## 5. Model Architecture

The model is built in `omfr/models/omfr.py`.

Input:

- Fingerprint image, grayscale, resized to 224x224.

Preprocessing branches:

- `self.gabor`: identity-oriented learnable Gabor bank, ridge-frequency tuned.
- `self.gabor_pad`: PAD-oriented learnable Gabor bank, initialized with higher
  frequency for pore/micro-texture cues.

Backbone:

- Selected by `backbone.name`.
- Current fine-tune uses `fastvit_sa24`.
- `FastViTBackbone` is configured with `in_chans=8`, MoE experts, and optional
  gradient checkpointing.
- Older configs can use TinyViT.

Heads:

- `IdentityHead`: consumes later backbone features and emits a 256-D embedding
  plus Matryoshka sub-embeddings at 64/128/256 dimensions.
- `PADHead`: consumes PAD stem features, early backbone features, and MoE
  routing stats to emit:
  - `pad_logit`
  - `pad_embedding`
  - `pad_features`

PAD branch policy:

- Implemented through `_run_pad_branch()` and `_phase2_pad_backbone_policy()`.
- In older Phase-2 configs, the backbone was effectively frozen/detached for
  PAD.
- In `grand_fusion_sa24_padft.yaml`, PAD uses light early backbone adaptation:
  - `pad_stage2.backbone_grad: true`
  - `pad_stage2.backbone_input: pad_gabor`
  - `pad_stage2.detach_backbone_features: false`
  - `pad_stage2.detach_routing_stats: true`

This lets PAD update early texture-facing groups while keeping late identity
features and MoE routing mostly protected.

## 6. Losses

Identity losses:

- ArcFace per MRL dimension.
- SupCon on the 256-D embedding.
- Current configs weight SupCon higher than ArcFace to favor open-set
  generalization.

PAD loss:

- Implemented in `omfr/models/losses/pad_loss.py`.
- Combines BCE, focal BCE, hard-spoof weighting, and spoof OHEM.

The latest fine-tune used:

- `pad_bce_weight: 0.5`
- `pad_focal_alpha: 0.35`
- `pad_focal_gamma: 2.0`
- `pad_focal_weight: 1.0`
- `pad_hard_spoof_weight: 3.0`
- `pad_ohem_fraction: 0.4`
- `pad_ohem_weight: 2.0`
- hard spoof sensor id: `[6]` (Swipe)
- hard spoof material ids: `[4, 6, 7]`

Sensor adversarial loss:

- `SensorAdversarialHead` and `SensorAdversarialLoss` exist.
- Unknown sensor id is masked out of the adversarial loss.
- Latest padft config keeps `alpha_adv_target: 0.0`, so this is effectively
  disabled in that run.

## 7. Data Pipeline

`OMFRDataModule` is in `omfr/data/datamodule.py`.

Identity data:

- Dataset class: `IdentityDataset`
- Current identity root in padft config:
  `/home/aiserver/works/fingerprint/PAD-joint/data/processed/identity_stage2_nist302a`
- Supports split files or folder-per-identity layout.
- Supports `group_by_subject` for NIST-style subject folders.
- Supports multi-view output for SupCon via `identity_num_views`.
- Uses `GroupBalancedPKSampler` when `balanced_identity_groups: true`.

PAD data:

- Dataset class: `PADDataset`
- Current PAD root in padft config:
  `/home/aiserver/works/fingerprint/dataset`
- Current PAD datasets:
  - `LivDet2011`
  - `LivDet2013`
  - `LivDet2015`
- Labels:
  - live = 1
  - spoof = 0
- Also infers sensor and spoof material ids from path ancestors.
- Training augmentation is created by `make_pad_train_transform()`.

Joint data:

- Dataset class: `JointDataset`
- Supports samples with both identity and liveness labels.
- Current padft config leaves `joint_data_root: ""`, so no joint dataset is
  active in that run.

Phase-aware loaders:

- Phase 1: identity loader only, PK/group-balanced sampler.
- Phase 2: dict loader with `identity` and `pad` loaders.
- Phase 3: legacy path, can include identity, PAD, and joint loaders.

The current fine-tune starts directly in Phase 2 by setting:

- `phase1_max_epochs: 0`
- `phase2_epochs: 20`
- `total_epochs: 20`

## 8. Training Pipeline

Main command shape:

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib \
/home/aiserver/miniconda3/bin/python train.py \
  --config configs/grand_fusion_sa24_padft.yaml \
  --init-from <checkpoint> \
  backbone.pretrained=false
```

Use `--init-from` when changing loss weights, LR multipliers, freeze policy, or
callback behavior. It loads compatible weights only and does not restore old
optimizer/callback state.

Use `--resume` only for crash continuation of the exact same run/config. It
restores optimizer, scheduler, callback, and loop state. It is not appropriate
for the padft experiment because the LR surgery and loss-policy changes must
take effect fresh.

`train.py` does the following:

1. Loads YAML config.
2. Applies CLI overrides like `key=value`.
3. Builds runtime config.
4. Builds and sets up `OMFRDataModule`.
5. Infers `num_classes` from the identity/joint datasets.
6. Builds `OMFRModule`.
7. Optionally loads weights with `--init-from`.
8. Builds callbacks and loggers.
9. Runs `trainer.fit()`.

Checkpoint callbacks:

- `best_phase1.ckpt`: saved only while `current_phase == 1`.
- `best_phase2.ckpt`: saved only while `current_phase == 2`.
- `last.ckpt`: rolling crash/resume checkpoint.

For deploy/eval, use `best_phase2.ckpt`, not `last.ckpt`, when it exists.

## 9. Phase Scheduler

`omfr/callbacks/phase_scheduler.py` implements the dynamic state machine.

Phase 1:

- Identity foundation.
- Runs until `val/cascaded_IM` plateaus or `phase1_max_epochs` is reached.
- ArcFace scale/margin are warmed up.

Phase 2:

- PAD adaptation.
- Data loader switches to identity + PAD.
- `OMFRModule.apply_phase2_lr_multipliers()` rewrites optimizer group LRs and
  scheduler base LRs.
- PAD loss weight and related ramps are applied.

The padft run uses Phase 2 directly, with `phase1_max_epochs: 0`.

## 10. Evaluation Pipeline

Main command shape:

```bash
MPLCONFIGDIR=/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 \
/home/aiserver/miniconda3/bin/python eval.py \
  --checkpoint <checkpoint> \
  --skip_identity \
  --livdet_roots \
    /home/aiserver/works/fingerprint/dataset/LivDet2011 \
    /home/aiserver/works/fingerprint/dataset/LivDet2013 \
    /home/aiserver/works/fingerprint/dataset/LivDet2015 \
  --device cuda:0 \
  --batch_size 64 \
  --num_workers 8
```

`eval.py` reports:

- EER and EER threshold.
- APCER/BPCER/ACER at threshold 0.5.
- APCER/BPCER/ACER at EER threshold.
- Identity metrics if identity evaluation is not skipped.

Definitions:

- APCER: spoof accepted as live rate.
- BPCER: live rejected as spoof rate.
- ACER: average of APCER and BPCER.
- EER: point where APCER and BPCER are approximately equal.

All these are rates. Multiply by 100 for percent. For example, `0.1323` means
`13.23%`.

## 11. Current Problem Interpretation

The latest model improved the target failure mode: spoof attacks are rejected
better. That is visible in lower APCER and lower EER. However, the model became
more conservative and rejects more live samples at a fixed threshold of 0.5,
which increases BPCER.

The right next step is not simply more training. The next useful work is:

1. Recover or recreate the missing latest `best_phase2.ckpt`.
2. Calibrate threshold on validation data for the desired operating point.
3. Report results at that threshold, not only at threshold 0.5.
4. If BPCER remains too high at the chosen security target, reduce the strict
   spoof weighting or add live-protecting calibration/training constraints.

Useful threshold strategies:

- EER threshold: balanced APCER/BPCER, useful for comparison.
- Fixed APCER target: choose threshold to keep spoof acceptance below a target,
  then report BPCER.
- Fixed BPCER target: choose threshold to keep live rejection tolerable, then
  report APCER.

## 12. Practical Safety Notes

Do not delete checkpoints until the current best model is confirmed backed up.
The most valuable artifacts are:

- Latest `best_phase2.ckpt`.
- The config used to create it.
- The `metrics.csv` log.
- External evaluation output.

Logs are tiny compared with checkpoints. Most disk savings come from deleting
old `.ckpt` files, not from deleting `logs/`.

Before cleanup, run:

```bash
find . -type f \( -name '*.ckpt' -o -name '*.pth' -o -name '*.pt' \) \
  -printf '%s %TY-%Tm-%Td %TH:%TM %p\n' | sort -nr | head -80
```

Keep at least one known-good deployable checkpoint and one rollback checkpoint.

## 13. Recommended Next Actions

Immediate:

- Search backup/W&B/snapshot for `checkpoints_sa24_stage2_padft/best_phase2.ckpt`.
- If recovered, store it somewhere explicit before more cleanup.
- Calibrate a single threshold using validation data or per-dataset thresholds
  if the deployment allows dataset/sensor-specific calibration.

If the latest checkpoint cannot be recovered:

- Re-run the padft fine-tune from the best available compatible checkpoint.
- Prefer the original seed and config.
- Expect results to be close but not guaranteed identical because the trainer
  is nondeterministic (`deterministic: false`).

Possible re-run command, replacing `<source_checkpoint>` with the best recovered
or fallback checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib \
/home/aiserver/miniconda3/bin/python train.py \
  --config configs/grand_fusion_sa24_padft.yaml \
  --init-from <source_checkpoint> \
  backbone.pretrained=false
```

Then evaluate with `eval.py` and compare EER, ACER, APCER, and BPCER on
LivDet2011/2013/2015.
