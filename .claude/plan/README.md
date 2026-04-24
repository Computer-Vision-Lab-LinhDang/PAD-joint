# OMFR Fix — Agent Runbook

**You are a Claude agent tasked with fixing OMFR domain collapse. Start here.**

## 1. Orient yourself

Read these files in this exact order:

1. `PLAN.md` — overall strategy, success criteria, constraints.
2. `TASK_01_pad_loss.md` — replace degenerate SupCon with focal BCE + MixUp.
3. `TASK_02_orth_loss.md` — fix orthogonality loss normalization.
4. `TASK_03_sensor_adv.md` — add gradient-reversal sensor classifier.
5. `TASK_04_phase_schedule.md` — soft-start + concatenated batches.
6. `TASK_05_frequency_bands.md` — ridge-aware MoE gate.
7. `TASK_06_identity_openset.md` — FVC cross-dataset lift.

Do not skip PLAN.md. The task ordering is load-bearing — TASK_04 breaks if
TASK_01, TASK_02, TASK_03 aren't done first.

## 2. Per-task workflow

For each task NN in order 01 → 06:

```
1. Read TASK_NN_*.md top to bottom
2. Inspect referenced files in the omfr/ repo
3. If patches/NN_*.py exists, copy it into the repo at the target path
   specified in the task file
4. Apply all "Files changed" edits in the task file in order
5. Run the smoke-test command from PLAN.md
6. Check the task's "Success criteria" block against wandb / logs
7. Commit on a branch fix/omfr-task-NN
8. Only then proceed to TASK_NN+1
```

If a success criterion fails:

```
STOP. Do not guess. Do not proceed to the next task.
Create TASK_99_<descriptor>.md documenting:
  - What criterion failed
  - What metric values you observed instead
  - What diagnostic you would run next
Hand control back to the user.
```

## 3. Hard rules

- **Never change** TinyViT topology, MRL dims (64/128/256), grad
  checkpointing flag. These are called out as stable.
- **Never remove** detach on stage1/stage2/stage3/stage4 features in
  `_run_pad`. That detach is the main defense against PAD loss
  corrupting the identity backbone.
- **Never merge** TASK_02 without running the three-case diagnostic in
  that file. Bad orth normalization silently kills training.
- **Never** hand-edit patches/NN_*.py to "simplify" — they are the
  canonical implementation.
- **Always** regenerate a Phase 1 checkpoint from scratch after TASK_05
  (frequency band change invalidates learned gate_proj weights).

## 4. Quick sanity after each task

Minimum run to confirm the task didn't break training:

```bash
python -m omfr.train \
    --config configs/smoke.yaml \
    --trainer.max_epochs 4 \
    --trainer.limit_train_batches 20 \
    --trainer.limit_val_batches 10 \
    --data.num_workers 2
```

Expected runtime: 5–10 min on a single A100, 15–25 min on a 3090.

At the end, check:

- Training completes without OOM.
- `total_loss` is finite (no NaN) throughout.
- The specific metric named in the task's "Success criteria" is in range.

## 5. What "done" looks like

After all 6 tasks merged, a 40-epoch run (20 Phase 1 + 20 Phase 2)
must satisfy the PLAN.md "Done-definition" block:

- `val/identity_rank1` ≥ 0.5 by end of Phase 1
- `train/identity_loss` no > 20% jump at Phase 2 boundary
- `train/pad_focal_loss` < 0.4 by end of Phase 2
- `val/apcer`, `val/bpcer` both < 35%
- `train/orth_loss` in [0.05, 2.0]

If any of these fails, report back — do not try to fix silently.

## 6. Files you will touch

Reference map of every file in the omfr repo that gets modified:

| File | Modified by task |
|---|---|
| `omfr/models/losses/focal_bce.py` | 01 (NEW) |
| `omfr/models/losses/mixup_consistency.py` | 01 (NEW) |
| `omfr/models/losses/orthogonal.py` | 02 (REPLACE) |
| `omfr/models/losses/sensor_adversarial.py` | 03 (NEW) |
| `omfr/data/datasets/pad_dataset.py` | 03 |
| `omfr/models/omfr.py` | 01, 02, 03, 04 |
| `omfr/callbacks/phase_scheduler.py` (or equivalent) | 02, 04 |
| `omfr/models/backbone/frequency_gate.py` | 05 (REPLACE) |
| `omfr/data/transforms.py` | 06 |
| `omfr/data/datasets/identity_dataset.py` | 06 |
| `omfr/data/datamodule.py` | 06 |
| `configs/*.yaml` | 03, 06 |

## 7. If the phase scheduler callback can't be found

The task files assume a callback at `omfr/callbacks/phase_scheduler.py`
with methods `on_train_epoch_start` mutating `pl_module.alpha`,
`pl_module.beta`, `pl_module.current_phase`. This file wasn't included
in the original code dump.

If you can't find it:

1. Search for assignments to `pl_module.alpha` or `self.model.alpha`:
   `grep -rn 'alpha' omfr/callbacks/ omfr/`.
2. Wherever that logic lives, apply TASK_04's schedule edits there.
3. If no such callback exists, create `omfr/callbacks/phase_scheduler.py`
   from scratch following the schedule in TASK_04, and register it in
   the training config `trainer.callbacks`.

## 8. Do not do these things

- **Do not run Phase 3 training** until all tasks are merged and Phase 2
  numbers meet success criteria. Phase 3 compounds the problems.
- **Do not "just tune hyperparameters"** if a success criterion fails.
  The hyperparams in each task file came from reasoning about the code,
  not cargo-cult. If something's wrong, the fix is documented — search
  the task for "Risks" and "Diagnostic".
- **Do not use `torch.compile`** on the modified model. MoE side-channel
  stats through wrapper attributes don't survive graph capture.
- **Do not delete** `_phase2_identity_step` or `_phase2_pad_step` (they
  become fallback paths after TASK_04 introduces `_phase2_joint_step`).

## 9. On questions

If a task is ambiguous, re-read it carefully — the task files are
written to be self-sufficient. If after careful reading the task is
still underspecified, pause and open `TASK_99_clarify_<what>.md` rather
than guessing.
