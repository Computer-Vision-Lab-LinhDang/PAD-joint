# OMFR Domain Collapse Fix — Implementation Plan

**Read order:** PLAN.md (this file) → each TASK_*.md in order.

## Context

OMFR is a two-branch fingerprint model (identity + PAD) with a shared TinyViT-5M backbone. After previous refactor adding dedicated `PADStem`, the model still collapses in Phase 2:

| Metric | Observed | Expected baseline |
|---|---|---|
| LivDet2013 EER | 49.19% | ≤ 15% |
| LivDet2015 APCER / BPCER | 74% / 23% | both ≤ 20% |
| FVC2004_DB1_A rank-1 | 3.88% | ≥ 60% |
| train/identity_loss | 5 → 12 at Phase 2 boundary | flat or decreasing |
| train/pad_supcon_loss | stuck 4.5–4.7 | should decrease to < 1 |
| train/orth_loss | 0.008 (trivially zero) | actually penalizing correlation |
| train/total_loss | periodic spikes | smooth |

## Root causes

1. **SupCon for PAD with τ=0.07 is wrong for binary liveness** — gradient landscape far too sharp for 2-class; features never develop contrastive structure. Evidence: pad_supcon stuck ~4.5.
2. **Orthogonality loss over-normalized** — `/ (d_p × d_i)` = `/ 8192` makes loss trivially small before β can matter. Evidence: train/orth_loss ~ 0.008 throughout.
3. **No sensor-shortcut regularization** — PAD head memorizes LivDet sensor signatures. Evidence: BCE drops on train but APCER jumps to 74% on held-out sensors.
4. **Frequency bands mis-tuned** — thresholds `max_dim/4` and `max_dim/2` don't align with fingerprint ridge frequency. MoE routing_stats are therefore semantically meaningless as PAD features.
5. **Phase 2 transition shock** — `batch_idx % 2` alternation + sudden full α, β kick-in causes loss oscillation and identity regression.
6. **SupCon τ=0.1 for identity on 200 NIST identities doesn't transfer to FVC** — open-set generalization gap, partially augmentation problem.

## Fix order (do not reorder)

Priority follows impact × implementation cost × independence:

| # | Task | File | Expected effect |
|---|---|---|---|
| 1 | Fix PAD loss | TASK_01_pad_loss.md | pad_supcon_loss drops below 1.0 within 5 epochs |
| 2 | Fix orthogonality normalization | TASK_02_orth_loss.md | train/orth_loss becomes meaningful (0.1–1.0 range) |
| 3 | Add sensor adversarial | TASK_03_sensor_adv.md | LivDet cross-sensor APCER drops |
| 4 | Fix phase transition schedule | TASK_04_phase_schedule.md | identity_loss doesn't regress at Phase 2 |
| 5 | Re-tune frequency bands | TASK_05_frequency_bands.md | MoE routing_stats become discriminative |
| 6 | Improve identity open-set | TASK_06_identity_openset.md | FVC rank-1 lifts |

**Tasks 1–4 are blocking** — after applying them run Phase 1 + Phase 2 for 10 epochs each and check the success criteria in each task. Only proceed to 5–6 if the blockers are resolved.

## Git workflow per task

Each task is atomic and commits cleanly. For every TASK_NN:

```bash
git checkout -b fix/omfr-task-NN
# apply changes as described in TASK_NN_*.md
python -m pytest tests/ -x   # if tests exist, else skip
git add -A
git commit -m "fix(omfr): <task title> (task NN)"
```

## Validation after each task

```bash
# Short smoke run — 2 epochs P1 + 2 epochs P2
python -m omfr.train --config configs/smoke.yaml \
    --trainer.max_epochs 4 \
    --trainer.limit_train_batches 20 \
    --trainer.limit_val_batches 10
```

Check `wandb` or `lightning_logs/` for the specific metric named in each task's "success criteria" block.

## File map

```
omfr_fix/
├── PLAN.md                         # this file
├── TASK_01_pad_loss.md             # replace SupCon with focal BCE + MixUp
├── TASK_02_orth_loss.md            # fix normalization
├── TASK_03_sensor_adv.md           # add sensor adversarial head
├── TASK_04_phase_schedule.md       # soft-start + concatenated batches
├── TASK_05_frequency_bands.md      # re-tune ridge-aware thresholds
├── TASK_06_identity_openset.md     # FVC-mimicking augmentation
└── patches/
    ├── 01_supcon_pad.py            # full replacement for losses/supcon.py's PAD usage
    ├── 02_orthogonal.py            # full replacement for losses/orthogonal.py
    ├── 03_sensor_adversarial.py    # new file: losses/sensor_adversarial.py
    ├── 03_pad_dataset_sensors.py   # patch snippet for pad_dataset.py
    ├── 04_phase_scheduler.py       # full replacement for callbacks/phase_scheduler.py (if exists)
    ├── 05_frequency_gate.py        # full replacement for backbone/frequency_gate.py
    └── 06_transforms.py            # additional augmentation ops for transforms.py
```

## Hard constraints

- **Do not change backbone architecture** (TinyViT, MoE topology, PADStem) — they were just stabilized.
- **Do not change MRL dimensions** (64, 128, 256) — downstream eval depends on these.
- **Do not remove gradient checkpointing** — it's required to fit Phase 2 on the GPU.
- **Do not touch tests that pass** — if a test fails after the task, fix the test only if the change is semantic (e.g., loss signature), otherwise debug the code change.
- **All numbers must come from code or logs** — if a recommended hyperparam conflicts with the current config, prefer the config and flag the discrepancy in a code comment.

## Done-definition for the whole fix

A run of 20 epochs Phase 1 + 20 epochs Phase 2 (smoke config) satisfies all of:

- `val/identity_rank1` (on NIST val) ≥ 0.5 by end of Phase 1
- `train/identity_loss` does not increase > 20% at Phase 2 boundary
- `train/pad_supcon_loss` (or replacement) < 1.5 by end of Phase 2 (this metric may be renamed per TASK_01)
- `val/apcer` and `val/bpcer` both < 35% on held-out LivDet split
- `train/orth_loss` in range [0.05, 2.0] (non-trivial)

If any criterion fails, open a new task file `TASK_99_<issue>.md` documenting what broke and stop. Do not patch blindly.
