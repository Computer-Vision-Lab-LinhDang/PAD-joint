# TASK 04 — Fix Phase 2 transition: soft-start α/β + concatenated batching

## Problem

Two separate issues compound at the Phase 1 → Phase 2 boundary:

1. **`batch_idx % 2` alternation** (in `training_step` Phase 2 branch) gives the optimizer two completely different loss landscapes on consecutive steps. Adam's β₂=0.999 second-moment estimate needs ~1000 steps to adapt to a new curvature, but the loss signature flips every step. Consequence: the observed `train/total_loss` spikes every ~20–50 steps (image 1, top-left panel).

2. **α, β kick in sharply.** Current PhaseScheduler (file not shown but behavior is inferrable) ramps from 0 to target over a handful of steps at Phase 2 start. Combined with the alternation, the first PAD batch at epoch 20 sees `α * (l_bce + l_supcon)` at near-target magnitude without the optimizer having any history on that loss's gradients. This is visible in `train/identity_loss` jumping from ~6 to ~12 right at step 500 (Phase 2 start with 25 steps/epoch).

The two interact: alternation makes α/β ramp per-batch-type rather than per-step. Softening only one of them doesn't fix the problem.

## Solution

Two changes, neither optional:

1. **Concatenate identity + PAD into a single mini-step.** Each step processes both batches in one forward and computes `L_total = L_id + α·L_pad + β·L_orth + γ·L_balance`. The optimizer sees a single stable landscape and Adam's second moment converges normally.

2. **Replace the α/β ramp with a cosine soft-start over the first 5 epochs of Phase 2.** Start at 0.01 × target, reach 1.0 × target at epoch `phase2_start + 5`. Same for `lam_adv` (from TASK_03).

## Files changed

### 1. `omfr/models/omfr.py` — rewrite the Phase 2 branch of `training_step`

Find:

```python
elif self.current_phase == 2:
    if isinstance(batch, dict) and "identity" in batch and "pad" in batch:
        if batch_idx % 2 == 0:
            return self._phase2_identity_step(batch["identity"])
        else:
            return self._phase2_pad_step(batch["pad"])
    if isinstance(batch, dict) and "liveness_labels" in batch:
        return self._phase2_pad_step(batch)
    return self._phase2_identity_step(batch)
```

Replace with:

```python
elif self.current_phase == 2:
    if isinstance(batch, dict) and "identity" in batch and "pad" in batch:
        return self._phase2_joint_step(batch["identity"], batch["pad"])
    if isinstance(batch, dict) and "liveness_labels" in batch:
        return self._phase2_pad_step(batch)
    return self._phase2_identity_step(batch)
```

### 2. `omfr/models/omfr.py` — add `_phase2_joint_step`

Add this method to `OMFRModule`. It replaces both `_phase2_identity_step` and `_phase2_pad_step` when both sub-batches are available:

```python
def _phase2_joint_step(self, id_batch: Any, pad_batch: Any) -> torch.Tensor:
    """Phase 2 — single optimizer step that sees BOTH identity and PAD
    batches. This is a full rewrite of the old alternating scheme
    (batch_idx % 2), which caused large total_loss oscillation every
    step and disrupted Adam's second-moment tracking.

    Forward structure:
        1. Identity branch on id_batch: full grad to backbone + id head.
        2. PAD branch on pad_batch: grad flows into pad_stem + pad_head
           (backbone stage1/2 still detached inside _run_pad; see
           OMFRModule._run_pad).
        3. Orthogonality uses PAD embedding from pad_batch and identity
           embedding from id_batch. Shapes must match on dim 0; if they
           don't, truncate to min(B_id, B_pad).
    """
    # --- Identity branch ---
    id_images, id_labels = self._unpack_identity_batch(id_batch)
    id_backbone = self._run_backbone(id_images)
    id_out = self._run_identity(id_backbone)
    id_parts = self._identity_loss(id_out["mrl_embeddings"], id_labels)
    l_balance_id = sum(id_backbone["balance_losses"])

    # --- PAD branch ---
    pad_images, liveness_labels = self._unpack_pad_batch(pad_batch)
    pad_backbone = self._run_backbone(pad_images)
    pad_out = self._run_pad(pad_backbone)
    l_focal = self.pad_focal_loss(
        pad_out["pad_logit"].squeeze(-1), liveness_labels.float(),
    )
    l_mixup = self.pad_mixup_loss(
        pad_images, liveness_labels,
        forward_fn=lambda x: self._pad_features_from_images(x),
    )
    l_balance_pad = sum(pad_backbone["balance_losses"])

    # Sensor adversarial (TASK_03)
    sensor_labels = self._unpack_sensor_labels(pad_batch)
    if sensor_labels is not None and self.lam_adv > 0:
        sensor_logits = self.sensor_adv_head(
            pad_out["pad_features"], lam=self.lam_adv,
        )
        l_sensor = self.sensor_adv_loss(sensor_logits, sensor_labels)
    else:
        l_sensor = pad_images.new_zeros(())

    # --- Orthogonality: identity embedding vs PAD embedding ---
    # Truncate to common batch size. Identity uses PKSampler (B=P*K,
    # usually 128); PAD uses BalancedPADSampler (B=pad_batch_size).
    # Typically these are equal, but guard anyway.
    Bmin = min(id_out["identity_embedding"].shape[0],
               pad_out["pad_embedding"].shape[0])
    l_orth = self.orth_loss(
        pad_out["pad_embedding"][:Bmin],
        id_out["identity_embedding"][:Bmin],
    )

    l_balance = 0.5 * (l_balance_id + l_balance_pad)

    loss = (id_parts["total"]
            + self.alpha * (l_focal + l_mixup)
            + self.alpha_adv * l_sensor
            + self.beta * l_orth
            + self.gamma * l_balance)

    self.log("train/identity_loss",  id_parts["total"], prog_bar=True, sync_dist=True)
    self.log("train/id_arcface",     id_parts["arcface"],                sync_dist=True)
    self.log("train/id_supcon",      id_parts["supcon"],                 sync_dist=True)
    self.log("train/pad_focal_loss", l_focal,                            sync_dist=True)
    self.log("train/pad_mixup_loss", l_mixup,                            sync_dist=True)
    self.log("train/pad_sensor_adv", l_sensor,                           sync_dist=True)
    self.log("train/orth_loss",      l_orth,                             sync_dist=True)
    self.log("train/balance_loss",   l_balance,                          sync_dist=True)
    self.log("train/total_loss",     loss,              prog_bar=True, sync_dist=True)
    self.log("train/alpha",          self.alpha,                         sync_dist=True)
    self.log("train/beta",           self.beta,                          sync_dist=True)
    self.log("train/lam_adv",        self.lam_adv,                       sync_dist=True)
    return loss
```

### 3. Delete the old `_phase2_identity_step` and `_phase2_pad_step`

They are still called when only one sub-batch is available (e.g., during validation sweeps or when CombinedLoader fails), so keep them but mark as legacy:

```python
def _phase2_identity_step(self, batch: Any) -> torch.Tensor:
    """Legacy path — Phase 2 identity batch with no PAD pair.
    Only invoked when CombinedLoader yields an identity-only dict.
    """
    # existing body unchanged
```

Remove the `# Detach stage3/4` duplicate forward inside `_phase2_pad_step` — the old second forward computed orth loss with detached identity but was never actually needed after TASK_01.

### 4. PhaseSchedulerCallback — cosine soft-start

Locate the callback. It likely has a method `on_train_epoch_start` that sets `self.model.alpha`, `self.model.beta`, and (now) `self.model.alpha_adv`, `self.model.lam_adv`.

Replace the linear ramp logic (if any) with:

```python
import math

def _cosine_ramp(self, epoch: int, start: int, end: int,
                 target: float, min_frac: float = 0.01) -> float:
    """Cosine ramp from min_frac*target (at epoch=start) to target (at epoch=end)."""
    if epoch < start:
        return 0.0
    if epoch >= end:
        return target
    p = (epoch - start) / max(end - start, 1)
    # Cosine ramp: 0 -> 1 as p: 0 -> 1
    ramp = 0.5 * (1.0 - math.cos(math.pi * p))
    return target * (min_frac + (1.0 - min_frac) * ramp)
```

And in the scheduler:

```python
def on_train_epoch_start(self, trainer, pl_module):
    epoch = trainer.current_epoch
    phase2_start = self.phase1_epochs                   # e.g., 20
    phase2_end   = self.phase1_epochs + self.phase2_epochs  # e.g., 40

    if epoch < phase2_start:
        pl_module.current_phase = 1
        pl_module.alpha = 0.0
        pl_module.beta = 0.0
        pl_module.alpha_adv = 0.0
        pl_module.lam_adv = 0.0
    elif epoch < phase2_end:
        pl_module.current_phase = 2
        # 5-epoch cosine soft-start so Adam state has time to adapt
        ramp_end = min(phase2_start + 5, phase2_end)
        pl_module.alpha     = self._cosine_ramp(epoch, phase2_start, ramp_end, self.alpha_target)
        pl_module.beta      = self._cosine_ramp(epoch, phase2_start, ramp_end, self.beta_target)
        # Sensor adversarial: slower ramp over full Phase 2 (DANN convention)
        pl_module.alpha_adv = self._cosine_ramp(epoch, phase2_start, phase2_end, self.alpha_adv_target)
        p = (epoch - phase2_start) / max(phase2_end - phase2_start, 1)
        pl_module.lam_adv   = 2.0 / (1.0 + math.exp(-10.0 * max(min(p, 1.0), 0.0))) - 1.0
    else:
        pl_module.current_phase = 3
        pl_module.alpha     = self.alpha_target
        pl_module.beta      = self.beta_target
        pl_module.alpha_adv = self.alpha_adv_target
        pl_module.lam_adv   = 1.0
```

With targets (after TASK_02 re-normalization and TASK_03 adv addition):

```python
alpha_target     = 1.0
beta_target      = 0.05        # NEW — was ~1.0 before TASK_02 re-norm
alpha_adv_target = 0.1
```

## Success criteria

After 10 epochs of Phase 2 on smoke config:

- `train/total_loss` is smooth (no periodic spikes exceeding 2× the rolling median).
- `train/identity_loss` at epoch 20 is within 15% of the value at epoch 19 (no jump).
- `train/alpha` and `train/beta` curves in wandb are smooth cosine ramps from ~0.01×target to target.
- Training step time increased by ~30–50% vs alternation (one extra backbone forward), which is expected.

## Risks

- **GPU memory.** Concatenated batch runs two backbones per step. With grad checkpointing enabled (already on in `TinyViTBackbone(use_grad_checkpoint=True)`), activation memory roughly doubles. If OOM, reduce `pk_P` from 32 to 24 (identity batch shrinks from 128 to 96).
- **Step count changes.** Each optimizer step now consumes one identity batch AND one PAD batch; an "epoch" ends when the shorter loader is exhausted. Total optimizer steps per phase drops. If the LR scheduler (warmup, cosine decay) is step-based, the effective schedule shortens. Check `configure_optimizers` — if the scheduler is `epoch`-based (it is in the current code: `"interval": "epoch"`), no change needed.
- **`_pad_features_from_images` in MixUp** (from TASK_01) is called by the PAD branch above and does a second PADStem forward. Combined with concatenated batching, the step does 2×Gabor + 2×backbone + 3×PADStem forwards. If this saturates GPU, lower `pad_batch_size` to 64 — PAD converges fine with smaller batches since focal loss focuses the signal.

## Why this ordering matters

TASK_04 must come after TASK_01, TASK_02, TASK_03 because `_phase2_joint_step` directly invokes `pad_focal_loss`, `pad_mixup_loss`, `sensor_adv_head`, and the re-normalized `orth_loss`. Applying TASK_04 without the others will crash on missing attributes.
