# TASK 02 — Fix orthogonality loss normalization

## Problem

Current `OrthogonalityLoss.forward`:

```python
loss = cross_corr.pow(2).sum() / (d_p * d_i)
```

With `d_p=32` and `d_i=256`, the denominator is 8192. The cross-correlation matrix has 8192 entries, but when PAD and identity embeddings come from separate branches of a freshly initialized network, their batch-wise cross-correlation is dominated by O(1/√B) noise per entry, giving a total Frobenius-squared norm on the order of `8192 × 1/B ≈ 64` for B=128. After normalization: 64 / 8192 ≈ 0.008.

That's exactly what the training log shows: `train/orth_loss ≈ 0.008` throughout. This value is **statistically indistinguishable from random noise correlation**. The loss never actually penalizes real subspace overlap — it just reflects finite-batch variance.

Beta multiplier can't fix this: even at β=100, the effective contribution is 0.8, and the gradient magnitude per embedding dimension is `β × 2/sqrt(8192) ≈ β × 0.022`. To get a useful signal we'd need β > 1000, at which point the gradient dynamics become unstable.

## Solution

Two targeted changes:

1. **Drop the `d_p × d_i` normalization**, replace with `max(d_p, d_i)` — this keeps the loss O(1) with respect to subspace size but doesn't crush the actual correlation signal. With the new normalizer, a "perfectly aligned" 32-D subspace embedded in 256-D gives loss ≈ 1, random embeddings give loss ≈ 32/128 = 0.25, fully orthogonal gives 0.
2. **Add a gate on the batch size** — the estimator is biased for small B because sample correlations converge slowly. Multiply by `(B / (B - 1))` (Bessel-like correction) and only compute the loss when `B ≥ max(d_p, d_i) / 4` to avoid punishing undersampled batches.

## Files changed

### 1. `omfr/models/losses/orthogonal.py` (full replacement)

Replace the file content with `patches/02_orthogonal.py`.

### 2. `omfr/models/omfr.py` — β schedule in `PhaseSchedulerCallback`

The β target value in the callback (not shown in the files but implied by the train logs) was probably ≈ 1.0. With the new normalization, the loss operates in the 0.1–1.0 range instead of the 0.005–0.015 range, so reduce the β target by ~20x.

Search for where `self.model.beta` is set in the phase scheduler (likely in a file called `omfr/callbacks/phase_scheduler.py` or similar). Find the final beta value (look for something like `beta_target=1.0` or `self.model.beta = self.beta_target`) and change it to `0.05`.

If the phase scheduler isn't in the files we have, add a comment in `omfr.py` __init__:

```python
# NOTE(task-02): β target should be ~0.05 after orth loss re-normalization
# (TASK_02). If PhaseSchedulerCallback is not updated, the old ~1.0 target
# will overpower identity loss.
```

and log a warning in `_phase2_identity_step` when `self.beta > 0.2`:

```python
if self.beta > 0.2 and self.trainer.global_step < 50:
    print(f"[omfr] WARNING: beta={self.beta:.3f} is high after orth re-norm; "
          f"see TASK_02. Expected target ~0.05.")
```

## Success criteria

After 2 epochs of Phase 2 on the smoke config:

- `train/orth_loss` lands in [0.05, 2.0] range (not stuck at 0.008).
- `train/identity_loss` does NOT suddenly jump by > 50% at Phase 2 boundary (with β=0.05, the orth loss contribution is bounded).
- Cross-correlation between `pad_embedding` and `identity_embedding` on a val batch (compute manually in a notebook) shows |corr| < 0.3 on average — up from the ~0 that was already achieved trivially but without actual learning.

## Risks

- If β target is not reduced: the new (larger) orth_loss will now overpower identity_loss, making identity_rank1 collapse. This is the symmetric failure to the current bug. **Always pair this task with the β adjustment.**
- The `B ≥ max(d_p, d_i) / 4` gate means for `d_i=256`, we need B ≥ 64. If `pad_batch_size` in the config is < 64, this loss will silently return 0 and the β→α symmetry breaks. Check the config and bump `pad_batch_size` to 128 if needed.

## Diagnostic to run before merging

Drop this into a scratch notebook and execute after Phase 1 checkpoint loads:

```python
import torch
from omfr.models.losses.orthogonal import OrthogonalityLoss

# Known cases
B = 128
d_p, d_i = 32, 256

orth = OrthogonalityLoss()

# Case 1: identical embeddings padded — should give ~1.0 (high overlap)
z_p = torch.randn(B, d_p)
z_i = torch.cat([z_p, torch.zeros(B, d_i - d_p)], dim=1)
print("overlap:", orth(z_p, z_i).item())     # expected > 0.5

# Case 2: independent random — should give small but non-trivial
z_p = torch.randn(B, d_p)
z_i = torch.randn(B, d_i)
print("random:",  orth(z_p, z_i).item())     # expected 0.1 – 0.4

# Case 3: truly orthogonal (orthogonal columns) — should give near zero
U, _, V = torch.linalg.svd(torch.randn(max(d_p, d_i), B), full_matrices=False)
z_p = U[:d_p].T[:B]
z_i = V[:d_i].T[:B] if d_i <= B else torch.randn(B, d_i)
print("ortho:",   orth(z_p, z_i).item())     # expected < 0.1
```

If the three cases don't order correctly (overlap > random > ortho), the normalization is still wrong. Do not merge until this sanity check passes.
