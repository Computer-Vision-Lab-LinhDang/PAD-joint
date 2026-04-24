# TASK 01 — Fix PAD loss (replace SupCon with focal BCE + liveness-aware augmentation)

## Problem

Current `_phase2_pad_step` applies:

```python
l_supcon = self.supcon_loss(pad_out["pad_features"], liveness_labels)  # τ=0.07
l_bce    = self.bce_loss(pad_out["pad_logit"].squeeze(-1), liveness_labels.float())
loss = self.alpha * (l_supcon + l_bce) + self.beta * l_orth + ...
```

This is wrong for two reasons.

1. **SupCon with τ=0.07 on 2 classes is degenerate.** SupCon was designed for ImageNet-scale class counts where each anchor has ~N/1000 positives and ~N negatives. On a 128-sample batch with 64 live + 64 spoof, each anchor has 63 positives and 64 negatives. Dividing similarity by 0.07 gives logit scale 14, meaning softmax concentrates all gradient on the single hardest negative — which is noise. The observed `train/pad_supcon_loss` plateau at 4.5–4.7 (vs theoretical min ~0.016 for the batch structure) confirms the features never develop contrastive geometry.

2. **BCE alone is under-regularized across the LivDet live/spoof imbalance of materials and sensors.** Within a balanced batch, the BCE gradient from easy samples (obvious live, obvious Ecoflex spoof) dominates, and hard samples (high-quality silicone on optical sensor) contribute negligibly. This is exactly the `focal loss` failure mode.

## Solution

Replace `L_pad = alpha * (SupCon + BCE)` with `L_pad = alpha * (FocalBCE + MixUpConsistency)`.

- **Focal BCE** weights hard examples by `(1 - p_t)^gamma` — pores, micro-texture edge cases contribute more gradient.
- **MixUp consistency** on `pad_features` enforces `pad_features(λx₁ + (1-λ)x₂) ≈ λ·pad_features(x₁) + (1-λ)·pad_features(x₂)`. This acts as a manifold-smoothness regularizer that's much better-suited to binary PAD than SupCon.

## Files changed

### 1. `omfr/models/losses/focal_bce.py` (new file)

Create with content from `patches/01_focal_bce.py`.

### 2. `omfr/models/losses/mixup_consistency.py` (new file)

Create with content from `patches/01_mixup_consistency.py`.

### 3. `omfr/models/omfr.py` — imports

Replace:
```python
from omfr.models.losses.supcon import SupConLoss
```
with:
```python
from omfr.models.losses.supcon import SupConLoss  # still used for identity branch
from omfr.models.losses.focal_bce import FocalBCELoss
from omfr.models.losses.mixup_consistency import MixUpConsistency
```

### 4. `omfr/models/omfr.py` — `__init__`

Inside `OMFRModule.__init__`, after the existing `self.supcon_loss = SupConLoss(...)` block, add:

```python
# PAD branch — focal BCE (hard-example mining) + MixUp consistency
# (manifold smoothness). Replaces SupCon(τ=0.07), which was degenerate
# on binary PAD and plateaued at ~4.5 (see TASK_01).
self.pad_focal_loss = FocalBCELoss(gamma=2.0, alpha=0.5)
self.pad_mixup_loss = MixUpConsistency(alpha=0.4)
```

Remove the old `self.supcon_loss = SupConLoss(temperature=0.07)` line — this is not used elsewhere for PAD after this task. `self.supcon_identity_loss` stays untouched.

### 5. `omfr/models/omfr.py` — `_phase2_pad_step`

Replace the `l_supcon` / `l_bce` lines:

```python
# --- OLD ---
l_supcon  = self.supcon_loss(pad_out["pad_features"], liveness_labels)
l_bce     = self.bce_loss(pad_out["pad_logit"].squeeze(-1), liveness_labels.float())
# ...
loss = (self.alpha * (l_supcon + l_bce)
        + self.beta * l_orth
        + self.gamma * l_balance)
self.log("train/pad_supcon_loss", l_supcon, sync_dist=True)
self.log("train/pad_bce_loss",    l_bce,    sync_dist=True)
```

with:

```python
# --- NEW ---
l_focal = self.pad_focal_loss(
    pad_out["pad_logit"].squeeze(-1), liveness_labels.float(),
)
l_mixup = self.pad_mixup_loss(
    images, liveness_labels,
    forward_fn=lambda x: self._pad_features_from_images(x),
)
# ...
loss = (self.alpha * (l_focal + l_mixup)
        + self.beta * l_orth
        + self.gamma * l_balance)
self.log("train/pad_focal_loss", l_focal, sync_dist=True)
self.log("train/pad_mixup_loss", l_mixup, sync_dist=True)
```

### 6. `omfr/models/omfr.py` — `_phase3_step`

Apply the same replacement in `_phase3_step` where the PAD loss terms are computed. Same import of `self.pad_focal_loss` / `self.pad_mixup_loss`, same log-key names.

### 7. `omfr/models/omfr.py` — new helper

Add this method to `OMFRModule` (anywhere after `_run_pad`):

```python
def _pad_features_from_images(self, images: torch.Tensor) -> torch.Tensor:
    """Forward (Gabor_pad → PADStem → pad_head) producing pad_features only.

    Used by MixUpConsistency to compute features on mixed images without
    re-running the full backbone. We intentionally skip the backbone here —
    PADStem is the only trainable path on the PAD side that depends on
    raw pixels, so a mixup consistency constraint is applied there.
    """
    gabor_pad = self.gabor_pad(images)                       # (B, 8, 224, 224)
    pad_stem_feat = self.pad_stem(gabor_pad)                 # (B, 256)
    # Feed only pad_stem to fusion — stage1/2 and routing features are
    # batch-dependent and can't be linearly mixed, so we pass zeros for
    # those and let the consistency constraint target pad_stem's manifold.
    B = images.shape[0]
    device = images.device
    zeros_s1 = torch.zeros(B, 64,  56, 56, device=device)
    zeros_s2 = torch.zeros(B, 128, 28, 28, device=device)
    zeros_rs = {
        "expert_weights": torch.zeros(B, 1, 4, device=device),
        "token_entropy":  torch.zeros(B, 1,    device=device),
    }
    # NOTE: PADHead currently indexes routing_stats with .mean(dim=1), so
    # feeding a length-1 sequence of zeros produces zero routing features.
    out = self.pad_head({
        "pad_stem_feat":     pad_stem_feat,
        "stage1_feat":       zeros_s1,
        "stage2_feat":       zeros_s2,
        "routing_stats_s2":  zeros_rs,
        "routing_stats_s3a": zeros_rs,
        "routing_stats_s3b": zeros_rs,
    })
    return out["pad_features"]   # (B, 128)
```

## Success criteria

After 5 epochs of Phase 2 on the smoke config:

- `train/pad_focal_loss` decreases monotonically below 0.4 (from initial ~0.69).
- `train/pad_mixup_loss` in [0.05, 0.5] range — not zero (collapse) and not > 1 (unstable).
- `val/bpcer` and `val/apcer` both < 60% on held-out LivDet sensor (replaces the 74% / 23% plateau).

## Risks

- **MixUpConsistency computes two extra PADStem forwards** per PAD batch. Training step time may rise ~15–20%. If memory becomes tight, lower `pad_batch_size` in the config by 25% — PADStem gradients don't benefit much from huge batches anyway.
- If `train/pad_mixup_loss` collapses to 0 in the first epoch, the pad_features space has collapsed. Check `pad_features.std()` — if < 1e-3, something earlier broke (pad_stem frozen? gradient killed by detach?). Revert and investigate before continuing.
