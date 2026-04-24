# TASK 05 — Re-tune frequency gate bands to match fingerprint ridge physics

## Problem

`frequency_gate.py` uses:

```python
low_thresh  = max_dim / 4.0
high_thresh = max_dim / 2.0
```

with band definitions:

- `low`:  `dist < max_dim/4`   — intended as ridge flow
- `mid`:  `max_dim/4 ≤ dist < max_dim/2`  — intended as minutiae
- `high`: `dist ≥ max_dim/2`   — intended as pores / spoof texture

At 500 DPI with images resized from native capture (~400–500 px, ~8–12 mm
of skin) to 224×224, ridge period works out to roughly 8–12 px in image
space, i.e. spatial frequency ~0.08–0.12 cycles/pixel.

The MoE FFN operates at two spatial resolutions:
- Stage s2: 28×28 (after patch_embed stride 8).
- Stage s3: 14×14 (after stage-2 downsampling stride 16).

At **28×28**, the Nyquist-domain distance of a ridge (12-px cycle in
original 224-px image → 224/12 ≈ 19 cycles in full image → at 28×28 grid,
≈ `28 × (1/12) × 8 = 19` grid-cycles). In `fftfreq` units (the code uses
`torch.fft.fftfreq(h) * h`, giving integer cycle counts over the grid):
ridge peaks around index **10–14** out of 14 Nyquist.

At **14×14**, ridge aliases down to index **5–7** out of 7 Nyquist.

Mapping those to the current thresholds:

| Band | Stage s2 (max_dim=28) | Stage s3 (max_dim=14) |
|---|---|---|
| low | dist < 7.0 | dist < 3.5 |
| mid | 7 ≤ dist < 14 | 3.5 ≤ dist < 7 |
| high | dist ≥ 14 | dist ≥ 7 |

Ridge frequency at stage s2 is around distance **10–14** — that's the
upper half of the "mid" band. At stage s3 it's **5–7** — the upper "mid"
band. In both cases, **ridge energy is split between "mid" and "high"**,
not concentrated where the code claims. Pore/texture signal (frequencies
above ridge) falls into the same "high" band as ridge harmonics, so the
"high" band is a muddy mixture.

The gate produces 3-band energies that don't correspond to 3 semantically
distinct concepts, so the MoE gate_proj cannot learn a sensible routing.
This is consistent with `train/balance_loss` being nearly flat throughout
training — experts are all doing similar things because the routing
signal has no meaningful structure.

## Solution

Re-tune so each band captures a distinct fingerprint-physics concept.
Target:

- **Band 0 (sub-ridge)**: global structure, finger boundary, sensor
  illumination gradient. Frequency < 60% of ridge.
- **Band 1 (ridge)**: the actual ridge-valley pattern. Centered on
  measured ridge frequency.
- **Band 2 (super-ridge)**: minutiae discontinuities, pore texture,
  sensor noise, spoof-material artifacts. Frequency > ridge harmonics.

Compute band thresholds **per-resolution** from ridge frequency, not
from grid size.

## Measuring ridge frequency (one-off calibration)

Run this once on a sample of NIST SD302 images to determine `ridge_freq_grid`:

```python
# scripts/calibrate_ridge_freq.py
import torch, numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

def ridge_freq_at_resolution(img_path, grid_h, img_size=224):
    """Return the dominant spatial frequency (in cycles per grid) after
    the image is downsampled to grid_h × grid_h.
    """
    img = Image.open(img_path).convert('L').resize((img_size, img_size))
    x = TF.to_tensor(img)                       # (1, 224, 224)
    # Downsample to grid resolution with mean pooling (matches stride-8/16
    # conv reduction roughly).
    stride = img_size // grid_h
    pool = torch.nn.functional.avg_pool2d(x.unsqueeze(0), stride, stride).squeeze(0)[0]
    # Radial FFT spectrum
    spec = torch.fft.fft2(pool, norm='ortho').abs()
    fy = torch.fft.fftfreq(grid_h) * grid_h
    fx = torch.fft.fftfreq(grid_h) * grid_h
    dist = (fy[:, None]**2 + fx[None, :]**2).sqrt()
    # Bin by integer radius, ignore DC
    radii = torch.arange(1, grid_h // 2 + 1, dtype=torch.float32)
    profile = torch.zeros(len(radii))
    for i, r in enumerate(radii):
        mask = ((dist >= r - 0.5) & (dist < r + 0.5))
        if mask.any():
            profile[i] = spec[mask].mean()
    peak_idx = int(profile.argmax().item())
    return float(radii[peak_idx].item())

# Run on ~50 NIST images, take median
paths = list(Path('data/raw/nist_sd302/train').rglob('*.png'))[:50]
for grid_h in (28, 14):
    freqs = [ridge_freq_at_resolution(p, grid_h) for p in paths]
    print(f"grid {grid_h}: median ridge freq = {np.median(freqs):.2f}")
```

Expected output (rough calibration, verify!):

```
grid 28: median ridge freq = 11.5
grid 14: median ridge freq =  5.8
```

If your measured values differ by more than ~20%, use yours instead of
the defaults below.

## Files changed

### 1. `omfr/models/backbone/frequency_gate.py` (full replacement)

Use `patches/05_frequency_gate.py`. Key differences from v1:

- Accepts optional `ridge_freq` per resolution from config.
- Defaults hardcoded to measured values (`ridge_freq_28=11.5`, `ridge_freq_14=5.8`).
- Bands: `low = dist < 0.6 * ridge`, `mid = [0.6, 1.6] * ridge`, `high = dist > 1.6 * ridge`.
- Still caches masks per (h, w) for efficiency.

### 2. No changes to `moe_ffn.py` or `tiny_vit.py`

The gate interface is unchanged — just the internal threshold computation
differs.

### 3. Config entry (optional)

In your training config, if you want to override the defaults:

```yaml
model:
  frequency_gate:
    ridge_freq_28: 11.5     # measured on NIST SD302
    ridge_freq_14: 5.8
```

And in the backbone/moe_ffn init path, thread these through to the
`FrequencyGate` constructor. If this isn't worth the plumbing, just
leave the hardcoded defaults — they'll be close enough on any 500 DPI
fingerprint dataset.

## Success criteria

After 5 epochs of Phase 1 on smoke config (just identity training):

- `train/balance_loss` actually changes over training (not flat). Initial
  value ≈ 1.0 (uniform routing), should decrease to ≤ 0.8 as experts
  specialize.
- Expert usage histogram (log manually in `_phase1_step` by tracking
  `expert_weights_gateonly.mean(dim=(0,1))`) should show diverging values
  across experts — ideally one expert ends up handling > 40% of tokens,
  which means routing found structure. Before this fix, all experts end
  up near 25% (uniform) throughout.
- PAD head's `routing_stats` features (passed through `_extract_routing_features`)
  become informative: if you freeze the rest of the model and train just
  the PAD head on these 24-D features alone, you should beat chance on
  a held-out sensor.

## Risks

- If the measured ridge frequency is way off from defaults (e.g., due
  to upstream resize pipeline differences), the new bands will be just
  as bad as the old ones. **Always run the calibration script before
  merging.**
- Changing band semantics means the gate projection weights learned in
  the current checkpoints become meaningless. **Discard Phase 1 checkpoints
  and restart from scratch.** This is a one-time cost.
- Stage s2 has 784 tokens × 3 bands = 2352 energy values computed per
  batch. Make sure `FrequencyGate._get_masks` cache is hit (not
  recomputed every forward) — the cache key is `(h, w, device)`, which
  stays constant per stage, so this should be fine. Verify by logging
  `len(self._mask_cache)` after a few steps — should be 2 (one per
  resolution).
