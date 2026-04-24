# TASK 03 — Add sensor-adversarial classifier with gradient reversal

## Problem

The most damaging failure mode in the current run is on LivDet2015:

```
APCER: 74.39%  BPCER: 23.20%
```

APCER = "spoofs classified as live". When APCER triples BPCER on a
balanced test set, it's almost always because the model learned a
**sensor shortcut** instead of a liveness concept. Evidence:

- `train/pad_bce_loss` drops 0.69 → 0.25 (the model clearly learns
  *something* on the train split).
- But `val/apcer` stays near chance, meaning whatever it learned does
  not generalize across sensors.

LivDet 2013/2015/2017 use 4 sensors per year (Biometrika, DigitalPersona,
GreenBit, HiScan, plus Italdata/CrossMatch/Swipe in 2013). Each sensor
has a distinctive noise pattern, DPI, and post-processing pipeline.
Without an explicit anti-sensor regularizer, minimizing BCE on a
balanced live/spoof split is easily solved by learning the sensor ID
of the train-time spoof samples, which leaks zero information at test
time.

## Solution

Add a **sensor classifier** that tries to predict the sensor ID from
`pad_features`, connected through a **Gradient Reversal Layer (GRL)**.
The sensor classifier learns normally, but the GRL flips the sign of
gradients flowing back into `pad_features`. This forces `pad_features`
to become sensor-invariant — a classic DANN (Ganin et al., 2015) setup.

Loss term:

```
L_sensor_adv = CE(sensor_classifier(GRL(pad_features, lam)), sensor_id)
```

Total PAD loss becomes:

```
L_pad = alpha * (FocalBCE + MixUpConsistency) + alpha_adv * L_sensor_adv
```

`alpha_adv` ramps from 0 to `lam_max=0.1` over the first half of Phase 2,
following the DANN schedule `lam(p) = 2/(1+exp(-10*p)) - 1`.

## Files changed

### 1. `omfr/models/losses/sensor_adversarial.py` (new file)

Create with content from `patches/03_sensor_adversarial.py`.

### 2. `omfr/data/datasets/pad_dataset.py` — emit sensor labels

Modify `PADDataset` to parse and return a sensor index.

Find the `_load_from_directory` method. Currently it only stores
`(path, label)`. Extend to `(path, label, sensor_id)`:

Add near the top of the class (next to LIVE_DIRS / SPOOF_DIRS):

```python
# Ordered list of sensor directory names seen in LivDet. The index in
# this list becomes the sensor_id integer used by the adversarial head.
# Add new sensors at the end to keep indices stable across runs.
SENSOR_NAMES = (
    'Biometrika',
    'CrossMatch',
    'DigitalPersona',
    'GreenBit',
    'HiScan',
    'Italdata',
    'Swipe',
    'Orcanthus',          # LivDet 2017
    'Unknown',            # catch-all; sensor_id = len - 1
)
```

Replace `_load_from_directory`:

```python
def _load_from_directory(self, root: Path):
    scan_root = self._resolve_split_root(root)

    for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=True):
        dirnames[:] = [name for name in dirnames if not name.startswith('.')]
        current_dir = Path(dirpath)

        for filename in filenames:
            img_path = current_dir / filename
            if img_path.suffix.lower() not in self.IMG_EXTS:
                continue

            label = self._infer_label(img_path, scan_root)
            if label is None:
                continue

            if self.sensor and not self._matches_sensor(img_path, scan_root):
                continue

            sensor_id = self._infer_sensor_id(img_path, scan_root)
            self.samples.append((img_path, label, sensor_id))
```

Add a new helper:

```python
def _infer_sensor_id(self, path: Path, scan_root: Path) -> int:
    """Walk ancestors until a directory matches a known sensor name.
    Returns index into SENSOR_NAMES, or the 'Unknown' index if no match.
    """
    normalized_targets = {
        self._normalize_sensor_name(name): idx
        for idx, name in enumerate(self.SENSOR_NAMES)
    }
    unknown_idx = self.SENSOR_NAMES.index('Unknown')
    for ancestor in self._ancestors_until(path.parent, scan_root):
        idx = normalized_targets.get(self._normalize_sensor_name(ancestor.name))
        if idx is not None:
            return idx
    return unknown_idx
```

Update `_load_from_csv` similarly:

```python
def _load_from_csv(self, root: Path, csv_file: Path):
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row.get('relative_path') or row.get('path', '')
            label_str = row.get('label', '0')
            path = root / rel_path
            label = self.LABEL_LIVE if label_str.lower() in {'live', '1', 'alive'} else self.LABEL_SPOOF
            if path.is_file() and path.suffix.lower() in self.IMG_EXTS:
                sensor_id = int(row.get('sensor_id', -1))
                if sensor_id < 0:
                    # Fall back to directory-based inference
                    sensor_id = self._infer_sensor_id(path, root)
                self.samples.append((path, label, sensor_id))
```

Update `__getitem__`:

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    path, label, sensor_id = self.samples[idx]
    try:
        image = self._load_image(path)
    except (OSError, IOError):
        return self.__getitem__((idx + 1) % len(self.samples))

    if self.transform is not None:
        image = self.transform(image)

    return {
        'images': image,
        'liveness_labels': torch.tensor(label, dtype=torch.long),
        'sensor_labels': torch.tensor(sensor_id, dtype=torch.long),
    }
```

Update `get_liveness_labels` — no change needed. Add:

```python
def get_sensor_labels(self) -> List[int]:
    return [sensor for _, _, sensor in self.samples]

@property
def num_sensors(self) -> int:
    return len(self.SENSOR_NAMES)
```

### 3. `omfr/models/omfr.py` — instantiate head + loss

In `OMFRModule.__init__`, after `self.pad_head = PADHead()`:

```python
# Sensor-adversarial head: classifies sensor ID from pad_features with
# gradient reversal. Forces pad_features to be sensor-invariant. See
# TASK_03. The sensor count comes from PADDataset.SENSOR_NAMES; pass
# num_sensors via config if it differs.
from omfr.models.losses.sensor_adversarial import (
    SensorAdversarialHead, SensorAdversarialLoss,
)
num_sensors = int(config.get("num_sensors", 9))
self.sensor_adv_head = SensorAdversarialHead(
    in_features=PADHead.PAD_FEATURES_DIM,
    num_sensors=num_sensors,
    hidden_dim=128,
)
self.sensor_adv_loss = SensorAdversarialLoss()
self.alpha_adv: float = 0.0        # ramped by PhaseScheduler, max ~0.1
self.lam_adv: float   = 0.0        # GRL lambda, ramped alongside alpha_adv
```

### 4. `omfr/models/omfr.py` — use in `_phase2_pad_step` and `_phase3_step`

In `_phase2_pad_step`, after computing `pad_out`:

```python
# Sensor adversarial: flip gradient into pad_features via GRL.
# No-op when lam_adv == 0 (early in Phase 2).
sensor_labels = self._unpack_sensor_labels(batch)
if sensor_labels is not None and self.lam_adv > 0:
    sensor_logits = self.sensor_adv_head(
        pad_out["pad_features"], lam=self.lam_adv,
    )
    l_sensor = self.sensor_adv_loss(sensor_logits, sensor_labels)
else:
    l_sensor = images.new_zeros(())

# ... existing l_focal / l_mixup / l_orth computation ...
loss = (self.alpha * (l_focal + l_mixup)
        + self.alpha_adv * l_sensor
        + self.beta * l_orth
        + self.gamma * l_balance)
self.log("train/pad_sensor_adv", l_sensor, sync_dist=True)
```

And add the helper to `OMFRModule`:

```python
@staticmethod
def _unpack_sensor_labels(batch):
    if isinstance(batch, dict):
        return batch.get("sensor_labels")
    return None
```

For `_phase3_step`, joint dataset (MSU-FPAD) likely doesn't have sensor
labels. Guard the same way — if `sensor_labels` is None, skip.

### 5. `omfr/models/omfr.py` — param groups

In `configure_optimizers`, add a new group:

```python
# Sensor adversarial head — standard LR, no special treatment.
{
    "params": list(self.sensor_adv_head.parameters()),
    "lr":     lr,
    "name":   "sensor_adv_head",
},
```

### 6. `PhaseSchedulerCallback` — ramp `alpha_adv` and `lam_adv`

During Phase 2 (epochs 20–39), the schedule should be:

```
epoch   lam_adv    alpha_adv
20      0.00       0.00
22      0.10       0.01
25      0.30       0.03
30      0.60       0.06
35      0.90       0.09
40+     1.00       0.10
```

Follow the DANN schedule: `lam(p) = 2 / (1 + exp(-10 * p)) - 1` where
`p = (epoch - phase2_start) / (phase2_end - phase2_start)`.

## Success criteria

After 10 epochs of Phase 2:

- `train/pad_sensor_adv` is roughly flat around `log(num_sensors)` ≈ 2.2 for 9 sensors — the adversary is confused, meaning pad_features have lost sensor info. If it keeps dropping < 1.5, the adversarial strength is too weak (lam_adv too low).
- `val/apcer` on LivDet2015 drops from 74% to < 45%.
- `val/bpcer` stays roughly in 15–30% range (should not degrade significantly; if it jumps > 40%, `alpha_adv` is too high).

## Risks

- **If `sensor_id = 'Unknown'` dominates** (directory layout not matching SENSOR_NAMES), the adversarial head sees a trivial problem (always predict Unknown) and contributes nothing useful. Before training, run `python -c "from omfr.data.datasets.pad_dataset import PADDataset; ds = PADDataset(root='data/raw/livdet2015', split='train'); from collections import Counter; print(Counter(ds.get_sensor_labels()))"` and verify sensor 8 (Unknown) is < 5% of samples.
- **Joint (MSU-FPAD) does not have sensor labels.** Phase 3 already bypasses sensor adversarial when labels are missing. But if you later add a joint dataset with sensors, extend the joint dataset loader the same way.

## Quick verification (before first training run)

```python
from omfr.data.datasets.pad_dataset import PADDataset
from collections import Counter

for year in (2013, 2015, 2017):
    ds = PADDataset(root=f'data/raw/livdet{year}', split='train')
    counts = Counter(ds.get_sensor_labels())
    print(f"LivDet{year}: {dict(sorted(counts.items()))}")
    # Expect 3-4 sensors with roughly equal counts, not all "Unknown".
```
