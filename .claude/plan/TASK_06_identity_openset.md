# TASK 06 — Identity open-set generalization (FVC cross-dataset transfer)

## Problem

Even after Phase 1 training stabilizes (fixing TASK_01–04), FVC rank-1
numbers are severely lagging NIST intra-dataset rank-1. Observed:

| Dataset | Rank-1 @ 256-D |
|---|---|
| FVC2000_DB1_A (optical) | 51.75% |
| FVC2000_DB2_A | 49.12% |
| FVC2002_DB1_A | 17.12% |
| FVC2004_DB1_A (optical, higher distortion) | 5.38% |
| FVC2004_DB2_A | 10.75% |

The drop from FVC2000 → FVC2002 → FVC2004 corresponds to increasing
distortion intensity. FVC2004 specifically was designed to have heavier
finger pressure variation, partial occlusions, and rotation — precisely
the invariances a fingerprint embedding should learn but currently
doesn't.

Train set (NIST SD302) uses carefully-captured plain + rolled impressions.
Impressions per identity are relatively similar. The ArcFace + SupCon
loss can minimize train objective while learning a "same-capture-session"
representation that lacks the invariances FVC demands.

## Solution

Two complementary changes:

1. **FVC-mimicking augmentation**: add an augmentation preset
   `"fingerprint_hard"` that simulates FVC2004 distortion characteristics
   — stronger elastic deformation, partial occlusion (RandomErasing with
   larger patches), and "partial capture" cropping (off-center ROIs).
   Apply only to identity dataset, not PAD (PAD has its own augmentation
   designed for cross-sensor).

2. **Multi-view SupCon for identity**: currently SupCon sees one
   augmented view per sample. Switch to two views per sample (standard
   SimCLR / SupCon protocol). This doubles the contrastive signal and
   teaches the embedding to be invariant to the augmentation.

## Files changed

### 1. `omfr/data/transforms.py` — add `FingerprintHardTransform`

Append to the existing `transforms.py` (keep all current classes):

```python
class FingerprintHardTransform:
    """FVC2004-mimicking augmentation for identity training.

    Adds:
      - Stronger elastic deformation (alpha=40, sigma=7) — simulates
        heavy finger pressure variation.
      - Wider rotation range (+/- 25 deg) — FVC capture allows larger
        finger misalignment.
      - RandomResizedCrop with scale (0.6, 1.0) — partial captures.
      - Larger RandomErasing (up to 30% of area).
    """

    def __init__(self, output_size: int = 224):
        self.transforms = [
            RandomRotation90(),
            ElasticDeformation(alpha=40.0, sigma=7.0, p=0.7),
            # Partial capture: crop down to as little as 60% of image
            _RandomResizedPartial(output_size=output_size,
                                  scale=(0.6, 1.0), p=0.7),
            RandomBrightnessContrast(brightness=0.4, contrast=0.4, p=0.9),
            RandomGaussianNoise(std=0.03, p=0.4),
            _RandomErasingBig(p=0.35, max_area=0.30),
        ]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


class _RandomResizedPartial:
    """Random crop of image area in [scale_min, scale_max], resize to output_size."""
    def __init__(self, output_size: int, scale=(0.6, 1.0), p: float = 0.7):
        self.output_size = output_size
        self.scale = scale
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return F.interpolate(img.unsqueeze(0), size=self.output_size,
                                 mode='bilinear', align_corners=False).squeeze(0)
        C, H, W = img.shape
        scale = random.uniform(*self.scale)
        h = int(H * (scale ** 0.5))
        w = int(W * (scale ** 0.5))
        top  = random.randint(0, max(H - h, 0))
        left = random.randint(0, max(W - w, 0))
        img = img[:, top:top + h, left:left + w]
        return F.interpolate(img.unsqueeze(0), size=self.output_size,
                             mode='bilinear', align_corners=False).squeeze(0)


class _RandomErasingBig:
    """RandomErasing with up to max_area patch size."""
    def __init__(self, p: float = 0.35, max_area: float = 0.30):
        self.p = p
        self.max_area = max_area

    def __call__(self, img):
        if random.random() > self.p:
            return img
        C, H, W = img.shape
        area = random.uniform(0.05, self.max_area) * H * W
        aspect = random.uniform(0.3, 3.3)
        h = int((area * aspect) ** 0.5)
        w = int((area / aspect) ** 0.5)
        if h >= H or w >= W:
            return img
        top  = random.randint(0, H - h)
        left = random.randint(0, W - w)
        img = img.clone()
        img[:, top:top + h, left:left + w] = 0.0
        return img
```

Update `get_transforms`:

```python
def get_transforms(split: str = 'train', output_size: int = 224,
                   preset: str = 'fingerprint'):
    """
    Args:
        split: 'train' | 'val' | 'test'
        preset: 'fingerprint' (default, current behavior) or
                'fingerprint_hard' (FVC-mimicking)
    """
    if split == 'train':
        if preset == 'fingerprint_hard':
            return FingerprintHardTransform(output_size=output_size)
        return FingerprintTrainTransform(output_size=output_size)
    return FingerprintValTransform(output_size=output_size)
```

### 2. `omfr/data/datasets/identity_dataset.py` — 2-view mode

Add an optional 2-view mode. Modify `__init__`:

```python
def __init__(
    self,
    root: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    image_size: int = 224,
    dataset_name: str = 'NIST_SD302',
    group_by_subject: bool = False,
    num_views: int = 1,          # NEW
):
    super().__init__()
    # ... existing ...
    self.num_views = num_views
```

Modify `__getitem__`:

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    path, label = self.samples[idx]
    try:
        image = self._load_image(path)
    except (OSError, IOError):
        return self.__getitem__((idx + 1) % len(self.samples))

    if self.num_views == 1:
        img_out = image if self.transform is None else self.transform(image)
        return {
            'images': img_out,
            'identity_labels': torch.tensor(label, dtype=torch.long),
        }

    # Multi-view: apply transform num_views times independently.
    if self.transform is None:
        # Without transform, all views would be identical. That defeats
        # the whole point. Fall back to single view.
        return {
            'images': image,
            'identity_labels': torch.tensor(label, dtype=torch.long),
        }

    views = torch.stack(
        [self.transform(image) for _ in range(self.num_views)],
        dim=0,
    )   # (num_views, C, H, W)
    return {
        'images': views,                                         # (V, C, H, W)
        'identity_labels': torch.tensor(label, dtype=torch.long),
    }
```

### 3. `omfr/data/datamodule.py` — pass `num_views` to identity dataset

In `setup`:

```python
num_views = int(self._cfg_get("identity_num_views", "data.identity_num_views", default=2))
preset    = str(self._cfg_get("identity_preset",   "data.identity_preset",   default="fingerprint_hard"))

from omfr.data.transforms import get_transforms

if identity_root:
    self.identity_ds = IdentityDataset(
        root=identity_root, split="train",
        group_by_subject=group_by_subject,
        num_views=num_views,
        transform=get_transforms('train', output_size=224, preset=preset),
    )
```

### 4. `omfr/models/omfr.py` — handle 2-view batches

In `_unpack_identity_batch`, detect the 2-view case and flatten:

```python
@staticmethod
def _unpack_identity_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        images = batch["images"]
        labels = batch["identity_labels"]
    else:
        images, labels = batch[0], batch[1]

    # Multi-view: (B, V, C, H, W) -> (B*V, C, H, W)
    if images.ndim == 5:
        B, V, C, H, W = images.shape
        images = images.reshape(B * V, C, H, W)
        labels = labels.repeat_interleave(V)
    return images, labels
```

This preserves the SupCon / ArcFace interfaces — they just see `B*V`
samples with labels repeated `V` times, which is exactly the correct
multi-view SupCon setup.

### 5. Config additions

```yaml
data:
  identity_num_views: 2
  identity_preset: "fingerprint_hard"
  # With 2 views and pk_P=32 / pk_K=4, actual batch becomes 256 samples.
  # Reduce pk_K to 2 if memory is tight (keeps 32 identities * 2 samples
  # * 2 views = 128 samples total).
  pk_P: 32
  pk_K: 2        # was 4
```

## Success criteria

After 20 epochs of Phase 1 on full identity dataset (not smoke):

- `val/identity_rank1` ≥ 0.7 on NIST val split.
- FVC2004_DB1_A rank-1 ≥ 0.30 (up from 0.0538) — this is the tightest
  bottleneck; any lift here is a win.
- FVC2000_DB1_A rank-1 ≥ 0.70 (up from 0.5175).
- `train/id_supcon` converges to a lower value than single-view baseline
  (more positives per anchor = tighter clustering), probably in the
  0.8–1.5 range.

## Risks

- **Memory doubles with 2 views.** If you're already borderline OOM after
  TASK_04 (concatenated batch), cut `pk_K` from 4 to 2 (config above),
  or cut `pk_P` from 32 to 24. Do NOT reduce num_views back to 1 — the
  multi-view SupCon is the main mechanism here.
- **Data-loading becomes 2x work per sample.** num_workers in DataLoader
  needs to scale. If CPU becomes the bottleneck (GPU utilization drops
  below 70%), bump `num_workers` by ~50%.
- **Hard augmentation can hurt intra-dataset rank-1 temporarily.** It's
  normal to see NIST val rank-1 drop slightly in the first 3–5 epochs
  before it recovers past baseline. If it hasn't recovered by epoch 10,
  the augmentation strength is too high — try `preset='fingerprint'`
  first and iterate upward.

## Rationale cross-check

This task explicitly does not touch the `IdentityHead` architecture.
The original analysis noted that `AttentivePooling(M=4)` may be
destroying minutiae spatial info. That may still be true, but changing
head architecture mid-debug is high-risk and the augmentation approach
is independent. If TASK_06 lifts FVC to 30–40% but plateaus there,
revisit head architecture as a follow-up.
