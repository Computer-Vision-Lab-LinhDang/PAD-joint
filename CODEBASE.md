# OMFR Codebase Overview

**Orthogonal Matryoshka Fingerprint Representation** — joint fingerprint identity matching and Presentation Attack Detection (PAD/liveness) in a single model.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Reference](#3-module-reference)
   - [Backbone](#31-backbone)
   - [Heads](#32-heads)
   - [Losses](#33-losses)
   - [Main Module](#34-main-module-omfrmodule)
4. [Data Pipeline](#4-data-pipeline)
   - [Datasets](#41-datasets)
   - [Samplers](#42-samplers)
   - [DataModule](#43-datamodule)
5. [Training Protocol](#5-training-protocol)
   - [Phase Schedule](#51-phase-schedule)
   - [Loss Formula](#52-loss-formula)
   - [Optimizer & Differential LR](#53-optimizer--differential-lr)
6. [Callbacks](#6-callbacks)
7. [Evaluation](#7-evaluation)
8. [Configuration](#8-configuration)
9. [Entry Points](#9-entry-points)
10. [Datasets Survey](#10-datasets-survey)

---

## 1. Project Structure

```
PAD/
├── train.py                        # Lightning Trainer entry point
├── export.py                       # ONNX export
├── configs/
│   └── base.yaml                   # Full hyperparameter config
├── omfr/
│   ├── models/
│   │   ├── backbone/
│   │   │   ├── gabor_stem.py       # LearnableGaborStem
│   │   │   ├── frequency_gate.py   # FrequencyGate (2D-FFT band energy)
│   │   │   ├── moe_ffn.py          # FreqGatedMoEFFN (4 experts, top-2)
│   │   │   └── vit_tiny.py         # ViTTinyBackbone (depth=12, MoE at {3,7,10})
│   │   ├── heads/
│   │   │   ├── identity_head.py    # IdentityHead → MRL embeddings {64,128,256}-D
│   │   │   └── pad_head.py         # PADHead → pad_logit, pad_embedding, pad_features
│   │   ├── losses/
│   │   │   ├── arcface.py          # ArcFaceLoss (margin=0.5, scale ramp 32→64)
│   │   │   ├── supcon.py           # SupConLoss (temperature=0.07)
│   │   │   └── orthogonal.py       # OrthogonalityLoss (Barlow Twins cross-corr)
│   │   └── omfr.py                 # OMFRModule (LightningModule, orchestrates all)
│   ├── data/
│   │   ├── datamodule.py           # OMFRDataModule (phase-aware DataLoader switching)
│   │   ├── transforms.py           # Fingerprint augmentations
│   │   ├── datasets/
│   │   │   ├── identity_dataset.py # FVC2004, NIST SD302/300/302a
│   │   │   ├── pad_dataset.py      # LivDet 2013/2015/2017
│   │   │   ├── joint_dataset.py    # MSU-FPAD (identity + liveness)
│   │   │   └── README.md           # Dataset download instructions + table
│   │   └── samplers/
│   │       ├── pk_sampler.py       # PKSampler (P identities × K samples)
│   │       └── balanced_pad_sampler.py  # 50/50 live/spoof sampler
│   ├── callbacks/
│   │   ├── phase_scheduler.py      # PhaseSchedulerCallback (phase transitions + ramps)
│   │   └── gradient_monitor.py     # GradientMonitor
│   └── evaluation/
│       ├── matching_eval.py        # TAR@FAR, CMC curves
│       ├── pad_eval.py             # ACER, APCER, BPCER
│       └── integrated_eval.py      # Cascaded accuracy (PAD → identity)
└── omfr_implementation_plan.md     # Master design document
```

---

## 2. Architecture Overview

```
Input (B, 1, 224, 224) — grayscale fingerprint
       │
       ▼
┌─────────────────────────────┐
│  LearnableGaborStem         │  (B,1,H,W) → (B,3,H,W)
│  8 orientations × 4 freqs  │  learnable Gabor filter bank
└────────────┬────────────────┘
             │ (B, 3, 224, 224)
             ▼
┌─────────────────────────────────────────────────────┐
│  ViTTinyBackbone (depth=12, embed_dim=192)           │
│                                                     │
│  PatchEmbed 16×16 → 196 tokens                      │
│  + CLS token + positional embedding                 │
│                                                     │
│  Layer  1 │ TransformerBlock (standard FFN)         │
│  Layer  2 │ TransformerBlock (standard FFN)         │
│  Layer  3 │ TransformerBlock + FreqGatedMoEFFN ◄─── PAD tap (layer3_tokens)
│  Layer  4 │ TransformerBlock (standard FFN)         │
│  Layer  5 │ TransformerBlock (standard FFN)         │
│  Layer  6 │ TransformerBlock (standard FFN)         │
│  Layer  7 │ TransformerBlock + FreqGatedMoEFFN ◄─── PAD tap (layer7_tokens)
│  Layer  8 │ TransformerBlock (standard FFN)         │
│  Layer  9 │ TransformerBlock (standard FFN)         │
│  Layer 10 │ TransformerBlock + FreqGatedMoEFFN      │
│  Layer 11 │ TransformerBlock (standard FFN)         │
│  Layer 12 │ TransformerBlock (standard FFN) ◄────── Identity tap (layer12_tokens + CLS)
│                                                     │
└─────┬──────────────────────┬────────────────────────┘
      │                      │
      ▼                      ▼
┌──────────────┐    ┌─────────────────────────────────┐
│ IdentityHead │    │ PADHead                         │
│              │    │                                 │
│ layer12 +    │    │ layer3_tokens + layer7_tokens   │
│ cls_token    │    │ + routing_stats_3 + routing_stats_7
│              │    │                                 │
│ → 256-D emb  │    │ → pad_logit     (scalar)        │
│ → MRL:       │    │ → pad_embedding (32-D)          │
│   64-D emb   │    │ → pad_features  (128-D)         │
│  128-D emb   │    └─────────────────────────────────┘
│  256-D emb   │
└──────────────┘
```

**Key design insight — MoE routing as PAD signal:** Live fingerprints exhibit diverse expert routing across MoE layers (high token entropy). Spoof images tend toward uniform routing (low entropy). The PADHead explicitly fuses `routing_stats` (expert weights, token entropy) with spatial token features.

---

## 3. Module Reference

### 3.1 Backbone

#### `gabor_stem.py` — LearnableGaborStem

| | |
|---|---|
| Input | `(B, 1, H, W)` grayscale fingerprint |
| Output | `(B, 3, H, W)` multi-channel enhanced fingerprint |
| Params | 8 orientations × 4 frequencies = 32 filter bank; kernel_size=31 |

Converts grayscale to a 3-channel output (grouped conv) suitable for ViT's `in_chans=3`. Filter parameters (orientation, frequency) are learnable starting from Gabor initialization.

---

#### `frequency_gate.py` — FrequencyGate

| | |
|---|---|
| Input | `tokens (B, 196, 192)` — spatial token sequence |
| Output | `gate_input (B, 196, 3)` — per-token [low, mid, high] band energy |

Computes 2D FFT of the 14×14 token grid (per channel), then integrates power spectral density into three radial bands. This produces a frequency fingerprint for each token that drives expert selection in the MoE-FFN.

---

#### `moe_ffn.py` — FreqGatedMoEFFN

| | |
|---|---|
| Input | `tokens (B, 196, 192)` |
| Output | `(output_tokens (B,196,192), routing_stats dict)` |
| Config | 4 experts, top-2 routing, hidden_dim=768 |

**Routing pipeline:**
```
gate_input  = FrequencyGate(tokens)        # (B, 196, 3)  — band energies
gate_logits = Linear(3→4)(gate_input)      # (B, 196, 4)
expert_weights = Softmax(gate_logits)       # (B, 196, 4)
top2_weights, top2_indices = TopK(k=2)
output = Σ_{k∈top2} weight_k × Expert_k(tokens)
```

**`routing_stats` dict:**
```python
{
    'expert_weights': (B, 196, 4),   # full softmax distribution
    'token_entropy':  (B, 196),       # H(p) = -Σ p·log(p)
    'balance_loss':   scalar,         # CV²(expert_load) — penalizes imbalance
}
```

**Balance loss:** CV² = Var(expert_load) / (Mean(expert_load)² + ε). Target: each expert receives ~25% of tokens.

**Parameter budget per MoE layer:** 4 × 295K (experts) + gate Linear(3→4) ≈ 1.18M params. FLOPs: ~2× standard FFN.

---

#### `vit_tiny.py` — ViTTinyBackbone

| | |
|---|---|
| Input | `(B, 3, 224, 224)` |
| Output | dict — see below |
| Config | embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, patch_size=16 |
| Params | ~8.35M (backbone only) |

**Output dict:**
```python
{
    'layer3_tokens':   (B, 196, 192),   # after block 3 (MoE), spatial only
    'layer7_tokens':   (B, 196, 192),   # after block 7 (MoE), spatial only
    'layer10_tokens':  (B, 196, 192),   # after block 10 (MoE), spatial only
    'layer12_tokens':  (B, 196, 192),   # after block 12 + final LayerNorm
    'cls_token':       (B, 192),        # CLS token after final LayerNorm
    'routing_stats': {
        3:  {'expert_weights': (B,196,4), 'token_entropy': (B,196)},
        7:  {'expert_weights': (B,196,4), 'token_entropy': (B,196)},
        10: {'expert_weights': (B,196,4), 'token_entropy': (B,196)},
    },
    'balance_losses': [scalar, scalar, scalar],  # one per MoE layer
}
```

**Why ViT-Tiny over hierarchical (TinyViT):**
- Flat architecture → global attention at every layer, no resolution downsampling
- Uniform FFN structure → MoE drop-in is trivial (no stage boundary complications)
- Routing statistics are learned signals, not hard-coded stage outputs

**Utility methods for differential LR:**
- `get_layer_params(layers: List[int])` — yields params for specific 1-indexed layers
- `get_moe_params()` — yields params for MoE-FFN blocks only (layers 3, 7, 10)
- `get_stem_and_embed_params()` — yields patch_embed, cls_token, pos_embed params

---

### 3.2 Heads

#### `identity_head.py` — IdentityHead

| | |
|---|---|
| Input | `{'layer12_tokens': (B,196,192), 'cls_token': (B,192)}` |
| Output | `{'identity_embedding': (B,256), 'mrl_embeddings': {64: (B,64), 128: (B,128), 256: (B,256)}}` |

**Internal pipeline:**
```
layer12_tokens (B,196,192) + cls_token (B,192)
    → Prepend CLS → (B, 197, 192)
    → Linear(192→256)                                         → (B, 197, 256)
    → StructuralAttentionBlock with RPE(Δx, Δy)               → (B, 197, 256)
    → AttentivePooling (learned query)                         → (B, 256)
    → LayerNorm → L2Norm                                      → identity_embedding (B, 256)
    → MRL slices: [:64], [:128], [:256]                       → 3 sub-embeddings
    → Per-dim ArcFaceClassifier (weights NOT shared)           → 3 logit vectors
```

**RPE (Relative Position Encoding):** Learned `(Δx, Δy)` bias table of shape `(num_heads, 27, 27)` for the 14×14 token grid. Added to attention logits before softmax — CLS-to-token and token-to-CLS biases are zero. Fingerprint-critical: minutia position on the spatial grid (core vs delta vs periphery) matters for identity.

**MRL (Matryoshka Representation Learning):** The first N dimensions of the 256-D embedding form a complete representation at N dimensions. All three are L2-normalized independently.

**`forward_embedding_only(inputs, dim=None)`:** Returns embedding without ArcFace logit computation. Used in:
- Phase 2 PAD batch (identity embedding for L_orth, with detached layer12/CLS)
- Inference / deployment (no classifier weights needed)
- ONNX export (excludes large ArcFace weight matrices)

---

#### `pad_head.py` — PADHead

| | |
|---|---|
| Input | `{'layer3_tokens': (B,196,192), 'layer7_tokens': (B,196,192), 'routing_stats_3': dict, 'routing_stats_7': dict}` |
| Output | `{'pad_logit': (B,1), 'pad_embedding': (B,32), 'pad_features': (B,128)}` |

**Internal pipeline:**
```
── Feature path ──
layer3_tokens (B,196,192) → TokenAttentionPool → (B, 256)
layer7_tokens (B,196,192) → TokenAttentionPool → (B, 256)
feat_combined = Concat → (B, 512)

── Routing path (12 features per layer × 2 layers = 24) ──
expert_weights per layer: mean(dim=1) → (B,4), std(dim=1) → (B,4)
token_entropy per layer:  [mean, std, max, min] → (B,4)
route_combined = Concat → (B, 24)

── Fusion ──
all_features = Concat(feat, route) → (B, 536)
    → LayerNorm → Linear(536→256) → GELU → Dropout(0.1) → Linear(256→128)
    → pad_features (128-D)

── Parallel output branches (both from 128-D, NOT serial) ──
pad_features → Linear(128→32) → L2Norm → pad_embedding (32-D, for SupCon + L_orth)
pad_features → Linear(128→1)            → pad_logit     (scalar, for BCE)
```

**Routing statistics as PAD signal:** Entropy [mean, std, max, min] captures spatial distribution — spoof has uniformly low entropy across tokens (synthetic texture), live has high variance (rich micro-texture at core/delta, low at background). Expert load std captures routing diversity — spoof routes tokens uniformly to the same experts.

---

### 3.3 Losses

#### `arcface.py` — ArcFaceLoss

```
margin: 0.5,  scale: 32.0 (Phase 1) → 64.0 (Phase 2 warmup end)
```

One ArcFaceLoss per MRL dimension (`"64"`, `"128"`, `"256"`). `set_scale(s)` method allows the `PhaseSchedulerCallback` to ramp the scale during Phase 2 warmup.

**Identity loss:** `L_id = (1/3) × (L_arc_64 + L_arc_128 + L_arc_256)`

---

#### `supcon.py` — SupConLoss

```
temperature: 0.07
contrast_mode: all
```

Applied to `pad_features` (128-D) with liveness labels. Pushes live embeddings together and away from spoof embeddings in feature space.

---

#### `orthogonal.py` — OrthogonalityLoss

Barlow Twins cross-correlation between `pad_embedding` (32-D) and `identity_embedding` (256-D). Penalizes any shared information between the two representation spaces, enforcing orthogonality.

```
L_orth = || C_cross ||_F²    where C_cross = Z_pad^T Z_id  (normalized)
```

---

### 3.4 Main Module: OMFRModule

**File:** `omfr/models/omfr.py`

`OMFRModule(LightningModule)` orchestrates the full forward pass and phase-aware training.

**Full forward path:**
```python
enhanced    = gabor(images)                              # (B,3,224,224)
backbone_out = backbone(enhanced)                        # dict with layer taps
id_out       = identity_head(backbone_out)               # MRL embeddings
pad_out      = pad_head(backbone_out)                    # logit + embeddings
```

**Phase-specific training steps:**

| Phase | Step method | Losses computed |
|-------|------------|-----------------|
| 1 | `_phase1_step` | `L_id + γ·L_balance` |
| 2 (ID batch) | `_phase2_identity_step` | `L_id + β·L_orth + γ·L_balance` |
| 2 (PAD batch) | `_phase2_pad_step` | `α·(L_SupCon + L_BCE) + β·L_orth + γ·L_balance` |
| 3 | `_phase3_step` | `L_id* + α·(L_SupCon+L_BCE) + β·L_orth + γ·L_balance` |

*Phase 3 ArcFace is spoof-masked: ArcFace only sees `live_mask == 1` samples.

**Phase 2 PAD batch gradient isolation:**
```
Layers 1-7 + PAD Head  →  full gradient (L_PAD + L_orth + L_balance)
Layers 8-12            →  NO gradient from PAD batch (layer12/CLS detached)
Identity Head          →  NO gradient from PAD batch (inputs detached)
```
`layer12_tokens` and `cls_token` are `.detach()`ed before passing to the identity head during PAD batches. This prevents L_PAD from destabilizing deep identity features learned in Phase 1. Layers 8-12 only receive gradients from identity batches (`_phase2_identity_step`).

**Validation metrics logged:**
- `val/pad_accuracy` — sigmoid(pad_logit) > 0.5 accuracy
- `val/identity_rank1` — cosine nearest-neighbor rank-1 accuracy (256-D)
- `val/cascaded_IM` — rank-1 accuracy computed only on PAD-accepted (predicted-live) samples; **primary checkpoint metric**

---

## 4. Data Pipeline

### 4.1 Datasets

#### `identity_dataset.py` — IdentityDataset

**Supported:** FVC2004, NIST SD302, NIST SD300, NIST SD302a

**Directory layout:**
```
root/
    subject_0001/
        img1.bmp
        img2.bmp
    subject_0002/
        ...
```

**Split support:** If `root/split_train.txt` exists (one relative path per line), it is used; otherwise all images are loaded and class labels derive from parent directory names.

**Returns:** `{'images': (1,224,224), 'identity_labels': LongTensor}`

---

#### `pad_dataset.py` — PADDataset

**Supported:** LivDet 2013, LivDet 2015, LivDet 2017

**Directory layout (LivDet convention):**
```
root/
    Train/
        Live/  *.bmp
        Fake/  *.bmp
    Test/
        Live/  *.bmp
        Fake/  *.bmp
```

**Optional:** `sensor` parameter filters to a specific sensor subdirectory. CSV fallback supported (`labels_{split}.csv`).

**Returns:** `{'images': (1,224,224), 'liveness_labels': LongTensor}` — 0=spoof, 1=live

---

#### `joint_dataset.py` — JointDataset

**Supported:** MSU-FPAD v2.0

**Directory layout:**
```
root/
    subject_001/
        live/   *.bmp
        spoof/  *.bmp
```

**Returns:** `{'images': (1,224,224), 'identity_labels': LongTensor, 'liveness_labels': LongTensor}`

---

### 4.2 Samplers

#### PKSampler

For identity training. Samples exactly P=32 identities per batch, K=4 images per identity → batch_size = 128. Guarantees multiple positives per batch for ArcFace and contrastive learning.

#### BalancedPADSampler

For PAD training. Maintains 50/50 live/spoof ratio per batch. batch_size=128.

---

### 4.3 DataModule

**File:** `omfr/data/datamodule.py` — `OMFRDataModule(LightningDataModule)`

Switches DataLoader configuration per phase:

| Phase | DataLoader(s) | Mode |
|-------|--------------|------|
| 1 | `identity_loader` (PKSampler, single loader) | — |
| 2 | `CombinedLoader({'identity': ..., 'pad': ...})` | `"min_size"` |
| 3 | `CombinedLoader({'identity': ..., 'pad': ..., 'joint': ...})` | `"min_size"` |

**CombinedLoader `"min_size"` mode:** Both loaders deliver batches simultaneously every step. The `training_step` receives `batch = {"identity": id_batch, "pad": pad_batch}` and selects which sub-batch to process based on `batch_idx`:
- Phase 2: `batch_idx % 2 == 0` → identity step, odd → PAD step
- Phase 3: `batch_idx % 3` → round-robin across identity/pad/joint

`current_phase` is set by `PhaseSchedulerCallback` at each phase transition.

---

## 5. Training Protocol

### 5.1 Phase Schedule

```
Epoch  0-19  │ Phase 1 — Identity Foundation
             │   • Only identity datasets (PKSampler)
             │   • Backbone learns global fingerprint features
             │   • MoE routing initializes
             │   • α=0, β=0 (PAD/orth losses inactive)

Epoch 20-39  │ Phase 2 — PAD Integration
             │   • Alternating ID/PAD batches (CombinedLoader)
             │   • 5-epoch linear warmup: α: 0→1.0, β: 0→0.1
             │   • ArcFace scale ramp: 32→64
             │   • PADHead weights re-initialized at epoch 20

Epoch 40-59  │ Phase 3 — Joint Refinement
             │   • All three datasets (identity + PAD + joint)
             │   • α=1.0, β=0.1 fixed
             │   • ArcFace scale=64, spoof-masked
             │   • Cascaded metric drives checkpointing
```

---

### 5.2 Loss Formula

```
L = L_Identity + α·L_PAD + β·L_orth + γ·L_balance

where:
    L_Identity = (1/3) × Σ_d∈{64,128,256} L_ArcFace_d(emb_d, id_labels)
    L_PAD      = L_SupCon(pad_features_128, live_labels)
               + L_BCE(pad_logit, live_labels)
    L_orth     = Barlow-Twins cross-corr(pad_embedding_32, identity_embedding_256)
    L_balance  = Σ_{l∈{3,7,10}} CV²(expert_load_l)

    α = 0→1.0  (ramped in Phase 2)
    β = 0→0.1  (ramped in Phase 2)
    γ = 0.01   (fixed throughout)
```

---

### 5.3 Optimizer & Differential LR

**Optimizer:** AdamW (β₁=0.9, β₂=0.999, weight_decay=0.05)
**Scheduler:** CosineAnnealingLR (T_max=60, eta_min=1e-6)
**Base LR:** 1e-4

| Param group | LR multiplier | Rationale |
|-------------|:---:|-----------|
| `gabor` | 0.1× | Stable learned filters, slow drift |
| `backbone_embed` | 1.0× | Patch embed, pos embed, CLS token |
| `backbone_early` (layers 1-6) | 1.0× | PAD-relevant layers |
| `backbone_late` (layers 7-12) | 1.0× | Identity-relevant layers |
| `moe_experts` | 2.0× | Newly initialized, need to catch up |
| `identity_head` | 1.0× | Standard |
| `pad_head` (Phase 2) | 2.0× | Higher LR to catch up in Phase 2 |
| `arcface` | 10.0× | Large sparse gradient classifiers |

MoE params are deduplicated from `backbone_early`/`backbone_late` groups to avoid double-counting.

---

## 6. Callbacks

### PhaseSchedulerCallback

**File:** `omfr/callbacks/phase_scheduler.py`

Called at `on_train_epoch_start`. Handles:
- Phase transitions at epochs 20 and 40
- Linear ramp of `pl_module.alpha`, `pl_module.beta`, and ArcFace `s` during Phase 2 warmup
- PADHead weight re-initialization at Phase 1→2 transition (Xavier uniform)
- Notifies `datamodule.current_phase` so the DataModule switches loaders

### GradientMonitor

**File:** `omfr/callbacks/gradient_monitor.py`

Logs per-layer gradient norms to TensorBoard for debugging vanishing/exploding gradients.

---

## 7. Evaluation

### `matching_eval.py`
- `compute_tar_at_far(scores, labels, far)` → TAR at FAR=1e-4
- CMC curve up to rank-10

### `pad_eval.py`
- `compute_acer(logits, labels)` → ACER = (APCER + BPCER) / 2
- APCER = Attack Presentation Classification Error Rate (spoof → live)
- BPCER = Bona Fide Presentation Classification Error Rate (live → spoof)

### `integrated_eval.py`
- `compute_cascaded_accuracy(id_scores, id_labels, pad_logits, live_labels)` → cascaded identification rate: only PAD-accepted samples participate in identity matching

---

## 8. Configuration

**File:** `configs/base.yaml`

All hyperparameters in one place. Key sections:

```yaml
backbone:
  embed_dim: 192 | depth: 12 | num_heads: 3
  moe_layers: [3, 7, 10] | num_experts: 4 | top_k: 2

losses:
  alpha: 0.0       # PAD weight (ramped)
  beta:  0.0       # Orth weight (ramped)
  gamma: 0.01      # Balance (fixed)

data:
  pk_p: 32         # identities per batch
  pk_k: 4          # samples per identity → batch = 128
  pad_batch_size: 128

trainer:
  precision: 16-mixed
  gradient_clip_val: 1.0
  accumulate_grad_batches: 2   # effective batch ≈ 256

checkpoint:
  monitor: val/cascaded_IM     # primary metric
  mode: max | save_top_k: 3
```

---

## 9. Entry Points

### `train.py`

```bash
python3 train.py --config configs/base.yaml
python3 train.py --config configs/base.yaml --resume checkpoints/last.ckpt
```

Instantiates `OMFRModule`, `OMFRDataModule`, `PhaseSchedulerCallback`, `GradientMonitor`, and `lightning.Trainer`. TensorBoard logs to `logs/`.

### `export.py`

```bash
python3 export.py --checkpoint checkpoints/omfr-best.ckpt --output exports/omfr.onnx
```

Exports to ONNX opset 17. Input shape: `[1, 1, 224, 224]` (grayscale). Runs `onnx-simplifier` if available. Exports two graph variants: identity-only head and PAD head.

---

## 10. Datasets Survey

| Dataset | Task | Images | Subjects/Sensors | Access |
|---------|------|--------|-----------------|--------|
| FVC2004 | Identity | 3,200 | 100 subjects, 4 sensors | Free (registration) |
| NIST SD302 | Identity | 27,000+ | 200 subjects, 6 sensors | $825 |
| NIST SD300 | Identity | 9,000+ | 75 subjects | $850 |
| NIST SD302a | Identity | 15,000+ | 200 subjects, quality-annotated | Same as SD302 |
| LivDet 2013 | PAD | 14,230 | 4 sensors | Free (registration) |
| LivDet 2015 | PAD | 19,000 | 4 sensors | Free (registration) |
| LivDet 2017 | PAD | 20,148 | 4 sensors | Free (registration) |
| MSU-FPAD v2.0 | Joint | 4,968 | 100 subjects | Request form |

**LivDet protocol:** All three editions use the **unknown attack** protocol — spoof materials in test set differ from training set materials.

**Dataset assignment by phase:**
```
Phase 1:  FVC2004, NIST SD302, NIST SD300, NIST SD302a
Phase 2:  + LivDet 2013, LivDet 2015, LivDet 2017
Phase 3:  + MSU-FPAD v2.0
```

**Expected directory layout:**
```
/data/fingerprint/
    identity/
        fvc2004/         DB{N}/subject_{NNN}_{impression}.bmp
        nist_sd302/      {subject_id}/*.png
        nist_sd300/      {subject_id}/*.png
        nist_sd302a/     {subject_id}/*.png
    pad/
        livdet2013/      Train/{Live,Fake}/*.bmp  Test/{Live,Fake}/*.bmp
        livdet2015/      Train/{Live,Fake}/*.bmp  Test/{Live,Fake}/*.bmp
        livdet2017/      Train/{Live,Fake}/*.bmp  Test/{Live,Fake}/*.bmp
    joint/
        msu_fpad/        subject_{NNN}/{live,spoof}/*.bmp
```

Override data roots in `configs/base.yaml` under `data.identity_data_root`, `data.pad_data_root`, `data.joint_data_root`.
