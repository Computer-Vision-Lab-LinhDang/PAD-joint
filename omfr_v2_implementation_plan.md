# OMFR v2: Orthogonal Matryoshka Fingerprint Representation
## Implementation Plan — TinyViT Hybrid + Frequency-Gated MoE

> **NOTE (2026-04-19)**: This document started as the original design plan and has
> been updated to reflect the *current* implementation. Sections that diverge from
> the original plan are flagged inline with **⚠ UPDATED**. See §0 below for a
> top-level summary of the current state and known issues.

---

## 0. Current Implementation Snapshot (2026-04-19)

### 0.1 What's confirmed in code

| Area | Current value | Notes |
|---|---|---|
| Backbone | TinyViT-5M, `pretrained: true` (ImageNet-22k) | ⚠ Plan said train-from-scratch |
| Gabor stem | 8-ch, learnable | as planned |
| MoE layers | 3 total: stage 1 block 1, stage 2 block 2, stage 2 block 5 | `num_experts=4`, `top_k=2` |
| Identity decoder | **3 stacked StructuralAttentionBlocks** + AttentivePool(4 queries) | ⚠ plan had 1 block |
| PAD fusion MLP | **3-layer** (536→512→256→128) with LN + GELU + Dropout(0.1) | ⚠ plan had 2-layer |
| Identity loss | **Hybrid**: 0.7·SupCon + 0.3·MRL-ArcFace | ⚠ plan had ArcFace only |
| MoE gate | **2-layer** MLP (Linear(3→16)→GELU→Linear(16→E)) + noisy top-k (σ=0.3 train) | ⚠ plan had single Linear |
| MoE softmax | fp32 promote + `1e-4` renorm epsilon | AMP safety |
| MoE dispatch | `index_add` (not in-place slice add) | AMP safety |
| Balance loss | Switch-Transformer `E·Σ(f·P)` summed across 3 layers | ⚠ plan said CV²; optimum value = **6.0**, NOT collapse |
| ArcFace schedule | `phase1_warmup_delay=5` epochs hold s=1/m=0, then ramp s:1→32, m:0→0.5 over 5 epochs; Phase 2 ramp s:32→64 | ⚠ plan had flat s=32 Phase 1 |
| PAD augmentation | HFlip/VFlip/Rot±15°/RandomResizedCrop/ColorJitter/GaussianBlur/RandomErasing (train only) | ⚠ plan had NO PAD aug |
| PAD routing-stat gradient | **Detached** before PAD head | ⚠ stops gate from memorising sensor routing as liveness signal |
| Identity dataset grouping | `group_by_subject: true` (NIST SD300 frgp merge) | ⚠ reduces 10,300 → ~1,030 classes |
| Trainer | AMP `16-mixed`, `gradient_clip_val=1.0`, `accumulate_grad_batches=2` | |
| γ (balance weight) | **0.05** (was 0.01) | |
| ArcFace LR multiplier | **1.0** (was 10.0) | to curb gradient explosion |

### 0.2 File map (what lives where)

| Responsibility | File |
|---|---|
| Lightning module, phase steps, optimizer | `omfr/models/omfr.py` |
| TinyViT backbone + MoE wiring | `omfr/models/backbone/tiny_vit.py` |
| Gate MLP + expert dispatch + balance loss | `omfr/models/backbone/moe_ffn.py` |
| FFT band-energy gate signal | `omfr/models/backbone/frequency_gate.py` |
| Identity decoder + MRL | `omfr/models/heads/identity_head.py` |
| PAD multi-scale + routing fusion | `omfr/models/heads/pad_head.py` |
| Identity dataset + subject grouping | `omfr/data/datasets/identity_dataset.py` |
| PAD dataset + `make_pad_train_transform` | `omfr/data/datasets/pad_dataset.py` |
| Phase-aware loaders + transform wiring | `omfr/data/datamodule.py` |
| Phase schedule, ArcFace/α/β/temp ramps | `omfr/callbacks/phase_scheduler.py` |
| Per-group gradient norm telemetry | `omfr/callbacks/gradient_monitor.py` |

### 0.3 Known training pathologies (v13 checkpoint, last full run)

Observed in `logs/omfr/version_13`:
- `grad_norm/total` ≈ 100K–500K throughout training — clip at 1.0 eats >99.999% of signal.
  Dominant contributors: `arcface` (~25K–95K), `identity_head` (~50K–160K).
- `train/id_arcface` *increases* 7.8 → 33 as scale ramps 1→64. `train/id_supcon` drops 5.9→2.1 (healthy).
- `train/balance_loss = 6.0` — this is the **uniform optimum** (`3 layers × E·Σ(f·P) = 3 × 2.0`), not collapse. Smoke-test confirms per-expert weights ≈ [0.27, 0.28, 0.24, 0.21].
- `train/pad_bce = 0.03` (near-perfect train) but `eval EER = 49–51%` on LivDet test splits → catastrophic PAD overfit even in-distribution.
- `val/cascaded_IM` peaks at **0.169 end of Phase 1** then drops to 0.13 — Phase 2 makes the model strictly worse.
- `val/identity_rank1` stuck at ~17% on held-out identities.

### 0.4 Diagnosis (verified, not hypothesis)

1. **PAD overfit** caused by (a) zero augmentation until today's fix, (b) routing stats feeding a sensor-fingerprint path into PAD head, (c) PAD head ≈ 2.85M params on 45K images.
2. **ArcFace gradient** drives `grad_norm` explosion — the 10× LR multiplier from the plan was the culprit, already dropped to 1.0.
3. **Identity non-learning**: hybrid SupCon is learning (loss drops) but ArcFace ramp is fighting it. val/rank1 plateaus because SupCon alone on 1,030 classes × 20 samples is weak signal without discriminative head.

### 0.5 Fixes applied to code as of 2026-04-19

| # | Fix | File(s) |
|---|---|---|
| a | `group_by_subject` for NIST SD300 (10,300→1,030 classes, ~20 samples/class) | `identity_dataset.py`, `datamodule.py`, `base.yaml` |
| b | `phase1_warmup_delay=5` — hold s=1, m=0 during LR warmup before ArcFace ramp | `phase_scheduler.py`, `train.py`, `base.yaml` |
| c | `FrequencyGate` per-token band IFFT (was global broadcast) | `frequency_gate.py` |
| d | MoE gate 2-layer MLP + noisy top-k + fp32 softmax + `index_add` dispatch + `1e-4` renorm ε | `moe_ffn.py` |
| e | Hybrid identity loss: SupCon(0.7) + ArcFace(0.3) | `omfr.py`, `base.yaml` |
| f | Identity decoder: 1 → 3 stacked attention blocks | `identity_head.py` |
| g | PAD fusion MLP: 2 → 3 layers | `pad_head.py` |
| h | `gradient_clip_val: 10.0 → 1.0`, `γ: 0.01 → 0.05`, ArcFace LR×10 → ×1 | `base.yaml` |
| i | **PAD train augmentation pipeline** (HFlip/VFlip/Rot/RRC/ColorJitter/Blur/Erase) | `pad_dataset.py`, `datamodule.py` |
| j | **Routing-stats detach** before PAD head (stops sensor-fingerprint leak) | `omfr.py` |

### 0.6 Outstanding work

- Gate gradient through `stage2_feat` → MoE is still a natural path for PAD signal to shape routing. If v14 still shows cross-split EER ~50%, consider: reduce PAD head capacity (`proj_s1/s2` 2.4M params are suspicious), or drop routing-stats features entirely.
- ArcFace gradient explosion is only partially mitigated by LR reduction — consider normalize W inside ArcFace, or drop ArcFace and run SupCon-only identity.

---

## 1. Codebase Structure

```
omfr/
├── configs/
│   └── base.yaml                   # All hyperparams
│
├── models/
│   ├── backbone/
│   │   ├── tiny_vit.py             # TinyViT-5M (forked Microsoft, adapted)
│   │   ├── gabor_stem.py           # 8-channel Learnable Gabor
│   │   ├── frequency_gate.py       # Band energy → gating signal
│   │   └── moe_ffn.py              # FreqGatedMoEFFN (4 experts, top-2)
│   │
│   ├── heads/
│   │   ├── identity_head.py        # Structural Attention + MRL-ArcFace
│   │   └── pad_head.py             # Multi-scale + routing features → 32-D
│   │
│   ├── losses/
│   │   ├── arcface.py              # ArcFace with scale ramp
│   │   ├── supcon.py               # Supervised Contrastive
│   │   └── orthogonal.py           # Barlow Twins cross-correlation
│   │
│   └── omfr.py                     # OMFRModule (LightningModule)
│
├── data/
│   ├── datasets/
│   │   ├── identity_dataset.py     # FVC2004, NIST SD302/300
│   │   ├── pad_dataset.py          # LivDet 2013/2015/2017
│   │   └── joint_dataset.py        # MSU-FPAD v2.0
│   ├── samplers/
│   │   ├── pk_sampler.py           # P=32 identities × K=4 samples
│   │   └── balanced_pad_sampler.py # 50/50 live/spoof
│   ├── transforms.py
│   └── datamodule.py               # Phase-aware DataLoader switching
│
├── callbacks/
│   ├── phase_scheduler.py          # Phase transitions + ramps
│   └── gradient_monitor.py         # Per-component gradient norms
│
├── evaluation/
│   ├── matching_eval.py            # TAR@FAR, CMC
│   ├── pad_eval.py                 # APCER, BPCER, ACER
│   └── integrated_eval.py          # Cascaded accuracy
│
├── train.py
└── export.py                       # ONNX export (2 variants)
```

---

## 2. Architecture Overview

```
Input (B, 1, 224, 224) — grayscale fingerprint
       │
       ▼
┌──────────────────────────────────┐
│  LearnableGaborStem              │
│  8 orientations × 4 frequencies │
│  → select 8 best channels       │
│  (B,1,224,224) → (B,8,224,224)  │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│  TinyViT-5M Backbone (forked + modified)                     │
│                                                              │
│  Stage 1: 2× MBConv blocks ──────────────── 56×56, 64-ch    │
│           (depthwise conv, NO attention)     HIGH-FREQ        │
│                    │                                         │
│           Patch Merging (2× downsample)                      │
│                    ▼                                         │
│  Stage 2: 2× TinyViTBlock ──────────────── 28×28, 128-ch    │
│           ★ Block 1: standard FFN                            │
│           ★ Block 2: FreqGatedMoEFFN ◄── routing_stats_s2   │
│                    │                      MID-FREQ            │
│           Patch Merging                                      │
│                    ▼                                         │
│  Stage 3: 6× TinyViTBlock ──────────────── 14×14, 160-ch    │
│           ★ Block 3: FreqGatedMoEFFN ◄── routing_stats_s3a  │
│           ★ Block 6: FreqGatedMoEFFN ◄── routing_stats_s3b  │
│           (other 4 blocks: standard FFN)   LOW-MID-FREQ      │
│                    │                                         │
│           Patch Merging                                      │
│                    ▼                                         │
│  Stage 4: 2× TinyViTBlock ──────────────── 7×7, 320-ch      │
│           window_size=7 (= feature map → GLOBAL attention)   │
│           (standard FFN, no MoE)           LOW-FREQ/GLOBAL   │
│                                                              │
└───┬──────────────┬──────────────┬────────────────────────────┘
    │              │              │
    │ stage1_feat  │ stage2_feat  │ stage3_feat    stage4_feat
    │ (B,64,56,56) │(B,128,28,28)│(B,160,14,14)  (B,320,7,7)
    │              │              │
    ▼              ▼              ▼                    ▼
┌─────────────────────────┐  ┌────────────────────────────────┐
│      PAD Branch         │  │      Identity Branch            │
│                         │  │                                │
│ Stage 1+2 features      │  │ Stage 3+4 features             │
│ + routing_stats (s2,s3) │  │                                │
│                         │  │ Stage 4 (7×7→49 tokens)        │
│ → Attention Pool        │  │ → Structural Attention + RPE   │
│ → Routing Stats Fusion  │  │   (Δx, Δy relative position)  │
│ → 128-D pad_features    │  │ → Attentive Pooling            │
│    ┌──────┴──────┐      │  │ → 256-D identity_embedding     │
│ 32-D emb     (B,1)logit │  │ → MRL: {64, 128, 256}-D       │
│ (SupCon)     (BCE)      │  │   (ArcFace per dimension)      │
└────────┬────────────────┘  └────────────────┬───────────────┘
         │                                    │
         └──────── L_orth (Barlow Twins) ─────┘
```

**MoE placement rationale:**
- Stage 1 (MBConv): NO MoE — convolution blocks, no FFN to replace
- Stage 2 (1 MoE block): mid-frequency features, shared PAD/identity zone
- Stage 3 (2 MoE blocks): block 3 and block 6 — largest stage, most capacity
- Stage 4: NO MoE — only 49 tokens at 7×7, global attention already cheap

**Total: 3 MoE-FFN layers** (down from 3 in pure ViT plan — same count, but placed more strategically)

---

## 3. Module Definitions — I/O Specs

### 3.1 `gabor_stem.py` — 8-Channel Learnable Gabor

```
Input:  (B, 1, 224, 224)  — grayscale fingerprint
Output: (B, 8, 224, 224)  — 8-channel enhanced (one per orientation)
```

```python
class LearnableGaborStem(nn.Module):
    """
    8 Gabor filters at fixed orientations, learnable frequency + bandwidth.
    
    Channels:
        ch 0: θ=0°      (horizontal ridges)
        ch 1: θ=22.5°
        ch 2: θ=45°     (diagonal ridges)
        ch 3: θ=67.5°
        ch 4: θ=90°     (vertical ridges)
        ch 5: θ=112.5°
        ch 6: θ=135°
        ch 7: θ=157.5°
    
    Learnable per filter: frequency σ, bandwidth γ
    Fixed per filter: orientation θ (evenly spaced), kernel_size=31
    
    Init: σ initialized to typical ridge frequency ~5 cycles/mm
          γ initialized to 0.5 (standard Gabor bandwidth)
    
    Why 8 channels (not 3):
        Training from scratch → no pretrained weight compatibility needed.
        8 orientations capture ridge flow direction explicitly.
        ViT patch_embed Conv2d(8, embed_dim, 16, 16) learns to combine them.
    """
    
    def __init__(self, n_orientations=8, kernel_size=31, init_frequency=0.12):
        # orientations: fixed, evenly spaced [0, π)
        # frequencies: nn.Parameter, learnable per filter
        # bandwidths: nn.Parameter, learnable per filter
    
    def forward(self, x):
        # x: (B, 1, H, W)
        # Apply depthwise grouped conv with Gabor kernels
        # gabor_responses: (B, 8, H, W) — one response per orientation
        return gabor_responses
```

---

### 3.2 `tiny_vit.py` — TinyViT-5M Backbone (Adapted)

```
Input:  (B, 8, 224, 224)  — 8-channel Gabor-enhanced fingerprint
Output: Dict — see below
```

```python
class TinyViTBackbone(nn.Module):
    """
    TinyViT-5M forked from github.com/microsoft/Cream/TinyViT
    
    Modifications from original:
    1. patch_embed: Conv2d(8, 64, ...) thay vì Conv2d(3, 64, ...)
       (8-channel Gabor input)
    2. Return intermediate stage outputs (original chỉ return final)
    3. Stage 2 block 2: FFN → FreqGatedMoEFFN
    4. Stage 3 block 3 + block 6: FFN → FreqGatedMoEFFN
    5. Stage 4: window_size = 7 (= spatial dim → effectively global attention)
    6. ⚠ UPDATED: pretrained=true (ImageNet-22k) — original plan was from-scratch
       but 5M params + 1,030 fingerprint classes makes pretraining essential
    
    TinyViT-5M config:
        embed_dims  = [64, 128, 160, 320]
        depths      = [2, 2, 6, 2]
        num_heads   = [2, 4, 5, 10]
        window_sizes = [7, 7, 14, 7]
        mbconv_expand_ratio = 4.0
    """
```

**Output dict:**
```python
{
    # ── Stage feature maps ──
    'stage1_feat': (B, 64, 56, 56),     # After Stage 1 (MBConv)
    'stage2_feat': (B, 128, 28, 28),    # After Stage 2 (Window Attn + MoE)
    'stage3_feat': (B, 160, 14, 14),    # After Stage 3 (Window Attn + MoE)
    'stage4_feat': (B, 320, 7, 7),      # After Stage 4 (Global Attn)
    
    # ── MoE routing statistics ──
    'routing_stats': {
        's2': {  # Stage 2, block 2
            'expert_weights': (B, 784, 4),   # 784 = 28×28 tokens
            'token_entropy':  (B, 784),
        },
        's3a': {  # Stage 3, block 3
            'expert_weights': (B, 196, 4),   # 196 = 14×14 tokens
            'token_entropy':  (B, 196),
        },
        's3b': {  # Stage 3, block 6
            'expert_weights': (B, 196, 4),
            'token_entropy':  (B, 196),
        },
    },
    'balance_losses': [scalar, scalar, scalar],  # One per MoE layer
}
```

**Utility methods:**
```python
def get_stage_params(self, stages: List[int]) -> Iterator[nn.Parameter]:
    """Get params for specific stages (1-indexed). Excludes MoE params."""

def get_moe_params(self) -> Iterator[nn.Parameter]:
    """Get params for MoE-FFN layers only (experts + gating)."""

def get_embed_params(self) -> Iterator[nn.Parameter]:
    """Get patch_embed, pos_embed params (modified for 8-ch input)."""
    
def freeze_stages(self, stages: List[int]):
    """Freeze specific stages. E.g. freeze_stages([1,2]) for PAD stability."""
```

---

### 3.3 `frequency_gate.py` — FrequencyGate

```
Input:  tokens (B, N, D)  — N varies per stage (784 for 28×28, 196 for 14×14)
Output: gate_input (B, N, 3)  — [low_energy, mid_energy, high_energy] per token
```

```python
class FrequencyGate(nn.Module):
    """
    Analyzes frequency content in spatial arrangement of tokens.
    Produces per-token band energy that drives MoE expert selection.
    
    Adapts to different spatial resolutions (Stage 2: 28×28, Stage 3: 14×14).
    
    Band definitions (radial in frequency domain):
        Low:  0 to H//4 cycles    — ridge flow, global structure
        Mid:  H//4 to H//2 cycles — minutiae, incipient ridges
        High: H//2+ cycles        — pores, micro-texture, spoof artifacts
    
    Implementation:
        1. Reshape tokens to spatial grid: (B, N, D) → (B, D, H, W)
        2. 2D FFT per channel → power spectrum
        3. Radial band integration → 3 energy values per spatial position
        4. LayerNorm for stable gating
    """
    
    def __init__(self):
        # No learnable params — pure signal processing
        # (gating network in MoE-FFN learns to interpret these)
    
    def forward(self, tokens, spatial_h, spatial_w):
        # tokens: (B, H*W, D)
        # Returns: (B, H*W, 3)  — normalized band energies
```

---

### 3.4 `moe_ffn.py` — FreqGatedMoEFFN

```
Input:  tokens (B, N, D)
Output: (output_tokens (B, N, D), routing_stats dict)
```

```python
class FreqGatedMoEFFN(nn.Module):
    """
    Mixture-of-Experts FFN with frequency-based gating.
    Drop-in replacement for standard FFN in TinyViT blocks.
    
    Config:
        num_experts: 4
        top_k: 2
        hidden_dim: D * mlp_ratio (matches original FFN hidden dim)
    
    Each expert = Linear(D, hidden) → GELU → Linear(hidden, D)
    Same architecture as standard FFN, just 4 copies.
    
    Routing (⚠ UPDATED — richer gate + AMP safety):
        gate_input = FrequencyGate(tokens)                # (B, N, 3)
        gate_proj  = Linear(3,16) → GELU → Linear(16, E)  # 2-layer MLP, not single Linear
        gate_logits = gate_proj(gate_input)               # (B, N, E)

        if training: gate_logits += 𝒩(0, 0.3²)            # Shazeer-style noisy top-k

        # fp32 softmax under AMP; renorm ε=1e-4 (> fp16 smallest positive ~6e-8)
        expert_weights = softmax(gate_logits.float() / temperature).to(dtype)
        top_w, top_idx = topk(expert_weights, k=2)
        top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-4)

        # Dispatch via index_add (AMP-safe, no in-place slice add)
        for k in range(top_k):
          for e in range(num_experts):
            idx  = nonzero(top_idx[:,k] == e)
            out  = experts[e](tokens_flat[idx])
            output_flat.index_add_(0, idx, top_w_k[idx]*out)

    Balance loss (⚠ UPDATED — Switch Transformer, not CV²):
        f_e = topk_indicator.mean over tokens     # hard dispatch fraction
        P_e = expert_weights.mean over tokens     # soft probability (differentiable)
        L   = E · Σ (f_e · P_e)                   # per layer
        Uniform optimum with E=4, top_k=2 → L = 2.0/layer
        Summed across 3 MoE layers (omfr.py:187) → 6.0 = optimum, NOT collapse
    
    routing_stats output:
        {
            'expert_weights': (B, N, 4),   # full softmax distribution (before top-k)
            'token_entropy':  (B, N),       # H = -Σ p·log(p) per token
            'balance_loss':   scalar,
        }
    """
    
    def __init__(self, dim, hidden_dim, num_experts=4, top_k=2):
        self.gate = FrequencyGate()
        self.gate_proj = nn.Linear(3, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            ) for _ in range(num_experts)
        ])
        self.temperature = 1.0  # Controlled by PhaseScheduler
```

**Parameter budget per MoE-FFN layer:**
```
Stage 2 (dim=128, hidden=512): 4 × (128×512 + 512×128) = 524K
Stage 3 (dim=160, hidden=640): 4 × (160×640 + 640×160) = 819K × 2 blocks = 1.64M
Gating: 3 × (3→4) per layer ≈ negligible

Total MoE overhead: ~2.16M params
```

---

### 3.5 `pad_head.py` — PAD Branch

```
Input: {
    'stage1_feat':      (B, 64, 56, 56),
    'stage2_feat':      (B, 128, 28, 28),
    'routing_stats_s2': dict,
    'routing_stats_s3a': dict,
    'routing_stats_s3b': dict,
}
Output: {
    'pad_embedding': (B, 32),     # L2-normalized, for SupCon + orthogonality
    'pad_logit':     (B, 1),      # Raw logit, for BCE
    'pad_features':  (B, 128),    # Pre-projection, for SupCon
}
```

```python
class PADHead(nn.Module):
    """
    Multi-scale feature aggregation + MoE routing stats fusion.
    
    ▌ FIX from codebase review:
    ▌ pad_logit computed from pad_features (128-D), NOT from pad_embedding (32-D)
    ▌ → Two PARALLEL branches from pad_features, not serial
    
    Architecture:
    
    ── Feature path ──
    stage1_feat (B,64,56,56)  → AdaptiveAvgPool(7×7) → Flatten → Linear(64*49, 256)
    stage2_feat (B,128,28,28) → AdaptiveAvgPool(7×7) → Flatten → Linear(128*49, 256)
    → concat → (B, 512)
    
    ── Routing path ── [⚠ UPDATED: tensors DETACHED at omfr.py:_run_pad]
    For each MoE layer (s2, s3a, s3b), from backbone_out['routing_stats']:
        expert_weights (B,N,4).detach() → mean over tokens     → (B, 4)
        token_entropy  (B,N).detach()   → [mean,std,max,min]   → (B, 4)
    → concat all 3 layers → (B, 3×8) = (B, 24)
    Why detach: without it, PAD gradient teaches the MoE gate to encode
    liveness in routing patterns, which becomes a sensor fingerprint
    (train BCE→0.03, test EER→50%). Detach keeps routing as an input
    feature only; gate is still trained via backbone task gradient.

    ── Fusion (⚠ UPDATED — 3-layer MLP, was 2-layer) ──
    concat(feature_512, routing_24) → (B, 536)
    → LayerNorm
    → Linear(536, 512) → GELU → Dropout(0.1)
    → Linear(512, 256) → GELU → Dropout(0.1)
    → Linear(256, 128) → pad_features (128-D)
    
    ── Two PARALLEL output branches ──
    pad_features → Linear(128, 32) → L2Norm → pad_embedding  (for SupCon + L_orth)
    pad_features → Linear(128, 1)            → pad_logit      (for BCE)
    
    ▌ Why parallel: BCE gradient doesn't flow through pad_embedding,
    ▌ SupCon gradient doesn't affect pad_logit. Two losses optimized independently.
    """
```

---

### 3.6 `identity_head.py` — Identity Branch

```
Input: {
    'stage3_feat': (B, 160, 14, 14),
    'stage4_feat': (B, 320, 7, 7),
}
Output: {
    'identity_embedding': (B, 256),    # L2-normalized, full dimension
    'mrl_embeddings': {
        64:  (B, 64),                  # L2-normalized prefix
        128: (B, 128),
        256: (B, 256),
    },
    'mrl_logits': {                    # Only during training
        64:  (B, num_classes),
        128: (B, num_classes),
        256: (B, num_classes),
    },
}
```

```python
class IdentityHead(nn.Module):
    """
    Structural Attention on 49 tokens (7×7) + MRL-ArcFace.
    
    ▌ FIX from codebase review:
    ▌ Added RPE (Δx, Δy) to structural attention — spatial awareness
    
    Architecture:
    
    stage3_feat (B,160,14,14) → AdaptiveAvgPool(7,7) → (B,160,7,7) → reshape → (B,49,160)
    stage4_feat (B,320,7,7)   → reshape → (B,49,320)
    
    → concat → (B, 49, 480)
    → Linear(480, 256) → (B, 49, 256)
    
    → Structural Attention (⚠ UPDATED: 3 stacked blocks, was 1):
        self.struct_attn = nn.ModuleList([
            StructuralAttentionBlock(embed_dim=256, num_heads=8, grid_size=7)
            for _ in range(3)
        ])
        for block in self.struct_attn: tokens = block(tokens)
      Each block: Pre-LN MHSA with RPE + FFN, residual.
        Q, K, V = separate Linear projections
        RPE bias = learned table indexed by (Δx, Δy) for 7×7 grid
                   Δx ∈ [-6, +6], Δy ∈ [-6, +6] → 13×13 = 169 entries
        attn = softmax(QK^T / √d_head + RPE_bias[Δx, Δy]) @ V
        + residual + FFN(4× expand)
      Rationale: match face-recognition decoder depth (ArcFace/AdaFace
      use ≥2 FC+conv decoders); 1 block was insufficient capacity.
    
    → Attentive Pooling (M=4 learned query seeds):
        queries = nn.Parameter(4, 256)  — 4 learnable "what to look for" vectors
        pool_attn = softmax(queries @ tokens.T) @ tokens  → (B, 4, 256)
        → flatten → Linear(1024, 256) → (B, 256)
    
    → L2Norm → identity_embedding (B, 256)
    
    → MRL slicing:
        z_64  = L2Norm(identity_embedding[:, :64])
        z_128 = L2Norm(identity_embedding[:, :128])
        z_256 = identity_embedding  (already normalized)
        
    → ArcFace classifiers (training only):
        logits_64  = ArcFace_64(z_64, labels)
        logits_128 = ArcFace_128(z_128, labels)
        logits_256 = ArcFace_256(z_256, labels)
    
    forward_embedding_only():
        Runs full pipeline EXCEPT ArcFace classifiers.
        Used for: Phase 2 PAD batch (L_orth only), inference, ONNX export.
        Returns: identity_embedding (B, 256)
    """
```

**RPE implementation detail:**
```python
# 7×7 grid → relative positions Δx ∈ [-6,+6], Δy ∈ [-6,+6]
self.rpe_table = nn.Parameter(torch.zeros(num_heads, 13, 13))
nn.init.trunc_normal_(self.rpe_table, std=0.02)

# In forward:
# coords: (49, 2) — (x, y) for each of 49 positions
# delta = coords[:, None, :] - coords[None, :, :]  → (49, 49, 2)
# rpe_bias = self.rpe_table[:, delta[...,0]+6, delta[...,1]+6]  → (heads, 49, 49)
# attn = softmax(QK^T/√d + rpe_bias) @ V
```

---

## 4. Loss Definitions

### 4.1 MRL-ArcFace (Identity)

```python
class ArcFaceLoss(nn.Module):
    """
    Input:  embedding (B, m), labels (B,)    m ∈ {64, 128, 256}
    Output: scalar loss
    
    Formula:
        W: (num_classes, m) — L2-normalized per row
        cos_θ = W^T · x
        θ = arccos(clamp(cos_θ, -1+ε, 1-ε))
        target_logit = s · cos(θ_yi + margin)
        other_logits = s · cos(θ_j)  for j ≠ yi
        L = CrossEntropy(logits, labels)
    
    EACH dimension m has its OWN W_m (not shared).
    
    set_scale(s): called by PhaseScheduler to ramp s from 32→64
    """
```

### 4.2 Supervised Contrastive (PAD)

```python
class SupConLoss(nn.Module):
    """
    Input:  features (B, 128) — pad_features, NOT pad_embedding
            labels (B,)       — 0=spoof, 1=live
    Output: scalar loss
    
    Temperature: τ = 0.07
    
    For anchor i:
        Positives P(i) = {j : y_j = y_i, j ≠ i}
        L_i = -1/|P(i)| Σ_{p∈P(i)} log[exp(z_i·z_p/τ) / Σ_{k≠i} exp(z_i·z_k/τ)]
    
    Why SupCon over BCE alone:
        SupCon creates TIGHT clusters → better generalization to unseen spoof materials
        BCE only pushes toward 0/1 boundary
    """
```

### 4.3 Orthogonality Loss

```python
class OrthogonalityLoss(nn.Module):
    """
    Input:  pad_emb (B, 32), id_emb (B, 256) — both L2-normalized
    Output: scalar loss
    
    Barlow Twins cross-correlation approach:
        Z_pad = (pad_emb - mean) / std    per feature, across batch
        Z_id  = (id_emb - mean) / std     per feature, across batch
        C = Z_pad.T @ Z_id / B            → (32, 256) cross-correlation matrix
        L = Σ_ij C_ij²                    penalize ALL cross-correlations
    
    Why Barlow Twins over cosine:
        Cosine measures direction similarity (1 number)
        Cross-correlation penalizes per-dimension correlations (32×256 constraints)
        → Stronger decorrelation guarantee
    """
```

---

## 5. Composite Loss — Per Phase

```
L_total = L_Identity + α·L_PAD + β·L_orth + γ·L_balance

where (⚠ UPDATED: hybrid identity, Switch balance, γ=0.05):
    L_Identity = 0.7·SupCon(z_256, id_labels)
               + 0.3·(1/3)·Σ_{m∈{64,128,256}} ArcFace_m(z_m, id_labels)
    L_PAD      = SupCon(pad_features, liveness_labels) + BCE(pad_logit, liveness_labels)
    L_orth     = BarlowCrossCorr(pad_embedding, identity_embedding)
    L_balance  = Σ_{l∈{s2,s3a,s3b}} E · Σ_e (f_e^l · P_e^l)   # Switch-Transformer

    α: PAD loss weight       (0 → 1.0, ramped in Phase 2)
    β: Orthogonality weight  (0 → 0.1, ramped in Phase 2)
    γ: Balance loss weight   (0.05, fixed all phases — up from 0.01)
    SupCon weight 0.7 / ArcFace weight 0.3 — SupCon dominates for open-set
    generalisation (FVC test IDs are disjoint from train); ArcFace adds
    closed-set discriminative structure without saturating the loss.
```

---

## 6. Gradient Flow — Per Phase

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        GRADIENT FLOW MATRIX                             ║
╠════════════════╦════════════╦════════════════╦═══════════╦══════════════╣
║  Component     ║  Phase 1   ║ Phase 2        ║ Phase 2   ║  Phase 3     ║
║                ║            ║ Identity batch ║ PAD batch ║              ║
╠════════════════╬════════════╬════════════════╬═══════════╬══════════════╣
║ Gabor Stem     ║ ✅ ArcFace ║ ✅ ArcFace+Ort ║ ✅ PAD+Ort║ ✅ ALL       ║
║ Stage 1 MBConv ║ ✅ ArcFace ║ ✅ ArcFace+Ort ║ ✅ PAD+Ort║ ✅ ALL       ║
║ Stage 2 (+MoE) ║ ✅ ArcFace ║ ✅ ArcFace+Ort ║ ✅ PAD+Ort║ ✅ ALL       ║
║ Stage 3 (+MoE) ║ ✅ ArcFace ║ ✅ ArcFace+Ort ║ ⚡ Ort    ║ ✅ ALL       ║
║ Stage 4        ║ ✅ ArcFace ║ ✅ ArcFace+Ort ║ ⚡ Ort    ║ ✅ ALL       ║
║ PAD Head       ║ ❌ frozen  ║ ⚡ Ort only     ║ ✅ PAD+Ort║ ✅ ALL       ║
║ Identity Head  ║ ✅ ArcFace ║ ✅ ArcFace+Ort ║ ⚡ Ort    ║ ✅ ALL       ║
║ ArcFace cls    ║ ✅ ArcFace ║ ✅ ArcFace     ║ ❌ no grad║ ✅ ArcFace   ║
╠════════════════╬════════════╬════════════════╬═══════════╬══════════════╣
║ MoE gating     ║ ✅ ArcFace ║ ✅ ArcFace+Ort ║ ✅ PAD+Ort║ ✅ ALL       ║
║ MoE experts    ║ ✅ ArcFace ║ ✅ ArcFace+Ort ║ mixed*    ║ ✅ ALL       ║
╠════════════════╬════════════╬════════════════╬═══════════╬══════════════╣
║ L_ArcFace_MRL  ║ ✅         ║ ✅             ║ ❌        ║ ✅ (masked)  ║
║ L_SupCon       ║ ❌         ║ ❌             ║ ✅        ║ ✅           ║
║ L_BCE          ║ ❌         ║ ❌             ║ ✅        ║ ✅           ║
║ L_orth         ║ ❌         ║ ✅             ║ ✅        ║ ✅           ║
║ L_balance      ║ ✅         ║ ✅             ║ ✅        ║ ✅           ║
╚════════════════╩════════════╩════════════════╩═══════════╩══════════════╝

⚡ = receives gradient but weak (only from L_orth, β=0.1)
* MoE experts in Stage 2: full PAD gradient. Stage 3: Ort only.
```

**Phase 2 PAD batch — gradient isolation implementation:**

```python
def _phase2_pad_step(self, batch):
    images, liveness_labels = batch['images'], batch['liveness_labels']
    
    enhanced = self.gabor(images)
    backbone_out = self.backbone(enhanced)
    
    # ── PAD head: full gradient ──
    pad_out = self.pad_head(backbone_out)
    
    # ── Identity head: embedding only, NO ArcFace ──
    # forward_embedding_only() returns identity_embedding
    # Gradients from L_orth will flow back through backbone Stage 3-4
    # BUT no ArcFace gradient (no classification computed)
    id_emb = self.identity_head.forward_embedding_only(backbone_out)
    
    # ── Losses ──
    l_supcon = self.supcon_loss(pad_out['pad_features'], liveness_labels)
    l_bce = self.bce_loss(pad_out['pad_logit'].squeeze(), liveness_labels.float())
    l_orth = self.orth_loss(pad_out['pad_embedding'], id_emb)
    l_balance = sum(backbone_out['balance_losses']) / 3
    
    loss = self.alpha * (l_supcon + l_bce) + self.beta * l_orth + self.gamma * l_balance
    
    self.log('train/pad_supcon', l_supcon)
    self.log('train/pad_bce', l_bce)
    self.log('train/orth', l_orth)
    return loss
```

---

## 7. Training Protocol — 3 Phases

### Phase 1: Identity Foundation (Epochs 0-19)   ⚠ UPDATED — ArcFace warm-up delay

```
Data:     Identity only (NIST SD300 with group_by_subject → ~1,030 classes × ~20 samples)
Sampler:  PKSampler(P=32, K=4) → batch=128
Loss:     L = 0.7·SupCon + 0.3·MRL-ArcFace + γ·L_balance           γ=0.05
Active:   Gabor, TinyViT (all stages incl. MoE), Identity Head, ArcFace classifiers
Frozen:   PAD Head

ArcFace warm-up schedule (prevents grad explosion with many classes):
    epochs 0..4  (phase1_warmup_delay=5): hold scale=1.0, margin=0.0
                  → only SupCon drives identity during LR warm-up
    epochs 5..9  (phase1_warmup_epochs=5): linear ramp
                  scale:  1.0 → 32.0
                  margin: 0.0 → 0.5
    epochs 10..19: scale=32, margin=0.5 (full Phase 1 targets)

MoE temperature=2.0 (soft routing, exploration). Load balance always active.
```

### Phase 2: PAD Integration (Epochs 20-39)

```
Data:     Alternating Identity + PAD batches
          CombinedLoader({"identity": ..., "pad": ...}, mode="min_size")
          training_step: batch_idx % 2 → choose which batch to process
          
Loss:     Identity batch: L = L_Identity + β·L_orth + γ·L_balance
          PAD batch:      L = α·(L_SupCon + L_BCE) + β·L_orth + γ·L_balance
          
Warmup:   Over first 5 epochs of Phase 2 (epochs 20-24):
          α: 0.0 → 1.0 (linear)
          β: 0.0 → 0.1 (linear)
          ArcFace scale: 32 → 64 (linear)
          MoE temperature: 2.0 → 1.0 (linear, sharper routing)
          
Events:   At epoch 20:
          - PAD Head weights initialized (Xavier uniform)
          - PAD Head added to optimizer with 2× LR
          - DataModule switches to CombinedLoader
```

### Phase 3: Joint Refinement (Epochs 40-59)

```
Data:     All three datasets (identity + PAD + joint)
          CombinedLoader({"identity": ..., "pad": ..., "joint": ...})
          
Loss:     Full composite: L_Identity* + α·L_PAD + β·L_orth + γ·L_balance
          * ArcFace masked: chỉ compute trên live samples (spoof excluded)
          
Config:   α=1.0, β=0.1, γ=0.01 (all fixed)
          ArcFace scale=64
          MoE temperature=0.5 (near-hard routing)
          
Features: Hard negative mining (identity: highest ArcFace loss)
          Hard positive mining (PAD: spoofs closest to live cluster)
          Gradient surgery (PCGrad) if conflict detected
```

---

## 8. DataModule — Phase-aware

```python
class OMFRDataModule(L.LightningDataModule):
    """
    current_phase set by PhaseSchedulerCallback at each transition.

    ⚠ UPDATED:
    - IdentityDataset(split='train', group_by_subject=True) merges NIST SD300
      frgp_0X folders per subject (10,300 → ~1,030 classes).
    - PADDataset(split='train', transform=make_pad_train_transform(...))
      applies strong augmentation (HFlip/VFlip/Rot±15°/RRC(0.85-1.0)/
      ColorJitter b=c=0.3/GaussianBlur p=0.3/RandomErasing p=0.25).
      val_pad_ds has NO transform (clean eval).

    Phase 1: Single DataLoader (identity only, PKSampler)
    Phase 2: CombinedLoader with mode="min_size"
             Delivers {"identity": batch, "pad": batch} simultaneously
             training_step selects which to process via batch_idx % 2
    Phase 3: CombinedLoader with mode="max_size_cycle"
             Cycles shorter loaders to match longest
             Delivers {"identity": batch, "pad": batch, "joint": batch}
    """
    
    def train_dataloader(self):
        if self.current_phase == 1:
            return DataLoader(
                self.identity_ds,
                batch_sampler=PKSampler(self.identity_ds, p=32, k=4),
                num_workers=8, pin_memory=True,
            )
        elif self.current_phase == 2:
            from lightning.pytorch.utilities import CombinedLoader
            return CombinedLoader({
                "identity": DataLoader(
                    self.identity_ds,
                    batch_sampler=PKSampler(self.identity_ds, p=32, k=4),
                    num_workers=4, pin_memory=True,
                ),
                "pad": DataLoader(
                    self.pad_ds,
                    batch_sampler=BalancedPADSampler(self.pad_ds, batch_size=128),
                    num_workers=4, pin_memory=True,
                ),
            }, mode="min_size")
        elif self.current_phase == 3:
            return CombinedLoader({
                "identity": DataLoader(self.identity_ds, ...),
                "pad": DataLoader(self.pad_ds, ...),
                "joint": DataLoader(self.joint_ds, ...),
            }, mode="max_size_cycle")
    
    def val_dataloader(self):
        # Always return both identity + PAD val loaders
        return [
            DataLoader(self.identity_val_ds, batch_size=256, ...),
            DataLoader(self.pad_val_ds, batch_size=256, ...),
        ]
```

---

## 9. Optimizer & Differential LR

```python
def configure_optimizers(self):
    param_groups = [
        {'params': self.gabor.parameters(),
         'lr': self.hparams.lr * 0.1,
         'name': 'gabor'},
        
        {'params': self.backbone.get_embed_params(),
         'lr': self.hparams.lr,
         'name': 'backbone_embed'},  # patch_embed (modified for 8-ch), pos_embed
         
        {'params': self.backbone.get_stage_params([1, 2]),  # excludes MoE
         'lr': self.hparams.lr,
         'name': 'backbone_early'},
         
        {'params': self.backbone.get_stage_params([3, 4]),
         'lr': self.hparams.lr,
         'name': 'backbone_late'},
         
        {'params': self.backbone.get_moe_params(),
         'lr': self.hparams.lr * 2.0,
         'name': 'moe_experts'},  # Higher LR — newly initialized experts
         
        {'params': self.identity_head.parameters(),
         'lr': self.hparams.lr,
         'name': 'identity_head'},
         
        {'params': self.pad_head.parameters(),
         'lr': self.hparams.lr * (2.0 if self.current_phase == 2 else 1.0),
         'name': 'pad_head'},
         
        {'params': [p for af in self.arcface_losses.values() for p in af.parameters()],
         'lr': self.hparams.lr * 10.0,
         'name': 'arcface'},
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups, lr=self.hparams.lr,  # base: 1e-4
        weight_decay=0.05, betas=(0.9, 0.999),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=1e-6,
    )
    
    return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
```

---

## 10. PhaseSchedulerCallback

```python
class PhaseSchedulerCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        # ── Phase transitions ──
        if epoch == 20:
            pl_module.current_phase = 2
            pl_module.pad_head.apply(self._xavier_init)  # Re-init PAD head
            trainer.datamodule.current_phase = 2          # Switch DataLoader
            # Note: configure_optimizers NOT re-called — PAD head params
            # already in optimizer from init, just with 0 gradient before
            
        elif epoch == 40:
            pl_module.current_phase = 3
            trainer.datamodule.current_phase = 3
        
        # ── Phase 2 warmup ramps (epochs 20-24) ──
        if pl_module.current_phase == 2:
            progress = min((epoch - 20) / 5, 1.0)
            pl_module.alpha = progress * 1.0
            pl_module.beta = progress * 0.1
            
            new_scale = 32.0 + progress * 32.0
            for af in pl_module.arcface_losses.values():
                af.set_scale(new_scale)
            
            # MoE temperature: 2.0 → 1.0
            new_temp = 2.0 - progress * 1.0
            for moe_layer in pl_module.backbone.get_moe_layers():
                moe_layer.temperature = new_temp
        
        # ── Phase 3: further decrease MoE temperature ──
        if pl_module.current_phase == 3:
            progress = min((epoch - 40) / 10, 1.0)
            new_temp = 1.0 - progress * 0.5  # 1.0 → 0.5
            for moe_layer in pl_module.backbone.get_moe_layers():
                moe_layer.temperature = new_temp
```

---

## 11. Evaluation

```python
def validation_step(self, batch, batch_idx, dataloader_idx=0):
    enhanced = self.gabor(batch['images'])
    backbone_out = self.backbone(enhanced)
    
    if dataloader_idx == 0:  # Identity val
        id_out = self.identity_head.forward_embedding_only(backbone_out)
        self.val_id_embeddings.append(id_out.cpu())
        self.val_id_labels.append(batch['identity_labels'].cpu())
        
    elif dataloader_idx == 1:  # PAD val
        pad_out = self.pad_head(backbone_out)
        scores = torch.sigmoid(pad_out['pad_logit']).squeeze()
        self.val_pad_scores.append(scores.cpu())
        self.val_pad_labels.append(batch['liveness_labels'].cpu())

def on_validation_epoch_end(self):
    # ── PAD metrics ──
    scores = torch.cat(self.val_pad_scores)
    labels = torch.cat(self.val_pad_labels)
    acer = compute_acer(scores, labels)
    self.log('val/ACER', acer, prog_bar=True)
    
    # ── Identity metrics per MRL dimension ──
    embs = torch.cat(self.val_id_embeddings)
    id_labels = torch.cat(self.val_id_labels)
    for dim in [64, 128, 256]:
        emb_m = F.normalize(embs[:, :dim], dim=-1)
        tar = compute_tar_at_far(emb_m, id_labels, far=0.0001)
        self.log(f'val/TAR@FAR0.01%_{dim}D', tar)
    
    # ── Cascaded metric (primary) ──
    # Step 1: PAD rejects spoofs. Step 2: Match remaining (live-predicted)
    cascaded = compute_cascaded_accuracy(scores, labels, embs, id_labels)
    self.log('val/cascaded_IM', cascaded, prog_bar=True)
```

---

## 12. Config

⚠ UPDATED — actual values in `configs/base.yaml` as of 2026-04-19:

```yaml
# configs/base.yaml  (excerpt, keys that matter)
backbone:
  name: tiny_vit_5m_224
  pretrained: true            # ⚠ ImageNet-22k (plan said false)
  in_chans: 8
  num_experts: 4
  top_k: 2

data:
  identity_data_root: /home/linhdang/workspace2/NIST-300-ds/combined_nist
  pad_data_root:      /home/linhdang/workspace2/PAD-joint
  identity_datasets:  [nist_sd300]
  pad_datasets:       [LivDet2013, LivDet2015]
  image_size: 224
  group_by_subject: true       # ⚠ merge NIST frgp_0X → subject
  pk_p: 32
  pk_k: 4
  pad_batch_size: 128
  num_workers: 8

losses:
  arcface: {margin: 0.5, scale_phase1: 32.0, scale_final: 64.0}
  supcon:  {temperature: 0.07}
  alpha: 0.0                   # ramped Phase 2
  beta:  0.0                   # ramped Phase 2
  gamma: 0.05                  # ⚠ up from 0.01 (balance loss)
  identity_supcon_weight: 0.7  # ⚠ hybrid identity
  identity_arcface_weight: 0.3

optimizer:
  name: AdamW
  lr: 1.0e-4
  weight_decay: 0.05
  lr_multipliers:
    gabor:              0.1
    backbone_embed:     1.0
    backbone_early:     1.0
    backbone_late:      1.0
    moe_experts:        2.0
    identity_head:      1.0
    pad_head_phase1:    1.0
    pad_head_phase2:    2.0
    arcface:            1.0     # ⚠ reduced from 10.0 (grad explosion)

phases:
  phase1_epochs: 20
  phase2_epochs: 20
  phase3_epochs: 20
  total_epochs:  60
  phase1_warmup_delay:   5      # ⚠ NEW — hold s=1,m=0 during LR warm-up
  phase1_warmup_epochs:  5      # then ramp s:1→32, m:0→0.5
  arcface_scale_init:    1.0
  arcface_margin_init:   0.0
  arcface_margin_target: 0.5
  warmup_epochs:         5      # Phase 2 ramp for α/β/arcface scale
  alpha_target:          1.0
  beta_target:           0.1
  arcface_scale_start:   32.0
  arcface_scale_end:     64.0
  moe_temp_phase1:       2.0
  moe_temp_phase2_end:   1.0
  moe_temp_phase3_end:   0.5

trainer:
  accelerator: gpu
  precision:   16-mixed
  gradient_clip_val:       1.0  # ⚠ down from 10.0
  accumulate_grad_batches: 2    # effective batch ≈ 256
```

---

## 13. Parameter Budget

```
TinyViT-5M backbone (original):      5.4M
 - Modified patch_embed (8-ch):       +0.003M (negligible)
 - 3× MoE-FFN overhead:              +2.16M
 = Backbone total:                    ~7.56M

Gabor Stem:                           ~0.02M
Identity Head:
  Projection (480→256):               0.12M
  Structural Attention (8 heads):     0.26M
  RPE table (8×13×13):               0.001M
  Attentive Pooling (4 seeds):        0.26M
  = Identity Head total:              ~0.65M

PAD Head:
  Feature path (stage1+2 pool+proj):  0.28M
  Routing path (linear layers):       0.01M
  Fusion MLP:                         0.10M
  Output branches:                    0.005M
  = PAD Head total:                   ~0.40M

ArcFace classifiers (train only):
  3 × (256×N) ≈ 2M (N≈2000)          ~2.0M
───────────────────────────────────────────
Total trainable:                      ~10.6M
Inference only (no ArcFace):          ~8.6M
```

---

## 14. Inference — Cascaded Pipeline

```
Input fingerprint
       │
       ▼
  Gabor Stem + TinyViT Stage 1-2 only
       │
       ▼
  PAD Head → 32-D → live/spoof decision
       │
       ├── SPOOF → REJECT                     Cost: ~1.5ms
       │
       └── LIVE ↓
                │
       TinyViT Stage 3-4 (continue forward)
                │
                ▼
       Identity Head → 64-D prefix
                │
                ├── 1:N FAISS search (fast)    Cost: ~3ms total
                │
                └── Top-K candidates ↓
                         │
                    Full 256-D re-ranking      Cost: ~4ms total
                         │
                         ▼
                    MATCH RESULT
```

**Early exit optimization:** TinyViT hierarchical structure allows Stage 1-2 to run independently. If PAD rejects, Stage 3-4 never executes → significant latency saving for spoof-heavy traffic.

---

## 15. ONNX Export

```python
# export.py — Two graph variants

# Variant 1: PAD-only (early rejection)
# Input: (1, 1, 224, 224) → Output: pad_logit (1, 1)
# Includes: Gabor + TinyViT Stage 1-2 + PAD Head

# Variant 2: Full pipeline
# Input: (1, 1, 224, 224) → Output: identity_embedding (1, 256) + pad_logit (1, 1)
# Includes: Gabor + TinyViT all stages + both heads (no ArcFace)

# MoE at inference: soft routing (all experts weighted, no top-k selection)
# → Static computation graph, TensorRT compatible
# FLOPs increase ~2× at 3 MoE layers, but total increase <15%
```
