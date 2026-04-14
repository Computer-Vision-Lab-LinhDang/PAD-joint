# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OMFR (Orthogonal Matryoshka Fingerprint Representation)** — A fingerprint recognition system that jointly learns identity matching and Presentation Attack Detection (PAD/liveness) in a single model. The core idea is orthogonal embedding spaces: identity features and PAD features are forced to be decorrelated via an orthogonality loss, while sharing a common TinyViT backbone.

## Architecture

The model pipeline flows: **Grayscale image (1,224,224) -> LearnableGaborStem (3,224,224) -> ViT-Tiny + MoE backbone -> two heads**.

- **LearnableGaborStem** (`gabor_stem.py`): Converts 1-channel grayscale to 3-channel (original + 2 Gabor responses) for ImageNet-pretrained compatibility. Gabor frequency/orientation are learnable parameters.
- **ViT-Tiny Backbone** (`vit_tiny.py`): Flat 12-layer ViT (embed_dim=192, 196 tokens at 14×14 throughout). FFN replaced with Frequency-Gated MoE-FFN at layers 3, 7, 10. Returns intermediate token tensors and routing statistics. ~8.35M params.
- **FrequencyGate** (`frequency_gate.py`): Reshapes tokens to 14×14 spatial grid, applies 2D-FFT, computes per-token low/mid/high band energy as gating signal (B,196,3).
- **FreqGatedMoEFFN** (`moe_ffn.py`): 4 experts, top-2 routing. Gate input = band energy from FrequencyGate. Returns `routing_stats` (expert_weights, token_entropy) and `balance_loss`.
- **PAD Head** (`pad_head.py`): Consumes layer-3 and layer-7 tokens **plus routing statistics**. Key novelty: routing patterns (expert load distribution, token entropy) are concatenated as PAD features alongside token embeddings — live images show diverse routing, spoofs show uniform routing.
- **Identity Head** (`identity_head.py`): Consumes layer-12 tokens + CLS token with structural attention + RPE(Δx,Δy,Δθ,Δf). Outputs 256-D identity embedding with MRL nesting at {64, 128, 256}, each with its own ArcFace classifier.

## Three-Phase Training Protocol

This is the most critical architectural concept. Phase transitions are managed by `PhaseSchedulerCallback`.

| Phase | Epochs | What trains | Key losses |
|-------|--------|-------------|------------|
| **Phase 1: Identity Foundation** | 1-20 | Gabor + full backbone + Identity Head | MRL-ArcFace (scale=32, gentle) |
| **Phase 2: PAD Integration** | 20-40 | Alternating ID/PAD batches, PAD Head introduced | ArcFace + SupCon + BCE + Orthogonality (alpha/beta ramp 0->1.0/0.1 over 5 epochs) |
| **Phase 3: Joint Refinement** | 40-60 | Everything jointly, 3 dataset types | All losses, spoof samples masked from ArcFace |

**Phase 2 gradient control is nuanced**: On PAD batches, stages 3-4 and Identity Head only receive weak orthogonality gradients. On identity batches, PAD Head only receives orthogonality gradients.

## Losses

Total: `L = L_Identity + α·L_PAD + β·L_orth + γ·L_balance`

- **MRL-ArcFace** (`L_Identity`): Separate ArcFace classifier per MRL dimension {64, 128, 256}. Weights NOT shared. Scale ramps 32->64 in Phase 2.
- **SupCon + BCE** (`L_PAD`): SupCon on PAD 128-D features (temperature=0.07) + BCE on binary logit. α ramps 0->1.0 over Phase 2 warmup.
- **Orthogonality** (`L_orth`, β=0.1): Barlow Twins-style cross-correlation matrix between PAD (32-D) and identity (256-D) embeddings.
- **Load Balancing** (`L_balance`, γ=0.01): CV² of expert load from all 3 MoE layers. Always active, ensures each expert handles ~25% of tokens.

## Datasets

- **Identity**: FVC2004 + NIST SD302 (identity labels only)
- **PAD**: LivDet 2015 + 2017 (liveness labels only)
- **Joint**: MSU-FPAD v2.0 (both identity + liveness labels)

Samplers: PKSampler (P=32 identities x K=4 samples) for identity; BalancedPADSampler (50/50 live/spoof) for PAD.

## Tech Stack

- **PyTorch Lightning** — LightningModule (`OMFRModule`), LightningDataModule, Callbacks
- **TinyViT** — from Microsoft repo, adapted for multi-stage output
- **AdamW** optimizer, CosineAnnealingLR, differential learning rates per component group
- **Mixed precision** (16-mixed), gradient clipping (1.0), gradient accumulation (x2 -> effective batch 256)
- **Single GPU** target

## Build & Run (Planned)

```bash
# Training
python train.py                              # Uses configs/base.yaml
python train.py --config configs/phase1_identity.yaml

# Export
python export.py                             # ONNX/TensorRT export
```

## Evaluation Metrics

- **PAD**: APCER, BPCER, ACER
- **Matching**: TAR@FAR=0.01% at each MRL dimension {64, 128, 256}
- **Integrated**: Cascaded accuracy (PAD reject spoofs first, then match remaining)
- Primary checkpoint metric: `val/cascaded_IM` (maximize)

## Key Design Decisions

- **ViT-Tiny over TinyViT**: Flat architecture gives global attention at every layer, uniform FFN structure enables clean MoE insertion. TinyViT's hierarchical stages make MoE placement impossible at early stages (MBConv, no FFN) and hard-code frequency separation.
- **Routing patterns as PAD signal**: The novel contribution — MoE routing statistics (expert load distribution, token entropy) are fed directly to the PAD head alongside token features. Live fingerprints exhibit diverse routing (rich micro-texture), spoofs show uniform routing (synthetic/uniform texture).
- **Orthogonal subspaces**: PAD and identity embeddings are forced decorrelated so liveness detection doesn't leak identity information and vice versa.
- **MRL (Matryoshka)**: Identity embedding supports variable-dimension matching — 64-D for speed, 256-D for accuracy, same model.
- **Layer tap strategy**: PAD reads layers 3+7 (early/mid features capture pores and micro-texture before global pooling washes them out). Identity reads layer 12 + CLS (deepest global semantics).
- **Spoof masking in Phase 3**: Spoof samples excluded from ArcFace loss — spoofs should not contribute to identity learning.

## Optimizer Configuration

Differential LR groups (relative to base LR=1e-4):
- Gabor stem: 0.1x (stable learned filters)
- ViT layers 1-6: 1x
- ViT layers 7-12: 1x
- MoE experts + gates: 2x (new parameters, need faster convergence)
- Identity/PAD heads: 1x (PAD gets 2x in Phase 2 to catch up)
- ArcFace classifiers: 10x (large, sparse gradients)
