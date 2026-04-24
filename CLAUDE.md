# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OMFR (Orthogonal Matryoshka Fingerprint Representation)** — joint fingerprint identity matching and Presentation Attack Detection (PAD/liveness) in a single model. Identity and PAD embeddings are forced to be decorrelated (orthogonality loss) while sharing a FastViT-SA12 backbone with Frequency-Gated Mixture-of-Experts FFNs.

## Architecture

Pipeline: **grayscale (B,1,224,224) → LearnableGaborStem (B,8,224,224) → FastViT-SA12 + MoE → (IdentityHead, PADHead)**.

- **LearnableGaborStem** (`backbone/gabor_stem.py`): 1→8-channel bank, 8 fixed orientations × learnable frequency/bandwidth. Stem input is 8-ch so the FastViT patch conv is re-seeded from ImageNet weights (first 3 channels copied, 4–7 = mean + small Gaussian noise).
- **PADStem** (`backbone/pad_stem.py`): a separate shallow convolutional stem dedicated to the PAD branch (independent of the identity Gabor path).
- **FastViT-SA12 Backbone** (`backbone/fastvit.py`, ~10.5M params, timm pretrained ImageNet-1k distilled):
  - `stage0`: 2× RepMixer → (B, 64, 56, 56)
  - `stage1`: 2× RepMixer → (B, 128, 28, 28)  — MoE at block 1 (`s2`)
  - `stage2`: 6× RepMixer → (B, 256, 14, 14)  — MoE at blocks 2 (`s3a`) and 5 (`s3b`)
  - `stage3`: 2× Self-Attention → (B, 512, 7, 7)  — genuine global attention for identity matching
- **FrequencyGate** (`backbone/frequency_gate.py`): ridge-aware spatial bands. 2D-FFT per resolution (cached masks for 28×28 / 14×14), three radial bands (sub-ridge / ridge / super-ridge) scaled to a measured ridge frequency (defaults: 11.5 @ 28, 5.8 @ 14). Forced to fp32 under AMP because cuFFT fp16 only supports power-of-2 sizes.
- **FreqGatedMoEFFN** (`backbone/moe_ffn.py`): 4 experts, top-2 routing, 2-layer gate MLP (3→16→E) on band energies. Noisy top-k (σ=0.3) in train only. Switch-Transformer balance loss `E·Σ(f·P)`. fp32 softmax + `index_add` dispatch for AMP safety; expert outputs cast to the buffer dtype before accumulation.
- **`ConvMoEMlpWrapper`** (`backbone/fastvit.py`): replaces each FastViT `ConvMlp` where MoE is inserted. Keeps the pretrained depthwise-7×7 spatial mixer, swaps only the 1×1 fc1/GELU/fc2 expansion for `FreqGatedMoEFFN` on flattened `(B, N, C)` tokens.
- **PADHead** (`heads/pad_head.py`): consumes **detached** `pad_stem_feat` + `stage1_feat` (64ch, 56²) + `stage2_feat` (128ch, 28²) **plus routing stats** from `s2/s3a/s3b`. Routing patterns (expert load, token entropy) are concatenated as PAD features — live shows diverse routing, spoof shows uniform routing. Emits `pad_logit`, 32-D `pad_embedding` (L2-normed), 128-D `pad_features`.
- **IdentityHead** (`heads/identity_head.py`): consumes `stage3_feat` (256ch, 14²) + `stage4_feat` (512ch, 7²). Upsamples stage4 to 14×14 (preserves minutiae), fuses to 196 tokens, runs Structural Attention + RPE(Δx,Δy), Attentive Pooling (4 learned queries), 256-D output with MRL nesting {64, 128, 256}, each with its own ArcFace classifier.

## Three-Phase Training Protocol

Managed by `callbacks/phase_scheduler.py`.

| Phase | Epochs | What trains | Key losses |
|-------|--------|-------------|------------|
| **Phase 1: Identity Foundation** | 0–19 | Gabor + backbone + Identity Head | hybrid (0.9·SupCon + 0.1·MRL-ArcFace) + γ·L_balance. ArcFace scale/margin held at 1.0/0.0 for `phase1_warmup_delay` epochs, then ramped 1→32 / 0→0.5 over `phase1_warmup_epochs`. |
| **Phase 2: PAD Integration** | 20–39 | Concatenated id+pad step (TASK_04 joint step, not `batch_idx%2` alternation); PAD Head + PADStem introduced | hybrid identity + α·(L_focal_BCE + L_mixup) + β·L_orth + γ·L_balance. Cosine soft-start α→1.0, β→0.02, ArcFace scale 32→48 over 5 epochs. |
| **Phase 3: Joint Refinement** | 40–59 | Everything jointly, 3 dataset types | All losses, spoof samples masked from ArcFace. |

**Phase 2 gradient isolation**: `_run_pad` detaches **all** backbone outputs (`stage1/2/3/4_feat` and routing stats) before they reach the PAD head, so PAD loss cannot back-propagate through the 10.5M-param backbone. The PAD branch is shaped only by PADStem + PADHead; the backbone continues to be shaped by identity loss.

## Losses

Total: `L = L_Identity + α·L_PAD + β·L_orth + γ·L_balance` (+ optional DANN `alpha_adv·L_sensor_adv`, currently disabled via `alpha_adv_target: 0.0`).

- **Hybrid Identity** (`L_Identity`): `0.9·SupCon(z_256) + 0.1·(1/3)·Σ_{m∈{64,128,256}} ArcFace_m(z_m)`. SupCon dominates for open-set generalization to unseen FVC identities; ArcFace adds discriminative structure. ArcFace scale ramps 1→32→48 across Phase 1 warmup + Phase 2.
- **PAD** (`L_PAD`): Focal BCE on `pad_logit` + MixUp consistency on `pad_features` (TASK_01 replacement for degenerate τ=0.07 SupCon). α ramps 0→1.0 over Phase 2.
- **Orthogonality** (`L_orth`, β=0.02): Barlow-Twins cross-correlation between 32-D `pad_embedding` and 256-D `identity_embedding`, normalized by `max(d_p,d_i)` and gated on `B ≥ max(d_p,d_i)/4` (TASK_02).
- **Load Balancing** (`L_balance`, γ=0.05): Switch-Transformer `E·Σ(f·P)` summed across 3 MoE layers. Uniform optimum = **6.0** (not collapse). Always active.

## Datasets

- **Identity**: FVC2004 + NIST SD302/300/302a. `group_by_subject=true` merges NIST SD300 `frgp_0X` folders per subject (~10,300 → ~1,030 classes × ~20 samples).
- **PAD**: LivDet 2013/2015/2017. PAD dataset emits `sensor_labels` for the optional DANN head.
- **Joint**: MSU-FPAD v2.0 (both identity + liveness labels).

Samplers: `PKSampler(p=32, k=2)` × `identity_num_views=2` = 128 identity samples/batch; `BalancedPADSampler(batch_size=64)` (lowered from 128 to fit dual-forward Phase 2 joint step on 11 GiB).

## Tech Stack

- **PyTorch Lightning** — `OMFRModule`, `OMFRDataModule`, `PhaseSchedulerCallback`, `GradientMonitor`.
- **timm** — FastViT-SA12 (`timm.create_model("fastvit_sa12", pretrained=True)`).
- **AdamW** (wd=0.05, β=(0.9, 0.999)), **CosineAnnealingLR** (T_max=60, eta_min=1e-6), differential LR per component group.
- **Mixed precision** (`16-mixed`), `gradient_clip_val=0.5`, `accumulate_grad_batches=2` (effective batch ≈ 256), `grad_checkpoint=true` on the backbone.
- **Single GPU** target (RTX 4070 12GB class).

## Build & Run

```bash
# Training (uses configs/base.yaml)
python train.py
python train.py --config configs/base.yaml --resume checkpoints/last.ckpt

# Export
python export.py --checkpoint checkpoints/omfr-best.ckpt --output exports/omfr.onnx
```

## Evaluation Metrics

- **PAD**: APCER, BPCER, ACER.
- **Matching**: TAR@FAR=1e-4 at each MRL dim {64, 128, 256}.
- **Integrated**: cascaded accuracy (PAD rejects spoofs first, then matches remaining).
- Primary checkpoint metric: `val/cascaded_IM` (maximize).

## Key Design Decisions

- **FastViT-SA12 over TinyViT/ViT-Tiny**: RepMixer stages 0–2 give efficient local mixing (important for ridge/minutiae detail) while stage 3 supplies genuine multi-head self-attention over 49 tokens for identity. Total ~10.5M params vs. 5.4M TinyViT — a quality-for-compute trade that fits the single-GPU budget.
- **MoE placed at stage1.block1 + stage2.blocks{2,5}**: after the first downsample (enough spatial context) but before the self-attention stage (still full 28²/14² token grids, useful routing signal). No MoE in stage 3: only 49 tokens, global attention already cheap.
- **Keep FastViT's depthwise-7×7 mixer in MoE blocks**: the `ConvMoEMlpWrapper` preserves the pretrained spatial mixer and only replaces the pointwise expansion. This keeps the pretrained inductive bias usable.
- **Routing patterns as PAD signal**: the core contribution. MoE routing stats (expert weights, token entropy) are concatenated with spatial token features in the PAD head. Live fingerprints route tokens diversely (rich micro-texture); spoofs route uniformly (synthetic/uniform texture).
- **Orthogonal subspaces**: PAD (32-D) and identity (256-D) embeddings are forced decorrelated so liveness detection does not leak identity and vice versa.
- **MRL (Matryoshka)**: identity embedding supports variable-dim matching — 64-D fast, 256-D accurate, one model.
- **PAD gradient isolation**: all backbone outputs are `.detach()`ed before PADHead to prevent LivDet sensor signatures from corrupting the identity backbone (previously caused catastrophic test-time EER despite near-zero train BCE).
- **Joint Phase 2 step (TASK_04)**: identity + PAD batches are processed in a **single** optimizer step, not alternated on `batch_idx%2`. This stabilizes Adam's second-moment estimate across loss-flavor changes and eliminates the periodic total-loss spikes seen in the previous scheme.

## Optimizer Configuration

Differential LR groups (relative to base LR=1e-4):
- `gabor`: 0.1× (stable learned filters)
- `backbone_embed`: 1× (patch/conv stem)
- `backbone_early` (stages 0–1): 1×
- `backbone_late` (stages 2–3): 1×
- `moe_experts`: 2× (new parameters, need to catch up)
- `identity_head`: 1×
- `pad_head`: 1× in Phase 1, 2× in Phase 2
- `arcface`: 0.1× (reduced from 10× — previously caused gradient explosion with many classes)

## AMP Notes

- `FrequencyGate` promotes the FFT path to fp32. cuFFT in fp16 only supports power-of-2 signal sizes; FastViT grid sizes 28 and 14 are not POT, so forcing fp32 is mandatory — do **not** try to run FFT in fp16.
- `FreqGatedMoEFFN.forward` casts `weighted` back to `output_flat.dtype` before `index_add`, because experts can emit a different precision than the accumulation buffer under autocast.
