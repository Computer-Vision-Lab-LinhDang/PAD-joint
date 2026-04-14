# OMFR: Orthogonal Matryoshka Fingerprint Representation
## Implementation Plan — Chi tiết Module, I/O, Loss Flow, và Training Protocol
### Revision: ViT-Tiny + Frequency-Gated MoE Backbone (replaces TinyViT)

---

## 1. Codebase Structure

```
omfr/
├── configs/
│   ├── base.yaml                # Shared hyperparams
│   ├── phase1_identity.yaml     # Phase 1 config
│   ├── phase2_integration.yaml  # Phase 2 config
│   └── phase3_refinement.yaml   # Phase 3 config
│
├── models/
│   ├── backbone/
│   │   ├── vit_tiny.py          # ViT-Tiny (embed_dim=192, depth=12, flat tokens)
│   │   ├── moe_ffn.py           # Frequency-Gated MoE-FFN block (layers 3, 7, 10)
│   │   ├── frequency_gate.py    # 2D-FFT band-energy gating network
│   │   └── gabor_stem.py        # Learnable Gabor preprocessing
│   │
│   ├── heads/
│   │   ├── identity_head.py     # MRL-ArcFace head (multi-dim, from layer-12 tokens)
│   │   └── pad_head.py          # SupCon PAD head (32-D, uses features + routing stats)
│   │
│   ├── losses/
│   │   ├── arcface.py           # ArcFace with MRL support
│   │   ├── supcon.py            # Supervised Contrastive Loss
│   │   └── orthogonal.py        # Orthogonality regularization
│   │
│   └── omfr.py                  # Main OMFRModule (LightningModule)
│
├── data/
│   ├── datasets/
│   │   ├── identity_dataset.py  # FVC, NIST SD302 (identity labels)
│   │   ├── pad_dataset.py       # LivDet 2015/2017 (liveness labels)
│   │   └── joint_dataset.py     # MSU-FPAD (both labels)
│   │
│   ├── samplers/
│   │   ├── pk_sampler.py        # P identities × K samples/identity
│   │   └── balanced_pad_sampler.py  # 50/50 live/spoof
│   │
│   ├── transforms.py            # Fingerprint-specific augmentations
│   └── datamodule.py            # OMFRDataModule (LightningDataModule)
│
├── callbacks/
│   ├── phase_scheduler.py       # Phase transition logic
│   ├── gradient_monitor.py      # Track gradient conflicts
│   └── embedding_visualizer.py  # t-SNE/UMAP visualization
│
├── evaluation/
│   ├── matching_eval.py         # TAR@FAR, CMC curves
│   ├── pad_eval.py              # APCER, BPCER, ACER
│   └── integrated_eval.py       # RIAPAR, cascaded metrics
│
├── train.py                     # Entry point
└── export.py                    # ONNX/TensorRT export
```

---

## 2. Module Definitions — I/O Specs

### 2.1. `gabor_stem.py` — Learnable Gabor Preprocessing

```
Input:  (B, 1, H, W) — grayscale fingerprint image, H=W=224
Output: (B, 3, H, W) — 3-channel enhanced image (original + 2 Gabor responses)
```

```python
class LearnableGaborStem(nn.Module):
    """
    Learnable Gabor filter bank → ridge/valley enhancement.
    Output 3 channels: [original, gabor_ridge, gabor_valley]
    để ViT-Tiny nhận input 3-channel (compatible với ImageNet pretrained weights).
    
    Parameters:
        n_orientations: int = 8     # Số hướng Gabor
        n_frequencies: int = 4      # Số tần số  
        kernel_size: int = 31       # Kích thước kernel
        learnable: bool = True      # Cho phép learn freq/orientation
    
    Learnable params: frequency (σ), orientation (θ) per filter
    Fixed params: kernel_size, spatial envelope
    
    Forward:
        gabor_bank(image) → (B, n_ori*n_freq, H, W)
        channel_attention → select top-2 responses
        concat [original, top1, top2] → (B, 3, H, W)
    """
```

**Lý do 3 channels:** ViT-Tiny pretrained trên ImageNet RGB (3-ch). Giữ 3-ch cho phép initialize từ pretrained weights mà không cần modify patch embedding projection.

---

### 2.2. `vit_tiny.py` — Backbone (ViT-Tiny with Freq-Gated MoE)

```
Input:  (B, 3, 224, 224) — enhanced fingerprint
Output: Dict {
    'layer3_tokens':   (B, 196, 192),   # After MoE layer 3 — PAD tap (early)
    'layer7_tokens':   (B, 196, 192),   # After MoE layer 7 — PAD tap (mid)
    'layer10_tokens':  (B, 196, 192),   # After MoE layer 10 — late features
    'layer12_tokens':  (B, 196, 192),   # Final layer output — identity tap
    'cls_token':       (B, 192),        # CLS token — identity tap
    'routing_stats': {
        3:  {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
        7:  {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
        10: {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
    },
    'balance_losses': [scalar, scalar, scalar],  # One per MoE layer
}
```

```python
class ViTTinyBackbone(nn.Module):
    """
    ViT-Tiny (flat, 12 layers) với MoE-FFN tại layers 3, 7, 10.
    Tất cả 196 tokens ở cùng resolution 14×14 qua toàn bộ 12 layers.

    Config:
        embed_dim:    192
        depth:        12 layers
        num_heads:    3
        mlp_ratio:    4.0  → FFN hidden = 768
        patch_size:   16   → 224/16 = 14×14 = 196 spatial tokens
        params:       ~8.35M (với MoE, không kể heads)

    MoE layers: {3, 7, 10}  — standard FFN tại tất cả layers còn lại

    Tại sao ViT-Tiny thay TinyViT:
        - Flat architecture: mọi token đều có thể attend to nhau ở MỌI layer
        - FFN structure đồng nhất → MoE drop-in replacement trivial
        - MoE routing là LEARNED (không hard-coded như hierarchical stages)
        - Routing stats là PAD signal mới: live vs spoof có routing pattern khác nhau
    """

def get_layer_params(self, layers: List[int]) -> Iterator[nn.Parameter]:
    """Get parameters of specific layers for differential LR."""

def get_moe_params(self) -> Iterator[nn.Parameter]:
    """Get parameters of all MoE-FFN blocks (experts + gates)."""
```

---

### 2.2a. `frequency_gate.py` — FrequencyGate

```
Input:  tokens (B, 196, 192)
Output: gate_input (B, 196, 3)  — [low_energy, mid_energy, high_energy] per token
```

```python
class FrequencyGate(nn.Module):
    """
    Phân tích frequency content trong spatial arrangement của tokens.
    196 tokens được reshape về grid 14×14, áp dụng 2D-FFT, phân band.

    Band definitions (trên 14×14 frequency grid, distance from DC):
        low  : freq_dist < 14//4  = 3.5 cycles  (~coarse structure)
        mid  : 3.5 ≤ freq_dist < 7  (minutiae-scale)
        high : freq_dist ≥ 7         (pores, micro-texture → PAD-critical)

    Forward:
        spatial = tokens.reshape(B, 14, 14, 192).permute(0, 3, 1, 2)  # (B, 192, 14, 14)
        power   = fft2(spatial, norm='ortho').abs() ** 2
        [low_energy, mid_energy, high_energy] = sum over D dimension per band mask
        gate_input = LayerNorm(stack([low, mid, high], dim=-1))        # (B, 196, 3)
    """
```

---

### 2.2b. `moe_ffn.py` — Frequency-Gated MoE-FFN Block

```
Input:  tokens (B, 196, 192)   — attention output
Output: tokens (B, 196, 192)   — same shape (residual added outside)
Side:   routing_stats {
            'expert_weights':  (B, 196, 4),
            'token_entropy':   (B, 196),
            'balance_loss':    scalar,
        }
```

```python
class FreqGatedMoEFFN(nn.Module):
    """
    Frequency-Gated MoE replaces standard FFN tại layers {3, 7, 10}.

    Config:
        num_experts:  4
        top_k:        2        (sparse routing, top-2 of 4)
        embed_dim:    192
        ffn_hidden:   768      (mlp_ratio=4.0)

    Parameters:
        Standard FFN (1 layer): 192×768×2 = 295K params
        MoE-FFN     (1 layer): 4 × 295K + gate(3→4) = 1.18M params

    FLOPs per token: 2× standard FFN (top-2 of 4 experts active)

    Pipeline:
        1. gate_input = FrequencyGate(tokens)             # (B, 196, 3)
        2. expert_weights = Softmax(Linear(3, 4)(gate_input))  # (B, 196, 4)
        3. top2_indices, top2_weights = TopK(expert_weights, k=2)
        4. output = Σ_{k∈top2} weight_k × Expert_k(tokens)
        5. token_entropy = -Σ p_k·log(p_k)               # (B, 196)
        6. balance_loss = CV(expert_load)²                # scalar

    Expert load = fraction of tokens routed to each expert (target: 25% each).
    L_balance penalizes coefficient of variation → encourages uniform utilization.
    """
```

**Parameter budget summary:**
```
ViT-Tiny base (standard FFN all layers):  5.70M
- Remove 3 standard FFN layers:          -0.89M
+ Add 3 MoE-FFN layers (4 experts each): +3.54M
+ Add 3 FrequencyGate (gating networks): +0.003M
────────────────────────────────────────────────
ViT-Tiny + Freq-Gated MoE:               ~8.35M

FLOPs increase: ~15% overall (MoE only at 3/12 layers, top-2 routing)
```

---

### 2.3. `pad_head.py` — PAD Branch

```
Input:  {
    'layer3_tokens':  (B, 196, 192),
    'layer7_tokens':  (B, 196, 192),
    'routing_stats_3': {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
    'routing_stats_7': {'expert_weights': (B, 196, 4), 'token_entropy': (B, 196)},
}
Output: Dict {
    'pad_embedding': (B, 32),           # L2-normalized, 32-D
    'pad_logit':     (B, 1),            # Sigmoid logit for BCE
    'pad_features':  (B, 128),          # Pre-projection features (for SupCon)
}
```

```python
class PADHead(nn.Module):
    """
    PAD feature aggregation từ layer 3 + 7 tokens, kết hợp routing patterns.

    ★ Novelty: PAD branch khai thác cả features VÀ routing statistics.
       Insight: spoof images có routing pattern đồng nhất (uniform texture →
       same experts cho mọi token), live images có routing đa dạng (rich
       micro-texture → diverse expert routing). Routing entropy = PAD signal.

    Architecture:
        ── Feature path ──
        layer3_tokens (B, 196, 192) → AttentionPool → (B, 256)
        layer7_tokens (B, 196, 192) → AttentionPool → (B, 256)
        feat_combined = Concat(256+256) → (B, 512)

        ── Routing path ──
        routing_stats_3['expert_weights'].mean(dim=1) → (B, 4)   # mean expert load
        routing_stats_7['expert_weights'].mean(dim=1) → (B, 4)
        routing_stats_3['token_entropy'].mean(dim=1)  → (B, 1)   # mean routing entropy
        routing_stats_7['token_entropy'].mean(dim=1)  → (B, 1)
        route_combined = Concat → (B, 10)

        ── Fusion ──
        all_features = Concat(feat_combined, route_combined) → (B, 522)
        pad_features = fusion_mlp(all_features) → (B, 128)
            [LayerNorm → Linear(522, 256) → GELU → Linear(256, 128)]

        pad_embedding = Linear(128, 32) → L2Norm → (B, 32)  [for SupCon]
        pad_logit     = Linear(128, 1)  → (B, 1)             [for BCE]

    Tại sao layer 3 + 7:
        Layer 3: early routing captures pixel-level texture (pores, ridges)
        Layer 7: mid routing captures higher-order texture patterns
        Late layers (10+) attend globally → wash out local spoof artifacts
    """
```

---

### 2.4. `identity_head.py` — Identity Branch (MRL-ArcFace)

```
Input:  {
    'layer12_tokens': (B, 196, 192),    # Final ViT layer output
    'cls_token':      (B, 192),         # CLS token (global summary)
}
Output: Dict {
    'identity_embedding': (B, 256),      # L2-normalized, full 256-D
    'mrl_embeddings': {                  # Nested embeddings
        64:  (B, 64),                    # L2-normalized prefix
        128: (B, 128),
        256: (B, 256),
    },
    'mrl_logits': {                      # ArcFace logits per dimension
        64:  (B, num_classes),
        128: (B, num_classes),
        256: (B, num_classes),
    },
}
```

```python
class IdentityHead(nn.Module):
    """
    MRL-ArcFace head with structural attention on layer-12 tokens.

    Architecture:
        layer12_tokens (B, 196, 192)  # all spatial tokens
        cls_token      (B, 192)       # prepend as extra token

        combined = Concat([cls_token.unsqueeze(1), layer12_tokens])  # (B, 197, 192)
        → Linear(192, 256) → (B, 197, 256)

        → Structural Attention Block (1 layer):
            Q, K, V = Linear(tokens)
            RPE = f(Δx, Δy, Δθ, Δf)   # relative position encoding
                  Δθ = angular diff between token positions (fingerprint-aware)
                  Δf = frequency-band affinity from FrequencyGate
            Attn = softmax(QK^T/√d + RPE) @ V

        → Attentive Pooling → (B, 256)
        → L2Norm → identity_embedding

        MRL: tại {64, 128, 256}:
            z_m = identity_embedding[:, :m]
            z_m = L2Norm(z_m)
            logits_m = ArcFaceClassifier_m(z_m)

    ArcFace params:
        num_classes: int     # Số identities trong training set
        scale: float = 64.0  # Angular scale
        margin: float = 0.5  # Additive angular margin

    MRL dims: [64, 128, 256]
    Mỗi dim có ArcFace classifier RIÊNG (không share weights)

    Tại sao layer 12 (không phải earlier layers):
        Layer 12 = deepest semantic features, global context qua 12 layers
        của full-resolution global attention → tốt nhất cho identity matching
    """
```

---

### 2.5. `omfr.py` — Main Lightning Module

```python
class OMFRModule(L.LightningModule):
    """
    Orchestrates backbone + heads + losses + phase switching.

    Key state:
        self.current_phase: int ∈ {1, 2, 3}
        self.phase_epoch_offset: int  # Global epoch khi phase bắt đầu
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.gabor    = LearnableGaborStem(...)
        self.backbone = ViTTinyBackbone(...)        # ← ViT-Tiny + MoE (not TinyViT)
        self.pad_head = PADHead(...)
        self.identity_head = IdentityHead(...)

        # Losses
        self.arcface_losses = nn.ModuleDict({
            '64':  ArcFaceLoss(embedding_dim=64,  num_classes=N, s=64, m=0.5),
            '128': ArcFaceLoss(embedding_dim=128, num_classes=N, s=64, m=0.5),
            '256': ArcFaceLoss(embedding_dim=256, num_classes=N, s=64, m=0.5),
        })
        self.supcon_loss = SupConLoss(temperature=0.07)
        self.bce_loss    = nn.BCEWithLogitsLoss()
        self.orth_loss   = OrthogonalityLoss()
        # γ·L_balance: sum of load-balancing losses from 3 MoE layers
        # returned directly by backbone as 'balance_losses' list

        # Loss weights
        self.alpha = 0.0   # PAD weight (ramped in Phase 2)
        self.beta  = 0.0   # Orthogonality weight (ramped in Phase 2)
        self.gamma = 0.01  # Balance loss weight (fixed, small)

        # Phase control
        self.current_phase = 1
```

---

## 3. Loss Flow — Chi tiết cho từng Phase

### 3.1. Phase 1: Identity Foundation

```
                    ┌─────────────────────────────────────────────┐
                    │            PHASE 1 — Identity Only           │
                    │         Epochs 1→20, Identity batches        │
                    └─────────────────────────────────────────────┘

    Input: (images, identity_labels)     # Từ IdentityDataset
           images: (B, 1, 224, 224)
           identity_labels: (B,) LongTensor

    ┌──────────┐    ┌─────────────────────┐    ┌──────────────────────┐
    │  Gabor   │───→│  ViT-Tiny + MoE     │───→│   Identity Head      │
    │  Stem    │    │  (layers 1-12)      │    │   (MRL-ArcFace)      │
    └──────────┘    │  MoE @ 3,7,10       │    └──────────────────────┘
                    └─────────────────────┘              │
                       layer12 + CLS ─────────────────────┘
                                              │
                         ┌────────────────────┼─────────────────────┐
                         │                    │                     │
                    ArcFace_64           ArcFace_128           ArcFace_256
                    loss_64              loss_128              loss_256
                         │                    │                     │
                         └────────────────────┼─────────────────────┘
                                              │
                                    L_phase1 = Σ loss_m / 3 + γ·L_balance
                                    (equal weight across dims)

    Gradients flow through:
        ✅ Gabor Stem
        ✅ ViT-Tiny ALL layers (1-12)
        ✅ MoE experts + gates (also receive L_balance gradient)
        ✅ Identity Head
        ✅ ArcFace classifiers (64, 128, 256)

    Frozen:
        ❌ PAD Head (not instantiated / zero grad)
        ❌ SupCon loss
        ❌ Orthogonality loss

    ArcFace config Phase 1:
        scale = 32.0  (thấp hơn bình thường → gentle gradients)
        margin = 0.5
```

```python
def training_step(self, batch, batch_idx):
    if self.current_phase == 1:
        images, identity_labels = batch

        enhanced = self.gabor(images)
        backbone_out = self.backbone(enhanced)

        # Identity head uses layer-12 tokens + CLS
        id_out = self.identity_head({
            'layer12_tokens': backbone_out['layer12_tokens'],
            'cls_token':      backbone_out['cls_token'],
        })

        loss = 0.0
        for dim_key, logits in id_out['mrl_logits'].items():
            loss += self.arcface_losses[dim_key](logits, identity_labels)
        loss = loss / len(id_out['mrl_logits'])  # Average over dims

        # Load balancing loss from MoE layers (always active)
        l_balance = sum(backbone_out['balance_losses'])
        loss += self.gamma * l_balance

        self.log('train/identity_loss', loss, prog_bar=True)
        self.log('train/balance_loss', l_balance)
        return loss
```

---

### 3.2. Phase 2: PAD Integration

```
                    ┌─────────────────────────────────────────────┐
                    │         PHASE 2 — PAD Integration            │
                    │   Epochs 20→40, Alternating batches          │
                    └─────────────────────────────────────────────┘

    Hai loại batch XOAY PHIÊN:

    ═══ Identity Batch (even batch_idx) ═══

    Input: (images, identity_labels)

    Gabor → ViT-Tiny+MoE → Identity Head (layer12+CLS) → MRL-ArcFace losses
                         └→ PAD Head (layer3+7+routing) → pad_embedding (Ort only)
                                                                │
                                             L_orth = CrossCorr(pad_emb, id_emb)²

    L_identity_batch = Σ ArcFace_m / 3 + β·L_orth + γ·L_balance

    ═══ PAD Batch (odd batch_idx) ═══

    Input: (images, liveness_labels)     # 0=spoof, 1=live

    Gabor → ViT-Tiny+MoE → PAD Head (layer3+7+routing_stats) → pad_emb, logit, feat
                         └→ Identity Head (layer12+CLS) → id_emb (Ort only, no ArcFace)

    L_supcon = SupConLoss(pad_features, liveness_labels)
    L_bce    = BCEWithLogitsLoss(pad_logit, liveness_labels)
    L_orth   = CrossCorr(pad_embedding, id_embedding)²

    L_pad_batch = α·(L_supcon + L_bce) + β·L_orth + γ·L_balance

    ═══ Gradient Control ═══

    Identity batch gradients:
        ✅ Gabor, ViT-Tiny layers 1-12, MoE experts, Identity Head, ArcFace classifiers
        ✅ PAD Head — gradient ONLY from L_orth

    PAD batch gradients:
        ✅ Gabor, ViT-Tiny layers 1-6 (PAD tap sources — layers 3+7)
        ⚡ ViT-Tiny layers 7-12: gradient from L_orth only (weak, β=0.1)
        ✅ MoE experts at layers 3, 7: full PAD + L_balance gradients
        ✅ PAD Head full gradient
        ✅ Identity Head — gradient ONLY from L_orth

    α ramp: 0.0 → 1.0 linearly over first 5 epochs of Phase 2
    β ramp: 0.0 → 0.1 linearly over first 5 epochs of Phase 2
    γ = 0.01 fixed throughout all phases
    ArcFace scale: 32 → 64 linearly over first 5 epochs
```

```python
def training_step(self, batch, batch_idx):
    if self.current_phase == 2:
        # Determine batch type
        is_identity_batch = (batch_idx % 2 == 0)

        images = batch['images']
        enhanced = self.gabor(images)
        backbone_out = self.backbone(enhanced)

        id_out = self.identity_head({
            'layer12_tokens': backbone_out['layer12_tokens'],
            'cls_token':      backbone_out['cls_token'],
        })
        pad_out = self.pad_head({
            'layer3_tokens':   backbone_out['layer3_tokens'],
            'layer7_tokens':   backbone_out['layer7_tokens'],
            'routing_stats_3': backbone_out['routing_stats'][3],
            'routing_stats_7': backbone_out['routing_stats'][7],
        })

        # Orthogonality loss (always computed)
        l_orth = self.orth_loss(
            pad_out['pad_embedding'],
            id_out['identity_embedding']
        )

        # Load balancing loss (always active)
        l_balance = sum(backbone_out['balance_losses'])

        if is_identity_batch:
            identity_labels = batch['identity_labels']
            l_id = sum(
                self.arcface_losses[k](v, identity_labels)
                for k, v in id_out['mrl_logits'].items()
            ) / 3
            loss = l_id + self.beta * l_orth + self.gamma * l_balance
            self.log('train/identity_loss', l_id)

        else:  # PAD batch
            liveness_labels = batch['liveness_labels']
            l_supcon = self.supcon_loss(
                pad_out['pad_features'], liveness_labels
            )
            l_bce = self.bce_loss(
                pad_out['pad_logit'].squeeze(),
                liveness_labels.float()
            )
            loss = self.alpha * (l_supcon + l_bce) + self.beta * l_orth + self.gamma * l_balance
            self.log('train/pad_loss', l_supcon + l_bce)

        self.log('train/orth_loss', l_orth)
        self.log('train/balance_loss', l_balance)
        self.log('train/total_loss', loss, prog_bar=True)
        return loss
```

---

### 3.3. Phase 3: Joint Refinement

```
                    ┌─────────────────────────────────────────────┐
                    │       PHASE 3 — Joint Refinement             │
                    │   Epochs 40→60, Mixed + alternating          │
                    └─────────────────────────────────────────────┘

    Ba loại batch:

    A) Identity-only batch:
       L = Σ ArcFace_m / 3 + β·L_orth
       
    B) PAD-only batch:
       L = α·(L_supcon + L_bce) + β·L_orth
       
    C) Joint batch (MSU-FPAD hoặc augmented data có cả hai labels):
       L = Σ ArcFace_m / 3 + α·(L_supcon + L_bce) + β·L_orth
       
       + BONUS cross-task signal:
         Nếu sample là spoof: ArcFace loss KHÔNG tính cho sample đó
         (spoof không nên contribute vào identity learning)
         → Mask ArcFace loss: chỉ compute trên live samples

    α = 1.0 (fixed)
    β = 0.1 (fixed)
    ArcFace scale = 64 (fixed)
    
    Gradients: TẤT CẢ modules đều nhận gradient
    
    Hard mining: 
        Identity: samples với highest ArcFace loss (hard negatives)
        PAD: spoofs với lowest SupCon loss (hard-to-detect spoofs)
```

```python
def training_step(self, batch, batch_idx):
    if self.current_phase == 3:
        images = batch['images']
        enhanced = self.gabor(images)
        backbone_out = self.backbone(enhanced)

        id_out = self.identity_head({
            'layer12_tokens': backbone_out['layer12_tokens'],
            'cls_token':      backbone_out['cls_token'],
        })
        pad_out = self.pad_head({
            'layer3_tokens':   backbone_out['layer3_tokens'],
            'layer7_tokens':   backbone_out['layer7_tokens'],
            'routing_stats_3': backbone_out['routing_stats'][3],
            'routing_stats_7': backbone_out['routing_stats'][7],
        })

        loss = 0.0

        # Orthogonality (always)
        l_orth = self.orth_loss(
            pad_out['pad_embedding'],
            id_out['identity_embedding']
        )
        loss += self.beta * l_orth

        # Load balancing (always)
        l_balance = sum(backbone_out['balance_losses'])
        loss += self.gamma * l_balance

        # Identity loss (if labels available)
        if 'identity_labels' in batch:
            labels = batch['identity_labels']
            # Mask: chỉ compute trên live samples nếu có liveness info
            if 'liveness_labels' in batch:
                live_mask = batch['liveness_labels'] == 1
                if live_mask.any():
                    for k, v in id_out['mrl_logits'].items():
                        loss += self.arcface_losses[k](
                            v[live_mask], labels[live_mask]
                        ) / 3
            else:
                for k, v in id_out['mrl_logits'].items():
                    loss += self.arcface_losses[k](v, labels) / 3

        # PAD loss (if labels available)
        if 'liveness_labels' in batch:
            liveness = batch['liveness_labels']
            l_supcon = self.supcon_loss(
                pad_out['pad_features'], liveness
            )
            l_bce = self.bce_loss(
                pad_out['pad_logit'].squeeze(), liveness.float()
            )
            loss += self.alpha * (l_supcon + l_bce)

        self.log('train/balance_loss', l_balance)
        return loss
```

---

## 4. Loss Definitions — Mathematical Detail

### 4.1. MRL-ArcFace Loss

```python
class ArcFaceLoss(nn.Module):
    """
    Input:  embedding (B, m) — L2-normalized, m ∈ {64, 128, 256}
            labels    (B,)   — identity class indices
    Output: scalar loss
    
    Formula:
        cos_θ = W^T · x  (W: (num_classes, m), L2-normalized per row)
        θ = arccos(cos_θ)
        
        Numerator: e^(s · cos(θ_yi + m))  — add margin to target class
        Denominator: Σ_j e^(s · cos(θ_j)) — sum over all classes
        
        L = -log(numerator / denominator)
    
    Mỗi dimension m có W_m riêng (not shared).
    """
```

### 4.2. Supervised Contrastive Loss (PAD)

```python
class SupConLoss(nn.Module):
    """
    Input:  features (B, 128) — PAD features (pre-projection)
            labels   (B,)     — 0=spoof, 1=live
    Output: scalar loss
    
    Formula (SupCon, Khosla et al. 2020):
        For anchor i with label y_i:
            P(i) = {j : y_j = y_i, j ≠ i}  — positive set
            
            L_i = -1/|P(i)| Σ_{p∈P(i)} log [
                exp(z_i · z_p / τ) / Σ_{k≠i} exp(z_i · z_k / τ)
            ]
            
        L = Σ_i L_i / B
    
    τ = 0.07 (temperature)
    
    Tại sao SupCon thay vì BCE alone:
    - SupCon tạo tighter clusters → better generalization to unseen materials
    - BCE chỉ push toward 0/1, SupCon pull same-class together AND push diff-class apart
    """
```

### 4.3. Orthogonality Loss

```python
class OrthogonalityLoss(nn.Module):
    """
    Input:  pad_emb (B, 32)  — L2-normalized PAD embedding
            id_emb  (B, 256) — L2-normalized Identity embedding
    Output: scalar loss

    Cross-correlation matrix (Barlow Twins style):
        C = (pad_emb.T @ id_emb) / B  → (32, 256)
        L = Σ C_ij²   — penalize ALL cross-correlations

    Penalizes toàn bộ cross-correlation (không chỉ direction-level similarity).
    """
```

### 4.4. Load Balancing Loss (MoE)

```python
class LoadBalancingLoss(nn.Module):
    """
    Ensures each expert receives ~equal token load across the batch.
    Computed inside FreqGatedMoEFFN and returned as 'balance_loss'.

    Input:  expert_weights (B, 196, 4) — routing weights per token
    Output: scalar loss

    Formula:
        expert_load_k = mean over (B, N) of routing_weight[:, :, k]  → (4,)
        CV = std(expert_load) / mean(expert_load)
        L_balance = CV²

    Target: each expert receives 25% of total token weight.
    γ = 0.01 (small weight — balancing is auxiliary, not primary objective)
    """
```

---

## 5. DataModule — Phase-aware Loading

```python
class OMFRDataModule(L.LightningDataModule):
    """
    Manages 3 datasets, switches DataLoader per phase.
    
    Datasets:
        identity_ds:  FVC2004 + NIST SD302
                      Returns: {'images': Tensor, 'identity_labels': LongTensor}
                      
        pad_ds:       LivDet 2015 + LivDet 2017
                      Returns: {'images': Tensor, 'liveness_labels': LongTensor}
                      
        joint_ds:     MSU-FPAD v2.0 (has both identity + liveness)
                      Returns: {'images': Tensor, 
                                'identity_labels': LongTensor,
                                'liveness_labels': LongTensor}
    """
    
    def train_dataloader(self):
        phase = self.trainer.lightning_module.current_phase
        
        if phase == 1:
            # Phase 1: Identity only
            sampler = PKSampler(
                self.identity_ds, 
                p=32,     # 32 identities per batch
                k=4,      # 4 samples per identity
            )  # → batch_size = 128
            return DataLoader(
                self.identity_ds, 
                batch_sampler=sampler,
                num_workers=8, pin_memory=True,
            )
            
        elif phase == 2:
            # Phase 2: Alternating identity + PAD
            # Dùng CombinedLoader của Lightning
            id_sampler = PKSampler(self.identity_ds, p=32, k=4)
            pad_sampler = BalancedPADSampler(self.pad_ds, batch_size=128)
            
            id_loader = DataLoader(
                self.identity_ds, batch_sampler=id_sampler,
                num_workers=4, pin_memory=True,
            )
            pad_loader = DataLoader(
                self.pad_ds, batch_sampler=pad_sampler,
                num_workers=4, pin_memory=True,
            )
            # "min_size" mode: stop when shorter loader exhausted
            return {"identity": id_loader, "pad": pad_loader}
            
        elif phase == 3:
            # Phase 3: Mix of all three datasets
            # Trọng số sampling: 50% identity, 30% PAD, 20% joint
            loaders = {
                "identity": DataLoader(self.identity_ds, ...),
                "pad": DataLoader(self.pad_ds, ...),
                "joint": DataLoader(self.joint_ds, ...),
            }
            return loaders
```

**Phase 2 batch handling trong training_step:**

```python
def training_step(self, batch, batch_idx):
    if self.current_phase == 2:
        # Lightning CombinedLoader delivers dict of batches
        # batch = {"identity": id_batch, "pad": pad_batch}
        # Ta xử lý xen kẽ:
        
        if batch_idx % 2 == 0 and 'identity' in batch:
            return self._identity_step(batch['identity'])
        elif 'pad' in batch:
            return self._pad_step(batch['pad'])
```

---

## 6. Phase Transition — Callback Implementation

```python
class PhaseSchedulerCallback(L.Callback):
    """
    Manages transitions between training phases.
    
    Config:
        phase1_epochs: 20
        phase2_epochs: 20  (total: 20-40)
        phase3_epochs: 20  (total: 40-60)
        
        alpha_warmup_epochs: 5   (in Phase 2)
        beta_warmup_epochs: 5    (in Phase 2)
        arcface_scale_warmup: 5  (in Phase 2, 32→64)
    """
    
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        # ── Phase Transition ──
        if epoch == self.phase1_epochs:
            self._transition_to_phase2(pl_module)
        elif epoch == self.phase1_epochs + self.phase2_epochs:
            self._transition_to_phase3(pl_module)
        
        # ── Ramp scheduling within Phase 2 ──
        if pl_module.current_phase == 2:
            phase2_epoch = epoch - self.phase1_epochs
            progress = min(phase2_epoch / self.alpha_warmup_epochs, 1.0)
            
            pl_module.alpha = progress * 1.0    # 0 → 1
            pl_module.beta = progress * 0.1     # 0 → 0.1
            
            # ArcFace scale warmup: 32 → 64
            new_scale = 32.0 + progress * 32.0
            for af in pl_module.arcface_losses.values():
                af.s = new_scale
    
    def _transition_to_phase2(self, pl_module):
        """Phase 1 → Phase 2"""
        pl_module.current_phase = 2
        
        # Khởi tạo PAD head weights (Xavier)
        pl_module.pad_head.apply(self._init_weights)
        
        # Re-configure optimizer: thêm PAD head params
        # → Trigger configure_optimizers() lại
        
        print(f"═══ PHASE 2 START ═══ Adding PAD objective")
    
    def _transition_to_phase3(self, pl_module):
        """Phase 2 → Phase 3"""
        pl_module.current_phase = 3
        pl_module.alpha = 1.0
        pl_module.beta = 0.1
        
        print(f"═══ PHASE 3 START ═══ Joint refinement")
```

---

## 7. Optimizer & LR — Per-phase Configuration

```python
def configure_optimizers(self):
    # Differential learning rates per component
    param_groups = [
        # Gabor stem — low LR (learned filters, don't over-update)
        {'params': self.gabor.parameters(),
         'lr': self.hparams.lr * 0.1,
         'name': 'gabor'},

        # ViT-Tiny layers 1-6 (PAD-tap source layers 3) — standard LR
        {'params': self.backbone.get_layer_params(list(range(1, 7))),
         'lr': self.hparams.lr,
         'name': 'backbone_early'},

        # ViT-Tiny layers 7-12 (identity-tap source layers 7,10,12) — standard LR
        {'params': self.backbone.get_layer_params(list(range(7, 13))),
         'lr': self.hparams.lr,
         'name': 'backbone_late'},

        # MoE expert + gate params — slightly higher LR (new params, need to catch up)
        {'params': self.backbone.get_moe_params(),
         'lr': self.hparams.lr * 2.0,
         'name': 'moe_experts'},

        # Identity Head — standard LR
        {'params': self.identity_head.parameters(),
         'lr': self.hparams.lr,
         'name': 'identity_head'},

        # PAD Head — higher LR in Phase 2 (catch up to backbone)
        {'params': self.pad_head.parameters(),
         'lr': self.hparams.lr * (2.0 if self.current_phase == 2 else 1.0),
         'name': 'pad_head'},

        # ArcFace classifiers — higher LR (large, sparse gradients)
        {'params': [p for af in self.arcface_losses.values()
                    for p in af.parameters()],
         'lr': self.hparams.lr * 10.0,
         'name': 'arcface'},
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=self.hparams.lr,        # default: 1e-4
        weight_decay=0.05,
        betas=(0.9, 0.999),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=self.hparams.total_epochs,  # 60
        eta_min=1e-6,
    )
    
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'epoch',
        }
    }
```

---

## 8. Gradient Flow Summary — Per Phase

```
╔══════════════════════════════════════════════════════════════════════╗
║                      GRADIENT FLOW MATRIX                           ║
╠════════════════════╦══════════╦══════════╦══════════╦══════════════╣
║  Component         ║ Phase 1  ║ Phase 2  ║ Phase 2  ║  Phase 3     ║
║                    ║          ║ ID batch ║ PAD batch║              ║
╠════════════════════╬══════════╬══════════╬══════════╬══════════════╣
║ Gabor Stem         ║ ✅ Arc   ║ ✅ Arc   ║ ✅ PAD+Ort✅ ALL        ║
║ ViT layers 1-6     ║ ✅ Arc   ║ ✅ Arc   ║ ✅ PAD+Ort✅ ALL        ║
║ ViT layers 7-12    ║ ✅ Arc   ║ ✅ Arc   ║ ⚡ Ort only ✅ ALL      ║
║ MoE experts (3,7,10)║ ✅+Bal  ║ ✅+Bal  ║ ✅+Bal    ✅ +Bal       ║
║ PAD Head           ║ ❌ frozen║ ⚡ Ort   ║ ✅ PAD+Ort✅ ALL        ║
║ Identity Head      ║ ✅ Arc   ║ ✅ Arc   ║ ⚡ Ort only ✅ ALL      ║
║ ArcFace classifiers║ ✅ Arc   ║ ✅ Arc   ║ ❌ no grad ✅ ArcFace  ║
╠════════════════════╬══════════╬══════════╬══════════╬══════════════╣
║ L_ArcFace_MRL      ║ ✅ active║ ✅ active║ ❌ skip   ✅ active     ║
║ L_SupCon           ║ ❌ skip  ║ ❌ skip  ║ ✅ active  ✅ active    ║
║ L_BCE              ║ ❌ skip  ║ ❌ skip  ║ ✅ active  ✅ active    ║
║ L_Orth             ║ ❌ skip  ║ ✅ active║ ✅ active  ✅ active    ║
║ L_balance          ║ ✅ always║ ✅ always║ ✅ always  ✅ always    ║
╚════════════════════╩══════════╩══════════╩══════════╩══════════════╝

⚡ = weak gradient (orthogonality only, β=0.1)
+Bal = always receives small gradient from L_balance (γ=0.01)
```

**Implementation Note — Phase 2 PAD batch gradient control:**

```python
def _pad_step(self, batch):
    """Phase 2: PAD batch — limit gradient through layers 7-12 to L_orth only"""
    images, liveness_labels = batch['images'], batch['liveness_labels']

    enhanced = self.gabor(images)
    backbone_out = self.backbone(enhanced)

    # PAD head: full gradient (layers 1-6 + MoE experts receive PAD gradients)
    pad_out = self.pad_head({
        'layer3_tokens':   backbone_out['layer3_tokens'],
        'layer7_tokens':   backbone_out['layer7_tokens'],
        'routing_stats_3': backbone_out['routing_stats'][3],
        'routing_stats_7': backbone_out['routing_stats'][7],
    })

    # Identity head: embedding only (no ArcFace logits)
    # layers 7-12 receive gradient ONLY from L_orth (weak, β=0.1)
    id_emb = self.identity_head.forward_embedding_only({
        'layer12_tokens': backbone_out['layer12_tokens'],
        'cls_token':      backbone_out['cls_token'],
    })
    # ↑ forward_embedding_only: compute embedding, skip ArcFace logits

    # Load balancing always active on all MoE layers
    l_balance = sum(backbone_out['balance_losses'])

    l_supcon = self.supcon_loss(pad_out['pad_features'], liveness_labels)
    l_bce    = self.bce_loss(pad_out['pad_logit'].squeeze(), liveness_labels.float())
    l_orth   = self.orth_loss(pad_out['pad_embedding'], id_emb)

    return self.alpha * (l_supcon + l_bce) + self.beta * l_orth + self.gamma * l_balance
```

---

## 9. Evaluation Protocol

```python
# Validation runs after EVERY epoch, computing task-specific metrics

def validation_step(self, batch, batch_idx, dataloader_idx=0):
    images = batch['images']
    enhanced = self.gabor(images)
    backbone_out = self.backbone(enhanced)

    # ── PAD metrics ──
    if 'liveness_labels' in batch:
        pad_out = self.pad_head({
            'layer3_tokens':   backbone_out['layer3_tokens'],
            'layer7_tokens':   backbone_out['layer7_tokens'],
            'routing_stats_3': backbone_out['routing_stats'][3],
            'routing_stats_7': backbone_out['routing_stats'][7],
        })
        pad_scores = torch.sigmoid(pad_out['pad_logit']).squeeze()
        self.pad_metrics.update(pad_scores, batch['liveness_labels'])

    # ── Matching metrics ──
    if 'identity_labels' in batch:
        id_out = self.identity_head({
            'layer12_tokens': backbone_out['layer12_tokens'],
            'cls_token':      backbone_out['cls_token'],
        })
        # Store embeddings for epoch-end computation
        self.val_embeddings.append(id_out['identity_embedding'].cpu())
        self.val_labels.append(batch['identity_labels'].cpu())

def on_validation_epoch_end(self):
    # PAD: compute APCER, BPCER, ACER
    pad_results = self.pad_metrics.compute()
    self.log('val/ACER', pad_results['ACER'], prog_bar=True)
    
    # Matching: compute TAR@FAR=0.01% for each MRL dimension
    embeddings = torch.cat(self.val_embeddings)
    labels = torch.cat(self.val_labels)
    
    for dim in [64, 128, 256]:
        emb_m = F.normalize(embeddings[:, :dim], dim=-1)
        tar_at_far = compute_tar_at_far(emb_m, labels, far=0.0001)
        self.log(f'val/TAR@FAR0.01%_dim{dim}', tar_at_far)
    
    # Integrated metric: cascaded accuracy
    # Step 1: PAD reject spoofs → Step 2: Match remaining
    cascaded_acc = compute_cascaded_accuracy(
        pad_scores, liveness_labels,
        embeddings, identity_labels,
        pad_threshold=0.5,
    )
    self.log('val/cascaded_IM', cascaded_acc, prog_bar=True)
```

---

## 10. Config Files

```yaml
# configs/base.yaml
seed: 42
image_size: 224
total_epochs: 60

backbone:
  name: vit_tiny_moe
  embed_dim: 192
  depth: 12
  num_heads: 3
  mlp_ratio: 4.0
  patch_size: 16            # → 14×14 = 196 tokens
  pretrained: true          # ImageNet ViT-Tiny pretrained weights
  moe_layers: [3, 7, 10]   # Replace FFN with MoE-FFN at these layers
  moe_num_experts: 4
  moe_top_k: 2
  pad_tap_layers: [3, 7]   # Layers from which PAD head reads tokens
  identity_tap_layer: 12   # Layer from which Identity head reads tokens

gabor:
  n_orientations: 8
  n_frequencies: 4
  kernel_size: 31
  learnable: true

identity_head:
  mrl_dims: [64, 128, 256]
  structural_attention_layers: 1
  num_heads: 8
  input_dim: 192            # ViT-Tiny embed_dim

pad_head:
  embedding_dim: 32
  hidden_dim: 128
  input_token_dim: 192      # ViT-Tiny embed_dim
  routing_stats_dim: 10     # 4+4+1+1 from routing_stats layers 3+7

losses:
  arcface_scale: 64.0
  arcface_margin: 0.5
  supcon_temperature: 0.07
  alpha: 1.0                # PAD loss weight
  beta: 0.1                 # Orthogonality weight
  gamma: 0.01               # Load balancing loss weight (MoE)

optimizer:
  lr: 1e-4
  weight_decay: 0.05

phases:
  phase1_epochs: 20
  phase2_epochs: 20
  phase3_epochs: 20
  alpha_warmup_epochs: 5
  beta_warmup_epochs: 5
  arcface_scale_phase1: 32.0
  arcface_scale_phase2: 64.0
```

---

## 11. Training Entry Point

```python
# train.py
import lightning as L
from omfr.models.omfr import OMFRModule                          # ViT-Tiny + MoE variant
from omfr.models.backbone.vit_tiny import ViTTinyBackbone
from omfr.models.backbone.moe_ffn import FreqGatedMoEFFN
from omfr.data.datamodule import OMFRDataModule
from omfr.callbacks.phase_scheduler import PhaseSchedulerCallback

def main():
    L.seed_everything(42)
    
    config = load_config('configs/base.yaml')
    
    model = OMFRModule(config)
    datamodule = OMFRDataModule(config)
    
    trainer = L.Trainer(
        max_epochs=config['total_epochs'],     # 60
        accelerator='gpu',
        devices=1,                              # Single GPU (Bình An's setup)
        precision='16-mixed',                   # AMP for speed
        gradient_clip_val=1.0,                  # Gradient clipping
        accumulate_grad_batches=2,              # Effective batch = 256
        
        callbacks=[
            PhaseSchedulerCallback(config['phases']),
            L.callbacks.ModelCheckpoint(
                monitor='val/cascaded_IM',
                mode='max',
                save_top_k=3,
                filename='omfr-{epoch:02d}-{val/cascaded_IM:.4f}',
            ),
            L.callbacks.LearningRateMonitor(logging_interval='epoch'),
            GradientConflictMonitor(),  # Custom: track PAD vs ID gradient cosine
            EmbeddingVisualizer(every_n_epochs=5),  # t-SNE plots
        ],
        
        logger=L.loggers.WandbLogger(
            project='OMFR',
            name='tinyVit5m_mrl_supcon_orth',
        ),
        
        deterministic=True,
    )
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
```

---

## 12. Estimated Resource Requirements

```
Training (single RTX 3090 24GB):
  Phase 1:  ~2h   (identity only, batch=128, 20 epochs on ~50K images)
  Phase 2:  ~3h   (alternating, batch=128, 20 epochs)
  Phase 3:  ~2h   (joint, batch=128, 20 epochs)
  Total:    ~7h

Model size:
  ViT-Tiny+MoE backbone:  ~8.35M params
  Gabor Stem:             ~0.02M params
  Identity Head:          ~1.5M params (incl. structural attention)
  PAD Head:               ~0.3M params
  ArcFace classifiers:    ~3×(256×N_classes) ≈ 2M params (N=2000)
  ─────────────────────────────────────────────
  Total trainable:        ~9.2M params
  Inference only:         ~7.2M params (sans ArcFace classifiers)

Inference (Jetson Orin NX):
  PAD only (reject):      ~1.5ms
  Fast search (64-D):     ~3ms
  Full match (256-D):     ~4ms
```
