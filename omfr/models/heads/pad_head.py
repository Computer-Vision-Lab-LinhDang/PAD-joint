"""
pad_head.py — PAD Branch Head (v6: Shared Spatial Latent + FPN-light Fusion)

Input: {
    'shared_spatial_feat':  (B, 256, 14, 14),  # post-attention spatial latent from IdentityHead
    'stage1_feat':          (B, 64, 56, 56),   # detached early backbone feature
    'stage2_feat':          (B, 128, 28, 28),  # detached early backbone feature
    'routing_stats_s2':  {'expert_weights': ..., 'token_entropy': ..., 'gate_input_gateonly': (B, 784, 3)},
    'routing_stats_s3a': {'expert_weights': ..., 'token_entropy': ..., 'gate_input_gateonly': (B, 196, 3)},
    'routing_stats_s3b': {'expert_weights': ..., 'token_entropy': ..., 'gate_input_gateonly': (B, 196, 3)},
}

Output: {
    'pad_embedding': (B, 32),    # L2-normalized PAD-specific embedding from pad_features
    'pad_logit':     (B, 1),     # Raw logit for BCE / Focal BCE
    'pad_features':  (B, 128),   # Pre-projection PAD features (sensor-adv head input)
}

v6 rationale:
    Earlier versions consumed `shared_repr_feat`, a single (B, 256) vector
    obtained by attentive-pooling the structural-attention tokens. PAD
    needs cues that are intrinsically spatial — background halo, edge
    artifacts, material texture, regions where ridges look fake — and
    pooling away the 14x14 grid before fusing with stage1/stage2 wipes
    out exactly those cues.

    v6 takes the 14x14 token grid produced by IdentityHead's structural
    attention block (`shared_spatial_feat`) and fuses it with downsampled
    `stage1_feat` (56→14) and `stage2_feat` (28→14) in a light FPN-style
    tower, applies depthwise-separable refinement, spatial attention,
    and GAP to derive `pad_features`. MoE routing stats stay as a small
    auxiliary signal injected post-pool.

    PAD no longer consumes an identity MRL prefix as its embedding. The PAD
    embedding is projected from `pad_features`, keeping identity MRL and PAD
    classification as separate downstream tasks over the shared trunk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def _pick_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class PADSpatialFusion(nn.Module):
    """
    FPN-light fusion of three resolutions onto the shared latent grid, followed by
    depthwise-separable refinement, spatial attention, and GAP.

    Inputs:
        stage1_feat: downsampled and aligned to shared_spatial_feat
        stage2_feat: downsampled and aligned to shared_spatial_feat
        shared_spatial_feat: projected to hidden_dim
    Output: (B, hidden_dim) pooled spatial vector
    """

    def __init__(
        self,
        stage1_dim: int = 64,
        stage2_dim: int = 128,
        shared_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        gn_h = _pick_groups(hidden_dim)

        # 56x56 -> 14x14: two stride-2 convs with channel projection.
        self.proj_s1 = nn.Sequential(
            nn.Conv2d(stage1_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_h, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_h, hidden_dim),
            nn.GELU(),
        )
        # 28x28 -> 14x14: one stride-2 conv with channel projection.
        self.proj_s2 = nn.Sequential(
            nn.Conv2d(stage2_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_h, hidden_dim),
            nn.GELU(),
        )
        # 14x14 channel reduction for the shared spatial latent.
        self.proj_shared = nn.Sequential(
            nn.Conv2d(shared_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(gn_h, hidden_dim),
            nn.GELU(),
        )

        fused_in = hidden_dim * 3
        gn_f = _pick_groups(fused_in)
        # Depthwise-separable refinement: DW3x3 -> PW1x1 -> DW3x3 on the
        # concatenated 384-ch tensor. Keeps params small while letting
        # each spatial position blend stage-1/stage-2/shared cues.
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, fused_in, kernel_size=3, padding=1, groups=fused_in, bias=False),
            nn.GroupNorm(gn_f, fused_in),
            nn.GELU(),
            nn.Conv2d(fused_in, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(gn_h, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.GroupNorm(gn_h, hidden_dim),
            nn.GELU(),
        )

        # Lightweight 1-channel spatial attention mask.
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1, bias=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        stage1_feat: torch.Tensor,
        stage2_feat: torch.Tensor,
        shared_spatial_feat: torch.Tensor,
    ) -> torch.Tensor:
        f1 = self.proj_s1(stage1_feat)              # (B, H, 14, 14)
        f2 = self.proj_s2(stage2_feat)              # (B, H, 14, 14)
        fs = self.proj_shared(shared_spatial_feat)  # (B, H, 14, 14)
        target_size = fs.shape[-2:]
        if f1.shape[-2:] != target_size:
            f1 = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        if f2.shape[-2:] != target_size:
            f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        x = torch.cat([f1, f2, fs], dim=1)          # (B, 3H, 14, 14)
        x = self.fuse(x)                            # (B, H, 14, 14)
        attn = torch.sigmoid(self.spatial_attn(x))  # (B, 1, 14, 14)
        x = x * attn
        return self.gap(x).flatten(1)               # (B, H)


class PADTextureBranch(nn.Module):
    """
    PAD-specific local texture branch from the earliest available signals.

    The shared identity path is optimized for minutiae/structure. Spoof
    artifacts are often local material cues, so this branch reads detached
    Gabor responses plus stage-0 features, downsamples them only to 28x28,
    and injects a compact residual into the PAD spatial vector.
    """

    def __init__(
        self,
        gabor_dim: int = 8,
        stage1_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        mid_dim = hidden_dim // 2
        gn_mid = _pick_groups(mid_dim)
        gn_h = _pick_groups(hidden_dim)

        self.gabor_tower = nn.Sequential(
            nn.Conv2d(gabor_dim, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_pick_groups(32), 32),
            nn.GELU(),
            nn.Conv2d(32, mid_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_mid, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_mid, mid_dim),
            nn.GELU(),
        )
        self.stage1_tower = nn.Sequential(
            nn.Conv2d(stage1_dim, mid_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_mid, mid_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.GroupNorm(gn_h, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(gn_h, hidden_dim),
            nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        gabor_feat: torch.Tensor,
        stage1_feat: torch.Tensor,
    ) -> torch.Tensor:
        fg = self.gabor_tower(gabor_feat)       # (B, H/2, 28, 28)
        fs = self.stage1_tower(stage1_feat)     # (B, H/2, 28, 28)
        x = torch.cat([fg, fs], dim=1)           # (B, H, 28, 28)
        x = self.fuse(x)
        return self.gap(x).flatten(1)            # (B, H)


class PADHead(nn.Module):
    """
    Spatial-fusion PAD head on top of the shared latent.

    Architecture (v6):
        -- Spatial fusion path --
        stage1_feat (B, 64, 56, 56)  -> proj 56->14 -> (B, 128, 14, 14)
        stage2_feat (B, 128, 28, 28) -> proj 28->14 -> (B, 128, 14, 14)
        shared_spatial_feat (B, 256, 14, 14) -> proj -> (B, 128, 14, 14)
        concat (3 x H) -> DSConv refinement -> (B, 128, 14, 14)
        spatial attention -> GAP -> spatial_vec (B, 128)

        -- Routing path (auxiliary, low-gain by default) --
        per-MoE-layer 20-D stats x 3 -> 60 -> route_mlp -> 24 -> route_vec

        -- Fusion -> pad_features --
        concat(spatial_vec, route_vec) -> Linear -> (B, 128)

        -- Output branches --
        pad_features -> Linear(128, 1) -> pad_logit
        pad_features -> Linear(128, 32) -> L2Norm -> pad_embedding
    """

    # Routing descriptor: 4 expert load + 4 entropy stats + 12 (3 bands x 4) gate stats = 20
    ROUTE_DIM_PER_LAYER = 20
    NUM_MOE_LAYERS = 3
    ROUTE_RAW_DIM = ROUTE_DIM_PER_LAYER * NUM_MOE_LAYERS
    ROUTE_FEATURE_DIM = 24

    SPATIAL_HIDDEN_DIM = 128
    PAD_FEATURES_DIM = 128
    PAD_EMBEDDING_DIM = 32

    def __init__(
        self,
        stage1_dim: int = 64,
        stage2_dim: int = 128,
        shared_dim: int = 256,
        gabor_dim: int = 8,
        dropout: float = 0.1,
        **_legacy_kwargs,
    ):
        super().__init__()

        self.spatial_fusion = PADSpatialFusion(
            stage1_dim=stage1_dim,
            stage2_dim=stage2_dim,
            shared_dim=shared_dim,
            hidden_dim=self.SPATIAL_HIDDEN_DIM,
        )
        self.texture_branch = PADTextureBranch(
            gabor_dim=gabor_dim,
            stage1_dim=stage1_dim,
            hidden_dim=self.SPATIAL_HIDDEN_DIM,
        )
        self.texture_gain = nn.Parameter(torch.tensor(-4.0))

        # Routing bottleneck — sensor-biased raw stats compressed through a
        # small MLP and gated by a learnable scalar that starts low so the
        # frequency cue does not dominate as a sensor shortcut early on.
        self.route_mlp = nn.Sequential(
            nn.LayerNorm(self.ROUTE_RAW_DIM),
            nn.Linear(self.ROUTE_RAW_DIM, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, self.ROUTE_FEATURE_DIM),
        )
        self.route_gain = nn.Parameter(torch.tensor(-2.2))

        fusion_in = self.SPATIAL_HIDDEN_DIM + self.ROUTE_FEATURE_DIM
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.PAD_FEATURES_DIM),
        )

        # Output branches.
        self.embedding_proj = nn.Linear(self.PAD_FEATURES_DIM, self.PAD_EMBEDDING_DIM)
        self.logit_head = nn.Linear(self.PAD_FEATURES_DIM, 1)

    def _extract_routing_features(self, stats: Dict) -> torch.Tensor:
        """20-D routing descriptor from a single MoE layer's stats."""
        ew = stats.get('expert_weights_gateonly', stats.get('expert_weights'))   # (B, N, 4)
        ent = stats.get('token_entropy_gateonly', stats.get('token_entropy'))    # (B, N)
        gate_input = stats.get('gate_input_gateonly', stats.get('gate_input'))   # (B, N, 3)

        load_mean = ew.mean(dim=1)                                                # (B, 4)
        ent_stats = torch.stack([
            ent.mean(dim=1),
            ent.std(dim=1),
            ent.max(dim=1).values,
            ent.min(dim=1).values,
        ], dim=-1)                                                                # (B, 4)

        features = [load_mean, ent_stats]
        if gate_input is not None:
            gate_stats = torch.cat([
                gate_input.mean(dim=1),         # (B, 3)
                gate_input.std(dim=1),          # (B, 3)
                gate_input.max(dim=1).values,   # (B, 3)
                gate_input.min(dim=1).values,   # (B, 3)
            ], dim=-1)                          # (B, 12)
            features.append(gate_stats)
        return torch.cat(features, dim=-1)      # (B, 20)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s1 = inputs['stage1_feat']                          # (B, 64, 56, 56)
        s2 = inputs['stage2_feat']                          # (B, 128, 28, 28)
        shared_spatial = inputs['shared_spatial_feat']      # (B, 256, 14, 14)
        gabor_feat = inputs.get('gabor_feat')               # (B, 8, 224, 224)

        # -- Spatial fusion --
        spatial_vec = self.spatial_fusion(s1, s2, shared_spatial)   # (B, 128)
        if gabor_feat is not None:
            texture_vec = self.texture_branch(gabor_feat, s1).to(spatial_vec.dtype)
            spatial_vec = spatial_vec + torch.sigmoid(self.texture_gain).to(spatial_vec.dtype) * texture_vec

        # -- Routing path --
        r_s2  = self._extract_routing_features(inputs['routing_stats_s2'])
        r_s3a = self._extract_routing_features(inputs['routing_stats_s3a'])
        r_s3b = self._extract_routing_features(inputs['routing_stats_s3b'])
        route_raw = torch.cat([r_s2, r_s3a, r_s3b], dim=-1)         # (B, 60)
        route_gain = torch.sigmoid(self.route_gain).to(route_raw.dtype)
        route_vec = self.route_mlp(route_raw) * route_gain          # (B, 24)

        # -- Fusion --
        fused = torch.cat([spatial_vec, route_vec], dim=-1)         # (B, 152)
        pad_features = self.fusion_mlp(fused)                       # (B, 128)

        # -- Output branches --
        pad_embedding = F.normalize(
            self.embedding_proj(pad_features), p=2, dim=-1,
        )
        pad_logit = self.logit_head(pad_features)                   # (B, 1)

        return {
            'pad_features': pad_features,
            'pad_embedding': pad_embedding,
            'pad_logit': pad_logit,
        }
