"""
pad_head.py — PAD Branch Head (v4: Dedicated Stem + DS-Conv Aux + Routing)

Input: {
    'pad_stem_feat':     (B, 128),              # from PADStem (TRAINABLE)
    'stage1_feat':       (B, 64, 56, 56),       # detached backbone feature
    'stage2_feat':       (B, 128, 28, 28),      # detached backbone feature
    'routing_stats_s2':  {'expert_weights': ..., 'token_entropy': ..., 'gate_input_gateonly': (B, 784, 3)},
    'routing_stats_s3a': {'expert_weights': ..., 'token_entropy': ..., 'gate_input_gateonly': (B, 196, 3)},
    'routing_stats_s3b': {'expert_weights': ..., 'token_entropy': ..., 'gate_input_gateonly': (B, 196, 3)},
}

Output: {
    'pad_embedding': (B, 32),    # L2-normalized, for SupCon + orthogonality
    'pad_logit':     (B, 1),     # Raw logit for BCE
    'pad_features':  (B, 128),   # Pre-projection features for SupCon
}

[NEW] v4.1 Update:
    Added 3-band frequency energy (gate_input) directly to the routing stats.
    This bypasses the MoE Softmax temperature collapse (T=0.5) that previously
    squashed token_entropy to 0. PAD now extracts statistical features (mean,
    std, max, min) for each frequency band, adding 12 robust features per MoE layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PADConvBlock(nn.Module):
    """
    Depthwise-separable conv block preserving spatial resolution.

    Flow: DW3x3 -> GN -> GELU -> PW1x1 -> GN -> GELU -> DW3x3 -> GN -> GELU
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        gn_in = self._pick_groups(in_ch)
        gn_out = self._pick_groups(out_ch)

        self.block = nn.Sequential(
            # DW 3x3 on input channels
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                      groups=in_ch, bias=False),
            nn.GroupNorm(gn_in, in_ch),
            nn.GELU(),
            # PW 1x1 to change channel count
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(gn_out, out_ch),
            nn.GELU(),
            # DW 3x3 to refine on new channel basis
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,
                      groups=out_ch, bias=False),
            nn.GroupNorm(gn_out, out_ch),
            nn.GELU(),
        )

    @staticmethod
    def _pick_groups(channels: int) -> int:
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PADHead(nn.Module):
    """
    Multi-scale feature aggregation + dedicated PAD stem + MoE routing.

    Architecture (v4.1):
        -- PAD stem path (TRAINABLE, runs upstream in OMFRModule) --
        pad_stem_feat (B, 128)

        -- Aux feature path on detached backbone stages --
        stage1_feat (B,64,56,56)  -> PADConvBlock(64->128)  -> GAP -> (B,128)
        stage2_feat (B,128,28,28) -> PADConvBlock(128->128) -> GAP -> (B,128)
        -> concat -> (B, 256)

        -- Routing path (20 features per MoE layer x 3 layers = 60) -- [NEW]
        Per MoE layer:
            expert_weights: mean over tokens -> (B, 4)
            token_entropy:  [mean, std, max, min] -> (B, 4)
            gate_input:     [mean, std, max, min] x 3 bands -> (B, 12)
        -> concat all 3 layers -> (B, 60)

        -- Fusion --
        concat(pad_stem_128, feature_256, routing_60) -> (B, 444) [NEW]
        -> LayerNorm -> MLP -> pad_features (B, 128)

        -- Two PARALLEL output branches --
        pad_features -> Linear(128, 32) -> L2Norm -> pad_embedding
        pad_features -> Linear(128, 1) -> pad_logit
    """

    FEAT_CH_PER_STAGE = 128
    PAD_STEM_DIM = 256
    
    # [NEW] Updated dimension: 4 expert + 4 entropy + 12 gate_input (3 bands * 4 stats) = 20
    ROUTE_DIM_PER_LAYER = 20   
    NUM_MOE_LAYERS = 3
    
    # [NEW] 20 * 3 = 60
    ROUTE_DIM = ROUTE_DIM_PER_LAYER * NUM_MOE_LAYERS   
    
    PAD_FEATURES_DIM = 128
    PAD_EMBEDDING_DIM = 32

    def __init__(
        self,
        stage1_dim: int = 64,
        stage2_dim: int = 128,
        pad_stem_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pad_stem_dim = pad_stem_dim

        # Aux feature path on detached backbone stages
        self.block_s1 = PADConvBlock(stage1_dim, self.FEAT_CH_PER_STAGE)
        self.block_s2 = PADConvBlock(stage2_dim, self.FEAT_CH_PER_STAGE)
        self.gap = nn.AdaptiveAvgPool2d(1)

        feat_dim = self.FEAT_CH_PER_STAGE * 2              # 256
        
        # [NEW] fusion_in automatically adapts to 128 + 256 + 60 = 444
        fusion_in = pad_stem_dim + feat_dim + self.ROUTE_DIM   

        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.PAD_FEATURES_DIM),
        )

        # Parallel output branches from pad_features
        self.embedding_proj = nn.Linear(self.PAD_FEATURES_DIM, self.PAD_EMBEDDING_DIM)
        self.logit_head = nn.Linear(self.PAD_FEATURES_DIM, 1)

    def _extract_routing_features(self, stats: Dict) -> torch.Tensor:
        """
        Extract 20-D feature vector from a single MoE layer's routing stats.
        """
        # Note: Depending on how OMFRModule passes the dict, the keys might have 
        # '_gateonly' suffix. We use .get() for robust fallback.
        ew = stats.get('expert_weights_gateonly', stats.get('expert_weights'))   # (B, N, 4)
        ent = stats.get('token_entropy_gateonly', stats.get('token_entropy'))    # (B, N)
        
        # [NEW] Retrieve the detached gate_input we added in moe_ffn.py
        gate_input = stats.get('gate_input_gateonly', stats.get('gate_input'))   # (B, N, 3)

        # 1. Expert load: mean across tokens -> (B, 4)
        load_mean = ew.mean(dim=1)

        # 2. Entropy statistics: [mean, std, max, min] -> (B, 4)
        ent_stats = torch.stack([
            ent.mean(dim=1),
            ent.std(dim=1),
            ent.max(dim=1).values,
            ent.min(dim=1).values,
        ], dim=-1)

        features_list = [load_mean, ent_stats]

        # 3. Gate Input statistics [NEW]
        # Calculates mean, std, max, min for EACH of the 3 frequency bands
        if gate_input is not None:
            gate_stats = torch.cat([
                gate_input.mean(dim=1),          # (B, 3)
                gate_input.std(dim=1),           # (B, 3)
                gate_input.max(dim=1).values,    # (B, 3)
                gate_input.min(dim=1).values,    # (B, 3)
            ], dim=-1)                           # Result: (B, 12)
            features_list.append(gate_stats)

        # Concat all into (B, 20)
        return torch.cat(features_list, dim=-1)  

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: Dictionary containing pad_stem_feat, stage1/2 feats, and routing stats.
        """
        pad_stem = inputs['pad_stem_feat']    # (B, 128)
        s1 = inputs['stage1_feat']            # (B, 64, 56, 56)
        s2 = inputs['stage2_feat']            # (B, 128, 28, 28)

        # -- Aux feature path --
        f1 = self.block_s1(s1)                 # (B, 128, 56, 56)
        f2 = self.block_s2(s2)                 # (B, 128, 28, 28)
        f1 = self.gap(f1).flatten(1)           # (B, 128)
        f2 = self.gap(f2).flatten(1)           # (B, 128)
        feat_combined = torch.cat([f1, f2], dim=-1)  # (B, 256)

        # -- Routing path --
        # [NEW] These are now (B, 20) each
        r_s2 = self._extract_routing_features(inputs['routing_stats_s2'])    
        r_s3a = self._extract_routing_features(inputs['routing_stats_s3a'])  
        r_s3b = self._extract_routing_features(inputs['routing_stats_s3b'])  
        
        # [NEW] route_combined is now (B, 60)
        route_combined = torch.cat([r_s2, r_s3a, r_s3b], dim=-1)  

        # -- Fusion --
        # [NEW] all_features is now (B, 444) -> 128 + 256 + 60
        all_features = torch.cat(
            [pad_stem, feat_combined, route_combined], dim=-1,
        )  
        pad_features = self.fusion_mlp(all_features)  # (B, 128)

        # -- Parallel output branches --
        pad_embedding = F.normalize(
            self.embedding_proj(pad_features), p=2, dim=-1
        )  # (B, 32)
        pad_logit = self.logit_head(pad_features)  # (B, 1)

        return {
            'pad_features': pad_features,
            'pad_embedding': pad_embedding,
            'pad_logit': pad_logit,
        }