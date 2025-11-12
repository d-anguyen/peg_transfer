import torch
import torch.nn as nn
from einops import rearrange, reduce

from .metaspikeformer import Spiking_vit_MetaFormer
from .x3d import X3D
from transformers import AutoModel


class ConsensusPooling(nn.Module):
    """Pool predictions across multiple clips."""
    
    def __init__(self, mode: str = "max"):
        super().__init__()
        self.mode = mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "max":
            return reduce(x, 'n d -> d', 'max')
        elif self.mode == "avg":
            return reduce(x, 'n d -> d', 'mean')
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")


class SpikeFormerVideoWrapper(nn.Module):
    """
    Wrapper for Spiking_vit_MetaFormer.
    Feeds the entire video clip directly to the model's forward_features().
    """
    
    def __init__(self, model: nn.Module, feature_dim: int):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Single clip (C, T, H, W) or batched clips (N, C, T, H, W)
        Returns:
            Features of shape (feature_dim,) for single input or (N, feature_dim) for batch
        """
        if x.ndim == 4:
            x_btchw = rearrange(x, 'c t h w -> t 1 c h w')
        else:
            # (N, C, T, H, W) -> (T, N, C, H, W)
            x_btchw = rearrange(x, 'n c t h w -> t n c h w')
        features = self.model.forward_features(x_btchw)  # (T, B, C, S)
        features = features.flatten(3).mean(3)  # (T, B, C)
        features = reduce(features, 't b c -> b c', 'mean')  # (B, C)
        if x.ndim == 4:
            return features.squeeze(0)
        return features  # (N, C)


class MultiClipClassifier(nn.Module):
    """Multi-clip video classifier."""
    
    def __init__(
        self,
        backbone_type: str = "spikeformer",
        feature_dim: int = 512,
        hidden_dim: int = 512,
        consensus_mode: str = "max",
        dropout: float = 0.5,
        **backbone_kwargs
    ):
        """
        Args:
            backbone_type: 'spikeformer', 'x3d'
            feature_dim: Backbone output dimension (unused, computed from backbone)
            hidden_dim: Hidden layer dimension (unused, computed from expansion_factor)
            consensus_mode: 'max' or 'avg'
            dropout: Dropout rate
            backbone_kwargs: Additional arguments for backbone (including expansion_factor)
        """
        super().__init__()
        
        # Read and remove head-only args from backbone kwargs
        head_expansion_factor = backbone_kwargs.pop('expansion_factor', 2)

        # Create backbone
        if backbone_type == "spikeformer":
            spikeformer = Spiking_vit_MetaFormer(
                detach_reset=backbone_kwargs.get('detach_reset', True),
                img_size_h=backbone_kwargs.get('img_size', 224),
                img_size_w=backbone_kwargs.get('img_size', 224),
                in_channels=backbone_kwargs.get('in_channels', 3),
                num_classes=2,  # Dummy, we'll replace the head
                depths=backbone_kwargs.get('depths', [2, 2, 2, 2]),
                embed_dim=backbone_kwargs.get('dims', [64, 128, 256, 512]),
                num_heads=[d // 32 for d in backbone_kwargs.get('dims', [64, 128, 256, 512])],
            )
            feature_dim = backbone_kwargs.get('dims', [64, 128, 256, 512])[-1]
            self.backbone = SpikeFormerVideoWrapper(spikeformer, feature_dim)
            
        elif backbone_type == "x3d":
            # X3D already handles video input natively
            self.backbone = X3D(**backbone_kwargs)
            feature_dim = self.backbone.feature_dim
        else:
            raise ValueError(
                f"Unknown backbone: {backbone_type}. "
                "Supported: 'spikeformer', 'x3d'"
            )
        self.consensus = ConsensusPooling(mode=consensus_mode)
        expanded_dim = feature_dim * head_expansion_factor
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, expanded_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, 1, bias=True),
        )
    
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        features = self.backbone(clips)  # (N, feature_dim)
        pooled = self.consensus(features)  # (feature_dim,)
        logit = self.classifier(pooled)  # (1,)
        return logit
    
    def get_clip_predictions(self, clips: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(clips)  # (N, feature_dim)
        logits = self.classifier(feats)  # (N, 1)
        return logits.squeeze(-1)  # (N,)

