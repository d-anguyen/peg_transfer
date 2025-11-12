import torch
import torch.nn as nn

from einops import rearrange
from pytorchvideo.models import hub

class X3D(nn.Module):
    """
    X3D backbone for video feature extraction.
    
    Loads pretrained X3D models from PyTorchVideo and converts them to feature extractors
    by replacing the classification head with global average pooling.
    
    Example:
        >>> model = X3D("x3d_m", pretrained=True)
        >>> features = model(video)  # (B, C, T, H, W) -> (B, feature_dim)
    
    Note: Requires PyTorchVideo to be installed:
        pip install pytorchvideo
    """
    
    def __init__(
        self,
        model_name: str = "x3d_m",
        pretrained: bool = True,
        checkpoint_path: str | None = None,
    ):
        """
        Args:
            model_name: X3D variant ('x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l')
            pretrained: Load pretrained Kinetics-400 weights from hub
            checkpoint_path: Path to custom checkpoint (optional, overrides pretrained)
        """
        super().__init__()
        
        # Load model from hub
        hub_models = {
            "x3d_xs": hub.x3d_xs,
            "x3d_s": hub.x3d_s,
            "x3d_m": hub.x3d_m,
            "x3d_l": hub.x3d_l,
        }
        
        if model_name not in hub_models:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {list(hub_models.keys())}")
        
        self.model = hub_models[model_name](pretrained=pretrained)
        # Load custom checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        # Convert classification head to feature extractor
        self._convert_to_feature_extractor()
        # Store feature dimension
        self.feature_dim = self.model.blocks[5].pool.pre_conv.out_channels

    def _convert_to_feature_extractor(self) -> None:
        """Remove classification head and replace with global average pooling."""
        # Remove final activation if present
        if hasattr(self.model, "post_act"):
            del self.model.post_act
        
        # Replace head components with identity/pooling
        head = self.model.blocks[5]
        head.proj = nn.Identity()
        head.dropout = nn.Identity()
        head.pool.post_act = nn.Identity()
        head.pool.post_conv = nn.Identity()
        head.pool.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load weights from checkpoint file (supports various formats)."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract state dict from various checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint
        
        # Remove DDP "module." prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
        # Load (non-strict to allow head mismatches)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video clips.
        Args:
            x: Video tensor of shape (B, C, T, H, W) or (C, T, H, W)
        Returns:
            features: Feature tensor of shape (B, feature_dim) or (feature_dim,)
        """
        # Handle single clip input
        if x.ndim == 4:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Extract features
        features = self.model(x)
        features = rearrange(features, 'b f ... -> b (f ...)')

        if squeeze_output:
            features = features.squeeze(0)

        return features


def x3d_xs(pretrained: bool = True) -> X3D:
    """X3D-XS: Extra small variant (3.8M params)."""
    return X3D(model_name="x3d_xs", pretrained=pretrained)

def x3d_s(pretrained: bool = True) -> X3D:
    """X3D-S: Small variant (3.8M params)."""
    return X3D(model_name="x3d_s", pretrained=pretrained)

def x3d_m(pretrained: bool = True) -> X3D:
    """X3D-M: Medium variant (3.8M params)."""
    return X3D(model_name="x3d_m", pretrained=pretrained)

def x3d_l(pretrained: bool = True) -> X3D:
    """X3D-L: Large variant (6.2M params)."""
    return X3D(model_name="x3d_l", pretrained=pretrained)
