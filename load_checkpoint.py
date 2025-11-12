from pathlib import Path

import torch
import torch.nn as nn
from pytorchvideo.models.x3d import create_x3d


x3d_config = {
    "width_factor": 2.0,         # Controls channel width scaling
    "depth_factor": 2.2,         # Controls network depth
    "bottleneck_factor": 2.25,   # Bottleneck expansion ratio
    "stem_dim_in": 12,           # Initial stem dimension (DIM_C1)
    "head_dim_out": 2048,        # Pre-pooling dimension (DIM_C5)
}
data_config = {
    "input_clip_length": 16,    # Number of frames per clip
    "input_crop_size": 224,     # Spatial crop size (height/width)
    "dropout_rate": 0.5,        # Dropout rate for backbone
}
head_config = {
    "expansion_factor": 2,      # hidden_dim = feature_dim * expansion_factor (2)
    "dropout_rate": 0.5,        # Dropout rate for head
}


def checkpoint_filter_fn(state_dict: dict) -> tuple[dict, dict]:
    backbone_state = {}
    head_state = {}
    
    for key, value in state_dict.items():
        key = key.replace("module.", "", 1) if key.startswith("module.") else key
        
        if key.startswith("backbone.model."):
            backbone_key = key.replace("backbone.model.", "", 1)
            backbone_state[backbone_key] = value
        elif key.startswith("head.head."):
            head_key = key.replace("head.head.", "head.", 1)
            head_state[head_key] = value
    
    return backbone_state, head_state


class X3DWithHead(nn.Module):
    """X3D backbone with classification head for drop detection."""
    
    def __init__(self, backbone: nn.Module, feature_dim: int, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.backbone = backbone
        # Head: Linear(no bias) -> GELU -> Dropout -> Linear(with bias)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) or (C, T, H, W)
        features = self.backbone(x)
        return self.head(features)


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cuda",
    load_head: bool = True,
) -> nn.Module:
    """
    Load pre-trained X3D checkpoint for drop detection.
    
    Example:
        >>> model = load_checkpoint("path/to/checkpoint.pth.tar", device="cuda")
        >>> video = torch.randn(1, 3, 16, 224, 224).cuda()
        >>> with torch.no_grad():
        ...     logit = model(video)  # (1, 1)
        ...     prob = torch.sigmoid(logit)  # Convert to probability
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_weights', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    # Filter and rename checkpoint keys
    backbone_state, head_state = checkpoint_filter_fn(state_dict)
    backbone = create_x3d(
        input_clip_length=data_config["input_clip_length"],
        input_crop_size=data_config["input_crop_size"],
        model_num_class=400,
        dropout_rate=data_config["dropout_rate"],
        width_factor=x3d_config["width_factor"],
        depth_factor=x3d_config["depth_factor"],
        bottleneck_factor=x3d_config["bottleneck_factor"],
        stem_dim_in=x3d_config["stem_dim_in"],
        head_dim_out=x3d_config["head_dim_out"],
        norm=nn.BatchNorm3d,
    )
    
    head = backbone.blocks[5]
    head.proj = nn.Identity()
    head.dropout = nn.Identity()
    head.pool.post_act = nn.Identity()
    head.pool.post_conv = nn.Identity()
    head.pool.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    # Load backbone weights
    # strict=True will trigger an error if any keys are missing / unexpected
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=True)
    
    if load_head and head_state:
        # Infer dimensions from head weights
        if "head.0.weight" in head_state:
            hidden_dim, feature_dim = head_state["head.0.weight"].shape
        else:
            # Fallback values if weights not found
            feature_dim = 432
            hidden_dim = feature_dim * head_config["expansion_factor"]
        
        model = X3DWithHead(
            backbone, 
            feature_dim, 
            hidden_dim, 
            dropout=head_config["dropout_rate"]
        )
        model.load_state_dict(head_state, strict=False)
    else:
        model = backbone
    
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    """Test loading the checkpoint."""
    checkpoint_path = "checkpoints/x3d-peg-transfer-error-detection.pth.tar"
    model = load_checkpoint(checkpoint_path, device="cpu", load_head=True)
    
    # Test forward pass
    dummy_video = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        logits = model(dummy_video)
        probs = torch.sigmoid(logits)
    
    print(f"Model loaded and tested successfully")
    print(f"Input shape: {dummy_video.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Prediction: {probs.item():.4f}")
