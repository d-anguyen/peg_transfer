from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading configuration."""
    annotations_path: Path = Path("/mnt/cluster/datasets/lassdas/annotation/PegTransfer.csv")
    frames_dir: Path = Path("/mnt/cluster/datasets/lassdas/extracted_frames/PegTransfer/left")
    
    num_frames: int = 16  # Frames per clip
    sampling_rate: int = 5  # Use every Nth frame
    image_size: int = 224
    mean: tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: tuple[float, float, float] = (0.225, 0.225, 0.225)
    train_sample_ratio: float = 0.8
    test_sample_ratio: float = 1.1 
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = "x3d"  # "spikeformer" or "x3d"
    
    spikeformer_depths: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    spikeformer_dims: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    spikeformer_heads: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    detach_reset: bool = True
    
    x3d_model_name: str = "x3d_m"  # "x4d_xs", "x3d_s", "x3d_m", "x3d_l"
    x3d_pretrained: bool = True
    frozen_bn: bool = True
    
    consensus_pooling: str = "max"  # "max" or "avg"
    hidden_dim: int = 512
    expansion_factor: int = 2  # Head expansion factor (hidden_dim = feature_dim * expansion_factor)
    dropout: float = 0.5


@dataclass
class TrainConfig:
    """Training configuration."""
    # Optimization
    epochs: int = 200
    lr: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    use_pos_weight: bool = True
    log_every: int = 10
    eval_every: int = 1
    output_dir: Path = Path("./outputs/simple_multiclip")
    save_best: bool = True
    # Augmentation
    augmentation_probability: float = 0.9
    # LR schedule (cosine with optional warmup)
    use_lr_scheduler: bool = False
    warmup_ratio: float = 0.0
    warmup_start_lr: float = 3e-6
    cosine_end_lr: float = 1e-8


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    seed: int = 42
    device: str = "cuda"

