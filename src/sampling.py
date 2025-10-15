"""
Frame sampling strategies for pre-extracted video frames.
"""

from dataclasses import dataclass
import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass
class SamplingConfig:
    """Configuration for frame sampling strategies."""
    
    method: str = "uniform"
    num_frames: int = 16
    random_offset: bool = False
    
    # Top-K sampling parameters
    topk_step: int = 2 
    topk_metric: str = "l1" 
    
    def __post_init__(self):
        """Validate configuration."""
        if self.method not in ["uniform", "topk"]:
            raise ValueError(f"Unsupported method: {self.method}. Use 'uniform' or 'topk'.")
        if self.topk_metric not in ["l1", "l2", "ssim"]:
            raise ValueError(f"Unsupported metric: {self.topk_metric}. Use 'l1', 'l2', or 'ssim'.")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "num_frames": self.num_frames,
            "random_offset": self.random_offset,
            "topk_step": self.topk_step,
            "topk_metric": self.topk_metric,
        }
    
    @classmethod
    def from_dict(cls, config: dict) -> "SamplingConfig":
        """Create from dictionary."""
        return cls(
            method=config.get("sampling_method", config.get("method", "uniform")),
            num_frames=config.get("num_frames", 16),
            random_offset=config.get("random_offset", False),
            topk_step=config.get("topk_step", 2),
            topk_metric=config.get("topk_metric", "l1"),
        )


def compute_frame_difference(frame1: np.ndarray, frame2: np.ndarray, metric: str = "l2") -> float:
    """Calculate the difference between two frames using a specified metric."""
    if metric == "l2":
        diff = frame1 - frame2
        return float(np.sum(diff**2))
    
    elif metric == "l1":
        diff = frame1 - frame2
        return float(np.sum(np.abs(diff)))
    
    elif metric == "ssim":
        # Return 1 - SSIM so higher values mean more difference
        return 1.0 - ssim(frame1, frame2, data_range=1.0, channel_axis=-1)
    
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'l1', 'l2', or 'ssim'.")


class FrameSampler:
    """Frame sampler for pre-extracted frames.
    
    Frames have been extracted at a fixed FPS and saved as images.
    Supports uniform sampling, topk based on motion/diffs.
    """
    
    def __init__(self, config: SamplingConfig):
        self.config = config
    
    def sample_indices(
        self,
        total_frames: int,
        frame_differences: np.ndarray | None = None,
    ) -> list[int]:
        """Sample frame indices from pre-extracted frames."""
        if self.config.method == "uniform":
            return self._sample_uniform(total_frames)
        elif self.config.method == "topk":
            if frame_differences is None:
                raise ValueError("frame_differences required for topk sampling")
            return self._sample_topk(total_frames, frame_differences)
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.method}")
    
    def _sample_uniform(self, total_frames: int) -> list[int]:
        """Sample frames uniformly with optional random offset."""
        if total_frames <= self.config.num_frames:
            # If not enough frames, sample all and repeat last
            frame_indices = list(range(total_frames))
            frame_indices += [total_frames - 1] * (self.config.num_frames - total_frames)
            return frame_indices
        
        # Calculate the stride for uniform sampling
        stride = (total_frames - 1) / (self.config.num_frames - 1)
        
        if self.config.random_offset:
            # Random offset within the stride
            max_offset = max(1, int(stride))
            offset = np.random.randint(0, max_offset)
        else:
            # No offset - deterministic sampling
            offset = 0
        
        # Sample with offset and stride
        frame_indices = []
        for i in range(self.config.num_frames):
            idx = min(offset + int(i * stride), total_frames - 1)
            frame_indices.append(idx)
        
        return frame_indices
    
    def _sample_topk(
        self,
        total_frames: int,
        frame_differences: np.ndarray,
    ) -> list[int]:
        """Sample frames with highest motion based on pre-computed differences."""
        # Subsample frame differences based on topk_step
        step = self.config.topk_step
        candidate_indices = list(range(0, len(frame_differences), step))
        candidate_diffs = frame_differences[candidate_indices]
        
        if len(candidate_indices) <= self.config.num_frames:
            # Not enough candidates, return all
            selected = candidate_indices
        else:
            # Select top-k frames with highest differences
            topk_positions = np.argpartition(
                candidate_diffs,
                -self.config.num_frames
            )[-self.config.num_frames:]
            selected = [candidate_indices[pos] for pos in topk_positions]
        
        # Sort chronologically
        selected.sort()
        
        # Ensure we don't exceed available frames
        selected = [min(idx, total_frames - 1) for idx in selected]
        
        return selected
    
    @staticmethod
    def compute_frame_differences(
        frames: list[np.ndarray],
        metric: str = "l2",
    ) -> np.ndarray:
        """Compute frame-to-frame differences for a sequence."""
        if len(frames) < 2:
            return np.array([])
        
        differences = []
        for i in range(len(frames) - 1):
            diff = compute_frame_difference(frames[i], frames[i + 1], metric)
            differences.append(diff)
        
        return np.array(differences)

