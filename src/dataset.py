"""
Dataset class for PegTransfer video classification.

Works with pre-extracted frames from videos, supporting flexible sampling strategies.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from functools import lru_cache

from sampling import SamplingConfig, FrameSampler


class PegTransferDataset(Dataset):
    
    def __init__(
        self,
        annotations_path: str | Path,
        frames_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (224, 224),
        sampling_config: SamplingConfig | None = None,
        motion_cache_dir: str | Path | None = None,
    ):
        """
        Args:
            annotations_path: Path to PegTransfer.csv
            frames_dir: Path to directory containing extracted frames
            split: 'train', 'val', or 'test'
            image_size: Target size for frames (H, W)
            sampling_config: Sampling configuration (defaults to uniform sampling with 16 frames)
            motion_cache_dir: Directory to cache frame differences for topk sampling (None disables)
        """
        self.frames_dir = Path(frames_dir)
        self.image_size = image_size
        self.split = split
        
        # Setup sampling
        if sampling_config is None:
            # No config provided - use defaults with auto-enable for training
            self.sampling_config = SamplingConfig()
            if split == "train" and self.sampling_config.method == "uniform":
                self.sampling_config.random_offset = True
        else:
            # Config provided - respect the user's settings
            self.sampling_config = sampling_config
        
        self.sampler = FrameSampler(self.sampling_config)
        
        # Setup motion cache for topk sampling
        self.motion_cache_dir = Path(motion_cache_dir) if motion_cache_dir is not None else None
        if self.motion_cache_dir is not None and self.sampling_config.method == "topk":
            self.motion_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Motion cache enabled at: {self.motion_cache_dir}")
        
        # Load annotations
        df = pd.read_csv(annotations_path)
        self.data = df[df["data_split"] == split].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} {split} samples")
        print(f"Frames directory: {self.frames_dir}")
        print(f"Sampling: {self.sampling_config.method}, {self.sampling_config.num_frames} frames")
        if self.sampling_config.method == "uniform":
            print(f"  Random offset: {self.sampling_config.random_offset}")
        elif self.sampling_config.method == "topk":
            print(f"  Top-k step: {self.sampling_config.topk_step}, metric: {self.sampling_config.topk_metric}")
        print(f"Label distribution: {self.data['object_dropped_within_fov'].value_counts().to_dict()}")
        
        # Cache frame paths for all videos (avoid repeated glob calls)
        print("Caching frame paths...")
        self._frame_paths_cache = {}
        for idx in tqdm(range(len(self.data)), desc="Loading paths"):
            video_id = self.data.iloc[idx]["id"]
            self._frame_paths_cache[video_id] = self._load_frame_paths(video_id)
        
        # Pre-compute and cache frame differences for topk sampling
        if self.sampling_config.method == "topk" and self.motion_cache_dir is not None:
            self._precompute_frame_differences()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _get_video_frame_dir(self, video_id: str) -> Path:
        """Get the directory containing extracted frames for a video."""
        return self.frames_dir / video_id
    
    @lru_cache(maxsize=1000)
    def _load_frame_paths(self, video_id: str) -> list[Path]:
        """
        Load paths to all extracted frames for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of frame paths sorted by frame number
        """
        frame_dir = self._get_video_frame_dir(video_id)
        
        if not frame_dir.exists():
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
        
        frame_paths = sorted(frame_dir.glob("frame_*.jpg")) # io op, therefore cached
        
        if len(frame_paths) == 0:
            raise ValueError(f"No frames found in {frame_dir}")
        
        return frame_paths
    
    def _get_motion_cache_path(self, video_id: str) -> Path:
        """Get the cache file path for frame differences."""
        metric = self.sampling_config.topk_metric
        return self.motion_cache_dir / f"{video_id}_motion_{metric}.npy"
    
    def _load_motion_cache(self, video_id: str) -> np.ndarray | None:
        """Load cached frame differences."""
        if self.motion_cache_dir is None:
            return None
        
        cache_path = self._get_motion_cache_path(video_id)
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def _save_motion_cache(self, video_id: str, differences: np.ndarray) -> None:
        """Save frame differences to cache."""
        if self.motion_cache_dir is None:
            return
        
        cache_path = self._get_motion_cache_path(video_id)
        np.save(cache_path, differences)
    
    def _compute_frame_differences(self, video_id: str) -> np.ndarray:
        """
        Compute frame-to-frame differences for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Array of frame differences
        """
        # Try to load from cache first
        cached_diffs = self._load_motion_cache(video_id)
        if cached_diffs is not None:
            return cached_diffs
        
        # Load all frames
        frame_paths = self._load_frame_paths(video_id)
        
        # Load and preprocess frames for difference computation
        frames = []
        for path in frame_paths:
            frame = cv2.imread(str(path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        # Compute differences
        differences = FrameSampler.compute_frame_differences(
            frames,
            metric=self.sampling_config.topk_metric
        )
        
        # Save to cache
        self._save_motion_cache(video_id, differences)
        
        return differences
    
    def _precompute_frame_differences(self) -> None:
        """Pre-compute and cache frame differences for all videos."""
        print(f"\nPre-computing frame differences for {len(self.data)} videos...")
        cached_count = 0
        computed_count = 0
        
        for idx in tqdm(range(len(self.data)), desc="Computing motion"):
            video_id = self.data.iloc[idx]["id"]
            
            # Check if already cached
            if self._load_motion_cache(video_id) is not None:
                cached_count += 1
                continue
            try:
                self._compute_frame_differences(video_id)
                computed_count += 1
            except Exception as e:
                print(f"\nWarning: Failed to compute differences for {video_id}: {e}")
        
        print(f"Motion computation complete: {cached_count} already cached, {computed_count} newly computed")
        print(f"Cache location: {self.motion_cache_dir}\n")
    
    def _load_frames_by_indices(
        self,
        frame_paths: list[Path],
        indices: list[int]
    ) -> np.ndarray:
        """
        Load specific frames by their indices.
        
        Args:
            frame_paths: List of all frame paths
            indices: Indices of frames to load
            
        Returns:
            Array of frames with shape (num_frames, H, W, 3)
        """
        frames = []
        
        for idx in indices:
            # Clamp index to valid range
            idx = min(idx, len(frame_paths) - 1)
            
            # Load frame
            frame = cv2.imread(str(frame_paths[idx]))
            if frame is None:
                # If frame loading fails, use last valid frame or black frame
                if frames:
                    frame = frames[-1]
                else:
                    frame = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            
            frames.append(frame)
        
        return np.stack(frames)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get video frames and label.
        
        Returns:
            Dictionary with:
                - 'frames': Tensor of shape (num_frames, 3, H, W)
                - 'label': Binary label (0 or 1)
                - 'video_id': Video ID string
        """
        row = self.data.iloc[idx]
        video_id = row["id"]
        
        # Get cached frame paths (no more repeated glob calls!)
        frame_paths = self._frame_paths_cache[video_id]
        total_frames = len(frame_paths)
        
        # Sample frame indices
        if self.sampling_config.method == "uniform":
            frame_indices = self.sampler.sample_indices(total_frames)
        elif self.sampling_config.method == "topk":
            # Load or compute frame differences
            frame_differences = self._compute_frame_differences(video_id)
            frame_indices = self.sampler.sample_indices(total_frames, frame_differences)
        else:
            raise ValueError(f"Unsupported sampling method: {self.sampling_config.method}")
        
        # Load selected frames
        frames = self._load_frames_by_indices(frame_paths, frame_indices)
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor and change to (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        # Get label
        label = int(row["object_dropped_within_fov"])
        
        return {
            "frames": frames,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": video_id,
        }
