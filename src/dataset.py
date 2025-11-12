import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from einops import rearrange


class MultiClipDataset(Dataset):
    """
    Dataset that loads multiple clips from each video.
    Each sample returns all clips from a single video.
    """
    
    def __init__(
        self,
        annotations_path: Path,
        frames_dir: Path,
        split: str,
        num_frames: int = 16,
        sampling_rate: int = 5,
        image_size: int = 224,
        sample_ratio: float = 0.75,
        augment: bool = False,
        augment_prob: float = 1.0,
        mean: tuple = (0.45, 0.45, 0.45),
        std: tuple = (0.225, 0.225, 0.225),
    ):
        """
        Args:
            annotations_path: Path to CSV with columns [id, data_split, object_dropped_within_fov]
            frames_dir: Directory with frames organized as frames_dir/video_id/frame_000000.jpg
            split: 'train', 'val', or 'test'
            num_frames: Number of frames per clip
            sampling_rate: Sample every Nth frame
            image_size: Target image size
            sample_ratio: Fraction of video to cover with clips
            augment: Whether to apply data augmentation
            mean: Normalization mean
            std: Normalization std
        """
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.sample_ratio = sample_ratio
        self.augment_prob = augment_prob
        
        # Load annotations
        df = pd.read_csv(annotations_path)
        self.data = df[df['data_split'] == split].reset_index(drop=True)
        
        # Build transforms
        self.transform = self._build_transform(image_size, augment, mean, std)
        
        print(f"Loaded {len(self.data)} videos for {split}")
        print(f"  Error distribution: {self.data['object_dropped_within_fov'].value_counts().to_dict()}")
    
    def _build_transform(self, size: int, augment: bool, mean: tuple, std: tuple):
        """Build image transforms."""
        if augment:
            # Training augmentations with probability knob: augment block applied with p=self.augment_prob,
            # while resize/crop/normalize/tensor always applied
            return A.Compose([
                A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(
                        brightness=(0.8, 1.2),
                        contrast=(0.8, 1.2),
                        saturation=(0.8, 1.2),
                        hue=(-0.5, 0.5),
                        p=0.9,
                    ),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True, p=0.75),
                    A.ShiftScaleRotate(
                        shift_limit_x=0.1,
                        shift_limit_y=0.01,
                        scale_limit=0.1,
                        rotate_limit=7.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.25,
                    ),
                    A.Affine(
                        scale=0.9,
                        interpolation=cv2.INTER_CUBIC,
                        cval=0,
                        mode=cv2.BORDER_CONSTANT,
                        keep_ratio=True,
                        p=0.25,
                    ),
                    A.ColorJitter(
                        brightness=(0.7, 1.2),
                        contrast=(0.8, 1.2),
                        saturation=(0.8, 1.2),
                        hue=(-0.5, 0.5),
                        p=0.9,
                    ),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True, p=0.75),
                    A.Affine(
                        scale=(0.95, 1.05),
                        translate_px={'x': (-15, 15), 'y': (-15, 0)},
                        rotate=(0, 5),
                        p=0.8,
                    ),
                ], p=float(self.augment_prob)),
                A.SmallestMaxSize(256, p=1.0),
                A.CenterCrop(size, size, p=1.0),
                A.Normalize(mean=mean, std=std, p=1.0),
                ToTensorV2(),
            ])
        else:
            # Test transforms (no augmentation)
            return A.Compose([
                A.Affine(scale=0.9, p=1.0),
                A.SmallestMaxSize(256, p=1.0),
                A.CenterCrop(size, size, p=1.0),
                A.Normalize(mean=mean, std=std, p=1.0),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
                - clips: Tensor of shape (num_clips, C, T, H, W)
                - label: Binary label (0 or 1)
                - video_id: Video identifier
        """
        row = self.data.iloc[idx]
        video_id = row['id']
        label = int(row['object_dropped_within_fov'])
        
        # Get all frame paths for this video
        video_dir = self.frames_dir / video_id
        frame_paths = sorted(video_dir.glob('frame_*.jpg'))
        
        # Sample frame indices for each clip
        total_frames = len(frame_paths) // self.sampling_rate
        clip_indices = self._sample_clip_indices(total_frames)
        
        # Load clips
        clips = []
        for start_idx in clip_indices:
            clip = self._load_clip(frame_paths, start_idx)
            clips.append(clip)
        
        # Stack clips: (num_clips, C, T, H, W)
        clips = torch.stack(clips, dim=0)
        
        return {
            'clips': clips,
            'label': torch.tensor(label, dtype=torch.long),
            'video_id': video_id,
        }
    
    def _sample_clip_indices(self, total_frames: int) -> list[int]:
        """Sample start indices for clips to cover sample_ratio of video."""
        # Calculate number of clips needed
        num_clips = max(1, int(np.ceil(self.sample_ratio * total_frames / self.num_frames)))
        
        # Uniformly sample start indices
        if num_clips == 1:
            return [total_frames // 2]  # Middle of video
        
        # Evenly spaced clips
        segment_length = total_frames / num_clips
        indices = [int(segment_length * i) for i in range(num_clips)]
        
        # Ensure indices are valid
        max_start = max(0, total_frames - self.num_frames)
        indices = [min(idx, max_start) for idx in indices]
        
        return indices
    
    def _load_clip(self, frame_paths: list[Path], start_idx: int) -> torch.Tensor:
        """
        Load a single clip (sequence of frames).
        
        Returns:
            Tensor of shape (C, T, H, W)
        """
        frames = []
        
        for i in range(self.num_frames):
            # Calculate frame index with sampling rate
            frame_idx = (start_idx + i) * self.sampling_rate
            frame_idx = min(frame_idx, len(frame_paths) - 1)  # Clamp to valid range
            
            # Load and transform frame
            frame = cv2.imread(str(frame_paths[frame_idx]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            transformed = self.transform(image=frame)
            frame_tensor = transformed['image']  # (C, H, W)
            
            frames.append(frame_tensor)
        
        # Stack into clip: (T, C, H, W) -> (C, T, H, W)
        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        clip = rearrange(clip, 't c h w -> c t h w')
        
        return clip
    
    def get_pos_weight(self) -> float:
        """Calculate positive class weight for loss balancing."""
        labels = self.data['object_dropped_within_fov'].values
        neg = (labels == 0).sum()
        pos = (labels == 1).sum()
        return neg / pos if pos > 0 else 1.0


def collate_multiclip(batch: list[dict]) -> dict:
    """
    Collate function that handles variable numbers of clips per video.
    
    Since each video can have different number of clips, we process videos
    one at a time during training (batch_size=1 for videos, but each video
    has multiple clips).
    """
    # For simplicity, we just return the first (and only) video in the batch
    # In practice, batch_size should be 1 for videos
    assert len(batch) == 1, "Use batch_size=1 for videos (each video has multiple clips)"
    return batch[0]
