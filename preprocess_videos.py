"""
Helper script to extract all frames.

Usage:
    uv run preprocess_videos.py --video_dir /path/to/videos --output_dir /path/to/frames
"""

import cv2
from pathlib import Path
import argparse
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path):
    """Extract all frames from a video."""
    video_id = video_path.stem 
    frames_dir = output_dir / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
    cap.release()
    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory with video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for frames")
    parser.add_argument("--extension", type=str, default=".mkv", help="Video file extension")
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted(video_dir.rglob(f"*{args.extension}"))
    print(f"Found {len(video_files)} videos")
    
    for video_path in tqdm(video_files, desc="Extracting frames"):
        num_frames = extract_frames(video_path, output_dir)
        print(f"  {video_path.name}: {num_frames} frames")
    
    print(f"\nDone! Frames saved to: {output_dir}")


if __name__ == "__main__":
    main()

