# /// script
# dependencies = ["opencv-python", "pandas", "tqdm"]
# ///

"""
Preprocess PegTransfer videos by extracting frames at a fixed FPS.

This script extracts frames from all videos in the dataset at a specified FPS
(default: 2.0) and saves them as individual JPEG images.

Usage:
    python preprocess_videos.py --fps 2.0 --quality 95
"""

import cv2
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    image_quality: int = 95,
    skip_existing: bool = True,
) -> dict[str, int | float]:
    """
    Extract frames from a video at specified FPS.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1.0)
        image_quality: JPEG quality (0-100, default: 95)
        skip_existing: Skip if output directory already exists and has frames
        
    Returns:
        Dictionary with extraction statistics
    """
    # Check if already extracted
    if skip_existing and output_dir.exists():
        existing_frames = list(output_dir.glob("frame_*.jpg"))
        if len(existing_frames) > 0:
            return {
                "status": "skipped",
                "frames_extracted": len(existing_frames),
                "video_fps": 0,
                "video_duration": 0,
            }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "status": "error",
            "error": "Cannot open video",
            "frames_extracted": 0,
        }
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    # Calculate frame interval
    if video_fps == 0:
        cap.release()
        return {
            "status": "error",
            "error": "Invalid FPS",
            "frames_extracted": 0,
        }
    
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1
    
    # Extract frames
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Calculate timestamp
            timestamp_sec = frame_count / video_fps
            
            # Save frame
            frame_filename = f"frame_{extracted_count:04d}_t{timestamp_sec:.2f}s.jpg"
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, image_quality]
            )
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    
    return {
        "status": "success",
        "frames_extracted": extracted_count,
        "video_fps": video_fps,
        "video_duration": duration,
        "total_frames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from PegTransfer videos at specified FPS"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="/mnt/cluster/datasets/lassdas/annotation/PegTransfer.csv",
        help="Path to annotations CSV file"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/mnt/cluster/datasets/lassdas/videos/PegTransfer/left",
        help="Directory containing video files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/cluster/datasets/lassdas/extracted_frames/PegTransfer/left/",
        help="Directory to save extracted frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="[1-100]",
        help="JPEG quality (1-100, default: 95)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Process only specific split (default: all)"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (for testing)"
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-extract frames even if they already exist"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without actually extracting frames"
    )
    
    args = parser.parse_args()
    annotations_path = Path(args.annotations)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    print("Loading annotations...")
    df = pd.read_csv(annotations_path)
    
    if args.split:
        df = df[df["data_split"] == args.split].reset_index(drop=True)
        print(f"Processing {args.split} split only")
    
    if args.max_videos:
        df = df.head(args.max_videos)
        print(f"Limiting to first {args.max_videos} videos")
    
    print(f"Total videos to process: {len(df)}")
    print(f"Extraction FPS: {args.fps}")
    print(f"JPEG quality: {args.quality}")
    print(f"Output directory: {output_dir}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "total_frames": 0,
    }
    
    errors = []
    
    for idx in tqdm(range(len(df)), desc="Extracting frames"):
        row = df.iloc[idx]
        video_id = row["id"]
        video_path = video_dir / f"{video_id}.mkv"
        video_output_dir = output_dir / video_id
        
        if not video_path.exists():
            stats["error"] += 1
            errors.append(f"{video_id}: Video file not found")
            continue
        
        result = extract_frames_from_video(
            video_path=video_path,
            output_dir=video_output_dir,
            fps=args.fps,
            image_quality=args.quality,
            skip_existing=not args.no_skip_existing,
        )
        
        status = result.get("status", "unknown")
        if status == "success":
            stats["success"] += 1
            stats["total_frames"] += result["frames_extracted"]
        elif status == "skipped":
            stats["skipped"] += 1
            stats["total_frames"] += result["frames_extracted"]
        elif status == "error":
            stats["error"] += 1
            errors.append(f"{video_id}: {result.get('error', 'Unknown error')}")
    
    print(f"Videos processed: {len(df)}")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped (already extracted): {stats['skipped']}")
    print(f"  Errors: {stats['error']}")
    print(f"Total frames extracted: {stats['total_frames']}")
    
    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    if stats["total_frames"] > 0:
        avg_frame_size_kb = 125
        estimated_size_mb = (stats["total_frames"] * avg_frame_size_kb) / 1024
        estimated_size_gb = estimated_size_mb / 1024
        print(f"\nEstimated storage used: ~{estimated_size_gb:.2f} GB")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

