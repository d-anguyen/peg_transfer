#!/usr/bin/env python3
import argparse
from pathlib import Path
from .config import Config
from .train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Simple multi-clip training")
    
    # Data
    parser.add_argument('--annotations', type=Path, help="Path to annotations CSV")
    parser.add_argument('--frames_dir', type=Path, help="Path to frames directory")
    
    # Model
    parser.add_argument('--backbone', type=str, default="x3d", 
                       choices=['spikeformer', 'x3d'], help="Backbone type")
    parser.add_argument('--pooling', type=str, default="max",
                       choices=['max', 'avg'], help="Consensus pooling mode")
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--output_dir', type=Path, default=Path("./outputs/peg-transfer"),
                       help="Output directory")
    
    # Data parameters
    parser.add_argument('--num_frames', type=int, default=16, help="Frames per clip")
    parser.add_argument('--sampling_rate', type=int, default=5, help="Frame sampling rate")
    parser.add_argument('--sample_ratio', type=float, default=0.75, 
                       help="Fraction of video to cover")
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--num_workers', type=int, default=8, help="Data loading workers")
    parser.add_argument('--image_size', type=int, default=224, help="Image size")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config
    config = Config()
    
    # Override with command-line args
    if args.annotations:
        config.data.annotations_path = args.annotations
    if args.frames_dir:
        config.data.frames_dir = args.frames_dir
    
    config.model.backbone = args.backbone
    config.model.consensus_pooling = args.pooling
    
    config.train.epochs = args.epochs
    config.train.lr = args.lr
    # Ensure Path type even if argparse/environment provides a string
    config.train.output_dir = Path(args.output_dir)
    
    config.data.num_frames = args.num_frames
    config.data.sampling_rate = args.sampling_rate
    config.data.train_sample_ratio = args.sample_ratio
    config.data.image_size = args.image_size
    
    config.seed = args.seed
    config.data.num_workers = args.num_workers
    
    # Print configuration
    print("=" * 60)
    print("SIMPLE MULTI-CLIP TRAINING")
    print("=" * 60)
    print(f"Data:")
    print(f"  Annotations: {config.data.annotations_path}")
    print(f"  Frames dir: {config.data.frames_dir}")
    print(f"Model:")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Pooling: {config.model.consensus_pooling}")
    print(f"Training:")
    print(f"  Epochs: {config.train.epochs}")
    print(f"  Learning rate: {config.train.lr}")
    print(f"  Output: {config.train.output_dir}")
    print(f"Sampling:")
    print(f"  Frames/clip: {config.data.num_frames}")
    print(f"  Sampling rate: {config.data.sampling_rate}")
    print(f"  Sample ratio: {config.data.train_sample_ratio}")
    print("=" * 60)
    print()
    
    # Train
    train(config)


if __name__ == '__main__':
    main()

