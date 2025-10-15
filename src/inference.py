"""
Inference script for video classification.
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from dataset import PegTransferDataset
from model import VideoSNN, VideoANN
from sampling import SamplingConfig


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float], list[str]]:
    """
    Run inference on a dataset.
    
    Returns:
        predictions: List of predicted class labels
        probabilities: List of probabilities for positive class
        video_ids: List of video IDs
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_video_ids = []
    
    pbar = tqdm(dataloader, desc="Inference")
    for batch in pbar:
        frames = batch["frames"].to(device)
        video_ids = batch["video_id"]
        
        # Forward pass
        logits = model(frames)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # Probability of positive class
        all_video_ids.extend(video_ids)
    
    return all_preds, all_probs, all_video_ids


def main():
    parser = argparse.ArgumentParser(description="Run inference on videos")
    parser.add_argument("--annotations", type=str, default="/mnt/cluster/datasets/lassdas/annotation/PegTransfer.csv")
    parser.add_argument("--frames_dir", type=str, default="/mnt/cluster/datasets/lassdas/extracted_frames/PegTransfer/left",
                        help="Directory containing pre-extracted frames")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to training config.json")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_random_offset", action="store_true",
                        help="Disable random offset (inference typically doesn't use it anyway)")
    parser.add_argument("--motion_cache_dir", type=str, default=None,
                        help="Directory for motion cache (for topk sampling)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create sampling configuration
    sampling_config = SamplingConfig.from_dict(config)
    
    # Override random_offset if flag is set
    if args.no_random_offset:
        sampling_config.random_offset = False
    
    # Create dataset
    dataset = PegTransferDataset(
        annotations_path=args.annotations,
        frames_dir=args.frames_dir,
        split=args.split,
        image_size=(config["image_size"], config["image_size"]),
        sampling_config=sampling_config,
        motion_cache_dir=args.motion_cache_dir,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    model_type = config["model_type"]
    if model_type == "snn":
        model = VideoSNN(
            num_classes=2,
            input_h=config["image_size"],
            input_w=config["image_size"],
        )
    else:  # ann
        model = VideoANN(
            num_classes=2,
            input_h=config["image_size"],
            input_w=config["image_size"],
        )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("Model loaded successfully")
    
    # Run inference
    predictions, probabilities, video_ids = predict(model, dataloader, device)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "video_id": video_ids,
        "predicted_label": predictions,
        "probability_dropped": probabilities,
    })
    
    # Add ground truth if available
    gt_df = dataset.data[["id", "object_dropped_within_fov"]].rename(
        columns={"id": "video_id", "object_dropped_within_fov": "ground_truth"}
    )
    results_df = results_df.merge(gt_df, on="video_id", how="left")
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    # Print summary
    if "ground_truth" in results_df.columns:
        accuracy = (results_df["predicted_label"] == results_df["ground_truth"].astype(int)).mean()
        print(f"\nAccuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
