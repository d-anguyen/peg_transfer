"""
Training script for video classification.
"""

import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import wandb

from dataset import PegTransferDataset
from model import VideoSNN, VideoANN
from sampling import SamplingConfig


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_wandb: bool = False,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(frames)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})
        
        # Log to wandb (per batch)
        if log_wandb:
            wandb.log({
                "train/batch_loss": loss.item(),
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        logits = model(frames)
        loss = criterion(logits, labels)
        
        # Track metrics
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Compute AUC if we have both classes
    auc = None
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "all_preds": all_preds,  # Return for wandb plotting
        "all_labels": all_labels,  # Return for wandb plotting
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train video classification model")
    parser.add_argument("--annotations", type=str, default="/mnt/cluster/datasets/lassdas/annotation/PegTransfer.csv")
    parser.add_argument("--frames_dir", type=str, default="/mnt/cluster/datasets/lassdas/extracted_frames/PegTransfer/left",
                        help="Directory containing pre-extracted frames")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample per video")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--model_type", type=str, default="ann", 
                        choices=["snn", "ann"],
                        help="Model architecture: snn (spiking) or ann (artificial neural network)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # Frame sampling parameters
    parser.add_argument("--sampling_method", type=str, default="uniform", choices=["uniform", "topk"],
                        help="Frame sampling method")
    parser.add_argument("--no_random_offset", action="store_true",
                        help="Disable random offset for uniform sampling (training only, enabled by default)")
    parser.add_argument("--topk_step", type=int, default=2,
                        help="Step size for topk sampling (process every Nth frame)")
    parser.add_argument("--topk_metric", type=str, default="l1", choices=["l1", "l2", "ssim"],
                        help="Metric for computing frame differences in topk sampling")
    # Motion caching parameters (for topk sampling)
    parser.add_argument("--cache_motion", action="store_true",
                        help="Enable caching of frame differences for topk sampling")
    parser.add_argument("--motion_cache_dir", type=str, default=None,
                        help="Directory for frame cache (defaults to <output_dir>/motion_cache)")
    # Weights & Biases parameters
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="peg-transfer",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Weights & Biases run name (auto-generated if not provided)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (username or team name)")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Make deterministic 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create sampling configuration
    sampling_config = SamplingConfig(
        method=args.sampling_method,
        num_frames=args.num_frames,
        random_offset=not args.no_random_offset, 
        topk_step=args.topk_step,
        topk_metric=args.topk_metric,
    )
    
    # Initialize Weights & Biases
    if args.use_wandb:
        # Auto-generate run name if not provided
        run_name = args.wandb_name or f"{args.model_type}_{args.sampling_method}"
        
        # Add offset suffix only for uniform sampling
        if sampling_config.method == "uniform":
            if sampling_config.random_offset:
                run_name += "_offset"
            else:
                run_name += "_no_offset"
        # For top-k, optionally add metric info
        elif sampling_config.method == "topk":
            run_name += f"_{sampling_config.topk_metric}"
        
        # Prepare config with sampling details for easy filtering
        config = vars(args).copy()
        config['random_offset'] = sampling_config.random_offset
        config['sampling_config'] = sampling_config.to_dict()
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config=config,
            dir=str(output_dir),
        )
        print(f"Initialized wandb run: {wandb.run.name}")
    
    # Save config
    config_to_save = vars(args).copy()
    config_to_save['sampling_config'] = sampling_config.to_dict()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)
    
    # Setup motion caching for topk sampling
    motion_cache_dir = None
    if args.cache_motion and args.sampling_method == "topk":
        if args.motion_cache_dir is not None:
            motion_cache_dir = Path(args.motion_cache_dir)
        else:
            motion_cache_dir = output_dir / "motion_cache"
        print(f"Motion caching enabled: {motion_cache_dir}")
    
    # Create datasets
    train_dataset = PegTransferDataset(
        annotations_path=args.annotations,
        frames_dir=args.frames_dir,
        split="train",
        image_size=(args.image_size, args.image_size),
        sampling_config=sampling_config,
        motion_cache_dir=motion_cache_dir,
    )
    
    test_dataset = PegTransferDataset(
        annotations_path=args.annotations,
        frames_dir=args.frames_dir,
        split="test",
        image_size=(args.image_size, args.image_size),
        sampling_config=sampling_config,
        motion_cache_dir=motion_cache_dir,
    )
    
    # Worker init function for deterministic dataloading
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    
    # Create model
    if args.model_type == "snn":
        model = VideoSNN(
            num_classes=2,
            input_h=args.image_size,
            input_w=args.image_size,
        )
    else:  # ann
        model = VideoANN(
            num_classes=2,
            input_h=args.image_size,
            input_w=args.image_size,
        )
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_type}")
    print(f"Total parameters: {total_params:,}")
    
    # Log model to wandb
    if args.use_wandb:
        wandb.config.update({"total_parameters": total_params})
        wandb.watch(model, log="all", log_freq=100)
    
    # Loss function with class weights for imbalanced data
    # Calculate class weights
    train_labels = train_dataset.data["object_dropped_within_fov"].values
    pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Optimizer (constant learning rate)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    best_f1 = 0.0
    results = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, log_wandb=args.use_wandb
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Acc: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        if test_metrics['auc'] is not None:
            print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")
        
        current_lr = args.lr
        
        # Log to wandb
        if args.use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "test/loss": test_metrics['loss'],
                "test/accuracy": test_metrics['accuracy'],
                "test/precision": test_metrics['precision'],
                "test/recall": test_metrics['recall'],
                "test/f1": test_metrics['f1'],
                "learning_rate": current_lr,
            }
            if test_metrics['auc'] is not None:
                log_dict["test/auc"] = test_metrics['auc']
            
            # Log confusion matrix using wandb's native support
            log_dict["test/confusion_matrix"] = wandb.plot.confusion_matrix(
                preds=test_metrics['all_preds'],
                y_true=test_metrics['all_labels'],
                class_names=["No Drop", "Drop"],
            )
            
            wandb.log(log_dict)
        
        # Save results
        results.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_metrics": test_metrics,
        })
        
        # Save best model
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_metrics": test_metrics,
            }, output_dir / "best_model.pt")
            print(f"Saved best model with F1: {best_f1:.4f}")
            
            # Log best model to wandb
            if args.use_wandb:
                wandb.run.summary["best_f1"] = best_f1
                wandb.run.summary["best_epoch"] = epoch + 1
        
        print(f"Learning rate: {current_lr:.6f}")
    
    # Save final results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    
    if args.use_wandb:
        wandb.finish()
    
    print("\nTraining completed!")
    print(f"Best F1 score: {best_f1:.4f}")


if __name__ == "__main__":
    main()
