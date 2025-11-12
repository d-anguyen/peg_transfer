import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from dataclasses import asdict, is_dataclass

from .config import Config
from .dataset import MultiClipDataset, collate_multiclip
from .model import MultiClipClassifier
from spikingjelly.clock_driven import functional


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_to_dict(config: Config) -> dict:
    """Convert Config dataclass to dict, handling nested dataclasses and Path objects."""
    def convert_value(value):
        if isinstance(value, Path):
            return str(value)
        elif is_dataclass(value):
            return asdict(value)
        elif isinstance(value, (list, tuple)):
            return [convert_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        else:
            return value
    
    result = {}
    for field_name, field_value in config.__dict__.items():
        result[field_name] = convert_value(field_value)
    return result


def compute_metrics(predictions: list, labels: list) -> dict:
    """Compute classification metrics."""
    preds = np.array(predictions)
    labs = np.array(labels)
    
    # Binary classification metrics
    tp = ((preds == 1) & (labs == 1)).sum()
    tn = ((preds == 0) & (labs == 0)).sum()
    fp = ((preds == 1) & (labs == 0)).sum()
    fn = ((preds == 0) & (labs == 1)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Matthews Correlation Coefficient
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator > 0 else 0
    mcc_normed = (mcc + 1) / 2  # Scale to [0, 1]
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_acc': (sensitivity + specificity) / 2,
        'mcc': mcc,
        'mcc_normed': mcc_normed,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | CosineAnnealingLR | SequentialLR | None,
    device: str,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    predictions = []
    labels = []
    logits = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        clips = batch['clips'].to(device)  # (num_clips, C, T, H, W)
        label = batch['label'].to(device)  # (,)
        
        # Forward pass
        optimizer.zero_grad()
        logit = model(clips)  # Shape (1,)
        # Ensure logit and label are both 1D tensors for loss
        if logit.ndim == 0:
            logit = logit.unsqueeze(0)
        loss = criterion(logit, label.float().unsqueeze(0))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Reset spiking neurons (if using SpikeFormer)
        functional.reset_net(model)
        
        # Track metrics (threshold at 0.5 for binary classification)
        total_loss += loss.item()
        # logit is (1,), extract scalar for prediction
        logit_scalar = logit.squeeze().item() if logit.ndim > 0 else logit.item()
        logits.append(logit_scalar)
        pred = (logit_scalar > 0.0)
        predictions.append(pred)
        labels.append(label.item())
        
        pbar.set_postfix({'loss': loss.item()})
    
    metrics = compute_metrics(predictions, labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    # Compute logit statistics
    logits_array = np.array(logits)
    metrics['logit_mean'] = float(np.mean(logits_array))
    metrics['logit_std'] = float(np.std(logits_array))
    metrics['frac_predicted_positive'] = float(np.mean(predictions))
    
    return metrics


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> dict:
    """Evaluate for one epoch."""
    model.eval()
    
    total_loss = 0
    predictions = []
    labels = []
    logits = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Eval]")
    for batch in pbar:
        clips = batch['clips'].to(device).float()
        label = batch['label'].to(device)
        
        # Forward pass
        logit = model(clips).float()  # Shape (1,)
        # Ensure logit and label are both 1D tensors for loss
        if logit.ndim == 0:
            logit = logit.unsqueeze(0)
        loss = criterion(logit, label.float().unsqueeze(0))
        
        # Track metrics (threshold at 0.5 for binary classification)
        total_loss += loss.item()
        # logit is (1,), extract scalar for prediction
        logit_scalar = logit.squeeze().item() if logit.ndim > 0 else logit.item()
        logits.append(logit_scalar)
        pred = (logit_scalar > 0.0)
        predictions.append(pred)
        labels.append(label.item())
    
    metrics = compute_metrics(predictions, labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    # Compute logit statistics
    logits_array = np.array(logits)
    metrics['logit_mean'] = float(np.mean(logits_array))
    metrics['logit_std'] = float(np.std(logits_array))
    metrics['frac_predicted_positive'] = float(np.mean(predictions))
    
    return metrics


def train(config: Config):
    """Main training function."""
    
    # Set seed
    set_seed(config.seed)
    
    # Create output directory
    config.train.output_dir.mkdir(parents=True, exist_ok=True)
    
    ## Save config (non-mutating, serialize Paths via default=str)
    #with open(config.train.output_dir / 'config.json', 'w') as f:
        #json.dump(config_to_dict(config), f, indent=2)
    
    print("Configuration:")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Epochs: {config.train.epochs}")
    print(f"  Learning rate: {config.train.lr}")
    print(f"  Output: {config.train.output_dir}")
    print()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MultiClipDataset(
        annotations_path=config.data.annotations_path,
        frames_dir=config.data.frames_dir,
        split='train',
        num_frames=config.data.num_frames,
        sampling_rate=config.data.sampling_rate,
        image_size=config.data.image_size,
        sample_ratio=config.data.train_sample_ratio,
        augment=True,
        augment_prob=config.train.augmentation_probability,
        mean=config.data.mean,
        std=config.data.std,
    )
    
    val_dataset = MultiClipDataset(
        annotations_path=config.data.annotations_path,
        frames_dir=config.data.frames_dir,
        split='val',
        num_frames=config.data.num_frames,
        sampling_rate=config.data.sampling_rate,
        image_size=config.data.image_size,
        sample_ratio=config.data.test_sample_ratio,
        augment=False,
        mean=config.data.mean,
        std=config.data.std,
    )
    
    test_dataset = MultiClipDataset(
        annotations_path=config.data.annotations_path,
        frames_dir=config.data.frames_dir,
        split='test',
        num_frames=config.data.num_frames,
        sampling_rate=config.data.sampling_rate,
        image_size=config.data.image_size,
        sample_ratio=config.data.test_sample_ratio,
        augment=False,
        mean=config.data.mean,
        std=config.data.std,
    )
    
    # Create dataloaders (batch_size=1 because each video has multiple clips)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_multiclip,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_multiclip,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_multiclip,
    )
    
    print()
    
    # Create model
    print("Creating model...")
    if config.model.backbone == "spikeformer":
        model = MultiClipClassifier(
            backbone_type="spikeformer",
            hidden_dim=config.model.hidden_dim,
            consensus_mode=config.model.consensus_pooling,
            dropout=config.model.dropout,
            expansion_factor=config.model.expansion_factor,
            img_size=config.data.image_size,
            depths=config.model.spikeformer_depths,
            dims=config.model.spikeformer_dims,
            detach_reset=config.model.detach_reset,
        )
    elif config.model.backbone == "x3d":
        model = MultiClipClassifier(
            backbone_type="x3d",
            hidden_dim=config.model.hidden_dim,
            consensus_mode=config.model.consensus_pooling,
            dropout=config.model.dropout,
            expansion_factor=config.model.expansion_factor,
            model_name=config.model.x3d_model_name,
            pretrained=config.model.x3d_pretrained,
        )
    else:
        raise ValueError(f"Unknown backbone: {config.model.backbone}")
    
    model = model.to(config.device)
    
    # Freeze BatchNorm layers if requested (eval mode + freeze affine params)
    if getattr(config.model, 'frozen_bn', False):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    
    # Create loss function 
    if config.train.use_pos_weight:
        pos_weight_val = float(train_dataset.get_pos_weight())
        pos_weight_tensor = torch.tensor([pos_weight_val], device=config.device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Using pos_weight: {pos_weight_val:.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Create optimizer
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier") or name.startswith("backbone.head") or "classifier" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optim_groups = [
        {"params": backbone_params, "lr": config.train.lr},              # e.g., 5e-5
        {"params": head_params, "lr": max(config.train.lr * 20, 1e-3)},  # e.g., 1e-3
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        weight_decay=config.train.weight_decay,
    )

    # LR scheduler: cosine with optional warmup (per-iteration)
    scheduler = None
    if config.train.use_lr_scheduler:
        steps_per_epoch = len(train_loader)
        total_steps = max(1, config.train.epochs * steps_per_epoch)
        warmup_steps = int(config.train.warmup_ratio * total_steps)
        base_lr = config.train.lr

        schedulers = []
        milestones = []
        if warmup_steps > 0:
            start_factor = config.train.warmup_start_lr / base_lr if base_lr > 0 else 0.0
            def lr_lambda(step: int, start_factor=start_factor, warmup_steps=warmup_steps):
                if step >= warmup_steps:
                    return 1.0
                return start_factor + (1.0 - start_factor) * (step / max(1, warmup_steps))
            warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
            schedulers.append(warmup)
            milestones.append(warmup_steps)

        cosine_steps = max(1, total_steps - warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=config.train.cosine_end_lr)
        schedulers.append(cosine)
        if milestones:
            scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
        else:
            scheduler = cosine
    
    # Training loop
    print("\nStarting training...")
    best_mcc = -1
    history = {
        'train': [],
        'val': [],
        'test': None,
    }
    
    for epoch in range(1, config.train.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, config.device, epoch)
        history['train'].append(train_metrics)
        
        print(f"Epoch {epoch} [Train] - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, MCC: {train_metrics['mcc_normed']:.4f}, "
              f"Logit μ: {train_metrics['logit_mean']:.4f}, Logit σ: {train_metrics['logit_std']:.4f}, "
              f"Frac Pos: {train_metrics['frac_predicted_positive']:.4f}")
        
        # Validate
        if epoch % config.train.eval_every == 0:
            val_metrics = eval_epoch(model, val_loader, criterion, config.device, epoch)
            history['val'].append(val_metrics)
            
            print(f"Epoch {epoch} [Val]   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, MCC: {val_metrics['mcc_normed']:.4f}, "
                  f"Logit μ: {val_metrics['logit_mean']:.4f}, Logit σ: {val_metrics['logit_std']:.4f}, "
                  f"Frac Pos: {val_metrics['frac_predicted_positive']:.4f}")
            
            # Save best model
            if config.train.save_best and val_metrics['mcc_normed'] > best_mcc:
                best_mcc = val_metrics['mcc_normed']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, config.train.output_dir / 'best_model.pt')
                print(f"  → Saved best model (MCC: {best_mcc:.4f})")

        # Persist history after each epoch
        with open(config.train.output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print()
    
    # Final test evaluation
    print("Final evaluation on test set...")
    
    # Load best model
    if config.train.save_best:
        checkpoint = torch.load(
            config.train.output_dir / 'best_model.pt',
            weights_only=False,
            map_location=config.device,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_metrics = eval_epoch(model, test_loader, criterion, config.device, -1)
    history['test'] = test_metrics
    
    print("\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Balanced Acc: {test_metrics['balanced_acc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    print(f"  MCC (normed): {test_metrics['mcc_normed']:.4f}")
    
    # Save history (including test)
    with open(config.train.output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), config.train.output_dir / 'final_model.pt')
    
    print(f"\nTraining complete! Results saved to {config.train.output_dir}")


if __name__ == '__main__':
    # Create config
    config = Config()
    
    # You can override config here
    # config.model.backbone = "spikeformer"
    # config.train.epochs = 50
    # config.train.lr = 1e-4
    
    # Train
    train(config)

