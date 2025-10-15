"""
Neural network models for video classification.
"""

import torch
import torch.nn as nn
import snntorch as snn


class VideoANN(nn.Module):
    """
    ANN for video classification with temporal averaging.
    
    Uses standard convolutional layers with ReLU activations.
    Processes frames independently and averages outputs over time.
    
    Architecture:
    - Two conv blocks (each: Conv2d -> MaxPool -> ReLU)
    - Fully connected layer for classification
    - Temporal averaging of frame-level predictions
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_h: int = 112,
        input_w: int = 112,
    ):
        """
        Args:
            num_classes: Number of output classes
            input_h: Input frame height
            input_w: Input frame width
        """
        super().__init__()
        
        # Fixed architecture with 2 convolutional blocks
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU(inplace=True)

        # Calculate flattened feature size after pooling
        # 112 -> 56 -> 28
        feature_size = 32 * (input_h // 4) * (input_w // 4)
        
        # Fully connected output layer
        self.fc = nn.Linear(feature_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
               where B=batch size, T=num_frames, C=channels, H=height, W=width
        
        Returns:
            logits: Output logits of shape (B, num_classes)
                   Computed by averaging frame-level predictions over time
        """
        batch, T, C, H, W = x.shape
        
        # Accumulate predictions over time
        sum_out = 0
        
        for t in range(T):
            out = x[:, t]  # B x C x H x W
            
            # Pass through convolutional blocks
            out = self.relu1(self.pool1(self.conv1(out)))
            out = self.relu2(self.pool2(self.conv2(out)))
            
            # Flatten and classify
            out = self.fc(out.view(batch, -1))
            sum_out += out
        
        # Average predictions over time
        out = sum_out / T
        
        return out


class VideoSNN(nn.Module):
    """
    Spiking Neural Network for video classification.
    
    Uses snnTorch Leaky Integrate-and-Fire neurons to process video frames
    with temporal dynamics. Accumulates spike outputs over time using rate coding.
    
    Architecture:
    - Two conv blocks (each: Conv2d -> MaxPool -> LIF neuron)
    - Fully connected layer for classification
    - Rate coding: average spike outputs over time
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_h: int = 112,
        input_w: int = 112,
        beta: float = 0.8,
        learn_beta: bool = True,
        learn_threshold: bool = True,
    ):
        """
        Args:
            num_classes: Number of output classes
            input_h: Input frame height
            input_w: Input frame width
            beta: Decay rate for membrane potential in LIF neurons
            learn_beta: Whether to learn the beta parameter
            learn_threshold: Whether to learn the firing threshold
        """
        super().__init__()
        
        # Fixed architecture with 2 convolutional blocks
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.lif1 = snn.Leaky(beta=beta, learn_beta=learn_beta, learn_threshold=learn_threshold)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=learn_beta, learn_threshold=learn_threshold)
        
        # Calculate flattened feature size after pooling
        # 112 -> 56 -> 28
        feature_size = 32 * (input_h // 4) * (input_w // 4)
        
        # Fully connected output layer
        self.fc = nn.Linear(feature_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
               where B=batch size, T=num_frames, C=channels, H=height, W=width
        
        Returns:
            logits: Output logits of shape (B, num_classes)
                   Computed using rate coding (average spike outputs over time)
        """
        batch, T, C, H, W = x.shape
        
        # Initialize membrane potentials for all LIF neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Accumulate spike outputs over time
        sum_out = 0
        
        for t in range(T):
            out = x[:, t]  # B x C x H x W
            # Pass through convolutional blocks with spiking neurons
            out = self.pool1(self.conv1(out))
            spk1, mem1 = self.lif1(out, mem1)

            out = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(out, mem2)
            # Flatten and classify
            spk_out = self.fc(spk2.view(batch, -1))
            sum_out += spk_out
        
        # Rate coding: average spike outputs over time
        out = sum_out / T
        
        return out