import torch
import torch.nn as nn
import snntorch as snn

class SNNVideoClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.lif1 = snn.Leaky(
            beta=0.8, threshold=1.0, learn_beta=True, learn_threshold=True
        )
        # Using pooling to reduce dimensionality:
        self.pool = nn.MaxPool2d(2, 2)  # reduces 112x112 -> 56x56
        self.fc = nn.Linear(16 * 56 * 56, num_classes)

    def forward(self, x):
        batch_size, num_steps, C, H, W = x.shape
        mem = self.lif1.init_leaky()
        spk_sum = torch.zeros(batch_size, 16, H // 2, W // 2, device=x.device)

        for step in range(num_steps):
            out = self.conv1(x[:, step])         # [batch, 16, 112, 112]
            out = self.pool(out)                 # [batch, 16, 56, 56]
            spk, mem = self.lif1(out, mem)       # SNN spiking neurons
            spk_sum += spk                       # accumulate spikes (rate coding)

        out = spk_sum.view(batch_size, -1)
        out = self.fc(out)
        return out
    

class ShallowSNNVideoNet(nn.Module):
    def __init__(self, num_classes=2, input_h=112, input_w=112):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)   # 112x112 -> 56x56
        self.lif1 = snn.Leaky(beta=0.8, learn_beta=True, learn_threshold=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)   # 56x56 -> 28x28
        self.lif2 = snn.Leaky(beta=0.8, learn_beta=True, learn_threshold=True)

        self.fc = nn.Linear(32 * (input_h // 4) * (input_w // 4), num_classes)

    def forward(self, x):  # x: [batch, T, 3, H, W]
        batch, T, C, H, W = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_sum = 0
        for t in range(T):
            out = self.pool1(self.conv1(x[:, t]))
            spk1, mem1 = self.lif1(out, mem1)
            out = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(out, mem2)
            spk_sum += spk2
        out = spk_sum.view(batch, -1)
        out = self.fc(out)
        return out

