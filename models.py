import torch
import torch.nn as nn
import snntorch as snn
    

class ShallowSNNVideoNet(nn.Module):
    def __init__(self, num_classes=2, input_h=112, input_w=112):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)   # 3xHxW -> 16 x H/2 x W/2
        self.lif1 = snn.Leaky(beta=0.8, learn_beta=True, learn_threshold=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)   # 16 x H/2 x W/2 -> 32 x H/4 x W/4
        self.lif2 = snn.Leaky(beta=0.8, learn_beta=True, learn_threshold=True)

        self.fc = nn.Linear(32 * (input_h // 4) * (input_w // 4), num_classes)

    def forward(self, x):  # x: [batch, T, 3, H, W]
        batch, T, C, H, W = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        sum_out = 0 #to_store_sum_of_spike_output_over_time
        for t in range(T):
            out = self.pool1(self.conv1(x[:, t])) #B x 16 x H/2 x W/2
            spk1, mem1 = self.lif1(out, mem1)
            
            out = self.pool2(self.conv2(spk1)) #B x 32 x H/4 x W/4
            spk2, mem2 = self.lif2(out, mem2)
            
            spk_out = self.fc( spk2.view(batch,-1) ) #B x 2
            sum_out += spk_out 
            
        out = sum_out/T # rate-coding
        return out
    
    