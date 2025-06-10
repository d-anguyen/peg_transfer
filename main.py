from load_data import *
from models import *
from train import *

import torch
import torch.nn as nn
import snntorch as snn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


VIDEO_PATH = './data/left/'
CSV_PATH = './data/PegTransfer.csv'
#VIDEO_PATH = '/home/groups/ai/ducanh/PegTransferData/left/'
#CSV_PATH = '/home/groups/ai/ducanh/PegTransferData/PegTransfer.csv'

#VIDEO_PATH = '../PegTransferData/left/'
#CSV_PATH = '../PegTransferData/PegTransfer.csv'

FRAMES_PER_CLIP = 50
FRAME_SIZE = (72,128) #HxW, original = 540x960
batch_size = 8

lr = 1e-3
weight_decay = 1e-4
num_epochs = 1

# Create the datasets and dataloaders
train_dataset = VideoDataset(VIDEO_PATH, CSV_PATH, 'train', resize_shape=FRAME_SIZE, frames_per_clip=FRAMES_PER_CLIP)
val_dataset = VideoDataset(VIDEO_PATH, CSV_PATH, 'val', resize_shape=FRAME_SIZE, frames_per_clip=FRAMES_PER_CLIP)
test_dataset = VideoDataset(VIDEO_PATH, CSV_PATH, 'test', resize_shape=FRAME_SIZE, frames_per_clip=FRAMES_PER_CLIP)
print(f'The train/validation/test dataset has {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} samples.')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create a SNN model and train
SNN = ShallowSNNVideoNet(num_classes=2, input_h=FRAME_SIZE[0], input_w=FRAME_SIZE[1]).to(device)
train(SNN, train_loader, val_loader, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay)
