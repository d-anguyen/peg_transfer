from load_data import *
from models import *
import torch
import torch.nn as nn
import snntorch as snn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

VIDEO_PATH = '/home/groups/ai/ducanh/PegTransferData/left/'
CSV_PATH = '/home/groups/ai/ducanh/PegTransferData/PegTransfer.csv'
FRAMES_PER_CLIP = 50
FRAME_SIZE = (112,112)
batch_size = 20
lr = 1e-3
weight_decay = 1e-4

# Create the datasets and dataloaders
train_dataset = VideoDataset(VIDEO_PATH, CSV_PATH, 'train', resize_shape=FRAME_SIZE, frames_per_clip=FRAMES_PER_CLIP)
val_dataset = VideoDataset(VIDEO_PATH, CSV_PATH, 'val', resize_shape=FRAME_SIZE, frames_per_clip=FRAMES_PER_CLIP)
test_dataset = VideoDataset(VIDEO_PATH, CSV_PATH, 'test', resize_shape=FRAME_SIZE, frames_per_clip=FRAMES_PER_CLIP)
print(f'The train/validation/test dataset has {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} samples.')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create
SNN = ShallowSNNVideoNet().to(device)

# frames, labels = next(iter(train_loader))
# frames = frames.to(device)
# labels = labels.to(device)





train(SNN, train_loader, num_epochs=1)
# start_time = time.time()



# num_epochs = 1

# optimizer = torch.optim.Adam(SNN.parameters(), lr=1e-3, weight_decay=1e-4)
# criterion = nn.CrossEntropyLoss()

# for epoch in range(num_epochs):
#     SNN.train()
#     for inputs, labels in train_loader:
#         inputs = inputs.float().to(device)    # [B, 50, 3, 112, 112]
#         labels = labels.long().to(device)

#         outputs = SNN(inputs)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    

# end_time = time.time()
# print(f"Elapsed time: {end_time - start_time:.4f} seconds")


# # Evaluation 
# SNN.eval()
# correct, total = 0, 0

# with torch.no_grad():
#     for inputs, labels in val_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = SNN(inputs)

#         predictions = outputs.argmax(dim=1)
#         correct += (predictions == labels).sum().item()
#         total += labels.size(0)

# accuracy = 100.0 * correct / total
# print(f"Validation Accuracy: {accuracy:.2f}%")