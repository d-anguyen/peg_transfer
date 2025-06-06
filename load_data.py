# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 13:54:22 2025

@author: ndab1
"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as T
# import torchvision.io as io  # For video reading



VID_PATH = './data/left/'
CSV_PATH = './data/PegTransfer.csv'
FRAMES_PER_CLIP = 50

annotations = pd.read_csv(CSV_PATH)
video_name = annotations.iloc[0]['id']
label1 = annotations.iloc[0]['object_dropped_within_fov']
label2 = annotations.iloc[0]['object_dropped_outside_of_fov']
label = label1 or label2



# Normalized 0-255 pixels to [0,1], resize, and convert to RGB 
def preprocess_video_frame(frame, to_normalize = True, resize = (112,112), to_RGB = True):
    if resize is not None:
        frame = cv2.resize(frame, resize)  # Resize frame
        frame.reshape(3, resize[0], resize[1])
    if to_RGB:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    if to_normalize:
        frame = frame / 255.0  # Normalize to [0, 1]
        
    return frame

# Read the video and turn it into a numpy array with given length consisting frames that 
# are preprocessed (renormalized as above)
def load_video(video_path, frames_per_clip=50):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [i* (total_frames//frames_per_clip) for i in range(frames_per_clip)]
    #frame_indices = list(range(0, total_frames, total_frames // (frames_per_clip) )) 
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame = preprocess_video_frame(frame)
            frames.append(frame)
        else:
            break
    
    cap.release()
    
    # Convert list to numpy array of shape (frames, height, width, channels)
    frames = np.array(frames)
    return frames


# if not cap.isOpened():
#     print("Error: Could not open video.")
    
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #number of frames (annotated in 2nd column)

# frame_indices = list(range(1, total_frames, total_frames // frames_per_clip)) #originally start at 0
# frames = []

# #cap = cv2.VideoCapture(VID_PATH+'blaox.mkv')  # Replace with your video path

# for idx in frame_indices:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#     ret, frame = cap.read()
#     if ret:
#         # Resize frame to a fixed size, for example 112x112
#         frame = cv2.resize(frame, (112, 112))
#         # Convert BGR to RGB (OpenCV uses BGR by default)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)
#     else:
#         break
# cap.release()
# # Convert list to numpy array of shape (frames, height, width, channels)
# frames = np.array(frames)

def show_frames(frames, waitKey=100):
# Display the frames as a video
    for frame in frames:
        # Show the frame
        cv2.imshow('Video', frame)
        # Wait for 30 milliseconds and check if the 'q' key is pressed to exit
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    # Close the OpenCV window
    cv2.destroyAllWindows()
    return 

frames = load_video(VID_PATH +'blaox.mkv')
print(show_frames(frames))

print(frames.shape)
class VideoDataset(Dataset):
    def __init__(self, csv_path, video_folder, transform=None, frames_per_clip=16):
        self.annotations = pd.read_csv(csv_path)
        self.video_folder = video_folder
        self.transform = transform
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_name = self.annotations.iloc[idx]['id']  # assuming first col is filename
        label1 = self.annotations.iloc[idx]['object_dropped_within_fov']
        label2 = self.annotations.iloc[idx]['object_dropped_outside_of_fov']
        label = label1 or label2
        
        video_path = f"{self.video_folder}/{video_name}.mkv"
        
        # Load video: video shape (T, H, W, C)
        video, _, _ = io.read_video(video_path, pts_unit='sec')

        # Sample or pad frames to fixed length
        video = self._sample_frames(video)

        # video shape -> (T, H, W, C), convert to (C, T, H, W)
        video = video.permute(3, 0, 1, 2).float() / 255.0

        if self.transform:
            video = self.transform(video)

        return video, label

    def _sample_frames(self, video):
        total_frames = video.shape[0]
        if total_frames >= self.frames_per_clip:
            indices = torch.linspace(0, total_frames-1, self.frames_per_clip).long()
            sampled = video[indices]
        else:
            # pad with last frame if less frames than required
            pad_len = self.frames_per_clip - total_frames
            pad_frames = video[-1].unsqueeze(0).repeat(pad_len, 1, 1, 1)
            sampled = torch.cat([video, pad_frames], dim=0)
        return sampled
