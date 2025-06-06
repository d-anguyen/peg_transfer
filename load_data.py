# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 13:54:22 2025

@author: ndab1
"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as T
# import torchvision.io as io  # For video reading



VIDEO_PATH = './data/left/'
CSV_PATH = './data/PegTransfer.csv'
FRAMES_PER_CLIP = 50

annotations = pd.read_csv(CSV_PATH)
video_name = annotations.iloc[0]['id']
label1 = annotations.iloc[0]['object_dropped_within_fov']
label2 = annotations.iloc[0]['object_dropped_outside_of_fov']
label = label1 or label2



# Normalized 0-255 pixels to [0,1], resize, and convert to RGB 
# def preprocess_video_frame(frame, to_normalize = True, resize = (112,112), to_RGB = True):
#     if resize is not None:
#         frame = cv2.resize(frame, resize)  # Resize frame
#         frame.reshape(3, resize[0], resize[1])
#     if to_RGB:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     if to_normalize:
#         frame = frame / 255.0  # Normalize to [0, 1]
        
#     return frame

# Read the video and turn it into a numpy array with given length consisting frames that 
# are preprocessed (renormalized as above)
# def load_video(video_path, frames_per_clip=50):
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return None
    
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_indices = [i* (total_frames//frames_per_clip) for i in range(frames_per_clip)]
#     #frame_indices = list(range(0, total_frames, total_frames // (frames_per_clip) )) 
#     frames = []
    
#     for idx in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
        
#         if ret:
#             frame = preprocess_video_frame(frame)
#             frames.append(frame)
#         else:
#             break
    
#     cap.release()
    
#     # Convert list to numpy array of shape (frames, height, width, channels)
#     frames = np.array(frames)
#     return frames


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

# frames = load_video(VID_PATH +'blaox.mkv')
# print(show_frames(frames))
# print(frames.shape)



class VideoDataset(Dataset):
    def __init__(self, video_folder, annotations_file, resize_shape=(112, 112), frames_per_clip=50):
        self.video_folder = video_folder
        self.annotations = pd.read_csv(annotations_file)
        self.resize_shape = resize_shape
        self.frames_per_clip = frames_per_clip
        
    def __len__(self):
        # Number of samples in the dataset (number of video files)
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the video filename and its label
        video_filename = self.annotations.iloc[idx, 0]
        label1 = self.annotations.iloc[idx]['object_dropped_within_fov']
        label2 = self.annotations.iloc[idx]['object_dropped_outside_of_fov']
        label = label1 or label2
        
        # Load the video frames
        frames = self.load_video_frames(video_filename)
        
        # Apply transformation (e.g., normalization)
        if self.transform:
            frames = self.transform(frames)
        
        # Convert frames to tensor and return as (frames, channels, height, width) format
        frames = torch.tensor(frames, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return frames, label
    
    def preprocess_video_frame(frame, to_normalize = True, resize = (112,112), to_RGB = True):
        if resize is not None:
            frame = cv2.resize(frame, resize)  # Resize frame
            frame.reshape(3, resize[0], resize[1])
        if to_RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if to_normalize:
            frame = frame / 255.0  # Normalize to [0, 1]
            
        return frame

    
    def load_video_frames(self, video_id):
        # Construct the full path to the video
        #video_path = os.path.join(self.video_folder, video_id)
        video_path = self.video_folder + video_id + '.mkv'
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [i* (total_frames//self.frames_per_clip) for i in range(self.frames_per_clip)]
        #frame_indices = list(range(0, total_frames, total_frames // (frames_per_clip) )) 
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = self.preprocess_video_frame(frame)
                frames.append(frame)
            else:
                break
        
        cap.release()
        
        # Convert list to numpy array of shape (frames, height, width, channels)
        frames = np.array(frames)
        return frames
    
    def show_frames(frames, waitKey=100):
    # Display the frames as a video
        for frame in frames:
            # Show the frame
            cv2.imshow('Video', frame)
            # Wait for 30 milliseconds and check if the 'q' key is pressed to exit
            if cv2.waitKey(waitKey) & 0xFF == ord('q'):
                break
        # Close the OpenCV window
        cv2.destroyAllWindows()
        return 


# Example usage:
video_folder = VIDEO_PATH  # Folder containing video files
annotations_file = CSV_PATH  # Path to the CSV file with annotations

# Create the dataset
dataset = VideoDataset(video_folder, annotations_file,)

# Create a DataLoader for batching the data during training
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate over the dataloader and print the shape of video data and labels
for frames, labels in dataloader:
    print(f"Frames shape: {frames.shape}")
    print(f"Labels: {labels}")
