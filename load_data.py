# import os
import torch
import torch.nn as nn
import snntorch as snn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd




device = 'cuda' if torch.cuda.is_available() else 'cpu'

VIDEO_PATH = './data/left/'
CSV_PATH = './data/PegTransfer.csv'
FRAMES_PER_CLIP = 50
FRAME_SIZE = (54,96)


class VideoDataset(Dataset):
    def __init__(self, video_folder, csv_path, data_split, resize_shape=(112, 112), frames_per_clip=50):
        self.video_folder = video_folder
        df = pd.read_csv(csv_path)
        self.annotations = df[df['data_split'] == data_split]
        self.resize_shape = resize_shape
        self.frames_per_clip = frames_per_clip
        
    def __len__(self):
        # Number of samples in the dataset (number of video files)
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the video filename and its label
        video_id = self.annotations.iloc[idx]['id']
        label1 = self.annotations.iloc[idx]['object_dropped_within_fov']
        label2 = self.annotations.iloc[idx]['object_dropped_outside_of_fov']
        label = int(label1 or label2)
        
        # Load the video frames
        frames = self.load_video_frames(video_id)
        
        return frames, label
    
    def preprocess_video_frame(self, frame, to_normalize = True, to_RGB = True):
        frame = cv2.resize(frame, self.resize_shape)  # Resize frame
        if to_RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = np.transpose(frame, (2, 0, 1)) # Convert HxWxC to CxHxW
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
        frames = np.array(frames, dtype=np.float32)
        return frames

def load_frames(video_folder, video_id, frames_per_clip, to_process=False):
    # Construct the full path to the video
    #video_path = os.path.join(self.video_folder, video_id)
    video_path = video_folder + video_id + '.mkv'
    cap = cv2.VideoCapture(video_path)
    frames = []
    
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
            if to_process==True:
                frame = preprocess_frame(frame)
            frames.append(frame)
        else:
            print('Frame not found!')
            break
    
    cap.release()
    
    # Convert list to numpy array of shape (frames, height, width, channels)
    frames = np.array(frames, dtype=np.float32)
    return frames

def preprocess_frame(resize_shape, frame, to_normalize = True, to_RGB = True):
    frame = cv2.resize(frame, resize_shape)  # Resize frame
    if to_RGB:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.transpose(frame, (2, 0, 1)) # Convert HxWxC to CxHxW
    if to_normalize:
        frame = frame / 255.0  # Normalize to [0, 1]    
    return frame
    
def show_frames(frames, waitKey=100):
# Display the frames as a video (HxWxC)
    for frame in frames:
        # Show the frame
        cv2.imshow('Video', frame)
        # Wait for 30 milliseconds and check if the 'q' key is pressed to exit
        if cv2.waitKey(waitKey) & 0xFF == ord('q'):
            break
    # Close the OpenCV window
    cv2.destroyAllWindows()
    return 

    
