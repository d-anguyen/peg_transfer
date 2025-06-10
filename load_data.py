# import os
import torch
import torch.nn as nn
import snntorch as snn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd




device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoDataset(Dataset):
    def __init__(self, video_folder, csv_path, data_split, resize_shape=(72,128), frames_per_clip=50):
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
        frames = self.load_frames(video_id)
        
        return frames, label
    
    def load_frames(self, video_id, to_RGB=True, to_CHW=True, to_normalize=True):
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
                frame = self.preprocess_frame(frame, to_RGB=to_RGB, to_CHW=to_CHW, to_normalize=to_normalize)
                frames.append(frame)
            else:
                print('Frame not found!')
                break
        
        cap.release()
        
        # Convert list to numpy array of shape (frames, height, width, channels)
        frames = np.array(frames, dtype=np.float32)
        return frames
    
    def preprocess_frame(self, frame, to_RGB=True, to_CHW=True, to_normalize=True):
        # Resize frame
        if self.resize_shape is not None: 
            height, width = self.resize_shape[0], self.resize_shape[1]
            frame = cv2.resize(frame, [width, height])  
        
        # Convert from BGR to RGB
        if to_RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        
        # Convert HxWxC to CxHxW
        if to_CHW: 
            frame = np.transpose(frame, (2, 0, 1)) 
        
        # Normalize to [0, 1]
        if to_normalize:
            frame = frame / 255.0      
        return frame
    
    def show_vid(self, idx, waitKey=100):
        video_id = self.annotations.iloc[idx]['id']
        frames = np.array(self.load_frames(video_id, to_RGB=False, to_CHW=False, to_normalize=False), dtype=np.uint8)
        for frame in frames:
            # Show the frame
            cv2.imshow('Video', frame)
            # Wait for 30 milliseconds and check if the 'q' key is pressed to exit
            if cv2.waitKey(waitKey) & 0xFF == ord('q'):
                break
        # Close the OpenCV window
        cv2.destroyAllWindows()
        return
    
