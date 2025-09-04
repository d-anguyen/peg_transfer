import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd



# Calculate the difference between two frames in some metric
# Here, frame1 and frame2 should not be in uint8 format to preven subtraction like 0-10 = 246
# Won't be any problem for us as we normalized each frame already before calling this function
def frame_diff(frame1, frame2, metric='l2'):
    if metric == 'l2':
        diff = frame1-frame2
        return np.sum(diff**2)
    
    if metric == 'l1':
        return np.sum(np.abs(diff))
    
    if metric == "ssim":
        from skimage.metrics import structural_similarity as ssim
        return 1.0 - ssim(frame1, frame2, data_range=1.0, channel_axis=0)
    
    else:
        raise RuntimeError('Metric not supported!')


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
    
    def load_frames(self, video_id, to_RGB=True, to_CHW=True, to_normalize=True, sampling_method='uniform'):
        # Construct the full path to the video
        #video_path = os.path.join(self.video_folder, video_id)
        video_path = self.video_folder + video_id + '.mkv'
        video = cv2.VideoCapture(video_path)
        frames = []
        
        if not video.isOpened():
            print("Error: Could not open video.")
            return None
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"total number of frames: {(total_frames)}")
        
        
        frame_indices = self.sample_frame_idxs(video, sampling_method=sampling_method, step=1, metric='l2')
        #frame_indices = np.array([0,1,2,4,5])   
        frames = []
        
        #video = cv2.VideoCapture(video_path)
        for idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video.read()
            
            if ret:
                frame = self.preprocess_frame(frame, to_RGB=to_RGB, to_CHW=to_CHW, to_normalize=to_normalize)
                frames.append(frame)
            else:
                print('Frame not found!')
                break
        
        video.release()
        
        # Convert list to numpy array of shape (T x C x H x W)
        frames = np.array(frames, dtype=np.float32)
        return frames
    
    
    def sample_frame_idxs(self, video, sampling_method='uniform', step=1, metric='l2'):
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if sampling_method == 'uniform':
            uniform_indices = [i* (total_frames//self.frames_per_clip) for i in range(self.frames_per_clip)]
            #uniform_indices = list(range(0, total_frames, total_frames // (frames_per_clip) ))
            return uniform_indices
        
        elif sampling_method == 'topk':
            if step >= total_frames//self.frames_per_clip:
                raise RuntimeError('Step to fast!')
            
            ret, prev = video.read()
            if not ret:
                #video.release()
                raise RuntimeError("Cannot open first frame")
            prev_frame = self.preprocess_frame(prev) # turn the frame to the input format/ maybe this is not necessary
            
            diffs, idxs = [], []
            i=0
            while i < total_frames:
                i+=step
                ret, curr = video.read()
                if not ret:
                    #video.release()
                    break
                curr_frame = self.preprocess_frame(curr)
                
                d = frame_diff(prev_frame, curr_frame, metric=metric)
                diffs.append(d)
                idxs.append(i)
                prev_frame = curr_frame
            #video.release()
            
            # Choose top-k indices and then order them in ascending order
            topk_idxs = np.argpartition(diffs, -self.frames_per_clip)[-self.frames_per_clip:]
            ordered_topk_idxs = np.sort(topk_idxs)
            
            return ordered_topk_idxs
    
                
    # Preprocess each frame, which includes resizing and possibly converting to RGB/CHW format and normalizing 
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
    
    def show_vid(self, idx, waitKey=400, save_to='./sampled_videos/', sampling_method="topk"):
        video_id = self.annotations.iloc[idx]['id']
        frames = np.array(self.load_frames(video_id, sampling_method=sampling_method,
                        to_RGB=False, to_CHW=False, to_normalize=False), dtype=np.uint8)
        if save_to:
            fps = 1000/waitKey
            out = cv2.VideoWriter(save_to + sampling_method +'_'+video_id+'.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 
                                  fps, (self.resize_shape[1],self.resize_shape[0]))
        
        for frame in frames:
            # Show the frame
            cv2.imshow('Video', frame)
            if save_to:
                out.write(frame)
            # Wait for 100 milliseconds and check if the 'q' key is pressed to exit
            if cv2.waitKey(waitKey) & 0xFF == ord('q'):
                break
            
        if save_to:
            out.release()
            
        # Close the OpenCV window
        cv2.destroyAllWindows()
        return
    
        
    
#test
VIDEO_PATH = './data/left/'
CSV_PATH = './data/PegTransfer.csv'
FRAMES_PER_CLIP = 30
FRAME_SIZE = (540,960)#HxW, (72,128)  original = 540x960
train_dataset = VideoDataset(VIDEO_PATH, CSV_PATH, 'train', resize_shape=FRAME_SIZE, frames_per_clip=FRAMES_PER_CLIP)

save_path = './sampled_' + str(FRAMES_PER_CLIP)+'_frames/'
os.makedirs(save_path, exist_ok=True)
train_dataset.show_vid(3, save_to = save_path, sampling_method='uniform')


label = train_dataset[3][1]
if label==1:
    print('dropped')
elif label==0:
    print('not dropped')
