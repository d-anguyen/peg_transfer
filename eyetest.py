import os
from load_data import *

# eye-test
#test
VIDEO_PATH = './data/left/'
CSV_PATH = './data/PegTransfer.csv'
FRAMES_PER_CLIP = 10
FRAME_SIZE = (540,960)#HxW, (72,128)  original = 540x960
sampling_method = 'topk'
train_dataset = VideoDataset(VIDEO_PATH, CSV_PATH, 'train', resize_shape=FRAME_SIZE, 
                             frames_per_clip=FRAMES_PER_CLIP, sampling_method=sampling_method)

save_path = './sampled_' + str(FRAMES_PER_CLIP)+'_frames/'
os.makedirs(save_path, exist_ok=True)
train_dataset.show_vid(3, save_to = save_path, sampling_method=sampling_method)


label = train_dataset[3][1]
if label==1:
    print('dropped')
elif label==0:
    print('not dropped')
