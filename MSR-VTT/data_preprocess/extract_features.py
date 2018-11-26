import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import skvideo.io
import time
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1  # Extract Resnet features in a tensor of batch_size videos.

video_path = '../../../dataset/MSR-VTT/videos_224/'

# Download pre trained Resnet101.
resnet50 = models.resnet152(pretrained=True)
modules = list(resnet50.children())[:-1]
resnet50 = nn.Sequential(*modules)
for p in resnet50.parameters():
    p.requires_grad = False
resnet50 = resnet50.to(device)


# Convert id's list to dictionary with key as id, value as id numpy array
def video_array(batch_ids):
    batch_array = {}
    try:
        video = skvideo.io.vread(video_path + batch_ids[0] + '.mp4')
        batch_array.update({batch_ids[0]:video})
        return batch_array
    except:
        return 0

# Convert array dictionary to resnet_dictionary with key as id, value as id tensor feature array
def resnet_features(batch_arrayd):

    with torch.no_grad():

        batch_feature = {}
        ids = list(batch_arrayd.keys())
        video_array = [x for x in batch_arrayd.values()]
        array_sizes = [x.shape[0] for x in batch_arrayd.values()]

        video1_array = np.array(video_array[0], dtype = np.float32)  # change datatype of frames to float32
        video_tensor = torch.from_numpy(video1_array)

        video_frames = video_tensor.size()[0]
        num_steps = math.ceil(video_frames / 100)
        resnet_feature = torch.zeros(video_frames,2048)

        video_tensor = video_tensor.permute(0,3,1,2) # change dimension to [?,3,224,224]

        for i in range(num_steps):
            start = i*100
            end = min((i+1)*100, video_frames)
            tensor_var = Variable(video_tensor[start:end]).to(device)
            feature = resnet50(tensor_var).data
            feature.squeeze_(3)
            feature.squeeze_(2)
            resnet_feature[start:end] = feature

        return {ids[0]:resnet_feature}


# seconds to minutes
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return m, s

# time in minutes b/w since to now
def timeSince(since):
    now = time.time()
    s = now - since
    min, sec = asMinutes(s)
    return min, sec


all_ids = os.listdir(video_path)
all_ids = [video[:-4] for video in all_ids]
total_ext = len(all_ids)

print('Total videos', total_ext)
print('batch size', batch_size)

total_iter = total_ext
batch_start =  0
batch_end = batch_size

iter = 1

while batch_start != batch_end:
    print('Iteration', iter, 'left', total_iter-iter, end=' ')
    start_time = time.time()
    batch_id = all_ids[batch_start:batch_start+batch_size]
    batch_arrayd = video_array(batch_id)

    if batch_arrayd != 0 : # video array present
        torch.cuda.empty_cache()
        batch_featuresd = resnet_features(batch_arrayd)

        state = batch_featuresd

        resnet_feat_path = '../../../dataset/MSR-VTT/resnet_features/' + batch_id[0] + '.pt'

        torch.save(state, resnet_feat_path)
        print('time taken (%dm %ds)'% timeSince(start_time))


    batch_start = batch_end
    batch_end = min(batch_start+batch_size, total_ext)
    iter += 1
