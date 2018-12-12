import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm
import cv2

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

object_path = '../../../dataset/MSVD/object_crop/'
output_path = '../../../dataset/MSVD/object_features/'

# Download pre trained Resnet152.
resnet152 = models.resnet152(pretrained=True)
modules = list(resnet152.children())[:-1]
resnet152 = nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False
resnet152 = resnet152.to(device)

def resize_image(image):
    return cv2.resize(image, (224,224))

def extract_features(image):
    image = torch.tensor(image).unsqueeze(0).permute(0,3,1,2)
    image = image.type(torch.float32)
    image_var = Variable(image).to(device)

    resnet_features = resnet152(image_var).data
    resnet_features = resnet_features.squeeze()

    return resnet_features

if __name__ == "__main__":

    video_ids = os.listdir(object_path)
    video_ids = [vids for vids in video_ids if vids[0] != '.']

    for vid in tqdm(video_ids):
        object_features = {}

        object_frames = os.listdir(object_path + vid)

        for frame in object_frames:

            object_in_frame = os.listdir(object_path + vid + '/' + frame + '/')

            if len(object_in_frame) == 0:
                object_features[int(frame)] = {}
                continue

            frame_features = {}

            for object in object_in_frame:

                object_image = cv2.imread(object_path + vid + '/' + frame + '/' + object)
                object_image = resize_image(object_image)
                object_ftrs= extract_features(object_image).to('cpu').numpy()
                torch.cuda.empty_cache()

                frame_features[object[:-4]] = object_ftrs

            object_features[int(frame)] = frame_features

        torch.save(object_features, output_path + vid + '.pt')






#
