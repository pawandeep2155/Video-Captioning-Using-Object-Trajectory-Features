import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm
import cv2

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
images_path = '/media/data2/pawan/FlowNetPytorch-master/flow_MSVD/'
output_path = '../../../dataset/MSVD/optical_flow/'

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

    video_ids = os.listdir(images_path)
    video_ids = [vids for vids in video_ids if vids[0] != '.']

    for vid in tqdm(video_ids):

        if not os.path.exists(output_path + vid + '.pt'):
            video_features = {}

            frames = os.listdir(images_path + vid)
            frames = [int(frm[1:-8]) for frm in frames]
            frames.sort()
            for frm in frames:
                frame_path = images_path + vid + '/' + 'f' + str(frm) + 'flow.png'
                image = cv2.imread(frame_path)
                image = resize_image(image)
                image_features = extract_features(image).to('cpu').numpy()
                torch.cuda.empty_cache()

                frame_number = frm
                video_features[frame_number + 1] = image_features

            torch.save(video_features, output_path + vid + '.pt')
