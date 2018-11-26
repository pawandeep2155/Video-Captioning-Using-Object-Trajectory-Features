import os
import time
import math
import glob
import subprocess
from tqdm import tqdm
import json
import cv2
from skvideo.io import vread, vwrite
import numpy as np
from multiprocessing import Pool

video_path = '../../../dataset/MSR-VTT/videos/train/'
videos_required_path = '../../../dataset/MSR-VTT/videos_required/'
json_path = '../../../dataset/MSR-VTT/train_info.json'

log_file = "required_unable.txt"
if not os.path.exists(log_file):
    open(log_file, 'w').close()  # empty log file

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def required_video(video_info):

    try:
        video_path = video_info[0]
        print(video_path)
        start_time = video_info[1]
        end_time = video_info[2]
        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        video = vread(video_path)
        video = video[math.floor(start_time*fps):math.floor(end_time*fps)]
        vwrite(videos_required_path + video_path.split('/')[-1], video)
    except:
        f = open(log_file, 'a')
        f.write(video_path + '\n')
        f.close()


if __name__ == "__main__":

    assert os.path.exists(video_path), "Source Directory does not exists"
    assert os.path.exists(videos_required_path), "Target Directory does not exists"
    assert os.path.exists(json_path), "Json file does not exists"

    with open(json_path) as f:
        data = json.load(f)

    videos_available = os.listdir(video_path)
    videos_available = [item[:-4] for item in videos_available]

    videos_already_requird = os.listdir(videos_required_path)
    videos_already_requird = [item[:-4] for item in videos_already_requird]


    videos_info = []


    for item in data['videos']:
        vid = item['video_id']
        start_time = item['start time']
        end_time = item['end time']
        if vid in videos_available and vid not in videos_already_requird:
            # mp4 downloaded but not present in videos_required_path
            videos_info.append([video_path + vid+'.mp4', start_time, end_time])

    print('videos info', len(videos_info))
    p = Pool(20)
    p.map(required_video, videos_info)
