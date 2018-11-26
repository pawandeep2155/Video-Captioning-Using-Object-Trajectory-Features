from multiprocessing import Pool

def f(x):
    return x[0]*x[1]

if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, [[1,2], [2,5], [3,7]]))


'''
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
from multiprocessing import pool

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


def required_video(video_path, start_time, end_time):

    try:
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

    videos_info = {}

    for item in data['videos']:
        key = item['video_id']
        start_time = item['start time']
        end_time = item['end time']
        videos_info[key] = [start_time, end_time]

    videos_available = os.listdir(video_path)

    videos_available = [item[:-4] for item in videos_available]

    for video_id in videos_available:
        start = time.time()
        start_time = videos_info[video_id][0]
        end_time = videos_info[video_id][1]
        required_video(video_path+video_id+'.mp4', start_time, end_time)
        print(timeSince(start))
        break
'''












# import json
# import os
# from pytube import YouTube
# from multiprocessing import Process, Pool
#
# train_info = json.load(open('../../../dataset/MSR-VTT/train_info.json'))
#
# def downloadYouTube(video_id, video_url, path=None):
#
#     try:
#         yt = YouTube(video_url)
#         yt.streams.first().download()
#         os.rename(yt.streams.first().default_filename, video_id + '.mp4')
#     except:
#         pass
#
# if __name__ == "__main__":
#
#     total_train_videos = len(train_info['videos'])
#
#
#     existing_videos = os.listdir( '../../../dataset/MSR-VTT/videos/train/')# + [fname for fname in os.listdir(os.getcwd()) if fname.endswith('.mp4')]
#     existing_videos = [video[:-4] for video in existing_videos]
#
#     print('total train videos', total_train_videos)
#     print('downloading train videos...')
#
#     video_to_download = train_info
#
#     p = Pool(10)
#     p.map(downloadYouTube, [1, 2, 3])
#
#
#
#     for i, item in enumerate(train_info['videos']):
#         video_id = item['video_id']
#         print('i', i,'video id', video_id)
#         video_url = item['url']
#
#         if video_id not in existing_videos:
#             # Extracting frames on multicores.
#             procs = []
#             proc = Process(target=downloadYouTube, args=(video_id, video_url))
#             procs.append(proc)
#             proc.start()
#             break
#
#     # complete the processes
#     for proc in procs:
#         proc.join()
