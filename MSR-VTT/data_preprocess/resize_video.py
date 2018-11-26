import os
import time
import math
import glob
import subprocess

video_path = '../../../dataset/MSR-VTT/videos_required/'
video_224_path = '../../../dataset/MSR-VTT/videos_224/'
log_file = "resize_unable.txt"
open(log_file, 'w').close()  # empty log file


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def resize_224(src_path):
    print('resizing...')
    assert os.path.exists(src_path), "video path exists"
    video_id = src_path.split('/')[-1]
    print('video id', video_id)
    try:
        sys_cmd = ["ffmpeg", "-i", src_path, "-vf", "scale=224:224", video_224_path + video_id]
        subprocess.check_call(sys_cmd, stdout=subprocess.DEVNULL, stderr= subprocess.DEVNULL)
    except:
        f = open(log_file,'a')
        f.write(video_id + '\n')
        f.close()


if __name__ == "__main__":

    assert os.path.exists(video_path), "Source Directory does not exists"
    assert os.path.exists(video_224_path), "Target Directory does not exists"

    total_videos = len([name for name in os.listdir(video_path)])
    print('Total videos',total_videos)

    for i, file_path in enumerate(glob.glob(video_path + "*")):
        start = time.time()
        resize_224(file_path)
        print(i+1, 'Left',total_videos-i-1 ,file_path, timeSince(start))

    total_224_videos = len([name for name in os.listdir(video_224_path)])
print('Total resize videos',total_224_videos)
