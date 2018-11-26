import json
import os
from pytube import YouTube

# Log for videos not downloaded.
log_file_path = '../../../dataset/MSR-VTT/logs/download_fail.txt'
open(log_file_path,'w').close()

# Log for last downloaded file
log_download_path = '../../../dataset/MSR-VTT/logs/last_download.txt'

if not os.path.exists(log_download_path):
    start = -1
    open(log_download_path, 'w').close()
else:
    f = open(log_download_path, 'r')
    start = int(f.readlines()[0]) + 1
    f.close()

test_info = json.load(open('../../../dataset/MSR-VTT/test_info.json'))

def downloadYouTube(video_id, video_url, path):

    try:
        yt = YouTube(video_url)
        yt.streams.first().download()
        os.rename(yt.streams.first().default_filename, path + video_id + '.mp4')
    except:
        video_log = open(log_file_path,'a')
        video_log.write(video_id + '\t' + video_url +'\n')
        video_log.close()

total_test_videos = len(test_info['videos'])

print('total test videos', total_test_videos)
print('downloading test videos...')

# Download test videos
for i, item in enumerate(test_info['videos']):
    video_id = item['video_id']
    video_url = item['url']

    if i >= start:
        downloadYouTube(video_id, video_url, '../../../dataset/MSR-VTT/videos/test/')
        f = open(log_download_path, 'w')
        f.write(str(i))
        f.close()

    print('Downloaded test', i+1, 'left', total_test_videos - i -1)
