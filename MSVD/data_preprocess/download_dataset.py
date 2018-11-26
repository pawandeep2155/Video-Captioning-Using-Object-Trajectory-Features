import csv
from pytube import YouTube
from multiprocessing import Pool
import os

csv_path = '../../../dataset/MSVD/dataset.csv'
videos_path = '/media/data/pawan/YouTubeClips/'

def download_video(video):
    try:
        video_id = video[0]
        video_url = video[1]
        yt = YouTube(video_url)
        yt.streams.first().download()
        os.rename(yt.streams.first().default_filename, video_id + '.mp4')
    except:
        pass

if __name__ == "__main__":

    video_details = []

    with open(csv_path,'r', newline='') as csvfile:
        dataset = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(dataset):
            if len(row)!=0 and i != 0:
                video_start = row[1]
                video_end = row[2]
                video_id = row[0] + '_' + video_start + '_' + video_end
                video_url = "http://www.youtube.com/watch?v=" + video_id
                if video_id not in video_details:
                    video_details.append(video_id)

    video_name = os.listdir(videos_path)
    video_name = [item[:-4] for item in video_name]
    print('video', len(video_name))
    print('csv ids', len(video_details))

    for ids in video_details:
        if ids not in video_name:
            print(ids)



    # p = Pool(10)
    # p.map(download_video, video_details)
