import os
from glob import glob
from multiprocessing import Pool

input_path = "../../../dataset/MSVD/videos_224/"
output_path = "../../../dataset/MSVD/video_frames/"

input_videos = glob(input_path+"*.avi")

print('input videos', len(input_videos))

def extract_frames(path):
    video_path = path[0]
    print('video', video_path)
    output_path = path[1]
    # extract frames
    command = "ffmpeg -loglevel panic -i " + video_path + " " + output_path + "/f%07d.png -hide_banner"
    os.system(command)

    # Rename all frames for optical flow code
    # frames = os.listdir(output_path)
    # nframes = len(frames)
    #
    # for frame in frames:
    #     frame_path = output_path + '/'
    #     frame_number = int(frame[1:-4], 10)
    #
    #     if frame_number == 1:
    #         name = 'f10.png'
    #         os.rename(frame_path + frame, frame_path + name)
    #     elif frame_number == nframes:
    #         name = 'f' + str(frame_number-1) + '1.png' # e.x f00276.png => f2751.png
    #         os.rename(frame_path + frame, frame_path + name)
    #     else:
    #         name1 = 'f' + str(frame_number-1) + '1.png'
    #         name2 = 'f' + str(frame_number) + '0.png'
    #         os.system('cp ' + frame_path + frame + ' ' + frame_path + name1)
    #         os.system('cp ' + frame_path + frame + ' ' + frame_path + name2)
    #         os.remove(frame_path + frame)


if __name__ == "__main__":

    video_paths = []

    for video in input_videos:
        video_name = (video.split("/")[-1])[:-4]
        if os.path.exists(output_path + video_name):
            continue

        os.mkdir(output_path + video_name)
        video_paths.append([video, output_path+video_name])

        p = Pool(5)
        # extract_frames(video, output_path + video_name)
        p.map(extract_frames, video_paths)
