import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Process

input_path = "../../../dataset/MSR-VTT/videos/train/"
output_path = "../../../dataset/MSR-VTT/video_frames/"

input_videos = glob(input_path+"*.mp4")

def extract_frames(video_path, output_path):
    # extract frames
    command = "ffmpeg -loglevel panic -i " + video_path + " " + output_path + "/f%07d.png -hide_banner"
    os.system(command)

    # Rename all frames for optical flow code
    frames = os.listdir(output_path)
    nframes = len(frames)

    for frame in frames:
        frame_path = output_path + '/'
        frame_number = int(frame[1:-4], 10)

        if frame_number == 1:
            name = 'f10.png'
            os.rename(frame_path + frame, frame_path + name)
        elif frame_number == nframes:
            name = 'f' + str(frame_number-1) + '1.png' # e.x f00276.png => f2751.png
            os.rename(frame_path + frame, frame_path + name)
        else:
            name1 = 'f' + str(frame_number-1) + '1.png'
            name2 = 'f' + str(frame_number) + '0.png'
            os.system('cp ' + frame_path + frame + ' ' + frame_path + name1)
            os.system('cp ' + frame_path + frame + ' ' + frame_path + name2)
            os.remove(frame_path + frame)


if __name__ == "__main__":


    for video in tqdm(input_videos):

        video_name = (video.split("/")[-1])[:-4]

        if os.path.exists(output_path + video_name):
            continue

        os.mkdir(output_path + video_name)
        extract_frames(video, output_path + video_name)

        # Extracting frames on multicores.
    #     procs = []
    #     proc = Process(target=extract_frames, args=(video, output_path + video_name))
    #     procs.append(proc)
    #     proc.start()
    #
    # # complete the processes
    # for proc in procs:
    #     proc.join()
