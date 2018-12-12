import cv2
import os
from multiprocessing import pool
import torch
from tqdm import tqdm
from skvideo.io import vread
from PIL import Image

input_path = '../../../dataset/MSVD/object_detection/'
video_path = '../../../dataset/MSVD/videos/'
output_path = '../../../dataset/MSVD/object_crop1/'

def load_video(vid_path):
    try:
        video = vread(vid_path)
        return video
    except:
        log_fail = open('fail_crop1.txt','a')
        log_fail.write(vid_path + '\n')
        log_fail.close()
        return []

def crop_object(image, obj_name, obj_cordnts):
    x1 = int(obj_cordnts[0])
    y1 = int(obj_cordnts[1])
    x2 = int(obj_cordnts[2])
    y2 = int(obj_cordnts[3])

    return image[y1:y2, x1:x2]

if __name__ == "__main__":
    video_ids = os.listdir(input_path)
    total_videos = len(video_ids)

    for i, vid in enumerate(tqdm(video_ids)):

        # print('id', vid, 'video num',i+1,'left',total_videos-i-1)

        if os.path.exists(output_path + vid[:-3] +'_done'):
            continue

        video_objects = torch.load(input_path + vid)
        video = load_video(video_path + vid[:-3] + '.avi')

        if len(video) == 0: # unable to read video.
            continue

        os.mkdir(output_path + vid[:-3])

        for frm_num, objects in video_objects.items():

            os.mkdir(output_path + vid[:-3] + '/' + str(frm_num))

            if len(objects) != 0:

                for obj_name, obj_cordnts in objects.items(): # object is dictionary with key as object name, value as object cordinates

                    croped_object = crop_object(video[frm_num-1], obj_name, obj_cordnts)
                    if not os.path.exists(output_path + vid[:-3] + '/' + str(frm_num) + '/'):
                        os.mkdir(output_path + vid[:-3] + '/' + str(frm_num) + '/')

                    img = Image.fromarray(croped_object)
                    # cv2.imwrite(output_path + vid[:-4] + '/' + str(frm_num) + '/' + obj_name + '.png', croped_object)
                    img.save(output_path + vid[:-3] + '/' + str(frm_num) + '/' + obj_name + '.png')

            else:
                if not os.path.exists(output_path + vid[:-3] + '/' + str(frm_num) + '/'):
                    os.mkdir(output_path + vid[:-3] + '/' + str(frm_num) + '/')


        os.rename(output_path + vid[:-3], output_path + vid[:-3] +'_done')
