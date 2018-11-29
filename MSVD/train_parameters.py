'''
Data Loader :

1. Each object file as : object/train/video1.pt
    Each pt file is in the format {1:['man':{position:[x1,y1,x2,y2],
    feature:[np array of shape 2048*1]}, 'car':{position:[x1,y1,x2,y2]}
    feature:[np array]], 2:[], 3:[],......,100:[] }
2. Optical flow as : optical_flow/train/video1.pt
    Each pt file is in the format {1:[np array of shape 2048], 2:[],....,100:[]}
3. Resnet features as : resnet/train/video1.pt
    Each pt file in the format {1:[np array of shape 2048], 2:[],......,100:[]}
4. Output caption as : captions/train
    A pt file in the format {video1:[cap1, cap2,...capi], video2:[cap..],..videoi:[]}
    A pt file in the format {video1:[cap1 feat size i1*500, cap feat i2*500,..], video2:[],...}
        where i1 = # captions per video

'''
# DATA PATHS
object_features_tr_path = '../../dataset/MSVD/object/train/'
optical_features_tr_path = '../../dataset/MSVD/optical/train/'
resnet_features_tr_path = '../../dataset/MSVD/resnet/train/'
caption_features_tr_path = '../../dataset/MSVD/caption/train/cap_feat_tr.pt'
caption_tr_path = '../../dataset/MSVD/captions/train/caption_tr.pt'

object_features_vl_path = '../../dataset/MSVD/object/valid/'
optical_features_vl_path = '../../dataset/MSVD/optical/valid/'
resnet_features_vl_path = '../../dataset/MSVD/resnet/valid/'
caption_features_vl_path = '../../dataset/MSVD/caption/valid/cap_feat_vl.pt'
caption_vl_path = '../../datset/MSVD/captions/valid/caption_vl.pt'

# GLOBAL VARIABLES
video_ids_tr = []
video_ids_vl = []

# GLOBAL HYPER PARAMETERS
epoch = 5
batch = 10
hidden_size = 1024
bi_dir = False
