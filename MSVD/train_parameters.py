'''
Data Loader :

1. Each object file as : object/train/video1.pt
    Each pt file is in the format {1:['man':{position:[x1,y1,x2,y2],
    feature:[np array of shape 2048*1]}, 'car':{position:[x1,y1,x2,y2]}
    feature:[np array]],..... 2:[], 3:[],......,100:[] }
2. Optical flow as : optical_flow/train/video1.pt
    Each pt file is in the format {1:[np array of shape 2048], 2:[],....,100:[]}
3. Resnet features as : resnet/train/video1.pt
    Each pt file in the format {1:[np array of shape 2048], 2:[],......,100:[]}
4. Output caption as : captions/train
    One pt file in the format {video1:[cap1, cap2,...capi]} caption_tr.pt
    Each pt file in the format {video1:[cap1 feat size i1*500, cap feat i2*500,..]}
        where i1 = #word per caption 1
    One pt file for vocabulary {word1:word2vec1,word2:word2vec2,......}.

'''
# DATA PATHS
object_features_tr_path = '../../dataset/MSVD/dummy/object/train/'
optical_features_tr_path = '../../dataset/MSVD/dummy/optical/train/'
resnet_features_tr_path = '../../dataset/MSVD/dummy/resnet/train/'
caption_features_tr_path = '../../dataset/MSVD/dummy/caption/train/'
caption_tr_path = '../../dataset/MSVD/dummy/caption/caption_tr.pt'

object_features_vl_path = '../../dataset/MSVD/dummy/object/valid/'
optical_features_vl_path = '../../dataset/MSVD/dummy/optical/valid/'
resnet_features_vl_path = '../../dataset/MSVD/dummy/resnet/valid/'
caption_features_vl_path = '../../dataset/MSVD/dummy/caption/valid/'
caption_vl_path = '../../dataset/MSVD/dummy/caption/caption_vl.pt'

# GLOBAL VARIABLES
video_ids_tr = []
video_ids_vl = []

# GLOBAL HYPER PARAMETERS
num_epoch = 5
batch_size = 10
hidden_size = 1024
num_layers = 1
learning_rate = 0.001
bi_dir = False
