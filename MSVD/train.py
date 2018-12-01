import torch
import numpy as np
from train_parameters import *

# GET VIDEO ID'S
for i,(key, _) in range(torch.load(caption_features_tr_path).items()):
    video_ids_tr.append(key)

for i,(key, _) in range(torch.load(caption_features_vl_path).items()):
    video_ids_vl.append(key)

# CAPTION FEATURES LOADER
class Caption_loader:
    '''
    Input : Video id list
    Task : Generate caption/video dict, index to word dictionary, caption tensor
    Dim : Caption Tensor of size [#captions, #words/caption, word_dim]
    '''
    def __init__(self, train=True):
        if train == True:
            self.caption_features = torch.load(caption_features_tr_path)
            caption_sents = torch.load(caption_tr_path)
        else:
            self.caption_features = torch.load(caption_features_vl_path)
            caption_sents = torch.load(caption_vl_path)

        #  dictionary maintaining #captions per video.
        self.capt_per_video = {}

        # Set up caption dictionary for visualisation of predicted sentences
        self.idx2word_dict = {}
        self.idx2word_dict[0] = 'BOS'
        self.idx2word_dict[1] = 'EOS'
        idx = 2
        for vid, captions in caption_sents.items():
            self.capt_per_video[vid] = len(captions)
            for capt in captions:
                for word in capt.split(' '):
                    if word not in list(self.idx2word_dict.values())
                        self.idx2word_dict[idx] = word
                        idx += 1

    def get_capt_per_video(self):
        return self.capt_per_video

    def get_tensor(self, vids):
        total_captions = 0
        max_word_per_sent = 0
        for vid in vids:
            for cap in vid:
                total_captions += 1
                word_size = cap.shape[1]
                max_word_per_sent = max(max_word_per_sent, cap.shape[0])

        captions = torch.zeros((total_captions, max_word_per_sent, word_size))

        cap_num = 0
        for vid in vids:
            for cap in vid:
                words_in_sent = cap.shape[0]
                captions[cap_num,0:words_in_sent-1,:] = torch.from_numpy(cap)
                cap_num += 1

        return captions

    def dictionary_size(self):
        return len(self.idx2word_dict)

    def get_word(self, idx):
        return self.idx2word_dict[idx]


# OBJECT FEATURES LOADER
class Object_features():
    '''
    Input : Video id list, #caption/video
    Task : Generate object tensor
    Dim : Object Tensor of size [#videos instances = #captions for video id list
                                , #frames, #objects ,object_size]
    '''

    def __init__(self, train=True):
        if train == True:
            self.object_features_path = object_features_tr_path
        else:
            self.object_features_path = object_features_vl_path

    def get_tensor(self, vids, captions_per_video):
        video_instances = sum(list(captions_per_video.values()))
        max_frm_per_video = 0
        max_obj_per_frm = 0
        object_size = 0

        for i, vid in enumerate(vids):
            object_features = torch.load(self.object_features_path + vid + '.pt')
            max_frm_per_video = max(max_frm_per_video, len(list(object_features.keys()))

            for frm in list(object_features.values):
                if len(frm) != 0: # object is detected in the frame.
                    object_size = np.prod(frm[0]['feature'].shape)
                    max_obj_per_frm = max(max_obj_per_frm, len(frm))

        object_tensor = torch.zeros((video_instances, max_frm_per_video, \
                                    max_obj_per_frm, object_size))

        vd_start_instance = 0
        for vid in vids:
            object_features = torch.load(self.object_features_path + vid + '.pt')
            for j, frame in enumerate(list(object_features.values())):
                if len(frame) != 0:
                    for k, (_, obj) in enumerate(list(frame.items()):
                        object_tensor[vd_start_instance,j,k] = torch.from_numpy(obj['feature'])

            object_tensor[vd_start_instance:vd_start_instance + captions_per_video[vid]] = \
            object_tensor[vd_start_instance].unsqueeze(0). \
            repeat(captions_per_video[vid],max_frm_per_video,max_obj_per_frm,object_size)

            vd_start_instance += captions_per_video[vid]

        return object_tensor


# RESNET FEATURES LOADER
class Resnet_features:
    '''
    Input : Video id list, #caption/video
    Task : Generate resnet tensor
    Dim : Resnet Tensor of size [#videos instances = #captions for video id list
                                , #frames, feature_size]
    '''

    def __init__(self, train=True):
        if train == True:
            self.resnet_features_path = resnet_features_tr_path
        else:
            self.resnet_features_path = resnet_features_vl_path

    def get_tensor(self, vids, captions_per_video):
        video_instances = sum(list(captions_per_video.values()))
        max_frm_per_video = 0
        feature_size = 0

        for vid in vids:
            video_features = torch.load(self.resnet_features_path + vid + '.pt')
            max_frm_per_video = max(max_frm_per_video, len(list(video_features.keys())))
            feature_size = np.prod(video_features[1].shape)

        resnet_tensor = torch.zeros(video_instances, max_frm_per_video, feature_size)

        vd_start_instance = 0
        for vid in vids:
            video_features = torch.load(self.resnet_features_path + vid + '.pt')
            for i,(frame_num,frame_feature) in enumerate(video_features.items()):
                resnet_tensor[vd_start_instance,i] = frame_feature

            resnet_tensor[vd_start_instance:vd_start_instance+captions_per_video[vid]] = \
             resnet_tensor[vd_start_instance].unsqueeze(0).repeat \
             (vd_start_instance+captions_per_video[vid], max_frm_per_video, feature_size)

             vd_start_instance += captions_per_video[vid]

        return renset_tensor

# OPTICAL FLOW FEATURES LOADER
class Optical_features:
    '''
    Input : Video id list, #caption/video
    Task : Generate optical tensor
    Dim : Optical Tensor of size [#videos instances = #captions for video id list
                                , #frames, feature_size]
    '''

    def __init__(self, train=True):
        if train == True:
            self.optical_features_path = optical_features_tr_path
        else:
            self.optical_features_path = optical_features_vl_path

    def get_tensor(self, vids, captions_per_video):
        video_instances = sum(list(captions_per_video.values()))
        max_frm_per_video = 0
        feature_size = 0

        for vid in vids:
            video_features = torch.load(self.optical_features_path + vid + '.pt')
            max_frm_per_video = max(max_frm_per_video, len(list(video_features.keys())))
            feature_size = np.prod(video_features[1].shape)

        optical_tensor = torch.zeros(video_instances, max_frm_per_video, feature_size)

        vd_start_instance = 0
        for vid in vids:
            video_features = torch.load(self.optical_features_path + vid + '.pt')
            for i,(frame_num,frame_feature) in enumerate(video_features.items()):
                optical_tensor[vd_start_instance,i] = frame_feature

            optical_tensor[vd_start_instance:vd_start_instance+captions_per_video[vid]] = \
             optical_tensor[vd_start_instance].unsqueeze(0).repeat \
             (vd_start_instance+captions_per_video[vid], max_frm_per_video, feature_size)

             vd_start_instance += captions_per_video[vid]

        return optical_tensor



# NETWORK




# TRAIN NETWORK
