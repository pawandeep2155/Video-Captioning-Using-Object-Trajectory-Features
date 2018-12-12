import torch
import numpy as np
from train_parameters import *
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import os
from glob import glob

# GET VIDEO ID'S
video_ids_tr = os.listdir(caption_tr_path)

video_ids_tr = [item[:-3] for item in video_ids_tr]

video_ids_vl = os.listdir(caption_vl_path)
video_ids_vl = [item[:-3] for item in video_ids_vl]

all_video_ids = video_ids_tr + video_ids_vl

def create_vocab():

    idx2word_dict = {}
    idx2word_dict[0] = 'SOS'
    idx2word_dict[1] = 'EOS'
    idx = 2

    for vid in all_video_ids:
        if vid in video_ids_tr:
            video_cap = torch.load(caption_tr_path + vid + '.pt')
        else:
            video_cap = torch.load(caption_vl_path + vid + '.pt')

        for cap in video_cap[vid]:
            for word in cap.split(' '):
                if word not in list(idx2word_dict.values()):
                    idx2word_dict[idx] = word
                    idx += 1

    return idx2word_dict

def caption_info():
    '''
    max Cap/video and max word/caption
    '''
    max_cap_per_vid = 0
    max_word_per_cap = 0

    for vid in all_video_ids:
        if vid in video_ids_tr:
            video_cap = torch.load(caption_tr_path + vid + '.pt')
        else:
            video_cap = torch.load(caption_vl_path + vid + '.pt')

        max_cap_per_vid = max(max_cap_per_vid, len(video_cap[vid]))

        for cap in video_cap[vid]:
            max_word_per_cap = max(max_word_per_cap, len(cap.split()))

    return max_cap_per_vid, max_word_per_cap

def max_frame_per_video():

    max_frame = 0

    for vid in all_video_ids:
        if vid in video_ids_tr:
            video_cap = torch.load(resnet_features_tr_path + vid + '.pt')
        else:
            video_cap = torch.load(resnet_features_vl_path + vid + '.pt')

        max_frame = max(max_frame, len(video_cap))

    return max_frame

# Change object file from temp.py

def max_object_per_frame():

    max_objects = 0

    for vid in all_video_ids:
        if vid in video_ids_tr:
            video_cap = torch.load(object_features_tr_path + vid + '.pt')
        else:
            video_cap = torch.load(object_features_vl_path + vid + '.pt')

        for frame, _ in video_cap.items():
            max_objects = max(max_objects, len(video_cap[frame]))

    return max_objects


# CAPTION FEATURES LOADER
class Caption_loader:
    '''
    Input : Video id list
    Task : Generate caption/video dict, index to word dictionary, caption tensor
    Dim : Caption Tensor of size [#captions, #words/caption, word_dim]
    '''
    def __init__(self, max_cap_per_vid, max_word_per_cap, train=True):
        if train == True:
            self.caption_features_path = caption_features_tr_path
        else:
            self.caption_features_path = caption_vl_path

        self.max_cap_per_vid = max_cap_per_vid
        self.max_word_per_cap = max_word_per_cap

    def video_instances(self, vids):
        video_inst = []
        for vid in vids:
            video_ftrs = torch.load(self.caption_features_path + vid + '.pt')
            video_inst.append(len(video_ftrs[vid]))

        return video_inst


    def get_tensor(self, vids):
        total_captions = 0
        for vid in vids:
            video_ftrs = torch.load(self.caption_features_path + vid + '.pt')
            total_captions += len(video_ftrs[vid])

        captions = torch.zeros((total_captions, self.max_word_per_cap, word_dim))

        cap_num = 0
        for vid in vids:
            video_ftrs = torch.load(self.caption_features_path + vid + '.pt')
            for cap in video_ftrs[vid]:
                words_in_sent = cap.shape[0]
                captions[cap_num,0:words_in_sent,:] = torch.from_numpy(cap)
                cap_num += 1

        return captions

# OBJECT FEATURES LOADER
class Object_features():
    '''
    Input : Video id list, #caption/video
    Task : Generate object tensor
    Dim : Object Tensor of size [#videos instances = #captions for video id list
                                , #frames, #objects ,object_size]
    '''

    def __init__(self, max_frame, max_object, train=True):
        if train == True:
            self.object_features_path = object_features_tr_path
        else:
            self.object_features_path = object_features_vl_path

        self.max_frm_per_video = max_frame
        self.max_obj_per_frm = max_object

    def get_tensor(self, vids, video_instances):

        total_instances = sum(video_instances)

        object_tensor = torch.zeros((total_instances, self.max_frm_per_video, \
                                    self.max_obj_per_frm, object_dim))

        vd_start_instance = 0
        for i, vid in enumerate(vids):
            object_features = torch.load(self.object_features_path + vid + '.pt')
            for j, frame in enumerate(object_features.values()):
                if len(frame) != 0:
                    for k, obj in enumerate(frame):
                        feature = list(obj.values())[0]['feature']
                        object_tensor[vd_start_instance,j,k] = torch.from_numpy(feature)

            object_tensor[vd_start_instance:vd_start_instance + video_instances[i]] = \
            object_tensor[vd_start_instance].unsqueeze(0). \
            repeat(video_instances[i], 1, 1, 1)

            vd_start_instance += video_instances[i]

        return object_tensor
#
# # RESNET FEATURES LOADER
# class Resnet_features:
#     '''
#     Input : Video id list, #caption/video
#     Task : Generate resnet tensor
#     Dim : Resnet Tensor of size [#videos instances = #captions for video id list
#                                 , #frames, feature_size]
#     '''
#
#     def __init__(self, train=True, all_vids):
#         if train == True:
#             self.resnet_features_path = resnet_features_tr_path
#         else:
#             self.resnet_features_path = resnet_features_vl_path
#
#         self.max_frm_per_video = 0
#         for vid in all_vids:
#             video_features = torch.load(self.resnet_features_path + vid + '.pt')
#             self.max_frm_per_video = max(self.max_frm_per_video, len(list(video_features.keys())))
#             self.feature_size = np.prod(video_features[1].shape)
#
#     def feature_size(self):
#         return self.feature_size
#
#     def get_tensor(self, vids, captions_per_video):
#         video_instances = sum(list(captions_per_video.values()))
#
#         resnet_tensor = torch.zeros(video_instances, self.max_frm_per_video, self.feature_size)
#
#         vd_start_instance = 0
#         for vid in vids:
#             video_features = torch.load(self.resnet_features_path + vid + '.pt')
#             for i,(frame_num,frame_feature) in enumerate(video_features.items()):
#                 resnet_tensor[vd_start_instance,i] = frame_feature
#
#             resnet_tensor[vd_start_instance:vd_start_instance+captions_per_video[vid]] = \
#              resnet_tensor[vd_start_instance].unsqueeze(0).repeat \
#              (vd_start_instance+captions_per_video[vid], max_frm_per_video, feature_size)
#
#              vd_start_instance += captions_per_video[vid]
#
#         return renset_tensor
#
# # OPTICAL FLOW FEATURES LOADER
# class Optical_features:
#     '''
#     Input : Video id list, #caption/video
#     Task : Generate optical tensor
#     Dim : Optical Tensor of size [#videos instances = #captions for video id list
#                                 , #frames, feature_size]
#     '''
#
#     def __init__(self, train=True, all_vids):
#         if train == True:
#             self.optical_features_path = optical_features_tr_path
#         else:
#             self.optical_features_path = optical_features_vl_path
#
#         self.max_frm_per_video = max_frm_per_video
#         self.feature_size = feature_size
#
#                 max_frm_per_video = 0
#                 feature_size = 0
#
#                 # for vid in vids:
#                 #     video_features = torch.load(self.optical_features_path + vid + '.pt')
#                 #     max_frm_per_video = max(max_frm_per_video, len(list(video_features.keys())))
#                 #     feature_size = np.prod(video_features[1].shape)
#
#     def get_tensor(self, vids, captions_per_video):
#         video_instances = sum(list(captions_per_video.values()))
#         max_frm_per_video = 0
#         feature_size = 0
#
#         # for vid in vids:
#         #     video_features = torch.load(self.optical_features_path + vid + '.pt')
#         #     max_frm_per_video = max(max_frm_per_video, len(list(video_features.keys())))
#         #     feature_size = np.prod(video_features[1].shape)
#
#         optical_tensor = torch.zeros(video_instances, self.max_frm_per_video, self.feature_size)
#
#         vd_start_instance = 0
#         for vid in vids:
#             video_features = torch.load(self.optical_features_path + vid + '.pt')
#             for i,(frame_num,frame_feature) in enumerate(video_features.items()):
#                 optical_tensor[vd_start_instance,i] = frame_feature
#
#             optical_tensor[vd_start_instance:vd_start_instance+captions_per_video[vid]] = \
#              optical_tensor[vd_start_instance].unsqueeze(0).repeat \
#              (vd_start_instance+captions_per_video[vid], max_frm_per_video, feature_size)
#
#              vd_start_instance += captions_per_video[vid]
#
#         return optical_tensor
#
# # NETWORK
# class Architecture(nn.Module):
#
#     def __init__(self, hidden_size, num_frames, obj_per_frame, resnet_ftr_size, \
#                  capt_ftr_size, num_layers, bidirectional=False):
#
#         super(Architecture, self).__init__()
#
#         num_directions = 2 is bidirectional else 1
#         video_instances = video_instances
#         self.num_frames = num_frames
#         self.obj_per_frame = obj_per_frame
#         self.resnet_ftr_size = resnet_ftr_size
#
#         obj_ftr_size = resnet_ftr_size
#
#         # attn level1 : among objects per frame
#         self.attn1 = nn.Sequential(nn.Linear(obj_per_frame*obj_ftr_size,obj_per_frame), \
#                                    nn.Softmax(dim=1))
#
#         # attn level2 : among <object,resnet,optical> per frame
#         self.attn2 = nn.Sequential(nn.Linear(3*obj_ftr_size, 3),
#                                    nn.Softmax(dim=1))
#
#         # attn level3 : among frames
#         self.attn3 = nn.Sequential(nn.Linear(num_frames*resnet_ftr_size,num_frames), \
#                                    nn.Softmax(dim=1))
#
#         # Encoder lstm
#         self.elstm = nn.LSTM(input_size=resnet_ftr_size, hidden_size = hidden_size, \
#                              num_layers=num_layers, batch_first=True, dropout=0.5, \
#                              bidirectional=bidirectional)
#
#         # Decoder lstm
#         self.dlstm = nn.LSTM(input_size=capt_ftr_size, hidden_size = hidden_size, \
#                              num_layers = num_layers, batch_first=True, dropout=0.5, \
#                              bidirectional=bidirectional)
#
#         self.caption = nn.Sequential(nn.Linear(num_directions*hidden_size, capt_ftr_size), \
#                                     nn.Softmax(dim=1))
#
#
#     def forward(self, video_instances, resnet_ftrs, optical_ftrs, object_ftrs, caption_ftrs):
#
#         #################### Attention Level 1 #################################
#
#         attn1 = self.attn1(object_ftrs)
#         # [700, 100, 4]
#         attn1 = attn1.view(video_instances*self.num_frames, self.obj_per_frame, 1)
#         # [70000, 4, 1]
#         object_ftrs = object_ftrs.view(video_instances*self.num_frames, \
#                         self.resnet_ftr_size, self.obj_per_frame)
#         # [700,100,4,2048] to [70000,2048,4] for bmm
#         object_attended = torch.bmm(object_ftrs, attn1)
#         # [70000, 2048, 4] and [70000, 4, 1] to [70000, 2048, 1]
#         object_attended = object_attended.view(video_instances, self.num_frames, self.resnet_ftr_size)
#         # [70000, 2048, 1] to [700, 100, 2048]
#
#         ###################### Attention Level 2 ###############################
#
#         all_features = torch.cat((object_attended, resnet_ftrs, optical_ftrs), 2)
#         # [700, 100, 3*2048]
#         attn2 = self.attn2(all_features)
#         # [700, 100, 3]
#         attn2 = attn2.view(video_instances*self.num_frames, 3, 1)
#         # [70000, 3, 1]
#
#         all_features = all_features.view(video_instances*self.num_frames, \
#                                          self.resnet_ftr_size, self.obj_per_frame)
#         # [700, 100, 3*2048] to [70000, 2048, 3]
#         features_attended = torch.bmm(all_features, attn2)
#         # [70000, 2048, 3] and [70000, 3, 1] to [70000, 2048, 1]
#         features_attended = features_attended.view(video_instances, self.num_frames, self.resnet_ftr_size)
#         # [70000, 2048, 1] to [700, 100, 2048]
#
#         ##################### Attention Level 3 ################################
#
#         video_feature = features_attended.view(video_instances, self.num_frames* self.resnet_ftr_size)
#         # [700, 100, 2048] to [700, 100*2048]
#
#         attn3 = self.attn3(video_feature)
#         # [700, 100]
#         attn3 = attn3.unsqueeze(2).repeat(video_instances, self.num_frames, self.resnet_ftr_size)
#         # [700, 100] to [700, 100, 1] to [700, 100, 2048]
#
#         video_attended = video_feature * attn3
#         # [700, 100, 2048]
#
#         ######################## Encoder LSTM ##################################
#         encoder_out, encoder_hidden = self.elstm(video_attended)
#         # encoder out [700,100, num_dir*hidden_size] , encoder_hidden <[700, num_layer*num_dir, hidden_size], same>
#
#         ######################### Decoder LSTM #################################
#         decoder_out, _ = self.dlstm(caption_ftrs, encoder_hidden)
#         # decoder out [700,10, num_dir*hidden_size] , encoder_hidden
#         # <[700, num_layer*num_dir, hidden_size], same> where 10 is max words per caption
#         caption = self.caption(decoder_out)
#         # [700, 10, 300] where 300 is the word size
#
#         return caption, self.attn1, self.attn2, self.attn3
#
# # TRAIN NETWORK
# def train(model, captions, objects, optical_flow, resnet, n_iters, lr_rate, batch_size):
#
#     model_optimizer = optim.Adam(model.parameters(), lr=lr_rate)
#     criterion = nn.MSELoss()
#
#     for epoch in tqdm(range(n_iters)):
#
#         loss = 0
#         data_iters = math.ceil(len(video_ids_tr) / batch_size)
#
#         for i in range(data_iters):
#
#             start = i*batch_size
#             end = min((i+1)*batch_size, len(video_ids_tr))
#             vids = video_ids_tr[start:end]
#
#             cap_gtruth = captions.get_tensor(vids)
#             obj_ftrs = objects.get_tensor(vids)
#             opt_ftrs = optical_flow.get_tensor(vids)
#             res_ftrs = resnet.get_tensor(vids)
#
#             video_instns = cap_ftr.shape[0]
#
#             caption_pred, attn1, attn2, attn3 = model(video_instns, res_ftrs, opt_ftrs, obj_ftrs, cap_gtruth)
#
#             loss += criterion(cap_gtruth, caption_pred)
#
#             loss.backward()
#             model_optimizer.step()
#
#         print(loss)
#
#
# def evaluate(model, captions, objects, optical_flow, resnet, n_iters, lr_rate, batch_size):
#
#
# def show_attn():
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
if __name__ == "__main__":

    vocabulary = create_vocab()
    max_cap_per_vid, max_word_per_cap = caption_info()

    captions = Caption_loader(max_cap_per_vid, max_word_per_cap, train=True)
    caption_tensor = captions.get_tensor(['vid1', 'vid2'])
    video_inst = captions.video_instances(['vid1', 'vid2'])

    print('caption tensor', caption_tensor.shape)
    print('video instances', video_inst)

    max_frame = max_frame_per_video()
    max_objects = max_object_per_frame()

    objects = Object_features(max_frame, max_objects, train=True)

    object_tensor = objects.get_tensor(['vid1', 'vid2'], video_inst)

    print('object tensor', object_tensor.shape)

#     optical_flow = Optical_features(train=True,all_vids=all_video_ids)
#     resnet = Resnet_features(train=True, all_vids=all_video_ids)
#
#     num_frames = objects.frame_per_video()
#     obj_per_frame = objects.object_per_frame()
#     resnet_feature_size = resnet.feature_size()
#     cap_feature_size = captions.dictionary_size()
#
#     model = Architecture(hidden_size, num_frames, obj_per_frame, resnet_feature_size, \
#                         cap_feature_size, num_layers, bi_dir).to(device)
#
#     train(model, captions, objects, optical_flow, resnet, num_epoch, learning_rate, batch_size)
#
#

















# comment
