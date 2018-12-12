import torch
import numpy as np
from train_parameters import *
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

# GET VIDEO ID'S
for (key, _) in (torch.load(caption_tr_path).items()):
    video_ids_tr.append(key)

for (key, _) in (torch.load(caption_vl_path).items()):
    video_ids_vl.append(key)

all_video_ids = video_ids_tr + video_ids_vl

# CAPTION FEATURES LOADER
class Caption_loader:
    '''
    Input : Video id list
    Task : Generate caption/video dict, index to word dictionary, caption tensor
    Dim : Caption Tensor of size [#captions, #words/caption, word_dim]
    '''
    def __init__(self, train=True, all_vids):
        if train == True:
            caption_features_path = caption_features_tr_path
            caption_sents_path = caption_tr_path
        else:
            caption_features_path = caption_vl_path
            caption_sents_path = caption_vl_path

        #  dictionary maintaining #captions per video.
        self.capt_per_video = {}

        # Set up caption dictionary for visualisation of predicted sentences
        self.idx2word_dict = {}
        self.idx2word_dict[0] = 'BOS'
        self.idx2word_dict[1] = 'EOS'
        idx = 2

        for vid in all_vids:
            video_cap = torch.load(caption_sents_path + vid + '.pt')
            for cap in video_cap[vid]:
                for word in cap.split(' '):
                    if word not in list(self.idx2word_dict.values())
                        self.idx2word_dict[idx] = word
                        idx += 1

        # Check max words per caption in all videos.
        self.max_word_per_sent = 0
        for vid in all_vids:
            video_features = torch.load(caption_features_path + vid + '.pt')
            for cap in video_features[vid]:
                self.word_size = cap.shape[1]
                self.max_word_per_sent = max(self.max_word_per_sent, cap.shape[0])


    def get_capt_per_video(self):
        return self.capt_per_video

    def get_tensor(self, vids):
        total_captions = 0
        for vid in vids:
            for cap in vid:
                total_captions += 1

        captions = torch.zeros((total_captions, self.max_word_per_sent, self.word_size))

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
#
#
# # OBJECT FEATURES LOADER
# class Object_features():
#     '''
#     Input : Video id list, #caption/video
#     Task : Generate object tensor
#     Dim : Object Tensor of size [#videos instances = #captions for video id list
#                                 , #frames, #objects ,object_size]
#     '''
#
#     def __init__(self, train=True, all_vids):
#         if train == True:
#             self.object_features_path = object_features_tr_path
#         else:
#             self.object_features_path = object_features_vl_path
#
#         self.max_frm_per_video = 0
#         self.max_obj_per_frm = 0
#
#         for i, vid in enumerate(all_vids):
#             object_features = torch.load(self.object_features_path + vid + '.pt')
#             self.max_frm_per_video = max(self.max_frm_per_video, len(list(object_features.keys()))
#
#             for frm in list(object_features.values):
#                 if len(frm) != 0: # object is detected in the frame.
#                     self.object_size = np.prod(frm[0]['feature'].shape)
#                     self.max_obj_per_frm = max(self.max_obj_per_frm, len(frm))
#
#     def frame_per_video(self):
#         return self.max_frm_per_video
#
#     def object_per_frame(self):
#         return self.max_obj_per_frm
#
#     def get_tensor(self, vids, captions_per_video):
#         video_instances = sum(list(captions_per_video.values()))
#
#         object_tensor = torch.zeros((video_instances, self.max_frm_per_video, \
#                                     self.max_obj_per_frm, self.object_size))
#
#         vd_start_instance = 0
#         for vid in vids:
#             object_features = torch.load(self.object_features_path + vid + '.pt')
#             for j, frame in enumerate(list(object_features.values())):
#                 if len(frame) != 0:
#                     for k, (_, obj) in enumerate(list(frame.items()):
#                         object_tensor[vd_start_instance,j,k] = torch.from_numpy(obj['feature'])
#
#             object_tensor[vd_start_instance:vd_start_instance + captions_per_video[vid]] = \
#             object_tensor[vd_start_instance].unsqueeze(0). \
#             repeat(captions_per_video[vid],max_frm_per_video,max_obj_per_frm,object_size)
#
#             vd_start_instance += captions_per_video[vid]
#
#         return object_tensor
#
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
# if __name__ == "__main__":
#
    captions = Caption_loader(train=True, all_vids=all_video_ids)
    



#     objects = Object_features(train=True, all_vids=all_video_ids)
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
