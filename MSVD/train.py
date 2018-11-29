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
    Output : Caption Tensor of size [batch, #caption/video, #words/caption, word_dim]
    '''
    def __init__(self, train=True):
        if train == True:
            self.caption_features = torch.load(caption_features_tr_path)
            caption_sents = torch.load(caption_tr_path)
        else:
            self.caption_features = torch.load(caption_features_vl_path)
            caption_sents = torch.load(caption_vl_path)

        # Set up caption dictionary for visualisation of predicted sentences
        self.idx2word_dict = {}
        self.idx2word_dict[0] = 'BOS'
        self.idx2word_dict[1] = 'EOS'
        idx = 2
        for vid in caption_sents:
            for capt in captions:
                for word in capt.split(' '):
                    if word not in list(self.idx2word_dict.values())
                        self.idx2word_dict[idx] = word
                        idx += 1

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



# RESNET FEATURES LOADER



# OPTICAL FLOW FEATURES LOADER




# DATALOADER




# NETWORK




# TRAIN NETWORK
