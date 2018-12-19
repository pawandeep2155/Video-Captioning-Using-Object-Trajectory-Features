import torch
import numpy as np
from train_parameters import *
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from glob import glob
import random
import math
from tensorboard_logger import configure, log_value

configure("../../dataset/MSVD/tensorboard/run-1")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# GET VIDEO ID'S
video_ids_tr = os.listdir(caption_tr_path)
video_ids_tr = [item[:-3] for item in video_ids_tr]

video_ids_vl = os.listdir(caption_vl_path)
video_ids_vl = [item[:-3] for item in video_ids_vl]

all_video_ids = video_ids_tr + video_ids_vl

def caption_info():
    print('caption info...')
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
    print('max object per frame...')
    max_objects = 0

    for vid in all_video_ids:
        if vid in video_ids_tr:
            video_cap = torch.load(object_features_tr_path + vid + '.pt')
        else:
            video_cap = torch.load(object_features_vl_path + vid + '.pt')

        for frame, _ in video_cap.items():
            max_objects = max(max_objects, len(video_cap[frame]))

    return max_objects

def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


# CAPTION FEATURES LOADER
class Caption_loader:
    '''
    Input : Video id list
    Task : Generate caption list [(10,300), (20, 300)]
    Dim : Caption Tensor of size [#captions, #words/caption, word_dim]
    '''
    def __init__(self, train=True):
        if train == True:
            self.caption_features_path = caption_features_tr_path
        else:
            self.caption_features_path = caption_features_vl_path

    def get_item(self, vids):

        captions = []

        for vid in vids:
            video_ftrs = torch.load(self.caption_features_path + vid + '.pt')
            for cap in video_ftrs[vid]:
                captions.append(torch.from_numpy(cap))

        return captions

# OBJECT FEATURES LOADER
class Object_features():
    '''
    Input : Video id list, #caption/video
    Task : Generate object list [[(4, 2048), (5, 2048)], [(6, 2048), (7, 2048)]]
    '''

    def __init__(self, train=True):
        if train == True:
            self.object_features_path = object_features_tr_path
        else:
            self.object_features_path = object_features_vl_path


    def get_item(self, vids):

        objects = []

        for vid in vids:
            object_features = torch.load(self.object_features_path + vid + '.pt')
            video_frames = []

            for frame in object_features.values():
                if len(frame) == 0:
                    video_frames.append(torch.zeros((1,object_dim)))
                    continue

                frame_tensor = torch.zeros((len(frame), object_dim))

                for i, obj in enumerate(frame):
                    feature = list(obj.values())[0]['feature']
                    frame_tensor[i] = torch.from_numpy(feature)

                video_frames.append(frame_tensor)

            objects.append(video_frames)

        return objects

# RESNET FEATURES LOADER
class Resnet_features:
    '''
    Input : Video id list
    Task : Generate resnet list [(100,2048), (200, 2048)]
    '''

    def __init__(self, train=True):
        if train == True:
            self.resnet_features_path = resnet_features_tr_path
        else:
            self.resnet_features_path = resnet_features_vl_path

        self.max_frame = max_frame

    def get_item(self, vids):
        resnet = []

        for vid in vids:
            video_features = torch.load(self.resnet_features_path + vid + '.pt')
            resnet_tensor = torch.zeros((len(video_features), resnet_dim))

            for j,(_, frame_feature) in enumerate(video_features.items()):
                resnet_tensor[j] = torch.from_numpy(frame_feature)

            resnet.append(resnet_tensor)

        return resnet

# OPTICAL FLOW FEATURES LOADER
class Optical_features:
    '''
    Input : Video id list
    Task : Generate optical list [(100,2048), (200, 2048)]
    '''

    def __init__(self, train=True):
        if train == True:
            self.optical_features_path = optical_features_tr_path
        else:
            self.optical_features_path = optical_features_vl_path

    def get_tensor(self, vids):

        optical = []

        for vid in vids:
            video_features = torch.load(self.optical_features_path + vid + '.pt')
            optical_tensor = torch.zeros(len(video_features), optical_dim)

            for j,(_,frame_feature) in enumerate(video_features.items()):
                optical_tensor[j] = torch.from_numpy(frame_feature)

            optical.append(optical_tensor)

        return optical

def pad_tensor(vec, pad, dim, max_cap):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    vec = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    vec = vec.repeat(max_cap,1,1)

    print('vec', vec.shape)
    return vec

class PadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_collate(self, batch):
        # find longest sequence
        max_lenx = max(map(lambda x: x[0].shape[self.dim], batch))
        max_leny = max(map(lambda x: x[1].shape[self.dim], batch))

        # pad according to max_len
        batchx = list(map(lambda x: pad_tensor(x[0], pad=max_lenx, dim=self.dim, max_cap=40), batch))
        batchy = list(map(lambda x: pad_tensor(x[1], pad=max_leny, dim=self.dim, max_cap=40), batch))

        xs = torch.cat(batchx, dim=0)
        ys = torch.stack(batchy)

        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


class VideoDataset(DataLoader):
    def __init__(self, max_cap_per_vid, caption, objects, optical, resnet):
        self.max_cap_per_vid = max_cap_per_vid

        self.caption = caption
        self.object = objects
        self.optical = optical
        self.resnet = resnet

    def __len__(self):
        return batch_size * self.max_cap_per_vid

    def __getitem__(self, vids):
        caption_list = self.caption.get_item(vids)
        object_list = self.object.get_item(vids)
        optical_list = self.optical.get_item(vids)
        resnet_list = self.resnet.get_item(vids)

        return object_list, optical_list, resnet_list, caption_list


# NETWORK
class Attention(nn.Module):

    def __init__(self, num_frames, obj_per_frame, bi_dir):

        super(Attention, self).__init__()
        print('num_frames', num_frames)
        print('object per frame', obj_per_frame)

        print('attention init')
        self.num_frames = num_frames
        self.obj_per_frame = obj_per_frame

        self.num_directions = 2 if bi_dir else 1

        # attn level1 : among objects per frame
        self.attn1 = nn.Sequential(nn.Linear(self.obj_per_frame*object_dim,self.obj_per_frame), \
                                   nn.Softmax(dim=1))
        print('attn1')
        # attn level2 : among <object,resnet,optical> per frame
        self.attn2 = nn.Sequential(nn.Linear(3*resnet_dim, 3),
                                   nn.Softmax(dim=1))
        print('attn2')
        # attn level3 : among frames
        self.attn3 = nn.Sequential(nn.Linear(self.num_frames*resnet_dim,self.num_frames), \
                                   nn.Softmax(dim=1))
        print('attn3')

    def forward(self, video_instances, resnet_ftrs, optical_ftrs, object_ftrs):

        #################### Attention Level 1 #################################
        total_instances = sum(video_instances)

        attn1 = self.attn1(object_ftrs)
        # [700, 100, 4, 4]
        attn1 = attn1.view(total_instances*self.num_frames, self.obj_per_frame, 1)
        # [70000, 4, 1]
        object_ftrs = object_ftrs.view(total_instances*self.num_frames, \
                        resnet_dim, self.obj_per_frame)
        # [700,100,4,2048] to [70000,2048,4] for bmm
        object_attended = torch.bmm(object_ftrs, attn1)
        # [70000, 2048, 4] and [70000, 4, 1] to [70000, 2048, 1]
        object_attended = object_attended.view(total_instances, self.num_frames, resnet_dim)
        # [70000, 2048, 1] to [700, 100, 2048]

        ###################### Attention Level 2 ###############################

        all_features = torch.cat((object_attended, resnet_ftrs, optical_ftrs), 2)
        # [700, 100, 3*2048]
        attn2 = self.attn2(all_features)
        # [700, 100, 3]
        attn2 = attn2.view(total_instances*self.num_frames, 3, 1)
        # [70000, 3, 1]

        all_features = all_features.view(total_instances*self.num_frames, \
                                         resnet_dim, 3)
        # [700, 100, 3*2048] to [70000, 2048, 3]
        features_attended = torch.bmm(all_features, attn2)
        # [70000, 2048, 3] and [70000, 3, 1] to [70000, 2048, 1]
        features_attended = features_attended.view(total_instances, self.num_frames, resnet_dim)
        # [70000, 2048, 1] to [700, 100, 2048]

        ##################### Attention Level 3 ################################

        video_feature = features_attended.view(total_instances, self.num_frames* resnet_dim)
        # [700, 100, 2048] to [700, 100*2048]

        attn3 = self.attn3(video_feature)
        # [700, 100]
        attn3 = attn3.unsqueeze(2).repeat(1, 1, resnet_dim)
        # [700, 100] to [700, 100, 1] to [700, 100, 2048]

        video_feature = video_feature.view(total_instances, self.num_frames, resnet_dim)
        video_attended = video_feature * attn3
        # [700, 100, 2048]
        return video_attended, self.attn1, self.attn2, self.attn3


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, bidirectional=False):

        super(Encoder, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Encoder lstm
        self.elstm = nn.LSTM(input_size=resnet_dim, hidden_size = self.hidden_size, \
                             num_layers=self.num_layers, batch_first=True, \
                             bidirectional=bidirectional)

    def forward(self, video_tensor, encoder_hidden):
        _, encoder_hidden = self.elstm(video_tensor, encoder_hidden)
        # output [1, 1, 2048], hidden [1, numlayer*num_dir, hidden_size]

        return encoder_hidden

    def init_hidden(self):
        return (torch.zeros((1, self.num_layers*self.num_directions, self.hidden_size))).to(device), \
               (torch.zeros((1, self.num_layers*self.num_directions, self.hidden_size))).to(device)


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, bidirectional=False):

        super(Decoder, self).__init__()

        num_directions = 2 if bidirectional else 1

        # Decoder lstm
        self.dlstm = nn.LSTM(input_size=word_dim, hidden_size = hidden_size, \
                             num_layers = num_layers, batch_first=True, \
                             bidirectional=bidirectional)

        self.caption = nn.Sequential(nn.Linear(num_directions*hidden_size, word_dim), \
                                    nn.Softmax(dim=1))


    def forward(self, word_tensor, encoder_hidden):

        decoder_out, _ = self.dlstm(word_tensor, encoder_hidden)
        decoder_out = self.caption(decoder_out)
        # output [1, 1, 300]

        return decoder_out


# TRAIN NETWORK
def train(attention, encoder, decoder, captions, objects, optical_flow, resnet, \
          objects_vl, resnet_vl, optical_vl, captions_vl, n_iters, lr_rate, \
          batch_size, dec_max_time_step):

    attention_optimizer = optim.Adam(attention.parameters(), lr=learning_rate)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    for epoch in tqdm(range(n_iters)):

        # Train mode
        attention = attention.train()
        encoder = encoder.train()
        decoder = decoder.train()

        loss = 0
        data_iters = math.ceil(len(video_ids_tr) / batch_size)

        for i in range(data_iters):

            start = i*batch_size
            end = min((i+1)*batch_size, len(video_ids_tr))
            vids = video_ids_tr[start:end]

            caption_tensor = captions.get_tensor(vids).to(device)
            video_inst = captions.video_instances(vids)

            object_tensor = objects.get_tensor(vids, video_inst).to(device)
            optical_tensor = optical_flow.get_tensor(vids, video_inst).to(device)
            resnet_tensor = resnet.get_tensor(vids, video_inst).to(device)

            video_attended, _, _, _ = attention(video_inst, resnet_tensor, optical_tensor, object_tensor)

            for i in range(sum(video_inst)):

                encoder_hidden = encoder.init_hidden()

                for frame_num in range(max_frame): # Run Encoder for one video.
                    frame = video_attended[i, frame_num].view(1, 1, resnet_dim)
                    encoder_hidden = encoder(frame, encoder_hidden)

                # Run Decoder for one sentence
                use_teacher_forcing = True if random.random() < teacher_force_ratio else False
                word_tensor = torch.zeros((1,1,word_dim)).to(device) # SOS

                if use_teacher_forcing:
                    # Decoder input is ground truth
                    for t in range(dec_max_time_step):
                        decoder_out = decoder(word_tensor, encoder_hidden)
                        word_ground_truth = caption_tensor[i,t].unsqueeze(0).unsqueeze(0)

                        loss += criterion(decoder_out, word_ground_truth)
                        word_tensor = word_ground_truth

                else:
                    # Decoder input is previous predicted word
                    for t in range(dec_max_time_step):
                        decoder_out = decoder(word_tensor, encoder_hidden)
                        word_ground_truth = caption_tensor[i,t].unsqueeze(0).unsqueeze(0)

                        loss += criterion(decoder_out, word_ground_truth)
                        word_tensor = decoder_out

            loss.backward()

            attention_optimizer.step()
            encoder_optimizer.step()
            decoder_optimizer.step()

        log_value('Training Loss', loss, epoch)

        # Save model parameters
        params = {'attention':attention.state_dict(), \
                  'encoder':encoder.state_dict(), 'decoder':decoder.state_dict()}
        torch.save(params, model_params_path + str(epoch) + '.pt')

        # Validation Loss, Bleu scores etc. after each epoch
        attention = attention.eval()
        encoder = encoder.eval()
        decoder = decoder.eval()
        validate(epoch, attention, encoder, decoder, captions_vl, objects_vl, optical_vl, resnet_vl, batch_size, dec_max_time_step)

def validate(tr_epoch, attention, encoder, decoder, captions, objects, optical_flow, resnet, batch_size,dec_max_time_step):

    criterion = nn.MSELoss()
    data_iters = math.ceil(len(video_ids_vl) / batch_size)

    loss = 0

    for batch_num in tqdm(range(data_iters)):

        start = batch_num*batch_size
        end = min((batch_num+1)*batch_size, len(video_ids_vl))
        vids = video_ids_vl[start:end]

        caption_tensor = captions.get_tensor(vids).to(device)
        video_inst = captions.video_instances(vids)

        object_tensor = objects.get_tensor(vids, video_inst).to(device)
        optical_tensor = optical_flow.get_tensor(vids, video_inst).to(device)
        resnet_tensor = resnet.get_tensor(vids, video_inst).to(device)

        video_attended, _, _, _ = attention(video_inst, resnet_tensor, optical_tensor, object_tensor)

        for i in range(sum(video_inst)):

            encoder_hidden = encoder.init_hidden()

            for frame_num in range(max_frame): # Run Encoder for one video.
                frame = video_attended[i, frame_num].view(1, 1, resnet_dim)
                encoder_hidden = encoder(frame, encoder_hidden)

            word_tensor = torch.zeros((1,1,word_dim)).to(device) # SOS

            # Decoder input is previous predicted word
            for t in range(dec_max_time_step):
                decoder_out = decoder(word_tensor, encoder_hidden)
                word_ground_truth = caption_tensor[i,t].unsqueeze(0).unsqueeze(0)

                loss += criterion(decoder_out, word_ground_truth)
                word_tensor = decoder_out

    log_value('Validation Loss', loss, tr_epoch)

if __name__ == "__main__":

    # Dataset info.
    vocabulary = create_vocab()
    max_cap_per_vid, max_word_per_cap = caption_info()
    max_frame = max_frame_per_video()
    max_objects = max_object_per_frame()

    print('dataloader starting...')

    # Data loader.
    objects = Object_features(max_frame, max_objects, train=True)
    resnet = Resnet_features(max_frame, train=True)
    optical = Optical_features(max_frame, train=True)
    captions = Caption_loader(max_cap_per_vid, max_word_per_cap, train=True)

    objects_vl = Object_features(max_frame, max_objects, train=False)
    resnet_vl = Resnet_features(max_frame, train=False)
    optical_vl = Optical_features(max_frame, train=False)
    captions_vl = Caption_loader(max_cap_per_vid, max_word_per_cap, train=False)

    attention = Attention(max_frame, max_objects, bi_dir).to(device)
    encoder = Encoder(hidden_size, num_layers, bi_dir).to(device)
    decoder = Decoder(hidden_size, num_layers, bi_dir).to(device)

    print('now training...')
    train(attention, encoder, decoder, captions, objects, optical, resnet, \
          objects_vl, resnet_vl, optical_vl, captions_vl, num_epoch, \
          learning_rate, batch_size, max_word_per_cap)

    print('done training...')















# comment
