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
video_ids_tr = os.listdir(video_features_tr_path)
video_ids_tr = [item[:-3] for item in video_ids_tr]

video_ids_vl = os.listdir(video_features_vl_path)
video_ids_vl = [item[:-3] for item in video_ids_vl]

all_video_ids = video_ids_tr + video_ids_vl

def caption_info():
    print('caption info...')
    '''
    max Cap/video
    '''
    max_cap_per_vid = 0

    for vid in all_video_ids:
        if vid in video_ids_tr:
            video_cap = torch.load(video_features_tr_path + vid + '.pt')
        else:
            video_cap = torch.load(video_features_vl_path + vid + '.pt')

        max_cap_per_vid = max(max_cap_per_vid, len(video_cap['caption']))

    return max_cap_per_vid

def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    if pad_size[dim] > 0:
        return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    return vec

class PadCollate:

    def __init__(self, max_caption):
        self.max_cap_per_vid = max_caption

    def pad_collate(self, batch):
        print('pad collate batch', len(batch), len(batch[0]), len(batch[1]))
        print('batch dims', batch[0][0].shape, batch[0][1].shape, batch[0][2].shape, batch[0][3].shape)
        print('batch dims', batch[1][0].shape, batch[1][1].shape, batch[1][2].shape, batch[1][3].shape)

        object_data = []
        optical_data = []
        resnet_data = []
        caption_data = []

        for x in batch:
            object_data.append(x[0])
            optical_data.append(x[1])
            resnet_data.append(x[2])
            caption_data.append(x[3])

        # object_data, optical_data, resnet_data, caption_data = batch[0], batch[1], batch[2], batch[3]

        # Padding Objects
        max_lenx = max(map(lambda x: x.shape[1], object_data))
        object_list = list(map(lambda x: pad_tensor(x, pad=max_lenx, dim=1), object_data))
        max_lenx = max(map(lambda x: x.shape[0], object_data))
        object_list = list(map(lambda x: pad_tensor(x, pad=max_lenx, dim=0), object_list))
        object_list = list(map(lambda x: x.unsqueeze(0).repeat(self.max_cap_per_vid,1,1,1), object_list))
        object_tensor = torch.cat(object_list, dim=0)
        print('object', object_tensor.shape)

        # Padding Optical
        max_lenx = max(map(lambda x: x.shape[0], optical_data))
        optical_list = list(map(lambda x: pad_tensor(x, pad=max_lenx, dim=0), optical_data))
        optical_list = list(map(lambda x: x.unsqueeze(0).repeat(self.max_cap_per_vid,1,1), optical_list))
        optical_tensor = torch.cat(optical_list, dim=0)
        print('optical', optical_tensor.shape)

        # Padding Resnet
        max_lenx = max(map(lambda x: x.shape[0], resnet_data))
        resnet_list = list(map(lambda x: pad_tensor(x, pad=max_lenx, dim=0), resnet_data))
        resnet_list = list(map(lambda x: x.unsqueeze(0).repeat(self.max_cap_per_vid,1,1), resnet_list))
        resnet_tensor = torch.cat(resnet_list, dim=0)
        print('resnet', resnet_tensor.shape)

        # Padding Captions
        max_lenx = max(map(lambda x: x.shape[1], caption_data))
        caption_list = list(map(lambda x: pad_tensor(x, pad=max_lenx, dim=1), caption_data))
        max_lenx = max(map(lambda x: x.shape[0], caption_data))
        caption_list = list(map(lambda x: pad_tensor(x, pad=max_lenx, dim=0), caption_list))
        caption_list = list(map(lambda x: pad_tensor(x, pad=self.max_cap_per_vid, dim=0), caption_list))
        caption_tensor = torch.cat(caption_list, dim=0)
        print('caption', caption_tensor.shape)

        return object_tensor, optical_tensor, resnet_tensor, caption_tensor

    def __call__(self, batch):
        return self.pad_collate(batch)


class VideoDataset(DataLoader):
    def __init__(self, video_ids, train=True):

        if train == True:
            self.video_path = video_features_tr_path
        else:
            self.video_path = video_features_vl_path

        self.video_ids = video_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):

        video_features = torch.load(self.video_path + self.video_ids[idx] + '.pt')
        objects = video_features['object']
        caption = video_features['caption']
        optical = video_features['optical']
        resnet = video_features['resnet']

        print('object', objects.shape, 'optical', optical.shape, 'resnet', resnet.shape, 'caption', caption.shape)

        return  objects, optical, resnet, caption


# NETWORK
class Attention(nn.Module):

    def __init__(self, num_frames, obj_per_frame, bi_dir):
        pass

    def forward(self, video_instances, resnet_ftrs, optical_ftrs, object_ftrs):
        pass

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
    max_cap_per_vid = 100#caption_info()
    print('dataloader starting...')

    dataset_tr = VideoDataset(video_ids_tr)
    dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=PadCollate(max_cap_per_vid))

    for i, data in enumerate(dataloader_tr):
        print('data', type(data))
        object_t, optical_t, resnet_t, caption_t = data[0], data[1], data[2], data[3]

        print('Tensors' ,object_t.shape, optical_t.shape, resnet_t.shape, caption_t.shape)

        break

    # Validation Data loader.
    # dataset_vl = VideoDataset(video_ids_vl)
    # dataloader_vl = DataLoader(dataset_vl, batch_size=batch_size, shuffle=True, collate_fn=PadCollate(max_cap_per_vid))

    # attention = Attention(max_frame, max_objects, bi_dir).to(device)
    # encoder = Encoder(hidden_size, num_layers, bi_dir).to(device)
    # decoder = Decoder(hidden_size, num_layers, bi_dir).to(device)

    # print('now training...')
    # train(attention, encoder, decoder, captions, objects, optical, resnet, \
    #       objects_vl, resnet_vl, optical_vl, captions_vl, num_epoch, \
    #       learning_rate, batch_size, max_word_per_cap)

    print('done training...')















# comment
