import torch
import numpy as np
import os
from tqdm import tqdm

'''
Every caption of a video => convert to 2D Matrix and store in pt file.
e.g = {'vid':[arry(10,300), array(20, 300)]} means 2 captions each containing
10 and 20 words resp. Each word is represented by vector of dim 300.
'''

dictionary_path = '../../../dataset/MSVD/word2vec/word2vec.pt'
dictionary = torch.load(dictionary_path)

dataset_path = '../../../dataset/MSVD/dataset_english.txt'
assert os.path.exists(dataset_path), 'dataset path does not exists'

output_path = '../../../dataset/MSVD/caption/'

previous_id = 'mv89psg6zh4_33_46'
video_cap_feature = []

def removeNonEnglish(s):
    english_word = "".join(i for i in s if i.isalpha()==True)
    return english_word

with open(dataset_path, 'r') as file:
    for _, row in enumerate(tqdm(file)):

        row_list = row.split('\t')
        vid = row_list[0]
        caption = row_list[1]

        # Determine num of words in caption
        word_per_caption = 0
        for word in caption.split():
            word = word.strip()
            if '-' in word:
                word_list = word.split('-')
            elif ',' in word:
                word_list = word.split(',')
            elif '/' in word:
                word_list = word.split('/')
            else:
                word_list = [word]

            for w in word_list:
                w = removeNonEnglish(w)
                if w != '' and w != ' ':
                    word_per_caption += 1

        caption_array = np.zeros((word_per_caption, 300))

        # Fill caption array with word vectors.
        word_index = 0
        for word in caption.split():
            word = word.strip()
            word_list = []
            if '-' in word:
                word_list = word.split('-')
            elif ',' in word:
                word_list = word.split(',')
            elif '/' in word:
                word_list = word.split('/')
            else:
                word_list = [word]

            for w in word_list:
                w = removeNonEnglish(w)
                if w != '' and w != ' ':
                    # print(w)
                    # print(dictionary[w].shape)
                    # print('caption array', caption_array.shape)
                    caption_array[word_index] = dictionary[w]
                    word_index += 1


        if vid != previous_id:
            caption_dict = {previous_id:video_cap_feature}
            torch.save(caption_dict, output_path + previous_id + '.pt')
            previous_id = vid
            video_cap_feature = []

        video_cap_feature.append(caption_array)





#
