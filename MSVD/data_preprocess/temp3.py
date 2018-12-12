# import os
#
# cap = os.listdir('../../../dataset/MSVD/caption')
# cap = [item[:-3] for item in cap]
#
# vid = os.listdir('../../../dataset/MSVD/videos')
# vid = [item[:-4] for item in vid]
#
# extra_caption = [item for item in cap if item not in vid]
# #
# # # delete extra caption files
# # for vid in extra_caption:
# #     os.remove('../../../dataset/MSVD/caption/' + vid + '.pt')
# #
# # print(len(cap))
# # print(len(vid))
# #
# # # check both are equal
# # extra_caption = [item for item in cap if item not in vid]
# print(extra_caption)
#
#
#
#
#
#
#












# import numpy as np
# import torch
# import torchtext.vocab as vocab
# import csv
# from tqdm import tqdm
# from autocorrect import spell
#
# '''
# Correct spelling mistakes line by line.
# '''
#
# dataset_path = 'english_corrected.txt'
#
# glove = vocab.GloVe(name='840B', dim=300)
#
# vec_not_available = []
#
# def word2vec(word):
#     vec = glove.vectors[glove.stoi[word]]
#
# def removeNonEnglish(s):
#     english_word = "".join(i for i in s if i.isalpha()==True)
#     return english_word
#
# if __name__ == "__main__":
#
#     # Extract all English words in dataset
#     vocab = []
#     print('Extracting Unique words...')
#
#     with open(dataset_path,'r') as dataset:
#         for i, row in enumerate(dataset):
#
#             caption = row.split('\t')[1]
#             print(i, caption)
#
#             for word in caption.split():
#                 word_list = []
#                 if '-' in word:
#                     word_list = word.split('-')
#                 if ',' in word:
#                     word_list = word.split(',')
#
#                 if len(word_list) > 1:
#                     for words in word_list:
#                         if words.isalpha(): # only english alphabet
#                             if len(words) > 0:
#                                 word2vec(words)
#                         else:
#                             words = removeNonEnglish(words)
#                             if len(words) > 0:
#                                 word2vec(words)
#                 else:
#                     if word not in vocab:
#                         if word.isalpha(): # only english alphabet
#                             if len(word) > 0:
#                                 word2vec(word)
#                         else:
#                             word = removeNonEnglish(word)
#                             if len(word) > 0:
#                                 word2vec(word)
#
#
#
#




#
