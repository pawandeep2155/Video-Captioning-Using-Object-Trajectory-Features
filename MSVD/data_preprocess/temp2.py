# import numpy as np
# import torch
# import torchtext.vocab as vocab
# from tqdm import tqdm
#
# '''
# CORRECT DATASET using incorrect and correct word files.
# '''
#
# dataset = open('english.txt')
# word_not_available = open('not_available.txt')
# word_corrected = open('not_available_correct.txt')
# dataset_corrected_path = open('english_corrected.txt', 'w')
#
# glove = vocab.GloVe(name='840B', dim=300)
#
# vec_not_available = []
#
# def word2vec(word):
#     try:
#         vec = glove.vectors[glove.stoi[word]]
#         return vec
#     except:
#         return []
#
# def removeNonEnglish(s):
#     english_word = "".join(i for i in s if i.isalpha()==True)
#     return english_word
#
# if __name__ == "__main__":
#
#     keys = word_not_available.readlines()
#     keys = [key.strip() for key in keys]
#
#     values = word_corrected.readlines()
#     values = [value.strip() for value in values]
#
#     correct_dict = dict(zip(keys, values))
#
#     for i, row in enumerate(dataset):
#         vid = row.split('\t')[0]
#         caption = row.split('\t')[1]
#
#         correct_cap = ''
#         for word in caption.split():
#             word = word.strip()
#             if word in correct_dict:
#                 word = correct_dict[word]
#             correct_cap += word + ' '
#         correct_cap = correct_cap[:-1] + '\n'
#         dataset_corrected_path.write(vid + '\t' + correct_cap)
#
#     dataset_corrected_path.close()
