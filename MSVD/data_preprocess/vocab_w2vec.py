import numpy as np
import torch
import torchtext.vocab as vocab
import csv
from tqdm import tqdm
from autocorrect import spell

'''
Build vocabulary for dataset
'''

word2vec_path = '../../../dataset/MSVD/word2vec/word2vec.pt'
dataset_path = '../../../dataset/MSVD/dataset_english.txt'

glove = vocab.GloVe(name='840B', dim=300)

vec_not_available = []

def removeNonEnglish(s):
    english_word = "".join(i for i in s if i.isalpha()==True)
    return english_word

def word2vec(word):
    vec = glove.vectors[glove.stoi[word]]
    return vec


if __name__ == "__main__":

    # Extract all English words in dataset
    vocab = {}
    print('Extracting Unique words...')

    with open(dataset_path) as file:
        for i, row in enumerate(tqdm(file)):
            caption = row.split('\t')[1]
            for word in caption.split():
                word = word.strip()

            for word in caption.split():
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
                    if w not in vocab and w != '' and w != ' ':
                        vec = word2vec(w)
                        vocab[w] = vec.numpy()

    torch.save(vocab, word2vec_path)











#
