import torch
import os
import numpy as np


# Object
vid1 = {1:{'man':{'position':[5,10,20,30], 'feature':np.zeros((2048))}}, 2:{'man':{'position':[5,15,30,40], 'feature':np.zeros((2048))}}}
vid2 = {1:{'car':{'position':[15,10,20,30], 'feature':np.zeros((2048))}}, 2:{'man':{'position':[15,15,30,40], 'feature':np.zeros((2048))}}}

torch.save(vid1, '../../../dataset/MSVD/dummy/object/valid/vid3.pt')
torch.save(vid2, '../../../dataset/MSVD/dummy/object/valid/vid4.pt')

# Optcial
opt1 = {1:np.zeros((2048)), 2:np.zeros((2048))}
opt2 = {1:np.zeros((2048)), 2:np.zeros((2048))}

torch.save(opt1, '../../../dataset/MSVD/dummy/optical/valid/vid3.pt')
torch.save(opt2, '../../../dataset/MSVD/dummy/optical/valid/vid4.pt')

# Resnet
res1 = {1:np.zeros((2048)), 2:np.zeros((2048))}
res2 = {1:np.zeros((2048)), 2:np.zeros((2048))}

torch.save(res1,'../../../dataset/MSVD/dummy/resnet/valid/vid3.pt')
torch.save(res2,'../../../dataset/MSVD/dummy/resnet/valid/vid4.pt')

# Captions

captions = {'vid1':['The man is cutting an onian', 'The man slices onion'], \
            'vid2':['A boy is playing','A child playing cricket']}

cap1 = {'vid3':[np.zeros((6,300)), np.zeros((4, 300))]}
cap2 = {'vid4':[np.zeros((4,300)), np.zeros((4, 300))]}

torch.save(captions, '../../../dataset/MSVD/dummy/caption/caption_tr.pt')

torch.save(cap1, '../../../dataset/MSVD/dummy/caption/valid/vid3.pt')
torch.save(cap2, '../../../dataset/MSVD/dummy/caption/valid/vid4.pt')
































# import numpy as np
# import torch
# import torchtext.vocab as vocab
# import csv
# from tqdm import tqdm
# from autocorrect import spell
#
# '''
# Check how many words not in word2vec after correcting on grammarly.
# '''
#
# dataset_path = 'english_corrected.txt'
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
#     # Extract all English words in dataset
#     vocab = []
#     print('Extracting Unique words...')
#
#     with open(dataset_path,'r') as dataset:
#         for i, row in enumerate(dataset):
#
#             caption = row.split('\t')[1]
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
#                         if words not in vocab:
#                             if words.isalpha(): # only english alphabet
#                                 if len(words) > 0:
#                                     vocab.append(words)
#                             else:
#                                 words = removeNonEnglish(words)
#                                 if len(words) > 0:
#                                     vocab.append(words)
#                 else:
#                     if word not in vocab:
#                         if word.isalpha(): # only english alphabet
#                             if len(word) > 0:
#                                 vocab.append(word)
#                         else:
#                             word = removeNonEnglish(word)
#                             if len(word) > 0:
#                                 vocab.append(word)
#
#     vocab = list(set(vocab))
#     print('building word2vec vocabulary...')
#     # word to vec
#     vocab_dict = {}
#
#     for word in tqdm(vocab):
#         vec = word2vec(word.lower())
#
#         if len(vec)==0:
#             vec_not_available.append(word)
#             continue
#
#         vocab_dict[word] = vec
#
#
#     print('Unique words in dataset', len(vocab))
#     print('word2vec avaialable', len(vocab_dict))
#     print('word2vec not avaialable ', len(vec_not_available))
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
# #
#
#
#
# # import numpy as np
# # import torch
# # import torchtext.vocab as vocab
# # import csv
# # from tqdm import tqdm
# #
# # word2vec_path = '../../../dataset/MSVD/word2vec/word2vec.pt'
# # dataset_path = 'dataset_eng.txt'
# #
# # glove = vocab.GloVe(name='840B', dim=300)
# #
# # vec_not_available = []
# #
# # def word2vec(word):
# #     try:
# #         vec = glove.vectors[glove.stoi[word]]
# #         return vec
# #     except:
# #         return []
# #
# # def removeNonEnglish(s):
# #     english_word = "".join(i for i in s if i.isalpha()==True)
# #     return english_word
# #
# # if __name__ == "__main__":
# #
# #     # Extract all English words in dataset
# #     vocab = []
# #     print('Extracting Unique words...')
# #
# #     with open(dataset_path,'r') as file:
# #         for i, row in enumerate(file):
# #             caption = row
# #             for word in caption.split():
# #                 word_list = []
# #                 if '-' in word:
# #                     word_list = word.split('-')
# #                 if ',' in word:
# #                     word_list = word.split(',')
# #
# #                 if len(word_list) > 1:
# #                     for words in word_list:
# #                         if words not in vocab:
# #                             if words.isalpha(): # only english alphabet
# #                                 vocab.append(words)
# #                             else:
# #                                 words = removeNonEnglish(words)
# #                                 vocab.append(words)
# #                 else:
# #                     if word not in vocab:
# #                         if word.isalpha(): # only english alphabet
# #                             vocab.append(word)
# #                         else:
# #                             word = removeNonEnglish(word)
# #                             vocab.append(word)
# #
# #     vocab = list(set(vocab))
# #     print('building word2vec vocabulary...')
# #     # word to vec
# #     vocab_dict = {}
# #
# #     for word in tqdm(vocab):
# #         vec = word2vec(word.lower())
# #
# #         if len(vec)==0:
# #             vec_not_available.append(word)
# #             continue
# #
# #         vocab_dict[word] = vec
# #
# #
# #     print('Unique words in dataset', len(vocab))
# #     print('word2vec avaialable', len(vocab_dict))
# #     print('word2vec not avaialable ', len(vec_not_available))
# #     print(vec_not_available)
# #
# #
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
# #
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
# # input_path = '../../../dataset/MSVD/object_detection/'
# # video_path = '../../../dataset/MSVD/videos/'
# # fail_videos_path = 'fail_crop1.txt'
# # output_path = '../../../dataset/MSVD/object_crop2/'
# #
# # def crop_object(image, obj_cordnts):
# #     x1 = int(obj_cordnts[0])
# #     y1 = int(obj_cordnts[1])
# #     x2 = int(obj_cordnts[2])
# #     y2 = int(obj_cordnts[3])
# #
# #     return image[y1:y2, x1:x2]
# #
# # if __name__ == "__main__":
# #
# #     video_ids = []
# #
# #     with open(fail_videos_path) as f:
# #
# #         for content in f:
# #             r = content
# #             video_id = r.split('/')[-1].strip()
# #             video_ids.append(video_id)
# #
# #     for i, vid in enumerate(tqdm(video_ids)):
# #
# #         if os.path.exists(output_path + vid[:-3] +'_done'):
# #             continue
# #
# #         os.mkdir(output_path + vid[:-4])
# #
# #         video_objects = torch.load(input_path + vid[:-4] + '.pt')
# #
# #         cap = cv2.VideoCapture(video_path + vid)
# #
# #         frm_num = 1
# #         while(cap.isOpened()):
# #             ret, frame = cap.read()
# #
# #             if frm_num not in video_objects:
# #                 break
# #
# #             objects = video_objects[frm_num]
# #
# #             os.mkdir(output_path + vid[:-4] + '/' + str(frm_num))
# #
# #             if len(objects) != 0:
# #
# #                 for obj_name, obj_cordnts in objects.items(): # object is dictionary with key as object name, value as object cordinates
# #
# #                     croped_object = crop_object(frame, obj_cordnts)
# #                     if not os.path.exists(output_path + vid[:-4] + '/' + str(frm_num) + '/'):
# #                         os.mkdir(output_path + vid[:-4] + '/' + str(frm_num) + '/')
# #
# #                     croped_object = croped_object[...,::-1]
# #                     img = Image.fromarray(croped_object)
# #                     # cv2.imwrite(output_path + vid[:-4] + '/' + str(frm_num) + '/' + obj_name + '.png', croped_object)
# #                     img.save(output_path + vid[:-4] + '/' + str(frm_num) + '/' + obj_name + '.png')
# #
# #             else:
# #                 if not os.path.exists(output_path + vid[:-4] + '/' + str(frm_num) + '/'):
# #                     os.mkdir(output_path + vid[:-4] + '/' + str(frm_num) + '/')
# #
# #             frm_num += 1
# #
# #         os.rename(output_path + vid[:-4], output_path + vid[:-4] +'_done')
# #
# #
# #
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
#             # frame = frame[...,::-1]
#             #
#             # img = Image.fromarray(frame)
#             # img.save('f1.png')
#
#         #
#         # for frm_num, objects in video_objects.items():
#         #
#         #     os.mkdir(output_path + vid[:-3] + '/' + str(frm_num))
#         #
#         #     if len(objects) != 0:
#         #
#         #         for obj_name, obj_cordnts in objects.items(): # object is dictionary with key as object name, value as object cordinates
#         #
#         #             croped_object = crop_object(video[frm_num-1], obj_name, obj_cordnts)
#         #             if not os.path.exists(output_path + vid[:-3] + '/' + str(frm_num) + '/'):
#         #                 os.mkdir(output_path + vid[:-3] + '/' + str(frm_num) + '/')
#         #
#         #             img = Image.fromarray(croped_object)
#         #             # cv2.imwrite(output_path + vid[:-4] + '/' + str(frm_num) + '/' + obj_name + '.png', croped_object)
#         #             img.save(output_path + vid[:-3] + '/' + str(frm_num) + '/' + obj_name + '.png')
#         #
#         #     else:
#         #         if not os.path.exists(output_path + vid[:-3] + '/' + str(frm_num) + '/'):
#         #             os.mkdir(output_path + vid[:-3] + '/' + str(frm_num) + '/')
#         #
#         #
#         # os.rename(output_path + vid[:-3], output_path + vid[:-3] +'_done')
