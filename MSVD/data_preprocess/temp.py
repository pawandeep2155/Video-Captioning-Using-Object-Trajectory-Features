class A():
    def __init__(self):
        print('i am called')
        self.a = 'hello name'
        self.b = 2
    def get_me(self):
        return self.b

class B(A):
    def __init__(self):
        A.__init__(self)
        p = A.get_me(self)
        print(self.a)
        print(self.b)

a = A()
b = B()




# import csv
# import os
#
# file_path = '../../../dataset/MSVD/dataset.csv'
#
# # Find number of english sentences per video
# video_dict = {}
#
# with open(file_path,'r', newline='') as csvfile:
#     dataset = csv.reader(csvfile, delimiter=',')
#     for i, row in enumerate(dataset):
#         if i%2 == 0:
#             language = row[6]
#             if language == 'English':
#                 key = row[0] + row[1] + row[2]
#                 if key in video_dict:
#                     video_dict[key] += 1
#                 else:
#                     video_dict[key] = 1
#
# print(len(video_dict))
# print(video_dict)
