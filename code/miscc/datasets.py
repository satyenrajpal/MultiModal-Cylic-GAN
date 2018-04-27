from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import torch
import sys
from miscc.config import cfg
import torchvision.datasets as dset
import torchvision.transforms as transforms

class TextImageDataset(data.Dataset):
    def __init__(self, data_dir, ann_file, vocab_file=None, embedding_type='cnn-rnn',
                 emb_model=None,imsize=64, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        # self.glove=self.load_embedding(emb_model)
        # if data_dir.find('birds') != -1:
        #     self.bbox = self.load_bbox()
        # else:
        #     self.bbox = None
        # split_dir = os.path.join(data_dir, split)
        self.cap = dset.CocoCaptions(root = data_dir,
                annFile = ann_file)
        self.vocab_file=vocab_file

        self.word_to_indx_dict = dict (zip(vocab_file.values(),vocab_file.keys()))

        
        #print(len(caps_ind))       
        #caps_ind = [word_to_indx_dict[word] for word in caption.replace(".","").replace(",","")lower().split()]


        # self.filenames = self.load_filenames(split_dir)
        # self.embeddings = self.load_embedding(split_dir, embedding_type)
        # self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # self.captions = self.load_all_captions()

    # def get_img(self, img_path, bbox):
    #     img = Image.open(img_path).convert('RGB')
    #     width, height = img.size
    #     # if bbox is not None:
    #     #     R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    #     #     center_x = int((2 * bbox[0] + bbox[2]) / 2)
    #     #     center_y = int((2 * bbox[1] + bbox[3]) / 2)
    #     #     y1 = np.maximum(0, center_y - R)
    #     #     y2 = np.minimum(height, center_y + R)
    #     #     x1 = np.maximum(0, center_x - R)
    #     #     x2 = np.minimum(width, center_x + R)
    #     #     img = img.crop([x1, y1, x2, y2])
    #     load_size = int(self.imsize * 76 / 64)
    #     img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img

    # def load_bbox(self):
    #     data_dir = self.data_dir
    #     bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
    #     df_bounding_boxes = pd.read_csv(bbox_path,
    #                                     delim_whitespace=True,
    #                                     header=None).astype(int)
    #     #
    #     filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
    #     df_filenames = \
    #         pd.read_csv(filepath, delim_whitespace=True, header=None)
    #     filenames = df_filenames[1].tolist()
    #     print('Total filenames: ', len(filenames), filenames[0])
    #     #
    #     filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    #     numImgs = len(filenames)
    #     for i in xrange(0, numImgs):
    #         # bbox = [x-left, y-top, width, height]
    #         bbox = df_bounding_boxes.iloc[i][1:].tolist()

    #         key = filenames[i][:-4]
    #         filename_bbox[key] = bbox
    #     #
    #     return filename_bbox
    # def load_captions(self, caption_name):
    #     cap_path = caption_name
    #     with open(cap_path, "r") as f:
    #         captions = f.read().decode('utf8').split('\n')
    #     captions = [cap.replace("\ufffd\ufffd", " ")
    #                 for cap in captions if len(cap) > 0]
    #     return captions

    # def load_all_captions(self):
    #     caption_dict = {}
    #     for key in self.filenames:
    #         caption_name = '%s/text/%s.txt' % (self.data_dir, key)
    #         captions = self.load_captions(caption_name)
    #         caption_dict[key] = captions
    #     return caption_dict

    # def load_embedding(self,emb_model):
    #     """
    #     creates a dictionary mapping words to vectors from a file in glove format.
    #     """
    #     with open(emb_model) as f:
    #         glove = {}
    #         for line in f.readlines():
    #             values = line.split()
    #             word = values[0]
    #             vector = np.array(values[1:], dtype='float32')
    #             glove[word] = vector
    #     return glove

    # def get_embedding(self, captions):
    #     # #if embedding_type == 'cnn-rnn':
    #     #     embedding_filename = '/char-CNN-RNN-embeddings.pickle'
    #     # elif embedding_type == 'cnn-gru':
    #     #     embedding_filename = '/char-CNN-GRU-embeddings.pickle'
    #     # elif embedding_type == 'skip-thought':
    #     #     embedding_filename = '/skip-thought-embeddings.pickle'

    #     # with open(data_dir + embedding_filename, 'rb') as f:
    #     #     embeddings = pickle.load(f)
    #     #     embeddings = np.array(embeddings)
    #     #     # embedding_shape = [embeddings.shape[-1]]
    #     #     print('embeddings: ', embeddings.shape)
    #     #Pick one sentence at random and pass an embedding of that in format 
    #     # (number_of_words x embedding_dim) format
    #     snt_idx=np.random.randint(0,len(captions)-1)
    #     sentence=captions[snt_idx].replace(".","").lower().split()
    #     # print(sentence)
    #     embeddings=[self.glove.get(x) for x in sentence]
    #     return np.array(embeddings),sentence

    # def load_class_id(self, data_dir, total_num):
    #     if os.path.isfile(data_dir + '/class_info.pickle'):
    #         with open(data_dir + '/class_info.pickle', 'rb') as f:
    #             class_id = pickle.load(f)
    #     else:
    #         class_id = np.arange(total_num)
    #     return class_id

    # def load_filenames(self, data_dir):
    #     filepath = os.path.join(data_dir, 'filenames.pickle')
    #     with open(filepath, 'rb') as f:
    #         filenames = pickle.load(f)
    #     print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    #     return filenames

    def convert_to_index(self,captions):
        captions_idx = []
        for descr in captions:
            words_curr = descr.replace(".","").replace(",","").lower().split()
            caps_ind = []
            for word in words_curr:
                if word in self.word_to_indx_dict.keys():
                    caps_ind.append(int(self.word_to_indx_dict[word]))
                else:
                    continue;
            captions_idx.append(np.array(caps_ind))
        return np.array(captions_idx)


    def __getitem__(self, index):
        # key = self.filenames[index]
        # cls_id = self.class_id[index]
        #
        # if self.bbox is not None:
        #     bbox = self.bbox[key]
        #     data_dir = '%s/CUB_200_2011' % self.data_dir
        # else:
        # bbox = None
        # data_dir = self.data_dir
        img,captions=self.cap[index]
        idx=np.random.randint(5,size=3)
        new_captions=[]
        for i in idx:
            new_captions.append(captions[i])
        captions=np.array(new_captions)
        captions_idx=self.convert_to_index(captions)

        img=np.array(img)
        if self.transform is not None:
            img = self.transform(img)
        # print("Image Size: ", img.shape)    
        # embeddings,sentence = self.get_embedding(captions)
        # img_name = '%s/images/%s.jpg' % (data_dir, key)
        # img = self.get_img(img_name, bbox)
        # while(True):
        # 	embedding_ix = random.randint(0, len(embeddings)-1)
        	
        # 	embedding = embeddings[embedding_ix]
        # 	if(embedding is not None):
        # 		break
        # # print(embedding.shape)
        # word_=sentence[embedding_ix]
        # if self.target_transform is not None:
        #     embedding = self.target_transform(embedding)
        return img, captions_idx[0],captions_idx[1],captions_idx[2],captions

    def __len__(self):
        return len(self.cap)
