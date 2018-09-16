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
from os import listdir, walk
from os.path import isfile, join
import cv2
import pickle

def getCaptions(data_dir):
    captionPath = os.path.join(data_dir,'text_c10')
    f = []
    for (dirpath, dirnames, filenames) in walk(captionPath):
        textNames = [join(dirpath,filename) for filename in filenames if filename.endswith('txt')]
        f.extend(textNames)

    captionDict = {}
    for file in f:
        key = int(file.split('/')[-1].split('.')[0].split('_')[1])
        with open(file) as temp:
            captionDict[key] = []
            for i in range(7):
                captionDict[key].append(temp.readline().rstrip())
    return captionDict

def getImagesAndCaptions(data_dir):
    captionDict = getCaptions(data_dir)

    imgCaptionsDict = {}
    imgPath = join(data_dir, 'jpg')
    jpegs = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]

    for i in jpegs:
        key = int(i.split('.')[0].split('_')[1])
        img=cv2.imread(join(imgPath,i))
        imgCaptionsDict[key-1]=(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), captionDict[key])                       #-1 to make sure that indexing starts at 0
    return imgCaptionsDict

def get_vocab(input_dict):
    vocab_set = set()
    for key, value in input_dict.items():
        img,captions = value
        for curr_caption in captions:
            temp = curr_caption.replace(".","").replace(",","").replace("?","").replace("-"," ").replace("(","").replace(")","").replace("!","").replace("/","").replace("\\","").replace('"',"").lower().split()
            vocab_set.update(temp)

    vocab_file = {}
    for key,word in enumerate(vocab_set):
        vocab_file[key] = word
    
    with open('Vocab_flowers.pkl','wb') as f:
        pickle.dump(vocab_file,f)

    print("Vocabulary loaded and pickle file saved in current working directory!")
    return vocab_file

class TextImageDataset(data.Dataset):
    def __init__(self, data_dir, ann_file, vocab_file=None, embedding_type='cnn-rnn',
                 emb_model=None,imsize=64, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.cap = getImagesAndCaptions(data_dir)
        if vocab_file is None:
            self.vocab_file = get_vocab(self.cap)
        else:
            self.vocab_file=vocab_file
            print("Vocabulary loaded from pickle file")
        self.word_to_indx_dict = dict(zip(self.vocab_file.values(),self.vocab_file.keys()))
        # img_val={}
        # captions_val={}
        # for i in range(20):
        #     img,captions=self.cap[i]
        #     img_val[i]=img
        #     captions_val[i]=self.convert_to_index(captions_val[:5])
        # self.val={'img':img_val,'captions':captions_val}

    def convert_to_index(self,captions):
        captions_idx = []
        for descr in captions:
            words_curr = descr.replace(".","").replace(",","").lower().split()                      #clearly each caption is in the form of a sentence
            caps_ind = []
            for word in words_curr:
                if word in self.word_to_indx_dict.keys():
                    caps_ind.append(int(self.word_to_indx_dict[word]))
                else:
                    continue;
            captions_idx.append(np.array(caps_ind))
        return np.array(captions_idx)


    def __getitem__(self, index):
        img,captions=self.cap[index]

        no_of_captions = len(captions)
        cap = np.random.randint(0, no_of_captions)
        idx=[(cap-1)%no_of_captions, cap, (cap+1)%no_of_captions]                                   #why the percentage???
        new_captions=[]
        for i in idx:
            new_captions.append(captions[i])
        captions=np.array(new_captions)
        captions_idx=self.convert_to_index(captions)

        if self.transform is not None:
            img = self.transform(img)
        return img, captions_idx[0], captions_idx[1], captions_idx[2], captions

    def __len__(self):
        return len(self.cap)