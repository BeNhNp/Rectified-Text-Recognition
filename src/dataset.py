#!/usr/bin/python
# -*- coding:utf-8 -*-

import os

import numpy as np

import cv2
import lmdb
from torch.utils.data import Dataset

import string
def get_vocabulary(voc_type, EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN'):
    '''
    voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
    '''

    voc = [EOS, PADDING]
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    if voc_type == 'LOWERCASE':
        voc+= list(string.digits + string.ascii_lowercase + string.punctuation)
    elif voc_type == 'ALLCASES':
        voc+= list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc+= list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')
    
    # update the voc with specifical chars
    voc.append(UNKNOWN)
    return voc


class LmdbDatasetConfig:
    '''
    configuration for the data set
    '''
    def __init__(self, voc_type = 'ALLCASES_SYMBOLS'):
        self.voc_type      = voc_type
        self.num_samples   = 0 #-\inf to 0 means use all data
        self.max_len_labels= 100
        self.use_bidecoder = True
        
        self.imgHeight = 36#64
        self.imgWidth = 128#256
        
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(self.voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.num_classes = len(self.voc)

class LmdbDataset(Dataset):
    def __init__(self, root, config = LmdbDatasetConfig()):
        super().__init__()
        self.env = lmdb.open(root, max_readers=8, readonly=True)
        assert self.env is not None, "cannot create lmdb from %s" % root
        
        self.txn = self.env.begin()
        
        self.voc_type = config.voc_type
        self.use_bidecoder = config.use_bidecoder
        self.imgHeight     = config.imgHeight
        self.imgWidth      = config.imgWidth
        
        self.max_len = config.max_len_labels
        self.nSamples = int(self.txn.get(b"num-samples"))
        if config.num_samples > 0:
            self.nSamples = min(self.nSamples, config.num_samples)
        
        assert self.voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = config.voc
        self.char2id = config.char2id
        self.id2char = config.id2char

        self.num_classes = len(self.voc)
        self.lowercase = (self.voc_type == 'LOWERCASE')

    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        assert 0<=index <= len(self), 'index range error'
        
        index += 1

        # reconition labels
        label_key = b'label-%09d' % index
        word = self.txn.get(label_key).decode()
        if not word:
            return self[index + 1]
        if len(word) > self.max_len:
            print("label too long", word)
            return self[index + 1]
        
        img_key = b'image-%09d' % index
        imgbuf = self.txn.get(img_key)

        # uncompress the jpeg/png file bytes
        file_bytes = np.asarray(bytearray(imgbuf), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # height = img.shape[0]
        # width = img.shape[1]
        # if height> width:
        #     img = cv2.resize(img, dsize=(self.imgHeight, self.imgWidth), interpolation=cv2.INTER_CUBIC)
        #     img = cv2.transpose(img)
        #     img = cv2.flip(img, 1)
        # else:
        img = cv2.resize(
            img, 
            dsize=(self.imgWidth, self.imgHeight), 
            interpolation=cv2.INTER_CUBIC
        )
        # if np.random.randint(3)==0:
        #     img = cv2.flip(img, -1)
        
        if self.lowercase:
            word = word.lower()
        
        ## fill with the padding token
        label_list = []
        for char in word:
            if char in self.char2id:
                label_list.append(self.char2id[char])
            else:
                ## add the unknown token
                print('{0} is out of vocabulary.'.format(char))
                label_list.append(self.char2id[self.UNKNOWN])
        
        ## add a stop token
        label_list = label_list + [self.char2id[self.EOS]]
        
        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
        label[:len(label_list)] = np.array(label_list)

        # label length
        label_len = len(label_list)

        if self.use_bidecoder:
            label_reverse = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
            label_reverse[:label_len] = np.array(label_list[-2::-1] + [self.char2id[self.EOS]])
            return img, label, label_reverse, label_len
        return img, label, label_len