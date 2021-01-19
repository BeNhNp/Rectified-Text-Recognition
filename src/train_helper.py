#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

import sys
import os
import time

import random
import numpy as np

from dataset import LmdbDatasetConfig
from model import TextRecognitionModelConfig

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, 
            size_average=True,
            sequence_normalize=False,
            sample_normalize=True,
            use_bidecoder = False
        ):
        super().__init__()
        self.use_bidecoder = use_bidecoder
        self.weight = (0.5, 0.5) if use_bidecoder else None
        self.size_average = size_average
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize

        assert (sequence_normalize and sample_normalize) == False

    def forward(self, prediction, target, length):
        if self.use_bidecoder:
            out1 = self._loss(prediction[0], target[0], length)
            out2 = self._loss(prediction[1], target[1], length)
            return self.weight[0]*out1 + self.weight[1]*out2
        else:
            return self._loss(prediction, target, length)
        
    def _loss(self, input, target, length):
        assert not target.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

        # length to mask
        batch_size, def_max_length = target.size(0), target.size(1)
        mask = torch.zeros(batch_size, def_max_length)
        for i in range(batch_size):
            mask[i,:length[i]].fill_(1)
        mask = mask.type_as(input)
        # truncate to the same size
        max_length = max(length)

        target = target[:, :max_length]
        mask   = mask[:, :max_length]

        input = input.reshape(-1, input.size(2))

        input = F.log_softmax(input, dim=1)

        target = target.reshape(-1, 1)
        mask = mask.reshape(-1, 1)

        output = - input.gather(1, target.long()) * mask

        output = torch.sum(output)
       
        if self.sequence_normalize:
            output = output / torch.sum(mask)
        if self.sample_normalize:
            output = output / batch_size

        return output

class Logger:
    def __init__(self, path=None):
        self.console = sys.stdout
        self.file = None
        if path is not None:
            dir = os.path.dirname(path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            self.file = open(path, 'a+')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

from test_helper import Accuracy

# version = torch.__version__.split('.')
# if version[0]=='1':
#     if int(version[1])==1:
#         # ignore bugs with pytorch 1.1 
#         import warnings
#         warnings.filterwarnings("ignore", category=RuntimeWarning)
#         torch.nn.RNNBase.flatten_parameters = lambda x: None
    
#     if int(version[1])>2:
#         grid_sample_ori = F.grid_sample
#         F.grid_sample = lambda *args, **kargs: grid_sample_ori(*args, **kargs, align_corners=True)

class TainTestConfig:
    def __init__(self, batch_size = 512, 
        device='cuda', 
        use_bidecoder = True
    ):

        self.batch_size = batch_size
        self.device = device

        self.train_data = [
#             '../data/lmdb_train/CVPR2016/',
#             '../data/lmdb_train/NIPS2014'
            "../data/data_lmdb_release/training/MJ/MJ_train",
            "../data/data_lmdb_release/training/ST/",
        ]
        
        self.use_bidecoder = use_bidecoder

        sys.stdout = Logger('./log.txt')
        self.iter_to_valid = 1024

        '''
        'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
        '''
        self.lmdb_config = LmdbDatasetConfig(voc_type = 'LOWERCASE')
        self.lmdb_config.use_bidecoder = self.use_bidecoder
        self.lmdb_config.num_samples   = 0#-\inf to 0 means use all data
        
        self.model_config = TextRecognitionModelConfig()
        self.model_config.num_classes = self.lmdb_config.num_classes
        self.model_config.max_len_labels = self.lmdb_config.max_len_labels
        self.model_config.eos = self.lmdb_config.char2id[self.lmdb_config.EOS]
        self.model_config.device = self.device
        self.model_config.use_bidecoder = self.use_bidecoder
        
        SEED = 1
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        self._best_score = -1
    
    def save_model(self, model, eval_score):
        if eval_score<0.8: return
        if self._best_score < eval_score:
            self._best_score = eval_score
            torch.save({
                    'state_dict': model.module.state_dict(),
                    'best_score': eval_score,
                }, "../data/models/model_best.pth"
            )
    def valid(self, test_dataset, data_loader, model):

        pred_rec = []
        targets = []
        device  = self.device
        for i, data_in in enumerate(data_loader):
            if self.use_bidecoder:
                imgs, labels1, labels2, lengths = data_in
                labels = (labels1.to(device), labels2.to(device))
            else:
                imgs, labels, lengths = data_in
                labels = labels.to(device)
            imgs = imgs.to(device)
            # lengths = lengths.to(device)
            
            with torch.no_grad():
                output_dict = model(imgs, labels)
            
            prediction = output_dict['prediction']
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            _, prediction = prediction.max(2)
            pred_rec.append(prediction.cpu())
            targets.append(labels1.cpu() if self.use_bidecoder else labels.cpu())
            
        eval_score = Accuracy(torch.cat(pred_rec), torch.cat(targets), self.lmdb_config)
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
            'valid score {0:.3f}'.format(eval_score),
            flush=True
        )
        self.save_model(model, eval_score)
        return eval_score
