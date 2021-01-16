#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

import torch

from torch.utils.data import DataLoader
import time

from train_helper import TainTestConfig
from model import TextRecognitionModel
from test_helper import Accuracy
from dataset import LmdbDataset

path = "../data/models/model_best.pth"
checkpoint = torch.load(path)

config = TainTestConfig()
config.batch_size = 32

model = TextRecognitionModel(config.model_config)
model.load_state_dict(checkpoint['state_dict'])

model = model.to(config.device)
if config.device == 'cuda':
    model = torch.nn.DataParallel(model)

model.eval()

test_data_dir = "../data/data_lmdb_release/evaluation/"
test_data_set= ["IIIT5k_3000", "SVT", "IC03_867", "IC13_1015", "IC15_1811", "SVTP", "CUTE80"]
device = config.device

for test_data in test_data_set:
    path = test_data_dir + test_data
    test_dataset = LmdbDataset(path, config.lmdb_config)
    data_loader = DataLoader(test_dataset, 
                        batch_size = 128,#config.batch_size, 
                        num_workers = 4,
                        shuffle = False, 
                        pin_memory = True, 
                        drop_last = False,
                    )
    test_data += '(%d)'%(len(test_dataset))
    targets = []
    pred_rec = []
    for i, data_in in enumerate(data_loader):
        
        if test_dataset.use_bidecoder:
            imgs, labels1, labels2, lengths = data_in
            labels = (labels1.to(device), labels2.to(device))
        else:
            imgs, labels, lengths = data_in
            labels = labels.to(device)
        imgs = imgs.to(device)
#         lengths = lengths.to(device)
        
        with torch.no_grad():
            output_dict = model(imgs, labels)
        
        prediction = output_dict['prediction']
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        _, prediction = prediction.max(2)
        pred_rec.append(prediction.cpu())
        targets.append(labels1.cpu() if test_dataset.use_bidecoder else labels.cpu())
        break
        
    eval_res = Accuracy(torch.cat(pred_rec), torch.cat(targets), test_dataset)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' '+ test_data)
    print('lexicon0: {0:.3f}'.format(eval_res))
