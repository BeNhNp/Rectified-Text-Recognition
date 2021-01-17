#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from train_helper import MaskedCrossEntropyLoss, AverageMeter, TainTestConfig
from dataset import LmdbDataset

from model import TextRecognitionModel

config = TainTestConfig()
writer = SummaryWriter(
    log_dir="../data/logs",
    flush_secs = 30,
)

# config.lmdb_config.num_samples = 4*10000
config.batch_size = 1024
config.iter_to_valid = 128*8

train_dataset = torch.utils.data.ConcatDataset([
    LmdbDataset(path, config.lmdb_config) for path in config.train_data
])

data_loader = DataLoader(train_dataset, 
    batch_size=config.batch_size, 
    num_workers = 4,
    shuffle = True, 
    pin_memory = True, 
    drop_last = True,
)

path = "../data/data_lmdb_release/evaluation/IIIT5k_3000"
config.lmdb_config.num_samples = 1000
test_dataset = LmdbDataset(path, config.lmdb_config)

data_loader_test = DataLoader(test_dataset, 
    batch_size = config.batch_size, 
    num_workers = 4,
    shuffle = False, 
    pin_memory = True, 
    drop_last = False,
)

model = TextRecognitionModel(config.model_config)

model = model.to(config.device)
model = torch.nn.DataParallel(model)

parameters = model.parameters()
parameters = filter(lambda p: p.requires_grad, parameters)

optimizer = torch.optim.Adadelta(parameters)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,4], gamma=0.1)

start_epoch = 0
epochs = 6
grad_clip = 1

with torch.no_grad():
    torch.cuda.empty_cache()

device = config.device
loss_function = MaskedCrossEntropyLoss(use_bidecoder=config.use_bidecoder)

end = time.time()
n_iter = 0

model.train()

for epoch in range(start_epoch, epochs):
    
    losses = AverageMeter()
    
    total_iter = len(data_loader)
    for i, data_in in enumerate(data_loader):
        n_iter += 1
        
        if config.use_bidecoder:
            imgs, labels1, labels2, lengths = data_in
            labels = (labels1.to(device), labels2.to(device))
        else:
            imgs, labels, lengths = data_in
            labels = labels.to(device)
        imgs = imgs.to(device)
#         lengths = lengths.to(device)
        
        output_dict = model(imgs, labels, lengths.max().item())
        
        batch_size = imgs.size(0)
        predictions = output_dict['prediction']
        loss = loss_function(predictions, labels, lengths)
        
        losses.update(loss.item(), batch_size)
        
        writer.add_scalar("Loss/train", loss.item(), n_iter)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if i % config.iter_to_valid==0: # valid on the test dataset
            model.eval()
            eval_score = config.valid(test_dataset, data_loader_test, model)
            writer.add_scalar("Accuracy/valid", eval_score)
            model.train()

        if i % 512==0: # output loss information
            time_elasped = time.time() - end
            time_remain = (total_iter*(epochs - epoch) - i) / n_iter *time_elasped
            print('%s epoch %d: %d/%d time elasped %s/remain %s loss: %f'%(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    epoch, i, total_iter, 
                    str(datetime.timedelta(seconds=int(time_elasped))), 
                    str(datetime.timedelta(seconds= int(time_remain))), 
                    losses.val
                    ), 
                flush=True
            )
    
    # config.valid(test_dataset, data_loader, model)
    scheduler.step(epoch)

writer.close()