#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"]='4,5,6,7'

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from train_helper import MaskedCrossEntropyLoss, TainTestConfig
from dataset import LmdbDataset

from model import TextRecognitionModel

config = TainTestConfig()
# TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
writer = SummaryWriter(
    log_dir="../data/logs/" + config.name,
    flush_secs = 30,
)

# config.lmdb_config.num_samples = 10000
# config.batch_size = 1024
n_device = torch.cuda.device_count()
config.batch_size = 256 * n_device
config.iter_to_valid = 128*8

train_dataset = torch.utils.data.ConcatDataset([
    LmdbDataset(path, config.lmdb_config) for path in config.train_data
])

data_loader = DataLoader(train_dataset, 
    batch_size=config.batch_size, 
    num_workers = 2 * n_device,#4,
    shuffle = True, 
    pin_memory = True, 
    drop_last = True,
)

path = "../data/lmdbs/evaluation/IIIT5K_3000"
config.batch_size = 1024
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

epochs = 4
optimizer = torch.optim.Adadelta(parameters)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs-2,epochs-1], gamma=0.2)

start_epoch = 0
grad_clip = 1

with torch.no_grad():
    torch.cuda.empty_cache()

device = config.device
loss_function = MaskedCrossEntropyLoss(use_bidecoder=config.use_bidecoder)

end = time.time()
n_iter = 0
n_iter_valid = 0

for epoch in range(start_epoch, epochs):
        
    total_iter = len(data_loader)
    for i, data_in in enumerate(data_loader):
        n_iter += 1
        model.train()
        
        if config.use_bidecoder:
            imgs, labels1, labels2, lengths = data_in
            labels = (labels1.to(device, non_blocking=True), labels2.to(device, non_blocking=True))
        else:
            imgs, labels, lengths = data_in
            labels = labels.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
#         lengths = lengths.to(device)
        
        output_dict = model(imgs, labels, lengths.max().item())
        
        batch_size = imgs.size(0)
        predictions = output_dict['prediction']
        loss = loss_function(predictions, labels, lengths)
                
        writer.add_scalar("Loss/train", loss.item(), n_iter)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if i % config.iter_to_valid==0: # valid on the test dataset
            model.eval()
            with torch.no_grad():
                eval_score = config.valid(test_dataset, data_loader_test, model)
            writer.add_scalar("Accuracy/valid", eval_score, n_iter_valid)
            n_iter_valid += 1
#             model.train()

        if i % 128==0: # output loss information
            time_elasped = time.time() - end
            time_remain = (total_iter*(epochs - epoch) - i) / n_iter *time_elasped
            print('%s epoch %d: %d/%d time elasped %s/remain %s loss: %f'%(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    epoch, i, total_iter, 
                    str(datetime.timedelta(seconds=int(time_elasped))), 
                    str(datetime.timedelta(seconds= int(time_remain))), 
                    loss.item()
                    ), 
                flush=True
            )
    
    scheduler.step(epoch)

writer.close()