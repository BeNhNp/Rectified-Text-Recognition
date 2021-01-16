import datetime
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import torch
from torch.utils.data import DataLoader

from train_helper import MaskedCrossEntropyLoss, AverageMeter, TainTestConfig
from dataset import LmdbDataset

from model import TextRecognitionModel

config = TainTestConfig()


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

model = TextRecognitionModel(config.model_config)

model = model.to(config.device)
model = torch.nn.DataParallel(model)

parameters = model.parameters()
parameters = filter(lambda p: p.requires_grad, parameters)

optimizer = torch.optim.Adadelta(parameters)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,5], gamma=0.1)

iters_old = 0
loss_collect = []


start_epoch = 0
epochs = 5
grad_clip = 1

with torch.no_grad():
    torch.cuda.empty_cache()

device = config.device
loss_function = MaskedCrossEntropyLoss(use_bidecoder=config.use_bidecoder)

for epoch in range(start_epoch, epochs):
    
    model.train()
    
    losses = AverageMeter()
    
    end = time.time()
    
    total_iter = len(data_loader)
    for i, data_in in enumerate(data_loader):
                
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
        total_loss = loss_function(predictions, labels, lengths).sum()
        
        losses.update(total_loss.item(), batch_size)
        
        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
                
        if i % 256==0:
            time_elasped = time.time() - end
            print('%s epoch %d: %d/%d time elasped %s/remain %s loss: %f'%(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                epoch, i, total_iter, 
                str(datetime.timedelta(seconds=int(time_elasped))), 
                str(datetime.timedelta(seconds= int((total_iter*(epochs - epoch) - i) / time_elasped))), 
                losses.val), flush=True
            )
    
    path = "/home/bit/data/wg/blobs/model_epoch%d.pth"%epoch
    torch.save({
            'state_dict': model.module.state_dict(),
    #         'best_res': best_res,
        }, path
    )
    scheduler.step(epoch)