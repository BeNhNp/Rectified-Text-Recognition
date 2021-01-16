import torch
from torch import nn
import torch.nn.functional as F

import sys
import os

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
        self.weight = (1, 1) if use_bidecoder else None
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

        output = torch.sum(output, 0)
       
        if self.sequence_normalize:
            output = output / torch.sum(mask)
        if self.sample_normalize:
            output = output / batch_size

        return output

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

class TainTestConfig:
    def __init__(self, batch_size = 1024, 
        device='cuda', 
        use_bidecoder = True
    ):

        self.batch_size = batch_size
        self.device = device

        self.train_data = [
#             '/data/text-recognition/train/CVPR2016/',
#             '/data/text-recognition/train/NIPS2014'
            "/data/data_lmdb_release/training/MJ/MJ_train",
            "/data/data_lmdb_release/training/ST/",
        ]
        
        self.use_bidecoder = use_bidecoder

        sys.stdout = Logger('./log.txt')

        '''
        'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
        '''
        self.lmdb_config = LmdbDatasetConfig(voc_type = 'ALLCASES_SYMBOLS')
        self.lmdb_config.use_bidecoder = self.use_bidecoder
        self.lmdb_config.num_samples   = 4*10000 #-\inf to 0 means use all data
        
        self.model_config = TextRecognitionModelConfig()
        self.model_config.num_classes = self.lmdb_config.num_classes
        self.model_config.max_len_labels = self.lmdb_config.max_len_labels
        self.model_config.eos = self.lmdb_config.char2id[self.lmdb_config.EOS]
        self.model_config.device = self.device
        self.model_config.use_bidecoder = self.use_bidecoder
        
        SEED = 1234
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True