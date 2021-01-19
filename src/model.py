#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from stn import TransformationConfig, Transformation
from crnn import ResNet, CRNN
from decoder import DecoderWithAttention

class TextRecognitionModelConfig:
    def __init__(self):
        self.device              = 'cuda'
        self.num_classes         = 97
        self.max_len_labels      = 100
        self.eos                 = 1

        self.with_STN            = True # add the STN layer

        self.attention_dim       = 512 # the dim for attention
        self.decoder_s_dim       = 512 # the dim of hidden layer in decoder
        self.use_bidecoder       = True

        self.with_beam_search    = False
        self.beam_width          = 5

class TextRecognitionModel(nn.Module):
    
    """
    This is the recognition integrated model.
    """
    
    def __init__(self, config = TextRecognitionModelConfig()):
        super().__init__()

        self.config = config
        
        self.cnn = ResNet()
        
        if self.config.with_STN:
            config_stn = TransformationConfig()
            # config_stn = TransformationConfig(self.cnn)
            # config_stn.outputsize = 256*2*16

            self.stn = Transformation(config_stn)

        self.encoder = CRNN(self.cnn)

        self.decoder = DecoderWithAttention(
            num_classes = config.num_classes,
            in_planes = self.encoder.out_planes,
            sDim = config.decoder_s_dim,
            attDim = config.attention_dim,
            max_len_labels = config.max_len_labels, 
            use_bidecoder = config.use_bidecoder,
            device = config.device,
        )

        for name, param in self.named_parameters():
            if 'fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param)
                else:
                    param.data.fill_(1)

    def forward(self, images, rec_targets, max_label_length = 0):
        '''
        images [batch_size, 64, 256, 3]
        rec_targets [batch_size, max_len_labels]
        '''
        
        if max_label_length>0:
            max_label_length = min(self.config.max_len_labels, max_label_length)
        else:
            max_label_length = self.config.max_len_labels
        
        return_dict = {}

        # normalize the images into x [batch_size, 3, 64, 256]
        # x = images.transpose(1, 2).transpose(1, 3).contiguous().float()
        x = images.permute(0,3,1,2).float()
        x.sub_(127.5).div_(127.5)
        
        # rectification
        if self.config.with_STN:
            if False and self.training and torch.rand(1).item()>0.5:
                x_new = F.interpolate(
                    x, 
                    (self.stn.target_height, self.stn.target_width), 
                    mode='bilinear', align_corners=True
                )
            else:
                output = self.stn(x, output_control_points = not self.training)

                if not self.training:
                    x_new, control_points, bias, weight = output
                    # save for visualization
                    return_dict['control_points'] = control_points
                    return_dict['rectified_images'] = x_new
                else:
                    x_new = output

        # x_new [batch_size, 3, 32, 100]
        encoder_feats = self.encoder(x_new)
        encoder_feats = encoder_feats.contiguous()

        if self.training or not self.config.with_beam_search:
            prediction = self.decoder(encoder_feats, rec_targets, max_label_length)
            return_dict['prediction'] = prediction
        else:
            
            prediction, prediction_scores = self.decoder.beam_search(
                encoder_feats, 
                self.config.beam_width, 
                self.config.eos)
            return_dict['prediction_beam'] = prediction
            return_dict['prediction_beam_score'] = prediction_scores

        return return_dict