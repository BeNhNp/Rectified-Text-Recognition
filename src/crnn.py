#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()
        self.inplanes = 32
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(self.inplanes,  3, [2, 2]) # [8, 25]
        self.layer2 = self._make_layer(64,  4, [2, 2]) # [8, 25]
        self.layer3 = self._make_layer(128, 6, [2, 1]) # [4, 25]
        self.layer4 = self._make_layer(256, 6, [2, 1]) # [2, 25]
        self.layer5 = self._make_layer(512, 3, [2, 1]) # [1, 25]
    
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(planes))

        layers = []
        layers.append(ResNetBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResNetBlock(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        '''
        input shape: [batch_size, 3, 32, 100]
        '''
        x0 = self.layer0(x) #[batch_size, 16, 32, 100]
        x1 = self.layer1(x0) #[batch_size, 32, 16, 50]
        x2 = self.layer2(x1) #[batch_size, 64, 8, 25]
        x3 = self.layer3(x2) #[batch_size, 128, 4, 25]
        x4 = self.layer4(x3) #[batch_size, 256, 2, 25]
        x5 = self.layer5(x4) #[batch_size, 512, 1, 25]
        return x5

class CRNN(nn.Module):
    """
    extract image feature and transform into sequence feature
    """

    def __init__(self, cnn = None):
        super().__init__()
    
        if cnn is None:
            self.cnn = ResNet()
        else:
            self.cnn = cnn
        
        self.rnn = nn.LSTM(512, 256, 
            bidirectional=True, num_layers=2, 
            dropout=0.2,
            batch_first=True
        )

        self.out_planes = 2 * 256
        
    def forward(self, x):
        '''
        x shape: [batch_size, 3, 32, 100]
        '''
        x5 = self.cnn(x) # [batch_size, c, 1, w]
        cnn_feat = x5.squeeze(2) # [batch_size, c, w] i.e. [batch_size, 512, 25]
        cnn_feat = cnn_feat.transpose(2, 1)
        self.rnn.flatten_parameters()
        rnn_feat, _ = self.rnn(cnn_feat)
        return rnn_feat