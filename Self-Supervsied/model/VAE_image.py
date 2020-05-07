import os
import random

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
import time

# The following defines the vae mapping for image used for pretraining
class encoder_for_vae_image(nn.Module):

    def __init__(self):
        super(encoder_for_vae_image, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.mu_dec = nn.Linear(10240, 512)
        self.logvar_dec = nn.Linear(10240, 512)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 10240)
        mu = self.mu_dec(x)
        logvar = self.logvar_dec(x)

        return mu, logvar
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class _SameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_layers, out_activation = 'relu'):
        super(_SameDecoder, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        ]*(num_layers-2)
        
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if out_activation == 'relu' else nn.Sigmoid(),
        ]
        self.decode = nn.Sequential(*layers)
    def forward(self, x):
        return self.decode(x)
class decoder_conv_image(nn.Module):
    def __init__(self):
        super(decoder_conv_image, self).__init__()
        self.dec1 = _SameDecoder(128, 256,kernel_size=2, stride=2, num_layers=4)
        self.dec2 = _SameDecoder(256, 256,kernel_size=3, stride=3, num_layers=4)
        self.dec3 = _SameDecoder(256, 256,kernel_size=2, stride=2, num_layers=4)
        self.dec4 = _SameDecoder(256, 256,kernel_size=2, stride=2, num_layers=2)
        self.dec5 = _SameDecoder(256, 256,kernel_size=2, stride=2, num_layers=2)
        self.dec6 = _SameDecoder(256, 256,kernel_size=2, stride=2, num_layers=2)

        self.dec7 = _SameDecoder(256, 128,kernel_size=2, stride=2, num_layers=2)
        self.conv_out = nn.Conv2d(128, 3,3 , padding=1)
        self.final_upsample = nn.Upsample((256, 306), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
                
    def forward(self, x):
        x = x.view(-1, 128, 2, 2)
        x = self.dec1(x)
        x= self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.dec7(x)
        x = self.conv_out(x)
        # print('after layers the size is', x.size())
        x = self.final_upsample(x)
        x = self.sigmoid(x)
        return x
class vae_mapping_image(nn.Module):
    def __init__(self):
        super(vae_mapping_image, self).__init__()
        
        self.feature_extractor = torchvision.models.resnet50(pretrained = False)
        self.encoder = list(self.feature_extractor.children())[:-3]
        self.encoder = nn.Sequential(*self.encoder)
        self.encoder_for_vae = encoder_for_vae_image()
        self.decoder = decoder_conv_image()
        
    def reparameterize(self, is_training, mu, logvar):
        if is_training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x, is_training, defined_mu = None):
        feature_maps = self.encoder(x)
        mu, logvar = self.encoder_for_vae(feature_maps)
        z = self.reparameterize(is_training, mu, logvar)
        
        pred_map = self.decoder(z)
        
        return pred_map, mu, logvar