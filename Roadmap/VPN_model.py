import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from torchvision import transforms, models

import copy
import torchgeometry.core as tgm 
class _DecoderBlock(nn.Module):
    """
    Taken from https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/seg_net.py
    """
    def __init__(self, in_channels, out_channels, num_conv_layers, out_activation = 'relu'):
        super(_DecoderBlock, self).__init__()
        middle_channels = int(in_channels / 2)
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if out_activation == 'relu' else nn.Sigmoid(),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)
    
class _SameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, out_activation = 'relu'):
        super(_SameDecoder, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if out_activation == 'relu' else nn.Sigmoid(),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)

# pyramid style decoder:
class PPMBilinear(nn.Module):
    def __init__(self, num_class=1, fc_dim=256,
                 pool_scales=(1, 2, 3, 6), out_size=800):
        super(PPMBilinear, self).__init__()
        self.out_size = out_size
        self.ppm = [nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                    nn.Conv2d(fc_dim, 256, kernel_size=1, bias= False), 
                    nn.ReLU(inplace=True))]
        for scale in pool_scales[1:]:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), 
                            nn.Conv2d(fc_dim, 256, kernel_size=1, bias= False), 
                            nn.BatchNorm2d(256), 
                            nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)

        self.pool_conv = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*256, 256,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            # nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last = nn.Sequential(*[_SameDecoder(256, 256), 
                                         _DecoderBlock(256, 128, 2),
                                         _DecoderBlock(128, 64, 2), 
                                         _DecoderBlock(64, 32, 2), 
                                         _DecoderBlock(32, 1, 2, 'Sigmoid')])
        
        # self.sigmoid = nn.Sigmoid()
    def forward(self, conv_out):
        conv5 = conv_out[-1]

        input_size = conv5.size()
             
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        # print(ppm_out.shape)
        x = self.pool_conv(ppm_out)
        #feat = x
        x  =self.conv_last(x)
        # print(x.shape)
        x = nn.functional.interpolate(x, (self.out_size,self.out_size), mode='bilinear', align_corners = False)
        
        return x

class TransformModule(nn.Module):
    '''
    Modified from https://github.com/pbw-Berwin/View-Parsing-Network/blob/dc0c4250302b84a8594f291a494b5e8969291e1b/segmentTool/models.py
    '''
    def __init__(self, dim1, dim2, num_view = 6):
        super(TransformModule, self).__init__()
        self.num_view = num_view
        self.dim1 = dim1
        self.dim2 = dim2
        self.mat_list = nn.ModuleList()
        
        for i in range(self.num_view):
            # weights are not shared
            fc_transform = nn.Sequential(
                        nn.Linear(dim1* dim2, dim1*dim2*4), 
                        nn.ReLU(), 
                        nn.Linear(dim1 * dim2*4, dim1*dim2),
                        nn.ReLU()
            )
            self.mat_list += [fc_transform]
    
    def forward(self, x):
        # shape B V C H W
        # flatten along the channel
        x = x.view(list(x.size()[:3]) + [self.dim1 * self.dim2,])
        # Transform the first image
        view_comb = self.mat_list[0](x[:, 0])
        for i in range(1, x.size(1)):
            # results are added(fusion func)
            view_comb += self.mat_list[i](x[:, i])
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.dim1, self.dim2]) 
        return view_comb
        

class vpn_model(nn.Module):
    def __init__(self, dim1, dim2, encoder, decoder):
        super(vpn_model, self).__init__()
        self.num_views = 6
        self.encoder = encoder
        
        self.transform = TransformModule(dim1=dim1, dim2=dim2)
        
        self.decoder = decoder
        
        
    def forward(self, x, return_feat = False):
        # flatten the output along channel: C x (HW)
        # weights are not shared, i.e. each first view input has
        # own VRM to get its top down view feature map 
        # i here in range 6(MN, N=6,M=1(MODALITY))
        # j here in range num_channels
        # 
        B,V,C,H,W = x.shape
        x = x.view(B*V, C, H, W)
        x = self.encoder(x)
        # return to B V 
        x = x.view([B,V] + list(x.size()[1:]))
        x =  self.transform(x) # B x c x h x w
        
        x = self.decoder([x])

        return x
    
class SpatialTransformer(nn.Module):

    def __init__(self, in_channels, kernel_size):
        '''
        Takes input in Bx 1024 x 16 x 20
        '''
        super(SpatialTransformer, self).__init__()
        self._in_ch = in_channels 
        self._ksize = kernel_size

        self.prep_warper = nn.Sequential(*[
            nn.Conv2d(self._in_ch, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
            
        ])
    
        self.warper_generator = nn.Sequential(*[
                    nn.Linear(32*8*10, 1024), 
                    nn.ReLU(inplace = True),
                    nn.Linear(1024, 9),
                    nn.Tanh()
        ])

    def forward(self, x): 
        """
        Forward pass of the STN module. 
        x -> input feature map 
        x should be the feature map for a single view
        """
        B, C, H, W = x.shape
        #localization net
        homo_mat = self.prep_warper(x)
        # concatenate 3 dim
        homo_mat = homo_mat.view(B, -1)
        
        homo_mat = self.warper_generator(homo_mat) # BV 3 X3 
        #reshape to homo matrix
        homo_mat = homo_mat.view(-1, 3, 3)
        # grid sample on original view
        warper = tgm.HomographyWarper(H, W)
        warpped = warper(x, homo_mat)
        return warpped
class ComplexTransformModule(nn.Module):
    def __init__(self, num_view=6):
        '''
        Takes in input B, V, C, H, W
        '''
        super(ComplexTransformModule, self).__init__()
        
        self.num_view = num_view

        self.mat_list = nn.ModuleList()
        
        for i in range(self.num_view):
            self.mat_list += [SpatialTransformer(1024, 3)]

    def forward(self, x):
        '''
        Takes in B,V,C,H, W, perform warpping on each image and concatenate by position
        '''
        B, V, C, H, W = x.size()
        view_comb = self.mat_list[0](x[:, 0])
        for i in range(1, V):
            view_comb += self.mat_list[i](x[:, i])
            # for each view, perform the warpped view
            #x[:, i] = self.mat_list[i](x[:, i])
        #Concatenate the view
        # x = position_concat_features(x)
        return view_comb

class vpn_model_v2(nn.Module):
    def __init__(self, encoder, decoder):
        super(vpn_model_v2, self).__init__()
        self.num_views = 6
        self.encoder = encoder
        
        self.transform = ComplexTransformModule()
        self.decoder = decoder
        
        
    def forward(self, x, return_feat = False):
        # flatten the output along channel: C x (HW)
        # weights are not shared, i.e. each first view input has
        # own VRM to get its top down view feature map 
        # i here in range 6(MN, N=6,M=1(MODALITY))
        # j here in range num_channels
        # 
        B,V,C,H,W = x.shape
        x = x.view(B*V, C, H, W)
        x = self.encoder(x)
        # return to B V 
        x = x.view([B,V] + list(x.size()[1:]))
        
        x =  self.transform(x) # B x c x h x w
        
        x = self.decoder([x])

        return x