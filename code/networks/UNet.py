# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpBlock_FBA(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 up_type=0):
        super(UpBlock_FBA, self).__init__()
        self.up_type = up_type
        if up_type == 0:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        elif up_type == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)

        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.up_type < 2:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        # self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=self.bilinear)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=self.bilinear)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=self.bilinear)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=self.bilinear)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class ContraModule(nn.Module):
    def __init__(self, channel, dim=128):
        super(ContraModule, self).__init__()

        self.activation_head = nn.Conv2d(channel, channel, 3, padding=1, stride=2, bias=False)
        self.f_mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())
        self.b_mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())

    def forward(self, x):
        # x: feature maps (output of U-Net)
        ccam = torch.sigmoid(self.activation_head(x))
        N, C, H, W = ccam.size()

        ccam_ = ccam.reshape(N, C, H * W)  # [N, C, H*W]
        fg_feats = ccam_ / (H * W)  # [N, C, H*W]
        bg_feats = (1 - ccam_) / (H * W)  # [N, C, H*W]

        fg_feats = self.f_mlp(fg_feats)
        bg_feats = self.b_mlp(bg_feats)
        return fg_feats, bg_feats


class ContraModuleNew(nn.Module):
    def __init__(self, channel, dim=128):
        super(ContraModuleNew, self).__init__()

        self.activation_head = nn.Conv2d(channel, channel, 3, padding=1, stride=2, bias=False)
        # self.bn_head = nn.BatchNorm2d(1)
        self.mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())
        # self.b_mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())

    def forward(self, x):
        # x: feature maps (output of U-Net)
        ccam = torch.sigmoid(self.activation_head(x))
        N, C, H, W = ccam.size()
        
        ccam_ = ccam.resize(N, C, H * W)
        ccam_ = self.mlp(ccam_)
        return ccam_
    


class Decoder_FBA(nn.Module):
    def __init__(self, params, n_classes=4, dim=128, new=False):
        super(Decoder_FBA, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_FBA(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, up_type=self.up_type)
        self.up2 = UpBlock_FBA(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, up_type=self.up_type)
        self.up3 = UpBlock_FBA(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, up_type=self.up_type)
        self.up4 = UpBlock_FBA(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, up_type=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        

        self.new = new
        if self.new:
            self.cm = ContraModuleNew(n_classes)
        else:
            self.cm = ContraModule(n_classes)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)

        if self.new:
            feats = self.cm(output)
            return feats, output
        else:
            fg_feats, bg_feats = self.cm(output)
            return fg_feats, bg_feats, output
    

def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': True,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output
    

class UNet_FBA2d(nn.Module):
    def __init__(self, in_chns, class_num, new=True):
        super(UNet_FBA2d3, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        self.new = new
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder_FBA(params1, new=self.new)
        self.decoder2 = Decoder_FBA(params2, new=self.new)
        self.decoder3 = Decoder_FBA(params3, new=self.new)
        
    def forward(self, x):
        feature = self.encoder(x)
        
        if self.new:
            feats1, output1 = self.decoder1(feature)
            feats2, output2 = self.decoder2(feature)
            feats3, output3 = self.decoder3(feature)
            return [feats1, feats2, feats3], [output1, output2, output3]
        else:
            fg_feats1, bg_feats1, output1 = self.decoder1(feature)
            fg_feats2, bg_feats2, output2 = self.decoder2(feature)
            fg_feats3, bg_feats3, output3 = self.decoder3(feature)
            return [fg_feats1, fg_feats2, fg_feats3], [bg_feats1, bg_feats2, bg_feats3], [output1, output2, output3]