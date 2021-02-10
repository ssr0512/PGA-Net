# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Chenru Jiang (chenru.jiang@student.xjtlu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn




############################################################
from tensorboardX import SummaryWriter
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import sys
import os

cfg_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  ################get upper directory

work_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  ################get upper directory
sys.path.append(work_path)

import pprint

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

import numpy as np



BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)




class AttentionBottleneck(nn.Module):
    # channelExpansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=True):
        super(AttentionBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)

        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)



        if self.downsample is True:
            out = self.conv4(out)
            out = self.bn4(out)
            residual = self.conv4(residual)
            residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out



class Img_Attention_Block(nn.Module):

    def __init__(self, inplanes, planes):
        super(Img_Attention_Block, self).__init__()


        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv4 = nn.Conv2d(planes, int(planes / 4), kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(planes / 4), momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        # out = self.relu(out)

        return out





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest'),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):

            if len(self.fuse_layers) == 1:
                x_fuse = x  ### make original stage4 also give four outputs
                continue

            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)



        self.convAttention1x1_stage1_1 = nn.Conv2d(48, 48, kernel_size=1, bias=False)
        self.bnAttention1x1_stage1 = nn.BatchNorm2d(48, momentum=BN_MOMENTUM)
        self.convAttention1x1_stage1_2 = nn.Conv2d(48, 48, kernel_size=1, bias=False)

        self.convAttention1x1_stage2_1 = nn.Conv2d(48, 48, kernel_size=1, bias=False)
        self.bnAttention1x1_stage2 = nn.BatchNorm2d(48, momentum=BN_MOMENTUM)
        self.convAttention1x1_stage2_2 = nn.Conv2d(48, 48, kernel_size=1, bias=False)

        self.convAttention1x1_stage3_1 = nn.Conv2d(48, 48, kernel_size=1, bias=False)
        self.bnAttention1x1_stage3 = nn.BatchNorm2d(48, momentum=BN_MOMENTUM)
        self.convAttention1x1_stage3_2 = nn.Conv2d(48, 48, kernel_size=1, bias=False)


        self.convAttention_stage1_0 = nn.Conv2d(96, 96, kernel_size=1, bias=False)
        self.bnAttentionstage1_0 = nn.BatchNorm2d(96, momentum=BN_MOMENTUM)

        self.convAttention_stage2_0 = nn.Conv2d(96, 96, kernel_size=1, bias=False)
        self.bnAttentionstage2_0 = nn.BatchNorm2d(96, momentum=BN_MOMENTUM)

        self.convAttention_stage2_1 = nn.Conv2d(192, 192, kernel_size=1, bias=False)
        self.bnAttentionstage2_1 = nn.BatchNorm2d(192, momentum=BN_MOMENTUM)


        self.convAttention_stage3_0 = nn.Conv2d(96, 96, kernel_size=1, bias=False)
        self.bnAttentionstage3_0 = nn.BatchNorm2d(96, momentum=BN_MOMENTUM)

        self.convAttention_stage3_1 = nn.Conv2d(192, 192, kernel_size=1, bias=False)
        self.bnAttentionstage3_1 = nn.BatchNorm2d(192, momentum=BN_MOMENTUM)

        self.convAttention_stage3_2 = nn.Conv2d(384, 384, kernel_size=1, bias=False)
        self.bnAttentionstage3_2 = nn.BatchNorm2d(384, momentum=BN_MOMENTUM)


        self.sigmAttention = nn.Sigmoid()


        self.attentionMask_2_1 = self._make_attention_mask(AttentionBottleneck, 96, 96)
        self.attentionMask_3_1 = self._make_attention_mask(AttentionBottleneck, 96, 96)
        self.attentionMask_3_2 = self._make_attention_mask(AttentionBottleneck, 192, 192)

        self.attention_stage_1_1 = self._make_conv1x1_layer(96, 48)

        self.attention_stage_2_1 = self._make_conv1x1_layer(96, 48)
        self.attention_stage_2_2 = self._make_conv1x1_layer(192, 96)

        self.attention_stage_3_1 = self._make_conv1x1_layer(96, 48)
        self.attention_stage_3_2 = self._make_conv1x1_layer(192, 96)
        self.attention_stage_3_3 = self._make_conv1x1_layer(384, 192)



        self.imgMaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.imgAttenConv1 = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)
        self.imgAttenBn1 = nn.BatchNorm2d(48, momentum=BN_MOMENTUM)

        self.imgAttenConv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False)
        self.imgAttenBn2 = nn.BatchNorm2d(96, momentum=BN_MOMENTUM)

        self.imgAttenConv3 = nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False)
        self.imgAttenBn3 = nn.BatchNorm2d(192, momentum=BN_MOMENTUM)

        self.imgAttenConv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False)
        self.imgAttenBn4 = nn.BatchNorm2d(384, momentum=BN_MOMENTUM)



        self.imgAtten_up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.imgAtten_deConv3_2 = self._make_deconv_layer(384, 192)  # for branch 3 to 2
        self.imgAtten_conv3_2 = nn.Conv2d(384, 192, kernel_size=1, bias=False)

        self.imgAtten_deConv2_1 = self._make_deconv_layer(192, 96)  # for branch 2 to 1
        self.imgAtten_conv2_1 = nn.Conv2d(192, 96, kernel_size=1, bias=False)

        self.imgAtten_deConv1_0 = self._make_deconv_layer(96, 48)  # for branch 1 to 0
        self.imgAtten_conv1_0 = nn.Conv2d(96, 48, kernel_size=1, bias=False)

        self.imgAtten_conv2 = nn.Conv2d(384, 192, kernel_size=1, bias=False)
        self.imgAtten_bn2 = nn.BatchNorm2d(192, momentum=BN_MOMENTUM)

        self.imgAtten_conv1 = nn.Conv2d(192, 96, kernel_size=1, bias=False)
        self.imgAtten_bn1 = nn.BatchNorm2d(96, momentum=BN_MOMENTUM)

        self.imgAtten_conv0 = nn.Conv2d(96, 48, kernel_size=1, bias=False)
        self.imgAtten_bn0 = nn.BatchNorm2d(48, momentum=BN_MOMENTUM)
        self.RELU = nn.ReLU(inplace=True)




        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.attentionModule2 = self._make_attention_layer(AttentionBottleneck, 96, 96, 2) #original planes=32  ##############################
        self.deConv1 = self._make_deconv_layer(96, 96) #for branch 2 to 1  #####################################

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.attentionModule3 = self._make_attention_layer(AttentionBottleneck, 192, 192, 2) #planes=32  ##############################
        self.deConv2_1 = self._make_deconv_layer(96, 96)   #for branch 3 to 2  #########################################
        self.deConv2_2 = self._make_deconv_layer(192, 192)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels)


        self.imgAttention1 = self._make_img_attention(Img_Attention_Block, 3, 192)
        self.imgAttention2 = self._make_img_attention(Img_Attention_Block, 3, 384)
        self.imgAttention3 = self._make_img_attention(Img_Attention_Block, 3, 768)
        self.imgAttention4 = self._make_img_attention(Img_Attention_Block, 3, 1536)


        self.attentionModule4 = self._make_attention_layer(AttentionBottleneck, 384, 384, 2)
        self.deConv3_1 = self._make_deconv_layer(96, 96)
        self.deConv3_2 = self._make_deconv_layer(192, 192)
        self.deConv3_3 = self._make_deconv_layer(384, 384)


        self.stage5_cfg = cfg['MODEL']['EXTRA']['STAGE5']
        num_channels = self.stage5_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage5_cfg['BLOCK']]
        num_channels = [
           num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition4 = self._make_transition_layer(
           pre_stage_channels, num_channels)

        self.stage5, pre_stage_channels = self._make_stage(
           self.stage5_cfg, num_channels, multi_scale_output=False)


        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']




    def _make_conv1x1_layer(self, inplanes, planes,):

        layers = []
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_attention_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = False
        if stride != 1 or inplanes != planes:
            downsample = False
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))

        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)



    def _make_attention_mask(self, block, inplanes, planes, stride=1):
        downsample = False

        layer = []
        layer.append(block(inplanes, planes, stride, downsample))

        return nn.Sequential(*layer)


    def _make_deconv_layer(self, inplanes, planes):
        layer = []

        layer.append(
            nn.ConvTranspose2d(
                in_channels = inplanes,
                out_channels = planes,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                output_padding = 0,
                bias = False))
        layer.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    def _make_img_attention(self, block, inplanes, planes):
        layer = []

        layer.append(block(inplanes, planes))

        return nn.Sequential(*layer)




    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        img = self.imgMaxPool(x)
        img = self.imgMaxPool(img)


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)





        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)




        ## feature attention
        ##########################################################################################attenetion_stage_1
        S1_attention = []
        S1_attention.append(self.attentionModule2(y_list[1]))

        S1_attention_mask_1 = self.convAttention_stage1_0(S1_attention[0])
        S1_attention_mask_1 = self.bnAttentionstage1_0(S1_attention_mask_1)
        S1_attention_mask_1 = self.relu(S1_attention_mask_1)
        S1_attention_mask_1 = self.sigmAttention(S1_attention_mask_1)
        y_list[1] = torch.add(torch.mul(y_list[1], S1_attention_mask_1), y_list[1])

        S1_attention_mask_up = self.imgAtten_up(S1_attention[0])

        S1_attention_mask = self.deConv1(S1_attention[0])

        S1_attention_mask = torch.add(S1_attention_mask, S1_attention_mask_up)

        S1_attention_mask = self.attention_stage_1_1(S1_attention_mask)

        S1_attention_mask = self.convAttention1x1_stage1_1(S1_attention_mask)
        S1_attention_mask = self.bnAttention1x1_stage1(S1_attention_mask)
        S1_attention_mask = self.relu(S1_attention_mask)
        S1_attention_mask = self.convAttention1x1_stage1_2(S1_attention_mask)
        S1_attention_mask = self.sigmAttention(S1_attention_mask)

        y_list[0] = torch.add(torch.mul(y_list[0], S1_attention_mask), y_list[0])
        ##########################################################################################


        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)


        ##########################################################################################attenetion_stage_2
        S2_attention = []
        S2_attention.append(self.attentionModule2(y_list[1]))
        S2_attention.append(self.attentionModule3(y_list[2]))


        S2_attention_mask_2 = self.convAttention_stage2_1(S2_attention[1])
        S2_attention_mask_2 = self.bnAttentionstage2_1(S2_attention_mask_2)
        S2_attention_mask_2 = self.relu(S2_attention_mask_2)
        S2_attention_mask_2 = self.sigmAttention(S2_attention_mask_2)
        y_list[2] = torch.add(torch.mul(y_list[2], S2_attention_mask_2), y_list[2])


        S2_attention_mask_1 = self.convAttention_stage2_0(S2_attention[0])
        S2_attention_mask_1 = self.bnAttentionstage2_0(S2_attention_mask_1)
        S2_attention_mask_1 = self.relu(S2_attention_mask_1)
        S2_attention_mask_1 = self.sigmAttention(S2_attention_mask_1)
        y_list[1] = torch.add(torch.mul(y_list[1], S2_attention_mask_1), y_list[1])

        S2_attention_mask_up1 = self.imgAtten_up(S2_attention[1])
        S2_attention_mask_up0 = self.imgAtten_up(S2_attention[0])
        S2_attention_mask = self.deConv2_2(S2_attention[1])
        S2_attention_mask = torch.add(S2_attention_mask, S2_attention_mask_up1)
        S2_attention_mask = self.attention_stage_2_2(S2_attention_mask)
        S2_attention_mask = torch.add(S2_attention_mask, S2_attention[0])


        S2_attention_mask = self.attentionMask_2_1(S2_attention_mask)


        S2_attention_mask = self.deConv2_1(S2_attention_mask)
        S2_attention_mask = torch.add(S2_attention_mask, S2_attention_mask_up0)
        S2_attention_mask = self.attention_stage_2_1(S2_attention_mask)
        S2_attention_mask = self.convAttention1x1_stage2_1(S2_attention_mask)
        S2_attention_mask = self.bnAttention1x1_stage2(S2_attention_mask)
        S2_attention_mask = self.relu(S2_attention_mask)
        S2_attention_mask = self.convAttention1x1_stage2_2(S2_attention_mask)
        S2_attention_mask = self.sigmAttention(S2_attention_mask)


        y_list[0] = torch.add(torch.mul(y_list[0], S2_attention_mask), y_list[0])
        #########################################################################################


        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)


        ##########################################################################################attenetion_stage_3
        S3_attention = []
        S3_attention.append(self.attentionModule2(y_list[1]))
        S3_attention.append(self.attentionModule3(y_list[2]))
        S3_attention.append(self.attentionModule4(y_list[3]))


        S3_attention_mask_3 = self.convAttention_stage3_2(S3_attention[2])
        S3_attention_mask_3 = self.bnAttentionstage3_2(S3_attention_mask_3)
        S3_attention_mask_3 = self.relu(S3_attention_mask_3)
        S3_attention_mask_3 = self.sigmAttention(S3_attention_mask_3)
        y_list[3] = torch.add(torch.mul(y_list[3], S3_attention_mask_3), y_list[3])


        S3_attention_mask_2 = self.convAttention_stage3_1(S3_attention[1])
        S3_attention_mask_2 = self.bnAttentionstage3_1(S3_attention_mask_2)
        S3_attention_mask_2 = self.relu(S3_attention_mask_2)
        S3_attention_mask_2 = self.sigmAttention(S3_attention_mask_2)
        y_list[2] = torch.add(torch.mul(y_list[2], S3_attention_mask_2), y_list[2])

        S3_attention_mask_1 = self.convAttention_stage3_0(S3_attention[0])
        S3_attention_mask_1 = self.bnAttentionstage3_0(S3_attention_mask_1)
        S3_attention_mask_1 = self.relu(S3_attention_mask_1)
        S3_attention_mask_1 = self.sigmAttention(S3_attention_mask_1)
        y_list[1] = torch.add(torch.mul(y_list[1], S3_attention_mask_1), y_list[1])


        S3_attention_mask_up2 = self.imgAtten_up(S3_attention[2])
        S3_attention_mask_up1 = self.imgAtten_up(S3_attention[1])
        S3_attention_mask_up0 = self.imgAtten_up(S3_attention[0])


        S3_attention_mask = self.deConv3_3(S3_attention[2])
        S3_attention_mask = torch.add(S3_attention_mask, S3_attention_mask_up2)
        S3_attention_mask = self.attention_stage_3_3(S3_attention_mask)
        S3_attention_mask = torch.add(S3_attention_mask, S3_attention[1])


        S3_attention_mask = self.attentionMask_3_2(S3_attention_mask)
        S3_attention_mask = self.deConv3_2(S3_attention_mask)
        S3_attention_mask = torch.add(S3_attention_mask, S3_attention_mask_up1)
        S3_attention_mask = self.attention_stage_3_2(S3_attention_mask)
        S3_attention_mask = torch.add(S3_attention_mask, S3_attention[0])


        S3_attention_mask = self.attentionMask_3_1(S3_attention_mask)
        S3_attention_mask = self.deConv3_1(S3_attention_mask)
        S3_attention_mask = torch.add(S3_attention_mask, S3_attention_mask_up0)
        S3_attention_mask = self.attention_stage_3_1(S3_attention_mask)


        S3_attention_mask = self.convAttention1x1_stage3_1(S3_attention_mask)
        S3_attention_mask = self.bnAttention1x1_stage3(S3_attention_mask)
        S3_attention_mask = self.relu(S3_attention_mask)
        S3_attention_mask = self.convAttention1x1_stage3_2(S3_attention_mask)
        S3_attention_mask = self.sigmAttention(S3_attention_mask)


        y_list[0] = torch.add(torch.mul(y_list[0], S3_attention_mask), y_list[0])
        #########################################################################################

        x_list = []
        for i in range(self.stage5_cfg['NUM_BRANCHES']):
            if self.transition4[i] is not None:
                x_list.append(self.transition4[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage5(x_list)


        ###################################################################### image attention
        img_attention1 = self.imgAttention1(img)
        img_attention1= self.RELU(img_attention1)
        y_list[0] = torch.add(y_list[0], img_attention1) / 2
        y_list[0] = self.imgAttenConv1(y_list[0])
        y_list[0] = self.imgAttenBn1(y_list[0])
        y_list[0] = self.RELU(y_list[0])


        #####
        img = self.imgMaxPool(img)
        img_attention2 = self.imgAttention2(img)

        img_attention2= self.RELU(img_attention2)
        y_list[1] = torch.add(y_list[1], img_attention2) / 2
        y_list[1] = self.imgAttenConv2(y_list[1])
        y_list[1] = self.imgAttenBn2(y_list[1])
        y_list[1] = self.RELU(y_list[1])

        #####
        img = self.imgMaxPool(img)
        img_attention3 = self.imgAttention3(img)
        img_attention3= self.RELU(img_attention3)

        y_list[2] = torch.add(y_list[2], img_attention3) / 2
        y_list[2] = self.imgAttenConv3(y_list[2])
        y_list[2] = self.imgAttenBn3(y_list[2])
        y_list[2] = self.RELU(y_list[2])


        #####
        img = self.imgMaxPool(img)
        img_attention4 = self.imgAttention4(img)
        img_attention4= self.RELU(img_attention4)
        y_list[3] = torch.add(y_list[3], img_attention4) / 2
        y_list[3] = self.imgAttenConv4(y_list[3])
        y_list[3] = self.imgAttenBn4(y_list[3])
        y_list[3] = self.RELU(y_list[3])


        ### 3-2
        y_list_final3_D = self.imgAtten_deConv3_2(y_list[3])
        y_list_final3_Up = self.imgAtten_up(y_list[3])
        y_list_final3_Up = self.imgAtten_conv3_2(y_list_final3_Up)

        y_list_final3 = torch.add(y_list_final3_D, y_list_final3_Up)

        ### 2-1
        y_list[2] = torch.cat((y_list[2], y_list_final3), 1)
        y_list[2] = self.imgAtten_conv2(y_list[2])
        y_list[2] = self.imgAtten_bn2(y_list[2])
        y_list[2] = self.RELU(y_list[2])

        y_list_final2_D = self.imgAtten_deConv2_1(y_list[2])
        y_list_final2_Up = self.imgAtten_up(y_list[2])
        y_list_final2_Up = self.imgAtten_conv2_1(y_list_final2_Up)

        y_list_final2 = torch.add(y_list_final2_D, y_list_final2_Up)

        ### 1-0
        y_list[1] = torch.cat((y_list[1], y_list_final2), 1)
        y_list[1] = self.imgAtten_conv1(y_list[1])
        y_list[1] = self.imgAtten_bn1(y_list[1])
        y_list[1] = self.RELU(y_list[1])

        y_list_final1_D = self.imgAtten_deConv1_0(y_list[1])
        y_list_final1_Up = self.imgAtten_up(y_list[1])
        y_list_final1_Up = self.imgAtten_conv1_0(y_list_final1_Up)

        y_list_final1 = torch.add(y_list_final1_D, y_list_final1_Up)

        ### 0
        y_list[0] = torch.cat((y_list[0], y_list_final1), 1)
        y_list[0] = self.imgAtten_conv0(y_list[0])
        y_list[0] = self.imgAtten_bn0(y_list[0])
        y_list[0] = self.RELU(y_list[0])

        ##############################################################################################

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model




