# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn



import math
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""

    return nn.Sequential(
        nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
                  padding=1, groups=in_planes, bias=False),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    )

    # return nn.Sequential(
    #     nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #               padding=1, bias=False)
    # )


# class conv3x3(nn.Module):
#     def __init__(self,  in_planes, out_planes, stride=1):
#         super(conv3x3, self).__init__()
#         self.depth_conv3x3 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
#                      padding=1, groups=in_planes, bias=False)
#         self.point_conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#     def forward(self, x):
#         out = self.depth_conv3x3(x)
#         out = self.point_conv3x3(out)
#         return out


#################################################################
# def  efficient_mul(inputs, efficient=True):
#     if efficient:
#         return checkpoint(lambda x: torch.mul(x, 1), inputs)

def dilatconv(in_planes, out_planes, stride=1, pad=1, dilate=1):

    return nn.Sequential(
        nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
                     padding=pad, dilation=dilate, groups=in_planes, bias=False),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    )

    # return nn.Sequential(
    #     nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=pad, dilation=dilate, bias=False)
    # )



class q_transform(nn.Conv2d):
    """Conv2d for q_transform"""


class k_transform(nn.Conv2d):
    """Conv2d for q_transform"""


class v_transform(nn.Conv2d):
    """Conv2d for q_transform"""


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=64,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.q_transform = q_transform(in_planes, out_planes // 2, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.k_transform = k_transform(in_planes, out_planes // 2, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.v_transform = v_transform(in_planes, out_planes, kernel_size=1, stride=1,
                                       padding=0, bias=False)

        self.bn_q = nn.BatchNorm2d(out_planes // 2)
        self.bn_k = nn.BatchNorm2d(out_planes // 2)
        self.bn_v = nn.BatchNorm2d(out_planes)

        self.bn_qk = nn.BatchNorm2d(groups)
        self.bn_qr = nn.BatchNorm2d(groups)
        self.bn_kr = nn.BatchNorm2d(groups)

        self.bn_sv = nn.BatchNorm2d(out_planes)
        self.bn_sve = nn.BatchNorm2d(out_planes)

        # Position embedding
        self.q_relative = nn.Parameter(torch.randn(self.group_planes // 2, kernel_size * 2 - 1, 1), requires_grad=True)
        self.k_relative = nn.Parameter(torch.randn(self.group_planes // 2, kernel_size * 2 - 1, 1), requires_grad=True)
        self.v_relative = nn.Parameter(torch.randn(self.group_planes, kernel_size * 2 - 1, 1), requires_grad=True)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.transpose(2, 3)
        N, C, H, W = x.shape

        self.kernel_size = H   ##############################################
        # Transformations
        q = self.q_transform(x)
        q = self.bn_q(q)
        k = self.k_transform(x)
        k = self.bn_k(k)
        v = self.v_transform(x)
        v = self.bn_v(v)

        # Calculate position embedding
        q_embedding = []
        k_embedding = []
        v_embedding = []
        for i in range(self.kernel_size):
            q_embedding.append(self.q_relative[:, self.kernel_size - 1 - i: self.kernel_size * 2 - 1 - i])
            k_embedding.append(self.k_relative[:, self.kernel_size - 1 - i: self.kernel_size * 2 - 1 - i])
            v_embedding.append(self.v_relative[:, self.kernel_size - 1 - i: self.kernel_size * 2 - 1 - i])
        q_embedding = torch.cat(q_embedding, dim=2)
        k_embedding = torch.cat(k_embedding, dim=2)
        v_embedding = torch.cat(v_embedding, dim=2)


        # a = q.reshape(N, self.groups, self.group_planes // 2, H, W)  ####################
        qr = torch.einsum('bgciw, cij->bgijw', q.reshape(N, self.groups, self.group_planes // 2, H, W), q_embedding)
        qr = self.bn_qr(qr.reshape(N, self.groups, -1, W)).reshape(N, self.groups, H, H, W)

        kr = torch.einsum('bgciw, cij->bgijw', k.reshape(N, self.groups, self.group_planes // 2, H, W), k_embedding)
        kr = self.bn_kr(kr.reshape(N, self.groups, -1, W)).reshape(N, self.groups, H, H, W)
        kr = kr.transpose(2, 3)

        # Blocks of axial attention
        q = q.reshape(N, self.groups, self.group_planes // 2, H, W)
        k = k.reshape(N, self.groups, self.group_planes // 2, H, W)

        # (q, k)
        qk = torch.einsum('bgciw, bgcjw->bgijw', q, k)
        qk = self.bn_qk(qk.reshape(N, self.groups, -1, W)).reshape(N, self.groups, H, H, W)

        # (N, groups, H, H, W)
        similarity = F.softmax(qk + qr + kr, dim=3)
        # b = v.reshape(N, self.groups, self.group_planes, H, W)   #########################
        sv = torch.einsum('bgijw, bgcjw->bgciw', similarity, v.reshape(N, self.groups, self.group_planes, H, W))
        sve = torch.einsum('bgijw, cji->bgciw', similarity, v_embedding)
        output = self.bn_sv(sv.reshape(N, -1, H, W)) + self.bn_sve(sve.reshape(N, -1, H, W))

        if self.width:
            output = output.transpose(2, 3)

        # if self.stride > 1:
        #     output = self.pooling(output)

        return output


    def reset_parameters(self):
        n = self.in_planes * self.group_planes
        self.q_transform.weight.data.normal_(0, math.sqrt(1. / n))
        n = self.in_planes
        self.k_transform.weight.data.normal_(0, math.sqrt(1. / n))
        self.v_transform.weight.data.normal_(0, math.sqrt(1. / n))
        n = self.out_planes // 2
        nn.init.normal_(self.q_relative, 0, math.sqrt(1. / n))
        nn.init.normal_(self.k_relative, 0, math.sqrt(1. / n))
        n = self.out_planes
        nn.init.normal_(self.v_relative, 0, math.sqrt(1. / n))


class conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(conv1x1, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)


        return out


class dense_conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(dense_conv, self).__init__()

        self.bn = nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)

        return out



class img_conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(img_conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class SKLayer(nn.Module):    ##########  SKNet
    def __init__(self, channel, reduction):
        super(SKLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel, channel // reduction, bias=False)

        self.fc0 = nn.Linear(channel // reduction, channel // 2, bias=False)
        self.fc1 = nn.Linear(channel // reduction, channel // 4, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel // 4, bias=False)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b,c)
        out = self.fc(out)

        out0 = self.fc0(out)
        out1 = self.fc1(out)
        out2 = self.fc2(out)

        out = torch.cat((out0, out1, out2), 1)
        out = self.softmax(out)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, stride=1):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2,1,kernel_size=3, stride=stride, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return out


class SALayer(nn.Module):   #########
    def __init__(self, in_planes, out_planes, stride=1):
        super(SALayer, self).__init__()

        self.depth_conv = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.point_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        out = self.bn(out)
        out = self.relu(out)

        return out

class SpatialMask(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(SpatialMask, self).__init__()


        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.sigmoid(out)

        return out


class SELayer(nn.Module):   ##########original  SENet
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel, bias=False),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        # b, c, _, _ = x.size()
        y = self.avg_pool(x)#.view(b,c)
        y = self.fc(y)

        # y = self.fc(y).view(b,c,1,1)
        # return x * y.expand_as(x)

        return y




class Img_Attention_Block(nn.Module):

    def __init__(self, inplanes, planes, reduction):
        super(Img_Attention_Block, self).__init__()


        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        # self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes//reduction, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes//reduction, momentum=BN_MOMENTUM)

        # self.bn3 = nn.BatchNorm2d(planes//reduction, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes//reduction, planes//reduction, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes//reduction, momentum=BN_MOMENTUM)

        # self.bn4 = nn.BatchNorm2d(planes//reduction, momentum=BN_MOMENTUM)
        self.conv4 = nn.Conv2d(planes//reduction, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)


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
        out = self.relu(out)

        return out
#################################################################

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(3, 5, 7), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        # if dimension == 1:
        #     prior = nn.AdaptiveAvgPool1d(output_size=size)
        if dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        # elif dimension == 3:
        #     prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class _SelfAttentionBlock(nn.Module):
    # '''
    # The basic implementation for self-attention block/non-local block
    # Input:c
    #     N X C X H X W
    # Parameters:
    #     in_channels       : the dimension of the input feature map
    #     key_channels      : the dimension after the key/query transform
    #     value_channels    : the dimension after the value transform
    #     scale             : choose the scale to downsample the input feature maps (save memory cost)
    # Return:
    #     N X C X H X W
    #     position-aware context features.(w/o concate or add with the input)
    # '''
    def __init__(self, low_in_channel, high_in_channel, neck_channel=16, psp_size=(3,5,7)):
        super(_SelfAttentionBlock, self).__init__()
        # self.scale = scale
        # self.in_channels = low_in_channels
        self.high_in_channel = high_in_channel
        # self.out_channels = out_channels
        self.neck_channel = neck_channel
        # self.value_channels = value_channels
        # if out_channels == None:
        #     self.out_channels = high_in_channels
        # self.pool = nn.MaxPool2d(kernel_size=(scale, scale))

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channel, out_channels=self.neck_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.neck_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channel, out_channels=self.neck_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.neck_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.f_value = nn.Conv2d(in_channels=high_in_channel, out_channels=high_in_channel,
                                 kernel_size=1, stride=1, bias=False)
        # self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
        #                    kernel_size=1, stride=1, bias=False)

        self.psp = PSPModule(psp_size)

        # nn.init.constant_(self.W.weight, 0)
        # nn.init.constant_(self.W.bias, 0)

    def forward(self, branch0_feats):

        batch_size, channel, h, w = branch0_feats.shape
        # if self.scale > 1:
        #     x = self.pool(x)

        value = self.psp(self.f_value(branch0_feats))

        query = self.f_query(branch0_feats).view(batch_size, self.neck_channel, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(branch0_feats)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.neck_channel ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.high_in_channel, *branch0_feats.size()[2:])
        # context = self.W(context)
        return context


######################################################################################################
class PatchSelfAttention(nn.Module):

    def __init__(self, low_in_channel, high_in_channel, neck_channel=16, patch_ratio=1):
        super(PatchSelfAttention, self).__init__()

        self.high_in_channel = high_in_channel
        self.neck_channel = neck_channel
        self.patch_ratio = patch_ratio
        self.low_in_channel = low_in_channel


        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channel, out_channels=self.neck_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.neck_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.high_in_channel, out_channels=self.neck_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.neck_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.f_value = nn.Conv2d(in_channels=self.high_in_channel, out_channels=self.neck_channel,
                                 kernel_size=1, stride=1, bias=False)


        self.up_patch = nn.Upsample(scale_factor=patch_ratio, mode='bilinear', align_corners=True)
        self.Wz = nn.Conv2d(in_channels=self.neck_channel, out_channels=self.high_in_channel,
                           kernel_size=1, stride=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, branch0_feats, other_feats):

        value = self.f_value(branch0_feats)
        query = self.f_query(branch0_feats)


        key = self.f_key(other_feats)
        key = self.up_patch(key)


        sim_map = torch.mul(query, key)

        # sim_map = self.softmax(sim_map)
        sim_map = self.sigmoid(sim_map)

        context = torch.mul(sim_map, value)

        context = self.Wz(context)
        return context



def transform(patch_num_axial, patch_size_H, patch_size_W, x):

    n,c,H,W = x.shape

    patch = []

    for i in range(patch_num_axial):
        for j in range(patch_num_axial):
            temp = x[:,:,i * patch_size_H:(i + 1) * patch_size_H, j * patch_size_W:(j + 1) * patch_size_W].reshape(n,c,-1).unsqueeze(2)
            patch.append(temp)

    patch = torch.cat(patch, 2)
    return patch


def transformBack(patch_num_axial, patch_size_H, patch_size_W, x):

    n, c, patch_num, _ = x.shape

    x = x.reshape(n, c, patch_num, patch_size_H, patch_size_W)
    patch = []

    for j in range(patch_num):
        patch.append(x[:,:,j,:,:])

    patchBack = []

    for i in range(0, patch_num, patch_num_axial):
        temp = torch.cat((patch[i:i+patch_num_axial]), dim=3)
        patchBack.append(temp)

    patchBack = torch.cat(patchBack, 2)
    return patchBack


class PatchSelfAttention2(nn.Module):

    def __init__(self, low_in_channel, high_in_channel, neck_channel=16, patch_ratio=1, k=1):
        super(PatchSelfAttention2, self).__init__()

        self.high_in_channel = high_in_channel
        self.neck_channel = neck_channel
        self.patch_ratio = patch_ratio
        self.low_in_channel = low_in_channel


        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channel, out_channels=self.neck_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.neck_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.high_in_channel, out_channels=self.neck_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.neck_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        #
        # self.f_key_loss = nn.Sequential(
        #     nn.Conv2d(in_channels=self.neck_channel, out_channels=1,
        #               kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(1, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True)
        # )
        # self.f_query_loss = nn.Sequential(
        #     nn.Conv2d(in_channels=self.neck_channel, out_channels=1,
        #               kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(1, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True)
        # )


        self.f_value = nn.Conv2d(in_channels=self.high_in_channel, out_channels=self.neck_channel,
                                 kernel_size=1, stride=1, bias=False)


        self.f_key2 = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channel, out_channels=self.neck_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.neck_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.f_query2 = nn.Sequential(
            nn.Conv2d(in_channels=self.high_in_channel, out_channels=self.neck_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.neck_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.f_value2 = nn.Conv2d(in_channels=self.high_in_channel, out_channels=self.neck_channel,
                                 kernel_size=1, stride=1, bias=False)

        self.up_patch = nn.Upsample(scale_factor=patch_ratio, mode='bilinear', align_corners=True)
        self.Wz = nn.Conv2d(in_channels=self.neck_channel * 2, out_channels=self.high_in_channel,
                           kernel_size=1, stride=1, bias=False)

        # self.Wz2 = nn.Conv2d(in_channels=self.neck_channel*3, out_channels=self.high_in_channel,
        #                    kernel_size=1, stride=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.zeros(1))
        # self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, branch0_feats, other_feats):

        _, _, H, W = branch0_feats.shape
        _, _, Hs, Ws = other_feats.shape

        value = self.f_value(branch0_feats)
        query = self.f_query(branch0_feats)

        key = self.f_key(other_feats)
        key = self.up_patch(key)

        nq, cq, _, _ = query.shape

        query_patch1 = query[:,:cq//2,:,:]
        query_patch2 = query[:,cq//2:,:,:]

        key_patch1 = key[:,:cq//2,:,:]
        key_patch2 = key[:,cq//2:,:,:]

        value_patch1 = value[:,:cq//2,:,:]
        value_patch2 = value[:,cq//2:,:,:]

        # a = value[0, :, :, :]        ############ print feature map
        # a = torch.mean(a, 0)
        # unloader = transforms.ToPILImage()
        # image = a.cpu().clone()
        # image = unloader(image)
        # image.save('/home/chenru/UDP_HRNet_work75/output/mpii/pose_hrnet/1_before.jpg')

        # query_loss = self.f_query_loss(query)
        # key_loss = self.f_key_loss(key)


        value2 = value
        query2 = self.f_query2(branch0_feats)
        key2 = self.f_key2(other_feats)
        key2 = self.up_patch(key2)


        ############# patch attention
        query_patch1 = query_patch1.view(nq, cq//2, H//Hs, Hs, W//Ws, Ws).permute(0,1,2,4,3,5)
        query_patch1 = torch.reshape(query_patch1, (nq, cq//2, (H//Hs)*(W//Ws), Hs*Ws))

        key_patch1 = key_patch1.view(nq, cq//2, H//Hs, Hs, W//Ws, Ws).permute(0,1,2,4,3,5)
        key_patch1 = torch.reshape(key_patch1, (nq, cq//2, (H//Hs)*(W//Ws), Hs*Ws))

        value_patch1 = value_patch1.view(nq, cq//2, H//Hs, Hs, W//Ws, Ws).permute(0,1,2,4,3,5)
        value_patch1 = torch.reshape(value_patch1, (nq, cq//2, (H//Hs)*(W//Ws), Hs*Ws))

        # query = transform(patch_num_axial=H//Hs, patch_size_H=Hs, patch_size_W=Ws, x=query)
        # key = transform(patch_num_axial=H//Hs, patch_size_H=Hs, patch_size_W=Ws, x=key)
        # value = transform(patch_num_axial=H//Hs, patch_size_H=Hs, patch_size_W=Ws, x=value)




        # query= transform(patch_num_axial=H//8, patch_size_H=H//8, patch_size_W=W//8, x=query)
        # key = transform(patch_num_axial=H//8, patch_size_H=H//8, patch_size_W=W//8, x=key)
        # value = transform(patch_num_axial=H//8, patch_size_H=H//8, patch_size_W=W//8, x=value)

        # sim_map = torch.einsum('bgiw, bgjw->bgij', query, key)

        # query_loss = torch.mean(query, 1)
        # key_loss = torch.mean(key, 1)
        # # query_loss = torch.mean(query_loss, -1).unsqueeze(-1)
        # # key_loss = torch.mean(key_loss, -1).unsqueeze(-1)
        #
        # # query_loss= transform(patch_num_axial=H//Hs, patch_size_H=Hs, patch_size_W=Ws, x=query_loss)
        # # key_loss = transform(patch_num_axial=H//Hs, patch_size_H=Hs, patch_size_W=Ws, x=key_loss)
        # key_loss = key_loss.permute(0,2,1)
        # sim_map_loss = torch.matmul(query_loss, key_loss)
        # # sim_map_loss = sim_map_loss.squeeze(1)
        # Max = sim_map_loss.max()
        # Min = sim_map_loss.min()
        # sim_map_loss = (sim_map_loss-Min)/(Max-Min)
        #
        # sim_map_loss = self.sigmoid(sim_map_loss)



        # n, c, _, _ = query.shape
        # query = self.f_query_k(query)
        # key = self.f_key_k(key)
        # query = query.reshape(n,c,-1).unsqueeze(-1)
        # key = key.reshape(n,c,-1).unsqueeze(-1)


        key_patch1 = key_patch1.permute(0,1,3,2)
        sim_map = torch.matmul(query_patch1, key_patch1)

        # sim_map_back = sim_map  #############################

        sim_map = self.sigmoid(sim_map)
        # sim_map = self.softmax(sim_map)

        #########################################
        sim_map1 = torch.mean(sim_map, -1).unsqueeze(-1)
        sim_map2 = torch.mean(sim_map, 2).unsqueeze(-1)

        context0 = torch.mul(sim_map1, value_patch1)
        # context = self.alpha * context

        context1 = torch.mul(sim_map2, value_patch1)
        # context1 = self.beta * context1

        context0 = context0 + context1

        # context = torch.cat((context0, context1), 1)




        # sim_map = torch.mean(sim_map, -1).unsqueeze(-1)
        # context = torch.mul(sim_map, value)

        context0 = context0.view(nq, cq//2, H//Hs, W//Ws, Hs, Ws).permute(0,1,2,4,3,5)
        context0 = torch.reshape(context0, (nq, cq//2, H, W))

        ##########################################################################################################


        # context = transformBack(patch_num_axial=H // Hs, patch_size_H=Hs, patch_size_W=Ws, x=context)
        # context = transformBack(patch_num_axial=H // 8, patch_size_H=H//8, patch_size_W=W//8, x=context)

        query_patch2 = query_patch2.view(nq, cq//2, H//Hs*2, Hs//2, W//Ws*2, Ws//2).permute(0, 1, 2, 4, 3, 5)
        query_patch2 = torch.reshape(query_patch2, (nq, cq//2, (H//Hs*2)*(W//Ws*2),Hs//2*Ws//2))

        key_patch2 = key_patch2.view(nq, cq//2, H//Hs*2, Hs//2, W//Ws*2, Ws//2).permute(0, 1, 2, 4, 3, 5)
        key_patch2 = torch.reshape(key_patch2, (nq, cq//2, (H//Hs*2)*(W//Ws*2),Hs//2*Ws//2))

        value_patch2 = value_patch2.view(nq, cq//2, H//Hs*2, Hs//2, W//Ws*2, Ws//2).permute(0, 1, 2, 4, 3, 5)
        value_patch2 = torch.reshape(value_patch2, (nq, cq//2, (H//Hs*2)*(W//Ws*2),Hs//2*Ws//2))


        key_patch2 = key_patch2.permute(0, 1, 3, 2)
        sim_map = torch.matmul(query_patch2, key_patch2)



        sim_map = self.sigmoid(sim_map)
        # sim_map = self.softmax(sim_map)

        #########################################
        sim_map1 = torch.mean(sim_map, -1).unsqueeze(-1)
        sim_map2 = torch.mean(sim_map, 2).unsqueeze(-1)

        context2 = torch.mul(sim_map1, value_patch2)
        # context = self.alpha * context

        context2_1 = torch.mul(sim_map2, value_patch2)
        # context1 = self.beta * context1

        context2 = context2 + context2_1

        context2 = context2.view(nq, cq//2, H//Hs*2, Hs//2, W//Ws*2, Ws//2).permute(0, 1, 2, 4, 3, 5)
        context2 = torch.reshape(context2, (nq, cq//2, H, W))

        context = torch.cat((context0, context2), 1)

        # a = context[0, :, :, :]
        # a = torch.mean(a, 0)
        # unloader = transforms.ToPILImage()
        # image = a.cpu().clone()
        # image = unloader(image)
        # image.save('/home/chenru/UDP_HRNet_work75/output/mpii/pose_hrnet/1_after.jpg')

        context = self.alpha * context
        #########################################

        # context = torch.einsum('bgmn, bgnh->bgmh', sim_map, value)
        # context = torch.matmul(sim_map, value)



        ###########################################
        sim_map = torch.mul(query2, key2)
        sim_map = self.sigmoid(sim_map)
        # context = context + sim_map


        # a = torch.mean(sim_map3, 1).unsqueeze(1)
        # a = a[0,:,:,:]
        # unloader = transforms.ToPILImage()
        # image = a.cpu().clone()
        # image = unloader(image)
        # image.save('/home/chenru/UDP_HRNet_work75/output/mpii/pose_hrnet/1.jpg')


        context3 = value2 * sim_map
        context3 = self.gamma * context3
        ###########################################

        context = torch.cat((context, context3), 1)

        context = self.Wz(context)
        ###############################






        ####### simplify patch attention
        # sim_map = torch.mul(query, key)
        #
        # # sim_map = self.softmax(sim_map)
        # sim_map = self.sigmoid(sim_map)
        #
        # context = torch.mul(sim_map, value)
        #
        # context = self.Wz(context)

        ######################################
        # sim_map_loss = torch.mean(sim_map_loss, 1)
        # Max = sim_map_loss.max()
        # Min = sim_map_loss.min()
        # sim_map_loss = (sim_map_loss-Min)/(Max-Min)
        # sim_map_back = (sim_map_back-0.5)*2
        # sim_map_back = self.sigmoid(sim_map_back)
        ######################################

        return context#, sim_map_loss


class FuseBlock(nn.Module):
    def __init__(self):
        super(FuseBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.UP1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.UP2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.UP3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        y_list1 = self.conv1(x[1])
        y_list1 = self.bn1(y_list1)
        y_list1 = self.UP1(y_list1)

        y_list2 = self.conv2(x[2])
        y_list2 = self.bn2(y_list2)
        y_list2 = self.UP2(y_list2)

        y_list3 = self.conv3(x[3])
        y_list3 = self.bn3(y_list3)
        y_list3 = self.UP3(y_list3)

        y_list = x[0] + y_list1 + y_list2 + y_list3
        y_list = self.relu(y_list)

        return y_list
######################################################################################################



class BilinearGate(nn.Module):
    def __init__(self, inplane, outplane, reduction):
        super(BilinearGate, self).__init__()

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels=inplane, out_channels=outplane//reduction, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(outplane//reduction, momentum=BN_MOMENTUM)
        )

        self.spatialRecover = nn.Sequential(
            nn.Conv2d(in_channels=(outplane//reduction)*(outplane//reduction+1)//2, out_channels=outplane, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(outplane, momentum=BN_MOMENTUM)
        )

        self.channelRecover = nn.Sequential(
            nn.Conv2d(in_channels=(outplane//reduction)*(outplane//reduction+1)//2, out_channels=outplane, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(outplane, momentum=BN_MOMENTUM)
        )

        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.reduce(x)

        out = torch.unsqueeze(out, 2)
        out_T = out.permute(0,2,1,3,4)

        bilinear = torch.einsum('ijlkn,ilmkn->ijmkn', out, out_T)
        _, c1, c2, _, _ = bilinear.shape
        idx = torch.triu_indices(c1,c2)

        Up_triangle =bilinear[:, idx[0], idx[1], :, :]

        # channel_atten = self.avgPool(Up_triangle)
        Up_triangle = self.channelRecover(Up_triangle)

        Up_triangle = torch.sign(Up_triangle) * torch.sqrt(torch.abs(Up_triangle) + 1e-6)

        # spatial_atten = self.spatialRecover(Up_triangle)
        #
        # spatial_atten = torch.mul(spatial_atten, channel_atten)
        # spatial_atten = self.sigmoid(spatial_atten)

        out = Up_triangle

        return out





class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1,
                      kernel_size=3, stride=1, padding=1, bias=False)
            # nn.BatchNorm2d(1, momentum=BN_MOMENTUM)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.spatial(out)
        scale = self.sigmoid(out)

        return x*scale

class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()

        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()

        x_out = self.SpatialGate(x)
        x_out = (x_out + x_out11 + x_out21)/3

        return x_out

class BasicBlock(nn.Module):  #####SENet + new type residual
    expansion = 1

    def __init__(self, inplanes, planes, reduction, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.split = inplanes*1//4

        self.conv1x1_Before = conv1x1(inplanes, planes)

        self.conv1x1_Before1 = conv1x1(inplanes//4, planes//4)
        self.conv1x1_Before2 = conv1x1(inplanes // 4, planes // 4)
        self.conv1x1_Before3 = conv1x1(inplanes // 4, planes // 4)
        self.conv1x1_Before4 = conv1x1(inplanes // 4, planes // 4)

        self.conv1x1_After = conv1x1(inplanes, planes)



        self.conv11 = conv3x3(inplanes, planes//2, stride)
        # self.conv12 = conv3x3(inplanes // 4, planes // 4, stride)



        self.conv21 = dilatconv(inplanes // 2, planes // 4, pad=2, dilate=2)
        # self.conv22 = dilatconv(inplanes // 4, planes // 4, pad=2, dilate=2)


        self.conv31 = dilatconv(inplanes // 4, planes // 8, pad=3, dilate=3)
        # self.conv32 = dilatconv(inplanes // 4, planes // 4, pad=3, dilate=3)


        self.conv41 = dilatconv(inplanes // 8, planes // 8, pad=4, dilate=4)


        self.downsample = downsample
        self.stride = stride


        self.relu = nn.ReLU(inplace=True)
        ######################SEBlock

        self.se = SELayer(planes, reduction)    ##########SENet


        self.spatialConv = SpatialAttention()


    def forward(self, x):
        # residual = x

        # split = torch.split(x, self.split, 1)

        out1 = self.conv1x1_Before(x)

        # out1 = self.conv1x1_Before1(split[0])
        out1 = self.conv11(out1)

        # out2 = self.conv1x1_Before2(split[1])
        out2 = self.conv21(out1)

        # out3 = self.conv1x1_Before3(split[2])
        out3 = self.conv31(out2)

        # out4 = self.conv1x1_Before4(split[3])
        out4 = self.conv41(out3)

        # out3 = self.conv32(out3 + out4)
        # out2 = self.conv22(out2 + out3)
        # out1 = self.conv12(out1 + out2)


        out = torch.cat((out1, out2, out3, out4), 1)  #### conact

        out = self.conv1x1_After(out)


        # #########################SEBlock
        se = self.se(out)  ##################################
        out = se * out


        sa = self.spatialConv(out)
        out = sa * out


        out += x
        # out = self.relu(out)

        return out


# class BasicBlock(nn.Module):  #####SENet + new type residual
#     expansion = 1
#
#     def __init__(self, inplanes, planes, reduction, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#
#
#         self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
#         self.conv1 = conv3x3(inplanes, planes // 2, stride)
#         # self.bn1 = nn.BatchNorm2d(planes // 2, momentum=BN_MOMENTUM)
#
#         self.bn2 = nn.BatchNorm2d(planes // 2, momentum=BN_MOMENTUM)
#         self.conv2 = dilatconv(planes // 2, planes // 4, pad=2, dilate=2)
#         # self.bn2 = nn.BatchNorm2d(planes // 4, momentum=BN_MOMENTUM)
#
#         self.bn3 = nn.BatchNorm2d(planes // 4, momentum=BN_MOMENTUM)
#         self.conv3 = dilatconv(planes // 4, planes // 8, pad=3, dilate=3)
#         # self.bn3 = nn.BatchNorm2d(planes // 4, momentum=BN_MOMENTUM)
#
#         self.bn4 = nn.BatchNorm2d(planes // 8, momentum=BN_MOMENTUM)
#         self.conv4 = dilatconv(planes // 8, planes // 8, pad=4, dilate=4)
#
#         self.downsample = downsample
#         self.stride = stride
#
#
#         self.relu = nn.ReLU(inplace=True)
#
#         # self.fn_bi_1 = BilinearGate(planes//4, planes//4, reduction)
#         # self.fn_bi_2 = BilinearGate(planes//4, planes//4, reduction)
#         ######################SEBlock
#
#         self.se = SELayer(planes, reduction)    ##########SENet
#         # self.se = SKLayer(planes, reduction)    ###########SKNet
#
#         # self.sigmoid = nn.Sigmoid()
#         #######################
#
#         # self.convLinear1 = nn.Linear(planes, planes // 2, bias=False)
#         # self.convLinear2 = nn.Linear(planes, planes // 4, bias=False)
#         # self.convLinear3 = nn.Linear(planes, planes // 8, bias=False)
#         # self.convLinear4 = nn.Linear(planes, planes // 8, bias=False)
#
#         ###################### spatial attention
#         # self.sa = SALayer(inplanes, 4)
#         #
#         # # self.spatialConv1 = SpatialMask(4, 1)
#         # self.spatialConv2 = SpatialMask(4, 1)
#         # self.spatialConv3 = SpatialMask(4, 1)
#         # self.spatialConv4 = SpatialMask(4, 1)
#
#         self.spatialConv = SpatialAttention()
#         # self.spatialConv1 = SpatialAttention()
#         # self.spatialConv2 = SpatialAttention()
#         # self.spatialConv3 = SpatialAttention()
#         # self.spatialConv4 = SpatialAttention()
#
#         # self.triplet = TripletAttention()
#
#         # self.bilinear = BilinearGate(inplanes, planes, 2)
#         # self.hight_block = AxialAttention(inplanes, planes, groups=reduction//2)
#         # self.width_block = AxialAttention(inplanes, planes, groups=reduction//2, width=True)
#         #
#         #
#         # self.asymm_block = _SelfAttentionBlock(inplanes, planes, inplanes//reduction, inplanes//reduction)
#
#     def forward(self, x):
#         # residual = x
#
#         # N, C, H, W = x.shape
#
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         # out = self.bn1(out)
#         # out = self.relu(out)
#         out_level_1 = out
#
#
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         # out = self.bn2(out)
#         # out = self.relu(out)
#         out_level_2 = out
#
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         # out = self.bn3(out)
#         # out = self.relu(out)
#         out_level_3 = out
#
#         out = self.bn4(out)
#         out = self.relu(out)
#         out = self.conv4(out)
#         # out = self.bn3(out)
#         # out = self.relu(out)
#         out_level_4 = out
#
#         # bifea_1 = self.fn_bi_1(out_level_1)
#
#         out = torch.cat((out_level_1, out_level_2, out_level_3, out_level_4), 1)  #### conact
#
#
#         # # out = self.asymm_block(out)
#         #
#         # # if C == 256:
#         # # out = self.relu(out)
#         # # out = self.hight_block(out)
#         # # out = self.width_block(out)
#         #
#         # # else:
#         # #########################SEBlock
#         # se = self.se(out)  ##################################
#         # out = se * out
#         #
#         # # b, c, _, _ = out.size()  ####################
#         # # #########################
#         # #
#         # # se1 = self.convLinear1(se).view(b, c // 2, 1, 1)  ####################
#         # # se2 = self.convLinear2(se).view(b, c // 4, 1, 1)  ####################
#         # # se3 = self.convLinear3(se).view(b, c // 8, 1, 1)  ####################
#         # # se4 = self.convLinear4(se).view(b, c // 8, 1, 1)  ####################
#         # #
#         # # se1 = self.sigmoid(se1)
#         # # se2 = self.sigmoid(se2)
#         # # se3 = self.sigmoid(se3)
#         # # se4 = self.sigmoid(se4)
#         #
#         # # out_level_1 = out_level_1 * se1.expand_as(out_level_1)  ####################
#         # # out_level_2 = out_level_2 * se2.expand_as(out_level_2)  ####################
#         # # out_level_3 = out_level_3 * se3.expand_as(out_level_3)  ####################
#         # # out_level_4 = out_level_4 * se4.expand_as(out_level_4)  ####################
#         #
#         # ########################################### spatial attention
#         # # sa = self.sa(out)
#         # #
#         # # # sa1 = self.spatialConv1(sa).expand_as(out_level_1)
#         # # sa2 = self.spatialConv2(sa).expand_as(out_level_2)
#         # # sa3 = self.spatialConv3(sa).expand_as(out_level_3)
#         # # sa4 = self.spatialConv4(sa).expand_as(out_level_4)
#         #
#         # sa = self.spatialConv(out)
#         # out = sa * out
#         # # sa1 = self.spatialConv1(out)
#         # # sa2 = self.spatialConv2(out)
#         # # sa3 = self.spatialConv3(out)
#         # # sa4 = self.spatialConv4(out)
#         #
#         # # out_level_1 = checkpoint(out_level_1, sa1)
#         # # out_level_1 = out_level_1 * sa1 ####################
#         # # out_level_2 = out_level_2 * sa2  ####################
#         # # out_level_3 = out_level_3 * sa3  ####################
#         # # out_level_4 = out_level_4 * sa4  ####################
#         #
#         #
#         # # bifea_2 = self.fn_bi_2(out_level_1)
#         # # out = torch.cat((out_level_1, out_level_2, out_level_3, out_level_4), 1)  #### conact
#         #
#         #
#         # # out = self.triplet(out)
#         # # out = self.bilinear(out)
#         #
#         # # se = se.view(b,c,1,1)   ##############SKNet
#         # # out = out * se.expand_as(out)   ##########SKNet
#         #
#         #
#         # # if self.downsample is not None:
#         # #     residual = self.downsample(x)
#
#         out += x
#         # out = self.relu(out)
#
#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
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

        ######################SEBlock

        self.se = SELayer(planes * self.expansion, reduction)

        #######################

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


        # out = self.se(out)  ##################################

        # b, c, _, _ = out.size()   ####################
        se = self.se(out)#.view(b,c,1,1)    ####################
        out = out * se#.expand_as(out)  ####################

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class DualPathBlock(nn.Module):
#     def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, G, _type='normal'):
#         super(DualPathBlock, self).__init__()
#         self.num_1x1_c = num_1x1_c
#
#         if _type is 'proj':
#             key_stride = 1
#             self.has_proj = True
#         if _type is 'down':
#             key_stride = 2
#             self.has_proj = True
#         if _type is 'normal':
#             key_stride = 1
#             self.has_proj = False
#
#         if self.has_proj:
#             self.c1x1_w = self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_c+2*inc, kernel_size=1, stride=key_stride)
#
#         self.layers = nn.Sequential(OrderedDict([
#             ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
#             ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride, padding=1, groups=G)),
#             ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c+inc, kernel_size=1, stride=1)),
#         ]))
#
#     def BN_ReLU_Conv(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1):
#         return nn.Sequential(OrderedDict([
#             ('norm', nn.BatchNorm2d(in_chs)),
#             ('relu', nn.ReLU(inplace=True)),
#             ('conv', nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)),
#         ]))
#
#     def forward(self, x):
#         data_in = torch.cat(x, dim=1) if isinstance(x, list) else x
#         if self.has_proj:
#             data_o = self.c1x1_w(data_in)
#             data_o1 = data_o[:,:self.num_1x1_c,:,:]
#             data_o2 = data_o[:,self.num_1x1_c:,:,:]
#         else:
#             data_o1 = x[0]
#             data_o2 = x[1]
#
#         out = self.layers(data_in)
#
#         summ = data_o1 + out[:,:self.num_1x1_c,:,:]
#         dense = torch.cat([data_o2, out[:,self.num_1x1_c:,:,:]], dim=1)
#         return [summ, dense]


class HighResolutionModule(nn.Module):
    # def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
    #              num_channels, fuse_method, multi_scale_output=True):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,  ###############################
                 num_channels, fuse_method, reduction, multi_scale_output=True):  #########################

        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        # self.branches = self._make_branches(
        #     num_branches, blocks, num_blocks, num_channels)

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels, reduction)   ###################


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

    # def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
    #                      stride=1):
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, reduction,
                             stride=1):   ################################
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
                reduction[branch_index],  ###############################
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
                    num_channels[branch_index],
                    reduction[branch_index]  ###############################
                )
            )

        return nn.Sequential(*layers)

    # def _make_branches(self, num_branches, block, num_blocks, num_channels):
    def _make_branches(self, num_branches, block, num_blocks, num_channels, reduction):  ##################
        branches = []

        for i in range(num_branches):
            branches.append(
                # self._make_one_branch(i, block, num_blocks, num_channels)
                self._make_one_branch(i, block, num_blocks, num_channels, reduction)  ################

            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        if not self.multi_scale_output:
            # num_inchannels[0] *= 4   #################%%%%%%%%%%%%%%%%%%%%
            num_inchannels[0] = 32  ##################################
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
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                            # nn.Upsample(scale_factor=2**(j-i), mode='nearest'),   ####################%%%%%%%%%%
                            nn.Upsample(scale_factor=2 ** (j - i), mode='bilinear', align_corners=True),  ##################################
                        )
                    )
                elif j == i:
                    if not self.multi_scale_output:
                        fuse_layer.append(nn.Sequential(
                                nn.Conv2d(
                                    # num_inchannels[j]//4,  ##############%%%%%%%%%%%%%%%%%%%%
                                    num_inchannels[j],  #####################
                                    num_inchannels[j],
                                    1, 1, 0, bias=False
                                )
                            ))
                    else:
                        fuse_layer.append(nn.Sequential())
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
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)
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
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
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
            y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
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


        #######################################################################################
        self.imgMaxPool = nn.MaxPool2d(kernel_size=2, stride=2)


        # self.hight_atten_stage1 = AxialAttention(32, 32)
        # self.width_atten_stage1 = AxialAttention(32, 32, width=True)
        #
        # self.hight_atten_stage2 = AxialAttention(32, 32)
        # self.width_atten_stage2 = AxialAttention(32, 32, width=True)

        # self.hight_atten_stage3 = AxialAttention(32, 32)
        # self.width_atten_stage3 = AxialAttention(32, 32, width=True)

        # self.convAttention1x1_stage1_1 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        # self.bnAttention1x1_stage1 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        # self.convAttention1x1_stage1_2 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        #
        # self.convAttention1x1_stage2_1 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        # self.bnAttention1x1_stage2 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        # self.convAttention1x1_stage2_2 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        #
        # self.convAttention1x1_stage3_1 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        # self.bnAttention1x1_stage3 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        # self.convAttention1x1_stage3_2 = nn.Conv2d(32, 32, kernel_size=1, bias=False)

        # #########**************################
        # self.convAttention_stage1_0 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        # self.bnAttentionstage1_0 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        #
        #
        # self.convAttention_stage2_0 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        # self.bnAttentionstage2_0 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        #
        # self.convAttention_stage2_1 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        # self.bnAttentionstage2_1 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        #
        #
        # self.convAttention_stage3_0 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        # self.bnAttentionstage3_0 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        #
        # self.convAttention_stage3_1 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        # self.bnAttentionstage3_1 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        #
        # self.convAttention_stage3_2 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        # self.bnAttentionstage3_2 = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        # #########**************################

        self.imgAttenConv1 = img_conv(64, 32)
        self.imgAttenConv2 = img_conv(128, 64)
        self.imgAttenConv3 = img_conv(256, 128)
        self.imgAttenConv4 = img_conv(512, 256)


        #######################################################################################
        self.dense0 = dense_conv(256,48)
        self.dense1 = dense_conv(32, 48)
        self.dense2 = dense_conv(32, 48)
        self.dense3 = dense_conv(32, 48)
        self.dense4 = dense_conv(96, 32)

        # self.asymm_block_stage1 = _SelfAttentionBlock(low_in_channel=32, high_in_channel=32, neck_channel=16)
        # self.asymm_block_stage2 = _SelfAttentionBlock(low_in_channel=32, high_in_channel=32, neck_channel=16)
        # self.asymm_block_stage3 = _SelfAttentionBlock(low_in_channel=32, high_in_channel=32, neck_channel=16)


        self.asymm_block01_stage1 = PatchSelfAttention2(low_in_channel=64, high_in_channel=32, neck_channel=16, patch_ratio=2)
        self.asymm_block02_stage1 = PatchSelfAttention2(low_in_channel=128, high_in_channel=32, neck_channel=16, patch_ratio=4)
        self.asymm_block03_stage1 = PatchSelfAttention2(low_in_channel=256, high_in_channel=32, neck_channel=16, patch_ratio=8)

        self.asymm_block01_stage2 = PatchSelfAttention2(low_in_channel=64, high_in_channel=32, neck_channel=16, patch_ratio=2)
        self.asymm_block02_stage2 = PatchSelfAttention2(low_in_channel=128, high_in_channel=32, neck_channel=16, patch_ratio=4)
        self.asymm_block03_stage2 = PatchSelfAttention2(low_in_channel=256, high_in_channel=32, neck_channel=16, patch_ratio=8)

        self.asymm_block01_stage3 = PatchSelfAttention2(low_in_channel=64, high_in_channel=32, neck_channel=16, patch_ratio=2)
        self.asymm_block02_stage3 = PatchSelfAttention2(low_in_channel=128, high_in_channel=32, neck_channel=16, patch_ratio=4)
        self.asymm_block03_stage3 = PatchSelfAttention2(low_in_channel=256, high_in_channel=32, neck_channel=16, patch_ratio=8)


        self.fuse1 = FuseBlock()
        self.fuse2 = FuseBlock()
        self.fuse3 = FuseBlock()

        # self.bilinear = BilinearGate(32, 32, 2)
        # self.UDP_change1 = conv1x1(128, 32)
        # self.UDP_change2 = conv1x1(128, 32)
        # self.UDP_change3 = conv1x1(128, 32)


        ###########################################################
        self.imgAtten_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.imgAtten_deConv3_2 = self._make_deconv_layer(256, 128)  # for branch 3 to 2
        self.imgAtten_conv3_2 = nn.Conv2d(256, 128, kernel_size=1, bias=False)

        self.imgAtten_deConv2_1 = self._make_deconv_layer(128, 64)  # for branch 2 to 1
        self.imgAtten_conv2_1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.imgAtten_deConv1_0 = self._make_deconv_layer(64, 32)  # for branch 1 to 0
        self.imgAtten_conv1_0 = nn.Conv2d(64, 32, kernel_size=1, bias=False)

        self.imgAtten_conv2 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.imgAtten_bn2 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)

        self.imgAtten_conv1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.imgAtten_bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.imgAtten_conv0 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.imgAtten_bn0 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)

        ###########################################################
        #
        # self.sigmAttention = nn.Sigmoid()
        # # self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        #
        # self.attentionMask_2_1 = self._make_attention_mask(AttentionBottleneck, 64, 64)
        # self.attentionMask_3_1 = self._make_attention_mask(AttentionBottleneck, 64, 64)
        # self.attentionMask_3_2 = self._make_attention_mask(AttentionBottleneck, 128, 128)
        #
        # self.attention_stage_1_1 = self._make_conv1x1_layer(64, 32)
        #
        # self.attention_stage_2_1 = self._make_conv1x1_layer(64, 32)
        # self.attention_stage_2_2 = self._make_conv1x1_layer(128, 64)
        #
        # self.attention_stage_3_1 = self._make_conv1x1_layer(64, 32)
        # self.attention_stage_3_2 = self._make_conv1x1_layer(128, 64)
        # self.attention_stage_3_3 = self._make_conv1x1_layer(256, 128)
        #
        # # self.convCPC = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        # # self.eps = torch.tensor(1e-6)
        # # self.sigmCPC = nn.Sigmoid()
        #
        # self.imgMaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.imgAttenConv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        # self.imgAttenBn1 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        #
        # self.imgAttenConv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        # self.imgAttenBn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        #
        # self.imgAttenConv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        # self.imgAttenBn3 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        #
        # self.imgAttenConv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        # self.imgAttenBn4 = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        #
        # ###########################################################
        # self.imgAtten_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        # self.imgAtten_deConv3_2 = self._make_deconv_layer(256, 128)  # for branch 3 to 2
        # self.imgAtten_conv3_2 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        #
        # self.imgAtten_deConv2_1 = self._make_deconv_layer(128, 64)  # for branch 2 to 1
        # self.imgAtten_conv2_1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        #
        # self.imgAtten_deConv1_0 = self._make_deconv_layer(64, 32)  # for branch 1 to 0
        # self.imgAtten_conv1_0 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        #
        # self.imgAtten_conv2 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        # self.imgAtten_bn2 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        #
        # self.imgAtten_conv1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        # self.imgAtten_bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        #
        # self.imgAtten_conv0 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        # self.imgAtten_bn0 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        #
        # ###########################################################
        #
        # # self.imgAtten_Final_conv4 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        # # self.imgAtten_Final_up4 = nn.Upsample(scale_factor=8, mode='bilinear')
        # #
        # # self.imgAtten_Final_conv3 = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        # # self.imgAtten_Final_up3 = nn.Upsample(scale_factor=4, mode='bilinear')
        # #
        # # self.imgAtten_Final_conv2 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        # # self.imgAtten_Final_up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # #
        # # self.imgAtten_Final_bn = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        #
        # self.RELU = nn.ReLU(inplace=True)
        #######################################################################################


        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # self.attentionModule2 = self._make_attention_layer(AttentionBottleneck, 64, 64, 2) #original planes=32  ##############################
        # self.deConv1 = self._make_deconv_layer(64, 64) #for branch 2 to 1  #####################################

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

        # self.attentionModule3 = self._make_attention_layer(AttentionBottleneck, 128, 128, 2) #planes=32  ##############################
        # self.deConv2_1 = self._make_deconv_layer(64, 64)   #for branch 3 to 2  #########################################
        # self.deConv2_2 = self._make_deconv_layer(128, 128)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)


        self.stage4, pre_stage_channels = self._make_stage(     #############################%%%%%%%%%%%%%%%%%%%%%%%%
            self.stage4_cfg, num_channels)  #############################%%%%%%%%%%%%%%%%%%%%%%%%
        # self.stage4, pre_stage_channels = self._make_stage(  #####################################
        #     self.stage4_cfg, num_channels)  #####################################


        ####################5, 6, 7
        self.stage5_cfg = cfg['MODEL']['EXTRA']['STAGE5']
        num_channels = self.stage5_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage5_cfg['BLOCK']]
        num_channels = [
           num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition4 = self._make_transition_layer([64], num_channels)  #############  [32]
        self.stage5, pre_stage_channels = self._make_stage(
           self.stage5_cfg, num_channels)


        self.stage6_cfg = cfg['MODEL']['EXTRA']['STAGE6']
        num_channels = self.stage6_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage6_cfg['BLOCK']]
        num_channels = [
           num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition5 = self._make_transition_layer(
           pre_stage_channels, num_channels)
        self.stage6, pre_stage_channels = self._make_stage(
           self.stage6_cfg, num_channels)


        self.stage7_cfg = cfg['MODEL']['EXTRA']['STAGE7']
        num_channels = self.stage7_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage7_cfg['BLOCK']]
        num_channels = [
           num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition6 = self._make_transition_layer(
           pre_stage_channels, num_channels)
        self.stage7, pre_stage_channels = self._make_stage(
           self.stage7_cfg, num_channels)

        ######################8, 9, 10

        self.stage8_cfg = cfg['MODEL']['EXTRA']['STAGE8']
        num_channels = self.stage8_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage8_cfg['BLOCK']]
        num_channels = [
           num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition7 = self._make_transition_layer([80], num_channels)    ######### [32]
        self.stage8, pre_stage_channels = self._make_stage(
           self.stage8_cfg, num_channels)


        self.stage9_cfg = cfg['MODEL']['EXTRA']['STAGE9']
        num_channels = self.stage9_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage9_cfg['BLOCK']]
        num_channels = [
           num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition8 = self._make_transition_layer(
           pre_stage_channels, num_channels)
        self.stage9, pre_stage_channels = self._make_stage(
           self.stage9_cfg, num_channels)


        self.stage10_cfg = cfg['MODEL']['EXTRA']['STAGE10']
        num_channels = self.stage10_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage10_cfg['BLOCK']]
        num_channels = [
           num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition9 = self._make_transition_layer(
           pre_stage_channels, num_channels)
        self.stage10, pre_stage_channels = self._make_stage(
           self.stage10_cfg, num_channels)

        ######################################################################################
        self.imgAttention1 = self._make_img_attention(Img_Attention_Block, 3, 32, 4)   ######### image attention
        self.imgAttention2 = self._make_img_attention(Img_Attention_Block, 3, 64, 8)   ######### image attention
        self.imgAttention3 = self._make_img_attention(Img_Attention_Block, 3, 128, 16)   ######### image attention
        self.imgAttention4 = self._make_img_attention(Img_Attention_Block, 3, 256, 32)   ######### image attention


        #####################################################################################


        if not cfg.MODEL.TARGET_TYPE=='offset':
            factor=1
        else:
            factor=3
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS*factor,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    ###################################################################################################
    def _make_attention_layer(self, block, inplanes, planes_1, planes_2, blocks, stride=1):   ##create attention module
        # downsample = True
        # if stride != 1 or inplanes != planes:
        #     downsample = False
        #
        #
        layers = []
        # layers.append(block(inplanes, planes, stride, downsample))


        for i in range(0, blocks):
            layers.append(block(inplanes, planes_1, planes_2))

        return nn.Sequential(*layers)

    def _make_attention_mask(self, block, inplanes, planes_1, planes_2, stride=1):
        # downsample = False

        layer = []
        layer.append(block(inplanes, planes_1, planes_2, stride))


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

    def _make_img_attention(self, block, inplanes, planes, reduction):
        layer = []

        layer.append(block(inplanes, planes, reduction))

        return nn.Sequential(*layer)

    ###################################################################################################


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
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
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
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
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


        reduction = layer_config['REDUCTION']  #######################################

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
                    reduction,  #######################################
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        ######################################## image attention
        img = self.imgMaxPool(x)
        img = self.imgMaxPool(img)


        ######################################################

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)


        DPN_x = self.dense0(x)  ###############densenet
        sum = DPN_x[:,:32,:,:]
        dense = DPN_x[:,32:,:,:]


        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y1 = x_list[0]  #######################################
        y2 = x_list[1]  #######################################
        y_list = self.stage2(x_list)

        y_list[0] = y_list[0] + y1  #######################################
        y_list[1] = y_list[1] + y2  #######################################




        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y1 = x_list[0]  #######################################
        y2 = x_list[1]  #######################################
        y3 = x_list[2]  #######################################
        y_list = self.stage3(x_list)

        y_list[0] = y_list[0] + y1  #######################################
        y_list[1] = y_list[1] + y2  #######################################
        y_list[2] = y_list[2] + y3  #######################################



        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y1 = x_list[0]  #######################################
        y2 = x_list[1]  #######################################
        y3 = x_list[2]  #######################################
        y4 = x_list[3]
        y_list = self.stage4(x_list)

        y_list[0] = y_list[0] + y1  #######################################
        y_list[1] = y_list[1] + y2  #######################################
        y_list[2] = y_list[2] + y3  #######################################
        y_list[3] = y_list[3] + y4  #######################################

        # y_list[0] = self.UDP_change1(y_list[0])

        y_list[0] = self.fuse1(y_list)

        # S1_x = self.final_layer(y_list[0])    ############internal supervision

        # y_list_01_temp_1, y_siMap4_1 = self.asymm_block01_stage1(y_list[0], y_list[1])
        # y_list_02_temp_1, y_siMap16_1 = self.asymm_block02_stage1(y_list[0], y_list[2])
        # y_list_03_temp_1, y_siMap64_1 = self.asymm_block03_stage1(y_list[0], y_list[3])

        # print(torch.max(y_list[0]), 'fea_before1')

        y_list_01_temp_1 = self.asymm_block01_stage1(y_list[0], y_list[1])
        # print(torch.max(y_list_01_temp_1), 'fea_after1')

        y_list_02_temp_1 = self.asymm_block02_stage1(y_list[0], y_list[2])
        # print(torch.max(y_list_02_temp_1), 'fea_after2')

        y_list_03_temp_1 = self.asymm_block03_stage1(y_list[0], y_list[3])
        # print(torch.max(y_list_03_temp_1), 'fea_after3')
        y_list[0] = y_list[0] + y_list_01_temp_1 + y_list_02_temp_1 + y_list_03_temp_1



        DPN_x = self.dense1(y_list[0])       ###########densenet
        sum = sum + DPN_x[:,:32,:,:]
        dense = torch.cat([dense, DPN_x[:,32:,:,:]], dim=1)

        # print(torch.max(dense), 'dense1')
        # print(torch.max(sum), 'sum1')

        y_list[0] = torch.cat((sum, dense), dim=1)    ###########densenet

        ####################5, 6, 7
        x_list = []
        for i in range(self.stage5_cfg['NUM_BRANCHES']):
            if self.transition4[i] is not None:
                x_list.append(self.transition4[i](y_list[0]))   ############### y_list[0]
            else:
                x_list.append(y_list[0])   ##################  y_list[0]

        y1 = x_list[0]  #######################################
        y2 = x_list[1]  #######################################
        y_list = self.stage5(x_list)

        y_list[0] = y_list[0] + y1  #######################################
        y_list[1] = y_list[1] + y2  #######################################


        x_list = []
        for i in range(self.stage6_cfg['NUM_BRANCHES']):
            if self.transition5[i] is not None:
                x_list.append(self.transition5[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y1 = x_list[0]  #######################################
        y2 = x_list[1]  #######################################
        y3 = x_list[2]  #######################################
        y_list = self.stage6(x_list)

        y_list[0] = y_list[0] + y1  #######################################
        y_list[1] = y_list[1] + y2  #######################################
        y_list[2] = y_list[2] + y3  #######################################


        x_list = []
        for i in range(self.stage7_cfg['NUM_BRANCHES']):
            if self.transition6[i] is not None:
                x_list.append(self.transition6[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y1 = x_list[0]  #######################################
        y2 = x_list[1]  #######################################
        y3 = x_list[2]  #######################################
        y4 = x_list[3]
        y_list = self.stage7(x_list)

        y_list[0] = y_list[0] + y1  #######################################
        y_list[1] = y_list[1] + y2  #######################################
        y_list[2] = y_list[2] + y3  #######################################
        y_list[3] = y_list[3] + y4  #######################################


        # y_list[0] = self.UDP_change2(y_list[0])


        y_list[0] = self.fuse2(y_list)

        # S2_x = self.final_layer(y_list[0])  ############internal supervision


        # y_list_01_temp_2, y_siMap4_2 = self.asymm_block01_stage2(y_list[0], y_list[1])
        # y_list_02_temp_2, y_siMap16_2 = self.asymm_block02_stage2(y_list[0], y_list[2])
        # y_list_03_temp_2, y_siMap64_2 = self.asymm_block03_stage2(y_list[0], y_list[3])

        # print(torch.max(y_list[0]), 'fea_before2')

        y_list_01_temp_2 = self.asymm_block01_stage2(y_list[0], y_list[1])
        # print(torch.max(y_list_01_temp_2), 'fea_after21')

        y_list_02_temp_2 = self.asymm_block02_stage2(y_list[0], y_list[2])
        # print(torch.max(y_list_02_temp_2), 'fea_after22')

        y_list_03_temp_2 = self.asymm_block03_stage2(y_list[0], y_list[3])
        # print(torch.max(y_list_03_temp_2), 'fea_after23')
        y_list[0] = y_list[0] + y_list_01_temp_2 + y_list_02_temp_2 + y_list_03_temp_2


        # y_list_02_temp = self.asymm_block_stage2(y_list[0])
        # y_list[0] = y_list[0] + y_list_02_temp

        # axial_attention = self.hight_atten_stage2(y_list[0])     ########### axial attention
        # axial_attention = self.width_atten_stage2(axial_attention)
        # axial_attention = self.relu(axial_attention)
        # y_list[0] = axial_attention


        DPN_x = self.dense2(y_list[0])    ##############densenet
        sum = sum + DPN_x[:, :32, :, :]
        dense = torch.cat([dense, DPN_x[:, 32:, :, :]], dim=1)
        y_list[0] = torch.cat((sum, dense), dim=1)  ###########densenet

        ####################8, 9, 10
        x_list = []
        for i in range(self.stage8_cfg['NUM_BRANCHES']):
            if self.transition7[i] is not None:
                x_list.append(self.transition7[i](y_list[0]))  ####### y_list[0]
            else:
                x_list.append(y_list[0])    ######## y_list[0]

        y1 = x_list[0]  #######################################
        y2 = x_list[1]  #######################################
        y_list = self.stage8(x_list)

        y_list[0] = y_list[0] + y1  #######################################
        y_list[1] = y_list[1] + y2  #######################################


        x_list = []
        for i in range(self.stage9_cfg['NUM_BRANCHES']):
            if self.transition8[i] is not None:
                x_list.append(self.transition8[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y1 = x_list[0]  #######################################
        y2 = x_list[1]  #######################################
        y3 = x_list[2]  #######################################
        y_list = self.stage9(x_list)

        y_list[0] = y_list[0] + y1  #######################################
        y_list[1] = y_list[1] + y2  #######################################
        y_list[2] = y_list[2] + y3  #######################################



        x_list = []
        for i in range(self.stage10_cfg['NUM_BRANCHES']):
            if self.transition9[i] is not None:
                x_list.append(self.transition9[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y1 = x_list[0]  #######################################
        y2 = x_list[1]  #######################################
        y3 = x_list[2]  #######################################
        y4 = x_list[3]
        y_list = self.stage10(x_list)

        y_list[0] = y_list[0] + y1  #######################################
        y_list[1] = y_list[1] + y2  #######################################
        y_list[2] = y_list[2] + y3  #######################################
        y_list[3] = y_list[3] + y4  #######################################

        # y_list[0] = self.UDP_change3(y_list[0])

        y_list[0] = self.fuse3(y_list)

        # S3_x = self.final_layer(y_list[0])  ############internal supervision


        # y_list_01_temp_3, y_siMap4_3 = self.asymm_block01_stage3(y_list[0], y_list[1])
        # y_list_02_temp_3, y_siMap16_3 = self.asymm_block02_stage3(y_list[0], y_list[2])
        # y_list_03_temp_3, y_siMap64_3 = self.asymm_block03_stage3(y_list[0], y_list[3])

        # print(torch.max(y_list[0]), 'fea_before3')

        y_list_01_temp_3 = self.asymm_block01_stage3(y_list[0], y_list[1])
        # print(torch.max(y_list_01_temp_3), 'fea_after31')

        y_list_02_temp_3 = self.asymm_block02_stage3(y_list[0], y_list[2])
        # print(torch.max(y_list_02_temp_3), 'fea_after32')

        y_list_03_temp_3 = self.asymm_block03_stage3(y_list[0], y_list[3])
        # print(torch.max(y_list_03_temp_3), 'fea_after33')

        y_list[0] = y_list[0] + y_list_01_temp_3 + y_list_02_temp_3 + y_list_03_temp_3

        # y_list_03_temp = self.asymm_block_stage3(y_list[0])
        # y_list[0] = y_list[0] + y_list_03_temp


        img_attention1 = self.imgAttention1(img)
        y_list[0] = torch.cat((y_list[0], img_attention1), 1)   #### conact
        # y_list[0] = torch.add(y_list[0], img_attention1) / 2
        # y_list[0] = torch.mul(y_list[0], img_attention1)
        y_list[0] = self.imgAttenConv1(y_list[0])





        #####
        img = self.imgMaxPool(img)
        img_attention2 = self.imgAttention2(img)
        y_list[1] = torch.cat((y_list[1], img_attention2), 1)  #### conact
        # y_list[1] = torch.add(y_list[1], img_attention2) / 2
        # y_list[1] = torch.mul(y_list[1], img_attention2)
        y_list[1] = self.imgAttenConv2(y_list[1])




        #####
        img = self.imgMaxPool(img)
        img_attention3 = self.imgAttention3(img)
        y_list[2] = torch.cat((y_list[2], img_attention3), 1)  #### conact
        # y_list[2] = torch.add(y_list[2], img_attention3) / 2
        # y_list[2] = torch.mul(y_list[2], img_attention3)
        y_list[2] = self.imgAttenConv3(y_list[2])





        #####
        img = self.imgMaxPool(img)
        img_attention4 = self.imgAttention4(img)
        y_list[3] = torch.cat((y_list[3], img_attention4), 1)  #### conact
        # y_list[3] = torch.add(y_list[3], img_attention4) / 2
        # y_list[3] = torch.mul(y_list[3], img_attention4)
        y_list[3] = self.imgAttenConv4(y_list[3])



        ##############################################################################################
        ### 3-2
        y_list_final3_D = self.imgAtten_deConv3_2(y_list[3])
        y_list_final3_Up = self.imgAtten_up(y_list[3])
        y_list_final3_Up = self.imgAtten_conv3_2(y_list_final3_Up)

        y_list_final3 = torch.add(y_list_final3_D, y_list_final3_Up)

        ### 2-1
        y_list[2] = torch.cat((y_list[2], y_list_final3), 1)
        y_list[2] = self.imgAtten_conv2(y_list[2])
        y_list[2] = self.imgAtten_bn2(y_list[2])
        y_list[2] = self.relu(y_list[2])

        y_list_final2_D = self.imgAtten_deConv2_1(y_list[2])
        y_list_final2_Up = self.imgAtten_up(y_list[2])
        y_list_final2_Up = self.imgAtten_conv2_1(y_list_final2_Up)

        y_list_final2 = torch.add(y_list_final2_D, y_list_final2_Up)

        ### 1-0
        y_list[1] = torch.cat((y_list[1], y_list_final2), 1)
        y_list[1] = self.imgAtten_conv1(y_list[1])
        y_list[1] = self.imgAtten_bn1(y_list[1])
        y_list[1] = self.relu(y_list[1])

        y_list_final1_D = self.imgAtten_deConv1_0(y_list[1])
        y_list_final1_Up = self.imgAtten_up(y_list[1])
        y_list_final1_Up = self.imgAtten_conv1_0(y_list_final1_Up)

        y_list_final1 = torch.add(y_list_final1_D, y_list_final1_Up)

        ### 0
        y_list[0] = torch.cat((y_list[0], y_list_final1), 1)
        y_list[0] = self.imgAtten_conv0(y_list[0])
        y_list[0] = self.imgAtten_bn0(y_list[0])
        y_list[0] = self.relu(y_list[0])


        # axial_attention = self.hight_atten_stage3(y_list[0])       ########### axial attention
        # axial_attention = self.width_atten_stage3(axial_attention)
        # axial_attention = self.relu(axial_attention)
        # y_list[0] = axial_attention

        DPN_x = self.dense3(y_list[0])    ##############densenet
        sum = sum + DPN_x[:,:32,:,:]
        dense = torch.cat([dense, DPN_x[:,32:,:,:]], dim=1)
        y_list[0] = torch.cat((sum, dense), dim=1)    ###########densenet
        y_list[0] = self.dense4(y_list[0])     ##############densenet


        # y_list[0] = self.bilinear(y_list[0])
        # print(torch.max(y_list[0]), 'fea_before4')
        x = self.final_layer(y_list[0])

        return x   #################%%%%%%%%%%%%%%%%%%%%
        # return x, y_siMap4_1, y_siMap16_1, y_siMap64_1, y_siMap4_2, y_siMap16_2, y_siMap64_2, y_siMap4_3, y_siMap16_3, y_siMap64_3
        # return x, S1_x, S2_x, S3_x  #######################################

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
                if 'stage4.2.fuse_layers' in name:
                    continue
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
