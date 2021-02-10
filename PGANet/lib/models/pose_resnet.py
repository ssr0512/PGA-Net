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


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)



class Img_Attention_Block(nn.Module):

    def __init__(self, inplanes, planes):
        super(Img_Attention_Block, self).__init__()


        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv4 = nn.Conv2d(planes, int(planes), kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(planes), momentum=BN_MOMENTUM)

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




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


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


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')


        self.layer1_downSample1 = self._make_downsample(256, 256, 256, 64)
        self.layer1_downSample2 = self._make_downsample(256, 256, 256, 64)
        self.layer1_downSample3 = self._make_downsample(256, 256, 256, 64)

        self.layer1_downSample_conv_stage1 = Bottleneck(256, 64)
        self.layer1_downSample_conv_stage2 = Bottleneck(256, 64)
        self.layer1_downSample_conv1_2 = Bottleneck(256, 64)
        self.layer1_downSample_conv2_2 = Bottleneck(256, 64)
        self.layer1_downSample_conv3_2 = Bottleneck(256, 64)

        self.layer1_deconv1 = self._make_attention_deconv_layer(256, 256)
        self.layer1_deconv2 = self._make_attention_deconv_layer(256, 256)
        self.layer1_deconv3 = self._make_attention_deconv_layer(256, 256)


        self.layer2_downSample1 = self._make_downsample(512, 512, 512, 128)
        self.layer2_downSample2 = self._make_downsample(512, 512, 512, 128)

        self.layer2_downSample_conv = Bottleneck(512, 128)
        self.layer2_downSample_conv1_2 = Bottleneck(512, 128)
        self.layer2_downSample_conv2_2 = Bottleneck(512, 128)

        self.layer2_deconv1 = self._make_attention_deconv_layer(512, 512)
        self.layer2_deconv2 = self._make_attention_deconv_layer(512, 512)


        self.layer3_downSample1 = self._make_downsample(1024, 1024, 1024, 256)

        self.layer3_downSample_conv = Bottleneck(1024, 256)

        self.layer3_deonv = self._make_attention_deconv_layer(1024, 1024)  ### plan A


        self.layer1_attention = self._make_attention_block(256, 256)
        self.layer2_attention = self._make_attention_block(512, 512)
        self.layer3_attention = self._make_attention_block(1024, 1024)


        #### img attention
        self.imgMaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.imgAttention1 = self._make_img_attention(Img_Attention_Block, 3, 256)
        self.imgAttention2 = self._make_img_attention(Img_Attention_Block, 3, 256)
        self.imgAttention3 = self._make_img_attention(Img_Attention_Block, 3, 256)
        self.imgAttention4 = self._make_img_attention(Img_Attention_Block, 3, 256)



        self.imgAtten_up = nn.Upsample(scale_factor=2, mode='bilinear')


        self.imgAttenChangeChannel4 = self._make_3x3_block(256, 256)
        self.imgAttenChangeChannel3_temp = self._make_3x3_block(256, 256)
        self.imgAttenChangeChannel2_temp = self._make_3x3_block(256, 256)
        self.imgAttenChangeChannel1_temp = self._make_3x3_block(256, 256)

        self.imgAttenChangeChannel3 = self._make_1x1_block(256, 128)
        self.imgAttenChangeChannel2 = self._make_1x1_block(256, 64)
        self.imgAttenChangeChannel1 = self._make_1x1_block(256, 32)



        self.imgAtten_conv4_3 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.imgAtten_conv3_2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.imgAtten_conv2_1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)

        self.imgAttenConv3 = self._make_3x3_block(256, 128)
        self.imgAttenConv2 = self._make_3x3_block(128, 64)
        self.imgAttenConv1 = self._make_3x3_block(64, 32)


        self.layer1_1 = self._make_layer(block, 64, 2)
        self.layer1_2 = self._make_layer(block, 64, 2)
        self.layer1_3 = self._make_layer(block, 64, 1)

        self.layer2_1 = self._make_layer(block, 128, 2, stride=2)
        self.layer2_2 = self._make_layer(block, 128, 2)
        self.layer2_3 = self._make_layer(block, 128, 1)

        self.layer3_1 = self._make_layer(block, 256, 2, stride=2)
        self.layer3_2 = self._make_layer(block, 256, 2)
        self.layer3_3 = self._make_layer(block, 256, 1)


        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)



        self.final_conv = nn.Conv2d(2048, 256, kernel_size=1, bias=False)
        self.final_bn = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        self.RELU = nn.ReLU(inplace=True)




        self.deconv_layers1 = self._make_deconv_layer(1, 256, 256, 4)
        self.deconv_layers2 = self._make_deconv_layer(1, 256, 256, 4)
        self.deconv_layers3 = self._make_deconv_layer(1, 256, 256, 4)

        self.deconv4_3 = self._make_deconv_layer(1, 256, 128, 4)
        self.deconv3_2 = self._make_deconv_layer(1, 128, 64, 4)
        self.deconv2_1 = self._make_deconv_layer(1, 64, 32, 4)

        self.final_layer = nn.Conv2d(
            in_channels=32,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )


    def _make_downsample(self, inplanes, planes, bottleneck_inplanes, bottleneck_planes, stride=1):
        layer = []

        layer.append(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=2, padding=1, bias=False))
        layer.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layer.append(nn.ReLU(inplace=True))

        downsample = None
        if bottleneck_inplanes != bottleneck_planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4, momentum=BN_MOMENTUM),
            )

        layer.append(Bottleneck(bottleneck_inplanes, bottleneck_planes, stride, downsample))

        return nn.Sequential(*layer)

    def _make_attention_block(self, inplanes, planes):
        layer = []

        layer.append(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False))
        layer.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False))
        layer.append(nn.Sigmoid())


        return nn.Sequential(*layer)

    def _make_attention_deconv_layer(self, inplanes, planes):
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
        layer.append(nn.BatchNorm2d(planes, momentum = BN_MOMENTUM))
        layer.append(nn.ReLU(inplace = True))

        return nn.Sequential(*layer)

    def _make_img_attention(self, block, inplanes, planes):
        layer = []

        layer.append(block(inplanes, planes))

        return nn.Sequential(*layer)


    def _make_3x3_block(self, inplanes, planes):
        layer = []

        layer.append(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1, bias=False))
        layer.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    def _make_1x1_block(self, inplanes, planes):
        layer = []

        layer.append(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False))
        layer.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding


    def _make_deconv_layer(self, num_layers, num_filters_in, num_filters_out, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels, i)

            planes = num_filters_out
            inplanes = num_filters_in
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):

        img = self.imgMaxPool(x)
        img = self.imgMaxPool(img)



        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ################################ Attenetion stage 1
        x = self.layer1_1(x)

        layer1_attention1_1 = self.layer1_downSample1(x)
        layer1_attention2_1 = self.layer1_downSample2(layer1_attention1_1)
        layer1_attention3_1 = self.layer1_downSample3(layer1_attention2_1)

        layer1_attention1_2 = self.layer1_downSample_conv1_2(layer1_attention1_1)
        layer1_attention2_2 = self.layer1_downSample_conv2_2(layer1_attention2_1)
        layer1_attention3_2 = self.layer1_downSample_conv3_2(layer1_attention3_1)

        layer1_attention3_2 = self.layer1_deconv1(layer1_attention3_2)

        layer1_attention2_2 = torch.add(layer1_attention3_2, layer1_attention2_2)
        layer1_attention2_2 = self.layer1_downSample_conv_stage2(layer1_attention2_2)

        layer1_attention2_2 = self.layer1_deconv2(layer1_attention2_2)

        layer1_attention1_2 = torch.add(layer1_attention2_2, layer1_attention1_2)
        layer1_attention1_2 = self.layer1_downSample_conv_stage1(layer1_attention1_2)

        layer1_attention1_2 = self.layer1_deconv3(layer1_attention1_2)

        layer1_attention_mask = self.layer1_attention(layer1_attention1_2)

        x = self.layer1_2(x)
        x = torch.add(torch.mul(layer1_attention_mask, x), x)
        x = self.layer1_3(x)

        ############################################ Attenetion stage 2
        x = self.layer2_1(x)

        layer2_attention1_1 = self.layer2_downSample1(x)
        layer2_attention2_1 = self.layer2_downSample2(layer2_attention1_1)

        layer2_attention1_2 = self.layer2_downSample_conv1_2(layer2_attention1_1)
        layer2_attention2_2 = self.layer2_downSample_conv2_2(layer2_attention2_1)

        layer2_attention2_2 = self.layer2_deconv1(layer2_attention2_2)

        layer2_attention1_2 = torch.add(layer2_attention2_2, layer2_attention1_2)
        layer2_attention1_2 = self.layer2_downSample_conv(layer2_attention1_2)

        layer2_attention1_2 = self.layer2_deconv2(layer2_attention1_2)

        layer2_attention_mask = self.layer2_attention(layer2_attention1_2)

        x = self.layer2_2(x)
        x = torch.add(torch.mul(layer2_attention_mask, x), x)
        x = self.layer2_3(x)

        ############################################ Attenetion stage 3
        x = self.layer3_1(x)

        layer3_attention1_1 = self.layer3_downSample1(x)

        layer3_attention1_2 = self.layer3_downSample_conv(layer3_attention1_1)

        layer3_attention1_2 = self.layer3_deonv(layer3_attention1_2)

        layer3_attention_mask = self.layer3_attention(layer3_attention1_2)

        x = self.layer3_2(x)
        x = torch.add(torch.mul(layer3_attention_mask, x), x)
        x = self.layer3_3(x)

        x = self.layer4(x)


        #### img attention mechanism
        ##########################################################################
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.RELU(x)

        img_attention1 = self.imgAttention1(img)

        img = self.imgMaxPool(img)
        img_attention2 = self.imgAttention2(img)

        img = self.imgMaxPool(img)
        img_attention3 = self.imgAttention3(img)

        img = self.imgMaxPool(img)
        img_attention4 = self.imgAttention4(img)


        x_4 = x
        x_3 = self.deconv_layers3(x_4)
        x_2 = self.deconv_layers2(x_3)
        x_1 = self.deconv_layers1(x_2)


        x_4 = torch.add(x_4, img_attention4) / 2
        x_3 = torch.add(x_3, img_attention3) / 2
        x_2 = torch.add(x_2, img_attention2) / 2
        x_1 = torch.add(x_1, img_attention1) / 2


        x_4 = self.imgAttenChangeChannel4(x_4)

        x_3 = self.imgAttenChangeChannel3_temp(x_3)
        x_3 = self.imgAttenChangeChannel3(x_3)

        x_2 = self.imgAttenChangeChannel2_temp(x_2)
        x_2 = self.imgAttenChangeChannel2(x_2)

        x_1 = self.imgAttenChangeChannel1_temp(x_1)
        x_1 = self.imgAttenChangeChannel1(x_1)


        ### step 4
        img_attention_4D = self.deconv4_3(x_4)
        img_attention_4Up = self.imgAtten_up(x_4)
        img_attention_4Up = self.imgAtten_conv4_3(img_attention_4Up)
        x_4 = torch.add(img_attention_4D, img_attention_4Up)


        ### step 3
        x_3 = torch.cat((x_3, x_4), 1)
        x_3 = self.imgAttenConv3(x_3)
        img_attention_3D = self.deconv3_2(x_3)
        img_attention_3Up = self.imgAtten_up(x_3)
        img_attention_3Up = self.imgAtten_conv3_2(img_attention_3Up)
        x_3 = torch.add(img_attention_3D, img_attention_3Up)


        ### step 2
        x_2 = torch.cat((x_2, x_3), 1)
        x_2 = self.imgAttenConv2(x_2)
        img_attention_2D = self.deconv2_1(x_2)
        img_attention_2Up = self.imgAtten_up(x_2)
        img_attention_2Up = self.imgAtten_conv2_1(img_attention_2Up)
        x_2 = torch.add(img_attention_2D, img_attention_2Up)


        ### step 1
        x_1 = torch.cat((x_1, x_2), 1)
        x_1 = self.imgAttenConv1(x_1)
        x = x_1



        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
