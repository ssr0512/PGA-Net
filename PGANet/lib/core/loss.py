# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

########################################################################
class DiversityLOSS(nn.Module):
    def __init__(self):
        super(DiversityLOSS, self).__init__()
        self.alpha = torch.tensor(1e-8)
        # self.beta = torch.tensor(1e-6)

    def forward(self, S2, S3):

        #### stage2
        # S2_total_T = torch.zeros(2,2)

        S2_m1 = S2[0]
        S2_m2 = S2[1]

        ######### initial S2_total_T in cuda
        temp_m1 = S2_m1[0, :].unsqueeze(0)
        temp_m2 = S2_m2[0, :].unsqueeze(0)

        S2_Mi = torch.cat((temp_m1, temp_m2), 0)
        S2_Mi_T = S2_Mi.permute(1, 0)

        S2_total_T = torch.matmul(S2_Mi, S2_Mi_T)
        ######### initial S2_total_T in cuda

        for i in range(1, S2_m1.size(0)):
            temp_m1 = S2_m1[i,:].unsqueeze(0)
            temp_m2 = S2_m2[i,:].unsqueeze(0)

            S2_Mi = torch.cat((temp_m1, temp_m2), 0)
            S2_Mi_T = S2_Mi.permute(1, 0)

            S2_T = torch.matmul(S2_Mi, S2_Mi_T)
            S2_total_T = S2_total_T + S2_T


        S2_total_T = S2_total_T / S2_m1.size(0)
        # print(S2_total_T, "S2_total_T")

        DiversityLOSS_S2 = torch.logdet(S2_total_T) * self.alpha
        # print(DiversityLOSS_S2, "DiversityLOSS_S2")
        ####


        #### stage3
        # S3_total_T = torch.zeros(3,3)

        S3_m1 = S3[0]
        S3_m2 = S3[1]
        S3_m3 = S3[2]

        ######### initial S3_total_T in cuda
        temp_m1 = S3_m1[0, :].unsqueeze(0)
        temp_m2 = S3_m2[0, :].unsqueeze(0)
        temp_m3 = S3_m3[0, :].unsqueeze(0)

        S3_Mi = torch.cat((temp_m1, temp_m2, temp_m3), 0)
        S3_Mi_T = S3_Mi.permute(1, 0)

        S3_total_T = torch.matmul(S3_Mi, S3_Mi_T)
        ######### initial S3_total_T in cuda

        for i in range(1, S3_m1.size(0)):
            temp_m1 = S3_m1[i,:].unsqueeze(0)
            temp_m2 = S3_m2[i,:].unsqueeze(0)
            temp_m3 = S3_m3[i,:].unsqueeze(0)

            S3_Mi = torch.cat((temp_m1, temp_m2, temp_m3), 0)
            S3_Mi_T = S3_Mi.permute(1, 0)

            S3_T = torch.matmul(S3_Mi, S3_Mi_T)
            S3_total_T = S3_total_T + S3_T

        S3_total_T = S3_total_T / S3_m1.size(0)
        # print(S3_total_T, "S3_total_T")

        DiversityLOSS_3 = torch.logdet(S3_total_T) * self.alpha
        # print(DiversityLOSS_3, "DiversityLOSS_3")
        ####

        return DiversityLOSS_S2, DiversityLOSS_3


class CPCLOSS(nn.Module):
    def __init__(self):
        super(CPCLOSS, self).__init__()
        self.eps = torch.tensor(1e-6)
        self.sigmCPC = nn.Sigmoid()

    def forward(self, cpc_A, cpc_B, cpc_B_fake):
        # cpc_B = cpc_B.permute(1, 0)
        cpc_B_fake = cpc_B_fake.permute(1, 0)



        cpc_feature = torch.mul(cpc_A, cpc_B)#/cpc_A.size(1)
        # a = torch.sum(torch.sum(cpc_feature))
        # cpc_feature = torch.mean(cpc_feature)
        # print("cpc: ", cpc_feature)

        cpc_feature_fake = torch.matmul(cpc_A, cpc_B_fake)/cpc_A.size(1)
        # b = torch.sum(torch.sum(cpc_feature_fake))
        # cpc_feature_fake = torch.mean(cpc_feature_fake)/cpc_A.size(1)
        # print("cpc_fake: ", cpc_feature_fake)

        cpc_loss_feature = -torch.log10(self.sigmCPC(cpc_feature) + self.eps)
        cpc_loss_feature_fake = -torch.log10(1-(self.sigmCPC(cpc_feature_fake)+self.eps))

        # a = torch.sum(torch.sum(cpc_loss_feature))
        b = torch.mean(cpc_loss_feature)
        # print("cpc: ", b)
        # c = torch.sum(torch.sum(cpc_loss_feature_fake))
        d = torch.mean(cpc_loss_feature_fake)
        # print("cpc_fake: ", d)

        cpc_loss = torch.mean(cpc_loss_feature) + torch.mean(cpc_loss_feature_fake)
        # cpc_loss = -torch.log10(self.sigmCPC(cpc_feature) + self.eps) - torch.log10(1-(self.sigmCPC(cpc_feature_fake)+self.eps))
        # print("cpc_loss: ", cpc_loss)
        return cpc_loss
########################################################################


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
