# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :dl_models
# @Time      :2024/8/15 23:38
# @Author    :Chen
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv1d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm1d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv1d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm1d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv1d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv1d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


class UNet(nn.Module):
    def __init__(self, first_kears_num=32, seg_length=4, device=''):
        super(UNet, self).__init__()
        self.seg_length = seg_length
        self.avg = nn.AvgPool1d(seg_length)
        self.device = device
        # 4次下采样
        self.C1 = Conv(1, first_kears_num)
        self.D1 = DownSampling(first_kears_num)
        self.C2 = Conv(first_kears_num, first_kears_num * 2)
        self.D2 = DownSampling(first_kears_num * 2)
        self.C3 = Conv(first_kears_num * 2, first_kears_num * 4)
        self.D3 = DownSampling(first_kears_num * 4)
        self.C4 = Conv(first_kears_num * 4, first_kears_num * 8)
        self.D4 = DownSampling(first_kears_num * 8)
        self.C5 = Conv(first_kears_num * 8, first_kears_num * 16)

        self.U1 = UpSampling(first_kears_num * 16)
        self.C6 = Conv(first_kears_num * 16, first_kears_num * 8)
        self.U2 = UpSampling(first_kears_num * 8)
        self.C7 = Conv(first_kears_num * 8, first_kears_num*4)
        self.U3 = UpSampling(first_kears_num * 4)
        self.C8 = Conv(first_kears_num * 4, first_kears_num * 2)
        self.U4 = UpSampling(first_kears_num * 2)
        self.C9 = Conv(first_kears_num * 2, first_kears_num)
        self.enc = torch.nn.Conv1d(first_kears_num, 1, 3, 1, 1)
        self.pred = torch.nn.Conv1d(1, 1, 3, 1, 1)
        self.Th = torch.nn.Sigmoid()

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        single_len = inputs.shape[3]
        lead_num = inputs.shape[2]
        outs = torch.empty(batch_size, lead_num, int(single_len/self.seg_length), device=self.device)
        for n in range(lead_num):
            x = inputs[:, :, n, :]
            R1 = self.C1(x)
            R2 = self.C2(self.D1(R1))
            R3 = self.C3(self.D2(R2))
            R4 = self.C4(self.D3(R3))
            Y1 = self.avg(self.C5(self.D4(R4)))

            O1 = self.C6(self.U1(Y1, self.avg(R4)))
            O2 = self.C7(self.U2(O1, self.avg(R3)))
            O3 = self.C8(self.U3(O2, self.avg(R2)))
            O4 = self.C9(self.U4(O3, self.avg(R1)))
            out = self.Th(self.pred(self.enc(O4)))
            outs[:, n, :] = out[:, 0, :]
        return outs


