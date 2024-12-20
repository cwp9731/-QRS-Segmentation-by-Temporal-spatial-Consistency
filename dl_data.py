# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :dl_data
# @Time      :2024/8/15 23:49
# @Author    :Chen
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class my_dataset(Dataset):
    def __init__(self, Dataset, idxes, lead_num, seg_len):
        self.x, self.y = self.get_data(Dataset, idxes, lead_num, seg_len)

    def __len__(self):
        return len(self.x)

    def get_data(self, Dataset, idxes, lead_num, seg_len):
        (data_dict, label_dict) = Dataset
        leads = np.array(list(data_dict[idxes[0]].keys()))
        leads[0], leads[1] = leads[1], leads[0]
        x = []
        y = []
        for idx in idxes:
            x_idx = []
            for lead in leads[:lead_num]:
                signal = data_dict[idx][lead]
                x_idx.append((signal - np.mean(signal)) / np.std(signal))
            label = label_dict[idx]['ii']
            y_idx = np.zeros(len(x_idx[0]))
            for i in label:
                y_idx[int(i[0]):int(i[1]) + 1] = 1
            y_idx = np.where(np.sum(y_idx.reshape(-1, seg_len), axis=1) > seg_len/2, 1, 0)
            x.append(x_idx)
            y.append(y_idx)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return [x, y]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        if x.dtype != float:
            x = np.array(x, dtype=float)
        x = torch.tensor(x, dtype=torch.float)
        x = x.unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float)
        return x, y


