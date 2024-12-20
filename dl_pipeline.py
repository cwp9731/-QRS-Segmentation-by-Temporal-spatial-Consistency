# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :dl_pipeline
# @Time      :2024/8/15 23:38
# @Author    :Chen
"""
import matplotlib.pyplot as plt
import os
from dl_models import UNet
import torch
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np


def train_single(data, model, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)
        outs = model(inputs)
        loss = 0.0
        for i in range(outs.shape[1]):
            loss += torch.nn.BCELoss()(outs[:, i, :], labels)/outs.shape[1]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * inputs.shape[0]
    epoch_loss = running_loss / len(data.dataset)
    return epoch_loss


def val_single(data, model, device):
    model.eval()
    running_loss = 0.0
    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)
        outs = model(inputs)
        loss = 0.0
        for i in range(outs.shape[1]):
            loss += torch.nn.BCELoss()(outs[:, i, :], labels)/outs.shape[1]
        running_loss += loss.item() * inputs.shape[0]
    epoch_loss = running_loss / len(data.dataset)
    return epoch_loss


def train_dl(train_data, test_data, time, data_name, lead_num, seg_len, kare_num, device):
    model = UNet(kare_num, seg_len, device).to(device)
    optimizer = optim.Adam(list(model.parameters()), lr=1e-4, weight_decay=0)
    model_path = os.path.join('models', data_name)
    if not os.path.exists(model_path):
        # 如果路径不存在，则创建该路径
        os.makedirs(model_path)
    model_filepath = os.path.join(model_path, str(lead_num)+'_'+str(seg_len)+'_'+str(kare_num)+'_'+'model_' + str(time))

    patience = 15
    max_epochs = 400
    best_loss = float('inf')
    stop_counter = 0
    with tqdm(range(max_epochs), dynamic_ncols=True) as tqdmEpochs:
        for epoch in tqdmEpochs:
            train_loss = train_single(train_data, model, optimizer, device)
            val_loss = val_single(test_data, model, device)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy((model.state_dict()))
                torch.save(best_model_wts, model_filepath)
                stop_counter = 0
            else:
                stop_counter += 1

            tqdmEpochs.set_postfix(ordered_dict={
                "l": '%.2f' % train_loss + ',%.2f' % val_loss,
                'b': '%.2f' % best_loss,
                's': stop_counter
            })

            if stop_counter >= patience:
                break

def obtain_label(out, seg_len):
    pred = torch.mean(out, dim=0)
    label = np.zeros(len(pred) * seg_len, dtype=int)
    for i in range(len(pred)):
        if pred[i].item() >= 0.5:
            label[i * seg_len:(i + 1) * seg_len] = 1
    indices = np.where(label == 1)[0]
    if len(indices) == 0:
        label = np.array([]).reshape(0, 2)
    else:
        # 找到连续区域的起止点
        diffs = np.diff(indices)
        split_indices = np.where(diffs != 1)[0]
        # 起点
        starts = np.insert(indices[split_indices + 1], 0, indices[0])
        # 终点
        ends = np.append(indices[split_indices], indices[-1])
        # 合并起点和终点
        label = np.vstack((starts, ends)).T
    return label


def test_dl(test_data, test_idx, time, data_name, pred_dict, lead_num, seg_len, kare_num, device):
    model = UNet(kare_num, seg_len, device).to(device)
    model_filepath = os.path.join('models', data_name,
                                  str(lead_num) + '_' + str(seg_len) + '_' + str(kare_num) + '_' + 'model_' + str(time))
    model.load_state_dict(torch.load(model_filepath, map_location='cuda:0'))
    idx = 0
    model.eval()
    for inputs, labels in test_data:
        inputs, labels = inputs.to(device), labels.to(device)
        outs = model(inputs)
        for i in range(outs.shape[0]):
            pred_dict[test_idx[idx]] = obtain_label(outs[i], seg_len)
            idx += 1
    return pred_dict

