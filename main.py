# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :main.py
# @Time      :2023/10/6 6:09
# @Author    :Chen
"""
import numpy as np
import random
import os
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from dl_data import my_dataset
from dl_pipeline import *
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import copy
import pickle

def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate_detection(label, pred, TOL):
    # 如果 pred 不为空，进行过滤
    if len(pred) > 0:
        # 第一个标签的起点
        first_label_start = label[0, 0]
        # 最后一个标签的终点
        last_label_end = label[-1, 1]

        # 过滤掉不符合条件的预测片段
        print(pred)
        pred = pred[
            ~((pred[:, 1] < first_label_start) & (np.abs(pred[:, 1] - first_label_start) > TOL)) &  # 左边的片段
            ~((pred[:, 0] > last_label_end) & (np.abs(pred[:, 0] - last_label_end) > TOL))  # 右边的片段
            ]

    # 初始化计数器和误差列表
    correct_start_count = 0
    correct_end_count = 0
    correct_segments = 0
    missed_start_count = 0
    missed_end_count = 0
    incorrect_start_count = 0
    incorrect_end_count = 0
    start_errors = []
    end_errors = []

    # 如果 pred 为空，直接统计所有起点和终点为遗漏
    if len(pred) == 0:
        missed_start_count = len(label)
        missed_end_count = len(label)
        return 0, 0, missed_start_count, missed_end_count, missed_start_count, missed_end_count, 0, 0, len(
            label), 0, 0

    # 获取起点和终点的误差矩阵
    start_diff_matrix = np.abs(label[:, 0].reshape(-1, 1) - pred[:, 0].reshape(1, -1))
    end_diff_matrix = np.abs(label[:, 1].reshape(-1, 1) - pred[:, 1].reshape(1, -1))

    # 找到满足误差范围 TOL 的匹配
    start_matches = start_diff_matrix <= TOL
    end_matches = end_diff_matrix <= TOL

    # 统计每个标签的起点和终点匹配情况
    for i in range(len(label)):
        if np.any(start_matches[i]):
            correct_start_count += 1
            # 取与标签匹配的第一个预测的误差
            start_errors.append(start_diff_matrix[i, start_matches[i]].min())
        if np.any(end_matches[i]):
            correct_end_count += 1
            # 取与标签匹配的第一个预测的误差
            end_errors.append(end_diff_matrix[i, end_matches[i]].min())

    # 找到同时满足起点和终点匹配的情况（片段预测正确的数量）
    correct_segments = np.sum(np.any(start_matches & end_matches, axis=1))

    # 统计起点和终点遗漏的数量（标签中没有匹配的情况）
    missed_start_count = len(label) - correct_start_count
    missed_end_count = len(label) - correct_end_count

    # 统计起点和终点错误的数量（预测中没有匹配的情况）
    incorrect_start_count = len(pred) - np.sum(np.any(start_matches, axis=0))
    incorrect_end_count = len(pred) - np.sum(np.any(end_matches, axis=0))

    # 片段预测错误的数量（预测中没有与任何标签匹配的片段）
    incorrect_segments = len(pred) - np.sum(np.any(start_matches & end_matches, axis=0))

    # 遗漏的片段数量（标签中没有任何匹配的预测）
    missed_segments = len(label) - correct_segments


    return [correct_start_count, correct_end_count, missed_start_count, missed_end_count,
            incorrect_start_count, incorrect_end_count, correct_segments, incorrect_segments, missed_segments,
            start_errors, end_errors]

def train_models(data_name):
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(data_path, 'data', data_name + '_resample.pkl'), 'rb') as input:
        Dataset = pickle.load(input)

    batch_size = 4
    record_list = np.array(list(Dataset[0].keys()))

    kf = KFold(n_splits=5, shuffle=True)
    fold_idx = kf.split(record_list)
    fold_idx = [[train_idx, test_idx] for train_idx, test_idx in fold_idx]


    lead_num_list = [1, 12]
    # seg_len_list = [1, 2, 4, 8][2:3]
    seg_len_list = [1,  4, ]
    kare_num_list = [4, 8, 16, 32, 64]
    TOL_list = [150, 70, 40, 20, 10]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for lead_num in lead_num_list:
        for seg_len in  seg_len_list:
            for kare_num in kare_num_list:
                time = 0
                pred_dict = dict()

                for train_idx, test_idx in fold_idx:
                    time += 1
                    print('lead_num={}, seg_len={}, kare_num={}：5 fold-No.{}'.format(lead_num, seg_len, kare_num, time))
                    train_data = DataLoader(my_dataset(Dataset, record_list[train_idx], lead_num, seg_len), batch_size=batch_size, shuffle=True, drop_last=False)
                    test_data = DataLoader(my_dataset(Dataset, record_list[test_idx], lead_num, seg_len), batch_size=batch_size, shuffle=False,  drop_last=False)

                    # train_dl(train_data, test_data, time, data_name, lead_num, seg_len, kare_num, device)
                    pred_dict = test_dl(test_data, record_list[test_idx],  time, data_name, pred_dict, lead_num, seg_len, kare_num, device)

                label_dict = dict()
                eval_dict = dict()
                for pid in record_list:
                    label_dict[pid] = Dataset[1][pid]['ii']
                    eval_dict[pid] = []
                    for TOL in TOL_list:
                        eval_dict[pid].append(evaluate_detection(label_dict[pid], pred_dict[pid], TOL*409.6/1000))


                save_path = os.path.join('resource', data_name)
                if not os.path.exists(save_path):
                    # 如果路径不存在，则创建该路径
                    os.makedirs(save_path)
                model_filepath = os.path.join(save_path, str(lead_num) + '_' + str(seg_len) + '_' + str(
                    kare_num) + '.pkl')
                with open(model_filepath, 'wb') as f:
                    pickle.dump((label_dict, pred_dict, eval_dict), f)



def main():
    # 设置随机种子
    setup_seed(1)
    data_list = ["ludb", "pdb"]
    for data_name in data_list:
        train_models(data_name)





if __name__ == "__main__":
    main()
