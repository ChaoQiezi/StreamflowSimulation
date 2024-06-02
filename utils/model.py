# @Author   : ChaoQiezi
# @Time     : 2024/4/26  11:27
# @FileName : model.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放模型相关等类和函数
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from Config import DEVICE


# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTMModelSame(nn.Module):
    def __init__(self, input_size=15, hidden_size=512, num_layers=3, output_size=1):
        super().__init__()
        # self.causal_conv1d = nn.Conv1d(input_size, 128, 5)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):  # x.shape=(batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out.shape=(batch_szie, seq_len<123>, hidden_size<128>)
        reg_out = self.regression(lstm_out).squeeze(-1)  # .squeeze(-1)  # 去除最后一个维度

        return reg_out


import torch
import torch.nn as nn


class LSTMModelFuture(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=1, num_layers=3, dropout_rate=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):  # x.shape=(batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out.shape=(batch_szie, seq_len<123>, hidden_size<128>)
        """
        lstm_out: shape=(batch_size, seq_len, hidden_size)  # 最后一层所有时间步的输出
        h_n, c_n: shape=(num_layer, batch_size, hidden_size), 最后一个时间步的隐藏状态和细胞状态
        """

        reg_out = self.regression(h_n[-1, :, :])  # .squeeze(-1)  # 去除最后一个维度

        return reg_out


def train_epoch(model, data_loader, optimizer, loss_func=nn.MSELoss(), pbar=None):
    """
    用于模型训练
    :param model: 定义好的模型
    :param data_loader: 数据加载器
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :param pbar: 进度条, 用于更新进度条显示内容
    :return: 返回loss
    """
    model.train()  # 训练模式

    # 每次迭代batch_size样本
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        # 清理存在NAN的样本
        valid_mask = ~(torch.isnan(inputs).any(dim=1).any(dim=1) | torch.isnan(targets).any(1))
        if valid_mask.any():  # 但凡有一个样本存在有效值
            inputs, targets = inputs[valid_mask], targets[valid_mask]
        else:
            continue

        optimizer.zero_grad()  # 清除存储梯度
        outputs = model(inputs)  # 模型预测
        loss = loss_func(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播,计算梯度
        optimizer.step()  # 更新权重

        if pbar is not None:
            pbar.set_postfix_str("Loss: {:.4f}".format(loss.item()))
    else:
        return loss.item()

# def train_epoch_ignore_nan(model, data_loader, optimizer, loss_func=nn.MSELoss(), pbar=None):
#     model.train()  # 训练模式
#
#     # 每次迭代batch_size样本
#     for inputs, targets in data_loader:
#         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#         # 检测nan
#         valid_mask = ~(torch.isnan(inputs).any(dim=1).any(dim=1) | torch.isnan(targets).any(1))
#         if not valid_mask.any():  # 所有样本均存在无效值
#             continue
#
#         optimizer.zero_grad()  # 清除存储梯度
#         outputs = model(inputs[[0]])  # 模型预测
#         loss = loss_func(outputs[valid_mask], targets[valid_mask])  # 计算损失,忽略存在nan的样本
#         loss.backward()  # 反向传播,计算梯度
#         optimizer.step()  # 更新权重
#
#         if pbar is not None:
#             pbar.set_postfix_str("Loss: {:.4f}".format(loss.item()))
#     else:
#         return loss.item()
