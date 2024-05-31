# @Author   : ChaoQiezi
# @Time     : 2024/5/30  22:28
# @FileName : train.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 模型训练
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils.model import LSTMModelFuture, DEVICE, train_epoch
import Config


# 准备
samples_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\Samples\model_train_test.h5'
save_model_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\ModelStorage\model_V01.pth'

# 读取样本
with h5py.File(samples_path, 'r') as f:
    train_x, train_y = torch.tensor(f['train_x'][:], dtype=torch.float32), torch.tensor(f['train_y'][:], dtype=torch.float32)
    train_size, seq_len, feature_size = train_x.shape
# 数据加载器
train_ds = TensorDataset(train_x, train_y)
train_ds_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
# 输出基本信息
print('当前训练特征项shape: {}\n当前训练目标项shape: {}'.format(train_x.shape, train_y.shape))
print('训练样本数目: {}\n单个样本时间长度: {}\n单个样本特征项数: {}'.format(train_size, seq_len, feature_size))
print('预测期数: {}'.format(train_y.shape[1]))

# 创建模型
model = LSTMModelFuture(feature_size, output_size=Config.pred_len_day)
summary(model, input_data=(210, 7))  # 输出模型结构
# 创建损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

# 训练模型
loss_epochs = []
pbar = tqdm.tqdm(range(Config.num_epochs))
for epoch in pbar:
    pbar.set_description('Epoch: {:03}'.format(epoch))
    train_epoch()
