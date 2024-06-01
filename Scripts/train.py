# @Author   : ChaoQiezi
# @Time     : 2024/5/30  22:28
# @FileName : train.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 模型训练

我似乎应该规范一下我关于模型训练的代码结构:
    -- 数据集
        -- 数据集读取
        -- 数据集装载器
        -- 数据集和训练的基本信息输出
    -- 模型
        -- 定义模型
        -- 模型-损失函数
        -- 模型-优化器
    -- 训练
        -- 迭代epoch
        -- 通过train_epoch类似函数实现
        -- 保存模型
        -- 可视化训练结果
        -- 适当计算评估指标(训练集)
"""

from utils.model import LSTMModelFuture, train_epoch
from utils.utils import plot_comparison, plot_loss, cal_nse, decode_time_col, show_samples_info
import Config

import os
import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchsummary import summary
import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_log_error

# 准备
samples_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\Samples\model_train_test_2day.h5'
save_model_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\ModelStorage\model_2day.pth'

# 读取样本
with h5py.File(samples_path, 'r') as f:
    train_x, train_y, train_ix = torch.tensor(f['train_x'][:], dtype=torch.float32), \
        torch.tensor(f['train_y'][:], dtype=torch.float32), f['train_ix'][:]
    train_size, seq_len, feature_size = train_x.shape
    # 对标识时间列解码和分割
    train_ix = decode_time_col(train_ix)
# 数据加载器
train_ds = TensorDataset(train_x, train_y)
train_ds_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
# 输出基本信息
show_samples_info(train_x.shape, train_y.shape, train_ix=train_ix)

# 模型训练准备
model = LSTMModelFuture(feature_size, output_size=Config.pred_len_day).to(Config.DEVICE)  # 创建模型
summary(model, input_data=(210, 7))  # 输出模型结构
criterion = nn.MSELoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr) # 优化器
# 训练模型
loss_epochs = []
pbar = tqdm.tqdm(range(Config.num_epochs))
best_loss = 0
for epoch in pbar:
    pbar.set_description('Epoch: {:03}'.format(epoch))
    loss_epochs.append(train_epoch(model, train_ds_loader, optimizer, criterion, pbar))
torch.save(model.state_dict(), save_model_path)  # 保存模型

# 输出模型训练情况
# 绘制迭代损失情况
plot_loss(loss_epochs)
# 绘制拟合情况和评估
model.load_state_dict(torch.load(save_model_path))  # 加载存储模型
model.eval()  # 评估模式
with torch.no_grad():
    for station_name in Config.station_names:
        # 预测
        temp_ix = train_ix[train_ix['站名'] == station_name]['date']
        temp_x = train_x[train_ix['站名'] == station_name].to(Config.DEVICE)
        temp_y_obs = train_y[train_ix['站名'] == station_name].squeeze()
        temp_y_pred = model(temp_x).detach().cpu().numpy().squeeze()
        temp_y_pred[temp_y_pred < 0] = 0  # 负数替换为0
        # 反归一化
        scalers = joblib.load(Config.scalers_path)
        # temp_y_obs = scalers['model__y_scaler'].inverse_transform(pd.DataFrame({Config.target_name[0]: temp_y_obs})).squeeze()
        # temp_y_pred = scalers['model__y_scaler'].inverse_transform(pd.DataFrame({Config.target_name[0]: temp_y_pred})).squeeze()
        temp_y_obs = scalers['model__y_scaler'].inverse_transform(
            pd.DataFrame(temp_y_obs)).squeeze()
        temp_y_pred = scalers['model__y_scaler'].inverse_transform(
            pd.DataFrame(temp_y_pred)).squeeze()
        # 绘制
        save_path = os.path.join(Config.Assets_charts_dir, 'pred_real_train_{}.png'.format(station_name))
        plot_comparison(temp_ix, temp_y_obs, temp_y_pred, station_name, save_path=save_path)
        # 计算训练集的评估指标
        r2 = r2_score(temp_y_obs, temp_y_pred)
        rmse = mean_squared_log_error(temp_y_obs, temp_y_pred)
        nse = cal_nse(temp_y_obs, temp_y_pred)
        print('训练集评估结果--站名: {}; R2: {:0.2}; RMSE: {:0.2}; NSE: {:0.2}'.format(station_name, r2, rmse, nse))
print('模型训练结束.')
