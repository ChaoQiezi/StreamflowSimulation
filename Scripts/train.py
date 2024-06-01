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
"""

from utils.model import LSTMModelFuture, train_epoch
from utils.utils import plot_comparison, plot_loss, cal_nse
import Config

import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchsummary import summary
import tqdm
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_log_error



# 准备
samples_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\Samples\model_train_test.h5'
save_model_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\ModelStorage\model_V01.pth'
# sns.set_style('darkgrid')  # 设置绘制风格

# 读取样本
with h5py.File(samples_path, 'r') as f:
    train_x, train_y, train_ix = torch.tensor(f['train_x'][:], dtype=torch.float32), \
        torch.tensor(f['train_y'][:], dtype=torch.float32), f['train_ix'][:]
    train_size, seq_len, feature_size = train_x.shape
# 数据加载器
train_ds = TensorDataset(train_x, train_y)
train_ds_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
# 输出基本信息
print('当前训练特征项shape: {}\n当前训练目标项shape: {}'.format(train_x.shape, train_y.shape))
print('训练样本数目: {}\n单个样本时间长度: {}\n单个样本特征项数: {}'.format(train_size, seq_len, feature_size))
print('预测期数: {}'.format(train_y.shape[1]))

# 创建模型
model = LSTMModelFuture(feature_size, output_size=Config.pred_len_day).to(Config.DEVICE)
summary(model, input_data=(210, 7))  # 输出模型结构
# 创建损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

# 训练模型
loss_epochs = []
pbar = tqdm.tqdm(range(Config.num_epochs))
for epoch in pbar:
    pbar.set_description('Epoch: {:03}'.format(epoch))
    loss_epochs.append(train_epoch(model, train_ds_loader, optimizer, criterion, pbar))
torch.save(model.state_dict(), save_model_path)  # 保存模型


# 输出模型训练情况
# 绘制迭代损失情况
plot_loss(loss_epochs)
# 绘制拟合情况和评估
model.load_state_dict(torch.load(save_model_path))  # 加载存储模型
train_ix = np.vectorize(lambda x: x.decode('utf-8'))(train_ix)  # 转码
train_ix = pd.DataFrame(train_ix, columns=['ID'])
train_ix[['站名', 'date']] = train_ix['ID'].str.split('_', expand=True)
train_ix['date'] = pd.to_datetime(train_ix['date'], format='%Y%m%d%H')
model.eval()  # 评估模式
for station_name in Config.station_names:
    # 预测
    temp_ix = train_ix[train_ix['站名'] == station_name]['date']
    temp_x = train_x[train_ix['站名'] == station_name].to(Config.DEVICE)
    temp_y_obs = train_y[train_ix['站名'] == station_name].squeeze()
    temp_y_pred = model(temp_x).detach().cpu().numpy().squeeze()
    temp_y_pred[temp_y_pred < 0] = 0  # 负数替换为0
    # 绘制
    save_path = os.path.join(Config.Assets_charts_dir, 'pred_real_train_{}.png'.format(station_name))
    plot_comparison(temp_ix, temp_y_obs, temp_y_pred, station_name, save_path=save_path)
    # 计算训练集的评估指标
    r2 = r2_score(temp_y_obs, temp_y_pred)
    rmse = mean_squared_log_error(temp_y_obs, temp_y_pred)
    nse = cal_nse(temp_y_obs, temp_y_pred)
    print('训练集评估结果--站名: {}; R2: {:0.2}; RMSE: {:0.2}; NSE: {:0.2}'.format(station_name, r2, rmse, nse))
print('模型训练结束.')
