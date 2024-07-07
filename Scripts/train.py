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


def train(seq_len_day=Config.seq_len_day, pred_len_day=Config.pred_len_day):
    """
    模型训练
    :param seq_len_day: 记忆期
    :param pred_len_day: 预见期
    :return: None
    """

    print('_' * 50)
    print('记忆期: {} day; 预见期: {} day;  训练中······'.format(seq_len_day, pred_len_day))
    print('_' * 50)

    # 准备
    samples_file_name = 'train_test_m{}day_p{}day.h5'.format(seq_len_day, pred_len_day)
    model_file_name = 'model_m{}day_p{}day.pth'.format(seq_len_day, pred_len_day)
    samples_path = os.path.join(Config.samples_dir, samples_file_name)
    save_model_path = os.path.join(Config.models_dir, model_file_name)

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
    show_samples_info(train_x.shape, train_y.shape, train_ix=train_ix, pred_len_day=pred_len_day)

    # 模型训练准备
    model = LSTMModelFuture(feature_size, output_size=pred_len_day).to(Config.DEVICE)  # 创建模型
    summary(model, input_data=(seq_len_day, Config.feature_size))  # 输入(samples_num<忽略不传入>, 时间步, 特征项数)输出模型结构
    criterion = nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)  # 优化器

    # 为减少训练时间, 若存在已训练好的模型则跳过模型训练部分
    if not os.path.exists(save_model_path):
        # 训练模型
        loss_epochs = []
        pbar = tqdm.tqdm(range(Config.num_epochs))
        for epoch in pbar:
            pbar.set_description('Epoch: {:03}'.format(epoch))
            loss_epochs.append(train_epoch(model, train_ds_loader, optimizer, criterion, pbar))
        torch.save(model.state_dict(), save_model_path)  # 保存模型

        # 输出模型训练情况
        # 绘制迭代损失情况
        save_loss_path = os.path.join(Config.result_dir, 'train_loss_m{}day_p{}day.png'.format(seq_len_day, pred_len_day))
        plot_loss(loss_epochs, save_path=save_loss_path)

    # 绘制拟合情况和评估
    model.load_state_dict(torch.load(save_model_path))  # 加载存储模型
    model.eval()  # 评估模式
    with torch.no_grad():
        for station_name in Config.station_names:
            # 预测
            temp_ix = train_ix[train_ix['站名'] == station_name][[x for x in train_ix.columns if x != '站名']]
            temp_x = train_x[train_ix['站名'] == station_name].to(Config.DEVICE)
            temp_y_obs = train_y[train_ix['站名'] == station_name]
            temp_y_pred = model(temp_x).detach().cpu().numpy()
            temp_y_pred[temp_y_pred < 0] = 0  # 负数替换为0
            # 反归一化
            scalers = joblib.load(Config.scalers_path)
            # temp_y_obs = scalers['model__y_scaler'].inverse_transform(pd.DataFrame({Config.target_name[0]: temp_y_obs})).squeeze()
            # temp_y_pred = scalers['model__y_scaler'].inverse_transform(pd.DataFrame({Config.target_name[0]: temp_y_pred})).squeeze()
            temp_y_obs = scalers['model__y_scaler'].inverse_transform(
                pd.DataFrame(temp_y_obs))
            temp_y_pred = scalers['model__y_scaler'].inverse_transform(
                pd.DataFrame(temp_y_pred))
            # 计算训练集的评估指标
            r2 = r2_score(temp_y_obs, temp_y_pred)
            rmse = mean_squared_log_error(temp_y_obs, temp_y_pred)
            nse = cal_nse(temp_y_obs, temp_y_pred)
            print('训练集评估结果--站名: {}; R2: {:0.4f}; RMSE: {:0.4f}; NSE: {:0.4f}'.format(station_name, r2, rmse,
                                                                                              nse))
            # 绘制
            # 合并重叠部分(简单均值)
            temp_y_pred = pd.DataFrame(temp_y_pred)
            for ix in range(temp_y_pred.shape[1]):
                temp_y_pred[ix] = temp_y_pred[ix].shift(ix)
            temp_y_pred = np.nanmean(temp_y_pred, axis=1)
            temp_y_obs = temp_y_obs[:, 0]  # 直接取第一列即可
            temp_ix = temp_ix['0_date']
            # combined_preds = np.zeros(temp_y_pred.shape[0] + Config.pred_len_day - 1)
            # counts = np.zeros_like(combined_preds)
            # for ix, line in enumerate(temp_y_pred):
            #     combined_preds[ix:ix+Config.pred_len_day] += line
            #     counts[ix:ix+Config.pred_len_day] += 1
            # combined_preds /= counts
            #
            # combined_obss = np.zeros(temp_y_obs.shape[0] + Config.pred_len_day - 1)
            # counts = np.zeros_like(combined_obss)
            # for ix, line in enumerate(temp_y_obs):
            #     combined_obss[ix:ix + Config.pred_len_day] += line
            #     counts[ix:ix + Config.pred_len_day] += 1
            # combined_obss /= counts

            # save_path = os.path.join(Config.Assets_charts_dir, 'pred_obs_train_{}.png'.format(station_name))
            save_path = os.path.join(Config.result_dir,
                                     'pred_obs_train_{}_m{}day_p{}day.png'.format(station_name, seq_len_day,
                                                                                 pred_len_day))
            plot_comparison(temp_ix, temp_y_obs, temp_y_pred, station_name, save_path=save_path)
    print('模型(记忆期:{}day; 预见期:{}day) 训练结束.'.format(seq_len_day, pred_len_day))
    print('=' * 80)


if __name__ == '__main__':
    train()
