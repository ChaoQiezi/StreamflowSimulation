# @Author   : ChaoQiezi
# @Time     : 2024/6/1  14:37
# @FileName : eval.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 预测和评估模型

"""

import Config
from utils.utils import decode_time_col, show_samples_info, plot_comparison, cal_nse
from utils.model import LSTMModelFuture

import os
import h5py
import joblib
from sklearn.metrics import r2_score, mean_squared_log_error, mean_absolute_error
import numpy as np
import pandas as pd
import torch

# 准备
samples_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\Samples\model_train_test.h5'
model_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\ModelStorage\model_V01.pth'
pred_obs_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\Result\pred_obs.xlsx'
eval_indicator_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\Result\eval_indicators.xlsx'

# 读取样本
with h5py.File(samples_path) as f:
    # 读取训练集和测试集的特征项、目标项和索引(标识时间列)
    train_x, train_y, train_ix, test_x, test_y, test_ix = torch.tensor(f['train_x'][:], dtype=torch.float32),\
        torch.tensor(f['train_y'][:], dtype=torch.float32),\
        f['train_ix'][:],\
        torch.tensor(f['test_x'][:], dtype=torch.float32),\
        torch.tensor(f['test_y'][:], dtype=torch.float32),\
        f['test_ix'][:]
    # 索引项编码和分割
    train_ix = decode_time_col(train_ix)
    test_ix = decode_time_col(test_ix)
# 获取和输出数据集基本信息
show_samples_info(train_x.shape, train_y.shape, test_x.shape, test_y.shape, train_ix, test_ix)
# 评估
model = LSTMModelFuture(Config.feature_size, output_size=1).to(Config.DEVICE)
model.load_state_dict(torch.load(model_path))  # 加载模型
model.eval()  # 评估模式
eval_indicators = {}
with torch.no_grad():
    with pd.ExcelWriter(pred_obs_path, mode='w', engine='openpyxl') as writer:
        for station_name in Config.station_names:
            # 预测
            # temp_ix = train_ix[train_ix['站名'] == station_name]['date']
            temp_ix = train_ix[train_ix['站名'] == station_name][[x for x in train_ix.columns if x != '站名']]
            temp_x = train_x[train_ix['站名'] == station_name].to(Config.DEVICE)
            temp_y_obs = train_y[train_ix['站名'] == station_name].detach().cpu().numpy().squeeze()
            temp_y_pred = model(temp_x).detach().cpu().numpy().squeeze()
            temp_y_pred[temp_y_pred < 0] = 0  # 负数替换为0
            # 反归一化
            scalers = joblib.load(Config.scalers_path)  # 加载标准化器
            temp_y_obs = scalers['model__y_scaler'].inverse_transform(
                pd.DataFrame({Config.target_name[0]: temp_y_obs})).squeeze()
            temp_y_pred = scalers['model__y_scaler'].inverse_transform(
                pd.DataFrame({Config.target_name[0]: temp_y_pred})).squeeze()
            # 计算评估指标
            r2 = r2_score(temp_y_obs, temp_y_pred)
            rmse = mean_squared_log_error(temp_y_obs, temp_y_pred)
            mae = mean_absolute_error(temp_y_obs, temp_y_pred)
            nse = cal_nse(temp_y_obs, temp_y_pred)
            eval_indicators[station_name] = [r2, rmse, mae, nse]
            # 绘制
            # 合并重叠部分
            combined_preds = np.zeros(temp_y_pred.shape[0] + Config.pred_len_day - 1)
            counts = np.zeros_like(combined_preds)
            for ix, line in enumerate(temp_y_pred):
                combined_preds[ix:ix + Config.pred_len_day] += line
                counts[ix:ix + Config.pred_len_day] += 1
            combined_preds /= counts

            combined_obss = np.zeros(temp_y_obs.shape[0] + Config.pred_len_day - 1)
            counts = np.zeros_like(combined_obss)
            for ix, line in enumerate(temp_y_obs):
                combined_obss[ix:ix + Config.pred_len_day] += line
                counts[ix:ix + Config.pred_len_day] += 1
            combined_obss /= counts

            save_path = os.path.join(Config.Assets_charts_dir, 'pred_obs_test_{}.png'.format(station_name))
            plot_comparison(temp_ix, temp_y_obs, temp_y_pred, station_name, save_path=save_path)
            # 存储结果
            temp_result = pd.DataFrame(
                {
                    'date': temp_ix,
                    'observation(mm)': temp_y_obs,
                    'prediction(mm)': temp_y_pred
                }
            )

            # 输出
            temp_result.to_excel(writer, sheet_name=station_name, index=False)
# 输出评估指标
eval_indicators = pd.DataFrame(eval_indicators).transpose()
eval_indicators.columns = ['R2', 'RMSE', 'MAE', 'NSE']
eval_indicators.to_excel(eval_indicator_path)
