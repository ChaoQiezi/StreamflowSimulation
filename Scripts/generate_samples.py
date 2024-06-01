# @Author   : ChaoQiezi
# @Time     : 2024/5/30  21:21
# @FileName : generate_samples.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 从time_series_processing.py预处理好的时间序列数据集中生成可供后续模型输入训练的样本集
"""
import h5py
import pandas as pd
from utils.utils import generate_samples  # 生成样本
import Config

# 准备
time_series_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\preprocessed_features_target.xlsx'
out_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\Samples\model_train_test_2day.h5'

# 读取
df = pd.read_excel(time_series_path)
generate_samples(df, Config.feature_names, Config.target_name, out_path, time_col_name='date', st_col_name='站名',
                 window_size=Config.seq_len_day, step_size=Config.pred_len_day)
print('_' * 50)
print('样本记忆期: {} day'.format(Config.seq_len_day))
print('样本预见期: {} day'.format(Config.pred_len_day))
print('训练集和测试集的划分时间节点: {}'.format(Config.split_time.strftime('%Y-%m-%d')))
print('_' * 50)
print('生成样本结束.')

