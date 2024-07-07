# @Author   : ChaoQiezi
# @Time     : 2024/5/30  21:21
# @FileName : generate_time_series_samples.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 从time_series_processing.py预处理好的时间序列数据集中生成可供后续模型输入训练的样本集
"""

import os
import pandas as pd
from utils.utils import generate_samples  # 生成样本
import Config


def generate_time_series_samples(seq_len_day=210, pred_len_day=1):
    """
    生成时间序列的样本供模型输入
    :param seq_len_day: 记忆期
    :param pred_len_day: 预测期
    :return: None
    """

    print('_' * 50)
    print('记忆期: {} day; 预见期: {} day;  生成样本中······'.format(seq_len_day, pred_len_day))
    print('_' * 50)
    # 准备
    time_series_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\preprocessed_features_target.xlsx'
    out_samples_file_name = 'train_test_m{}day_p{}day.h5'.format(seq_len_day, pred_len_day)
    out_path = os.path.join(Config.samples_dir, out_samples_file_name)

    # 读取
    df = pd.read_excel(time_series_path)
    generate_samples(df, Config.feature_names, Config.target_name, out_path, time_col_name='date', st_col_name='站名',
                     window_size=seq_len_day, future_size=pred_len_day)
    print('-' * 50)
    print('样本记忆期: {} day'.format(seq_len_day))
    print('样本预见期: {} day'.format(pred_len_day))
    print('训练集和测试集的划分时间节点: {}'.format(Config.split_time.strftime('%Y-%m-%d')))
    print('-' * 50)
    print('生成样本(记忆期:{}day; 预见期:{}day)结束.'.format(seq_len_day, pred_len_day))
    print('=' * 80)


if __name__ == '__main__':
    generate_time_series_samples(210, 1)

