# @Author   : ChaoQiezi
# @Time     : 2024/6/2  11:11
# @FileName : main.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 主程序, 基于LSTM模型利用气象因子(特征项)和径流(目标项)训练和预测在不同记忆期和不同预见期的结果
"""

from Scripts.generate_time_series_samples import generate_time_series_samples
from Scripts.train import train
from Scripts.eval import eval

import numpy as np

memory_days = np.arange(150, 360, 20)  # 生成记忆期列表
predict_days = np.arange(1, 14)  # 生成预见期列表

if __name__ == '__main__':
    for seq_len_day in memory_days:
        for pred_len_day in predict_days:
            generate_time_series_samples(seq_len_day, pred_len_day)
            train(seq_len_day, pred_len_day)
            eval(seq_len_day, pred_len_day)
    print('主程序结束.')
