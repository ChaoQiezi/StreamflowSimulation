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
import tqdm
import time

start_time = time.time()
# 记忆期和预见期参考: 考虑记忆时间的LSTM模型在赣江流域径流预报中的应用_胡乐怡.pdf
memory_days = [210]
"""
似乎论文均使用过去210天的样本作为输入
(基于先前研究，且考虑到流域径流形成过程，积雪形成消融周期约为7个月，并经过多次调整序列长度实验，序列长度设定为 210)
"""
predict_days = [1, 2, 3, 4, 5, 7, 14, 21]
# memory_days = np.arange(150, 300, 20)  # 生成记忆期列表
# predict_days = np.arange(1, 15)  # 生成预见期列表

if __name__ == '__main__':
    for seq_len_day in memory_days:
        for pred_len_day in predict_days:
            generate_time_series_samples(seq_len_day, pred_len_day)
            train(seq_len_day, pred_len_day)
            eval(seq_len_day, pred_len_day)

    end_time = time.time()
    print('主程序结束(耗时: {:0.2f} min).'.format((end_time - start_time) / 60))
