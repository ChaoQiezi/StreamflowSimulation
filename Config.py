# @Author   : ChaoQiezi
# @Time     : 2024/5/15  21:16
# @FileName : Config.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 作为配置文件, 存储基本参数、模型参数、基础路径等等变量以及初始化字体、创建初始文件夹等操作
"""

import os
import joblib
import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from datetime import datetime

# 设置相关
mpl.rcParams['font.family'] = 'Microsoft YaHei'  # 可正常显示中文
mpl.rcParams['axes.unicode_minus'] = True  # 显示正负号
sns.set_style('darkgrid')  # 设置绘制风格
# plt.rcParams['font.family'] = 'Simhei'
# plt.rcParams['font.family'] = 'Times New Roman'

# 路径相关
Assets_dir = r'I:\PyProJect\StreamflowSimulation\Assets'
Assets_charts_dir = os.path.join(Assets_dir, 'Charts')
if not os.path.exists(Assets_charts_dir):  # 判断文件夹绝对路径是否存在, 不存在则创建该文件夹
    os.makedirs(Assets_charts_dir)
samples_dir = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\Samples'  # 存储训练和预测样本文件(.h5)的目录
models_dir = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\ModelStorage'  # 存储训练好的模型的目录
result_dir = r'H:\Datasets\Objects\StreamflowSimulation\Data\Result'  # 输出图表等结果的目录

# 数据基本信息相关
station_names = ['巴河桥', '更张', '工布江达']
feature_names = ['气温', '气压', '相对湿度', '风速', '日照', '地温', '降水量']
target_name = ['平均流量']
feature_target_names = feature_names + target_name
feature_size = len(feature_names)
# 数据集划分相关
split_time = datetime(2014, 1, 1)  # 数据集的划分时间节点
seq_len_day = 210  # 记忆时间(时间分辨率: day)
pred_len_day = 1  # 预见期(day)

# 模型相关
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 是否有显卡并加速模型训练
num_epochs = 30  # 训练次数
lr = 1e-4  # 学习率
batch_size = 32  # 批次大小
best_loss = np.inf
scalers_path = os.path.join(Assets_dir, 'scalers.pkl')  # 归一化器存储, 用于后续预测值的反归一化
if not os.path.exists(scalers_path):
    joblib.dump({}, scalers_path)
else:
    scalers = joblib.load(scalers_path)

