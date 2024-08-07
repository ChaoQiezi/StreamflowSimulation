# @Author   : ChaoQiezi
# @Time     : 2024/5/18  17:11
# @FileName : time_series_processing.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 将各个特征项及其目标项的时间序列数据集进行整合、去除无效值(插值)、滤波操作
"""

import Config

import os
from datetime import datetime
import pandas as pd
from pykalman import KalmanFilter  # 卡尔曼滤波器
from utils.utils import set_show_nan, fast_viewing


# 准备
era5_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\ERA5\Era5_2010_2015.xlsx'
rainfall_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\模拟资料\尼洋河数据\Rainfall.xlsx'
streamflow_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\模拟资料\尼洋河数据\Streamflow.xlsx'
out_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\preprocessed_features_target.xlsx'
station_names = Config.station_names  # 站点列表
# feature_names = ['气温', '降水', '气压', '相对湿度', '风速', '风向', '日照', '地温', '蒸发', '降水量', '平均流量']
# feature_names = ['气温', '气压', '相对湿度', '风速', '日照', '地温', '降水量']  # 去除风向和蒸发(NAN过多)
feature_names = Config.feature_names   # 特征项名称
target_name = Config.target_name  # 目标项(径流)
feature_target_names = Config.feature_target_names  # 特征项和目标项的名称列表

# 读取和整合
dfs = []  # df: pd.DataFrame, 存储
for station_name in station_names:
    # 读取当前站点的特征项和目标项excel表
    rainfall_df = pd.read_excel(rainfall_path, sheet_name=station_name)
    streamflow_df = pd.read_excel(streamflow_path, sheet_name=station_name)
    era5_df = pd.read_excel(era5_path, sheet_name=station_name)
    # 日期格式控制
    rainfall_df['日期'] = pd.to_datetime(rainfall_df['日期'], format='%Y/%m/%d')
    streamflow_df['日期'] = pd.to_datetime(streamflow_df['日期'], format='%Y/%m/%d')
    era5_df['日期'] = era5_df.apply(lambda x: datetime(x['年'], x['月'], x['日']), axis=1)
    # 整合为XY数据集
    era5_df = era5_df.merge(rainfall_df, how='left', on=['日期'])
    era5_df = era5_df.merge(streamflow_df, how='left', on=['日期'])
    era5_df['站点'] = station_name
    dfs.append(era5_df)
    print('匹配站点数据: {}'.format(station_name))
df = pd.concat(dfs, axis=0)
df['date'] = df.apply(lambda x: pd.to_datetime('{}{:02}{:02}'.format(x['年'], x['月'], x['日'])), axis=1)
df = df[['date', '纬度', '经度', '站名', '站号'] + feature_target_names]

# 检查数据集的基本信息和无效值
print('插值前: ')
set_show_nan(df, feature_target_names)
# 快速浏览概况(绘制简单的折线图展示)
save_path = os.path.join(Config.Assets_charts_dir, 'pre_interpolation_fast_view.png')
fast_viewing(df, station_names, feature_target_names, save_path)
# 多项式插值
for station_name in station_names:
    temp_df = df[df['站名'] == station_name]
    df.loc[df['站名'] == station_name, feature_target_names] = temp_df[feature_target_names].interpolate(
        method='polynomial', order=3)
# 再次检查info和nan
print('插值后: ')
set_show_nan(df, feature_target_names)
# 再次快速浏览概况比对插值情况(绘制简单的折线图展示)
save_path = os.path.join(Config.Assets_charts_dir, 'post_interpolation_fast_view.png')
fast_viewing(df, station_names, feature_target_names, save_path)

# 进行卡尔曼滤波
for station_name in station_names:
    observations = df.loc[df['站名'] == station_name, feature_names]
    filter = KalmanFilter(n_dim_obs=len(feature_names), n_dim_state=len(feature_names))  # 实例化滤波器
    state_means, _ = filter.em(observations, n_iter=10).smooth(observations)
    df.loc[df['站名'] == station_name, feature_names] = state_means
# 检查滤波后基本信息
print('滤波后: ')
set_show_nan(df, feature_target_names)
# 再次快速浏览概况查看滤波情况
save_path = os.path.join(Config.Assets_charts_dir, 'filtered_fast_view.png')
fast_viewing(df, station_names, feature_names, save_path)

# 输出
df.to_excel(out_path, index=False)
print('时间序列数据集预处理结束.')
