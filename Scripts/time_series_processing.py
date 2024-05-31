# @Author   : ChaoQiezi
# @Time     : 2024/5/18  17:11
# @FileName : time_series_processing.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 将各个特征项及其目标项的时间序列数据集进行整合、去除无效值、滤波等操作
"""

import os
from datetime import datetime
import pandas as pd
from pykalman import KalmanFilter  # 卡尔曼滤波器
from utils.utils import view_info, fast_viewing
import Config

# 准备
era5_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\ERA5\Era5_2010_2015.xlsx'
rainfall_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\模拟资料\尼洋河数据\Rainfall.xlsx'
streamflow_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\模拟资料\尼洋河数据\Streamflow.xlsx'
out_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\LSTM\preprocessed_features_target.xlsx'
station_names = Config.station_names
# feature_names = ['气温', '降水', '气压', '相对湿度', '风速', '风向', '日照', '地温', '蒸发', '降水量', '平均流量']
# feature_names = ['气温', '气压', '相对湿度', '风速', '日照', '地温', '降水量']  # 去除风向和蒸发(NAN过多)
feature_names = Config.feature_names
target_name = Config.target_name
feature_target_names = Config.feature_target_names

# 读取和整合
dfs = []
for station_name in station_names:
    rainfall_df = pd.read_excel(rainfall_path, sheet_name=station_name)
    streamflow_df = pd.read_excel(streamflow_path, sheet_name=station_name)
    era5_df = pd.read_excel(era5_path, sheet_name=station_name)
    rainfall_df['日期'] = pd.to_datetime(rainfall_df['日期'], format='%Y/%m/%d')
    streamflow_df['日期'] = pd.to_datetime(streamflow_df['日期'], format='%Y/%m/%d')
    era5_df['日期'] = era5_df.apply(lambda x: datetime(x['年'], x['月'], x['日']), axis=1)
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
view_info(df, feature_target_names)
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
view_info(df, feature_target_names)
# 再次快速浏览概况比对插值情况(绘制简单的折线图展示)
save_path = os.path.join(Config.Assets_charts_dir, 'post_interpolation_fast_view.png')
fast_viewing(df, station_names, feature_target_names, save_path)
# 进行卡尔曼滤波
for station_name in station_names:
    observations = df.loc[df['站名'] == station_name, feature_names]
    filter = filter = KalmanFilter(n_dim_obs=len(feature_names), n_dim_state=len(feature_names))  # 实例化滤波器
    state_means, _ = filter.em(observations, n_iter=10).smooth(observations)
    df.loc[df['站名'] == station_name, feature_names] = state_means
# 检查滤波后基本信息
print('滤波后: ')
view_info(df, feature_target_names)
# 再次快速浏览概况查看滤波情况
save_path = os.path.join(Config.Assets_charts_dir, 'filtered_fast_view.png')
fast_viewing(df, station_names, feature_names, save_path)

# 输出
df.to_excel(out_path, index=False)
print('时间序列数据集预处理结束.')