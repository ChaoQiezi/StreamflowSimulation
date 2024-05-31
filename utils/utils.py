# @Author   : ChaoQiezi
# @Time     : 2024/4/25  19:21
# @FileName : utils.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放常用工具和函数
"""

import os.path
import h5py
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Union

import Config
# 初始化参数
# split_time = datetime(2020, 7, 10)  # 划分时间节点, 5~7月为训练集, 8月为验证集, 约为3:1
from Config import split_time, seq_len_day, pred_len_day

time_name = None


def create_xy_same(dataset, x_col_names: list, y_col_name: list, window_size=30, step_size=1, st_col_name='st'):
    """
    为时间序列创建滑动窗口生成样本, XY同时期
    :param dataset:待划分的数据集
    :param window_size: 时间滑动窗口大小
    :param step_size: 窗口移动步长
    :return: 元组(train_x, train_y, train_ix)
    """
    global time_name

    xs, ys, ixs = [], [], []
    for st_id in dataset[st_col_name].unique():
        cur_data = dataset[dataset[st_col_name] == st_id].reset_index(drop=True)
        for start in range(0, len(cur_data) - window_size + 1, step_size):
            end = start + window_size - 1
            xs.append(cur_data.loc[start:end, x_col_names])
            ys.append(cur_data.loc[start:end, y_col_name])
            ixs.append(cur_data.loc[start:end, [time_name]].apply(lambda x: x[time_name].strftime('%Y%m%d'), axis=1))
            # ixs.append(cur_data.loc[start:end, ['st', 'ymdh']].apply(
            #     lambda x: str(x['st']) + '_' + x['ymdh'].strftime('%Y%m%d'), axis=1))
    xs = np.array(xs)
    ys = np.array(ys).squeeze(axis=-1)  # 去除最后一个形状大小为1的维度
    ixs = np.array(ixs).squeeze()

    # return train_x, train_y, train_ix
    return xs, ys, ixs


def create_xy_future(dataset, x_col_names: list, y_col_name: list, window_size=seq_len_day, step_size=1, future_size=pred_len_day,
                     st_col_name='st', format_date='%Y%m%d%H'):
    """
    为时间序列基于滑动窗口生成样本, X和Y不同时期
    :param dataset: 待划分的数据
    :param window_size: 时间窗口的大小(理解为LSTM的记忆时间)
    :param step_size: 窗口移动的步长(避免样本之间的高度相似性)
    :param future_size: y对应的未来大小(对应LSTM的可预见期)
    :return: 元组(train_x, train_y)
    """
    global time_name

    xs, ys, ixs = [], [], []
    for st_id in dataset[st_col_name].unique():  # 迭代站点
        cur_data = dataset[dataset[st_col_name] == st_id].reset_index(drop=True)
        for x_start in range(0, len(cur_data) - (window_size + future_size) + 1, step_size):
            x_end = x_start + window_size - 1  # -1是因为.loc两边为闭区间
            y_start = x_end + 1
            y_end = y_start + future_size - 1
            # x_cols = ['PRCP'] + ['mwhs{:02d}'.format(_ix) for _ix in range(1, 16)]
            xs.append(cur_data.loc[x_start:x_end, x_col_names])
            ys.append(cur_data.loc[y_start:y_end, y_col_name])
            # ixs.append(cur_data.loc[y_start:y_end, time_name])
            ixs.append(
                cur_data.loc[y_start:y_end, :].apply(lambda x: str(x[st_col_name]) + '_' + x[time_name].strftime(format_date), axis=1))
    xs = np.array(xs)
    ys = np.array(ys).squeeze(axis=-1)      # 去除最后一个形状大小为1的维度, 下同
    ixs = np.array(ixs).squeeze()

    return xs, ys, ixs


def generate_samples(df, x_col_names: list, y_col_name: Union[str, list], out_path, time_col_name='date', format_date=None,
                     split_time=split_time, is_same_periods=False, model_fix='', **kwargs):
    """
    基于滑动窗口对时间序列数据集进行训练和测试样本的生成, 此外还进行了数据集的标准化
    :param df: 包含特征项和目标项、以及标识时间列的pd.Dataframe结构
    :param x_col_names: 特征项的列名称
    :param y_col_name: 目标项的列名称
    :param out_path: 生成的样本集的输出路径
    :param time_col_name: 标识时间列的名称
    :param format_date: 时间列的format样式
    :param split_time: 划分训练集和测试机的时间节点
    :param is_same_periods: 单个样本中的目标项和特征项是否为同时期
    :param model_fix:
    :param kwargs:
    :return:
    """
    global time_name
    time_name = time_col_name

    if isinstance(y_col_name, str):
        y_col_name = [y_col_name]

    df[time_col_name] = pd.to_datetime(df[time_col_name], format=format_date)  # 转换成时间对象
    # 训练测试集划分
    train_ds = df[df[time_col_name] < split_time]
    test_ds = df[df[time_col_name] >= split_time]
    # 标准化
    train_ds, test_ds = normalize_xy(train_ds,  test_ds, x_col_names, y_col_name, scaler_path=Config.scalers_path,
                                     model_fix=model_fix)
    # 特征项(x/features)和目标项(y/targets)划分
    if not is_same_periods:  # 基于过去的特征项预测未来的目标项
        train_x, train_y, train_ix = create_xy_future(train_ds, x_col_names, y_col_name, **kwargs)
        test_x, test_y, test_ix = create_xy_future(test_ds, x_col_names, y_col_name, **kwargs)
    else:  # 基于同时期的特征项预测同时期的目标项
        train_x, train_y, train_ix = create_xy_same(train_ds, x_col_names, y_col_name, **kwargs)
        test_x, test_y, test_ix = create_xy_same(test_ds, x_col_names, y_col_name, **kwargs)

    # 输出为HDF5文件
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('train_x', data=train_x, dtype=np.float32)  # float32是模型训练的默认精度
        f.create_dataset('train_y', data=train_y, dtype=np.float32)
        f.create_dataset('train_ix', data=train_ix)
        f.create_dataset('test_x', data=test_x, dtype=np.float32)
        f.create_dataset('test_y', data=test_y, dtype=np.float32)
        f.create_dataset('test_ix', data=test_ix)

    print(f'model{model_fix} 样本数据已经生成.')


def normalize_xy(train_ds, test_ds, x_names: list, y_name: list, scaler_path='', model_fix=None):
    """
    对训练集和测试集分别进行X和Y的标准化
    :param train_ds: 训练集
    :param test_ds: 测试集
    :param x_names: 特征项的名称
    :param y_name: 目标项的名称
    :param scaler_path: 归一化器的存储路径
    :param model_fix: 模型标识字符
    :return: 标准化后的训练集和测试集
    """
    # 标准化
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()  # 标准化器

    train_ds.loc[:, x_names] = x_scaler.fit_transform(train_ds.loc[:, x_names])
    test_ds.loc[:, x_names] = x_scaler.transform(test_ds.loc[:, x_names])  # 注意标准化不能独立对测试集进行, 标准化参数应来源于训练集

    train_ds.loc[:, y_name] = y_scaler.fit_transform(train_ds[y_name])
    test_ds.loc[:, y_name] = y_scaler.transform(test_ds[y_name])  # 标准化目标变量

    if os.path.exists(scaler_path):
        scalers = joblib.load(scaler_path)
        scalers.update({f'model_{model_fix}_x_scaler': x_scaler, f'model_{model_fix}_y_scaler': y_scaler})
        joblib.dump(scalers, scaler_path)

    return train_ds, test_ds


def view_info(df, col_names, min_value=-9999, max_value=9999):
    print('-' * 50)
    for col_name in col_names:
        df.loc[df[col_name] < min_value, col_name] = np.nan
        df.loc[df[col_name] > max_value, col_name] = np.nan
        invalid_sum = df[[col_name]].isna().any(axis=1).sum()
        print('{} -- 最大值: {};\t最小值: {};\t无效值数量: {}'.format(col_name, df[col_name].max(),
                                                                      df[col_name].min(), invalid_sum))
    print('总无效数量: {}'.format(df.isna().any(axis=1).sum()))
    print('-' * 50)


def fast_viewing(df, station_names, feature_names, out_path=None):
    fig, axs = plt.subplots(len(feature_names), len(station_names), figsize=(50, 50))
    for col_ix, station_name in enumerate(station_names):
        temp_df = df[df['站名'] == station_name]
        for row_ix, feature_name in enumerate(feature_names):
            ax = axs[row_ix, col_ix]
            ax.plot(temp_df['date'], temp_df[feature_name])
            ax.set_title('Times series line of {}-{}'.format(station_name, feature_name))
            ax.set_xlabel('Date')
            ax.set_ylabel(feature_name)
    if out_path is not None:
        fig.savefig(out_path, dpi=177)
    fig.show()
