# @Author   : ChaoQiezi
# @Time     : 2024/4/25  19:21
# @FileName : utils.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放常用工具和函数
"""

import Config
# 初始化参数
from Config import split_time, seq_len_day, pred_len_day

import os.path
import h5py
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from typing import Union

mpl.rcParams['font.family'] = 'Microsoft YaHei'  # 设置字体(显示中文)
mpl.rcParams['axes.unicode_minus'] = True  # 显示负号
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
    ixs = np.array(ixs)

    # return train_x, train_y, train_ix
    return xs, ys, ixs


def create_xy_future(dataset, x_col_names: list, y_col_name: list, window_size=seq_len_day, step_size=1,
                     future_size=pred_len_day,
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
                cur_data.loc[y_start:y_end, :].apply(
                    lambda x: str(x[st_col_name]) + '_' + x[time_name].strftime(format_date), axis=1))
    xs = np.array(xs)
    ys = np.array(ys).squeeze(axis=-1)  # 去除最后一个形状大小为1的维度, 下同
    ixs = np.array(ixs)

    return xs, ys, ixs


def generate_samples(df, x_col_names: list, y_col_name: Union[str, list], out_path, time_col_name='date',
                     format_date=None,
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
    train_ds, test_ds = normalize_xy(train_ds, test_ds, x_col_names, y_col_name, scaler_path=Config.scalers_path,
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


def set_show_nan(df, col_names, min_value=-9999, max_value=9999):
    """
    设置无效值并打印无效值情况
    :param df:
    :param col_names:
    :param min_value:
    :param max_value:
    :return:
    """
    print('-' * 50)
    for col_name in col_names:
        df.loc[df[col_name] < min_value, col_name] = np.nan
        df.loc[df[col_name] > max_value, col_name] = np.nan
        invalid_sum = df[[col_name]].isna().any(axis=1).sum()
        print('{} -- 最大值: {};\t最小值: {};\t无效值数量: {}'.format(col_name, df[col_name].max(),
                                                                      df[col_name].min(), invalid_sum))
    print('总无效数量: {}'.format(df.isna().any(axis=1).sum()))
    print('-' * 50)


def show_samples_info(train_x_shape=None, train_y_shape=None, test_x_shape=None, test_y_shape=None, train_ix=None, test_ix=None):
    """
    打印样本数据集的基本情况
    :param train_shape:
    :param test_shape:
    :param train_ix_shape:
    :param test_ix_shape:
    :return:
    """

    output = []
    train_flag, test_flag = None, False

    if (train_x_shape is not None) and (train_y_shape is not None):
        output.append('-' * 50)
        # 训练集特征项
        train_size, seq_len, feature_size = train_x_shape
        output.append('当前训练集特征项Shape: {};'.format(train_x_shape))
        # 训练集目标项
        output.append('当前训练集目标项Shape: {};'.format(train_y_shape))
        pred_seq_len = train_y_shape[1]

        output.append('单个样本特征数: {}'.format(feature_size))
        output.append('预测期数: {} day'.format(pred_seq_len))

        train_flag = True

    if (test_x_shape is not None) and (test_y_shape is not None):
        test_size, _, _ = test_x_shape
        output.append('-' * 50)
        output.append('当前测试集特征项Shape: {};'.format(test_x_shape))
        output.append('当前测试集目标项Shape: {};'.format(test_y_shape))

        test_flag = True

    if train_flag and test_flag:
        output.append('训练集数目: {}; 测试集数目: {}; 比例: {:0.2f}:1'.format(train_size, test_size, train_size / test_size))

    if train_ix is not None:
        train_start_date, train_end_date = train_ix['0_date'].min(), train_ix[f'{Config.pred_len_day - 1}_date'].max()
        output.append('-' * 50)
        output.append('训练集的时间范围: {} ~ {}'.format(train_start_date, train_end_date))
    if test_ix is not None:
        test_start_date, test_end_date = test_ix['0_date'].min(), test_ix[f'{Config.pred_len_day - 1}_date'].max()
        output.append('-' * 50)
        output.append('测试集的时间范围: {} ~ {}'.format(test_start_date, test_end_date))

    output.append('-' * 50)
    for line in output:
        print(line)



def fast_viewing(df, station_names, feature_names, out_path=None):
    """
    快速查看各个站点各个特征的图
    :param df:
    :param station_names:
    :param feature_names:
    :param out_path:
    :return:
    """
    fig, axs = plt.subplots(len(feature_names), len(station_names), figsize=(50, 50))
    for col_ix, station_name in enumerate(station_names):
        temp_df = df[df['站名'] == station_name]
        for row_ix, feature_name in enumerate(feature_names):
            ax = axs[row_ix, col_ix]
            sns.lineplot(x=temp_df['date'], y=temp_df[feature_name], ax=ax)
            # ax.plot(temp_df['date'], temp_df[feature_name])
            ax.set_title('Times series line of {}-{}'.format(station_name, feature_name))
            ax.set_xlabel('Date')
            ax.set_ylabel(feature_name)
    if out_path is not None:
        fig.savefig(out_path, dpi=177)
    fig.show()


def plot_comparison(x, y_obs, y_pred, station_name, save_path=None):
    """
    绘制预测结果和真实结果的折线图
    :param x:
    :param y_obs:
    :param y_pred:
    :param station_name:
    :param save_path:
    :return:
    """
    fig, axs = plt.subplots(3, 1, figsize=(19, 24))
    ax_upper, ax_middle, ax_under = axs[0], axs[1], axs[2]

    # 上部子图: 真实降雨
    sns.lineplot(x=x, y=y_obs, ax=ax_upper, linewidth=3, color='#75813C', label='Real Precipitation')
    ax_upper.set_xlabel('Date', fontsize=26)
    ax_upper.set_title('The real precipitation of {}'.format(station_name), fontsize=30)
    ax_upper.set_ylabel('Real precipitation (mm)', fontsize=26)
    ax_upper.tick_params(axis='both', labelsize=18)
    ax_upper.legend(fontsize=26, loc='upper right')

    # 中部子图: 预测降雨
    sns.lineplot(x=x, y=y_pred, ax=ax_middle, linewidth=3, color='#1E0785', label='Predicted Precipitation')
    ax_middle.set_xlabel('Date', fontsize=26)
    ax_middle.set_title('The predicted precipitation of {}'.format(station_name), fontsize=30)
    ax_middle.set_ylabel('Predicted precipitation (mm)', fontsize=26)
    ax_middle.tick_params(axis='both', labelsize=18)
    ax_middle.legend(fontsize=26, loc='upper right')

    # 底部子图: 真实和预测降雨
    sns.lineplot(x=x, y=y_obs, ax=ax_under, linewidth=3, color='#75813C', label='Real Precipitation')
    sns.lineplot(x=x, y=y_pred, ax=ax_under, linewidth=3, color='#1E0785', label='Predicted Precipitation')
    ax_under.set_xlabel('Date', fontsize=26)
    ax_under.set_title('The predicted and real precipitation of {}'.format(station_name), fontsize=30)
    ax_under.set_ylabel('Predicted and real precipitation (mm)', fontsize=26)
    ax_under.tick_params(axis='both', labelsize=18)
    ax_under.legend(fontsize=26, loc='upper right')

    # 输出
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    fig.show()


def plot_loss(loss_epochs):
    """
    绘制迭代损失
    :param loss_epochs:
    :return:
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    plt.plot(loss_epochs)
    ax.set_xlabel('Epoch 次数', fontsize=24)
    ax.set_ylabel('MSE Loss', fontsize=24)
    ax.set_title('LSTM training loss diagram', fontsize=30)
    ax.legend(['MSE Loss'], fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(linestyle='--', alpha=0.6)
    plt.show()


def cal_nse(y_obs, y_pred):
    """
    计算纳什效率系数NSE
    :param y_obs:
    :param y_pred:
    :return:
    """

    # 转换为np.numpy数组
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    # 计算观测值平均值
    y_obs_mean = np.mean(y_obs)

    # 计算分子分母
    numerator = np.sum(np.power(y_obs - y_pred, 2))
    denominator = np.sum(np.power(y_obs - y_obs_mean, 2))

    # 计算NSE
    nse = 1 - numerator / denominator

    return nse


def decode_time_col(time_col, sep='_'):
    """
    用于对字符串型时间列进行解码和分割
    :param time_col: 待处理的时间数组
    :param sep: 分割符,默认_(下划线)
    :return: 返回处理好的时间数据集(pd.Dataframe)
    """
    # 解码
    time_col = np.vectorize(lambda x: x.decode('utf-8'))(time_col)
    time_col = pd.DataFrame(time_col)
    # 分割
    col_names = time_col.columns
    for col_name in col_names:
        cur_date_name = '{}_date'.format(col_name)
        time_col[['站名', cur_date_name]] = time_col[col_name].str.split(sep, expand=True)
        time_col[cur_date_name] = pd.to_datetime(time_col[cur_date_name], format='%Y%m%d%H')
        del time_col[col_name]
    return time_col
