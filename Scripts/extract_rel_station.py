# @Author   : ChaoQiezi
# @Time     : 2024/4/22  21:55
# @FileName : extract_rel_station.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 基于站点ID提取相关因子数据
"""

import os.path
from glob import glob
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta  # 方便时间的计算
import pandas as pd

# 准备
start_t = datetime(2010, 1, 1)
end_t = datetime(2015, 12, 31)
station_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\模拟资料\尼洋河数据\Station.xlsx'
rainfall_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\模拟资料\尼洋河数据\Rainfall.xlsx'
streamflow_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\模拟资料\尼洋河数据\Streamflow.xlsx'
factors_dir = r'H:\Datasets\Objects\StreamflowSimulation\Data\全国日气象数据1960-2016'
log_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\match_station_log.txt'
factor_names = {
    'PRS': ['ID', 'lat', 'lon', 'dem', 'year', 'month', 'day', 'aver_prs', 'max_prs', 'min_prs', 'aver_q', 'max_q',
            'min_q'],
    'TEM': ['ID', 'lat', 'lon', 'dem', 'year', 'month', 'day', 'aver_tem', 'max_tem', 'min_tem', 'aver_q', 'max_q',
            'min_q'],
    'RHU': ['ID', 'lat', 'lon', 'dem', 'year', 'month', 'day', 'aver_rhu', 'min_rhu', 'aver_q', 'min_q'],
    'PRE': ['ID', 'lat', 'lon', 'dem', 'year', 'month', 'day', '20-8_pre', '8-20_pre', '20-20_pre', '20-8_q', '8-20_q',
            '20-20_q'],
    'EVP': ['ID', 'lat', 'lon', 'dem', 'year', 'month', 'day', 'small_evp', 'large_evp', 'small_q', 'large_q'],
    'WIN': ['ID', 'lat', 'lon', 'dem', 'year', 'month', 'day', 'aver_w_spd', 'max_w_spd', 'max_w_dir', 'gust_w_spd',
            'gust_w_dir', 'aver_w_spd_q', 'max_w_spd_q', 'max_w_dir_q', 'gust_w_spd_q', 'gust_w_dir_q'],
    'SSD': ['ID', 'lat', 'lon', 'dem', 'year', 'month', 'day', 'sun_hours', 'sun_q'],
    'GST': ['ID', 'lat', 'lon', 'dem', 'year', 'month', 'day', 'aver_gst', 'max_gst', 'min_gst', 'aver_q', 'max_q',
            'min_q']
}
cols_required = [
    {'aver_prs': 0.1, 'max_prs': 0.1, 'min_prs': 0.1},  # hPa
    {'aver_tem': 0.1, 'max_tem': 0.1, 'min_tem': 0.1},  # ℃
    {'aver_rhu': 0.01, 'min_rhu': 0.01},  # 1
    {'20-8_pre': 0.1, '8-20_pre': 0.1, '20-20_pre': 0.1},  # mm
    {'small_evp': 0.1, 'large_evp': 0.1},  # mm
    {'aver_w_spd': 0.1, 'max_w_spd': 0.1, 'max_w_dir': None, 'gust_w_spd': 0.1, 'gust_w_dir': None},  # m/s, 方位
    {'sun_hours': 0.1},  # hours
    {'aver_gst': 0.1, 'max_gst': 0.1, 'min_gst': 0.1}  # ℃
]
total_months = (end_t.year - start_t.year) * 12 + (end_t.month - start_t.month) + 1
log_content = []

# 读取数据集
station = pd.read_excel(station_path)
station = station[station['ID'] == 56312]  # 暂时仅有该站点的处理需求
# 迭代月份
alls = []  # 存储所有提取的数据
for month in range(total_months):
    # 迭代当前循环月的因子
    factors = []  # 存储当前月份的提取数据
    col_names_required = ['ID', 'lat', 'lon', 'dem', 'date']
    cur_t = start_t + relativedelta(months=month)
    cur_month_days = calendar.monthrange(cur_t.year, cur_t.month)[1]  # monthrange返回(month, days)
    cur_date = pd.DataFrame(pd.date_range(cur_t, periods=cur_month_days, freq='D').strftime('%Y%m%d'), columns=['date'])

    for (factor_name, col_names), cols in zip(factor_names.items(), cols_required):
        # print(factor_name, col_names, cols)
        # print('-' * 50)
        # 匹配信息
        match_file_name = 'SURF_CLI_CHN_MUL_DAY*{}*{}{:02}.TXT'.format(factor_name, cur_t.year, cur_t.month)
        match_path = os.path.join(factors_dir, '**', match_file_name)
        matched_path = glob(match_path, recursive=True)
        if len(matched_path) != 1:
            log_content.append(match_file_name + ' ==> {:02}'.format(len(matched_path)))
            continue
        matched_path = matched_path[0]

        # 匹配
        factor = pd.read_csv(matched_path, sep='\s+', header=None, names=col_names)
        # factor = factor[factor['ID'].isin(station['ID'])]
        factor['date'] = factor.apply(lambda x: '{}{:02}{:02}'.format(x.year, x.month, x.day), axis=1)
        # 循环匹配每一个站点
        targets = []
        for station_id in station.ID:
            cur_factor = factor[factor['ID'] == station_id]
            targets.append(pd.merge(cur_date, cur_factor, how='left', on='date'))
        targets = pd.concat(targets, axis=0)
        # 选择指定列和列单位换算
        if not factors:  # 空列表
            col_names_required.extend(cols.keys())
            targets = targets[col_names_required]
        else:
            targets = targets[cols.keys()]
        for col_name, col_scale in cols.items():
            if not col_scale:
                continue
            targets[col_name] *= col_scale
        factors.append(targets)

    # 拼接
    factors = pd.concat(factors, axis=1)
    alls.append(factors)

    print('完成: {}年{:02}月-数据集提取'.format(cur_t.year, cur_t.month))
# 整理和单位换算
alls = pd.concat(alls, axis=0)
alls['dem'] *= 0.1  # convert to m
alls['lat'] = alls['lat'].astype(str)
alls['lon'] = alls['lon'].astype(str)
alls['lat'] = alls.apply(lambda x: float(x['lat'][:2]) + float(x['lat'][2:]) / 60, axis=1)  # 十进制
alls['lon'] = alls.apply(lambda x: float(x['lon'][:-2]) + float(x['lon'][-2:]) / 60, axis=1)
# 匹配径流(streamflow)和降水(rainfall)
rainfall = pd.read_excel(rainfall_path, )
# 写入
with open(log_path, 'w') as f:
    if not (log_content == []):
        f.writelines(log_content)
    else:
        f.write("No errors")
