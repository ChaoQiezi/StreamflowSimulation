# @Author   : ChaoQiezi
# @Time     : 2024/5/18  17:17
# @FileName : era5_preprocessing.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 将张琪处理ERA5转csv文件进行处理,方便后续时间序列样本的生成
"""

import os
import pandas as pd

# 准备
in_dir = r'H:\Datasets\Objects\StreamflowSimulation\Data\ERA5'
out_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\ERA5\Era5_2010_2015.xlsx'
station_names = ['巴河桥', '更张', '工布江达']

dfs = []
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    for station_name in station_names:
        csv_path = os.path.join(in_dir, station_name + '.csv')
        df = pd.read_csv(csv_path, encoding='ANSI')
        df.to_excel(writer, sheet_name=station_name, index=False)
        print('处理: {}'.format(station_name))
print('程序结束.')
