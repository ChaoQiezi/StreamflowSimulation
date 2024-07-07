# @Author   : ChaoQiezi
# @Time     : 2024/4/22  21:01
# @FileName : dead_code.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放临时无效代码
"""

import os.path
import Config
import pandas as pd
import numpy as np

# 打乱数据集后分享
era5_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\ERA5\Era5_2010_2015.xlsx'
rainfall_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\模拟资料\尼洋河数据\Rainfall.xlsx'
streamflow_path = r'H:\Datasets\Objects\StreamflowSimulation\Data\模拟资料\尼洋河数据\Streamflow.xlsx'
out_dir = Config.Assets_dir
st_names = ['巴河桥', '更张', '工布江达']
feature_names = ['气温', '降水', '气压', '相对湿度', '风速', '风向', '日照', '地温', '蒸发']

for process_path in [era5_path, rainfall_path, streamflow_path]:
    write_path = os.path.join(out_dir, 'random_' + os.path.basename(process_path))
    # if not os.path.exists(write_path):
    with pd.ExcelWriter(write_path, mode='w') as writer:
        pd.DataFrame().to_excel(writer)
# 读取
for st_name in st_names:
    with pd.ExcelWriter(os.path.join(out_dir, 'random_' + os.path.basename(era5_path)), mode='a') as writer:
        era5_df = pd.read_excel(era5_path)
        process_columns = era5_df.loc[:, '气温':].columns
        era5_df[process_columns] = era5_df[process_columns].sample(frac=1).reset_index(drop=True)
        era5_df.to_excel(writer, sheet_name=st_name)
    with pd.ExcelWriter(os.path.join(out_dir, 'random_' + os.path.basename(rainfall_path)), mode='a') as writer:
        rainfall_df = pd.read_excel(rainfall_path)
        process_columns = rainfall_df.loc[:, '降水量':].columns
        rainfall_df[process_columns] = rainfall_df[process_columns].sample(frac=1).reset_index(drop=True)
        rainfall_df.to_excel(writer, sheet_name=st_name)
    with pd.ExcelWriter(os.path.join(out_dir, 'random_' + os.path.basename(streamflow_path)), mode='a') as writer:
         streamflow_df = pd.read_excel(streamflow_path)
         process_columns = streamflow_df.loc[:, '平均流量':].columns
         streamflow_df[process_columns] = streamflow_df[process_columns].sample(frac=1).reset_index(drop=True)
         streamflow_df.to_excel(writer, sheet_name=st_name)
