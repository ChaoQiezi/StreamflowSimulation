# @Author   : ChaoQiezi
# @Time     : 2024/6/3  16:25
# @FileName : postprocessing.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 对main.py产生的诸多文件进行整合
"""

import os
from tqdm import tqdm
from glob import glob
import pandas as pd

def mult_excel2sheet(in_dir, prefix, out_path, pbar):
    excel_paths = glob(os.path.join(in_dir, '{}*.xlsx'.format(prefix)))
    with pd.ExcelWriter(out_path, mode='w', engine='openpyxl') as writer:
        for excel_path in excel_paths:
            file_name = os.path.basename(excel_path).split('.xlsx')[0].split(prefix + '_')[1]
            pd.read_excel(excel_path, index_col=0).to_excel(writer, sheet_name=file_name)
            pbar.set_postfix_str(file_name)

# 准备
in_dir = r'H:\Datasets\Objects\StreamflowSimulation\Data\Result'
out_dir = r'H:\Datasets\Objects\StreamflowSimulation\Data\Result\postprocessing'
if not os.path.exists(out_dir): os.makedirs(out_dir)

pbar = tqdm(['eval_indicators', 'pred_obs'])
for prefix in pbar:
    pbar.set_description(prefix)
    out_path = os.path.join(out_dir, 'all_{}.xlsx'.format(prefix))
    mult_excel2sheet(in_dir, prefix, out_path, pbar)
print('文件整合完毕.')
