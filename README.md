# 径流模拟

本项目基于2010年~2015年气温, 气压, 相对湿度, 风速, 日照, 地温, 降水量特征因子的去
模拟径流数据(包括巴河桥, 更张, 工布江达三个站点)。

# 2024/6/1说明

# 项目结构

 - Assets  `目录: 存储运行时产生的中间结果和图表`
     - Charts  `目录: 存储图表文件`  
       - ···  `文件: 图表`
     - scalers.pkl  `文件: 标准化参数存储器`
 - Scripts  `目录: 脚本文件`
   - Download  `目录: 下载文件相关脚本文件`
     - era5temperature.py  `代码: 下载ERA5再分析资料脚本文件`
   - era5_preprocessing.py  `代码: 预处理ERA5(.nc)的脚本文件`
   - extract_rel_station.py  `代码: 提取和整合气象数据(站点不匹配搁置)`
   - time_series_processing.py  `代码: 时间序列数据集预处理`
   - generate_samples.py  `代码: 生成样本数据集`
   - train.py  `代码: 模型训练`
   - eval.py  `代码: 模型评估`
 - utils  `目录: 存储函数和类`
   - model.py  `代码: 模型相关函数和类`
   - utils.py  `代码: 常用函数和类`
 - Config.py  `配置文件`
 - README.MD  `项目说明文档`

# 后期可能增加
1. 增加多步预测
2. 未来或许换训练集、验证集、测试集(遥遥无期)
3. 完善整个流程, 方便记忆期和预见期调整后, 快速进行模型整体运行


 
