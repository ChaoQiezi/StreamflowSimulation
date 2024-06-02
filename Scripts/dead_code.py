# @Author   : ChaoQiezi
# @Time     : 2024/4/22  21:01
# @FileName : dead_code.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放临时无效代码
"""

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 生成数据
np.random.seed(0)
time_steps = np.linspace(0, 8 * np.pi, 800)
data = np.sin(time_steps) + np.random.normal(size=len(time_steps)) * 0.5

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# 创建数据窗口
def create_dataset(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in)])
        y.append(data[(i + n_steps_in):(i + n_steps_in + n_steps_out)])
    return np.array(X), np.array(y)

n_steps_in, n_steps_out = 7, 2
X, y = create_dataset(data, n_steps_in, n_steps_out)

# 划分训练和测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 构建模型
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps_in, 1)),
    Dense(n_steps_out)
])

model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=30, verbose=1)

# 进行预测
yhat = model.predict(X_test[:1])
print("Predicted values:", yhat)
print("True values:", y_test[:1])



import matplotlib.pyplot as plt

# 进行预测
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
import numpy as np

# 假设y_pred是模型的预测输出，形状为(300, 2)
# 假设y_test是测试数据的实际输出，形状为(300, 2)

# 首先，我们需要从每个样本的预测中提取连续的时间序列
continuous_pred = np.zeros((y_pred.shape[0] + y_pred.shape[1] - 1, ))
continuous_true = np.zeros((y_test.shape[0] + y_test.shape[1] - 1, ))

# 填充预测和真实值的连续时间序列
for i in range(y_pred.shape[0]):
    continuous_pred[i:i + y_pred.shape[1]] += y_pred[i]
    continuous_true[i:i + y_test.shape[1]] += y_test[i]

# 简化版：只将重叠的预测平均处理
overlap_count = np.full((len(continuous_pred),), 1)  # 计算重叠次数
for i in range(1, y_pred.shape[1]):
    overlap_count[i:-i] += 1

continuous_pred /= overlap_count
continuous_true /= overlap_count

# 绘制连续的时间序列预测图
plt.figure(figsize=(15, 6))
plt.plot(continuous_true, label='True Values', marker='o')
plt.plot(continuous_pred, label='Predictions', marker='x', linestyle='--')
plt.title('Continuous Time Series Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Predicted/True Value')
plt.legend()
plt.show()

