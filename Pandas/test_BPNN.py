import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------- 1. 数据准备（贴合论文场景）--------------------------
# 模拟数据：替换为你的FY-3C MWHTS数据（15列亮温）+ TMPA 3B42降水数据（1列降雨率）
# 实际使用时：用pd.read_csv('你的数据.csv')加载，列名设为TB1-TB15（亮温）、RR（降雨率mm/hr）
np.random.seed(42)  # 固定随机种子，结果可复现
n_samples = 5000    # 模拟样本数，实际替换为你的真实样本数
# 模拟15个通道亮温（TB：200-300K，符合微波探测亮温范围）
TB = np.random.uniform(200, 300, (n_samples, 15))
# 模拟降雨率（RR：0-35mm/hr，贴合论文台风降水范围）
RR = 0.1*TB[:,0] - 0.05*TB[:,9] + 0.02*TB[:,10] + np.random.normal(0, 2, n_samples)
RR = np.clip(RR, 0, 35)  # 降雨率非负，截断到0-35mm/hr

# 构造数据框
data = pd.DataFrame(TB, columns=[f'TB{i+1}' for i in range(15)])
data['RR'] = RR

# 划分输入（X：15个亮温）和输出（y：降雨率）
X = data.iloc[:, 0:15].values  # 输入层：15个节点（论文MWHTS15通道）
y = data.iloc[:, 15].values    # 输出层：1个节点（降雨率RR）

# -------------------------- 2. 数据预处理（关键步骤）--------------------------
# （1）划分训练集（80%）和测试集（20%）：论文用2016.1-8训练，9月验证
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# （2）标准化：微波亮温量纲一致但数值范围不同，标准化提升网络训练效率
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # 测试集用训练集的均值/方差，避免数据泄露

# -------------------------- 3. 搭建BP神经网络（贴合论文）--------------------------
# 论文为3层BP：输入层(15) → 隐藏层(自选) → 输出层(1)
# 隐藏层节点数经验公式：2*输入节点+1 或 根号(输入+输出)，这里选32（可调）
model = Sequential(name='BP_NN_Typhoon_Precipitation')
# 输入层+隐藏层1：Dense为全连接层，ReLU为非线性激活（贴合论文非线性拟合）
model.add(Dense(32, activation='relu', input_dim=15, name='Hidden_Layer1'))
model.add(Dropout(0.1))  # 防止过拟合，随机丢弃10%的神经元
# 可选：增加隐藏层（复杂场景），论文单隐藏层即可满足
# model.add(Dense(16, activation='relu', name='Hidden_Layer2'))
# 输出层：线性激活（regression回归任务，降雨率为连续值）
model.add(Dense(1, activation='linear', name='Output_Layer'))

# 编译模型：优化器+损失函数+评估指标
model.compile(
    optimizer=Adam(learning_rate=0.001),  # 自适应学习率，收敛更快
    loss='mse',  # 均方误差：回归任务核心损失，对应论文RMSE
    metrics=['mae']  # 平均绝对误差，辅助评估
)

# 打印网络结构
model.summary()

# -------------------------- 4. 训练BP神经网络--------------------------
# 训练参数：epochs(训练轮数)、batch_size(批次大小)可根据数据量调
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),  # 验证集实时监控
    verbose=1  # 打印训练过程
)

# -------------------------- 5. 模型验证与反演结果评估（论文指标）--------------------------
# 对测试集进行反演
y_pred = model.predict(X_test).flatten()  # 展平为一维数组，方便计算

# 计算论文核心评估指标：相关系数Corr、偏差Bias、均方根误差RMSE
def calculate_metrics(y_true, y_pred):
    corr = np.corrcoef(y_true, y_pred)[0,1]  # 皮尔逊相关系数
    bias = np.mean(y_pred - y_true)          # 偏差
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))  # 均方根误差
    r2 = r2_score(y_true, y_pred)            # 决定系数（辅助）
    return corr, bias, rmse, r2

corr, bias, rmse, r2 = calculate_metrics(y_test, y_pred)
# 打印指标（贴合论文表3/表4格式）
print('-'*50)
print(f'BP神经网络反演结果评估指标：')
print(f'相关系数(Corr)：{corr:.2f}')
print(f'偏差(Bias, mm/hr)：{bias:.2f}')
print(f'均方根误差(RMSE, mm/hr)：{rmse:.2f}')
print(f'决定系数(R²)：{r2:.2f}')
print('-'*50)

# -------------------------- 6. 结果可视化（贴合论文图7/图8）--------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示

# 子图1：训练过程的损失变化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练集损失(MSE)')
plt.plot(history.history['val_loss'], label='测试集损失(MSE)')
plt.title('BP神经网络训练损失变化')
plt.xlabel('训练轮数(Epochs)')
plt.ylabel('均方误差(MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：反演值vs真实值散点图（论文图8格式）
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6, s=10)  # 散点图，透明化避免重叠
# 绘制y=x参考线
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2)
plt.title(f'BP神经网络反演值vs真实值\nCorr={corr:.2f}, Bias={bias:.2f}, RMSE={rmse:.2f}')
plt.xlabel('TMPA 3B42真实降雨率(mm/hr)')
plt.ylabel('BP反演降雨率(mm/hr)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------- 7. 新数据反演（实际应用）--------------------------
# 模拟新的MWHTS亮温数据（15个通道），实际替换为你的观测数据
new_TB = np.random.uniform(200, 300, (10, 15))  # 10个新样本
new_TB_scaled = scaler.transform(new_TB)        # 标准化
new_RR_pred = model.predict(new_TB_scaled).flatten()  # 反演降雨率
print('新样本亮温的台风降水反演结果(mm/hr)：', np.round(new_RR_pred, 2))
