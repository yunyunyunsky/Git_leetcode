import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pylab import mpl

# 设置matplotlib中文字体，以便图表标题和标签能显示中文
mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体为黑体
mpl.rcParams["axes.unicode_minus"] = False    # 解决保存图像是负号'-'显示为方块的问题

# 检查是否有可用的CUDA设备，若有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用设备：", device)

# ===================== 1. 构建模拟数据集（使用更复杂的非线性关系） =====================
# 设置随机种子以确保结果可复现
np.random.seed(42)
n_samples = 1000  # 样本数量
n_channels = 15   # 模拟的微波亮温通道数

# 模拟MWHTS亮温特征（X），数值范围在180K到300K之间
X_raw = np.random.uniform(low=180, high=300, size=(n_samples, n_channels))

# 【核心修改】引入更复杂的非线性关系来生成降水量标签
bright_temp_mean = X_raw.mean(axis=1)  # 计算每个样本的亮温均值

# 方案四：复合非线性函数
# 结合多项式和指数特性，模拟更真实的非线性亮温-降水关系
# 这种关系比简单的线性关系更复杂，更接近真实物理过程
coefficients = [0.0005, -0.2, 25] # [a, b, c] for ax^2 + bx + c part
poly_part = coefficients[0] * (bright_temp_mean**2) + coefficients[1] * bright_temp_mean + coefficients[2]
exp_coeff = 0.005
base_precip = 40
y_raw_nonlinear = base_precip * np.exp(-exp_coeff * (bright_temp_mean - 180)) + poly_part
# 为了保证生成的降水范围在 0-50 mm/hr，进行裁剪
y_raw_nonlinear = np.clip(y_raw_nonlinear, 0, 50)

# 用新生成的非线性 y_raw 替换原来的 y_raw
y_raw = y_raw_nonlinear

print(f"生成的降水数据范围: {y_raw.min():.2f} ~ {y_raw.max():.2f} mm/hr")

# ===================== 2. 数据预处理 =====================
# 使用StandardScaler对亮温数据进行标准化（Z-score标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 将数据重塑为CNN所需的格式：(样本数, 通道数, 高度, 宽度)
X = X_scaled.reshape(-1, n_channels, 1, 1)  # shape=(1000, 15, 1, 1)
y = y_raw.reshape(-1, 1)                   # 标签保持一维

# 按比例划分训练集(70%)、验证集(15%)和测试集(15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 将numpy数组转换为PyTorch张量，并指定数据类型
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("训练集特征形状（X）：", X_train_tensor.shape)
print("训练集基准标签形状（y）：", y_train_tensor.shape)

# ===================== 定义CNN模型 =====================
class MWHTS_Precip_CNN(nn.Module):
    """
    用于从微波亮温反演降水量的CNN模型
    """
    def __init__(self, in_channels=15):
        super(MWHTS_Precip_CNN, self).__init__()
        # 第一个卷积层：将输入的1个“通道”（通过permute调整后的结构）转换为32个特征图
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化层，加速训练并提高稳定性
        self.relu = nn.ReLU()          # 激活函数，引入非线性

        # 第二个卷积层：将32个特征图转换为64个特征图
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(64)

        # 展平层：将多维特征图展平为一维向量，供全连接层使用
        self.flatten = nn.Flatten()

        # 全连接层（隐藏层）：接收展平后的特征，输出128维特征
        # 输入维度：64个卷积核 * 15个原始通道 * 1 * 1
        self.fc1 = nn.Linear(64 * in_channels * 1 * 1, 128)
        self.dropout = nn.Dropout(0.3) # Dropout层，防止过拟合
        self.fc2 = nn.Linear(128, 1)   # 输出层：预测1个值（降水量）

    def forward(self, x):
        """
        定义模型的前向传播过程
        """
        # 调整输入张量的维度顺序，使其符合Conv2d的要求 (N, C_in, H, W)
        # 原始输入是 (N, 15, 1, 1)，需要变成 (N, 1, 15, 1)
        x = x.permute(0, 3, 1, 2)  # 维度重排

        # 通过第一个卷积块：卷积 -> 批归一化 -> ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        # 通过第二个卷积块：卷积 -> 批归一化 -> ReLU
        x = self.relu(self.bn2(self.conv2(x)))

        # 将卷积层输出的多维特征展平
        x = self.flatten(x)
        # 通过第一个全连接层
        x = self.relu(self.fc1(x))
        # 应用Dropout
        x = self.dropout(x)
        # 通过第二个全连接层（输出层）得到最终预测值
        output = self.fc2(x)
        return output

# 初始化模型实例，并将其移动到指定设备（CPU或GPU）
model = MWHTS_Precip_CNN(in_channels=15).to(device)
print("CNN模型结构：\n", model)

# 将所有数据张量移动到指定设备（CPU或GPU）以进行计算
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# ===================== 模型训练 =====================
# 定义损失函数：均方误差（MSE），适用于回归任务
criterion = nn.MSELoss()
# 定义优化器：Adam，一种自适应学习率的优化算法
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 设置训练超参数
epochs = 100    # 训练轮数
batch_size = 32 # 批处理大小
train_loss_list = []  # 记录训练损失
val_loss_list = []    # 记录验证损失

# 开始训练循环
model.train()  # 设置模型为训练模式
for epoch in range(epochs):
    # 分批处理训练数据，避免一次性加载全部数据导致内存不足
    for i in range(0, len(X_train_tensor), batch_size):
        # 获取当前批次的输入特征和标签
        batch_X = X_train_tensor[i:i + batch_size]
        batch_y = y_train_tensor[i:i + batch_size]

        # 1. 前向传播：模型预测
        pred_y = model(batch_X)
        # 2. 计算损失：预测值与真实值的差异
        loss = criterion(pred_y, batch_y)

        # 3. 反向传播和参数更新
        optimizer.zero_grad()  # 清空上一轮迭代的梯度
        loss.backward()        # 计算当前损失相对于模型参数的梯度
        optimizer.step()       # 根据梯度更新模型参数

    # 每个epoch结束后，在验证集上评估模型性能
    model.eval()  # 设置模型为评估模式（关闭dropout等）
    with torch.no_grad():  # 禁用梯度计算，节省内存
        val_pred = model(X_val_tensor)  # 验证集预测
        val_loss = criterion(val_pred, y_val_tensor)  # 计算验证损失

    # 记录当前epoch的训练和验证损失
    train_loss_list.append(loss.item())
    val_loss_list.append(val_loss.item())

    # 每10个epoch打印一次训练进度
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], 训练损失：{loss.item():.4f}, 验证损失：{val_loss.item():.4f}")

print("模型训练完成！")

# ===================== 降水反演 =====================
model.eval()  # 将模型切换到评估/推理模式
with torch.no_grad():  # 推理时不需要计算梯度
    # 使用测试集进行最终的降水反演预测
    test_pred = model(X_test_tensor)
    # 计算测试集上的总损失（MSE）
    test_loss = criterion(test_pred, y_test_tensor)

# 将预测结果和真实标签从GPU/CPU移回CPU，并转换为numpy数组，以便后续绘图和计算
test_true_np = y_test_tensor.cpu().numpy().flatten()
test_pred_np = test_pred.cpu().numpy().flatten()

# 计算RMSE（均方根误差）
rmse_value = torch.sqrt(test_loss).item()
print("\n===== 降水反演结果示例 =====")
print("测试集平均反演误差（RMSE）：", rmse_value)

# 计算R²（决定系数）
from sklearn.metrics import r2_score
r2_value = r2_score(test_true_np, test_pred_np)
print(f"测试集决定系数 R²: {r2_value:.4f}")

# ===================== 绘图展示 =====================
# 创建一个包含三个子图的画布
plt.figure(figsize=(15, 5))

# 子图1：训练损失曲线
plt.subplot(1, 3, 1)
plt.plot(train_loss_list, label='训练损失', color='blue')     # 绘制训练损失曲线
plt.plot(val_loss_list, label='验证损失', color='red')       # 绘制验证损失曲线
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('模型训练过程损失变化')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 子图2：测试集反演效果散点图
plt.subplot(1, 3, 2)
plt.scatter(test_true_np, test_pred_np, alpha=0.6, s=30, edgecolors='w', linewidth=0.5)
# 绘制理想预测线 (y=x)
min_val = min(min(test_true_np), min(test_pred_np))
max_val = max(max(test_true_np), max(test_pred_np))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线 (y=x)')
plt.xlabel('真实降水量 (TMPA基准, mm/hr)')
plt.ylabel('CNN反演降水量 (mm/hr)')
plt.title(f'测试集反演效果散点图\nRMSE: {rmse_value:.3f}, R²: {r2_value:.3f}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 子图3：原始数据的非线性关系
plt.subplot(1, 3, 3)
# 绘制所有样本的亮温均值与真实降水量的关系
plt.scatter(bright_temp_mean, y_raw, alpha=0.6, s=30, edgecolors='w', linewidth=0.5, c='green', label='原始非线性关系')
plt.xlabel('亮温均值 (K)')
plt.ylabel('真实降水量 (mm/hr)')
plt.title('原始数据的非线性关系')
plt.grid(True, linestyle='--', alpha=0.6)
# 通常亮温越低降水越强，反转x轴使趋势更直观
plt.gca().invert_xaxis()
plt.legend()

plt.tight_layout()  # 自动调整子图间距
plt.show()

# 绘制残差图
plt.figure(figsize=(8, 6))
# 计算残差（真实值 - 预测值）
residuals = test_true_np - test_pred_np
# 以预测值为x轴，残差为y轴绘制散点图
plt.scatter(test_pred_np, residuals, alpha=0.6, s=30, edgecolors='w', linewidth=0.5)
# 绘制零误差参考线
plt.axhline(y=0, color='r', linestyle='--', label='零误差线')
plt.xlabel('CNN反演降水量 (mm/hr)')
plt.ylabel('真实值 - 预测值 (残差, mm/hr)')
plt.title('残差分布图')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(f"最终评估指标：RMSE: {rmse_value:.4f}, R²: {r2_value:.4f}")
