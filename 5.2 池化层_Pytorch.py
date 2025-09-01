import torch
import torch.nn as nn

# ==============================
# 1. 最大池化 (Max Pooling)
# ==============================
maxpool = nn.MaxPool2d(
    kernel_size=2,  # 池化窗口大小 (例如 2x2)
    stride=2,       # 步幅 (默认为 kernel_size)
    padding=0,      # 边缘填充，默认不填充
    dilation=1,     # 膨胀池化 (很少用)
    return_indices=False,  # 是否返回最大值的索引 (用于反池化)
    ceil_mode=False        # 是否向上取整计算输出大小
)

# ==============================
# 2. 平均池化 (Average Pooling)
# ==============================
avgpool = nn.AvgPool2d(
    kernel_size=2,  # 池化窗口大小
    stride=2,       # 步幅
    padding=0       # 边缘填充
)

# ==============================
# 3. 输入张量
# ==============================
x = torch.randn(1, 3, 32, 32)  # batch=1, 3通道, 32x32图像

# ==============================
# 4. 前向传播
# ==============================
y_max = maxpool(x)
y_avg = avgpool(x)

print("输入 shape:", x.shape)        # [1, 3, 32, 32]
print("最大池化输出 shape:", y_max.shape)  # [1, 3, 16, 16]  (下采样一半)
print("平均池化输出 shape:", y_avg.shape)  # [1, 3, 16, 16]
