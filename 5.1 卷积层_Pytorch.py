import torch
import torch.nn as nn

# 通用写法：定义一个二维卷积层
conv = nn.Conv2d(
    in_channels=3,   # 输入通道数 (例如 RGB 图像就是 3)
    out_channels=16, # 卷积核（滤波器）个数，也就是输出通道数
    kernel_size=3,   # 卷积核大小，可以是 int 或 (k_h, k_w)
    stride=1,        # 步幅，默认 1
    padding=1,       # 填充，默认 0；常用 same padding 就是 kernel_size // 2
    dilation=1,      # 膨胀卷积参数，默认 1（不扩张）
    groups=1,        # 分组卷积，默认 1；depthwise 卷积时 groups=in_channels
    bias=True        # 是否带偏置项，默认 True
)

# 输入：batch_size=1，3通道，高宽都是32
x = torch.randn(1, 3, 32, 32)

# 前向传播
y = conv(x)
print(y.shape)  # [1, 16, 32, 32]  (因为 stride=1, padding=1, kernel=3，尺寸不变)
