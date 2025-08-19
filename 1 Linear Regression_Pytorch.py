# 1. 导入库
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 2. 读取加载数据
def make_dataloader(tensors, batch_size, shuffle=True):
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features = torch.normal(mean=0.0, std=1.0, size=(1000, 2))
targets = features @ true_w + true_b
targets += torch.normal(mean=0.0, std=0.01, size=targets.shape)
targets = targets.reshape(-1, 1)

train_loader = make_dataloader((features, targets), batch_size=10, shuffle=True)

# 3. 定义模型
model = nn.Sequential(nn.Linear(in_features=2, out_features=1))

# 4. 损失函数
loss_fn = nn.MSELoss()

# 5. 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

# 6. 参数初始化
with torch.no_grad():
    model[0].weight.normal_(mean=0.0, std=0.01)
    model[0].bias.zero_()

# 7. 训练循环
EPOCHS = 3
for epoch in range(1, EPOCHS + 1):
    for X_batch, y_batch in train_loader:
        preds = model(X_batch)                 # 前向传播
        loss = loss_fn(preds, y_batch)         # 计算损失

        optimizer.zero_grad()                  # 梯度清零
        loss.backward()                        # 反向传播
        optimizer.step()                       # 参数更新

    with torch.no_grad():
        total_loss = loss_fn(model(features), targets).item()
    print(f"Epoch {epoch}, Loss: {total_loss:.6f}")
