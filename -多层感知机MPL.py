# 1. 导入库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# 2. 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 官方均值/方差
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          num_workers=0, pin_memory=False, persistent_workers=False)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False,
                          num_workers=0, pin_memory=False, persistent_workers=False)

# 3. MLP模型
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                 # [B,1,28,28] -> [B,784]
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 4. 损失函数
criterion = nn.CrossEntropyLoss()

# 5. 优化器
model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 6. 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计
        loss_sum += loss.item() * x.size(0)   # ✅ 用 size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)                    # ✅ 记得累计 total
    return loss_sum / total, correct / total

# 7. 评估模型
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

# 8. GPU等配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 9. 训练
def main():
    epochs = 5
    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1:02d} | "
              f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
              f"test_loss={te_loss:.4f}, test_acc={te_acc:.4f}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
