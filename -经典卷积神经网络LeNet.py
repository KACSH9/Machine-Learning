# 1. 导入库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# 2. 数据集
# 2.1 数据预处理
transform = transforms.Compose([
    transforms.Pad(2),                          # 28 -> 32
    transforms.RandomRotation(10),     # 数据增强
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 官方均值/方差
])

# 2.2 数据下载
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)

# 2.3 数据加载
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          num_workers=0, pin_memory=False, persistent_workers=False)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False,
                          num_workers=0, pin_memory=False, persistent_workers=False)

# 3. LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=0)          # 32->28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)            # 28->14
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)         # 14->10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)            # 10->5
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)  # 5->1
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.act = nn.ReLU()  
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = self.act(self.conv3(x))          # [B,120,1,1]
        x = x.view(x.size(0), -1)            # [B,120]
        x = self.act(self.fc1(x))
        x = self.dropout(x)                  # 丢弃法
        x = self.fc2(x)                      # [B,10] (logits)
        return x

# 4. 损失函数
criterion = nn.CrossEntropyLoss()

# 5. 优化器
model = LeNet5()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)   # 正则化

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

