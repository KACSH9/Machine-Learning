# 1 导入库
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

def main():
    # 2 数据加载
    train_iter, test_iter = d2l.load_data_fashion_mnist(256)  

    # 3 参数初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)

    # 4 神经网络模型
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5)   # 丢弃法
        nn.Linear(256, 10)
    )
    net.apply(init_weights)

    # 5 损失函数
    criterion = nn.CrossEntropyLoss()

    # 6 优化器
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 7 训练模型
    EPOCHS = 10
    train_losses, test_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        net.train()
        running_loss, count = 0.0, 0
        for X, y in train_iter:
            logits = net(X)
            l = criterion(logits, y)
            trainer.zero_grad(set_to_none=True)
            l.backward()
            trainer.step()
            bsz = y.size(0)
            running_loss += l.item() * bsz
            count += bsz
        train_loss = running_loss / count

        # 8 评估模型
        net.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for X, y in test_iter:
                pred = net(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        test_acc = correct / total

        train_losses.append(train_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch}, TrainLoss: {train_loss:.6f}, TestAcc: {test_acc:.4f}")

    # 9 画图
    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label="Train Loss")
    plt.plot(epochs, test_accs, marker='s', label="Test Accuracy")
    plt.title("Training Loss & Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
