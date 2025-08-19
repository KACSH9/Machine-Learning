# 1. 导入库
import torch
from torch import nn
from d2l import torch as d2l

def main():
    # 2. 加载数据集
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 3. 定义模型
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10)
    )

    with torch.no_grad():   # 初始化参数
        net[1].weight.normal_(mean=0.0, std=0.01)
        net[1].bias.zero_()

    # 4. 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 5. 优化器
    optim = torch.optim.SGD(net.parameters(), lr=0.1)

    # 6. 训练循环
    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        net.train()   # 切换到训练模式
        running_loss, count = 0.0, 0

        for X, y in train_iter:
            logits = net(X)
            loss = loss_fn(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bsz = y.size(0)
            running_loss += loss.item() * bsz
            count += bsz

        train_loss = running_loss / count

        # 7. 评估
        net.eval()   # 评估模式
        with torch.no_grad():
            correct, total = 0, 0
            for X, y in test_iter:
                pred = net(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
            test_acc = correct / total

        # 8. 打印该轮结果（训练损失 + 测试准确率）
        print(f"Epoch {epoch}, TrainLoss: {train_loss:.6f}, TestAcc: {test_acc:.4f}")

# 9. main 保护
if __name__ == '__main__':
    main()
