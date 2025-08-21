# 1 导入库
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

def run_experiment(weight_decay, EPOCHS=10):
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
        nn.Linear(256, 10)
    )
    net.apply(init_weights)

    # 5 损失函数
    criterion = nn.CrossEntropyLoss()

    # 6 优化器（带正则项）
    decay, no_decay = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    trainer = torch.optim.SGD(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=0.1,
    )

    # 7 训练模型
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

    return train_losses, test_accs


def main():
    # 测试的正则项系数（weight_decay）
    lambdas = [0.0, 1e-5, 1e-4, 1e-3]

    results = {}
    for lam in lambdas:
        print(f"Running experiment with weight_decay={lam}")
        train_losses, test_accs = run_experiment(lam)
        results[lam] = (train_losses, test_accs)

    # 画图比较
    epochs = range(1, 11)
    plt.figure(figsize=(12, 5))

    # 子图1：训练损失
    plt.subplot(1, 2, 1)
    for lam, (train_losses, _) in results.items():
        plt.plot(epochs, train_losses, marker='o', label=f"λ={lam}")
    plt.title("Training Loss vs Weight Decay")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 子图2：测试准确率
    plt.subplot(1, 2, 2)
    for lam, (_, test_accs) in results.items():
        plt.plot(epochs, test_accs, marker='s', label=f"λ={lam}")
    plt.title("Test Accuracy vs Weight Decay")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
