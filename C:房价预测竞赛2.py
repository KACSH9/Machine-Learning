# 1 导入库
import numpy as np
import pandas as pd
import math
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# 2 数据读取
train_data = pd.read_csv('./california-house-prices/train.csv')
test_data = pd.read_csv('./california-house-prices/test.csv')

# 3 数据处理 对每一个特征进行分析
# 3.1 特征与目标提取
train_features = train_data.drop(columns='Sold Price')
train_labels = train_data['Sold Price']

test_features = test_data.copy()

id_train = train_features['Id'].copy() if 'Id' in train_features.columns else None
id_test  = test_features['Id'].copy()  if 'Id' in test_features.columns  else None

# 3.2 删除无效特征
drop_cols = ['Id', 'Address', 'Summary', 'City', 'State']
train_features = train_features.drop(columns=drop_cols)
test_features = test_features.drop(columns=drop_cols)

# 3.3 缺失值处理
# 3.3.1 数值列 / 类别列分列
num_cols = train_features.select_dtypes(include=[np.number]).columns
cat_cols = train_features.columns.difference(num_cols)

# 3.3.2 数值列：用“训练集中位数”填充
medians = train_features[num_cols].median()
train_features[num_cols] = train_features[num_cols].fillna(medians)
test_features[num_cols]  = test_features[num_cols].fillna(medians)

# 3.3.3 类别列：用占位符填充
train_features[cat_cols] = train_features[cat_cols].fillna('Unknown')
test_features[cat_cols]  = test_features[cat_cols].fillna('Unknown')

# 3.4 特殊值处理
# 3.4.1 极偏值处理
skew_cols = ['Lot', 'Total interior livable area', 'Total spaces', 'Garage spaces']
for c in skew_cols:
    if c in train_features.columns:
        train_features[c] = np.log1p(train_features[c].clip(lower=0))  # 负值防御
    if c in test_features.columns:
        test_features[c]  = np.log1p(test_features[c].clip(lower=0))

# 3.4.2 日期处理
date_cols = ['Listed On', 'Last Sold On']
base = pd.Timestamp('2000-01-01')

for col in date_cols:
    if col in train_features.columns:
        tr = pd.to_datetime(train_features[col], format='%Y/%m/%d', errors='coerce')
        te = pd.to_datetime(test_features[col],  format='%Y/%m/%d', errors='coerce')

        # 只保留一个数值特征：距 2000-01-01 的天数
        newc = col + '_since2000_days'
        train_features[newc] = (tr - base).dt.days.astype('float32')
        test_features[newc]  = (te - base).dt.days.astype('float32')

        # 用训练集的中位数填充新特征的缺失
        med = train_features[newc].median()
        train_features[newc] = train_features[newc].fillna(med)
        test_features[newc] = test_features[newc].fillna(med)

        # 删除原始日期列，避免 One-Hot 爆列
        train_features.drop(columns=[col], inplace=True)
        test_features.drop(columns=[col],  inplace=True)

# 3.5 One-hot类别编码
# 3.5.1 记录“原始数值列”（后面只对这些做标准化）
num_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()

# 3.5.2 纵向拼接（不打乱行顺序）
all_X = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# 3.5.3 一次性做 One-Hot（所有类别列都会被展开）
all_X = pd.get_dummies(all_X, dummy_na=False, dtype=np.uint8)

# 3.5.4 切回 train/test
n_train = len(train_features)
train_X = all_X.iloc[:n_train].copy()
test_X = all_X.iloc[n_train:].copy()

# 3.6 数值标准化
# 确保原始数值列仍在 One-Hot 后的数据里
missing = [c for c in num_cols if c not in train_X.columns]
assert not missing, f"这些原始数值列在编码后缺失了：{missing}"

means = train_X[num_cols].mean()
stds = train_X[num_cols].std().replace(0, 1)

train_X[num_cols] = (train_X[num_cols] - means) / stds
test_X[num_cols] = (test_X[num_cols] - means) / stds

# 3.7 格式转化
# 3.7.1 转tensor
X_all = torch.tensor(train_X.values, dtype=torch.float32)
y_all = torch.tensor(train_labels.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(test_X.values, dtype=torch.float32)

# 3.7.2 切分 train/valid（8:2）
n_total = len(X_all)
n_train = int(n_total * 0.8)
n_valid = n_total - n_train
train_ds, valid_ds = random_split(TensorDataset(X_all, y_all), [n_train, n_valid], generator=torch.Generator().manual_seed(42))

# 3.7.3 DataLoader
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=512, shuffle=False)

# 4 模型函数LR
def get_net(in_dim: int):
    return nn.Linear(in_dim, 1)

# 5 损失函数
criterion = nn.MSELoss()

def log_rmse(net, features, labels):
    preds = net(features)
    clipped_preds = torch.clamp(preds, min=1.0)   # 避免 log(<=0)
    log_preds = torch.log1p(clipped_preds)        # log(1 + y_hat)
    log_labels = torch.log1p(labels)              # log(1 + y)
    rmse = torch.sqrt(criterion(log_preds, log_labels))
    return rmse.item()

def evaluate_log_rmse(net, loader):
    net.eval()
    device = next(net.parameters()).device
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = net(xb)
            log_preds = torch.log1p(torch.clamp(preds, min=1.0))
            log_labels = torch.log1p(yb)
            batch_mse = criterion(log_preds, log_labels).item()
            total_loss += batch_mse * yb.size(0)
            total_n += yb.size(0)
    return math.sqrt(total_loss / max(total_n, 1))

# 6 训练函数
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def kfold_train_and_predict(
    X_all, y_all, X_test_tensor,
    k=5, num_epochs=50, batch_size=256, lr=1e-3, weight_decay=1e-4,
    device=None, seed=42, eval_every=1
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = X_all.shape[0]
    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    oof_logrmse = []
    oof_pred = np.zeros(N, dtype=np.float32)
    test_pred_sum = np.zeros(X_test_tensor.shape[0], dtype=np.float32)

    histories = []  # 保存每折的逐 epoch 训练/验证 log-RMSE

    for fold, val_idx in enumerate(folds, 1):
        train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_va, y_va = X_all[val_idx], y_all[val_idx]

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=512, shuffle=False)

        net = get_net(X_all.shape[1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        hist = {"epoch": [], "train_logrmse": [], "valid_logrmse": []}

        for epoch in range(1, num_epochs + 1):
            net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                preds = net(xb)
                loss = criterion(preds, yb)
                loss.backward()
                opt.step()

            if epoch % eval_every == 0:
                tr_lr = evaluate_log_rmse(net, train_loader)
                va_lr = evaluate_log_rmse(net, valid_loader)
                hist["epoch"].append(epoch)
                hist["train_logrmse"].append(tr_lr)
                hist["valid_logrmse"].append(va_lr)
                print(f"[Fold {fold}/{k}] Epoch {epoch:03d}  Train log-RMSE: {tr_lr:.4f} | Valid log-RMSE: {va_lr:.4f}")

        # 本折验证分数（用最后一次的 valid log-RMSE）
        va_logrmse = hist["valid_logrmse"][-1] if hist["valid_logrmse"] else evaluate_log_rmse(net, valid_loader)
        oof_logrmse.append(va_logrmse)
        histories.append(hist)

        # 导出每折曲线 CSV
        pd.DataFrame(hist).to_csv(f"kfold_fold{fold}_history.csv", index=False)

        # OOF 与测试集预测
        net.eval()
        with torch.no_grad():
            oof_pred[val_idx] = net(X_va.to(device)).cpu().numpy().reshape(-1)
            test_pred_sum += net(X_test_tensor.to(device)).cpu().numpy().reshape(-1)

    mean_logrmse = float(np.mean(oof_logrmse))
    std_logrmse  = float(np.std(oof_logrmse))
    print(f"K-fold mean valid log-RMSE: {mean_logrmse:.4f} ± {std_logrmse:.4f}")

    test_pred = test_pred_sum / k
    return {
        "oof_logrmse_per_fold": oof_logrmse,
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "mean_valid_logrmse": mean_logrmse,
        "std_valid_logrmse": std_logrmse,
        "histories": histories,  # 新增：所有折的学习曲线
    }

# 7 调用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k = 5  # 可改 5/10
res = kfold_train_and_predict(
    X_all=X_all, y_all=y_all, X_test_tensor=X_test_tensor,
    k=k, num_epochs=50, batch_size=256, lr=1e-3, weight_decay=1e-4, device=device
)

'''
# 生成提交（可选）
sub = pd.DataFrame({
    "Id": (id_test if id_test is not None else test_data["Id"]),
    "Sold Price": res["test_pred"]
})
sub.to_csv(f"submission_lr_kfold{k}.csv", index=False)
print(f"Saved submission_lr_kfold{k}.csv")
'''

# 8 画图
plt.figure()
for i, hist in enumerate(res["histories"], 1):
    epochs = hist["epoch"]
    vals = hist["valid_logrmse"]
    plt.plot(epochs, vals, label=f"fold {i}")
plt.xlabel("Epoch")
plt.ylabel("Valid log-RMSE")
plt.title("K-Fold Validation Curves")
plt.legend()
plt.tight_layout()
plt.savefig("kfold_valid_curves.png", dpi=150)
plt.close()

# （可选）训练集曲线
plt.figure()
for i, hist in enumerate(res["histories"], 1):
    epochs = hist["epoch"]
    vals = hist["train_logrmse"]
    plt.plot(epochs, vals, label=f"fold {i}")
plt.xlabel("Epoch")
plt.ylabel("Train log-RMSE")
plt.title("K-Fold Training Curves")
plt.legend()
plt.tight_layout()
plt.savefig("kfold_train_curves.png", dpi=150)
plt.close()

# 2) 各折最终验证分数的箱线图/散点
final_scores = np.array(res["oof_logrmse_per_fold"])
plt.figure()
plt.boxplot(final_scores)
plt.scatter(np.ones_like(final_scores), final_scores)
plt.ylabel("Valid log-RMSE")
plt.title("Fold Valid Scores Distribution")
plt.tight_layout()
plt.savefig("kfold_valid_scores_box.png", dpi=150)
plt.close()

# 3) 测试集预测分布（直方图）
plt.figure()
plt.hist(res["test_pred"], bins=50)
plt.xlabel("Predicted Sold Price")
plt.ylabel("Count")
plt.title("Test Predictions Histogram")
plt.tight_layout()
plt.savefig("kfold_test_pred_hist.png", dpi=150)
plt.close()

print("Saved plots: kfold_valid_curves.png, kfold_train_curves.png, kfold_valid_scores_box.png, kfold_test_pred_hist.png")
