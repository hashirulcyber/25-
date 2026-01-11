# ============================================
# Wimbledon 2023 
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 0) 参数配置
# =========================
DATA_PATH = "data/wimbledon_2023_points_formatted_like_featured.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2  # 测试集占比（按比赛 match_id 切分）

# 网格搜索范围
SEQ_LENS = [5, 10, 20]       # 序列长度（时间窗口）
HIDDEN_DIMS = [32, 64]       # LSTM隐藏单元数

EPOCHS = 30
PATIENCE = 3                 # 连续几轮AUC不提升则早停
BATCH_SIZE = 64

LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
GRAD_CLIP = 1.0

# 动量特征窗口（最近3/5/10分）
MOM_WINDOWS = (3, 5, 10)


# =========================
# 1) 指标计算函数
# =========================
def evaluate_metrics(y_true, y_prob, threshold=0.5):
    """
    y_true: 真实标签 (0/1)
    y_prob: 预测概率 (0~1)
    threshold: 分类阈值（用于F1）
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    f1 = f1_score(y_true, y_pred)
    # AUC 需要 y_true 同时有 0 和 1
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    # MAE/RMSE：把预测概率当连续值，衡量概率误差
    mae = mean_absolute_error(y_true, y_prob)
    rmse = np.sqrt(mean_squared_error(y_true, y_prob))
    return {"F1": f1, "AUC": auc, "MAE": mae, "RMSE": rmse}


# =========================
# 2) 读取与清洗
#    - 把空值/空格等识别为 NaN
#    - speed_mph==0 作为“无测速”处理
# =========================
def read_and_clean(path: str) -> pd.DataFrame:
    # 把常见的占位符当作缺失
    na_tokens = ["", " ", "NA", "N/A", "null", "None", "-", "--"]
    df = pd.read_csv(path, na_values=na_tokens, keep_default_na=True)

    # 纯空白字符串 -> NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # 按比赛内时间顺序排序（LSTM/滑窗必须依赖正确顺序）
    df = df.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)

    # elapsed_time -> elapsed_seconds
    def hhmmss_to_seconds(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        parts = x.split(":")
        if len(parts) != 3:
            return np.nan
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)

    df["elapsed_seconds"] = df["elapsed_time"].apply(hhmmss_to_seconds)

    # 类别字段缺失填 Unknown，避免 one-hot 报错
    for c in ["serve_width", "serve_depth", "return_depth"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    # winner_shot_type：原始“0”表示没有winner，这里改成更清晰的类别值
    if "winner_shot_type" in df.columns:
        df["winner_shot_type"] = df["winner_shot_type"].fillna("0").astype(str)
        df["winner_shot_type"] = df["winner_shot_type"].replace({"0": "NoneShot"})

    # speed_mph 转数值
    df["speed_mph"] = pd.to_numeric(df["speed_mph"], errors="coerce")

    df["speed_mph_zero"] = ((df["speed_mph"].isna()) | (df["speed_mph"] <= 0)).astype(int)

    # 先构造当前分 p1 是否赢（后面做标签/动量用）
    df["p1_point_win"] = (df["point_victor"] == 1).astype(int)

    # 双误标记（用于解释 speed_mph==0 的机制）
    if "p1_double_fault" in df.columns and "p2_double_fault" in df.columns:
        df["double_fault_any"] = ((df["p1_double_fault"] == 1) | (df["p2_double_fault"] == 1)).astype(int)
    else:
        df["double_fault_any"] = 0

    df["speed0_and_df"] = ((df["speed_mph_zero"] == 1) & (df["double_fault_any"] == 1)).astype(int)
    df["speed0_no_df"]  = ((df["speed_mph_zero"] == 1) & (df["double_fault_any"] == 0)).astype(int)

    # 比分字段鲁棒编码：0/15/30/40/AD + 抢七数字
    def parse_tennis_score(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().upper()
        score_map = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
        if x in score_map:
            return score_map[x]
        if x.isdigit():
            return int(x)
        return np.nan

    df["p1_score_num"] = df["p1_score"].apply(parse_tennis_score)
    df["p2_score_num"] = df["p2_score"].apply(parse_tennis_score)

    # 如果 point_victor 出现 0（无效分），直接删掉
    df = df[df["point_victor"].isin([1, 2])].copy().reset_index(drop=True)

    return df


# =========================
# 3) 构造标签：预测“下一分 p1 是否获胜”
# =========================
def add_label_next_point(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)

    df["p1_point_win"] = (df["point_victor"] == 1).astype(int)
    df["y_next_p1_win"] = df.groupby("match_id")["p1_point_win"].shift(-1)

    # 每场最后一分没有下一分标签，删掉
    df = df.dropna(subset=["y_next_p1_win"]).reset_index(drop=True)
    df["y_next_p1_win"] = df["y_next_p1_win"].astype(int)
    return df


# =========================
# 4) 动量特征（改进方向C）
#    - 最近 N 分赢分率、winner/UE/ace/df趋势、平均回合、平均速度等
#    - 所有 rolling 均使用 shift(1)，保证“只用过去信息”（防泄漏）
# =========================
def add_momentum_features(match_df: pd.DataFrame, windows=MOM_WINDOWS) -> pd.DataFrame:
    g = match_df.copy()

    # 事件列确保为数值（缺失按0处理）
    event_cols = ["p1_winner","p1_unf_err","p1_ace","p1_double_fault",
                  "p2_winner","p2_unf_err","p2_ace","p2_double_fault"]
    for col in event_cols:
        if col not in g.columns:
            g[col] = 0
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0)

    # 数值性能列
    perf_cols = ["rally_count","speed_mph","p1_distance_run","p2_distance_run"]
    for col in perf_cols:
        if col not in g.columns:
            g[col] = np.nan
        g[col] = pd.to_numeric(g[col], errors="coerce")

    # 速度rolling时：把无测速(<=0)转成 NaN，否则平均会被0拉低
    speed_for_roll = g["speed_mph"].where(g["speed_mph"] > 0, np.nan)

    # lagged 赢分（用于 winrate / streak）
    g["p1_point_win_lag1"] = g["p1_point_win"].shift(1)

    # 连胜长度（基于上一分是否赢）
    lag_win = g["p1_point_win_lag1"].fillna(0).astype(int).to_numpy()
    streak = np.zeros_like(lag_win)
    run = 0
    for i, v in enumerate(lag_win):
        if v == 1:
            run += 1
        else:
            run = 0
        streak[i] = run
    g["p1_streak_lag1"] = streak

    # rolling 动量统计
    for w in windows:
        # 过去w分赢分率
        g[f"p1_winrate_{w}"] = g["p1_point_win"].shift(1).rolling(w, min_periods=1).mean()

        # 过去w分事件计数
        g[f"p1_winner_cnt_{w}"] = g["p1_winner"].shift(1).rolling(w, min_periods=1).sum()
        g[f"p1_ue_cnt_{w}"]     = g["p1_unf_err"].shift(1).rolling(w, min_periods=1).sum()
        g[f"p1_ace_cnt_{w}"]    = g["p1_ace"].shift(1).rolling(w, min_periods=1).sum()
        g[f"p1_df_cnt_{w}"]     = g["p1_double_fault"].shift(1).rolling(w, min_periods=1).sum()

        g[f"p2_winner_cnt_{w}"] = g["p2_winner"].shift(1).rolling(w, min_periods=1).sum()
        g[f"p2_ue_cnt_{w}"]     = g["p2_unf_err"].shift(1).rolling(w, min_periods=1).sum()
        g[f"p2_ace_cnt_{w}"]    = g["p2_ace"].shift(1).rolling(w, min_periods=1).sum()
        g[f"p2_df_cnt_{w}"]     = g["p2_double_fault"].shift(1).rolling(w, min_periods=1).sum()

        # 过去w分均值特征
        g[f"avg_rally_{w}"]  = g["rally_count"].shift(1).rolling(w, min_periods=1).mean()
        g[f"avg_speed_{w}"]  = speed_for_roll.shift(1).rolling(w, min_periods=1).mean()
        g[f"p1_run_avg_{w}"] = g["p1_distance_run"].shift(1).rolling(w, min_periods=1).mean()
        g[f"p2_run_avg_{w}"] = g["p2_distance_run"].shift(1).rolling(w, min_periods=1).mean()

        # 动量差（p1 - p2）
        g[f"mom_diff_winner_{w}"] = g[f"p1_winner_cnt_{w}"] - g[f"p2_winner_cnt_{w}"]
        g[f"mom_diff_ue_{w}"]     = g[f"p1_ue_cnt_{w}"] - g[f"p2_ue_cnt_{w}"]

    # 初期rolling会出现NaN，用0填补
    mom_cols = [c for c in g.columns if (
        c.startswith("p1_winrate_") or c.startswith("p1_winner_cnt_") or c.startswith("p1_ue_cnt_") or
        c.startswith("p1_ace_cnt_") or c.startswith("p1_df_cnt_") or
        c.startswith("p2_winner_cnt_") or c.startswith("p2_ue_cnt_") or
        c.startswith("p2_ace_cnt_") or c.startswith("p2_df_cnt_") or
        c.startswith("avg_") or c.startswith("p1_run_avg_") or c.startswith("p2_run_avg_") or
        c.startswith("mom_diff_") or c.startswith("p1_streak_")
    )]
    g[mom_cols] = g[mom_cols].fillna(0)

    return g


# =========================
# 5) LSTM 模型相关组件
# =========================
class SeqDataset(Dataset):
    """把 (样本数, seq_len, 特征维) 的序列数据包装成 PyTorch Dataset"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # BCEWithLogitsLoss 要 float
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    """LSTM + Dropout + Dense（二分类输出logit）"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (B, T, H)
        last = out[:, -1, :]           # 取最后一个时间步作为序列表示 (B, H)
        last = self.dropout(last)
        logit = self.fc(last).squeeze(1)  # (B,)
        return logit

def predict_proba(model, loader, device):
    """得到测试集预测概率（sigmoid之后）"""
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logit = model(xb)
            p = torch.sigmoid(logit).cpu().numpy()
            probs.append(p)
            ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)

def make_sequences_local(df_part, X_flat, y_flat, seq_len=10):
    """
    按 match_id 分组做滑动窗口，保证窗口不跨比赛
    """
    X_list, y_list = [], []
    start = 0
    for mid, g in df_part.groupby("match_id", sort=False):
        n = g.shape[0]
        Xg = X_flat[start:start+n]
        yg = y_flat[start:start+n]
        for i in range(n - seq_len + 1):
            X_list.append(Xg[i:i+seq_len])
            y_list.append(yg[i+seq_len-1])
        start += n
    return np.array(X_list), np.array(y_list)

def train_with_early_stopping(X_train, y_train, X_test, y_test, hidden_dim, dropout=DROPOUT):
    """
    训练 LSTM，并按 AUC early stopping，返回最佳模型对应的指标
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定随机种子方便复现
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader  = DataLoader(SeqDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = LSTMClassifier(input_dim=X_train.shape[2], hidden_dim=hidden_dim, num_layers=1, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1
    best_state = None
    no_improve = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()

            # 梯度裁剪：防止LSTM梯度爆炸/震荡
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

        # 每轮在测试集算指标（用于早停）
        y_prob, y_true = predict_proba(model, test_loader, device)
        metrics = evaluate_metrics(y_true, y_prob, threshold=0.5)

        # AUC 改善则记录模型
        if metrics["AUC"] > best_auc + 1e-4:
            best_auc = metrics["AUC"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终（最佳模型）指标
    y_prob, y_true = predict_proba(model, test_loader, device)
    final_metrics = evaluate_metrics(y_true, y_prob, threshold=0.5)
    return final_metrics


# =========================
# 6) 主流程：清洗 -> 标签 -> 动量 -> 切分 -> 特征工程 -> 窗口 -> 训练/调参
# =========================
df = read_and_clean(DATA_PATH)
df = add_label_next_point(df)

# 按比赛生成动量特征（防止跨比赛rolling）
df = df.groupby("match_id", group_keys=False).apply(add_momentum_features)

print("加入动量特征后 shape:", df.shape)
print("标签分布:\n", df["y_next_p1_win"].value_counts(normalize=True))

# 按 match_id 切分训练/测试，避免泄漏
match_ids = df["match_id"].unique()
train_mids, test_mids = train_test_split(match_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)

train_df = df[df["match_id"].isin(train_mids)].copy().reset_index(drop=True)
test_df  = df[df["match_id"].isin(test_mids)].copy().reset_index(drop=True)

print("训练比赛数:", len(train_mids), "测试比赛数:", len(test_mids))
print("训练行数:", train_df.shape[0], "测试行数:", test_df.shape[0])

# speed_mph_filled：用训练集“有效速度(>0)”的中位数填补无测速点（避免测试集统计泄漏）
train_valid_speed = train_df.loc[train_df["speed_mph"] > 0, "speed_mph"]
speed_median_train = train_valid_speed.median()

for part in [train_df, test_df]:
    part["speed_mph_filled"] = part["speed_mph"]
    part.loc[part["speed_mph_zero"] == 1, "speed_mph_filled"] = speed_median_train

# -------- 特征列表（可按需要增减）--------
# 数值特征：原始数值 + 动量数值（全部会做标准化）
base_num_features = [
    "elapsed_seconds",
    "set_no","game_no","point_no",
    "p1_sets","p2_sets","p1_games","p2_games",
    "p1_points_won","p2_points_won",
    "p1_distance_run","p2_distance_run",
    "rally_count",
    "p1_score_num","p2_score_num",
    "speed_mph_filled"
]

# 类别特征：one-hot
cat_features = ["server","serve_no","serve_width","serve_depth","return_depth","winner_shot_type"]

# 二值特征：直接拼接
bin_features = [
    "p1_ace","p2_ace","p1_winner","p2_winner",
    "p1_double_fault","p2_double_fault","p1_unf_err","p2_unf_err",
    "p1_net_pt","p2_net_pt","p1_net_pt_won","p2_net_pt_won",
    "p1_break_pt","p2_break_pt","p1_break_pt_won","p2_break_pt_won",
    "p1_break_pt_missed","p2_break_pt_missed",
    # 信息性缺失相关
    "speed_mph_zero","speed0_and_df","speed0_no_df"
]

# 动量特征列
momentum_features = [c for c in df.columns if (
    c.startswith("p1_winrate_") or c.startswith("p1_winner_cnt_") or c.startswith("p1_ue_cnt_") or
    c.startswith("p1_ace_cnt_") or c.startswith("p1_df_cnt_") or
    c.startswith("p2_winner_cnt_") or c.startswith("p2_ue_cnt_") or
    c.startswith("p2_ace_cnt_") or c.startswith("p2_df_cnt_") or
    c.startswith("avg_") or c.startswith("p1_run_avg_") or c.startswith("p2_run_avg_") or
    c.startswith("mom_diff_") or c.startswith("p1_streak_")
)]

num_features = base_num_features + momentum_features

# 防止缺列
for c in num_features + bin_features + cat_features:
    if c not in train_df.columns:
        train_df[c] = 0
        test_df[c] = 0

# -------- 类别 one-hot（只fit训练集）--------
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(train_df[cat_features])
X_train_cat = ohe.transform(train_df[cat_features])
X_test_cat  = ohe.transform(test_df[cat_features])

# -------- 数值标准化（只fit训练集）--------
scaler = StandardScaler()
scaler.fit(train_df[num_features])
X_train_num = scaler.transform(train_df[num_features])
X_test_num  = scaler.transform(test_df[num_features])

# -------- 二值特征直接拼接 --------
X_train_bin = train_df[bin_features].to_numpy(dtype=float)
X_test_bin  = test_df[bin_features].to_numpy(dtype=float)

# 拼成 flat 特征
X_train_flat = np.concatenate([X_train_num, X_train_bin, X_train_cat], axis=1)
X_test_flat  = np.concatenate([X_test_num,  X_test_bin,  X_test_cat], axis=1)

y_train_flat = train_df["y_next_p1_win"].to_numpy(dtype=int)
y_test_flat  = test_df["y_next_p1_win"].to_numpy(dtype=int)

print("Flat 特征维度:", X_train_flat.shape[1])

# =========================
# 7) 网格搜索：seq_len + hidden_dim
# =========================
results = []

for seq_len in SEQ_LENS:
    # 按比赛做滑窗（窗口不跨比赛）
    X_train, y_train = make_sequences_local(train_df, X_train_flat, y_train_flat, seq_len=seq_len)
    X_test,  y_test  = make_sequences_local(test_df,  X_test_flat,  y_test_flat,  seq_len=seq_len)

    print(f"\n--- 序列长度 seq_len={seq_len} ---")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    for hidden_dim in HIDDEN_DIMS:
        metrics = train_with_early_stopping(X_train, y_train, X_test, y_test,
                                            hidden_dim=hidden_dim, dropout=DROPOUT)
        row = {"seq_len": seq_len, "hidden_dim": hidden_dim, **metrics}
        results.append(row)
        print("完成:", row)

results_df = pd.DataFrame(results).sort_values(["AUC", "F1"], ascending=False).reset_index(drop=True)

print("\n=== 网格搜索结果（按AUC再按F1排序） ===")
print(results_df)

# 保存结果
out_results_path = "data/2023_lstm_momentum_grid_results.csv"
results_df.to_csv(out_results_path, index=False)
print("\n已保存网格结果到:", out_results_path)

# ============ 固定最优参数 ============
seq_len_best = 5
hidden_best = 64

# 重新生成序列数据（窗口化）
X_train_best, y_train_best = make_sequences_local(train_df, X_train_flat, y_train_flat, seq_len=seq_len_best)
X_test_best,  y_test_best  = make_sequences_local(test_df,  X_test_flat,  y_test_flat,  seq_len=seq_len_best)

print("X_train_best:", X_train_best.shape, "y_train_best:", y_train_best.shape)
print("X_test_best :", X_test_best.shape,  "y_test_best :",  y_test_best.shape)

# ======== 替换版训练函数（带history与y_prob导出）========
def train_with_early_stopping_best(X_train, y_train, X_test, y_test, hidden_dim, dropout=DROPOUT):
    """
    训练 LSTM，并按 AUC early stopping
    返回：model, final_metrics, history, best_y_prob, best_y_true
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定随机种子
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader  = DataLoader(SeqDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = LSTMClassifier(input_dim=X_train.shape[2], hidden_dim=hidden_dim, num_layers=1, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "test_loss": [], "F1": [], "AUC": [], "MAE": [], "RMSE": []}

    best_auc = -1
    best_state = None
    best_y_prob = None
    best_y_true = None
    no_improve = 0

    for ep in range(1, EPOCHS + 1):
        # -------- 训练 --------
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        # -------- 测试 loss --------
        model.eval()
        test_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logit = model(xb)
                loss = criterion(logit, yb)
                test_loss_sum += loss.item() * xb.size(0)
        test_loss = test_loss_sum / len(test_loader.dataset)

        # -------- 指标 --------
        y_prob, y_true = predict_proba(model, test_loader, device)
        metrics = evaluate_metrics(y_true, y_prob, threshold=0.5)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        for k in ["F1", "AUC", "MAE", "RMSE"]:
            history[k].append(metrics[k])

        print(
            f"Epoch {ep:02d} | train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"| F1={metrics['F1']:.4f} AUC={metrics['AUC']:.4f} "
            f"MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f}"
        )

        # -------- Early stopping（按AUC）--------
        if metrics["AUC"] > best_auc + 1e-4:
            best_auc = metrics["AUC"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_y_prob = y_prob.copy()
            best_y_true = y_true.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {ep} | best AUC={best_auc:.4f}")
                break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_metrics(best_y_true, best_y_prob, threshold=0.5)
    return model, final_metrics, history, best_y_prob, best_y_true


# ======== 训练最佳模型 ========
best_model, final_metrics, history, y_prob, y_true = train_with_early_stopping_best(
    X_train_best, y_train_best, X_test_best, y_test_best,
    hidden_dim=hidden_best, dropout=DROPOUT
)

print("\nFinal(best) metrics:", final_metrics)

# 保存预测结果
pred_path = "data/2023_best_seq5_hid64_test_predictions.csv"
pd.DataFrame({"y_true": y_true.astype(int), "y_prob": y_prob.astype(float)}).to_csv(pred_path, index=False)
print("已保存预测结果到:", pred_path)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["test_loss"], label="test_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (Early Stopping)")
plt.legend()
plt.tight_layout()
plt.show()

# 从测试集中选一场比赛展示
example_mid = test_df["match_id"].iloc[0]
g = test_df[test_df["match_id"] == example_mid].copy().reset_index(drop=True)

# 对这一场比赛单独做 transform（用已fit好的 scaler/ohe）
Xg_cat = ohe.transform(g[cat_features])
Xg_num = scaler.transform(g[num_features])
Xg_bin = g[bin_features].to_numpy(dtype=float)
Xg_flat = np.concatenate([Xg_num, Xg_bin, Xg_cat], axis=1)

yg = g["y_next_p1_win"].to_numpy(dtype=int)

# 生成窗口序列
Xg_seq, yg_seq = [], []
for i in range(len(g) - seq_len_best + 1):
    Xg_seq.append(Xg_flat[i:i+seq_len_best])
    yg_seq.append(yg[i+seq_len_best-1])
Xg_seq = np.array(Xg_seq)
yg_seq = np.array(yg_seq)

# 用最佳模型预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(device)
best_model.eval()

with torch.no_grad():
    xb = torch.tensor(Xg_seq, dtype=torch.float32).to(device)
    probs = torch.sigmoid(best_model(xb)).cpu().numpy()

plt.figure()
plt.plot(probs, label="Predicted Probability P(next p1 win)")
plt.plot(yg_seq, label="True Label", alpha=0.7)
plt.xlabel("Point Index (Window End)")
plt.ylabel("Value")
plt.title(f"Match-level Prediction Trend| match_id={example_mid}")
plt.legend()
plt.tight_layout()
plt.show()

# 选择动量特征
mom_show = [
    "p1_winrate_5",
    "mom_diff_winner_5",
    "mom_diff_ue_5",
    "p1_streak_lag1",
    "avg_rally_5",
    "avg_speed_5"
]

# 如果某些列不存在，就自动过滤掉
mom_show = [c for c in mom_show if c in g.columns]

H = g[mom_show].to_numpy(dtype=float).T  # (特征数, 时间)

plt.figure(figsize=(10, 4))
plt.imshow(H, aspect="auto")
plt.yticks(range(len(mom_show)), mom_show)
plt.xlabel("Point Index")
plt.title(f"Momentum Feature Heatmap | match_id={example_mid}")
plt.colorbar()
plt.tight_layout()
plt.show()
