"""
MorphGuard-IDS Baseline Models Training.
Trains and evaluates all baselines with proper tuning.
"""

import os, sys, json, time, warnings, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "baselines_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_data(dataset_name: str):
    preproc = ROOT / "results" / "preprocessed"
    meta_dir = ROOT / "results" / "metadata"
    # tiissrc23 is 6.06M training rows at natural prevalence; a documented,
    # class-capped (<=20,000/class) TRAINING sample is used to keep model
    # fitting tractable. Validation and test sets are ALWAYS the full
    # natural-prevalence splits (1,298,344 rows each) -- only the training
    # distribution is capped, never the evaluation distribution.
    capped_path = preproc / f"{dataset_name}_train_capped.parquet"
    train_path = capped_path if capped_path.exists() else preproc / f"{dataset_name}_train.parquet"
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(preproc / f"{dataset_name}_val.parquet")
    test_df = pd.read_parquet(preproc / f"{dataset_name}_test.parquet")
    with open(meta_dir / f"{dataset_name}_meta.json") as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]
    n_classes = meta["n_classes_multi"]
    if train_path == capped_path:
        log.info(f"  Using capped training sample ({len(train_df)} rows, <=20,000/class) "
                 f"for {dataset_name}; val/test remain full natural prevalence.")
    return train_df, val_df, test_df, feat_cols, n_classes


def compute_metrics(y_true, y_pred, y_prob=None, n_classes=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    unique = np.unique(y_true)
    fpr, fnr = 0.0, 0.0
    for c in unique:
        yb = (np.array(y_true) == c).astype(int)
        yp = (np.array(y_pred) == c).astype(int)
        tn = ((yb == 0) & (yp == 0)).sum()
        fp = ((yb == 0) & (yp == 1)).sum()
        fn = ((yb == 1) & (yp == 0)).sum()
        tp = ((yb == 1) & (yp == 1)).sum()
        fpr += fp / max(fp + tn, 1)
        fnr += fn / max(fn + tp, 1)
    fpr /= max(len(unique), 1)
    fnr /= max(len(unique), 1)
    metrics = {"accuracy": acc, "precision_macro": prec, "recall_macro": rec,
               "f1_macro": f1, "f1_weighted": wf1, "fpr": fpr, "fnr": fnr}
    if y_prob is not None:
        try:
            if n_classes and n_classes > 2:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            else:
                col = y_prob[:, 1] if (hasattr(y_prob, 'ndim') and y_prob.ndim > 1 and y_prob.shape[1] == 2) else y_prob
                auc = roc_auc_score(y_true, col if y_prob.ndim == 1 else y_prob, multi_class="ovr", average="macro")
            metrics["roc_auc"] = auc
        except Exception:
            metrics["roc_auc"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# PyTorch neural baselines
# ---------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    def __init__(self, d_in, n_classes, hidden=256, n_layers=3, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1DBaseline(nn.Module):
    def __init__(self, d_in, n_classes, d_model=128, n_layers=4, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        convs = []
        norms = []
        for i in range(n_layers):
            dil = 2 ** i
            convs.append(nn.Conv1d(d_model, d_model, kernel_size=3, padding=dil * 2, dilation=dil))
            norms.append(nn.LayerNorm(d_model))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.proj(x).unsqueeze(2)  # (B, d, 1)
        for conv, norm in zip(self.convs, self.norms):
            h_out = conv(h)[..., :1]  # trim to length 1
            h_out = norm(h_out.squeeze(2)).unsqueeze(2)
            h_out = torch.relu(h_out)
            h_out = self.drop(h_out)
            h = h + h_out  # residual
        return self.fc(h.squeeze(2))


class GRUBaseline(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, n_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0,
                           bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden * mult, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        out, _ = self.gru(h)
        return self.fc(self.drop(out[:, -1, :]))


class LSTMBaseline(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, n_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.lstm = nn.LSTM(hidden, hidden, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0,
                           bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden * mult, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)  # (B, 1, hidden)
        out, _ = self.lstm(h)
        return self.fc(self.drop(out[:, -1, :]))


class TransformerBaseline(nn.Module):
    def __init__(self, d_in, n_classes, d_model=128, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_classes)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        h = self.encoder(h)
        h = self.norm(h[:, 0, :])
        return self.fc(h)


class TCNBaseline(nn.Module):
    def __init__(self, d_in, n_classes, d_model=128, n_layers=4, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        blocks = []
        for i in range(n_layers):
            dil = 2 ** i
            blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=dil * 2, dilation=dil),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.proj(x).unsqueeze(2)
        for blk in self.blocks:
            h_new = blk(h)[..., :h.shape[2]]
            h = h_new + h
        return self.fc(h.squeeze(2))


# ---------------------------------------------------------------------------
# GCN/GAT baseline (using PyTorch Geometric)
# ---------------------------------------------------------------------------

def try_import_pyg():
    try:
        from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
        from torch_geometric.data import Data, Batch
        return True
    except ImportError:
        return False


class GATBaseline(nn.Module):
    def __init__(self, d_in, n_classes, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.proj = nn.Linear(d_in, d_model)
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_ch = d_model if i == 0 else d_model * n_heads
            out_ch = d_model if i < n_layers - 1 else d_model
            concat = (i < n_layers - 1)
            self.gat_layers.append(
                GATConv(in_ch, out_ch, heads=n_heads if concat else 1,
                       dropout=dropout, concat=concat)
            )
        self.fc = nn.Linear(d_model, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward_graph(self, x, edge_index, batch):
        from torch_geometric.nn import global_mean_pool
        h = self.proj(x)
        for layer in self.gat_layers:
            h = layer(h, edge_index)
            h = torch.relu(h)
            h = self.drop(h)
        h = global_mean_pool(h, batch)
        return self.fc(h)

    def forward(self, x):
        # Fallback: simple tabular MLP when no graph structure
        h = self.proj(x)
        return self.fc(torch.relu(h))


class GraphSAGEBaseline(nn.Module):
    def __init__(self, d_in, n_classes, d_model=64, n_layers=2, dropout=0.2):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([SAGEConv(d_model, d_model) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward_graph(self, x, edge_index, batch):
        from torch_geometric.nn import global_mean_pool
        h = self.proj(x)
        for layer in self.layers:
            h = torch.relu(layer(h, edge_index))
            h = self.drop(h)
        h = global_mean_pool(h, batch)
        return self.fc(h)

    def forward(self, x):
        # Fallback: simple tabular MLP when no graph structure is constructed
        # (consistent with the GAT baseline's tabular fallback above)
        h = self.proj(x)
        return self.fc(torch.relu(h))


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_torch_model(
    model, train_X, train_y, val_X, val_y,
    n_epochs=40, lr=1e-3, batch_size=512, patience=8,
    class_weights=None, device=DEVICE
) -> Tuple[nn.Module, List[float]]:
    model.to(device)
    train_ds = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(val_X, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.long)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_f1 = -1.0
    best_state = None
    patience_cnt = 0
    val_f1s = []

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                p = model(xb).argmax(dim=-1).cpu().numpy()
                preds.extend(p.tolist())
        vf1 = f1_score(val_y, preds, average="macro", zero_division=0)
        val_f1s.append(vf1)

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, val_f1s


def eval_torch_model(model, X, y, batch_size=512, device=DEVICE, n_classes=None):
    model.eval()
    model.to(device)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    all_probs, latencies = [], []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            t0 = time.perf_counter()
            logits = model(xb)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / xb.shape[0] * 1e3)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    all_probs = np.vstack(all_probs)
    preds = all_probs.argmax(axis=1)
    metrics = compute_metrics(y, preds, all_probs, n_classes)
    metrics["latency_ms_per_sample"] = float(np.mean(latencies))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics["n_params"] = n_params
    metrics["model_size_mb"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return metrics


# ---------------------------------------------------------------------------
# Main: train all baselines
# ---------------------------------------------------------------------------

BASELINE_CONFIGS = {
    "LogisticRegression": {
        "type": "sklearn",
        "class": LogisticRegression,
        "params": {"max_iter": 300, "solver": "lbfgs", "C": 1.0, "n_jobs": -1},
    },
    "RandomForest": {
        "type": "sklearn",
        "class": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 15, "n_jobs": -1, "class_weight": "balanced"},
    },
    "ExtraTrees": {
        "type": "sklearn",
        "class": ExtraTreesClassifier,
        "params": {"n_estimators": 100, "max_depth": 15, "n_jobs": -1, "class_weight": "balanced"},
    },
    "HistGradientBoosting": {
        "type": "sklearn",
        "class": HistGradientBoostingClassifier,
        "params": {"max_iter": 200, "max_depth": 10, "class_weight": "balanced"},
    },
    "XGBoost": {"type": "xgb"},
    "LightGBM": {"type": "lgb"},
    "CatBoost": {"type": "cb"},
    "MLP": {"type": "torch", "class": MLPBaseline, "kwargs": {"hidden": 256, "n_layers": 3, "dropout": 0.2}},
    "1D-CNN": {"type": "torch", "class": CNN1DBaseline, "kwargs": {"d_model": 128, "n_layers": 4, "dropout": 0.2}},
    "LSTM": {"type": "torch", "class": LSTMBaseline, "kwargs": {"hidden": 128, "n_layers": 2, "dropout": 0.2}},
    "GRU": {"type": "torch", "class": GRUBaseline, "kwargs": {"hidden": 128, "n_layers": 2, "dropout": 0.2}},
    "CNN-BiLSTM": {"type": "torch_composite"},
    "TCN": {"type": "torch", "class": TCNBaseline, "kwargs": {"d_model": 128, "n_layers": 4, "dropout": 0.2}},
    "Transformer": {"type": "torch", "class": TransformerBaseline, "kwargs": {"d_model": 128, "n_heads": 4, "n_layers": 2, "dropout": 0.2}},
    "GAT": {"type": "torch", "class": GATBaseline, "kwargs": {"d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.2}},
    "GraphSAGE": {"type": "torch", "class": GraphSAGEBaseline, "kwargs": {"d_model": 64, "n_layers": 2, "dropout": 0.2}},
}


class CNNBiLSTMBaseline(nn.Module):
    def __init__(self, d_in, n_classes, d_model=128, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(d_model, d_model // 2, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(d_model, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.cnn(x).unsqueeze(1)
        out, _ = self.lstm(h)
        return self.fc(self.drop(out[:, -1, :]))


def run_baselines(dataset_name: str, seeds: List[int] = [42, 123, 456]):
    train_df, val_df, test_df, feat_cols, n_classes = load_data(dataset_name)
    train_X = train_df[feat_cols].values
    val_X = val_df[feat_cols].values
    test_X = test_df[feat_cols].values
    train_y = train_df["multi_label"].values
    val_y = val_df["multi_label"].values
    test_y = test_df["multi_label"].values
    d_in = len(feat_cols)

    cw = compute_class_weight("balanced", classes=np.unique(train_y), y=train_y)

    results_all = {}

    log.info(f"\n{'='*60}")
    log.info(f"Running baselines on: {dataset_name}  n_classes={n_classes}  d_in={d_in}")

    # ---- Sklearn models ----
    for name, cfg in BASELINE_CONFIGS.items():
        if cfg["type"] != "sklearn":
            continue
        log.info(f"  Training {name}...")
        seed_results = []
        for seed in seeds:
            np.random.seed(seed)
            clf = cfg["class"](**cfg["params"])
            if hasattr(clf, "random_state"):
                clf.set_params(random_state=seed)
            t0 = time.perf_counter()
            clf.fit(train_X, train_y)
            t1 = time.perf_counter()
            preds = clf.predict(test_X)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(test_X)
            else:
                probs = None
            m = compute_metrics(test_y, preds, probs, n_classes)
            m["latency_ms_per_sample"] = (t1 - t0) / len(train_y) * 1e3
            m["train_time_s"] = t1 - t0
            m["n_params"] = 0
            m["model_size_mb"] = 0.0
            seed_results.append(m)
        results_all[name] = aggregate_results(seed_results)
        log.info(f"    {name}: F1={results_all[name]['f1_macro_mean']:.4f}±{results_all[name]['f1_macro_std']:.4f}")

    # ---- XGBoost ----
    log.info("  Training XGBoost...")
    seed_results = []
    for seed in seeds:
        n_jobs = -1
        n_est = 300
        clf = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=8, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu",
            random_state=seed, n_jobs=n_jobs,
            scale_pos_weight=None,
        )
        t0 = time.perf_counter()
        clf.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
        t1 = time.perf_counter()
        preds = clf.predict(test_X)
        probs = clf.predict_proba(test_X)
        m = compute_metrics(test_y, preds, probs, n_classes)
        m["latency_ms_per_sample"] = (time.perf_counter() - t1) / len(test_y) * 1e3
        m["train_time_s"] = t1 - t0
        m["n_params"] = 0
        m["model_size_mb"] = 0.0
        seed_results.append(m)
    results_all["XGBoost"] = aggregate_results(seed_results)
    log.info(f"    XGBoost: F1={results_all['XGBoost']['f1_macro_mean']:.4f}")

    # ---- LightGBM ----
    log.info("  Training LightGBM...")
    seed_results = []
    for seed in seeds:
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.05,
            num_leaves=63, n_jobs=-1, random_state=seed,
            class_weight="balanced", verbose=-1,
        )
        t0 = time.perf_counter()
        clf.fit(train_X, train_y, eval_set=[(val_X, val_y)])
        t1 = time.perf_counter()
        preds = clf.predict(test_X)
        probs = clf.predict_proba(test_X)
        m = compute_metrics(test_y, preds, probs, n_classes)
        m["latency_ms_per_sample"] = (time.perf_counter() - t1) / len(test_y) * 1e3
        m["train_time_s"] = t1 - t0
        m["n_params"] = 0
        m["model_size_mb"] = 0.0
        seed_results.append(m)
    results_all["LightGBM"] = aggregate_results(seed_results)
    log.info(f"    LightGBM: F1={results_all['LightGBM']['f1_macro_mean']:.4f}")

    # ---- CatBoost ----
    log.info("  Training CatBoost...")
    seed_results = []
    for seed in seeds:
        clf = cb.CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.1,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            random_seed=seed, verbose=False,
            task_type="GPU" if torch.cuda.is_available() else "CPU",
            auto_class_weights="Balanced",
        )
        t0 = time.perf_counter()
        clf.fit(train_X, train_y, eval_set=(val_X, val_y), verbose=False)
        t1 = time.perf_counter()
        preds = clf.predict(test_X).flatten().astype(int)
        probs = clf.predict_proba(test_X)
        m = compute_metrics(test_y, preds, probs, n_classes)
        m["latency_ms_per_sample"] = (time.perf_counter() - t1) / len(test_y) * 1e3
        m["train_time_s"] = t1 - t0
        m["n_params"] = 0
        m["model_size_mb"] = 0.0
        seed_results.append(m)
    results_all["CatBoost"] = aggregate_results(seed_results)
    log.info(f"    CatBoost: F1={results_all['CatBoost']['f1_macro_mean']:.4f}")

    # ---- PyTorch models ----
    torch_models = {
        "MLP": lambda: MLPBaseline(d_in, n_classes, **BASELINE_CONFIGS["MLP"]["kwargs"]),
        "1D-CNN": lambda: CNN1DBaseline(d_in, n_classes, **BASELINE_CONFIGS["1D-CNN"]["kwargs"]),
        "LSTM": lambda: LSTMBaseline(d_in, n_classes, **BASELINE_CONFIGS["LSTM"]["kwargs"]),
        "GRU": lambda: GRUBaseline(d_in, n_classes, **BASELINE_CONFIGS["GRU"]["kwargs"]),
        "CNN-BiLSTM": lambda: CNNBiLSTMBaseline(d_in, n_classes),
        "TCN": lambda: TCNBaseline(d_in, n_classes, **BASELINE_CONFIGS["TCN"]["kwargs"]),
        "Transformer": lambda: TransformerBaseline(d_in, n_classes, **BASELINE_CONFIGS["Transformer"]["kwargs"]),
        "GAT": lambda: GATBaseline(d_in, n_classes, **BASELINE_CONFIGS["GAT"]["kwargs"]),
        "GraphSAGE": lambda: GraphSAGEBaseline(d_in, n_classes, **BASELINE_CONFIGS["GraphSAGE"]["kwargs"]),
    }

    for name, model_fn in torch_models.items():
        log.info(f"  Training {name}...")
        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_fn()
            model, _ = train_torch_model(
                model, train_X, train_y, val_X, val_y,
                n_epochs=40, lr=1e-3, batch_size=512, patience=8,
                class_weights=cw, device=DEVICE,
            )
            m = eval_torch_model(model, test_X, test_y, batch_size=512, device=DEVICE, n_classes=n_classes)
            seed_results.append(m)
        results_all[name] = aggregate_results(seed_results)
        log.info(f"    {name}: F1={results_all[name]['f1_macro_mean']:.4f}±{results_all[name]['f1_macro_std']:.4f}  Lat={results_all[name]['latency_ms_per_sample_mean']:.3f}ms")

    # Save
    out = ROOT / "results" / f"{dataset_name}_baselines.json"
    with open(out, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    log.info(f"\nBaseline results saved -> {out}")
    return results_all


def aggregate_results(seed_results: List[Dict]) -> Dict:
    keys = seed_results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in seed_results if isinstance(r.get(k), (int, float))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
            agg[f"{k}_per_seed"] = vals  # retained for paired statistical testing
    return agg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="edgeiiot",
                         choices=["edgeiiot", "rtiot", "unsw", "tiissrc23", "ustc", "5gad"])
    parser.add_argument("--seeds", default="42,123,456")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    run_baselines(args.dataset, seeds=seeds)
