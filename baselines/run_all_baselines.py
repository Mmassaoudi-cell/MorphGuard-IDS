"""
Run all baselines across all three datasets with incremental saving.
Designed to be resumable if interrupted.
"""

import os, sys, json, time, warnings, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "all_baselines.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---- Metrics ----

def metrics(y_true, y_pred, y_prob=None, n_cls=None):
    acc = accuracy_score(y_true, y_pred)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    unique = np.unique(y_true)
    fpr_sum = fnr_sum = 0.0
    for c in unique:
        yb = (np.array(y_true) == c).astype(int)
        yp = (np.array(y_pred) == c).astype(int)
        fp = ((yb == 0) & (yp == 1)).sum(); tn = ((yb == 0) & (yp == 0)).sum()
        fn = ((yb == 1) & (yp == 0)).sum(); tp = ((yb == 1) & (yp == 1)).sum()
        fpr_sum += fp / max(fp + tn, 1); fnr_sum += fn / max(fn + tp, 1)
    n = max(len(unique), 1)
    m = {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_w,
         "fpr": fpr_sum / n, "fnr": fnr_sum / n}
    if y_prob is not None:
        try:
            if n_cls and n_cls > 2:
                m["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            else:
                col = y_prob[:, 1] if (y_prob.ndim > 1 and y_prob.shape[1] == 2) else (y_prob[:, 0] if y_prob.ndim > 1 else y_prob)
                m["roc_auc"] = roc_auc_score(y_true, col, multi_class="ovr", average="macro")
        except:
            m["roc_auc"] = 0.0
    return m


def agg(rs):
    out = {}
    for k in rs[0]:
        vals = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    return out


# ---- PyTorch models ----

class MLP(nn.Module):
    def __init__(self, d, nc, h=256, nl=3, dr=0.2):
        super().__init__()
        ls = [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dr)]
        for _ in range(nl - 1):
            ls += [nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dr)]
        ls.append(nn.Linear(h, nc))
        self.net = nn.Sequential(*ls)
    def forward(self, x): return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, d, nc, h=128, nl=3, dr=0.2):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.convs = nn.ModuleList([nn.Conv1d(h, h, 3, padding=2**i, dilation=2**i) for i in range(nl)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(nl)])
        self.drop = nn.Dropout(dr)
        self.fc = nn.Linear(h, nc)
    def forward(self, x):
        h = self.proj(x).unsqueeze(2)
        for c, n in zip(self.convs, self.norms):
            r = c(h)[..., :1]
            r = n(r.squeeze(2)).unsqueeze(2)
            h = torch.relu(h + r)
        return self.fc(self.drop(h.squeeze(2)))


class LSTM(nn.Module):
    def __init__(self, d, nc, h=128, nl=2, dr=0.2, bi=False):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.rnn = nn.LSTM(h, h, nl, batch_first=True, dropout=dr if nl > 1 else 0, bidirectional=bi)
        self.fc = nn.Linear(h * (2 if bi else 1), nc)
        self.drop = nn.Dropout(dr)
    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        o, _ = self.rnn(h)
        return self.fc(self.drop(o[:, -1, :]))


class CNNBiLSTM(nn.Module):
    def __init__(self, d, nc, h=128, dr=0.2):
        super().__init__()
        self.cnn = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dr))
        self.rnn = nn.LSTM(h, h // 2, 2, batch_first=True, bidirectional=True, dropout=dr)
        self.fc = nn.Linear(h, nc)
        self.drop = nn.Dropout(dr)
    def forward(self, x):
        h = self.cnn(x).unsqueeze(1)
        o, _ = self.rnn(h)
        return self.fc(self.drop(o[:, -1, :]))


class TCN(nn.Module):
    def __init__(self, d, nc, h=128, nl=4, dr=0.2):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.convs = nn.ModuleList([nn.Conv1d(h, h, 3, padding=2**i * 2, dilation=2**i) for i in range(nl)])
        self.drop = nn.Dropout(dr)
        self.fc = nn.Linear(h, nc)
    def forward(self, x):
        h = self.proj(x).unsqueeze(2)
        for c in self.convs:
            r = torch.relu(c(h)[..., :1])
            h = h + self.drop(r)
        return self.fc(h.squeeze(2))


class TransformerCls(nn.Module):
    def __init__(self, d, nc, h=128, nh=4, nl=2, dr=0.2):
        super().__init__()
        self.proj = nn.Linear(d, h)
        enc = nn.TransformerEncoderLayer(h, nh, dim_feedforward=256, dropout=dr, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=nl)
        self.fc = nn.Linear(h, nc)
        self.norm = nn.LayerNorm(h)
    def forward(self, x):
        h = self.enc(self.proj(x).unsqueeze(1))
        return self.fc(self.norm(h[:, 0]))


class GAT(nn.Module):
    """Simple GAT approximation using multi-head self-attention over chunked features."""
    def __init__(self, d, nc, h=64, nh=4, nl=2, dr=0.2):
        super().__init__()
        n_chunks = 4
        chunk = max(d // n_chunks, 1)
        self.proj = nn.Linear(chunk, h)
        self.attn = nn.MultiheadAttention(h, nh, dropout=dr, batch_first=True)
        self.fc = nn.Linear(h, nc)
        self.norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(dr)
        self.chunk = chunk
        self.n_chunks = n_chunks
        self.d = d
    def forward(self, x):
        chunks = []
        for i in range(self.n_chunks):
            s = i * self.chunk
            e = min(s + self.chunk, self.d)
            part = x[:, s:e]
            if part.shape[1] < self.chunk:
                part = torch.nn.functional.pad(part, (0, self.chunk - part.shape[1]))
            chunks.append(self.proj(part))
        nodes = torch.stack(chunks, dim=1)  # (B, n_chunks, h)
        out, _ = self.attn(nodes, nodes, nodes)
        out = self.norm(out + nodes).mean(1)
        return self.fc(self.drop(out))


def train_nn(model, Xtr, ytr, Xvl, yvl, ep=30, lr=1e-3, bs=512, pat=7, cw=None, dev=DEVICE):
    model.to(dev)
    ds_tr = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    ds_vl = TensorDataset(torch.tensor(Xvl, dtype=torch.float32), torch.tensor(yvl, dtype=torch.long))
    dl_tr = DataLoader(ds_tr, bs, shuffle=True, pin_memory=True)
    dl_vl = DataLoader(ds_vl, bs, shuffle=False, pin_memory=True)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(dev) if cw is not None else None)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    best_f1, best_st, pat_cnt = -1.0, None, 0
    for epoch in range(ep):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in dl_vl:
                preds.extend(model(xb.to(dev)).argmax(1).cpu().tolist())
        vf1 = f1_score(yvl, preds, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= pat: break
    if best_st: model.load_state_dict(best_st)
    return model


def eval_nn(model, Xte, yte, bs=512, dev=DEVICE, n_cls=None):
    model.eval().to(dev)
    ds = TensorDataset(torch.tensor(Xte, dtype=torch.float32))
    dl = DataLoader(ds, bs, pin_memory=True)
    probs, lats = [], []
    with torch.no_grad():
        for (xb,) in dl:
            t0 = time.perf_counter()
            p = torch.softmax(model(xb.to(dev)), -1).cpu().numpy()
            lats.append((time.perf_counter() - t0) / xb.shape[0] * 1e3)
            probs.append(p)
    probs = np.vstack(probs)
    preds = probs.argmax(1)
    m = metrics(yte, preds, probs, n_cls)
    m["latency_ms"] = float(np.mean(lats))
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    m["n_params"] = n_p
    m["size_mb"] = n_p * 4 / 1e6
    return m


# ---- Main ----

def run_dataset(dsname):
    pdir = ROOT / "results" / "preprocessed"
    mdir = ROOT / "results" / "metadata"
    rdir = ROOT / "results"

    # Resume check
    out_path = rdir / f"{dsname}_baselines.json"
    if out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
    else:
        saved = {}

    tr = pd.read_parquet(pdir / f"{dsname}_train.parquet")
    vl = pd.read_parquet(pdir / f"{dsname}_val.parquet")
    te = pd.read_parquet(pdir / f"{dsname}_test.parquet")
    with open(mdir / f"{dsname}_meta.json") as f:
        meta = json.load(f)
    fc = meta["feature_cols"]
    nc = meta["n_classes_multi"]
    d = len(fc)
    Xtr, ytr = tr[fc].values, tr["multi_label"].values
    Xvl, yvl = vl[fc].values, vl["multi_label"].values
    Xte, yte = te[fc].values, te["multi_label"].values
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)

    log.info(f"\n{'='*60}\n{dsname}: n={len(Xtr)} d={d} nc={nc}")

    def save():
        with open(out_path, "w") as f:
            json.dump(saved, f, indent=2, default=str)

    # --- Logistic Regression ---
    if "LogisticRegression" not in saved:
        log.info("  LogisticRegression...")
        rs = []
        for seed in SEEDS:
            np.random.seed(seed)
            clf = LogisticRegression(max_iter=300, solver="lbfgs", C=1.0, n_jobs=-1, random_state=seed)
            t0 = time.time()
            clf.fit(Xtr, ytr)
            t1 = time.time()
            preds = clf.predict(Xte)
            probs = clf.predict_proba(Xte)
            m = metrics(yte, preds, probs, nc)
            m["latency_ms"] = (time.time() - t1) / len(Xte) * 1e3
            m["n_params"] = 0; m["size_mb"] = 0.0; m["train_time_s"] = t1 - t0
            rs.append(m)
        saved["LogisticRegression"] = agg(rs)
        log.info(f"    LR: F1={saved['LogisticRegression']['f1_macro_mean']:.4f}")
        save()

    # --- Random Forest ---
    if "RandomForest" not in saved:
        log.info("  RandomForest...")
        rs = []
        for seed in SEEDS:
            np.random.seed(seed)
            clf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1,
                                         class_weight="balanced", random_state=seed)
            t0 = time.time()
            clf.fit(Xtr, ytr)
            t1 = time.time()
            preds = clf.predict(Xte)
            probs = clf.predict_proba(Xte)
            m = metrics(yte, preds, probs, nc)
            m["latency_ms"] = (time.time() - t1) / len(Xte) * 1e3
            m["n_params"] = 0; m["size_mb"] = 0.0; m["train_time_s"] = t1 - t0
            rs.append(m)
        saved["RandomForest"] = agg(rs)
        log.info(f"    RF: F1={saved['RandomForest']['f1_macro_mean']:.4f}")
        save()

    # --- XGBoost ---
    if "XGBoost" not in saved:
        log.info("  XGBoost...")
        rs = []
        for seed in SEEDS:
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.1,
                eval_metric="mlogloss", tree_method="hist",
                device="cuda" if torch.cuda.is_available() else "cpu",
                random_state=seed, verbosity=0,
            )
            t0 = time.time()
            clf.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
            t1 = time.time()
            preds = clf.predict(Xte)
            probs = clf.predict_proba(Xte)
            m = metrics(yte, preds, probs, nc)
            m["latency_ms"] = (time.time() - t1) / len(Xte) * 1e3
            m["n_params"] = 0; m["size_mb"] = 0.0; m["train_time_s"] = t1 - t0
            rs.append(m)
        saved["XGBoost"] = agg(rs)
        log.info(f"    XGB: F1={saved['XGBoost']['f1_macro_mean']:.4f}")
        save()

    # --- LightGBM ---
    if "LightGBM" not in saved:
        log.info("  LightGBM...")
        rs = []
        for seed in SEEDS:
            clf = lgb.LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.05,
                                      num_leaves=63, n_jobs=-1, random_state=seed,
                                      class_weight="balanced", verbose=-1)
            t0 = time.time()
            clf.fit(Xtr, ytr, eval_set=[(Xvl, yvl)])
            t1 = time.time()
            preds = clf.predict(Xte)
            probs = clf.predict_proba(Xte)
            m = metrics(yte, preds, probs, nc)
            m["latency_ms"] = (time.time() - t1) / len(Xte) * 1e3
            m["n_params"] = 0; m["size_mb"] = 0.0; m["train_time_s"] = t1 - t0
            rs.append(m)
        saved["LightGBM"] = agg(rs)
        log.info(f"    LGB: F1={saved['LightGBM']['f1_macro_mean']:.4f}")
        save()

    # --- PyTorch NN models ---
    nn_models = {
        "MLP":         lambda: MLP(d, nc),
        "1D-CNN":      lambda: CNN1D(d, nc),
        "LSTM":        lambda: LSTM(d, nc),
        "CNN-BiLSTM":  lambda: CNNBiLSTM(d, nc),
        "TCN":         lambda: TCN(d, nc),
        "Transformer": lambda: TransformerCls(d, nc),
        "GAT":         lambda: GAT(d, nc),
    }

    for name, mfn in nn_models.items():
        if name in saved:
            log.info(f"  {name}: already done (F1={saved[name].get('f1_macro_mean', '?'):.4f})")
            continue
        log.info(f"  {name}...")
        rs = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            model = mfn()
            model = train_nn(model, Xtr, ytr, Xvl, yvl, ep=30, lr=1e-3, bs=512, pat=7, cw=cw)
            m = eval_nn(model, Xte, yte, n_cls=nc)
            rs.append(m)
            del model; torch.cuda.empty_cache()
        saved[name] = agg(rs)
        log.info(f"    {name}: F1={saved[name]['f1_macro_mean']:.4f}±{saved[name]['f1_macro_std']:.4f}  Lat={saved[name]['latency_ms_mean']:.3f}ms")
        save()

    log.info(f"\nAll baselines done for {dsname}")
    return saved


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="edgeiiot,rtiot,unsw")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        run_dataset(ds.strip())
    log.info("\n=== ALL DATASETS COMPLETE ===")
