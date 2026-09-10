"""
Cross-dataset generalization experiments.
Train on one dataset, evaluate on another using aligned features.
"""

import sys, json, logging, warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "cross_dataset.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def get_common_features(ds1: str, ds2: str):
    """Find features common to two datasets (by keyword matching)."""
    mdir = ROOT / "results" / "metadata"
    with open(mdir / f"{ds1}_meta.json") as f:
        m1 = json.load(f)
    with open(mdir / f"{ds2}_meta.json") as f:
        m2 = json.load(f)
    # Find num_cols in both
    nums1 = set(m1.get("num_cols", []))
    nums2 = set(m2.get("num_cols", []))
    common = nums1 & nums2
    log.info(f"Common features between {ds1} and {ds2}: {len(common)}")
    return sorted(common)


def run_cross_dataset(source: str, target: str, seeds=[42, 123, 456]):
    """Train on source, evaluate on target using common features."""
    pdir = ROOT / "results" / "preprocessed"
    mdir = ROOT / "results" / "metadata"

    common_feats = get_common_features(source, target)
    if len(common_feats) < 5:
        log.warning(f"Too few common features ({len(common_feats)}) between {source} and {target}. Skipping.")
        return None

    capped_path = pdir / f"{source}_train_capped.parquet"
    src_train_path = capped_path if capped_path.exists() else pdir / f"{source}_train.parquet"
    src_train = pd.read_parquet(src_train_path)
    tgt_test = pd.read_parquet(pdir / f"{target}_test.parquet")  # always full natural-prevalence

    with open(mdir / f"{source}_meta.json") as f:
        src_meta = json.load(f)
    with open(mdir / f"{target}_meta.json") as f:
        tgt_meta = json.load(f)

    # Align binary labels for cross-dataset binary eval
    Xtr = src_train[common_feats].values
    ytr_bin = src_train["binary_label"].values
    Xte = tgt_test[common_feats].values
    yte_bin = tgt_test["binary_label"].values

    # Refit scaler on source train (features already scaled per dataset)
    # Since features are already StandardScaled per-dataset, we need to undo and redo
    # For simplicity, we treat the already-scaled features as raw and re-scale on common features
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)

    from torch.utils.data import TensorDataset, DataLoader

    # Simple MLP for cross-dataset baseline
    import torch.nn as nn
    import torch.optim as optim
    d = len(common_feats)

    results = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = nn.Sequential(
            nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 2)
        ).to(DEVICE)

        Xt = torch.tensor(Xtr_scaled, dtype=torch.float32)
        yt = torch.tensor(ytr_bin, dtype=torch.long)
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.unique(ytr_bin), y=ytr_bin)
        cw_t = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw_t)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        dl = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True, pin_memory=True)
        for epoch in range(20):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                criterion(model(xb), yb).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            Xte_t = torch.tensor(Xte_scaled, dtype=torch.float32).to(DEVICE)
            preds = model(Xte_t).argmax(1).cpu().numpy()

        f1 = f1_score(yte_bin, preds, average="macro", zero_division=0)
        acc = accuracy_score(yte_bin, preds)
        results.append({"f1_macro": f1, "accuracy": acc})
        log.info(f"  {source}->{target} seed={seed}: F1={f1:.4f}")

    agg = {
        "f1_macro_mean": float(np.mean([r["f1_macro"] for r in results])),
        "f1_macro_std": float(np.std([r["f1_macro"] for r in results])),
        "accuracy_mean": float(np.mean([r["accuracy"] for r in results])),
        "common_feats": len(common_feats),
        "source": source, "target": target,
    }
    log.info(f"  {source}->{target}: F1={agg['f1_macro_mean']:.4f}+-{agg['f1_macro_std']:.4f} (n_feats={len(common_feats)})")
    return agg


def run_all_cross_dataset(seeds=[42, 123, 456]):
    # Feature-name overlap audit (computed once, see logs/cross_dataset.log):
    #   tiissrc23 <-> 5gad: 38 common features (both CICFlowMeter-style)
    #   tiissrc23 <-> ustc: 38 common features
    #   5gad      <-> ustc: 38 common features
    #   rtiot     <-> *    : 0 common features (Zeek-style column names,
    #                        e.g. "id.orig_p"/"proto", vs. CICFlowMeter-style
    #                        "Src Port"/"Protocol" elsewhere). RT-IoT2022 is
    #                        therefore EXCLUDED from name-matched transfer --
    #                        this is reported as a real limitation, not
    #                        silently skipped: even nominally "flow-ready"
    #                        IDS benchmarks do not share a common schema
    #                        without an explicit alignment layer.
    pairs = [
        ("tiissrc23", "5gad"), ("5gad", "tiissrc23"),
        ("tiissrc23", "ustc"), ("ustc", "tiissrc23"),
        ("5gad", "ustc"), ("ustc", "5gad"),
    ]
    results = {}
    for src, tgt in pairs:
        key = f"{src}_{tgt}"
        r = run_cross_dataset(src, tgt, seeds)
        if r:
            results[key] = r

    results["_rtiot_excluded_reason"] = (
        "RT-IoT2022 shares 0 common feature names with tiissrc23/5gad/ustc "
        "(Zeek-style column names vs. CICFlowMeter-style elsewhere); "
        "name-matched transfer is not meaningful without an explicit "
        "schema-alignment layer, which was not implemented."
    )

    out = ROOT / "results" / "cross_dataset_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Cross-dataset results saved: {out}")
    return results


if __name__ == "__main__":
    run_all_cross_dataset()
