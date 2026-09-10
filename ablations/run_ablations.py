"""
MorphGuard-IDS Ablation Study.
Systematically removes components and measures impact.
"""

import os, sys, json, logging, warnings
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "ablations.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

from models.morphguard import (
    MorphGuardTeacher, MorphGuardStudent, FocalLoss,
    MultiScaleTCN, TemperatureScaler
)
from models.hypergraph import build_hypergraph_batch, HypergraphEncoder
from training.train_morphguard import (
    IDSDataset, make_loader, eval_model, train_teacher, distill_student, count_params, model_size_mb
)


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

class AblationNoHypergraph(nn.Module):
    """Without hypergraph: only TCN + MLP classifier."""
    def __init__(self, d_flow, n_classes, d_model=128, dropout=0.15):
        super().__init__()
        self.tcn = MultiScaleTCN(d_flow, d_model, n_layers=4, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        self.temp_scaler = TemperatureScaler()

    def encode(self, flow_feats, hg=None):
        return self.tcn(flow_feats)

    def forward(self, flow_feats, hg=None, labels=None, augment=False):
        emb = self.encode(flow_feats)
        logits = self.classifier(emb)
        cal_logits = self.temp_scaler(logits)
        return {"logits": logits, "cal_logits": cal_logits, "embeddings": emb}


class AblationNoSSL(MorphGuardTeacher):
    """Without self-supervised pretraining (will be trained from scratch supervised)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Same architecture, but skips SSL pretraining in runner
        self._skip_ssl = True


class AblationNoTCN(nn.Module):
    """Without temporal encoder: only HGNN."""
    def __init__(self, d_flow, n_classes, d_model=128, n_heads=4, dropout=0.15):
        super().__init__()
        self.hgnn = HypergraphEncoder(d_flow=d_flow, d_model=d_model, n_layers=2, n_heads=n_heads, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        self.temp_scaler = TemperatureScaler()

    def encode(self, flow_feats, hg):
        return self.hgnn(hg)

    def forward(self, flow_feats, hg, labels=None, augment=False):
        emb = self.encode(flow_feats, hg)
        logits = self.classifier(emb)
        cal_logits = self.temp_scaler(logits)
        return {"logits": logits, "cal_logits": cal_logits, "embeddings": emb}


class AblationNoCausal(MorphGuardTeacher):
    """Without causal-invariant regularization (alpha_causal=0)."""
    def __init__(self, *args, **kwargs):
        kwargs["n_envs"] = 1  # disables causal penalty
        super().__init__(*args, **kwargs)


class AblationNoAugmentation(MorphGuardTeacher):
    """Without counterfactual augmentation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_augment = True

    def forward(self, flow_feats, hg, labels=None, augment=False):
        # Override to disable augmentation
        return super().forward(flow_feats, hg, labels, augment=False)


class AblationNoCalibration(MorphGuardTeacher):
    """Without calibration (returns raw logits)."""
    def forward(self, flow_feats, hg, labels=None, augment=False):
        out = super().forward(flow_feats, hg, labels, augment)
        out["cal_logits"] = out["logits"]  # use raw logits
        return out


class AblationNoDistillation(MorphGuardStudent):
    """Student trained from scratch, without teacher KD."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_distill = True


# ---------------------------------------------------------------------------
# Train ablation model
# ---------------------------------------------------------------------------

def train_ablation_model(
    model, train_loader, val_loader, n_classes, device=DEVICE,
    n_epochs=40, lr=5e-4, class_weights=None, patience=8,
    is_teacher_like=True,
) -> dict:
    model.to(device)
    if class_weights is not None:
        cw = class_weights.to(device)
        criterion = FocalLoss(gamma=2.0, weight=cw)
    else:
        criterion = FocalLoss(gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_f1 = -1.0
    best_state = None
    patience_cnt = 0

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)

            if is_teacher_like:
                hg = build_hypergraph_batch(feats, device=device)
                out = model(feats, hg, labels)
                logits = out["logits"]
            else:
                out = model(feats)
                logits = out["logits"]

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Val check
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device)
                if is_teacher_like:
                    hg = build_hypergraph_batch(feats, device=device)
                    out = model(feats, hg)
                    logits = out["cal_logits"]
                else:
                    out = model(feats)
                    logits = out["cal_logits"]
                preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        val_y = []
        for batch in val_loader:
            val_y.extend(batch["labels"].numpy().tolist())
        vf1 = f1_score(val_y, preds, average="macro", zero_division=0)

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
    return {"best_val_f1": best_f1}


def eval_ablation(model, test_loader, device=DEVICE, is_teacher_like=True, n_classes=None):
    import time
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    latencies = []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            labels = batch["labels"]
            t0 = time.perf_counter()
            if is_teacher_like:
                hg = build_hypergraph_batch(feats, device=device)
                out = model(feats, hg)
            else:
                out = model(feats)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / feats.shape[0] * 1e3)
            logits = out["cal_logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_probs.append(probs)
            all_labels.extend(labels.numpy().tolist())

    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    # ECE computation
    all_probs_np = np.vstack(all_probs)
    confidence = all_probs_np.max(axis=1)
    correct = (np.array(all_preds) == np.array(all_labels)).astype(float)
    n_bins = 10
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() > 0:
            acc_bin = correct[mask].mean()
            conf_bin = confidence[mask].mean()
            ece += mask.sum() / len(all_labels) * abs(acc_bin - conf_bin)

    return {
        "f1_macro": f1,
        "accuracy": acc,
        "recall_macro": rec,
        "ece": ece,
        "latency_ms": float(np.mean(latencies)),
        "n_params": count_params(model),
    }


# ---------------------------------------------------------------------------
# Run all ablations
# ---------------------------------------------------------------------------

def run_ablations(dataset_name: str, seeds: List = [42, 123, 456]):
    import json
    preproc = ROOT / "results" / "preprocessed"
    meta_dir = ROOT / "results" / "metadata"
    capped_path = preproc / f"{dataset_name}_train_capped.parquet"
    train_path = capped_path if capped_path.exists() else preproc / f"{dataset_name}_train.parquet"
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(preproc / f"{dataset_name}_val.parquet")
    test_df = pd.read_parquet(preproc / f"{dataset_name}_test.parquet")
    if train_path == capped_path:
        log.info(f"  Using capped training sample ({len(train_df)} rows) for {dataset_name}.")
    with open(meta_dir / f"{dataset_name}_meta.json") as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]
    n_classes = meta["n_classes_multi"]
    d_flow = len(feat_cols)

    cw = compute_class_weight("balanced", classes=np.unique(train_df["multi_label"].values), y=train_df["multi_label"].values)
    cw_tensor = torch.tensor(cw, dtype=torch.float32)

    train_loader = make_loader(train_df, feat_cols, "multi_label", batch_size=512, shuffle=True)
    val_loader = make_loader(val_df, feat_cols, "multi_label", batch_size=512, shuffle=False)
    test_loader = make_loader(test_df, feat_cols, "multi_label", batch_size=512, shuffle=False)

    # Define ablation variants
    ablation_variants = {
        "Full MorphGuard-IDS": {
            "factory": lambda: MorphGuardTeacher(d_flow, n_classes, d_model=128, n_tcn_layers=4, n_hgnn_layers=2),
            "teacher_like": True,
            "use_ssl": True,
        },
        "w/o Hypergraph": {
            "factory": lambda: AblationNoHypergraph(d_flow, n_classes),
            "teacher_like": True,  # still takes hg arg (ignored internally)
            "use_ssl": False,
        },
        "w/o SSL Pretrain": {
            "factory": lambda: MorphGuardTeacher(d_flow, n_classes, d_model=128, n_tcn_layers=4, n_hgnn_layers=2),
            "teacher_like": True,
            "use_ssl": False,
        },
        "w/o Temporal Encoder": {
            "factory": lambda: AblationNoTCN(d_flow, n_classes),
            "teacher_like": True,
            "use_ssl": False,
        },
        "w/o Causal Regularization": {
            "factory": lambda: AblationNoCausal(d_flow, n_classes, d_model=128, n_tcn_layers=4, n_hgnn_layers=2),
            "teacher_like": True,
            "use_ssl": False,
        },
        "w/o Augmentation": {
            "factory": lambda: AblationNoAugmentation(d_flow, n_classes, d_model=128, n_tcn_layers=4, n_hgnn_layers=2),
            "teacher_like": True,
            "use_ssl": False,
        },
        "w/o Calibration": {
            "factory": lambda: AblationNoCalibration(d_flow, n_classes, d_model=128, n_tcn_layers=4, n_hgnn_layers=2),
            "teacher_like": True,
            "use_ssl": False,
        },
        "w/o Distillation (Student)": {
            "factory": lambda: MorphGuardStudent(d_flow, n_classes, d_model=64, n_layers=2),
            "teacher_like": False,
            "use_ssl": False,
        },
    }

    ablation_results = {}

    for variant_name, cfg in ablation_variants.items():
        log.info(f"\nAblation: {variant_name}")
        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = cfg["factory"]().to(DEVICE)

            if cfg["use_ssl"] and hasattr(model, "ssl_pretrain_loss"):
                # Quick SSL pretraining (5 epochs)
                from training.train_morphguard import ssl_pretrain
                ssl_pretrain(model, train_loader, n_epochs=5, device=DEVICE)

            is_tl = cfg["teacher_like"]
            train_ablation_model(
                model, train_loader, val_loader, n_classes,
                device=DEVICE, n_epochs=30, lr=5e-4,
                class_weights=cw_tensor, patience=8,
                is_teacher_like=is_tl,
            )

            res = eval_ablation(model, test_loader, device=DEVICE,
                               is_teacher_like=is_tl, n_classes=n_classes)
            seed_results.append(res)

        # Aggregate
        keys = seed_results[0].keys()
        agg = {}
        for k in keys:
            vals = [r[k] for r in seed_results if isinstance(r[k], (int, float))]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
        ablation_results[variant_name] = agg
        log.info(f"  F1={agg['f1_macro_mean']:.4f}±{agg['f1_macro_std']:.4f}  ECE={agg['ece_mean']:.4f}  Lat={agg['latency_ms_mean']:.3f}ms")

    out = ROOT / "results" / f"{dataset_name}_ablations.json"
    with open(out, "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    log.info(f"\nAblation results saved -> {out}")
    return ablation_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="edgeiiot",
                         choices=["edgeiiot", "rtiot", "unsw", "tiissrc23", "ustc", "5gad"])
    parser.add_argument("--seeds", default="42,123,456")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    run_ablations(args.dataset, seeds=seeds)
