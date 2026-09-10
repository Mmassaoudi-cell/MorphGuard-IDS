"""
MorphGuard-IDS Training Script.
Trains teacher model with optional SSL pretraining, then distills to student.
Reports mean ± std over multiple seeds.
"""

import os, sys, json, time, logging, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    f1_score, confusion_matrix, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.hypergraph import build_hypergraph_batch, HypergraphBatch
from models.morphguard import (
    MorphGuardTeacher, MorphGuardStudent, MorphGuardLoss,
    RelationalKDLoss, FocalLoss, TemperatureScaler
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "morphguard_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class IDSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], label_col: str = "multi_label"):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.long)
        if "binary_label" in df.columns:
            self.binary_labels = torch.tensor(df["binary_label"].values, dtype=torch.long)
        else:
            self.binary_labels = (self.labels > 0).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "binary_label": self.binary_labels[idx],
        }


def make_loader(df, feature_cols, label_col="multi_label", batch_size=512, shuffle=True, num_workers=0):
    ds = IDSDataset(df, feature_cols, label_col)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None, n_classes=None) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # FPR, FNR (binary)
    unique_classes = np.unique(y_true)
    fpr_macro, fnr_macro = 0.0, 0.0
    for c in unique_classes:
        y_bin = (np.array(y_true) == c).astype(int)
        yp_bin = (np.array(y_pred) == c).astype(int)
        tn = ((y_bin == 0) & (yp_bin == 0)).sum()
        fp = ((y_bin == 0) & (yp_bin == 1)).sum()
        fn = ((y_bin == 1) & (yp_bin == 0)).sum()
        tp = ((y_bin == 1) & (yp_bin == 1)).sum()
        fpr_macro += fp / max(fp + tn, 1)
        fnr_macro += fn / max(fn + tp, 1)
    fpr_macro /= len(unique_classes)
    fnr_macro /= len(unique_classes)

    metrics = {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "f1_weighted": wf1,
        "fpr": fpr_macro,
        "fnr": fnr_macro,
    }

    if y_prob is not None:
        try:
            if n_classes and n_classes > 2:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            else:
                if y_prob.ndim > 1:
                    auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob, multi_class="ovr", average="macro")
                else:
                    auc = roc_auc_score(y_true, y_prob)
            metrics["roc_auc"] = auc
        except Exception:
            metrics["roc_auc"] = 0.0

        # PR-AUC (macro average)
        try:
            if y_prob.ndim > 1 and y_prob.shape[1] > 2:
                pr_auc = 0.0
                for c in range(y_prob.shape[1]):
                    y_c = (np.array(y_true) == c).astype(int)
                    if y_c.sum() > 0:
                        pr_auc += average_precision_score(y_c, y_prob[:, c])
                metrics["pr_auc"] = pr_auc / y_prob.shape[1]
            elif y_prob.ndim > 1:
                metrics["pr_auc"] = average_precision_score(y_true, y_prob[:, 1])
            else:
                metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics["pr_auc"] = 0.0

    return metrics


def eval_model(model, loader, device, is_teacher=True, n_classes=None):
    """Evaluate model on a DataLoader, return metrics + predictions."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    latencies = []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)

            t0 = time.perf_counter()
            if is_teacher:
                hg = build_hypergraph_batch(feats, device=device)
                out = model(feats, hg)
                logits = out["cal_logits"]
            else:
                out = model(feats)
                logits = out["cal_logits"]
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / feats.shape[0] * 1e3)  # ms/sample

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_probs.append(probs)
            all_labels.extend(labels.cpu().numpy().tolist())

    all_probs = np.vstack(all_probs)
    metrics = compute_metrics(all_labels, all_preds, all_probs, n_classes)
    metrics["latency_ms_per_sample"] = float(np.mean(latencies))
    return metrics, all_preds, all_probs, all_labels


# ---------------------------------------------------------------------------
# SSL Pretraining
# ---------------------------------------------------------------------------

def ssl_pretrain(
    teacher: MorphGuardTeacher,
    train_loader,
    n_epochs: int = 10,
    lr: float = 1e-3,
    device: str = DEVICE,
) -> List[float]:
    """Self-supervised pretraining with masked/contrastive loss."""
    teacher.to(device)
    teacher.train()
    optimizer = optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    losses = []

    log.info(f"  SSL pretraining for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            feats = batch["features"].to(device)
            hg = build_hypergraph_batch(feats, device=device)

            ssl_loss = teacher.ssl_pretrain_loss(feats, hg)
            optimizer.zero_grad()
            ssl_loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
            epoch_loss += ssl_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            log.info(f"    SSL Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Supervised Training
# ---------------------------------------------------------------------------

def train_teacher(
    teacher: MorphGuardTeacher,
    train_loader,
    val_loader,
    n_epochs: int = 40,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    n_classes: int = 2,
    device: str = DEVICE,
    patience: int = 8,
    augment: bool = True,
    alpha_causal: float = 0.1,
    focal_gamma: float = 2.0,
    checkpoint_path: Optional[Path] = None,
) -> Dict:
    """Train teacher with focal loss + causal regularization."""
    teacher.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = MorphGuardLoss(
        n_classes=n_classes,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        alpha_causal=alpha_causal,
    )

    optimizer = optim.AdamW(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_f1 = -1.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_f1": [], "val_loss": []}

    log.info(f"  Training teacher for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        teacher.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            hg = build_hypergraph_batch(feats, device=device)

            out = teacher(feats, hg, labels, augment=augment)
            aug_labels = out.get("aug_labels", labels)

            # Recompute logits for augmented labels (augmented embeddings -> classifier)
            logits = out["logits"]
            cal_logits = out["cal_logits"]

            # Trim to match aug_labels if needed
            if aug_labels.shape[0] != logits.shape[0]:
                # Augmenter changed batch size - need forward pass on augmented embeddings
                aug_emb = out["embeddings"]
                logits = teacher.classifier(aug_emb)
                cal_logits = teacher.temp_scaler(logits)

            # Causal penalty
            try:
                causal_penalty = teacher.causal_reg(logits, aug_labels)
            except Exception:
                causal_penalty = torch.tensor(0.0, device=device)

            total_loss, loss_dict = criterion(
                logits, cal_logits, aug_labels, causal_penalty,
                list(teacher.parameters()),
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation
        val_metrics, _, _, _ = eval_model(teacher, val_loader, device, is_teacher=True, n_classes=n_classes)
        val_f1 = val_metrics["f1_macro"]
        history["val_f1"].append(val_f1)
        history["val_loss"].append(val_metrics.get("fpr", 0.0))

        if (epoch + 1) % 5 == 0:
            log.info(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}  val_f1={val_f1:.4f}  fpr={val_metrics['fpr']:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
            patience_counter = 0
            if checkpoint_path:
                try:
                    tmp = Path(str(checkpoint_path) + ".tmp")
                    torch.save(best_state, tmp)
                    tmp.replace(checkpoint_path)
                except Exception as e:
                    log.warning(f"Checkpoint save failed: {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        teacher.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------------------
# Student Distillation
# ---------------------------------------------------------------------------

def distill_student(
    teacher: MorphGuardTeacher,
    student: MorphGuardStudent,
    train_loader,
    val_loader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    temperature: float = 4.0,
    n_classes: int = 2,
    device: str = DEVICE,
    patience: int = 6,
    checkpoint_path: Optional[Path] = None,
) -> Dict:
    """Train student via relational knowledge distillation."""
    teacher.to(device).eval()
    student.to(device)

    kd_loss_fn = RelationalKDLoss(temperature=temperature, lambda_z=0.5, lambda_r=0.1)
    cls_loss_fn = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_f1 = -1.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_f1": []}

    log.info(f"  Distilling student for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        student.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            hg = build_hypergraph_batch(feats, device=device)

            with torch.no_grad():
                teacher_out = teacher(feats, hg)
                t_logits = teacher_out["logits"].detach()
                t_emb = teacher_out["embeddings"].detach()

            student_out = student(feats)
            s_logits = student_out["logits"]
            s_emb = student_out["embeddings"]

            # KD loss
            kd_loss, _ = kd_loss_fn(s_logits, t_logits, s_emb, t_emb)
            # Classification loss on true labels
            cls_loss = cls_loss_fn(s_logits, labels)
            total = 0.5 * cls_loss + 0.5 * kd_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        val_metrics, _, _, _ = eval_model(student, val_loader, device, is_teacher=False, n_classes=n_classes)
        val_f1 = val_metrics["f1_macro"]
        history["val_f1"].append(val_f1)

        if (epoch + 1) % 5 == 0:
            log.info(f"    Student Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
            if checkpoint_path:
                try:
                    tmp = Path(str(checkpoint_path) + ".tmp")
                    torch.save(best_state, tmp)
                    tmp.replace(checkpoint_path)
                except Exception as e:
                    log.warning(f"Student checkpoint save failed: {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"    Student early stopping at epoch {epoch+1}")
                break

    if best_state:
        student.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------------------
# Count model parameters
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1e6


# ---------------------------------------------------------------------------
# Main runner: train on one dataset with multiple seeds
# ---------------------------------------------------------------------------

def run_dataset(
    dataset_name: str,
    seeds: List[int] = [42, 123, 456],
    use_ssl: bool = True,
    batch_size: int = 512,
    n_teacher_epochs: int = 40,
    n_student_epochs: int = 30,
    n_ssl_epochs: int = 10,
    d_model: int = 128,
    lr: float = 5e-4,
    focal_gamma: float = 2.0,
    alpha_causal: float = 0.1,
    n_hgnn_layers: int = 2,
) -> Dict:
    """Train MorphGuard-IDS on a dataset over multiple seeds."""
    preproc_dir = ROOT / "results" / "preprocessed"
    meta_dir = ROOT / "results" / "metadata"
    ckpt_dir = ROOT / "checkpoints"
    res_dir = ROOT / "results"
    ckpt_dir.mkdir(exist_ok=True)

    # Load preprocessed data. tiissrc23 uses a documented, class-capped
    # (<=20,000/class) TRAINING sample for tractability; val/test are always
    # the full natural-prevalence splits -- see baselines/train_baselines.py
    # load_data() for the same convention.
    capped_path = preproc_dir / f"{dataset_name}_train_capped.parquet"
    train_path = capped_path if capped_path.exists() else preproc_dir / f"{dataset_name}_train.parquet"
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(preproc_dir / f"{dataset_name}_val.parquet")
    test_df = pd.read_parquet(preproc_dir / f"{dataset_name}_test.parquet")
    if train_path == capped_path:
        log.info(f"  Using capped training sample ({len(train_df)} rows) for {dataset_name}.")
    with open(meta_dir / f"{dataset_name}_meta.json") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    n_classes = meta["n_classes_multi"]
    d_flow = len(feature_cols)

    log.info(f"\n{'='*60}")
    log.info(f"Dataset: {meta['dataset']}  n_classes={n_classes}  d_flow={d_flow}")
    log.info(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Compute class weights (from train set)
    train_labels = train_df["multi_label"].values
    cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    cw_tensor = torch.tensor(cw, dtype=torch.float32)

    # Data loaders
    train_loader = make_loader(train_df, feature_cols, "multi_label", batch_size, shuffle=True)
    val_loader = make_loader(val_df, feature_cols, "multi_label", batch_size, shuffle=False)
    test_loader = make_loader(test_df, feature_cols, "multi_label", batch_size, shuffle=False)

    teacher_results = []
    student_results = []
    histories = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        log.info(f"\n--- Seed {seed} ---")

        # Build teacher
        teacher = MorphGuardTeacher(
            d_flow=d_flow,
            n_classes=n_classes,
            d_model=d_model,
            n_tcn_layers=4,
            n_hgnn_layers=n_hgnn_layers,
            n_heads=4,
            dropout=0.15,
            n_envs=3,
        ).to(DEVICE)

        log.info(f"  Teacher params: {count_params(teacher):,}  size: {model_size_mb(teacher):.2f} MB")

        # SSL pretraining
        if use_ssl:
            ssl_losses = ssl_pretrain(teacher, train_loader, n_ssl_epochs, lr=lr, device=DEVICE)
        else:
            ssl_losses = []

        # Supervised fine-tuning
        ckpt_path = ckpt_dir / f"{dataset_name}_teacher_seed{seed}.pt"
        history = train_teacher(
            teacher, train_loader, val_loader,
            n_epochs=n_teacher_epochs,
            lr=lr,
            weight_decay=1e-4,
            class_weights=cw_tensor,
            n_classes=n_classes,
            device=DEVICE,
            patience=10,
            augment=True,
            alpha_causal=alpha_causal,
            focal_gamma=focal_gamma,
            checkpoint_path=ckpt_path,
        )
        history["ssl_losses"] = ssl_losses
        histories.append(history)

        # Evaluate teacher
        t_metrics, t_preds, t_probs, t_labels = eval_model(
            teacher, test_loader, DEVICE, is_teacher=True, n_classes=n_classes
        )
        t_metrics["n_params"] = count_params(teacher)
        t_metrics["model_size_mb"] = model_size_mb(teacher)
        teacher_results.append(t_metrics)
        log.info(f"  Teacher Test: F1={t_metrics['f1_macro']:.4f}  AUC={t_metrics.get('roc_auc',0):.4f}  FPR={t_metrics['fpr']:.4f}  Lat={t_metrics['latency_ms_per_sample']:.3f}ms")

        # Build + distill student
        student = MorphGuardStudent(
            d_flow=d_flow,
            n_classes=n_classes,
            d_model=d_model // 2,
            n_layers=2,
            dropout=0.1,
        ).to(DEVICE)

        log.info(f"  Student params: {count_params(student):,}  size: {model_size_mb(student):.2f} MB")

        s_ckpt = ckpt_dir / f"{dataset_name}_student_seed{seed}.pt"
        distill_student(
            teacher, student, train_loader, val_loader,
            n_epochs=n_student_epochs,
            lr=lr * 2,
            temperature=4.0,
            n_classes=n_classes,
            device=DEVICE,
            patience=8,
            checkpoint_path=s_ckpt,
        )

        s_metrics, s_preds, s_probs, s_labels = eval_model(
            student, test_loader, DEVICE, is_teacher=False, n_classes=n_classes
        )
        s_metrics["n_params"] = count_params(student)
        s_metrics["model_size_mb"] = model_size_mb(student)
        student_results.append(s_metrics)
        log.info(f"  Student Test: F1={s_metrics['f1_macro']:.4f}  Lat={s_metrics['latency_ms_per_sample']:.3f}ms")

    # Aggregate over seeds
    def aggregate(results: List[Dict]) -> Dict:
        keys = results[0].keys()
        agg = {}
        for k in keys:
            vals = [r[k] for r in results if isinstance(r[k], (int, float))]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
        return agg

    agg_teacher = aggregate(teacher_results)
    agg_student = aggregate(student_results)

    output = {
        "dataset": dataset_name,
        "teacher": agg_teacher,
        "student": agg_student,
        "teacher_per_seed": teacher_results,
        "student_per_seed": student_results,
        "training_history": histories,
    }

    out_path = res_dir / f"{dataset_name}_morphguard_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nSaved results -> {out_path}")

    log.info(f"\nTeacher: F1={agg_teacher['f1_macro_mean']:.4f}±{agg_teacher['f1_macro_std']:.4f}  AUC={agg_teacher.get('roc_auc_mean',0):.4f}  FPR={agg_teacher['fpr_mean']:.4f}")
    log.info(f"Student: F1={agg_student['f1_macro_mean']:.4f}±{agg_student['f1_macro_std']:.4f}  Lat={agg_student['latency_ms_per_sample_mean']:.3f}ms")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="edgeiiot",
                         choices=["edgeiiot", "rtiot", "unsw", "tiissrc23", "ustc", "5gad"])
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--ssl", action="store_true", default=True)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--teacher_epochs", type=int, default=40)
    parser.add_argument("--student_epochs", type=int, default=30)
    parser.add_argument("--ssl_epochs", type=int, default=10)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    run_dataset(
        args.dataset,
        seeds=seeds,
        use_ssl=args.ssl,
        batch_size=args.batch_size,
        n_teacher_epochs=args.teacher_epochs,
        n_student_epochs=args.student_epochs,
        n_ssl_epochs=args.ssl_epochs,
        d_model=args.d_model,
    )
