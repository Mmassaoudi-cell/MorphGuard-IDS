"""
Generate IEEE-style LaTeX tables from simulation results.
All tables show mean ± std over 3 random seeds.
"""

import json, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RES_DIR = ROOT / "results"
TAB_DIR = ROOT / "tables"
TAB_DIR.mkdir(exist_ok=True)

DATASETS = ["5gad", "ustc", "tiissrc23", "rtiot"]
DS_LABELS = {"5gad": "5GAD-2022", "ustc": "USTC-TFC2016", "tiissrc23": "TII-SSRC-23", "rtiot": "RT-IoT2022"}
MODEL_DISPLAY = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "ExtraTrees": "Extra Trees",
    "HistGradientBoosting": "HistGradBoosting",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
    "MLP": "MLP",
    "1D-CNN": "1D-CNN",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "CNN-BiLSTM": "CNN-BiLSTM",
    "TCN": "TCN",
    "Transformer": "Transformer",
    "GAT": "GAT",
    "GraphSAGE": "GraphSAGE",
    "MorphGuard Teacher": "\\textbf{MorphGuard Teacher}",
    "MorphGuard Student": "\\textbf{MorphGuard Student}",
}
MAIN_BASELINES_TO_BEAT = ["XGBoost", "LightGBM", "CatBoost", "HistGradientBoosting", "RandomForest"]


def fmt(mean, std, bold=False, pct=False):
    """Format mean ± std for LaTeX table cell."""
    if mean == 0 and std == 0:
        return "--"
    scale = 100 if pct else 1
    s = f"{mean * scale:.2f}$\\pm${std * scale:.2f}"
    if pct:
        s = f"{mean * 100:.1f}$\\pm${std * 100:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def load_all():
    baselines, morphguard, ablations = {}, {}, {}
    for ds in DATASETS:
        bp = RES_DIR / f"{ds}_baselines.json"
        mp = RES_DIR / f"{ds}_morphguard_results.json"
        ap = RES_DIR / f"{ds}_ablations.json"
        if bp.exists():
            with open(bp) as f:
                baselines[ds] = json.load(f)
        if mp.exists():
            with open(mp) as f:
                morphguard[ds] = json.load(f)
        if ap.exists():
            with open(ap) as f:
                ablations[ds] = json.load(f)
    return baselines, morphguard, ablations


def save(name, content):
    p = TAB_DIR / f"{name}.tex"
    with open(p, "w") as f:
        f.write(content)
    print(f"Saved {p}")


# ============================================================
# Table I: Dataset Statistics
# ============================================================

def table_dataset_stats():
    meta_dir = ROOT / "results" / "metadata"
    rows = []
    for ds in DATASETS:
        mp = meta_dir / f"{ds}_meta.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            m = json.load(f)
        label = DS_LABELS[ds]
        total = m.get("train_size", 0) + m.get("val_size", 0) + m.get("test_size", 0)
        n_feat = len(m.get("feature_cols", []))
        n_cls = m.get("n_classes_multi", 0)
        # Prefer the explicit binary distribution saved by the newer
        # preprocessors (tiissrc23/ustc/5gad); fall back to the heuristic
        # used for the older edgeiiot/rtiot/unsw metadata.
        bin_dist = m.get("class_distribution_binary")
        if bin_dist:
            normal_keys = [k for k in bin_dist if k in ("Benign", "Normal")]
            n_normal = sum(bin_dist[k] for k in normal_keys)
            n_attack = sum(v for k, v in bin_dist.items() if k not in normal_keys)
        else:
            dist = m.get("class_distribution") or m.get("class_distribution_train", {})
            n_attack = sum(v for k, v in dist.items() if k != "Normal" and str(k) != "7")
            n_normal = dist.get("Normal", dist.get("7", 0))
        imb = f"{n_attack / max(n_normal, 1):.1f}:1" if n_normal > 0 else "N/A"
        rows.append(f"    {label} & {total:,} & {n_feat} & {n_cls} & {imb} \\\\")

    tex = r"""\begin{table}[!t]
\centering
\caption{Dataset Statistics and Class Distribution}
\label{tab:datasets}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
Dataset & Samples & Features & Classes & Attack:Normal Ratio \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    save("table1_dataset_stats", tex)


# ============================================================
# Table II: MorphGuard Architecture
# ============================================================

def table_architecture():
    tex = r"""\begin{table}[!t]
\centering
\caption{MorphGuard-IDS Architecture Configuration}
\label{tab:arch}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lll}
\toprule
Component & Teacher & Student \\
\midrule
Input & Flow feature vector & Flow feature vector \\
Temporal encoder & Multi-scale TCN (4 layers, $d=128$) & Compact TCN (2 layers, $d=64$) \\
Graph module & Typed HGNN (2 layers, 4 heads, $d=128$) & None (MLP fallback) \\
Temporal fusion & Multi-head self-attention & Dropout MLP \\
Causal reg. & IRM penalty ($\lambda_{inv}=0.1$) & None \\
Classifier & 2-layer MLP, GELU & 2-layer MLP, GELU \\
Calibration & Temperature scaling & Temperature scaling \\
Augmentation & Latent counterfactual ($\sigma=0.1$) & None \\
Loss & Focal ($\gamma=2$) + Causal + Calib. & 0.5 CE + 0.5 KD ($T=4$) \\
Optimizer & AdamW, CosineLR, wd=$10^{-4}$ & AdamW, CosineLR \\
Parameters & $\sim$892K & $\sim$59K \\
Size & $\sim$3.6 MB & $\sim$0.24 MB \\
\bottomrule
\end{tabular}}
\end{table}
"""
    save("table2_architecture", tex)


# ============================================================
# Table III: Binary IDS Performance Comparison
# ============================================================

def table_binary_comparison(baselines, morphguard):
    order = ["LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting",
             "XGBoost", "LightGBM", "CatBoost",
             "MLP", "1D-CNN", "LSTM", "GRU", "CNN-BiLSTM", "TCN", "Transformer", "GAT", "GraphSAGE",
             "MorphGuard Teacher", "MorphGuard Student"]

    col_spec = "l" + "".join(["cc" for _ in DATASETS])
    ds_header = " & ".join([f"\\multicolumn{{2}}{{c}}{{{DS_LABELS[ds]}}}" for ds in DATASETS])
    n = len(DATASETS)
    cmidrules = "".join([f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(n)])
    metric_header = "Model & " + " & ".join(["F1 & FPR" for _ in DATASETS]) + r" \\"

    header = r"""\begin{table*}[!t]
\centering
\caption{Binary IDS Detection Performance Comparison (macro-F1 $\uparrow$, FPR $\downarrow$; mean\,$\pm$\,std over 3 seeds)}
\label{tab:binary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{""" + col_spec + r"""}
\toprule
& """ + ds_header + r""" \\
""" + cmidrules + "\n" + metric_header + r"""
\midrule
"""

    rows_tex = []
    for m in order:
        label = MODEL_DISPLAY.get(m, m)
        cells = []
        is_morph = "MorphGuard" in m
        for ds in DATASETS:
            bl = baselines.get(ds, {})
            mg = morphguard.get(ds, {})
            if m in bl:
                d = bl[m]
            elif m == "MorphGuard Teacher" and "teacher" in mg:
                d = mg["teacher"]
            elif m == "MorphGuard Student" and "student" in mg:
                d = mg["student"]
            else:
                cells.extend(["--", "--"])
                continue

            f1 = d.get("f1_macro_mean", 0); f1_s = d.get("f1_macro_std", 0)
            fpr = d.get("fpr_mean", 0); fpr_s = d.get("fpr_std", 0)

            f1_str = f"{f1:.3f}$\\pm${f1_s:.3f}" if f1 > 0 else "--"
            fpr_str = f"{fpr:.4f}$\\pm${fpr_s:.4f}" if fpr > 0 else "--"
            cells.append(f"\\textbf{{{f1_str}}}" if is_morph and f1 > 0 else f1_str)
            cells.append(f"\\textbf{{{fpr_str}}}" if is_morph and fpr > 0 else fpr_str)

        rows_tex.append(f"    {label} & " + " & ".join(cells) + r" \\")
        if m in ["CatBoost", "GraphSAGE"]:
            rows_tex.append("    \\midrule")

    footer = r"""\bottomrule
\end{tabular}}
\end{table*}
"""
    tex = header + "\n".join(rows_tex) + "\n" + footer
    save("table3_binary_comparison", tex)


# ============================================================
# Table IV: Multi-class Attack Classification
# ============================================================

def table_multiclass(baselines, morphguard):
    order = ["XGBoost", "LightGBM", "CatBoost", "MLP", "TCN", "Transformer", "GAT", "GraphSAGE",
             "MorphGuard Teacher", "MorphGuard Student"]

    col_spec = "l" + "c" * len(DATASETS)
    ds_header = " & ".join(DS_LABELS[ds] for ds in DATASETS)
    header = r"""\begin{table}[!t]
\centering
\caption{Multi-Class Attack/Application Classification (Macro-F1, mean\,$\pm$\,std)}
\label{tab:multiclass}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{""" + col_spec + r"""}
\toprule
Model & """ + ds_header + r""" \\
\midrule
"""
    rows_tex = []
    for m in order:
        label = MODEL_DISPLAY.get(m, m)
        cells = []
        for ds in DATASETS:
            bl = baselines.get(ds, {})
            mg = morphguard.get(ds, {})
            if m in bl:
                d = bl[m]
            elif m == "MorphGuard Teacher" and "teacher" in mg:
                d = mg["teacher"]
            elif m == "MorphGuard Student" and "student" in mg:
                d = mg["student"]
            else:
                cells.append("--")
                continue
            f1 = d.get("f1_macro_mean", 0); f1s = d.get("f1_macro_std", 0)
            is_morph = "MorphGuard" in m
            s = f"{f1*100:.1f}$\\pm${f1s*100:.1f}"
            cells.append(f"\\textbf{{{s}}}" if is_morph and f1 > 0 else (s if f1 > 0 else "--"))
        rows_tex.append(f"    {label} & " + " & ".join(cells) + r" \\")

    footer = r"""\bottomrule
\end{tabular}}
\end{table}
"""
    tex = header + "\n".join(rows_tex) + "\n" + footer
    save("table4_multiclass", tex)


# ============================================================
# Table V: Cross-Dataset Generalization
# ============================================================

def table_cross_dataset():
    cd_path = RES_DIR / "cross_dataset_results.json"
    rows_tex = []
    note = ""
    if cd_path.exists():
        with open(cd_path) as f:
            cd = json.load(f)
        note = cd.pop("_rtiot_excluded_reason", "")
        for key, d in cd.items():
            if not d:
                continue
            src, tgt = d.get("source", "?"), d.get("target", "?")
            f1 = d.get("f1_macro_mean", 0); f1s = d.get("f1_macro_std", 0)
            nfeat = d.get("common_feats", 0)
            label = f"{DS_LABELS.get(src, src)} $\\to$ {DS_LABELS.get(tgt, tgt)}"
            rows_tex.append(f"    {label} & {f1:.3f}$\\pm${f1s:.3f} & {nfeat} \\\\")
    if not rows_tex:
        rows_tex = ["    \\multicolumn{3}{l}{No cross-dataset results available yet.} \\\\"]

    note_tex = f"\\multicolumn{{3}}{{p{{0.9\\columnwidth}}}}{{\\small Note: {note}}} \\\\" if note else ""

    tex = r"""\begin{table}[!t]
\centering
\caption{Cross-Dataset Generalization (Train$\to$Test, Macro-F1, simple MLP on common features)}
\label{tab:crossdataset}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcc}
\toprule
Train $\to$ Test & Macro-F1 & \#Common Feats. \\
\midrule
""" + "\n".join(rows_tex) + r"""
\midrule
""" + note_tex + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    save("table5_cross_dataset", tex)


# ============================================================
# Table VI: Ablation Study
# ============================================================

def table_ablation(ablations, ds="5gad"):
    abl = ablations.get(ds, {})

    order = ["Full MorphGuard-IDS",
             "w/o Hypergraph", "w/o SSL Pretrain",
             "w/o Temporal Encoder", "w/o Causal Regularization",
             "w/o Augmentation", "w/o Calibration", "w/o Distillation (Student)"]

    header = r"""\begin{table}[!t]
\centering
\caption{Ablation Study -- Component Contribution (""" + DS_LABELS.get(ds, ds) + r""")}
\label{tab:ablation}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Variant & Macro-F1 & Recall & ECE & Lat.(ms) \\
\midrule
"""
    rows_tex = []
    for v in order:
        d = abl.get(v, {})
        f1 = d.get("f1_macro_mean", 0); f1s = d.get("f1_macro_std", 0)
        rec = d.get("recall_macro_mean", 0); recs = d.get("recall_macro_std", 0)
        ece = d.get("ece_mean", 0); eces = d.get("ece_std", 0)
        lat = d.get("latency_ms_mean", 0); lats = d.get("latency_ms_std", 0)

        def c(m, s, best=False):
            if m == 0: return "--"
            s_str = f"{m:.4f}$\\pm${s:.4f}"
            return f"\\textbf{{{s_str}}}" if best else s_str

        is_full = "Full" in v
        r = f"    {v} & {c(f1,f1s,is_full)} & {c(rec,recs,is_full)} & {c(ece,eces)} & {c(lat,lats)} \\\\"
        rows_tex.append(r)
        if "Full" in v:
            rows_tex.append("    \\midrule")

    footer = r"""\bottomrule
\end{tabular}}
\end{table}
"""
    tex = header + "\n".join(rows_tex) + "\n" + footer
    save("table6_ablation", tex)


# ============================================================
# Table VII: Latency and Complexity
# ============================================================

def table_complexity(baselines, morphguard, ds="tiissrc23"):
    order = ["MLP", "TCN", "Transformer", "GAT", "GraphSAGE",
             "MorphGuard Teacher", "MorphGuard Student"]

    header = r"""\begin{table}[!t]
\centering
\caption{Model Complexity and Inference Latency (""" + DS_LABELS.get(ds, ds) + r""")}
\label{tab:complexity}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
Model & Params & Size (MB) & Lat. (ms/sample) & Macro-F1 \\
\midrule
"""
    bl = baselines.get(ds, {})
    mg = morphguard.get(ds, {})

    rows_tex = []
    for m in order:
        label = MODEL_DISPLAY.get(m, m)
        if m in bl:
            d = bl[m]
        elif m == "MorphGuard Teacher" and "teacher" in mg:
            d = mg["teacher"]
        elif m == "MorphGuard Student" and "student" in mg:
            d = mg["student"]
        else:
            rows_tex.append(f"    {label} & -- & -- & -- & -- \\\\")
            continue
        np_ = d.get("n_params_mean", d.get("n_params", 0))
        sz = d.get("model_size_mb_mean", d.get("size_mb_mean", 0))
        lat = d.get("latency_ms_mean", d.get("latency_ms_per_sample_mean", 0))
        f1 = d.get("f1_macro_mean", 0)
        is_morph = "MorphGuard" in m

        def c(v, fmt=".2f"):
            if v == 0: return "--"
            return f"{v:{fmt}}"

        row = f"    {label} & {c(np_, '.0f')} & {c(sz)} & {c(lat, '.3f')} & {c(f1, '.4f')} \\\\"
        rows_tex.append(row)

    footer = r"""\bottomrule
\end{tabular}}
\end{table}
"""
    tex = header + "\n".join(rows_tex) + "\n" + footer
    save("table7_complexity", tex)


# ============================================================
# Table VIII: Calibration and FPR
# ============================================================

def table_calibration(ablations, morphguard, ds="5gad"):
    header = r"""\begin{table}[!t]
\centering
\caption{Calibration and False-Positive Analysis (""" + DS_LABELS.get(ds, ds) + r""")}
\label{tab:calibration}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Variant & ECE ($\downarrow$) & FPR ($\downarrow$) & FNR ($\downarrow$) & Macro-F1 \\
\midrule
"""
    abl = ablations.get(ds, {})
    mg = morphguard.get(ds, {})

    rows_tex = []
    for v, src in [("Full MorphGuard-IDS", abl),
                    ("w/o Calibration", abl),
                    ("MorphGuard Student", {"MorphGuard Student": mg.get("student", {})}),
                    ]:
        d = src.get(v, {})
        ece = d.get("ece_mean", 0); eces = d.get("ece_std", 0)
        fpr = d.get("fpr_mean", 0); fprs = d.get("fpr_std", 0)
        fnr = d.get("fnr_mean", 0); fnrs = d.get("fnr_std", 0)
        f1 = d.get("f1_macro_mean", 0); f1s = d.get("f1_macro_std", 0)

        def c(m, s):
            if m == 0: return "--"
            return f"{m:.4f}$\\pm${s:.4f}"

        rows_tex.append(f"    {MODEL_DISPLAY.get(v, v)} & {c(ece,eces)} & {c(fpr,fprs)} & {c(fnr,fnrs)} & {c(f1,f1s)} \\\\")

    footer = r"""\bottomrule
\end{tabular}}
\end{table}
"""
    tex = header + "\n".join(rows_tex) + "\n" + footer
    save("table8_calibration", tex)


# ============================================================
# Generate All Tables
# ============================================================

def generate_all_tables():
    print("Generating all LaTeX tables...")
    baselines, morphguard, ablations = load_all()

    table_dataset_stats()
    table_architecture()
    table_binary_comparison(baselines, morphguard)
    table_multiclass(baselines, morphguard)
    table_cross_dataset()
    table_ablation(ablations)
    table_complexity(baselines, morphguard)
    table_calibration(ablations, morphguard)
    print(f"All tables saved to {TAB_DIR}")


if __name__ == "__main__":
    generate_all_tables()
