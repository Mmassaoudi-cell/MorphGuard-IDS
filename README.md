# MorphGuard-IDS: Self-Supervised Causal Hypergraph Distillation for Real-Time Intrusion Detection

## Overview
Complete implementation and reproducible experimental pipeline for the MorphGuard-IDS framework.

**Benchmark suite (revised):** 5GAD-2022, TII-SSRC-23, USTC-TFC2016, RT-IoT2022.
This replaces the earlier Edge-IIoTset/UNSW-NB15 pairing; their preprocessing/baseline/
MorphGuard code paths remain in the repo for reference but are not part of the current
manuscript's headline results.

## Project Structure
```
MorphGuard_IDS_Proposal/
├── Data/
│   ├── 5GAD-2022/5GAD-lfs/ # 5G-core PCAPNG captures (attacks + normal)
│   ├── TII-SSRC-23/        # csv/data.csv (flow-ready) + pcap/ (benign, malicious)
│   ├── USTC-TFC2016/       # Malware/*.pcap, Benign/*.pcap (incl. SMB/, Weibo/ subfolders)
│   ├── RT_IOT/             # Real-time IoT validation dataset (flow-ready CSV)
│   ├── Edge-IIoTset/       # (legacy, not in current headline results)
│   └── UNSW_NB15/          # (legacy, not in current headline results)
├── preprocessing/
│   ├── preprocess_all.py       # All dataset-specific preprocessing functions
│   ├── pcap_to_flow.py         # scapy-based bidirectional flow extractor (shared)
│   ├── extract_5gad_flows.py   # 5GAD-2022 PCAP extraction (documented sampling)
│   └── extract_ustc_flows.py   # USTC-TFC2016 PCAP extraction (full corpus)
├── models/                 # MorphGuard-IDS model implementation
│   ├── hypergraph.py       # Typed cyber-physical hypergraph construction (dataset-agnostic)
│   └── morphguard.py       # Teacher and student model architectures
├── training/               # Training and evaluation scripts
│   ├── train_morphguard.py # Teacher + student training
│   └── cross_dataset.py    # Cross-dataset generalization experiments
├── baselines/              # Baseline model implementations (16 models)
│   └── train_baselines.py
├── ablations/              # Ablation study scripts
│   └── run_ablations.py
├── statistics/              # Statistical significance testing
│   └── run_statistical_tests.py
├── figures/                # Figure generation scripts
├── tables/                 # LaTeX table generation
├── manuscript/             # Final IEEE manuscript
├── results/                # Saved results (JSON), preprocessed/extracted parquet
├── checkpoints/            # Model checkpoints (.pt)
└── logs/                   # Training logs
```

## Hardware Requirements
- GPU: NVIDIA RTX 5090 (25.6 GB VRAM) or equivalent CUDA-capable GPU
- RAM: ≥16 GB
- Storage: ≥10 GB for datasets and results

## Software Requirements
```
Python 3.10+
torch >= 2.0
torch_geometric >= 2.7
scikit-learn >= 1.5
xgboost >= 2.0
lightgbm >= 4.0
optuna >= 3.0
matplotlib, seaborn, pandas, numpy, scipy
pyarrow (for parquet files)
imbalanced-learn
```

## Reproduction Instructions

### Step 1: Install dependencies
```bash
pip install torch torchvision torch_geometric scikit-learn xgboost lightgbm
pip install optuna matplotlib seaborn pyarrow imbalanced-learn
```

### Step 2: Extract PCAP-first datasets to flow tables (one-time, cached)
```bash
cd MorphGuard_IDS_Proposal
python preprocessing/extract_5gad_flows.py   # ~5 min; writes results/extracted_flows/5gad2022_flows.parquet
python preprocessing/extract_ustc_flows.py   # ~15-50 min depending on CPU contention; writes ustc_tfc2016_flows.parquet
```
Sampling protocol for 5GAD-2022 (documented, not silent): all 10 attack scenarios are
used in full via their canonical `Attacks_<scenario>.pcapng` capture; Normal traffic is
sampled from 2 of 24 available Normal-1UE/2UE captures because this machine has no
tshark/libpcap backend (scapy-only fallback, ~366 KB/s) and the full Normal corpus is
~33 GB. See `results/extracted_flows/5gad2022_sampling_protocol.json` for exact files used.
USTC-TFC2016 is processed in full (no sampling).

### Step 3: Preprocess all datasets (leakage-safe splits, scalers/encoders fit on train only)
```bash
python -c "from preprocessing.preprocess_all import *; preprocess_tii_ssrc23(); preprocess_ustc(); preprocess_5gad()"
```
TII-SSRC-23 is preprocessed at **full natural prevalence** (8,656,767 rows; 8,655,466
Malicious : 1,301 Benign) for validation/test splits. A documented, class-capped
(<=20,000 rows/class) **training** sample (`tiissrc23_train_capped.parquet`, 289,829
rows) is used by all training scripts for tractability; this is logged explicitly at
runtime and is never applied to validation/test. Two 5GAD-2022 attack scenarios
(`automatedDropWithTimer`, `automatedRedirectWithTimer`) produced only 1 flow each from
their canonical capture and are excluded from the supervised learning task (reported in
metadata as `excluded_rare_classes`), while still counted in the raw capture audit.

Output: `results/preprocessed/` (parquet files), `results/metadata/` (JSON metadata).

### Step 4: Train all baselines (16 models: LR, RF, ExtraTrees, HistGradientBoosting,
XGBoost, LightGBM, CatBoost, MLP, 1D-CNN, LSTM, GRU, CNN-BiLSTM, TCN, Transformer, GAT, GraphSAGE)
```bash
python -c "from baselines.train_baselines import run_baselines; run_baselines('5gad', seeds=[42,123,456])"
python -c "from baselines.train_baselines import run_baselines; run_baselines('ustc', seeds=[42,123,456])"
python -c "from baselines.train_baselines import run_baselines; run_baselines('tiissrc23', seeds=[42,123,456])"
python -c "from baselines.train_baselines import run_baselines; run_baselines('rtiot', seeds=[42,123,456])"
```
Each call saves `results/{dataset}_baselines.json`, including per-seed raw scores
(`*_per_seed`) needed for paired statistical testing.

### Step 5: Train MorphGuard-IDS (teacher + student, 3 seeds each)
```bash
python training/train_morphguard.py --dataset 5gad --seeds 42,123,456
python training/train_morphguard.py --dataset ustc --seeds 42,123,456
python training/train_morphguard.py --dataset tiissrc23 --seeds 42,123,456
python training/train_morphguard.py --dataset rtiot --seeds 42,123,456
```

### Step 6: Run ablation study
```bash
python ablations/run_ablations.py --dataset 5gad --seeds 42,123,456
```

### Step 7: Cross-dataset transfer
```bash
python training/cross_dataset.py
```
Feature-name overlap audit: TII-SSRC-23, 5GAD-2022, and USTC-TFC2016 share 38 common
CICFlowMeter-style feature names (enabling 6 directed transfer pairs among them).
RT-IoT2022 shares **zero** feature names with the other three (Zeek-style columns,
e.g. `id.orig_p`/`proto`, vs. CICFlowMeter-style elsewhere) and is excluded from
name-matched transfer — reported as a real limitation, not silently skipped.

### Step 8: Statistical significance testing
```bash
python statistics/run_statistical_tests.py
```
Compares MorphGuard teacher/student against the five main baselines to beat (XGBoost,
LightGBM, CatBoost, HistGradientBoosting, RandomForest) per dataset, using a **paired**
t-test + Wilcoxon signed-rank test + bootstrap 95% CI when per-seed raw scores are
available on both sides, falling back to an explicitly-labeled **approximate** (Welch's,
summary-stats-only) test otherwise. Holm-Bonferroni correction is applied across all
comparisons within each dataset. Output: `results/statistical_tests/morphguard_vs_main_baselines.json`.

### Step 9: Generate figures and tables
```bash
python figures/generate_figures.py
python tables/generate_tables.py
```

### Step 10: Complete manuscript
```bash
python manuscript/update_manuscript.py
```
Output: `manuscript/morphguard_ids_final.tex`

## Scientific Integrity Notes
- All encoders, scalers, feature selectors fit on training set only
- No oversampling before train/val/test split; no test-set hyperparameter tuning
- Results reported as mean ± std over 3 random seeds (42, 123, 456)
- Every disclosed training-set sampling decision (5GAD-2022 Normal traffic, TII-SSRC-23
  training cap) is logged at runtime and recorded in `results/metadata/*.json` /
  `results/extracted_flows/*_sampling_protocol.json` — never applied silently, and never
  applied to validation/test splits
- Tree/boosting baselines are reported even where they outperform MorphGuard
  (e.g., on 5GAD-2022 and Edge-IIoTset, XGBoost/LightGBM/CatBoost beat the MorphGuard
  teacher on raw macro-F1); statistical tests determine whether such gaps are significant

## Citation
```bibtex
@article{morphguard2026,
  author  = {Mohamed Massaoudi and Maymouna Ez Eddin and Katherine R. Davis},
  title   = {MorphGuard-IDS: Self-Supervised Causal Hypergraph Distillation for
             Real-Time Intrusion Detection in Cyber-Physical and IIoT Systems},
  year    = {2026}
}
```
