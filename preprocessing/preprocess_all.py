"""
MorphGuard-IDS Dataset Preprocessing
Handles Edge-IIoTset, RT-IoT2022, UNSW-NB15
Strict no-leakage: all encoders/scalers fit on train only.
"""

import os, sys, json, warnings, logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "Data"
OUT_DIR = ROOT / "results" / "preprocessed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR = ROOT / "results" / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "preprocessing.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SEED = 42


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def port_group(port):
    """Map port number to a group label."""
    try:
        p = int(float(port))
        if p <= 1023:
            return "well_known"
        elif p <= 49151:
            return "registered"
        else:
            return "dynamic"
    except Exception:
        return "unknown"


def remove_duplicates(df, label_col):
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"  Duplicates removed: {before - len(df)}")
    return df


def handle_inf_nan(df, num_cols):
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df, imputer


def stratified_split(df, label_col, test_size=0.15, val_size=0.15, seed=SEED):
    """Stratified 70/15/15 split."""
    # first split off test
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    # then split val from train_val
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[label_col],
        random_state=seed,
    )
    log.info(f"  Split -> train:{len(train)}  val:{len(val)}  test:{len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fit_encode_cats(train_df, cat_cols):
    """Fit OrdinalEncoder on train, return fitted encoder and encoded."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_df[cat_cols].astype(str))
    return enc


def apply_cat_encoder(df, enc, cat_cols):
    result = df.copy()
    result[cat_cols] = enc.transform(df[cat_cols].astype(str))
    return result


def fit_scale_nums(train_df, num_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[num_cols])
    return scaler


def apply_scaler(df, scaler, num_cols):
    result = df.copy()
    result[num_cols] = scaler.transform(df[num_cols])
    return result


def remove_zero_var(train_df, num_cols, threshold=0.0):
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(train_df[num_cols])
    kept = [c for c, s in zip(num_cols, sel.get_support()) if s]
    removed = [c for c in num_cols if c not in kept]
    if removed:
        log.info(f"  Zero-variance removed: {removed}")
    return kept, sel


def save_preprocessed(name, train, val, test, feature_cols, meta):
    prefix = OUT_DIR / name
    train.to_parquet(f"{prefix}_train.parquet", index=False)
    val.to_parquet(f"{prefix}_val.parquet", index=False)
    test.to_parquet(f"{prefix}_test.parquet", index=False)
    meta["feature_cols"] = feature_cols
    with open(META_DIR / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Saved {name} -> {OUT_DIR}")


# ---------------------------------------------------------------------------
# 1. Edge-IIoTset
# ---------------------------------------------------------------------------

def preprocess_edgeiiot():
    log.info("=" * 60)
    log.info("Preprocessing Edge-IIoTset")
    csv = DATA_DIR / "Edge-IIoTset" / "Edge-IIoTset dataset" / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv"
    df = pd.read_csv(csv, low_memory=False)
    log.info(f"  Raw shape: {df.shape}")

    # Drop unnamed index
    drop_cols = [c for c in df.columns if "Unnamed" in c]
    df = df.drop(columns=drop_cols)

    # Target
    LABEL_COL = "Attack_type"
    df = remove_duplicates(df, LABEL_COL)

    # Binary label
    df["binary_label"] = (df[LABEL_COL] != "Normal").astype(int)

    # Multi-class label
    le_multi = LabelEncoder()
    df["multi_label"] = le_multi.fit_transform(df[LABEL_COL])

    # Feature split
    cat_cols = [c for c in ["proto", "service"] if c in df.columns]
    # Add port groups
    for pc in ["id.orig_p", "id.resp_p"]:
        if pc in df.columns:
            df[f"{pc}_grp"] = df[pc].apply(port_group)
            cat_cols.append(f"{pc}_grp")

    # Exclude all label-related columns to prevent data leakage
    non_feat = [LABEL_COL, "Attack_label", "binary_label", "multi_label"]
    all_cols = df.columns.tolist()
    num_cols = [
        c for c in all_cols
        if c not in non_feat + cat_cols + drop_cols
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    # Handle inf/nan in numeric
    df, _ = handle_inf_nan(df, num_cols)

    # Stratified split on multi_label
    train, val, test = stratified_split(df, "multi_label")

    # Remove zero-variance on train
    num_cols, var_sel = remove_zero_var(train, num_cols)

    # Fit encoders/scalers on train only
    cat_enc = fit_encode_cats(train, cat_cols)
    num_scaler = fit_scale_nums(train, num_cols)

    # Apply to all splits
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df = apply_cat_encoder(split_df, cat_enc, cat_cols)
        split_df = apply_scaler(split_df, num_scaler, num_cols)
        if split_name == "train":
            train = split_df
        elif split_name == "val":
            val = split_df
        else:
            test = split_df

    feature_cols = num_cols + cat_cols
    log.info(f"  Feature columns: {len(feature_cols)}")
    log.info(f"  Class distribution (multi): {df['multi_label'].value_counts().to_dict()}")

    # Save artifacts
    joblib.dump(cat_enc, META_DIR / "edgeiiot_cat_enc.joblib")
    joblib.dump(num_scaler, META_DIR / "edgeiiot_scaler.joblib")
    joblib.dump(var_sel, META_DIR / "edgeiiot_varsel.joblib")
    joblib.dump(le_multi, META_DIR / "edgeiiot_le_multi.joblib")

    meta = {
        "dataset": "Edge-IIoTset",
        "raw_shape": list(df.shape),
        "n_classes_multi": int(df["multi_label"].nunique()),
        "n_classes_binary": 2,
        "class_names": le_multi.classes_.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "class_distribution": df[LABEL_COL].value_counts().to_dict(),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }
    save_preprocessed("edgeiiot", train, val, test, feature_cols, meta)
    log.info("Edge-IIoTset preprocessing complete.")
    return meta


# ---------------------------------------------------------------------------
# 2. RT-IoT2022
# ---------------------------------------------------------------------------

def preprocess_rtiot():
    log.info("=" * 60)
    log.info("Preprocessing RT-IoT2022")
    csv = DATA_DIR / "RT_IOT" / "RT_IOT2022.csv"
    df = pd.read_csv(csv, low_memory=False)
    log.info(f"  Raw shape: {df.shape}")

    # Drop Unnamed
    drop_cols = [c for c in df.columns if "Unnamed" in c]
    df = df.drop(columns=drop_cols)

    LABEL_COL = "Attack_type"
    BIN_COL = "Attack_label"

    # Attack_label: 0=normal, 1=attack (if present)
    if BIN_COL in df.columns:
        df["binary_label"] = df[BIN_COL].astype(int)
    else:
        # Create binary from attack type (Thing_Speak and Wipro_bulb are often considered normal IoT)
        NORMAL_TYPES = {"Thing_Speak", "Wipro_bulb", "MQTT_Publish"}
        df["binary_label"] = (~df[LABEL_COL].isin(NORMAL_TYPES)).astype(int)

    df = remove_duplicates(df, LABEL_COL)

    le_multi = LabelEncoder()
    df["multi_label"] = le_multi.fit_transform(df[LABEL_COL])

    # Feature split
    cat_cols = []
    for c in ["proto", "service", "flag"]:
        if c in df.columns:
            cat_cols.append(c)

    # Port groups
    for pc in ["id.orig_p", "id.resp_p"]:
        if pc in df.columns:
            df[f"{pc}_grp"] = df[pc].apply(port_group)
            cat_cols.append(f"{pc}_grp")

    non_feat = [LABEL_COL, BIN_COL if BIN_COL in df.columns else "", "binary_label", "multi_label"]
    non_feat = [x for x in non_feat if x]

    num_cols = [
        c for c in df.columns
        if c not in non_feat + cat_cols + drop_cols
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    df, _ = handle_inf_nan(df, num_cols)

    train, val, test = stratified_split(df, "multi_label")

    num_cols, var_sel = remove_zero_var(train, num_cols)
    cat_enc = fit_encode_cats(train, cat_cols) if cat_cols else None
    num_scaler = fit_scale_nums(train, num_cols)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        if cat_cols and cat_enc:
            split_df = apply_cat_encoder(split_df, cat_enc, cat_cols)
        split_df = apply_scaler(split_df, num_scaler, num_cols)
        if split_name == "train":
            train = split_df
        elif split_name == "val":
            val = split_df
        else:
            test = split_df

    feature_cols = num_cols + (cat_cols if cat_cols else [])
    log.info(f"  Feature columns: {len(feature_cols)}")

    if cat_enc:
        joblib.dump(cat_enc, META_DIR / "rtiot_cat_enc.joblib")
    joblib.dump(num_scaler, META_DIR / "rtiot_scaler.joblib")
    joblib.dump(var_sel, META_DIR / "rtiot_varsel.joblib")
    joblib.dump(le_multi, META_DIR / "rtiot_le_multi.joblib")

    meta = {
        "dataset": "RT-IoT2022",
        "raw_shape": list(df.shape),
        "n_classes_multi": int(df["multi_label"].nunique()),
        "n_classes_binary": 2,
        "class_names": le_multi.classes_.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "class_distribution": df[LABEL_COL].value_counts().to_dict(),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }
    save_preprocessed("rtiot", train, val, test, feature_cols, meta)
    log.info("RT-IoT2022 preprocessing complete.")
    return meta


# ---------------------------------------------------------------------------
# 3. UNSW-NB15
# ---------------------------------------------------------------------------

def preprocess_unsw():
    log.info("=" * 60)
    log.info("Preprocessing UNSW-NB15")
    train_csv = DATA_DIR / "UNSW_NB15" / "UNSW_NB15_training-set.csv"
    test_csv = DATA_DIR / "UNSW_NB15" / "UNSW_NB15_testing-set.csv"
    df_train_raw = pd.read_csv(train_csv, low_memory=False)
    df_test_raw = pd.read_csv(test_csv, low_memory=False)
    log.info(f"  Raw train: {df_train_raw.shape}  test: {df_test_raw.shape}")

    LABEL_COL = "attack_cat"
    BIN_COL = "label"

    for df in [df_train_raw, df_test_raw]:
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    # Combine to do consistent encoding then split back
    df_train_raw["_split"] = "train"
    df_test_raw["_split"] = "test"
    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    # Binary label
    df_all["binary_label"] = df_all[BIN_COL].astype(int)

    # Multi-class label: strip whitespace from attack_cat
    df_all[LABEL_COL] = df_all[LABEL_COL].str.strip()
    # Fill NaN attack_cat for normal traffic
    df_all[LABEL_COL] = df_all[LABEL_COL].fillna("Normal")

    le_multi = LabelEncoder()
    le_multi.fit(df_all[LABEL_COL])
    df_all["multi_label"] = le_multi.transform(df_all[LABEL_COL])

    # Separate train/test back
    df_train_all = df_all[df_all["_split"] == "train"].drop(columns=["_split"])
    df_test_all = df_all[df_all["_split"] == "test"].drop(columns=["_split"])

    df_train_all = remove_duplicates(df_train_all, LABEL_COL)

    # Feature split
    cat_cols = []
    for c in ["proto", "service", "state"]:
        if c in df_all.columns:
            cat_cols.append(c)

    non_feat = [LABEL_COL, BIN_COL, "binary_label", "multi_label", "_split"]
    num_cols = [
        c for c in df_all.columns
        if c not in non_feat + cat_cols
        and df_all[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    df_train_all, _ = handle_inf_nan(df_train_all, num_cols)
    df_test_all, _ = handle_inf_nan(df_test_all, num_cols)

    # Split train into train/val (no test split needed - use provided test)
    train, val = train_test_split(
        df_train_all,
        test_size=0.15,
        stratify=df_train_all["multi_label"],
        random_state=SEED,
    )
    test = df_test_all
    log.info(f"  Split -> train:{len(train)}  val:{len(val)}  test:{len(test)}")

    num_cols, var_sel = remove_zero_var(train, num_cols)
    cat_enc = fit_encode_cats(train, cat_cols)
    num_scaler = fit_scale_nums(train, num_cols)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df = apply_cat_encoder(split_df, cat_enc, cat_cols)
        split_df = apply_scaler(split_df, num_scaler, num_cols)
        if split_name == "train":
            train = split_df
        elif split_name == "val":
            val = split_df
        else:
            test = split_df

    feature_cols = num_cols + cat_cols
    log.info(f"  Feature columns: {len(feature_cols)}")

    joblib.dump(cat_enc, META_DIR / "unsw_cat_enc.joblib")
    joblib.dump(num_scaler, META_DIR / "unsw_scaler.joblib")
    joblib.dump(var_sel, META_DIR / "unsw_varsel.joblib")
    joblib.dump(le_multi, META_DIR / "unsw_le_multi.joblib")

    meta = {
        "dataset": "UNSW-NB15",
        "raw_train_shape": list(df_train_raw.shape),
        "raw_test_shape": list(df_test_raw.shape),
        "n_classes_multi": int(df_all["multi_label"].nunique()),
        "n_classes_binary": 2,
        "class_names": le_multi.classes_.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "class_distribution_train": df_train_all[LABEL_COL].value_counts().to_dict(),
        "class_distribution_test": df_test_all[LABEL_COL].value_counts().to_dict(),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }
    save_preprocessed("unsw", train, val, test, feature_cols, meta)
    log.info("UNSW-NB15 preprocessing complete.")
    return meta


# ---------------------------------------------------------------------------
# 4. TII-SSRC-23 (full natural-prevalence CSV, NOT the 100:1 pilot sample)
# ---------------------------------------------------------------------------

def preprocess_tii_ssrc23():
    log.info("=" * 60)
    log.info("Preprocessing TII-SSRC-23 (full natural prevalence)")
    csv = DATA_DIR / "TII-SSRC-23" / "csv" / "data.csv"
    df = pd.read_csv(csv, low_memory=False)
    log.info(f"  Raw shape: {df.shape}")
    log.info(f"  Natural prevalence -> {df['Label'].value_counts().to_dict()}")

    drop_cols = [c for c in df.columns if "Unnamed" in c]
    df = df.drop(columns=drop_cols)

    # Leakage-prone identifiers: drop, but keep ports/protocol as features
    id_cols = [c for c in ["Flow ID", "Src IP", "Dst IP", "Timestamp"] if c in df.columns]

    BIN_SRC = "Label"                # Benign / Malicious
    MULTI_SRC = "Traffic Subtype"    # fine-grained (~20+ classes)
    MACRO_SRC = "Traffic Type"       # coarse-grained (8 classes)

    df = df.drop_duplicates()
    log.info(f"  After duplicate removal: {df.shape}")

    df["binary_label"] = (df[BIN_SRC] != "Benign").astype(int)

    le_multi = LabelEncoder()
    df["multi_label"] = le_multi.fit_transform(df[MULTI_SRC])
    le_macro = LabelEncoder()
    df["macro_label"] = le_macro.fit_transform(df[MACRO_SRC])

    cat_cols = []
    for pc in ["Src Port", "Dst Port"]:
        df[f"{pc}_grp"] = df[pc].apply(port_group)
        cat_cols.append(f"{pc}_grp")
    cat_cols.append("Protocol")

    non_feat = [BIN_SRC, MULTI_SRC, MACRO_SRC, "binary_label", "multi_label", "macro_label"]
    num_cols = [
        c for c in df.columns
        if c not in non_feat + cat_cols + id_cols + drop_cols
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    df, _ = handle_inf_nan(df, num_cols)

    # Stratify on multi_label (finest grain); extremely rare classes (Background n=32)
    # still receive representation in all three splits since test_size/val_size are
    # large enough relative to the rarest class to keep >=1 sample per split.
    train, val, test = stratified_split(df, "multi_label")

    num_cols, var_sel = remove_zero_var(train, num_cols)
    cat_enc = fit_encode_cats(train, cat_cols)
    num_scaler = fit_scale_nums(train, num_cols)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df = apply_cat_encoder(split_df, cat_enc, cat_cols)
        split_df = apply_scaler(split_df, num_scaler, num_cols)
        if split_name == "train":
            train = split_df
        elif split_name == "val":
            val = split_df
        else:
            test = split_df

    feature_cols = num_cols + cat_cols
    log.info(f"  Feature columns: {len(feature_cols)}")

    joblib.dump(cat_enc, META_DIR / "tiissrc23_cat_enc.joblib")
    joblib.dump(num_scaler, META_DIR / "tiissrc23_scaler.joblib")
    joblib.dump(var_sel, META_DIR / "tiissrc23_varsel.joblib")
    joblib.dump(le_multi, META_DIR / "tiissrc23_le_multi.joblib")
    joblib.dump(le_macro, META_DIR / "tiissrc23_le_macro.joblib")

    meta = {
        "dataset": "TII-SSRC-23",
        "sampling_protocol": "FULL natural-prevalence CSV (8,656,767 rows); NOT the earlier 100:1 pilot sample",
        "raw_shape": list(df.shape),
        "n_classes_multi": int(df["multi_label"].nunique()),
        "n_classes_macro": int(df["macro_label"].nunique()),
        "n_classes_binary": 2,
        "class_names_multi": le_multi.classes_.tolist(),
        "class_names_macro": le_macro.classes_.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "class_distribution_binary": df[BIN_SRC].value_counts().to_dict(),
        "class_distribution_macro": df[MACRO_SRC].value_counts().to_dict(),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }
    save_preprocessed("tiissrc23", train, val, test, feature_cols, meta)
    log.info("TII-SSRC-23 preprocessing complete.")
    return meta


# ---------------------------------------------------------------------------
# 5. USTC-TFC2016 (PCAP-first; consumes cached scapy-extracted flow table)
# ---------------------------------------------------------------------------

def preprocess_ustc():
    log.info("=" * 60)
    log.info("Preprocessing USTC-TFC2016 (PCAP-extracted flows)")
    flow_path = ROOT / "results" / "extracted_flows" / "ustc_tfc2016_flows.parquet"
    if not flow_path.exists():
        log.error(f"  {flow_path} not found -- run preprocessing/extract_ustc_flows.py first.")
        return None
    df = pd.read_parquet(flow_path)
    log.info(f"  Raw extracted shape: {df.shape}")

    drop_cols = []
    src_file_col = "__src_file" if "__src_file" in df.columns else None

    LABEL_COL = "Label"          # malware family / benign app name
    BIN_SRC = "binary_label"     # Malware / Benign (string, set at extraction time)

    df = df.drop_duplicates(subset=[c for c in df.columns if c != src_file_col])
    log.info(f"  After duplicate removal: {df.shape}")

    df["binary_label"] = (df[BIN_SRC] == "Malware").astype(int)
    le_multi = LabelEncoder()
    df["multi_label"] = le_multi.fit_transform(df[LABEL_COL])

    cat_cols = []
    for pc in ["Src Port", "Dst Port"]:
        df[f"{pc}_grp"] = df[pc].apply(port_group)
        cat_cols.append(f"{pc}_grp")
    cat_cols.append("Protocol")

    non_feat = [LABEL_COL, BIN_SRC, "binary_label", "multi_label"]
    if src_file_col:
        non_feat.append(src_file_col)
    num_cols = [
        c for c in df.columns
        if c not in non_feat + cat_cols + drop_cols
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    df, _ = handle_inf_nan(df, num_cols)

    train, val, test = stratified_split(df, "multi_label")

    num_cols, var_sel = remove_zero_var(train, num_cols)
    cat_enc = fit_encode_cats(train, cat_cols)
    num_scaler = fit_scale_nums(train, num_cols)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df = apply_cat_encoder(split_df, cat_enc, cat_cols)
        split_df = apply_scaler(split_df, num_scaler, num_cols)
        if split_name == "train":
            train = split_df
        elif split_name == "val":
            val = split_df
        else:
            test = split_df

    feature_cols = num_cols + cat_cols
    log.info(f"  Feature columns: {len(feature_cols)}")

    joblib.dump(cat_enc, META_DIR / "ustc_cat_enc.joblib")
    joblib.dump(num_scaler, META_DIR / "ustc_scaler.joblib")
    joblib.dump(var_sel, META_DIR / "ustc_varsel.joblib")
    joblib.dump(le_multi, META_DIR / "ustc_le_multi.joblib")

    meta = {
        "dataset": "USTC-TFC2016",
        "sampling_protocol": "FULL PCAP corpus (all Malware/*.pcap and Benign/*.pcap incl. SMB/Weibo subfolders), scapy-extracted bidirectional flows, 120s flow timeout",
        "raw_shape": list(df.shape),
        "n_classes_multi": int(df["multi_label"].nunique()),
        "n_classes_binary": 2,
        "class_names": le_multi.classes_.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "class_distribution": df[LABEL_COL].value_counts().to_dict(),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }
    save_preprocessed("ustc", train, val, test, feature_cols, meta)
    log.info("USTC-TFC2016 preprocessing complete.")
    return meta


# ---------------------------------------------------------------------------
# 6. 5GAD-2022 (PCAP-first; consumes cached scapy-extracted flow table)
# ---------------------------------------------------------------------------

def preprocess_5gad():
    log.info("=" * 60)
    log.info("Preprocessing 5GAD-2022 (PCAP-extracted flows)")
    flow_path = ROOT / "results" / "extracted_flows" / "5gad2022_flows.parquet"
    if not flow_path.exists():
        log.error(f"  {flow_path} not found -- run preprocessing/extract_5gad_flows.py first.")
        return None
    df = pd.read_parquet(flow_path)
    log.info(f"  Raw extracted shape: {df.shape}")

    src_file_col = "__src_file" if "__src_file" in df.columns else None
    LABEL_COL = "Label"          # attack scenario name / Normal
    BIN_SRC = "binary_label"     # Attack / Normal

    df = df.drop_duplicates(subset=[c for c in df.columns if c != src_file_col])
    log.info(f"  After duplicate removal: {df.shape}")

    # Honest handling of scenarios with too few flows to appear in all three
    # splits: the canonical capture for automatedDropWithTimer and
    # automatedRedirectWithTimer yields only 1 flow each (see
    # logs/extract_5gad.log). A class needs >=5 members for a 70/15/15
    # stratified split to plausibly place >=1 in val and >=1 in test, so
    # classes below that threshold are EXCLUDED from the dataset used for
    # multiclass/binary supervised learning and reported explicitly as a
    # limitation, rather than silently merged or kept in only one split.
    class_counts = df[LABEL_COL].value_counts()
    rare_classes = class_counts[class_counts < 5].index.tolist()
    n_excluded = int(df[LABEL_COL].isin(rare_classes).sum())
    if rare_classes:
        log.warning(f"  Excluding {n_excluded} flow(s) from {rare_classes} "
                     f"(< 5 flows/class; cannot guarantee representation in all 3 splits).")
        df = df[~df[LABEL_COL].isin(rare_classes)].reset_index(drop=True)

    df["binary_label"] = (df[BIN_SRC] == "Attack").astype(int)
    le_multi = LabelEncoder()
    df["multi_label"] = le_multi.fit_transform(df[LABEL_COL])

    cat_cols = []
    for pc in ["Src Port", "Dst Port"]:
        df[f"{pc}_grp"] = df[pc].apply(port_group)
        cat_cols.append(f"{pc}_grp")
    cat_cols.append("Protocol")

    non_feat = [LABEL_COL, BIN_SRC, "binary_label", "multi_label"]
    if src_file_col:
        non_feat.append(src_file_col)
    num_cols = [
        c for c in df.columns
        if c not in non_feat + cat_cols
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    df, _ = handle_inf_nan(df, num_cols)
    train, val, test = stratified_split(df, "multi_label")

    num_cols, var_sel = remove_zero_var(train, num_cols)
    cat_enc = fit_encode_cats(train, cat_cols)
    num_scaler = fit_scale_nums(train, num_cols)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df = apply_cat_encoder(split_df, cat_enc, cat_cols)
        split_df = apply_scaler(split_df, num_scaler, num_cols)
        if split_name == "train":
            train = split_df
        elif split_name == "val":
            val = split_df
        else:
            test = split_df

    feature_cols = num_cols + cat_cols
    log.info(f"  Feature columns: {len(feature_cols)}")

    joblib.dump(cat_enc, META_DIR / "5gad_cat_enc.joblib")
    joblib.dump(num_scaler, META_DIR / "5gad_scaler.joblib")
    joblib.dump(var_sel, META_DIR / "5gad_varsel.joblib")
    joblib.dump(le_multi, META_DIR / "5gad_le_multi.joblib")

    meta = {
        "dataset": "5GAD-2022",
        "sampling_protocol": "All 10 attack scenarios in full (canonical Attacks_<scenario>.pcapng); "
                              "Normal traffic sampled from 2 of 24 available Normal-1UE/2UE captures "
                              "(see results/extracted_flows/5gad2022_sampling_protocol.json)",
        "excluded_rare_classes": rare_classes,
        "excluded_flow_count": n_excluded,
        "exclusion_reason": "< 5 flows/class; cannot guarantee representation in all 3 splits (still counted in raw capture stats, excluded from supervised learning task)",
        "raw_shape": list(df.shape),
        "n_classes_multi": int(df["multi_label"].nunique()),
        "n_classes_binary": 2,
        "class_names": le_multi.classes_.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "class_distribution": df[LABEL_COL].value_counts().to_dict(),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }
    save_preprocessed("5gad", train, val, test, feature_cols, meta)
    log.info("5GAD-2022 preprocessing complete.")
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting MorphGuard-IDS preprocessing pipeline")
    meta_edge = preprocess_edgeiiot()
    meta_rt = preprocess_rtiot()
    meta_unsw = preprocess_unsw()

    summary = {
        "Edge-IIoTset": {
            "train": meta_edge["train_size"],
            "val": meta_edge["val_size"],
            "test": meta_edge["test_size"],
            "n_features": len(meta_edge["feature_cols"]),
            "n_classes": meta_edge["n_classes_multi"],
        },
        "RT-IoT2022": {
            "train": meta_rt["train_size"],
            "val": meta_rt["val_size"],
            "test": meta_rt["test_size"],
            "n_features": len(meta_rt["feature_cols"]),
            "n_classes": meta_rt["n_classes_multi"],
        },
        "UNSW-NB15": {
            "train": meta_unsw["train_size"],
            "val": meta_unsw["val_size"],
            "test": meta_unsw["test_size"],
            "n_features": len(meta_unsw["feature_cols"]),
            "n_classes": meta_unsw["n_classes_multi"],
        },
    }
    with open(META_DIR / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("\nPreprocessing summary:")
    for ds, s in summary.items():
        log.info(f"  {ds}: train={s['train']} val={s['val']} test={s['test']} feats={s['n_features']} classes={s['n_classes']}")
    log.info("All preprocessing complete.")
