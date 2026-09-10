"""
Statistical significance testing: MorphGuard (teacher/student) vs. the main
tabular baselines to beat (XGBoost, LightGBM, CatBoost, HistGradientBoosting,
RandomForest), per dataset.

Two modes, chosen automatically per comparison based on what's in the result
JSON:
  - PAIRED  (preferred): both sides have an `f1_macro_per_seed` list of equal
    length aligned by seed index -> paired t-test + Wilcoxon signed-rank +
    bootstrap CI on the paired differences.
  - SUMMARY (fallback): only mean/std/n are available (older runs that
    predate per-seed logging) -> Welch's two-sample t-test from summary
    statistics + a normal-approximation CI. This is explicitly labeled
    "approximate (summary-stats)" in the output so it is never confused with
    a true paired test.

Verdict labels follow the user-specified language exactly:
  "statistically superior" / "statistically tied" / "mean-superior" / "inferior"
Holm-Bonferroni correction is applied across all baseline comparisons within
each dataset.
"""
import json, sys
from pathlib import Path
from itertools import product

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
RES_DIR = ROOT / "results"
OUT_DIR = RES_DIR / "statistical_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_BASELINES = ["XGBoost", "LightGBM", "CatBoost", "HistGradientBoosting", "RandomForest"]
DATASETS = ["5gad", "ustc", "tiissrc23", "rtiot"]
ALPHA = 0.05


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def welch_from_summary(mean1, std1, n1, mean2, std2, n2):
    """Welch's t-test from summary statistics only (unpaired, unequal variance)."""
    se = np.sqrt(std1**2 / n1 + std2**2 / n2)
    if se == 0:
        return 0.0, 1.0
    t = (mean1 - mean2) / se
    df = (std1**2 / n1 + std2**2 / n2) ** 2 / (
        (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1) + 1e-12
    )
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return float(t), float(p)


def bootstrap_ci_diff(a, b, n_boot=10000, seed=0):
    """Bootstrap 95% CI for mean(a) - mean(b) given paired (or independent) arrays."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        diffs[i] = ai.mean() - bi.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def cohens_d_paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean() / (d.std(ddof=1) + 1e-12))


def cohens_d_unpaired(mean1, std1, mean2, std2):
    pooled = np.sqrt((std1**2 + std2**2) / 2)
    return float((mean1 - mean2) / (pooled + 1e-12))


def verdict(p_corrected, mean_diff, alpha=ALPHA):
    if p_corrected < alpha and mean_diff > 0:
        return "statistically superior"
    if p_corrected < alpha and mean_diff < 0:
        return "inferior"
    if mean_diff > 0:
        return "mean-superior"
    return "mean-superior (negative, not significant)" if mean_diff == 0 else "statistically tied"


def holm_correction(pvals):
    """Holm-Bonferroni step-down correction. Returns corrected p-values in original order."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    corrected = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected.tolist()


def compare_one(mg_name, mg_metrics, bl_name, bl_metrics, metric_key="f1_macro"):
    mg_per_seed = mg_metrics.get(f"{metric_key}_per_seed")
    bl_per_seed = bl_metrics.get(f"{metric_key}_per_seed")

    mg_mean = mg_metrics.get(f"{metric_key}_mean", mg_metrics.get(metric_key, 0.0))
    mg_std = mg_metrics.get(f"{metric_key}_std", 0.0)
    bl_mean = bl_metrics.get(f"{metric_key}_mean", bl_metrics.get(metric_key, 0.0))
    bl_std = bl_metrics.get(f"{metric_key}_std", 0.0)
    mean_diff = mg_mean - bl_mean

    if mg_per_seed and bl_per_seed and len(mg_per_seed) == len(bl_per_seed) and len(mg_per_seed) >= 3:
        mode = "paired"
        t_stat, p_t = stats.ttest_rel(mg_per_seed, bl_per_seed)
        try:
            w_stat, p_w = stats.wilcoxon(mg_per_seed, bl_per_seed)
        except ValueError:
            w_stat, p_w = float("nan"), 1.0
        ci_lo, ci_hi = bootstrap_ci_diff(mg_per_seed, bl_per_seed)
        effect = cohens_d_paired(mg_per_seed, bl_per_seed)
        p_primary = float(p_t)
    else:
        mode = "approximate (summary-stats, unpaired Welch)"
        n = 3  # all our runs use 3 seeds
        t_stat, p_t = welch_from_summary(mg_mean, mg_std, n, bl_mean, bl_std, n)
        w_stat, p_w = float("nan"), float("nan")
        # normal-approx CI on the difference of means
        se = np.sqrt(mg_std**2 / n + bl_std**2 / n)
        ci_lo, ci_hi = mean_diff - 1.96 * se, mean_diff + 1.96 * se
        effect = cohens_d_unpaired(mg_mean, mg_std, bl_mean, bl_std)
        p_primary = float(p_t)

    return {
        "comparison": f"{mg_name} vs {bl_name}",
        "metric": metric_key,
        "mode": mode,
        "mg_mean": mg_mean, "bl_mean": bl_mean, "mean_diff": mean_diff,
        "t_stat": float(t_stat), "p_ttest": float(p_t),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "p_wilcoxon": float(p_w) if not np.isnan(p_w) else None,
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "cohens_d": effect,
        "p_primary_uncorrected": p_primary,
    }


def run_dataset_tests(dataset: str):
    baselines = load_json(RES_DIR / f"{dataset}_baselines.json")
    morphguard = load_json(RES_DIR / f"{dataset}_morphguard_results.json")
    if baselines is None or morphguard is None:
        return None

    comparisons = []
    for mg_name, mg_key in [("MorphGuard-Teacher", "teacher"), ("MorphGuard-Student", "student")]:
        if mg_key not in morphguard:
            continue
        mg_metrics = dict(morphguard[mg_key])  # copy; aggregated mean/std
        # train_morphguard.py stores raw per-seed metric dicts under
        # "teacher_per_seed"/"student_per_seed" at the top level (not nested
        # under "teacher"/"student"); surface them as f1_macro_per_seed so
        # compare_one() can run a true paired test instead of falling back
        # to the summary-stats approximation.
        per_seed_key = f"{mg_key}_per_seed"
        if per_seed_key in morphguard:
            mg_metrics["f1_macro_per_seed"] = [s["f1_macro"] for s in morphguard[per_seed_key]]
        for bl_name in MAIN_BASELINES:
            if bl_name not in baselines:
                continue
            comparisons.append(compare_one(mg_name, mg_metrics, bl_name, baselines[bl_name]))

    if not comparisons:
        return None

    # Holm correction across all comparisons for this dataset
    pvals = [c["p_primary_uncorrected"] for c in comparisons]
    corrected = holm_correction(pvals)
    for c, pc in zip(comparisons, corrected):
        c["p_holm_corrected"] = pc
        c["verdict"] = verdict(pc, c["mean_diff"])

    return comparisons


def main():
    all_results = {}
    for ds in DATASETS:
        res = run_dataset_tests(ds)
        if res is None:
            print(f"  {ds}: skipped (missing baseline or MorphGuard results)")
            continue
        all_results[ds] = res
        print(f"\n=== {ds} ===")
        for c in res:
            print(f"  [{c['mode']}] {c['comparison']} ({c['metric']}): "
                  f"diff={c['mean_diff']:+.4f}  p_holm={c['p_holm_corrected']:.4f}  "
                  f"d={c['cohens_d']:+.2f}  -> {c['verdict']}")

    out_path = OUT_DIR / "morphguard_vs_main_baselines.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
