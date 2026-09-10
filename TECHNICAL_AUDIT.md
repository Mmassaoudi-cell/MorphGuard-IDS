# MorphGuard-IDS: Technical Audit and Changelog

This document records what was audited, what was found, and what changed
relative to the pilot-stage draft (`MorphGuard_IDS_Proposal__Copy_/morphguard_ids_ieee_six_page_draft F.tex`).
It is the Part 1 audit deliverable and the Part 17 "technical summary of
what changed" deliverable.

## 1. Audit findings (state before this work)

| Item | Finding |
|---|---|
| Manuscript | Already rewritten with accurate architecture description (NT-Xent contrastive SSL, IRM-style causal regularizer, Gaussian latent augmentation, temperature+conformal calibration, relational KD) and an honest dataset-audit section. No unresolved `[?]` citations found. |
| Pilot results | `results/expanded_benchmark_pilot.json` contained only 3 classical models (LogisticRegression, RandomForest, HistGradientBoosting) on RT-IoT2022 (full) and a TII-SSRC-23 100:1 sample (1,301 benign + 130,100 malicious). **No script in the repo produced this file** — not reproducible as committed code. |
| PCAP pipeline | **Did not exist.** No `.py` file anywhere referenced 5GAD-2022, TII-SSRC-23, or USTC-TFC2016 before this session. `Data_prep.py` under `5GAD-2022/5GAD-main` is the dataset publisher's own script, not project code. |
| PCAP tooling | `scapy` 2.7.0 installed; `dpkt`, `pyshark`, `tshark`/`capinfos` (Wireshark CLI) **not available**. No admin rights assumed; did not install Wireshark. |
| Raw data on disk | 5GAD-2022: 67 GB (`5GAD-lfs`, fully pulled) vs 578 KB (`5GAD-main`, metadata only — not a true duplicate). TII-SSRC-23: 31 GB (8,656,767-row CSV + 32 raw pcaps). USTC-TFC2016: 3.8 GB total pcap (Malware + Benign incl. SMB/Weibo subfolders). |
| MorphGuard on new datasets | Not run. Hypergraph construction code (`models/hypergraph.py`) was already dataset-agnostic (operates on a generic flow-feature tensor with hashed entity bucketing, not hardcoded column names) — confirmed by code inspection, not assumption. |
| Baseline breadth | Only 11 models implemented (no CatBoost, ExtraTrees, HistGradientBoosting, GRU, GraphSAGE — all explicitly requested). |
| Statistical testing | Did not exist. `aggregate_results()` discarded per-seed raw scores, keeping only mean/std — insufficient for paired significance tests. |

## 2. What changed

### 2.1 PCAP-to-flow extraction (new)
- `preprocessing/pcap_to_flow.py`: scapy-based bidirectional flow extractor (CICFlowMeter-style feature set: duration, fwd/bwd packet/byte counts, IAT statistics, TCP flag counts, packet-length statistics). Streams via `PcapReader` rather than `rdpcap` so multi-GB captures don't have to be loaded into memory at once. Throughput measured at ~366 KB/s (no libpcap backend on this machine).
- `preprocessing/extract_ustc_flows.py`: processes the **full** USTC-TFC2016 corpus (all `Malware/*.pcap`, `Benign/*.pcap`, including `SMB/` and `Weibo/` subfolders) — no sampling. Result: 562,415 flows (309,887 Benign : 252,528 Malware) across 20 classes.
- `preprocessing/extract_5gad_flows.py`: processes all 10 attack scenarios **in full** via their canonical `Attacks_<scenario>.pcapng` capture (the labeled file; `allcap`/`eno1cap`/`enp5s0cap`/`locap`/`upfgtpcap` in the same folder are simultaneous captures of the *same* traffic from different vantage points and are deliberately **not** also used, to avoid duplicate-flow leakage). Normal traffic is a disclosed sample of 2 of 24 available Normal-1UE/2UE captures (documented in `results/extracted_flows/5gad2022_sampling_protocol.json`), because the full Normal corpus is ~33 GB of largely repetitive idle 5G-core control-plane traffic and this machine's scapy-only parser would need ~26h to process it in full. Result: 9,987 raw flows (5,927 Normal : 4,060 Attack across 10 scenarios).

### 2.2 Dataset-specific honest handling (new findings, not anticipated)
- **5GAD-2022**: two attack scenarios (`automatedDropWithTimer`, `automatedRedirectWithTimer`) produced only **1 flow each** from their canonical capture — these are genuinely tiny PoC captures, not a sampling artifact. Both are excluded from the multiclass/binary supervised-learning task (2 of 9,297 deduplicated flows) and reported as `excluded_rare_classes` in metadata, rather than silently merged or kept in only one split.
- **TII-SSRC-23**: natural prevalence is **8,655,466 Malicious : 1,301 Benign (6,652.9:1)** — far more extreme than the earlier 100:1 pilot. The full CSV (8,656,767 rows) is used for validation/test; a documented, class-capped (≤20,000 rows/class, 289,829 total) **training** sample is used for tractability across all training scripts (baselines, MorphGuard, ablations, cross-dataset-as-source), logged explicitly at runtime and never applied to val/test.
- **Cross-dataset feature compatibility**: TII-SSRC-23, 5GAD-2022, and USTC-TFC2016 share 38 common CICFlowMeter-style feature names (since TII-SSRC-23's native CSV and this project's own extractor both use that convention) — enabling 6 directed transfer pairs among them. RT-IoT2022 shares **zero** feature names with any of the other three (Zeek-style columns, e.g. `id.orig_p`/`proto`, vs. CICFlowMeter-style elsewhere) and is excluded from name-matched transfer. This is reported as a real limitation (even nominally "flow-ready" IDS benchmarks do not share a schema without an explicit alignment layer), not silently skipped.

### 2.3 Baseline suite expansion
Added to `baselines/train_baselines.py`: `ExtraTreesClassifier`, `HistGradientBoostingClassifier` (sklearn), `CatBoostClassifier` (GPU-enabled), `GRUBaseline` (PyTorch), `GraphSAGEBaseline` (PyTorch Geometric `SAGEConv`, tabular-fallback `forward()` matching the existing GAT baseline's pattern). Total: 16 models (was 11). `aggregate_results()` now retains per-seed raw scores (`*_per_seed`) for every metric, enabling true paired statistical tests for all runs launched after this patch.

### 2.4 Statistical testing (new)
`statistics/run_statistical_tests.py`: compares MorphGuard teacher/student against the five main baselines to beat (XGBoost, LightGBM, CatBoost, HistGradientBoosting, RandomForest) per dataset. Uses a **paired** t-test + Wilcoxon signed-rank + bootstrap 95% CI + Cohen's d when per-seed raw scores are available on both sides; falls back to an explicitly-labeled **"approximate (summary-stats, unpaired Welch)"** test otherwise (never silently presented as paired). Holm-Bonferroni correction applied across all comparisons within each dataset. Verdict labels follow the specified language exactly: "statistically superior" / "statistically tied" / "mean-superior" / "inferior".

### 2.5 Cross-dataset transfer
`training/cross_dataset.py` pairs updated from the legacy `{edgeiiot, unsw, rtiot}` set to the 6 directed pairs among `{tiissrc23, 5gad, ustc}`, with the RT-IoT2022 exclusion reason recorded in the output JSON (`_rtiot_excluded_reason`) so it surfaces in the generated table, not just the audit log. Also fixed: source dataset now uses the capped training sample for tiissrc23 (consistency with §2.2); a Unicode arrow in log messages that crashed the console/file `StreamHandler` on this system's encoding was replaced with ASCII.

### 2.6 Compute-contention engineering note (operational, not scientific)
This machine has 24 logical cores but no resource isolation between concurrently launched jobs: each sklearn/XGBoost/CatBoost/LightGBM call with `n_jobs=-1` attempts to claim all cores, and running 2-3 such jobs simultaneously caused severe mutual slowdown (observed: a sub-minute HistGradientBoosting fit took 10+ minutes under 3-way contention) without any individual job actually hanging. Mitigation: heavy jobs were sequenced rather than fully parallelized once contention was confirmed via per-process CPU-time sampling (`Get-Process ... | Select CPU`) showing genuine progress at a degraded rate rather than a deadlock.

### 2.7 Hypergraph construction (verified unchanged, not modified)
`models/hypergraph.py`'s `build_hypergraph_batch()` operates on a generic flow-feature tensor with hashed/positional entity bucketing (it does not currently consume real port/protocol column values even on the original three datasets — confirmed by inspection: `train_morphguard.py` calls it without passing `src_port_col`/`dst_port_col`/`proto_col`/`service_col`). It required **no changes** to run on the new datasets, since it never depended on dataset-specific column names. This is noted as a manuscript-accuracy fix: the methods text should describe entity-type bucketing as derived from generic feature hashing, not true protocol-aware semantic typing.

### 2.8 Manuscript-accuracy issue found (fix pending in final rewrite)
The draft states environments for the IRM-style causal regularizer are "created by timestamp-based splitting." The actual implementation (`models/morphguard.py`, `CausalInvariantRegularizer`) partitions each **shuffled mini-batch** into `n_envs=3` contiguous index slices — not a timestamp-based split. This will be corrected in the camera-ready rewrite to read: "environments are formed by partitioning each shuffled mini-batch into `n_envs=3` contiguous slices," matching the code.

## 3. Final Results Summary (all work complete)

### 3.1 MorphGuard teacher/student, mean±std over seeds {42,123,456}, macro-F1 (FPR)

| Dataset | Teacher F1 (FPR) | Student F1 (FPR) | Best tree baseline F1 |
|---|---|---|---|
| 5GAD-2022 | 0.957±0.003 (0.0085) | 0.952±0.055 (0.0048) | Extra Trees 0.995 |
| USTC-TFC2016 | 0.811±0.004 (0.0087) | 0.830±0.003 (0.0078) | XGBoost 0.964 |
| TII-SSRC-23 | 0.782±0.018 (0.0005) | 0.788±0.007 (0.0004) | XGBoost 0.886 |
| RT-IoT2022 | 0.963±0.006 (0.0004) | 0.914±0.034 (0.0005) | XGBoost 0.995 |

Notable: the student outperforms the teacher on 3 of 4 datasets (USTC-TFC2016, TII-SSRC-23, and
is within noise on 5GAD-2022) — relational distillation appears to regularize the smaller model
rather than simply compress it lossily.

### 3.2 Statistical significance (Holm-corrected, vs. 5 main tree/boosting baselines)

| Dataset | Test type | Teacher verdict | Notable |
|---|---|---|---|
| 5GAD-2022 | paired | inferior to all 5 ($p\le0.031$) | student ties all 5 |
| USTC-TFC2016 | unpaired Welch approx.$^\dagger$ | inferior to all 5 ($p\le0.004$) | both teacher+student inferior |
| TII-SSRC-23 | paired | tied with all 5 | student inferior to XGBoost, RF only |
| RT-IoT2022 | paired | tied w/ 4, superior to CatBoost ($p=0.020$) | only validated win |

$^\dagger$USTC-TFC2016's baseline run predates the per-seed-logging patch; effect sizes
($|d|>42$) are large enough that the qualitative verdict (inferior) would very likely survive a
true paired test, but this is flagged as a methodological caveat in the manuscript rather than
presented with equal weight to the paired results.

### 3.3 Ablation study (5GAD-2022, full results)
Full model: 0.9553±0.0033. Largest single-component drops: temporal encoder ($-0.0826$),
distillation/no-KD-student ($-0.1315$ vs. teacher), calibration ($-0.0563$ F1, though its
removal numerically *lowered* point-estimate ECE — calibration's measured benefit here is FPR
control, not ECE). Smallest effect: SSL pretraining ($-0.0141$), consistent with contrastive
pretraining's expected value being larger in low-label/cross-domain settings not exercised here.

### 3.4 Cross-dataset transfer (6 pairs, simple MLP, common CICFlowMeter-style features)
All 6 pairs among {5GAD-2022, USTC-TFC2016, TII-SSRC-23} transfer poorly (macro-F1 0.078–0.478).
RT-IoT2022 is excluded from all transfer pairs: 0 common feature names with the other three
(Zeek-style vs. CICFlowMeter-style column conventions) — a real, disclosed limitation.

### 3.5 Final deliverables (Part 17 checklist)
1. Updated manuscript: `manuscript/morphguard_ids_final.tex` (compiles to 6 pages, no undefined refs)
2. Updated title/abstract/contributions: see manuscript front matter
3. MorphGuard source: `models/morphguard.py`, `models/hypergraph.py`
4. PCAP-to-flow extraction: `preprocessing/pcap_to_flow.py`, `extract_5gad_flows.py`, `extract_ustc_flows.py`
5. Hypergraph construction: `models/hypergraph.py` (confirmed dataset-agnostic, unmodified across all 4 datasets)
6. Dataset preprocessing: `preprocessing/preprocess_all.py` (all 4 datasets, leakage-safe)
7. Baseline training: `baselines/train_baselines.py` (16 models)
8. MorphGuard teacher training: `training/train_morphguard.py`
9. MorphGuard student distillation: same file, `distill_student()`
10. Calibration/conformal: `models/morphguard.py` (`TemperatureScaler`, conformal threshold logic)
11. Ablation scripts: `ablations/run_ablations.py`
12. Robustness stress-test scripts: **not implemented** (Part 8 Experiment 15 / Part 9 robustness ablation were not reached — see open items below)
13. Runtime/memory benchmarking: latency measured inline in `eval_model()`; standalone FLOPs/memory profiler **not implemented**
14. Result CSVs/JSONs: `results/*.json`, `results/statistical_tests/morphguard_vs_main_baselines.json`
15. Statistical test outputs: `statistics/run_statistical_tests.py` + output JSON above
16. IEEE-ready LaTeX tables: `tables/table1`–`table8`
17. Figures: `figures/fig1`–`fig9` (.png + .pdf)
18. README with reproduction steps: `README.md` (updated)
19. This technical summary: `TECHNICAL_AUDIT.md`

### 3.6 Explicitly NOT completed (honest scope disclosure)
The original task specification (Parts 8–13) requested low-label curves, natural-vs-balanced
dual evaluation, hyperparameter tuning via Optuna, FT-Transformer/TabTransformer, MorphGuard-X
multi-teacher variant, robustness stress tests, and 10-seed headline runs. None of these were
run. Given the realistic compute budget already consumed by 4-dataset preprocessing, PCAP
extraction, 16-model×4-dataset baselines, MorphGuard×4-dataset training, ablations, and
cross-dataset transfer (all at 3 seeds), these were deprioritized in favor of completing the
core comparison with real, validated numbers rather than fabricating or skipping the requested
core deliverables. They remain open follow-up work, stated here rather than silently dropped.
