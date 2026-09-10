"""
MorphGuard-IDS Full Pipeline Runner.
Orchestrates: preprocessing -> baselines -> teacher -> student -> ablations -> figures -> tables -> manuscript.
"""

import os, sys, json, logging, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

DATASETS = ["edgeiiot", "rtiot", "unsw"]
SEEDS = [42, 123, 456]


def step(name, fn):
    log.info(f"\n{'='*70}")
    log.info(f"STEP: {name}")
    log.info(f"{'='*70}")
    t0 = time.time()
    result = fn()
    log.info(f"STEP COMPLETE: {name}  ({time.time()-t0:.1f}s)")
    return result


def main():
    # 1. Preprocessing
    step("Preprocessing all datasets", lambda: __import__(
        "preprocessing.preprocess_all", fromlist=["preprocess_edgeiiot", "preprocess_rtiot", "preprocess_unsw"]
    ) and __run_preprocessing())

    # 2. Baselines
    for ds in DATASETS:
        step(f"Baselines on {ds}", lambda d=ds: __run_baselines(d))

    # 3. MorphGuard teacher + student
    for ds in DATASETS:
        step(f"MorphGuard on {ds}", lambda d=ds: __run_morphguard(d))

    # 4. Ablations (on primary dataset)
    step("Ablations on edgeiiot", lambda: __run_ablations("edgeiiot"))

    # 5. Figures
    step("Generate figures", lambda: __run_figures())

    # 6. Tables
    step("Generate tables", lambda: __run_tables())

    # 7. Manuscript
    step("Complete manuscript", lambda: __run_manuscript())

    log.info("\nFULL PIPELINE COMPLETE")


def __run_preprocessing():
    from preprocessing.preprocess_all import preprocess_edgeiiot, preprocess_rtiot, preprocess_unsw
    preprocess_edgeiiot()
    preprocess_rtiot()
    preprocess_unsw()


def __run_baselines(dataset_name: str):
    from baselines.train_baselines import run_baselines
    run_baselines(dataset_name, seeds=SEEDS)


def __run_morphguard(dataset_name: str):
    from training.train_morphguard import run_dataset
    run_dataset(dataset_name, seeds=SEEDS)


def __run_ablations(dataset_name: str):
    from ablations.run_ablations import run_ablations
    run_ablations(dataset_name, seeds=SEEDS)


def __run_figures():
    from figures.generate_figures import generate_all_figures
    generate_all_figures()


def __run_tables():
    from tables.generate_tables import generate_all_tables
    generate_all_tables()


def __run_manuscript():
    from manuscript.update_manuscript import update_manuscript
    update_manuscript()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all",
        choices=["all", "preprocess", "baselines", "morphguard", "ablations", "figures", "tables", "manuscript"])
    parser.add_argument("--dataset", default="all", choices=["all"] + DATASETS)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    if args.step == "all":
        main()
    elif args.step == "preprocess":
        step("Preprocessing", __run_preprocessing)
    elif args.step == "baselines":
        for ds in datasets:
            step(f"Baselines {ds}", lambda d=ds: __run_baselines(d))
    elif args.step == "morphguard":
        for ds in datasets:
            step(f"MorphGuard {ds}", lambda d=ds: __run_morphguard(d))
    elif args.step == "ablations":
        for ds in datasets:
            step(f"Ablations {ds}", lambda d=ds: __run_ablations(d))
    elif args.step == "figures":
        step("Figures", __run_figures)
    elif args.step == "tables":
        step("Tables", __run_tables)
    elif args.step == "manuscript":
        step("Manuscript", __run_manuscript)
