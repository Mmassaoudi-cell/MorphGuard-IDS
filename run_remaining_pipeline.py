"""
Run all remaining pipeline steps sequentially:
1. MorphGuard training: rtiot, unsw (edgeiiot already running separately)
2. Ablation study on edgeiiot
3. Cross-dataset generalization experiments
4. Generate figures and tables
5. Update manuscript

Run AFTER edgeiiot MorphGuard training completes.
"""

import sys, json, logging, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "remaining_pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SEEDS = [42, 123, 456]
RES_DIR = ROOT / "results"


def wait_for_result(path: Path, poll_sec: int = 30, timeout_sec: int = 7200):
    """Poll until a result JSON file exists."""
    t0 = time.time()
    while not path.exists():
        elapsed = time.time() - t0
        if elapsed > timeout_sec:
            log.error(f"Timeout waiting for {path}")
            return False
        log.info(f"  Waiting for {path.name}... ({elapsed/60:.1f} min elapsed)")
        time.sleep(poll_sec)
    log.info(f"  Found: {path}")
    return True


def step(name, fn, *args, **kwargs):
    log.info(f"\n{'='*70}")
    log.info(f"STEP: {name}")
    log.info(f"{'='*70}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    log.info(f"STEP COMPLETE: {name}  ({time.time()-t0:.1f}s)")
    return result


def run_morphguard(dataset_name: str):
    from training.train_morphguard import run_dataset
    out_path = RES_DIR / f"{dataset_name}_morphguard_results.json"
    if out_path.exists():
        log.info(f"  MorphGuard {dataset_name} already done, skipping.")
        return
    log.info(f"  Training MorphGuard on {dataset_name}...")
    run_dataset(dataset_name, seeds=SEEDS)


def run_ablations_step():
    from ablations.run_ablations import run_ablations
    out_path = RES_DIR / "edgeiiot_ablations.json"
    if out_path.exists():
        log.info("  Ablations already done, skipping.")
        return
    log.info("  Running ablation study on edgeiiot...")
    run_ablations("edgeiiot", seeds=SEEDS)


def run_cross_dataset_step():
    out_path = RES_DIR / "cross_dataset_results.json"
    if out_path.exists():
        log.info("  Cross-dataset results already exist, skipping.")
        return
    from training.cross_dataset import run_all_cross_dataset
    log.info("  Running cross-dataset generalization experiments...")
    run_all_cross_dataset(seeds=SEEDS)


def finalize():
    log.info("  Generating all figures...")
    from figures.generate_figures import generate_all_figures
    generate_all_figures()

    log.info("  Generating all LaTeX tables...")
    from tables.generate_tables import generate_all_tables
    generate_all_tables()

    log.info("  Writing BibTeX and populating manuscript [PENDING] markers...")
    from manuscript.update_manuscript import write_bibtex, populate_pending, load_results
    write_bibtex()
    _, morphguard, _ = load_results()
    populate_pending(morphguard)

    log.info("  Figures, tables, and manuscript updated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-edgeiiot", action="store_true",
                        help="Wait for edgeiiot MorphGuard to complete first")
    parser.add_argument("--skip-rtiot", action="store_true")
    parser.add_argument("--skip-unsw", action="store_true")
    parser.add_argument("--only-finalize", action="store_true")
    args = parser.parse_args()

    if args.only_finalize:
        step("Generate figures + tables", finalize)
        sys.exit(0)

    # Wait for edgeiiot MorphGuard if requested
    if args.wait_for_edgeiiot:
        log.info("Waiting for edgeiiot MorphGuard training to complete...")
        wait_for_result(RES_DIR / "edgeiiot_morphguard_results.json")

    # Train remaining datasets
    if not args.skip_rtiot:
        step("MorphGuard on RT-IoT2022", run_morphguard, "rtiot")

    if not args.skip_unsw:
        step("MorphGuard on UNSW-NB15", run_morphguard, "unsw")

    # Ablation study (requires edgeiiot MorphGuard results)
    step("Ablation Study", run_ablations_step)

    # Cross-dataset generalization
    step("Cross-Dataset Generalization", run_cross_dataset_step)

    # Generate all outputs
    step("Generate Figures + Tables", finalize)

    log.info("\n" + "="*70)
    log.info("REMAINING PIPELINE COMPLETE")
    log.info("Results are in results/ directory")
    log.info("Figures are in figures/ directory")
    log.info("Tables are in tables/ directory")
    log.info("Run 'python manuscript/update_manuscript.py' to finalize the manuscript")
    log.info("="*70)
