"""
PCAP-to-flow extraction for 5GAD-2022.

SAMPLING PROTOCOL (documented honestly, per the no-fabrication rules):
  - All 10 attack scenarios are used IN FULL via their canonical
    "Attacks_<scenario>.pcapng" capture (the single labeled file per
    scenario; the allcap/eno1cap/enp5s0cap/locap/upfgtpcap files in the
    same folder are redundant simultaneous captures of the SAME traffic
    from different vantage points/interfaces and are NOT separately
    used, to avoid duplicate-flow leakage). Combined attack captures
    total ~15 MB.
  - Normal traffic is a SAMPLE of 2 of 7 available Normal-1UE captures
    (allcap_00005_*, 887 MB; allcap_00006_*, 97 MB), chosen because the
    full Normal-1UE + Normal-2UE corpus is ~33 GB of largely repetitive
    idle 5G-core control-plane traffic, and this machine has no
    tshark/libpcap backend (scapy-only fallback parser, ~366 KB/s),
    making full processing impractical for a single session (would take
    ~26 hours). The 2 files used are stated explicitly so any reviewer
    can reproduce or extend the sample.
  - This is reported in the manuscript as a STATED sampling protocol,
    not presented as full-corpus evidence.
"""
import sys, time, logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from preprocessing.pcap_to_flow import extract_flows

DATA_DIR = ROOT / "Data" / "5GAD-2022" / "5GAD-lfs"
OUT_DIR = ROOT / "results" / "extracted_flows"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(ROOT / "logs" / "extract_5gad.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ATTACK_SCENARIOS = [
    "AMFLookingForUDM", "automatedDropWithTimer", "automatedRedirectWithTimer",
    "CrashNRF", "FakeAMFDelete", "FakeAMFInsert", "GetAllNFs", "GetUserData",
    "randomAMFInsert", "randomDataDump",
]
NORMAL_SAMPLE_FILES = [
    "Normal-1UE/allcap_00005_20220607051008.pcapng",
    "Normal-1UE/allcap_00006_20220607091008.pcapng",
]


def main():
    all_rows = []
    t0 = time.time()

    for scenario in ATTACK_SCENARIOS:
        path = DATA_DIR / "Attacks" / scenario / f"Attacks_{scenario}.pcapng"
        if not path.exists():
            log.warning(f"  MISSING: {path}")
            continue
        log.info(f"Extracting attack scenario {scenario} ...")
        rows = extract_flows(path, label=scenario)
        for r in rows:
            r["binary_label"] = "Attack"
        all_rows.extend(rows)
        log.info(f"  Running total: {len(all_rows)} flows, elapsed {time.time()-t0:.0f}s")

    for rel in NORMAL_SAMPLE_FILES:
        path = DATA_DIR / rel
        if not path.exists():
            log.warning(f"  MISSING: {path}")
            continue
        log.info(f"Extracting Normal sample {rel} ...")
        rows = extract_flows(path, label="Normal")
        for r in rows:
            r["binary_label"] = "Normal"
        all_rows.extend(rows)
        log.info(f"  Running total: {len(all_rows)} flows, elapsed {time.time()-t0:.0f}s")

    df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "5gad2022_flows.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"DONE. {len(df)} total flows saved -> {out_path}  ({time.time()-t0:.0f}s)")
    log.info(f"Scenario distribution:\n{df['Label'].value_counts()}")
    log.info(f"Binary distribution:\n{df['binary_label'].value_counts()}")

    # Save the sampling protocol alongside the data for the manuscript/audit
    protocol = {
        "attack_scenarios_used": ATTACK_SCENARIOS,
        "attack_capture_type": "canonical Attacks_<scenario>.pcapng only (not allcap/eno1cap/enp5s0cap/locap/upfgtpcap duplicates)",
        "normal_files_used": NORMAL_SAMPLE_FILES,
        "normal_files_available_but_not_used": 7 + 17 - len(NORMAL_SAMPLE_FILES),
        "reason": "no tshark/libpcap backend; scapy-only fallback parser throughput ~366 KB/s; full 33GB Normal-1UE+2UE corpus would require ~26h",
    }
    import json
    with open(OUT_DIR / "5gad2022_sampling_protocol.json", "w") as f:
        json.dump(protocol, f, indent=2)


if __name__ == "__main__":
    main()
