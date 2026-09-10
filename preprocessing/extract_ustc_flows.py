"""
Full PCAP-to-flow extraction for USTC-TFC2016.
Processes every .pcap file in Benign/ and Malware/ (including the Weibo/SMB
subfolders), in full -- no sampling. ~3.8 GB total; expect multi-hour
runtime with the scapy-only fallback parser (no tshark/libpcap on this
machine). Caches the raw flow table so preprocess_all.py never has to
re-run PCAP parsing.
"""
import sys, time, logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from preprocessing.pcap_to_flow import extract_flows

DATA_DIR = ROOT / "Data" / "USTC-TFC2016" / "USTC-TFC2016-master"
OUT_DIR = ROOT / "results" / "extracted_flows"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(ROOT / "logs" / "extract_ustc.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# (relative path, malware_family_or_app_label, binary_label)
MALWARE_FILES = ["Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus"]
BENIGN_FLAT = ["BitTorrent", "FTP", "Facetime", "Gmail", "MySQL", "Outlook", "Skype", "WorldOfWarcraft"]
BENIGN_SMB = ["SMB/SMB-1", "SMB/SMB-2"]
BENIGN_WEIBO = ["Weibo/Weibo-1", "Weibo/Weibo-2", "Weibo/Weibo-3", "Weibo/Weibo-4"]


def main():
    jobs = []
    for name in MALWARE_FILES:
        jobs.append((DATA_DIR / "Malware" / f"{name}.pcap", name, "Malware"))
    for name in BENIGN_FLAT:
        jobs.append((DATA_DIR / "Benign" / f"{name}.pcap", name, "Benign"))
    for rel in BENIGN_SMB:
        jobs.append((DATA_DIR / "Benign" / f"{rel}.pcap", "SMB", "Benign"))
    for rel in BENIGN_WEIBO:
        jobs.append((DATA_DIR / "Benign" / f"{rel}.pcap", "Weibo", "Benign"))

    all_rows = []
    t0 = time.time()
    for path, app_label, bin_label in jobs:
        if not path.exists():
            log.warning(f"  MISSING: {path}")
            continue
        log.info(f"Extracting {path.name} (app={app_label}, label={bin_label}) ...")
        rows = extract_flows(path, label=app_label)
        for r in rows:
            r["binary_label"] = bin_label
        all_rows.extend(rows)
        log.info(f"  Running total: {len(all_rows)} flows, elapsed {time.time()-t0:.0f}s")

    df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "ustc_tfc2016_flows.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"DONE. {len(df)} total flows saved -> {out_path}  ({time.time()-t0:.0f}s)")
    log.info(f"App-label distribution:\n{df['Label'].value_counts()}")
    log.info(f"Binary distribution:\n{df['binary_label'].value_counts()}")


if __name__ == "__main__":
    main()
