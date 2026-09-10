"""
PCAP/PCAPNG -> bidirectional flow-feature extraction (CICFlowMeter-style).

Used for the two PCAP-first datasets in the revised benchmark suite
(USTC-TFC2016, 5GAD-2022) where no flow-ready CSV exists. Implemented with
scapy (the only PCAP library confirmed installed on this machine; no
tshark/dpkt/pyshark available), streaming packet-by-packet via PcapReader
so multi-GB captures do not have to be loaded into memory at once.

Flow key is the canonicalized 5-tuple (unordered src/dst so A->B and B->A
packets land in the same flow). A flow is closed and re-opened after
FLOW_TIMEOUT seconds of inactivity, matching common flow-exporter behavior
(CICFlowMeter default is 120s; this affects flow *count*, not feature
honesty, and is reported in the manuscript).
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, IPv6, TCP, UDP, SCTP

log = logging.getLogger(__name__)

FLOW_TIMEOUT = 120.0  # seconds, CICFlowMeter convention


class _FlowAccumulator:
    __slots__ = (
        "src_ip", "dst_ip", "src_port", "dst_port", "proto",
        "start_t", "last_t",
        "fwd_lens", "bwd_lens", "fwd_times", "bwd_times", "all_times",
        "fin", "syn", "rst", "psh", "ack", "urg",
    )

    def __init__(self, src_ip, dst_ip, src_port, dst_port, proto, t0):
        self.src_ip, self.dst_ip = src_ip, dst_ip
        self.src_port, self.dst_port = src_port, dst_port
        self.proto = proto
        self.start_t = t0
        self.last_t = t0
        self.fwd_lens, self.bwd_lens = [], []
        self.fwd_times, self.bwd_times, self.all_times = [], [], []
        self.fin = self.syn = self.rst = self.psh = self.ack = self.urg = 0

    def add(self, is_fwd, length, t, flags):
        if is_fwd:
            self.fwd_lens.append(length)
            self.fwd_times.append(t)
        else:
            self.bwd_lens.append(length)
            self.bwd_times.append(t)
        self.all_times.append(t)
        self.last_t = t
        if flags:
            if "F" in flags: self.fin += 1
            if "S" in flags: self.syn += 1
            if "R" in flags: self.rst += 1
            if "P" in flags: self.psh += 1
            if "A" in flags: self.ack += 1
            if "U" in flags: self.urg += 1

    def to_row(self, label=None, src_file=None):
        dur = max(self.last_t - self.start_t, 1e-6)
        fwd = np.asarray(self.fwd_lens, dtype=np.float64)
        bwd = np.asarray(self.bwd_lens, dtype=np.float64)
        allp = np.asarray(self.fwd_lens + self.bwd_lens, dtype=np.float64)
        iat = np.diff(sorted(self.all_times)) if len(self.all_times) > 1 else np.array([0.0])
        fwd_iat = np.diff(sorted(self.fwd_times)) if len(self.fwd_times) > 1 else np.array([0.0])
        bwd_iat = np.diff(sorted(self.bwd_times)) if len(self.bwd_times) > 1 else np.array([0.0])

        def s(x, f, default=0.0):
            return float(f(x)) if len(x) else default

        row = {
            "Src Port": self.src_port, "Dst Port": self.dst_port, "Protocol": self.proto,
            "Flow Duration": dur * 1e6,  # microseconds, matches CICFlowMeter convention
            "Total Fwd Packet": len(fwd), "Total Bwd packets": len(bwd),
            "Total Length of Fwd Packet": s(fwd, np.sum), "Total Length of Bwd Packet": s(bwd, np.sum),
            "Fwd Packet Length Max": s(fwd, np.max), "Fwd Packet Length Min": s(fwd, np.min),
            "Fwd Packet Length Mean": s(fwd, np.mean), "Fwd Packet Length Std": s(fwd, np.std),
            "Bwd Packet Length Max": s(bwd, np.max), "Bwd Packet Length Min": s(bwd, np.min),
            "Bwd Packet Length Mean": s(bwd, np.mean), "Bwd Packet Length Std": s(bwd, np.std),
            "Flow Bytes/s": float(allp.sum() / dur), "Flow Packets/s": float(len(allp) / dur),
            "Flow IAT Mean": s(iat, np.mean), "Flow IAT Std": s(iat, np.std),
            "Flow IAT Max": s(iat, np.max), "Flow IAT Min": s(iat, np.min),
            "Fwd IAT Mean": s(fwd_iat, np.mean), "Fwd IAT Std": s(fwd_iat, np.std),
            "Bwd IAT Mean": s(bwd_iat, np.mean), "Bwd IAT Std": s(bwd_iat, np.std),
            "Packet Length Min": s(allp, np.min), "Packet Length Max": s(allp, np.max),
            "Packet Length Mean": s(allp, np.mean), "Packet Length Std": s(allp, np.std),
            "FIN Flag Count": self.fin, "SYN Flag Count": self.syn, "RST Flag Count": self.rst,
            "PSH Flag Count": self.psh, "ACK Flag Count": self.ack, "URG Flag Count": self.urg,
            "Down/Up Ratio": float(len(bwd) / max(len(fwd), 1)),
            "Average Packet Size": s(allp, np.mean),
            "Subflow Fwd Packets": len(fwd), "Subflow Bwd Packets": len(bwd),
        }
        if label is not None:
            row["Label"] = label
        if src_file is not None:
            row["__src_file"] = src_file
        return row


def extract_flows(pcap_path, label=None, max_packets=None, flow_timeout=FLOW_TIMEOUT):
    """Stream a single pcap/pcapng file into bidirectional flow-feature rows.

    Returns a list of dict rows (one per closed flow). `label` is attached
    to every row from this file (caller maps file -> attack/benign class).
    `__src_file` is kept so capture-disjoint splitting is possible later.
    """
    pcap_path = str(pcap_path)
    flows = {}
    closed_rows = []
    n_packets = 0
    n_ip_packets = 0

    try:
        reader = PcapReader(pcap_path)
    except Exception as e:
        log.warning(f"  Failed to open {pcap_path}: {e}")
        return []

    for pkt in reader:
        n_packets += 1
        if max_packets and n_packets > max_packets:
            break
        t = float(pkt.time)

        if IP in pkt:
            src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
        elif IPv6 in pkt:
            src_ip, dst_ip = pkt[IPv6].src, pkt[IPv6].dst
        else:
            continue
        n_ip_packets += 1

        if TCP in pkt:
            proto, sport, dport = 6, int(pkt[TCP].sport), int(pkt[TCP].dport)
            flags = str(pkt[TCP].flags)
        elif UDP in pkt:
            proto, sport, dport = 17, int(pkt[UDP].sport), int(pkt[UDP].dport)
            flags = None
        elif SCTP in pkt:
            proto, sport, dport = 132, int(pkt[SCTP].sport), int(pkt[SCTP].dport)
            flags = None
        else:
            proto, sport, dport = int(pkt.proto) if hasattr(pkt, "proto") else 0, 0, 0
            flags = None

        length = len(pkt)

        # canonical undirected key so A->B and B->A share one flow
        fwd_key = (src_ip, dst_ip, sport, dport, proto)
        bwd_key = (dst_ip, src_ip, dport, sport, proto)
        if fwd_key in flows:
            key, is_fwd = fwd_key, True
        elif bwd_key in flows:
            key, is_fwd = bwd_key, False
        else:
            key, is_fwd = fwd_key, True

        acc = flows.get(key)
        if acc is not None and (t - acc.last_t) > flow_timeout:
            closed_rows.append(acc.to_row(label=label, src_file=Path(pcap_path).name))
            acc = None

        if acc is None:
            acc = _FlowAccumulator(src_ip, dst_ip, sport, dport, proto, t)
            flows[key] = acc
            is_fwd = True

        acc.add(is_fwd, length, t, flags)

    reader.close()

    for acc in flows.values():
        closed_rows.append(acc.to_row(label=label, src_file=Path(pcap_path).name))

    log.info(f"  {Path(pcap_path).name}: {n_packets} packets ({n_ip_packets} IP), "
             f"{len(closed_rows)} flows")
    return closed_rows


def extract_dataset(file_label_pairs, flow_timeout=FLOW_TIMEOUT, max_packets_per_file=None):
    """file_label_pairs: list of (path, label) -> concatenated flow DataFrame."""
    all_rows = []
    for path, label in file_label_pairs:
        rows = extract_flows(path, label=label, max_packets=max_packets_per_file,
                              flow_timeout=flow_timeout)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)
