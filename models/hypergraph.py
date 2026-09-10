"""
Typed Cyber-Physical Hypergraph Construction for MorphGuard-IDS.

For each tabular network flow sample, we construct a heterogeneous
hypergraph where:
  Nodes = entity types (src_port_group, dst_port_group, protocol, service,
           traffic_cluster, temporal_segment)
  Hyperedges = typed relations (comm_event, session, burst_event, probe_event)

We implement the HNHN-style bipartite representation:
  - Incidence matrix H: (n_nodes x n_hyperedges)
  - Node features X: (n_nodes x d_node)
  - Hyperedge features E: (n_hyperedges x d_edge)
  - Apply attention-based HGNN propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Entity vocabulary (shared across datasets)
# ---------------------------------------------------------------------------

PORT_GROUPS = ["well_known", "registered", "dynamic", "unknown"]
PROTOCOLS = ["tcp", "udp", "icmp", "arp", "dns", "mqtt", "http", "other"]
SERVICES = ["http", "https", "dns", "ftp", "ssh", "smtp", "mqtt", "modbus", "other", "unknown"]
TRAFFIC_CLUSTERS = [f"cluster_{i}" for i in range(8)]
TEMPORAL_SEGS = [f"tseg_{i}" for i in range(6)]  # 4-hour blocks


def make_vocab():
    """Return ordered entity lists and offsets for node indexing."""
    entities = {
        "src_port_grp": PORT_GROUPS,
        "dst_port_grp": PORT_GROUPS,
        "protocol": PROTOCOLS,
        "service": SERVICES,
        "traffic_cluster": TRAFFIC_CLUSTERS,
        "temporal_seg": TEMPORAL_SEGS,
    }
    offsets = {}
    idx = 0
    for etype, lst in entities.items():
        offsets[etype] = {name: idx + i for i, name in enumerate(lst)}
        idx += len(lst)
    total = idx
    return entities, offsets, total


ENTITY_VOCAB, NODE_OFFSETS, N_NODES = make_vocab()


def port_to_group(port_val) -> str:
    try:
        p = int(float(port_val))
        if p <= 1023:
            return "well_known"
        elif p <= 49151:
            return "registered"
        else:
            return "dynamic"
    except Exception:
        return "unknown"


def proto_to_cat(proto_val) -> str:
    p = str(proto_val).lower().strip()
    for name in PROTOCOLS[:-1]:
        if name in p:
            return name
    return "other"


def service_to_cat(svc_val) -> str:
    s = str(svc_val).lower().strip()
    for name in SERVICES[:-2]:
        if name in s:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Hyperedge type enum
# ---------------------------------------------------------------------------

HYPEREDGE_TYPES = {
    "comm_event": 0,     # src_port + dst_port + proto + service
    "session": 1,         # src + dst + proto (bidir session)
    "burst_event": 2,     # traffic_cluster + temporal_seg + proto
    "probe_event": 3,     # dst_port (well_known) + proto + temporal
}
N_EDGE_TYPES = len(HYPEREDGE_TYPES)


# ---------------------------------------------------------------------------
# Batch hypergraph construction from a feature tensor
# ---------------------------------------------------------------------------

@dataclass
class HypergraphBatch:
    """Sparse representation of a batch of hypergraphs."""
    node_features: torch.Tensor          # (N_NODES, d_node) — shared vocab embeddings
    edge_features: torch.Tensor          # (B * N_EDGE_TYPES, d_edge) — per-sample edge feats
    incidence_row: torch.Tensor          # node indices (for sparse incidence)
    incidence_col: torch.Tensor          # edge indices
    incidence_val: torch.Tensor          # 1.0 for membership
    n_nodes: int
    n_edges: int
    batch_size: int
    # Per-sample assignments
    edge_to_sample: torch.Tensor         # (n_edges,) — which sample each edge belongs to
    edge_type: torch.Tensor              # (n_edges,) — type index
    # Flow features (raw tabular, for MLP branch)
    flow_features: torch.Tensor          # (B, d_flow)


def build_hypergraph_batch(
    flow_feats: torch.Tensor,
    src_port_col: Optional[torch.Tensor] = None,
    dst_port_col: Optional[torch.Tensor] = None,
    proto_col: Optional[torch.Tensor] = None,
    service_col: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> HypergraphBatch:
    """
    Build a batched hypergraph from a batch of flow feature vectors.

    This is a simplified typed hypergraph where for each sample we create
    4 typed hyperedges connecting entity nodes.
    """
    B = flow_feats.shape[0]
    d_flow = flow_feats.shape[1]

    # Each sample has N_EDGE_TYPES hyperedges
    total_edges = B * N_EDGE_TYPES

    # -----------------------------------------------------------------------
    # Determine which entity node each sample maps to
    # For simplicity we use dummy assignments when port/proto/service not given
    # -----------------------------------------------------------------------

    def safe_col(col, B, default_val=0):
        if col is None:
            return torch.zeros(B, dtype=torch.long)
        c = col.long()
        return c % 4  # clamp to group range (0-3 for port groups)

    src_grp = safe_col(src_port_col, B) % len(PORT_GROUPS)
    dst_grp = safe_col(dst_port_col, B) % len(PORT_GROUPS)
    proto_idx = safe_col(proto_col, B) % len(PROTOCOLS)
    svc_idx = safe_col(service_col, B) % len(SERVICES)
    # Assign traffic cluster: simple hash from flow features
    cluster_idx = (flow_feats[:, 0].abs() * 8).long() % len(TRAFFIC_CLUSTERS)
    tseg_idx = torch.zeros(B, dtype=torch.long)  # default temporal seg

    # -----------------------------------------------------------------------
    # Build incidence matrix entries
    # -----------------------------------------------------------------------
    # Node global IDs
    src_node = torch.tensor([NODE_OFFSETS["src_port_grp"][PORT_GROUPS[g]] for g in src_grp.tolist()], dtype=torch.long)
    dst_node = torch.tensor([NODE_OFFSETS["dst_port_grp"][PORT_GROUPS[g]] for g in dst_grp.tolist()], dtype=torch.long)
    proto_node = torch.tensor([NODE_OFFSETS["protocol"][PROTOCOLS[g]] for g in proto_idx.tolist()], dtype=torch.long)
    svc_node = torch.tensor([NODE_OFFSETS["service"][SERVICES[g]] for g in svc_idx.tolist()], dtype=torch.long)
    cluster_node = torch.tensor([NODE_OFFSETS["traffic_cluster"][TRAFFIC_CLUSTERS[g]] for g in cluster_idx.tolist()], dtype=torch.long)
    tseg_node = torch.tensor([NODE_OFFSETS["temporal_seg"][TEMPORAL_SEGS[g]] for g in tseg_idx.tolist()], dtype=torch.long)

    # Edge IDs: sample i has edges [i*N_EDGE_TYPES ... i*N_EDGE_TYPES+3]
    # comm_event connects: src, dst, proto, service
    # session connects: src, dst, proto
    # burst_event connects: cluster, temporal, proto
    # probe_event connects: dst, proto, temporal

    row_list = []
    col_list = []

    for i in range(B):
        base = i * N_EDGE_TYPES
        e_comm = base + 0
        e_sess = base + 1
        e_burst = base + 2
        e_probe = base + 3

        # comm_event: src + dst + proto + service
        for n in [src_node[i], dst_node[i], proto_node[i], svc_node[i]]:
            row_list.append(n.item())
            col_list.append(e_comm)

        # session: src + dst + proto
        for n in [src_node[i], dst_node[i], proto_node[i]]:
            row_list.append(n.item())
            col_list.append(e_sess)

        # burst_event: cluster + tseg + proto
        for n in [cluster_node[i], tseg_node[i], proto_node[i]]:
            row_list.append(n.item())
            col_list.append(e_burst)

        # probe_event: dst + proto + tseg
        for n in [dst_node[i], proto_node[i], tseg_node[i]]:
            row_list.append(n.item())
            col_list.append(e_probe)

    inc_row = torch.tensor(row_list, dtype=torch.long)
    inc_col = torch.tensor(col_list, dtype=torch.long)
    inc_val = torch.ones(len(row_list), dtype=torch.float32)

    # Edge type and sample assignment
    edge_type = torch.tensor(
        [t for _ in range(B) for t in range(N_EDGE_TYPES)], dtype=torch.long
    )
    edge_to_sample = torch.tensor(
        [i for i in range(B) for _ in range(N_EDGE_TYPES)], dtype=torch.long
    )

    # Hyperedge features: concatenate relevant flow statistics
    # Use chunks of flow_feats as edge features (4 slices per sample)
    d_edge = d_flow  # edge feature dim = flow feature dim
    edge_features = flow_feats.repeat_interleave(N_EDGE_TYPES, dim=0)  # (B*4, d_flow)

    # Node features: uniform learnable (will be in embedding table)
    # We return zeros here; the model embeds them via nn.Embedding
    node_features = torch.zeros(N_NODES, d_flow)

    return HypergraphBatch(
        node_features=node_features.to(device),
        edge_features=edge_features.to(device),
        incidence_row=inc_row.to(device),
        incidence_col=inc_col.to(device),
        incidence_val=inc_val.to(device),
        n_nodes=N_NODES,
        n_edges=total_edges,
        batch_size=B,
        edge_to_sample=edge_to_sample.to(device),
        edge_type=edge_type.to(device),
        flow_features=flow_feats.to(device),
    )


# ---------------------------------------------------------------------------
# HGNN Propagation Layer (typed hypergraph attention)
# ---------------------------------------------------------------------------

class HypergraphAttentionLayer(nn.Module):
    """
    One layer of hypergraph attention:
      1. Node -> HyperEdge: attention-weighted aggregation
      2. HyperEdge -> Node: attention-weighted aggregation
    Uses sparse incidence for efficiency.
    """
    def __init__(self, d_node: int, d_edge: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.n_heads = n_heads
        self.d_head = d_node // n_heads

        # Node -> Edge attention
        self.W_Q_n2e = nn.Linear(d_node, d_node)
        self.W_K_n2e = nn.Linear(d_edge, d_node)
        self.W_V_n2e = nn.Linear(d_node, d_node)

        # Edge -> Node attention
        self.W_Q_e2n = nn.Linear(d_edge, d_node)
        self.W_K_e2n = nn.Linear(d_node, d_node)
        self.W_V_e2n = nn.Linear(d_edge, d_node)

        # Edge update MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_edge + d_node, d_edge),
            nn.LayerNorm(d_edge),
            nn.GELU(),
        )
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(d_node + d_node, d_node),
            nn.LayerNorm(d_node),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.d_head) ** -0.5

    def forward(
        self,
        node_emb: torch.Tensor,         # (N_NODES, d_node)
        edge_emb: torch.Tensor,          # (n_edges, d_edge)
        inc_row: torch.Tensor,           # node indices
        inc_col: torch.Tensor,           # edge indices
        n_nodes: int,
        n_edges: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # --- Node -> Edge aggregation ---
        # For each hyperedge, aggregate member nodes with attention
        node_q = self.W_Q_n2e(node_emb)     # (N, d)
        edge_k = self.W_K_n2e(edge_emb)      # (E, d)
        node_v = self.W_V_n2e(node_emb)      # (N, d)

        # Attention score for each (node, edge) membership pair
        q_ij = node_q[inc_row]               # (nnz, d)
        k_ij = edge_k[inc_col]               # (nnz, d)
        attn_score = (q_ij * k_ij).sum(-1) * self.scale  # (nnz,)

        # Softmax per hyperedge (over member nodes)
        attn_score_e = torch.zeros(n_edges, device=node_emb.device)
        # Use scatter softmax
        attn_exp = torch.exp(attn_score - attn_score.max())
        denom = torch.zeros(n_edges, device=node_emb.device).scatter_add(0, inc_col, attn_exp)
        denom = denom[inc_col].clamp(min=1e-8)
        attn_w = attn_exp / denom  # (nnz,)
        attn_w = self.dropout(attn_w)

        # Weighted sum of node values -> edge update
        node_v_ij = node_v[inc_row] * attn_w.unsqueeze(-1)
        edge_agg = torch.zeros(n_edges, self.d_node, device=node_emb.device)
        edge_agg.scatter_add_(0, inc_col.unsqueeze(-1).expand_as(node_v_ij), node_v_ij)

        # Update edge embeddings
        edge_emb_new = self.edge_mlp(torch.cat([edge_emb, edge_agg], dim=-1))

        # --- Edge -> Node aggregation ---
        edge_q = self.W_Q_e2n(edge_emb_new)  # (E, d)
        node_k = self.W_K_e2n(node_emb)       # (N, d)
        edge_v = self.W_V_e2n(edge_emb_new)   # (E, d_node)

        q2_ij = edge_q[inc_col]
        k2_ij = node_k[inc_row]
        attn2_score = (q2_ij * k2_ij).sum(-1) * self.scale

        attn2_exp = torch.exp(attn2_score - attn2_score.max())
        denom2 = torch.zeros(n_nodes, device=node_emb.device).scatter_add(0, inc_row, attn2_exp)
        denom2 = denom2[inc_row].clamp(min=1e-8)
        attn2_w = attn2_exp / denom2
        attn2_w = self.dropout(attn2_w)

        edge_v_ij = edge_v[inc_col] * attn2_w.unsqueeze(-1)
        node_agg = torch.zeros(n_nodes, self.d_node, device=node_emb.device)
        node_agg.scatter_add_(0, inc_row.unsqueeze(-1).expand_as(edge_v_ij), edge_v_ij)

        # Update node embeddings
        node_emb_new = self.node_mlp(torch.cat([node_emb, node_agg], dim=-1))

        return node_emb_new, edge_emb_new


class HypergraphEncoder(nn.Module):
    """
    Multi-layer hypergraph encoder with typed node/edge embeddings.
    Produces per-sample graph-level embeddings by pooling edge embeddings.
    """
    def __init__(
        self,
        d_flow: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        n_edge_types: int = N_EDGE_TYPES,
        n_nodes: int = N_NODES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_nodes = n_nodes

        # Learnable node vocabulary embeddings
        self.node_embedding = nn.Embedding(n_nodes, d_model)

        # Edge type embeddings
        self.edge_type_embedding = nn.Embedding(n_edge_types, d_model)

        # Project flow features to d_model for edge features
        self.flow_proj = nn.Sequential(
            nn.Linear(d_flow, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # HGNN layers
        self.layers = nn.ModuleList([
            HypergraphAttentionLayer(d_model, d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Readout: pool edge embeddings per sample → graph embedding
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, hg: HypergraphBatch) -> torch.Tensor:
        """
        Returns graph-level embeddings: (B, d_model)
        """
        B = hg.batch_size
        device = hg.flow_features.device

        # Node embeddings: all N_NODES vocab nodes
        node_ids = torch.arange(self.n_nodes, device=device)
        node_emb = self.node_embedding(node_ids)  # (N_NODES, d_model)

        # Edge embeddings: flow features + edge type
        edge_emb = self.flow_proj(hg.edge_features)  # (n_edges, d_model)
        edge_type_emb = self.edge_type_embedding(hg.edge_type)  # (n_edges, d_model)
        edge_emb = edge_emb + edge_type_emb

        # HGNN propagation
        for layer in self.layers:
            node_emb, edge_emb = layer(
                node_emb, edge_emb,
                hg.incidence_row, hg.incidence_col,
                hg.n_nodes, hg.n_edges,
            )
            node_emb = self.dropout(node_emb)
            edge_emb = self.dropout(edge_emb)

        # Readout: mean-pool edge embeddings per sample
        graph_emb = torch.zeros(B, self.d_model, device=device)
        edge_out = self.readout(edge_emb)  # (n_edges, d_model)
        count = torch.zeros(B, device=device)
        graph_emb.scatter_add_(0, hg.edge_to_sample.unsqueeze(-1).expand_as(edge_out), edge_out)
        count.scatter_add_(0, hg.edge_to_sample, torch.ones(hg.n_edges, device=device))
        count = count.clamp(min=1).unsqueeze(-1)
        graph_emb = graph_emb / count  # mean pooling

        return graph_emb  # (B, d_model)


# ---------------------------------------------------------------------------
# Graph construction variants for comparison
# ---------------------------------------------------------------------------

class SimpleFlowGraph(nn.Module):
    """Variant 1: Simple MLP on raw flow features (no graph)."""
    def __init__(self, d_flow: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_flow, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, hg: HypergraphBatch) -> torch.Tensor:
        return self.net(hg.flow_features)


class ProtocolRoleGraph(nn.Module):
    """
    Variant 2: Simple pairwise graph attention over entity features.
    Treats each entity (src, dst, proto, service) as a node and
    applies multi-head self-attention over them.
    """
    def __init__(self, d_flow: int, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        n_entities = 4  # src_port, dst_port, proto, service
        self.entity_proj = nn.Linear(d_flow // n_entities + 1, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.flow_proj = nn.Linear(d_flow, d_model)
        self.combine = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hg: HypergraphBatch) -> torch.Tensor:
        B = hg.flow_features.shape[0]
        d = hg.flow_features.shape[1]
        # Split flow features into 4 entity-like segments
        chunk = d // 4
        parts = []
        for i in range(4):
            start = i * chunk
            end = start + chunk if i < 3 else d
            parts.append(hg.flow_features[:, start:end])

        # Project each part with a simple linear (pad to same size if needed)
        max_len = max(p.shape[1] for p in parts)
        padded = [F.pad(p, (0, max_len - p.shape[1])) for p in parts]
        entities = torch.stack(padded, dim=1)  # (B, 4, max_len)
        entities = self.entity_proj(entities)   # (B, 4, d_model)

        attended, _ = self.attn(entities, entities, entities)
        attended = self.norm(attended + entities)
        pooled = attended.mean(1)  # (B, d_model)

        flow_emb = self.flow_proj(hg.flow_features)
        combined = self.combine(torch.cat([pooled, flow_emb], dim=-1))
        return F.gelu(combined)
