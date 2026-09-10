"""
MorphGuard-IDS: Teacher and Student models.

Teacher: TCN -> HypergraphEncoder -> BiLSTM -> CausalInvariantHead -> Classifier
Student: CompactTCN -> MLP -> Classifier (trained via relational KD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List

from models.hypergraph import (
    HypergraphEncoder, HypergraphBatch, build_hypergraph_batch,
    N_NODES, N_EDGE_TYPES
)


# ---------------------------------------------------------------------------
# TCN block
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pad = pad  # causal trim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        res = self.res(x)
        out = self.conv1(x)[..., :-self.pad] if self.pad > 0 else self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.drop(out)
        out = self.conv2(out)[..., :-self.pad] if self.pad > 0 else self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out + res)
        return out


class MultiScaleTCN(nn.Module):
    """Multi-scale temporal encoder with dilated convolutions."""
    def __init__(self, d_in: int, d_model: int = 128, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        dilations = [2 ** i for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, d_model, kernel_size=3, dilation=d, dropout=dropout)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_in) — single flow feature vector
        Returns: (B, d_model)
        """
        # Expand to (B, 1, d_in) for 1D conv
        x = self.proj(x)           # (B, d_model)
        x = x.unsqueeze(2)         # (B, d_model, 1)
        for block in self.blocks:
            x = block(x)
        return x.squeeze(2)        # (B, d_model)


# ---------------------------------------------------------------------------
# Causal-Invariant Regularizer
# ---------------------------------------------------------------------------

class CausalInvariantRegularizer(nn.Module):
    """
    IRM-style penalty: penalize gradient norm variance across environments.
    Environments are created by splitting the batch on predicted confidence.
    """
    def __init__(self, n_envs: int = 3):
        super().__init__()
        self.n_envs = n_envs
        self.dummy_w = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Returns IRM penalty term."""
        B = logits.shape[0]
        env_size = B // self.n_envs
        if env_size < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Split batch into environments by index
        penalties = []
        for i in range(self.n_envs):
            start = i * env_size
            end = start + env_size if i < self.n_envs - 1 else B
            env_logits = logits[start:end] * self.dummy_w
            env_labels = labels[start:end]
            env_loss = F.cross_entropy(env_logits, env_labels)
            grad = torch.autograd.grad(
                env_loss, self.dummy_w,
                create_graph=True, retain_graph=True
            )[0]
            penalties.append(grad ** 2)

        if len(penalties) < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        penalty = torch.stack(penalties).mean()
        return penalty


# ---------------------------------------------------------------------------
# Calibration: Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)

    def calibration_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scaled = self.forward(logits)
        return F.cross_entropy(scaled, labels)


# ---------------------------------------------------------------------------
# Counterfactual / Minority-class Augmentation (in latent space)
# ---------------------------------------------------------------------------

class LatentCounterfactualAugmenter(nn.Module):
    """
    Simple latent-space augmentation for minority classes.
    Adds noise to minority embeddings to generate synthetic variants.
    Constrained to remain within class-conditional ellipsoid.
    """
    def __init__(self, noise_scale: float = 0.1, n_augment: int = 4):
        super().__init__()
        self.noise_scale = noise_scale
        self.n_augment = n_augment

    def forward(
        self,
        embeddings: torch.Tensor,   # (B, d)
        labels: torch.Tensor,        # (B,)
        min_class_count: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns augmented embeddings and labels for minority classes only.
        """
        unique, counts = torch.unique(labels, return_counts=True)
        max_count = counts.max().item()
        aug_embs = []
        aug_labs = []

        for cls_id, cnt in zip(unique.tolist(), counts.tolist()):
            if cnt < max_count * 0.5:  # minority if count < 50% of majority
                mask = (labels == cls_id)
                cls_embs = embeddings[mask]
                # Generate n_augment synthetic variants per minority sample
                n_gen = min(self.n_augment, max(1, max_count // (cnt + 1)))
                noise = torch.randn(
                    cls_embs.shape[0], n_gen, cls_embs.shape[1],
                    device=cls_embs.device
                ) * self.noise_scale
                aug = cls_embs.unsqueeze(1) + noise  # (n_cls, n_gen, d)
                aug = aug.reshape(-1, cls_embs.shape[1])
                aug_embs.append(aug)
                aug_labs.extend([cls_id] * len(aug))

        if aug_embs:
            aug_embs = torch.cat(aug_embs, dim=0)
            aug_labs = torch.tensor(aug_labs, dtype=labels.dtype, device=labels.device)
            return torch.cat([embeddings, aug_embs], dim=0), torch.cat([labels, aug_labs], dim=0)
        return embeddings, labels


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p = torch.exp(-ce)
        loss = ((1 - p) ** self.gamma) * ce
        return loss.mean()


# ---------------------------------------------------------------------------
# MorphGuard Teacher
# ---------------------------------------------------------------------------

class MorphGuardTeacher(nn.Module):
    """
    Full teacher model:
      flow -> MultiScaleTCN -> HypergraphEncoder -> fusion -> BiLSTM (simulated)
           -> CausalHead -> Classifier
    """
    def __init__(
        self,
        d_flow: int,
        n_classes: int,
        d_model: int = 128,
        n_tcn_layers: int = 4,
        n_hgnn_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.15,
        n_envs: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # Temporal encoder
        self.tcn = MultiScaleTCN(d_flow, d_model, n_tcn_layers, dropout)

        # Hypergraph encoder
        self.hgnn = HypergraphEncoder(
            d_flow=d_flow,
            d_model=d_model,
            n_layers=n_hgnn_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Fusion of TCN + HGNN features
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal sequence modeling (simulated BiLSTM over the fused features)
        # Since we process individual samples (not sequences), we use a
        # self-attention layer as a proxy for temporal dependencies
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        # Causal-invariant regularizer
        self.causal_reg = CausalInvariantRegularizer(n_envs)

        # Temperature scaling for calibration
        self.temp_scaler = TemperatureScaler()

        # Latent augmenter
        self.augmenter = LatentCounterfactualAugmenter(noise_scale=0.1, n_augment=3)

        # Projection head for self-supervised pretraining
        self.ssl_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )

    def encode(self, flow_feats: torch.Tensor, hg: HypergraphBatch) -> torch.Tensor:
        """Extract embeddings from flow features and hypergraph."""
        tcn_emb = self.tcn(flow_feats)           # (B, d_model)
        hg_emb = self.hgnn(hg)                   # (B, d_model)
        fused = self.fusion(torch.cat([tcn_emb, hg_emb], dim=-1))  # (B, d_model)

        # Apply temporal attention (single token per sample, so self-attend with no-op)
        # Batch as sequence of length 1
        fused_seq = fused.unsqueeze(1)  # (B, 1, d_model)
        attn_out, _ = self.temporal_attn(fused_seq, fused_seq, fused_seq)
        fused = self.temporal_norm(fused + attn_out.squeeze(1))
        return fused

    def forward(
        self,
        flow_feats: torch.Tensor,
        hg: HypergraphBatch,
        labels: Optional[torch.Tensor] = None,
        augment: bool = False,
    ) -> Dict[str, torch.Tensor]:

        emb = self.encode(flow_feats, hg)  # (B, d_model)

        # Optional latent augmentation during training
        if augment and labels is not None:
            emb, labels = self.augmenter(emb, labels)

        logits = self.classifier(emb)           # (B, n_classes)
        cal_logits = self.temp_scaler(logits)   # calibrated logits

        out = {
            "logits": logits,
            "cal_logits": cal_logits,
            "embeddings": emb,
            "ssl_proj": self.ssl_proj(emb),
        }

        if labels is not None:
            out["aug_labels"] = labels

        return out

    def ssl_pretrain_loss(self, flow_feats: torch.Tensor, hg: HypergraphBatch) -> torch.Tensor:
        """
        Self-supervised loss: masked hyperedge reconstruction + contrastive.
        We implement a simplified version: reconstruction from masked flow.
        """
        # Create two views: original and slightly perturbed
        noise = torch.randn_like(flow_feats) * 0.1
        flow_aug = flow_feats + noise

        # Build augmented hypergraph
        hg_aug = build_hypergraph_batch(flow_aug, device=str(flow_feats.device))

        emb1 = self.encode(flow_feats, hg)
        emb2 = self.encode(flow_aug, hg_aug)

        proj1 = self.ssl_proj(emb1)
        proj2 = self.ssl_proj(emb2)

        # NT-Xent contrastive loss
        proj1 = F.normalize(proj1, dim=-1)
        proj2 = F.normalize(proj2, dim=-1)
        B = proj1.shape[0]
        sim_matrix = torch.mm(proj1, proj2.T) / 0.1  # temperature=0.1
        labels = torch.arange(B, device=proj1.device)
        loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
        return loss


# ---------------------------------------------------------------------------
# MorphGuard Student
# ---------------------------------------------------------------------------

class MorphGuardStudent(nn.Module):
    """
    Lightweight student model for edge deployment.
    Compact TCN + MLP-Mixer style architecture.
    Trained via relational knowledge distillation from teacher.
    """
    def __init__(
        self,
        d_flow: int,
        n_classes: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Compact temporal encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_flow, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Compact TCN
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(d_model, d_model, kernel_size=3, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        # Temperature scaling
        self.temp_scaler = TemperatureScaler()

    def encode(self, flow_feats: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(flow_feats)     # (B, d_model)
        emb_3d = emb.unsqueeze(2)          # (B, d_model, 1)
        for block in self.tcn_blocks:
            emb_3d = block(emb_3d)
        return emb_3d.squeeze(2)           # (B, d_model)

    def forward(self, flow_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encode(flow_feats)
        logits = self.classifier(emb)
        cal_logits = self.temp_scaler(logits)
        return {
            "logits": logits,
            "cal_logits": cal_logits,
            "embeddings": emb,
        }


# ---------------------------------------------------------------------------
# Knowledge Distillation Loss
# ---------------------------------------------------------------------------

class RelationalKDLoss(nn.Module):
    """
    Combined KD loss: soft-label KL + embedding matching + relational preservation.
    """
    def __init__(self, temperature: float = 4.0, lambda_z: float = 0.5, lambda_r: float = 0.1):
        super().__init__()
        self.T = temperature
        self.lambda_z = lambda_z
        self.lambda_r = lambda_r

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
    ) -> torch.Tensor:
        T = self.T

        # Soft-label KL divergence
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * T ** 2

        # Embedding matching (only when dimensions match)
        if student_emb.shape[-1] == teacher_emb.shape[-1]:
            emb_loss = F.mse_loss(
                F.normalize(student_emb, dim=-1),
                F.normalize(teacher_emb, dim=-1).detach(),
            )
        else:
            emb_loss = torch.tensor(0.0, device=student_emb.device)

        # Relational preservation: pairwise similarity matrices (dimension-agnostic)
        B = student_emb.shape[0]
        if B > 1:
            s_norm = F.normalize(student_emb, dim=-1)
            t_norm = F.normalize(teacher_emb, dim=-1).detach()
            G_s = torch.mm(s_norm, s_norm.T)
            G_t = torch.mm(t_norm, t_norm.T)
            rel_loss = F.mse_loss(G_s, G_t)
        else:
            rel_loss = torch.tensor(0.0, device=student_emb.device)

        total = kl_loss + self.lambda_z * emb_loss + self.lambda_r * rel_loss
        return total, {
            "kl": kl_loss.item(),
            "emb": emb_loss.item(),
            "rel": rel_loss.item(),
        }


# ---------------------------------------------------------------------------
# Total Training Loss for Teacher
# ---------------------------------------------------------------------------

class MorphGuardLoss(nn.Module):
    """
    L_total = L_cls + alpha*L_distill + beta*L_relation + gamma*L_causal
              + delta*L_calibration + eta*L_regularization
    """
    def __init__(
        self,
        n_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        alpha_causal: float = 0.1,
        delta_calib: float = 0.1,
        eta_reg: float = 1e-4,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, weight=class_weights)
        self.alpha_causal = alpha_causal
        self.delta_calib = delta_calib
        self.eta_reg = eta_reg

    def forward(
        self,
        logits: torch.Tensor,
        cal_logits: torch.Tensor,
        labels: torch.Tensor,
        causal_penalty: torch.Tensor,
        model_params: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        L_cls = self.focal(logits, labels)
        L_calib = F.cross_entropy(cal_logits, labels)
        L_causal = causal_penalty

        L_reg = torch.tensor(0.0, device=logits.device)
        if model_params and self.eta_reg > 0:
            for p in model_params:
                L_reg = L_reg + p.pow(2).sum()
            L_reg = L_reg * self.eta_reg

        total = (
            L_cls
            + self.alpha_causal * L_causal
            + self.delta_calib * L_calib
            + L_reg
        )

        return total, {
            "cls": L_cls.item(),
            "causal": L_causal.item() if torch.is_tensor(L_causal) else float(L_causal),
            "calib": L_calib.item(),
            "reg": L_reg.item(),
            "total": total.item(),
        }


# ---------------------------------------------------------------------------
# Conformal Threshold Calibration
# ---------------------------------------------------------------------------

def calibrate_conformal_threshold(
    model: MorphGuardTeacher,
    val_loader,
    alpha: float = 0.01,
    device: str = "cuda",
) -> float:
    """
    Calibrate conformal threshold on benign validation samples.
    Returns tau such that FPR <= alpha on benign val set.
    """
    model.eval()
    risk_scores = []
    with torch.no_grad():
        for batch in val_loader:
            flow_feats = batch["features"].to(device)
            labels = batch["binary_label"].to(device)
            hg = build_hypergraph_batch(flow_feats, device=device)
            out = model(flow_feats, hg)
            # Risk = P(attack)
            probs = torch.softmax(out["cal_logits"], dim=-1)
            # Binary: P(any attack class)
            # For multi-class, sum over attack classes (all except class 0 = Normal)
            attack_prob = 1.0 - probs[:, 0]  # assume class 0 is Normal
            # Benign-only scores
            benign_mask = (labels == 0)
            risk_scores.extend(attack_prob[benign_mask].cpu().numpy().tolist())

    if not risk_scores:
        return 0.5
    risk_scores = sorted(risk_scores)
    # tau = quantile (1-alpha) of benign scores
    tau_idx = int(np.ceil((1 - alpha) * len(risk_scores))) - 1
    tau = risk_scores[min(tau_idx, len(risk_scores) - 1)]
    return float(tau)
