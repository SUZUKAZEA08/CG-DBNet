"""
CG-DBNet model definitions
==========================
Backbone shared by the prediction and the quality-control variants
(CyclicPositionalEncoding + multi-scale MSAC CNN + dual-attention GAP +
AdaptiveAffineFusion + GRU). The two variants differ only in the output head and
in whether RevIN is applied:

  - PredictionNet: RevIN built in (applied to the temporal dimensions only, the
    spatial dimensions are concatenated unchanged), outputs
    (mu * target_std, sigma). mu is used in de-normalized space for the
    MSE loss on the differenced target.
  - QCNet: no RevIN (inputs are already normalized by StandardScaler), outputs
    (mu, sigma) in normalized space for Gaussian NLL training.

All ablation switches (use_pos / use_msac / use_dagap / use_affine_fusion /
context_direct_gru / use_conv3 / use_conv5 / use_conv7) can be combined freely.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int
    seq_len: int
    dropout: float


# ===================== Shared building blocks =====================
class CyclicPositionalEncoding(nn.Module):
    """Sinusoidal/cosine cyclic positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :]


class AdaptiveAffineFusion(nn.Module):
    """Adaptive affine fusion: sigmoid-gated weighting of (local + global) and (local * global)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        w_alpha = torch.sigmoid(self.alpha)
        w_beta = torch.sigmoid(self.beta)
        w_sum = w_alpha + w_beta + 1e-8
        return (w_alpha / w_sum) * (x_local + x_global) + (w_beta / w_sum) * (x_local * x_global)


class GaussianHead(nn.Module):
    """Gaussian output head: linear layer for mu + linear layer with Softplus for sigma."""

    def __init__(self, hidden_dim: int, sigma_min: float = 1e-4):
        super().__init__()
        self.head_mu = nn.Linear(hidden_dim, 1)
        self.head_sigma = nn.Linear(hidden_dim, 1)
        self.sigma_activate = nn.Softplus()
        self.sigma_min = sigma_min

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.head_mu(x)
        sigma = self.sigma_activate(self.head_sigma(x)) + self.sigma_min
        return mu, sigma


class RevIN(nn.Module):
    """RevIN instance normalization: (x - mean) / std over the first n_temporal_dims
    dimensions (detached); the trailing n_spatial dimensions (geospatial features) are
    concatenated unchanged. Also exposes target_std for de-normalizing mu."""

    def __init__(self, n_temporal_dims: int, n_spatial: int):
        super().__init__()
        self.n_temporal_dims = n_temporal_dims
        self.n_spatial = n_spatial

    def forward(self, x: torch.Tensor):
        if self.n_spatial > 0:
            x_temp = x[:, :, : self.n_temporal_dims]
            x_spat = x[:, :, self.n_temporal_dims:]
        else:
            x_temp = x
            x_spat = None
        mean = x_temp.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(torch.var(x_temp, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_temp = (x_temp - mean) / std
        x_norm = torch.cat([x_temp, x_spat], dim=-1) if x_spat is not None else x_temp
        return x_norm, mean, std

    def target_std(self, std: torch.Tensor) -> torch.Tensor:
        """std of the target column (last temporal dimension), used to de-normalize mu."""
        return std[:, :, self.n_temporal_dims - 1]


# ===================== Shared backbone =====================
class CGDBNetBackbone(nn.Module):
    """CG-DBNet shared backbone: input projection -> (optional) positional encoding ->
    MSAC multi-scale CNN -> dual-attention GAP -> adaptive affine fusion ->
    (optional) temporal/feature-attention GRU.
    Takes the un-projected input x and returns the feature vector of the last time
    step, x_last, for the output head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        dropout: float = 0.15,
        use_pos: bool = True,
        use_msac: bool = True,
        use_dagap: bool = True,
        use_affine_fusion: bool = True,
        context_direct_gru: bool = False,
        use_conv3: bool = True,
        use_conv5: bool = True,
        use_conv7: bool = True,
    ):
        super().__init__()
        self.use_pos = use_pos
        self.use_msac = use_msac
        self.use_dagap = use_dagap
        self.use_affine_fusion = use_affine_fusion
        self.context_direct_gru = context_direct_gru
        self.use_conv3 = use_conv3
        self.use_conv5 = use_conv5
        self.use_conv7 = use_conv7
        self.msac_num_branches = max(int(use_conv3) + int(use_conv5) + int(use_conv7), 1)

        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.pos_encoder = CyclicPositionalEncoding(hidden_dim, max_len=seq_len * 2)
        self.pos_weight = nn.Parameter(torch.ones(hidden_dim))

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2)
        self.conv5 = nn.Conv1d(hidden_dim, hidden_dim, 5, padding=4, dilation=2)
        self.conv7 = nn.Conv1d(hidden_dim, hidden_dim, 7, padding=6, dilation=2)
        self.norm_concat = nn.LayerNorm(hidden_dim * self.msac_num_branches)
        self.conv_1x1 = nn.Conv1d(hidden_dim * self.msac_num_branches, hidden_dim, 1)
        self.norm_conv = nn.LayerNorm(hidden_dim)

        self.affine_fusion = AdaptiveAffineFusion(hidden_dim)
        self.norm_global = nn.LayerNorm(hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len), nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid(),
        )

        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.norm_gru = nn.LayerNorm(hidden_dim)
        self.gru_res_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.input_proj(x)

        # Direct GRU shortcut (ablation: skip MSAC/DAGAP, GRU only)
        if self.context_direct_gru:
            x_gru_out, _ = self.gru(x_emb)
            x_gru_out = self.norm_gru(x_gru_out)
            return x_gru_out[:, -1, :]

        # The last 4 projected channels are treated as temporal encoding channels
        non_time_dim = max(x_emb.shape[-1] - 4, 1)
        x_emb_non_time = x_emb[..., :non_time_dim]
        x_emb_time = x_emb[..., non_time_dim:]

        if self.use_pos:
            x_pos_full = self.pos_encoder(x_emb)
            x_pos_non_time = x_pos_full[..., :non_time_dim]
            pos_weight = self.pos_weight[:non_time_dim].view(1, 1, -1)
            x_conv_in_non_time = x_emb_non_time + x_pos_non_time * pos_weight
        else:
            x_conv_in_non_time = x_emb_non_time
        x_conv_in = torch.cat([x_conv_in_non_time, x_emb_time], dim=-1)

        # MSAC: parallel multi-scale convolution branches
        if self.use_msac and (self.use_conv3 or self.use_conv5 or self.use_conv7):
            x_in_t = x_conv_in.transpose(1, 2)
            branches = []
            if self.use_conv3:
                branches.append(self.conv3(x_in_t))
            if self.use_conv5:
                branches.append(self.conv5(x_in_t))
            if self.use_conv7:
                branches.append(self.conv7(x_in_t))
            x_cat = torch.cat(branches, dim=1).transpose(1, 2)
            x_cat = self.norm_concat(x_cat)
            x_local = self.conv_1x1(x_cat.transpose(1, 2)).transpose(1, 2)
            x_local = self.norm_conv(x_local)
        else:
            x_local = x_conv_in

        # Gated local features
        x_gate = torch.tanh(x_local) * torch.sigmoid(x_local) + x_conv_in

        # Dual-attention GAP: feature-dimension gating * time-dimension gating * input embedding
        w_feat = torch.sigmoid(x_emb.mean(dim=1, keepdim=True))
        w_time = torch.sigmoid(x_emb.mean(dim=2, keepdim=True))
        a_global = w_feat * w_time * x_emb

        if self.use_affine_fusion:
            x_fused = self.norm_global(self.affine_fusion(x_gate, a_global) + x_conv_in)
        else:
            x_fused = self.norm_global(x_gate + a_global + x_conv_in)

        # DAGAP: temporal attention modulates the GRU input; feature attention modulates the GRU output
        if self.use_dagap:
            time_gap = x_fused.mean(dim=2)
            time_attn = self.time_mlp(time_gap)
            x_gru_in = x_emb * time_attn.unsqueeze(-1)
            feat_gap = x_fused.mean(dim=1)
            feat_attn = self.mlp(feat_gap)
        else:
            x_gru_in = x_emb
            feat_attn = torch.ones(x_emb.shape[0], x_emb.shape[2], device=x_emb.device)

        x_gru_out, _ = self.gru(x_gru_in)
        x_gru_out = self.norm_gru(x_gru_out)
        x_gru_out_feat = x_gru_out * feat_attn.unsqueeze(1)
        gru_res_weight = torch.sigmoid(self.gru_res_weight)
        x_last = x_gru_out_feat[:, -1, :] + gru_res_weight * x_gru_out[:, -1, :]
        return x_last


# ===================== Prediction model (RevIN + mu * target_std output) =====================
class PredictionNet(nn.Module):
    """CG-DBNet for prediction: RevIN built in, outputs (mu * target_std, sigma)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        dropout: float = 0.15,
        n_spatial: int = 0,
        sigma_min: float = 1e-4,
        use_pos: bool = True,
        use_msac: bool = True,
        use_dagap: bool = True,
        use_affine_fusion: bool = True,
        context_direct_gru: bool = False,
        use_conv3: bool = True,
        use_conv5: bool = True,
        use_conv7: bool = True,
    ):
        super().__init__()
        self.n_spatial = n_spatial
        self.n_temporal_dims = input_dim - n_spatial
        self.revin = RevIN(self.n_temporal_dims, n_spatial)
        self.backbone = CGDBNetBackbone(
            input_dim, hidden_dim, seq_len, dropout,
            use_pos=use_pos, use_msac=use_msac, use_dagap=use_dagap,
            use_affine_fusion=use_affine_fusion, context_direct_gru=context_direct_gru,
            use_conv3=use_conv3, use_conv5=use_conv5, use_conv7=use_conv7,
        )
        self.head = GaussianHead(hidden_dim, sigma_min)

    def forward(self, x: torch.Tensor):
        x_norm, _mean, std = self.revin(x)
        x_last = self.backbone(x_norm)
        mu, sigma = self.head(x_last)
        target_std = self.revin.target_std(std)
        return mu * target_std, sigma


# ===================== Quality-control model (no RevIN, mu/sigma in normalized space) =====================
class QCNet(nn.Module):
    """CG-DBNet for quality control: no RevIN (inputs are already normalized by
    StandardScaler), outputs (mu, sigma) in normalized space for Gaussian NLL training."""

    def __init__(
        self,
        cfg: ModelConfig,
        sigma_min: float = 1e-4,
        use_pos: bool = True,
        use_msac: bool = True,
        use_dagap: bool = True,
        use_affine_fusion: bool = True,
        context_net_direct_gru: bool = False,
        use_conv3: bool = True,
        use_conv5: bool = True,
        use_conv7: bool = True,
    ):
        super().__init__()
        self.backbone = CGDBNetBackbone(
            cfg.input_dim, cfg.hidden_dim, cfg.seq_len, cfg.dropout,
            use_pos=use_pos, use_msac=use_msac, use_dagap=use_dagap,
            use_affine_fusion=use_affine_fusion, context_direct_gru=context_net_direct_gru,
            use_conv3=use_conv3, use_conv5=use_conv5, use_conv7=use_conv7,
        )
        self.head = GaussianHead(cfg.hidden_dim, sigma_min)

    def forward(self, x: torch.Tensor):
        x_last = self.backbone(x)
        return self.head(x_last)


# ===================== Model factories =====================
def build_pred_model(kind: str, input_dim: int, n_spatial: int, seq_len: int,
                     hidden_dim: int, dropout: float, sigma_min: float = 1e-4, toggles: dict | None = None):
    """Model factory for the prediction experiment. kind: proposed.
    The ablation switches are read from the toggles dictionary."""
    if kind == "proposed":
        t = toggles or {}
        return PredictionNet(
            input_dim, hidden_dim, seq_len, dropout, n_spatial, sigma_min,
            use_pos=t.get("use_pos", True), use_msac=t.get("use_msac", True),
            use_dagap=t.get("use_dagap", True), use_affine_fusion=t.get("use_affine_fusion", True),
            context_direct_gru=t.get("context_direct_gru", False),
            use_conv3=t.get("use_conv3", True), use_conv5=t.get("use_conv5", True),
            use_conv7=t.get("use_conv7", True),
        )
    raise ValueError(f"Unknown model_kind: {kind}")


def build_qc_model(kind: str, input_dim: int, seq_len: int, hidden_dim: int,
                   dropout: float, sigma_min: float = 1e-4):
    """Model factory for the quality-control experiment.
    kind: proposed|abl_no_msac|abl_no_dagap|abl_no_pos|abl_affine_add|abl_serial_gru.
    The network structure is selected by kind."""
    cfg = ModelConfig(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, dropout=dropout)
    if kind == "proposed":
        return QCNet(cfg, sigma_min)
    if kind == "abl_no_msac":
        return QCNet(cfg, sigma_min, use_msac=False)
    if kind == "abl_no_dagap":
        return QCNet(cfg, sigma_min, use_dagap=False)
    if kind == "abl_no_pos":
        return QCNet(cfg, sigma_min, use_pos=False)
    if kind == "abl_affine_add":
        return QCNet(cfg, sigma_min, use_affine_fusion=False)
    if kind == "abl_serial_gru":
        return QCNet(cfg, sigma_min, context_net_direct_gru=True)
    raise ValueError(f"Unknown model_kind: {kind}")
