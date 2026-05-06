from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import FEATURE_INDEX


class ChannelLayerNorm1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def build_norm_1d(norm_type: str, channels: int, groupnorm_groups: int = 8) -> nn.Module:
    norm_name = str(norm_type).lower()
    if norm_name == "batchnorm":
        return nn.BatchNorm1d(channels)
    if norm_name == "groupnorm":
        groups = max(1, min(int(groupnorm_groups), channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_name == "layernorm":
        return ChannelLayerNorm1d(channels)
    raise ValueError(f"unsupported norm_type: {norm_type}")


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, norm_type: str = "groupnorm", groupnorm_groups: int = 8):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
        self.norm = build_norm_1d(norm_type, out_ch, groupnorm_groups)
        self.act = nn.GELU()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = F.pad(x, (self.left_pad, 0))
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x + residual


class CausalTCNStem(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, channels: int = 64, norm_type: str = "groupnorm", groupnorm_groups: int = 8):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.block1 = CausalConvBlock(channels, channels, kernel_size=3, dilation=1, norm_type=norm_type, groupnorm_groups=groupnorm_groups)
        self.block2 = CausalConvBlock(channels, channels, kernel_size=3, dilation=2, norm_type=norm_type, groupnorm_groups=groupnorm_groups)
        self.block3 = CausalConvBlock(channels, hidden_dim, kernel_size=3, dilation=4, norm_type=norm_type, groupnorm_groups=groupnorm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.transpose(1, 2)


@dataclass
class LACSPCORNetConfig:
    input_dim: int
    stem_dim: int = 64
    state_dim: int = 64
    dropout: float = 0.10
    norm_type: str = "groupnorm"
    groupnorm_groups: int = 8
    residual_on_input: bool = True
    switch_gate_mode: str = "learned"
    explicit_gate_switch_weight: float = 0.35
    explicit_gate_turn_weight: float = 0.35
    explicit_gate_lost_weight: float = 0.15
    explicit_gate_coast_weight: float = 0.10
    explicit_gate_zoom_weight: float = 0.05
    explicit_gate_jitter_weight: float = 0.25
    explicit_gate_jitter_scale: float = 1.50
    explicit_gate_threshold: float = 0.45
    explicit_gate_temperature: float = 0.12
    recover_residual_cap_enabled: bool = True
    recover_residual_cap_value: float = 8.0
    recover_residual_min_scale: float = 0.25
    recover_residual_trigger_threshold: float = 0.55
    residual_base_gain: float = 0.10
    residual_gate_gain: float = 0.90
    lead_enabled: bool = False
    future_head: bool = False
    velocity_head: bool = False


class LACSPCORNet(nn.Module):
    def __init__(self, cfg: LACSPCORNetConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = CausalTCNStem(
            cfg.input_dim,
            cfg.stem_dim,
            channels=cfg.stem_dim,
            norm_type=cfg.norm_type,
            groupnorm_groups=cfg.groupnorm_groups,
        )
        self.slow_gru = nn.GRU(cfg.stem_dim, cfg.state_dim, batch_first=True)
        self.fast_gru = nn.GRU(cfg.stem_dim, cfg.state_dim, batch_first=True)
        gate_in = cfg.stem_dim + 6
        self.correction_gate = nn.Sequential(nn.Linear(gate_in, cfg.state_dim), nn.GELU(), nn.Linear(cfg.state_dim, 1))
        self.switch_gate = None
        if str(cfg.switch_gate_mode).lower() == "learned":
            self.switch_gate = nn.Sequential(nn.Linear(gate_in, cfg.state_dim), nn.GELU(), nn.Linear(cfg.state_dim, 1))
        self.slow_corrector = nn.Sequential(nn.Linear(cfg.stem_dim + cfg.state_dim, cfg.state_dim), nn.GELU(), nn.Linear(cfg.state_dim, cfg.state_dim))
        self.fast_corrector = nn.Sequential(nn.Linear(cfg.stem_dim + cfg.state_dim, cfg.state_dim), nn.GELU(), nn.Linear(cfg.state_dim, cfg.state_dim))
        self.direction_mix = None
        self.lead_magnitude = None
        self.lead_gate = None
        if cfg.lead_enabled:
            self.direction_mix = nn.Sequential(nn.Linear(cfg.state_dim, cfg.state_dim), nn.GELU(), nn.Linear(cfg.state_dim, 3))
            self.lead_magnitude = nn.Sequential(nn.Linear(cfg.state_dim, cfg.state_dim), nn.GELU(), nn.Linear(cfg.state_dim, 1))
            self.lead_gate = nn.Sequential(nn.Linear(gate_in, cfg.state_dim), nn.GELU(), nn.Linear(cfg.state_dim, 1))
        self.clean_head = nn.Sequential(nn.Linear(cfg.state_dim, cfg.state_dim), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.state_dim, 2))
        self.velocity_head = None
        if cfg.velocity_head:
            self.velocity_head = nn.Sequential(nn.Linear(cfg.state_dim, cfg.state_dim), nn.GELU(), nn.Linear(cfg.state_dim, 2))
        self.future_head = None
        if cfg.future_head:
            self.future_head = nn.Sequential(nn.Linear(cfg.state_dim + 1, cfg.state_dim), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.state_dim, 2))

    def _event_features(self, stem_seq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        switch_score = x[:, :, FEATURE_INDEX.switch_score : FEATURE_INDEX.switch_score + 1]
        turn_score = x[:, :, FEATURE_INDEX.turn_score : FEATURE_INDEX.turn_score + 1]
        det_conf = x[:, :, FEATURE_INDEX.det_conf : FEATURE_INDEX.det_conf + 1]
        lost_flag = x[:, :, FEATURE_INDEX.lost_flag : FEATURE_INDEX.lost_flag + 1]
        zoom_delta = x[:, :, FEATURE_INDEX.zoom_delta : FEATURE_INDEX.zoom_delta + 1]
        meas_age = x[:, :, FEATURE_INDEX.meas_age_ms : FEATURE_INDEX.meas_age_ms + 1]
        return torch.cat([stem_seq, switch_score, turn_score, det_conf, lost_flag, zoom_delta, meas_age], dim=-1)

    def _direction_bases(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        innovation = x[:, :, FEATURE_INDEX.dx_raw : FEATURE_INDEX.dy_raw + 1] - x[:, :, FEATURE_INDEX.trend_dx : FEATURE_INDEX.trend_dy + 1]
        trend_v = x[:, :, FEATURE_INDEX.trend_vx : FEATURE_INDEX.trend_vy + 1]
        raw_v = x[:, :, FEATURE_INDEX.d1_dx : FEATURE_INDEX.d1_dy + 1]
        eps = 1e-6
        d_innov = innovation / (innovation.norm(dim=-1, keepdim=True) + eps)
        d_trend = trend_v / (trend_v.norm(dim=-1, keepdim=True) + eps)
        d_raw = raw_v / (raw_v.norm(dim=-1, keepdim=True) + eps)
        return {"innov": d_innov, "trend": d_trend, "raw": d_raw}

    def _input_baseline_seq(self, x: torch.Tensor, baseline_seq: torch.Tensor | None = None) -> torch.Tensor:
        if baseline_seq is None:
            raise ValueError("baseline_seq is required when residual_on_input=true")
        return baseline_seq

    def _recover_score_seq(self, x: torch.Tensor) -> torch.Tensor:
        lost = x[:, :, FEATURE_INDEX.lost_flag : FEATURE_INDEX.lost_flag + 1]
        coast = x[:, :, FEATURE_INDEX.coast_count : FEATURE_INDEX.coast_count + 1]
        return torch.clamp(torch.maximum(lost, coast), 0.0, 1.0)

    def _explicit_switch_gate(self, x: torch.Tensor) -> torch.Tensor:
        switch = x[:, :, FEATURE_INDEX.switch_score : FEATURE_INDEX.switch_score + 1]
        turn = x[:, :, FEATURE_INDEX.turn_score : FEATURE_INDEX.turn_score + 1]
        lost = x[:, :, FEATURE_INDEX.lost_flag : FEATURE_INDEX.lost_flag + 1]
        coast = x[:, :, FEATURE_INDEX.coast_count : FEATURE_INDEX.coast_count + 1]
        zoom = torch.abs(x[:, :, FEATURE_INDEX.zoom_delta : FEATURE_INDEX.zoom_delta + 1])
        jitter = torch.clamp(
            torch.sqrt(
                x[:, :, FEATURE_INDEX.d1_dx : FEATURE_INDEX.d1_dx + 1] ** 2
                + x[:, :, FEATURE_INDEX.d1_dy : FEATURE_INDEX.d1_dy + 1] ** 2
            )
            / max(float(self.cfg.explicit_gate_jitter_scale), 1e-6),
            0.0,
            1.0,
        )
        weights = torch.tensor(
            [
                self.cfg.explicit_gate_switch_weight,
                self.cfg.explicit_gate_turn_weight,
                self.cfg.explicit_gate_lost_weight,
                self.cfg.explicit_gate_coast_weight,
                self.cfg.explicit_gate_zoom_weight,
                self.cfg.explicit_gate_jitter_weight,
            ],
            dtype=x.dtype,
            device=x.device,
        )
        weights = weights / torch.clamp(weights.sum(), min=1e-6)
        gate_score = (
            weights[0] * switch
            + weights[1] * turn
            + weights[2] * lost
            + weights[3] * coast
            + weights[4] * zoom
            + weights[5] * jitter
        )
        threshold = float(self.cfg.explicit_gate_threshold)
        temperature = max(float(self.cfg.explicit_gate_temperature), 1e-4)
        return torch.sigmoid((gate_score - threshold) / temperature)

    def forward(
        self,
        x: torch.Tensor,
        horizon_ms: Optional[torch.Tensor] = None,
        baseline_seq: Optional[torch.Tensor] = None,
        stable_baseline_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        stem_seq = self.stem(x)
        slow_seq, _ = self.slow_gru(stem_seq)
        fast_seq, _ = self.fast_gru(stem_seq)
        event_seq = self._event_features(stem_seq, x)
        corr_gate = torch.sigmoid(self.correction_gate(event_seq))
        if str(self.cfg.switch_gate_mode).lower() == "explicit":
            switch_gate = self._explicit_switch_gate(x)
        else:
            switch_gate = torch.sigmoid(self.switch_gate(event_seq))
        slow_corr = self.slow_corrector(torch.cat([stem_seq, slow_seq], dim=-1))
        fast_corr = self.fast_corrector(torch.cat([stem_seq, fast_seq], dim=-1))
        slow_post = slow_seq + corr_gate * slow_corr
        fast_post = fast_seq + corr_gate * fast_corr
        latent_seq = (1.0 - switch_gate) * slow_post + switch_gate * fast_post
        clean_residual_seq = self.clean_head(latent_seq)
        if self.cfg.recover_residual_cap_enabled:
            recover_score = self._recover_score_seq(x)
            strong_recover = (recover_score >= float(self.cfg.recover_residual_trigger_threshold)).to(clean_residual_seq.dtype)
            cap_value = max(float(self.cfg.recover_residual_cap_value), 1e-3)
            min_scale = float(self.cfg.recover_residual_min_scale)
            min_scale = min(max(min_scale, 0.0), 1.0)
            capped_residual = cap_value * torch.tanh(clean_residual_seq / cap_value)
            recover_residual = min_scale * capped_residual
            clean_residual_seq = (1.0 - strong_recover) * clean_residual_seq + strong_recover * recover_residual
        clean_seq = clean_residual_seq
        if self.cfg.residual_on_input:
            residual_gain = float(self.cfg.residual_base_gain) + float(self.cfg.residual_gate_gain) * switch_gate
            event_seq = self._input_baseline_seq(x, baseline_seq=baseline_seq) + residual_gain * clean_residual_seq
            if stable_baseline_seq is None:
                clean_seq = event_seq
            else:
                clean_seq = (1.0 - switch_gate) * stable_baseline_seq + switch_gate * event_seq
        if self.cfg.lead_enabled:
            bases = self._direction_bases(x)
            mix = torch.softmax(self.direction_mix(latent_seq), dim=-1)
            lead_dir = mix[:, :, 0:1] * bases["innov"] + mix[:, :, 1:2] * bases["trend"] + mix[:, :, 2:3] * bases["raw"]
            lead_dir = lead_dir / (lead_dir.norm(dim=-1, keepdim=True) + 1e-6)
            lead_mag = F.softplus(self.lead_magnitude(latent_seq))
            lead_gate = torch.sigmoid(self.lead_gate(event_seq))
            lead_delta = lead_gate * lead_mag * lead_dir
        else:
            lead_delta = torch.zeros_like(clean_seq)

        out: Dict[str, torch.Tensor] = {
            "clean_seq": clean_seq,
            "clean": clean_seq[:, -1, :],
            "lead_delta_seq": lead_delta,
            "lead_delta": lead_delta[:, -1, :],
            "switch_gate_seq": switch_gate,
            "switch_gate": switch_gate[:, -1, :],
            "correction_gate_seq": corr_gate,
            "correction_gate": corr_gate[:, -1, :],
            "latent_seq": latent_seq,
            "gate_enabled": True,
        }
        if self.velocity_head is not None:
            vel_seq = self.velocity_head(latent_seq)
            out["velocity_seq"] = vel_seq
            out["velocity"] = vel_seq[:, -1, :]
        if self.future_head is not None:
            if horizon_ms is None:
                horizon_ms = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            horizon_norm = horizon_ms / 1000.0
            future_in = torch.cat([latent_seq[:, -1, :], horizon_norm], dim=-1)
            out["future"] = self.future_head(future_in)
        return out
