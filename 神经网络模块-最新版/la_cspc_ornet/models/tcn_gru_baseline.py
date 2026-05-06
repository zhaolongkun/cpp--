from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .la_cspc_ornet import CausalTCNStem


@dataclass
class TCNGRUBaselineConfig:
    input_dim: int
    stem_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.10
    norm_type: str = "groupnorm"
    groupnorm_groups: int = 8
    residual_on_input: bool = True
    velocity_head: bool = False
    future_head: bool = False


class TCNGRUCleanBaseline(nn.Module):
    def __init__(self, cfg: TCNGRUBaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = CausalTCNStem(
            cfg.input_dim,
            cfg.stem_dim,
            channels=cfg.stem_dim,
            norm_type=cfg.norm_type,
            groupnorm_groups=cfg.groupnorm_groups,
        )
        self.gru = nn.GRU(cfg.stem_dim, cfg.hidden_dim, batch_first=True)
        self.clean_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.velocity_head = None
        if cfg.velocity_head:
            self.velocity_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, 2),
            )

    def forward(
        self,
        x: torch.Tensor,
        horizon_ms=None,
        baseline_seq: torch.Tensor | None = None,
        stable_baseline_seq: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        stem_seq = self.stem(x)
        latent_seq, _ = self.gru(stem_seq)
        clean_residual_seq = self.clean_head(latent_seq)
        clean_seq = clean_residual_seq
        if self.cfg.residual_on_input:
            if baseline_seq is None:
                raise ValueError("baseline_seq is required when residual_on_input=true")
            clean_seq = baseline_seq + clean_residual_seq
        out: Dict[str, torch.Tensor] = {
            "clean_seq": clean_seq,
            "clean": clean_seq[:, -1, :],
            "lead_delta_seq": torch.zeros_like(clean_seq),
            "lead_delta": torch.zeros_like(clean_seq[:, -1, :]),
            "switch_gate_seq": torch.zeros(clean_seq.shape[0], clean_seq.shape[1], 1, device=clean_seq.device),
            "switch_gate": torch.zeros(clean_seq.shape[0], 1, device=clean_seq.device),
            "correction_gate_seq": torch.zeros(clean_seq.shape[0], clean_seq.shape[1], 1, device=clean_seq.device),
            "correction_gate": torch.zeros(clean_seq.shape[0], 1, device=clean_seq.device),
            "latent_seq": latent_seq,
            "gate_enabled": False,
        }
        if self.velocity_head is not None:
            vel_seq = self.velocity_head(latent_seq)
            out["velocity_seq"] = vel_seq
            out["velocity"] = vel_seq[:, -1, :]
        return out
