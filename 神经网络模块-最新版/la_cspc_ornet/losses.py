from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .features import FEATURE_INDEX


@dataclass
class Stage1LossConfig:
    clean_weight: float = 1.0
    smooth_weight: float = 0.01
    turn_weight: float = 0.16
    delta_weight: float = 0.12
    peak_weight: float = 0.10
    peak_margin: float = 0.85
    event_recon_boost: float = 0.75
    gate_align_weight: float = 0.0
    gate_sparse_weight: float = 0.0


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float = 2.5) -> torch.Tensor:
    err = F.huber_loss(pred, target, reduction="none", delta=delta)
    while weight.ndim < err.ndim:
        weight = weight.unsqueeze(-1)
    return (err * weight).mean()


def last_event_score(x: torch.Tensor) -> torch.Tensor:
    switch = x[:, -1, FEATURE_INDEX.switch_score : FEATURE_INDEX.switch_score + 1]
    turn = x[:, -1, FEATURE_INDEX.turn_score : FEATURE_INDEX.turn_score + 1]
    lost = x[:, -1, FEATURE_INDEX.lost_flag : FEATURE_INDEX.lost_flag + 1]
    coast = x[:, -1, FEATURE_INDEX.coast_count : FEATURE_INDEX.coast_count + 1]
    zoom = torch.abs(x[:, -1, FEATURE_INDEX.zoom_delta : FEATURE_INDEX.zoom_delta + 1])
    return torch.clamp(torch.maximum(torch.maximum(switch, turn), torch.maximum(lost, torch.maximum(coast, zoom))), 0.0, 1.0)


def smoothness_loss(clean_seq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if clean_seq.size(1) < 3:
        return clean_seq.new_zeros(())
    clean_d2 = clean_seq[:, 2:, :] - 2.0 * clean_seq[:, 1:-1, :] + clean_seq[:, :-2, :]
    switch = x[:, 2:, FEATURE_INDEX.switch_score : FEATURE_INDEX.switch_score + 1]
    turn = x[:, 2:, FEATURE_INDEX.turn_score : FEATURE_INDEX.turn_score + 1]
    lost = x[:, 2:, FEATURE_INDEX.lost_flag : FEATURE_INDEX.lost_flag + 1]
    coast = x[:, 2:, FEATURE_INDEX.coast_count : FEATURE_INDEX.coast_count + 1]
    zoom = torch.abs(x[:, 2:, FEATURE_INDEX.zoom_delta : FEATURE_INDEX.zoom_delta + 1])
    jitter_proxy = torch.clamp(
        torch.sqrt(
            x[:, 2:, FEATURE_INDEX.d1_dx : FEATURE_INDEX.d1_dx + 1] ** 2
            + x[:, 2:, FEATURE_INDEX.d1_dy : FEATURE_INDEX.d1_dy + 1] ** 2
        )
        / 2.0,
        0.0,
        1.0,
    )
    non_event = 1.0 - torch.clamp(
        torch.maximum(
            torch.maximum(torch.maximum(switch, turn), torch.maximum(lost, torch.maximum(coast, zoom))),
            jitter_proxy,
        ),
        0.0,
        1.0,
    )
    return (non_event * torch.abs(clean_d2)).mean()


def turn_preservation_loss(clean_seq: torch.Tensor, target_seq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if clean_seq.size(1) < 3:
        return clean_seq.new_zeros(())
    clean_d1 = clean_seq[:, 1:, :] - clean_seq[:, :-1, :]
    target_d1 = target_seq[:, 1:, :] - target_seq[:, :-1, :]
    clean_d2 = clean_d1[:, 1:, :] - clean_d1[:, :-1, :]
    target_d2 = target_d1[:, 1:, :] - target_d1[:, :-1, :]
    event = torch.maximum(
        x[:, 2:, FEATURE_INDEX.switch_score : FEATURE_INDEX.switch_score + 1],
        x[:, 2:, FEATURE_INDEX.turn_score : FEATURE_INDEX.turn_score + 1],
    )
    return (event * (torch.abs(clean_d1[:, 1:, :] - target_d1[:, 1:, :]) + torch.abs(clean_d2 - target_d2))).mean()


def delta_consistency_loss(clean_seq: torch.Tensor, target_seq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if clean_seq.size(1) < 2:
        return clean_seq.new_zeros(())
    clean_d1 = clean_seq[:, 1:, :] - clean_seq[:, :-1, :]
    target_d1 = target_seq[:, 1:, :] - target_seq[:, :-1, :]
    switch = x[:, 1:, FEATURE_INDEX.switch_score : FEATURE_INDEX.switch_score + 1]
    turn = x[:, 1:, FEATURE_INDEX.turn_score : FEATURE_INDEX.turn_score + 1]
    lost = x[:, 1:, FEATURE_INDEX.lost_flag : FEATURE_INDEX.lost_flag + 1]
    zoom = torch.abs(x[:, 1:, FEATURE_INDEX.zoom_delta : FEATURE_INDEX.zoom_delta + 1])
    jitter_proxy = torch.clamp(
        torch.sqrt(
            x[:, 1:, FEATURE_INDEX.d1_dx : FEATURE_INDEX.d1_dx + 1] ** 2
            + x[:, 1:, FEATURE_INDEX.d1_dy : FEATURE_INDEX.d1_dy + 1] ** 2
        )
        / 2.0,
        0.0,
        1.0,
    )
    event = torch.clamp(torch.maximum(torch.maximum(torch.maximum(switch, turn), torch.maximum(lost, zoom)), jitter_proxy), 0.0, 1.0)
    return (event * torch.abs(clean_d1 - target_d1)).mean()


def peak_preservation_loss(clean_seq: torch.Tensor, target_seq: torch.Tensor, x: torch.Tensor, margin: float) -> torch.Tensor:
    if clean_seq.size(1) < 2:
        return clean_seq.new_zeros(())
    clean_d1 = clean_seq[:, 1:, :] - clean_seq[:, :-1, :]
    target_d1 = target_seq[:, 1:, :] - target_seq[:, :-1, :]
    switch = x[:, 1:, FEATURE_INDEX.switch_score : FEATURE_INDEX.switch_score + 1]
    turn = x[:, 1:, FEATURE_INDEX.turn_score : FEATURE_INDEX.turn_score + 1]
    jitter_proxy = torch.clamp(
        torch.sqrt(
            x[:, 1:, FEATURE_INDEX.d1_dx : FEATURE_INDEX.d1_dx + 1] ** 2
            + x[:, 1:, FEATURE_INDEX.d1_dy : FEATURE_INDEX.d1_dy + 1] ** 2
        )
        / 2.0,
        0.0,
        1.0,
    )
    event = torch.clamp(torch.maximum(torch.maximum(switch, turn), jitter_proxy), 0.0, 1.0)
    clean_mag = torch.abs(clean_d1)
    target_mag = torch.abs(target_d1)
    missing_peak = torch.relu(margin * target_mag - clean_mag)
    return (event * missing_peak).mean()


def gate_target_seq(x: torch.Tensor) -> torch.Tensor:
    switch = x[:, :, FEATURE_INDEX.switch_score : FEATURE_INDEX.switch_score + 1]
    turn = x[:, :, FEATURE_INDEX.turn_score : FEATURE_INDEX.turn_score + 1]
    lost = x[:, :, FEATURE_INDEX.lost_flag : FEATURE_INDEX.lost_flag + 1]
    coast = x[:, :, FEATURE_INDEX.coast_count : FEATURE_INDEX.coast_count + 1]
    zoom = torch.abs(x[:, :, FEATURE_INDEX.zoom_delta : FEATURE_INDEX.zoom_delta + 1])
    jitter_proxy = torch.clamp(
        torch.sqrt(
            x[:, :, FEATURE_INDEX.d1_dx : FEATURE_INDEX.d1_dx + 1] ** 2
            + x[:, :, FEATURE_INDEX.d1_dy : FEATURE_INDEX.d1_dy + 1] ** 2
        )
        / 2.0,
        0.0,
        1.0,
    )
    return torch.clamp(
        torch.maximum(
            torch.maximum(torch.maximum(switch, turn), torch.maximum(lost, torch.maximum(coast, zoom))),
            jitter_proxy,
        ),
        0.0,
        1.0,
    )


def gate_alignment_loss(gate_seq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    target = gate_target_seq(x)
    gate_seq = torch.clamp(gate_seq, 1e-4, 1.0 - 1e-4)
    return F.binary_cross_entropy(gate_seq, target)


def gate_sparsity_loss(gate_seq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    target = gate_target_seq(x)
    non_event = 1.0 - target
    return (non_event * gate_seq).mean()


def compute_stage1_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Stage1LossConfig) -> Dict[str, torch.Tensor]:
    x = batch["x"]
    y_clean = batch["y_clean"]
    y_clean_seq = batch["y_clean_seq"]
    base_weight = batch["weight"]
    event = last_event_score(x)
    recon_weight = base_weight * (1.0 + cfg.event_recon_boost * event)

    losses: Dict[str, torch.Tensor] = {}
    losses["clean"] = weighted_huber(outputs["clean"], y_clean, recon_weight)
    losses["smooth"] = smoothness_loss(outputs["clean_seq"], x)
    losses["turn"] = turn_preservation_loss(outputs["clean_seq"], y_clean_seq, x)
    losses["delta"] = delta_consistency_loss(outputs["clean_seq"], y_clean_seq, x)
    losses["peak"] = peak_preservation_loss(outputs["clean_seq"], y_clean_seq, x, cfg.peak_margin)
    gate_enabled = bool(outputs.get("gate_enabled", False))
    if gate_enabled and "switch_gate_seq" in outputs:
        losses["gate_align"] = gate_alignment_loss(outputs["switch_gate_seq"], x)
        losses["gate_sparse"] = gate_sparsity_loss(outputs["switch_gate_seq"], x)
    else:
        losses["gate_align"] = losses["clean"].new_zeros(())
        losses["gate_sparse"] = losses["clean"].new_zeros(())
    losses["total"] = (
        cfg.clean_weight * losses["clean"]
        + cfg.smooth_weight * losses["smooth"]
        + cfg.turn_weight * losses["turn"]
        + cfg.delta_weight * losses["delta"]
        + cfg.peak_weight * losses["peak"]
        + cfg.gate_align_weight * losses["gate_align"]
        + cfg.gate_sparse_weight * losses["gate_sparse"]
    )
    return losses
