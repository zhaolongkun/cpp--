from __future__ import annotations

from typing import Any, Dict, Tuple

from .la_cspc_ornet import LACSPCORNet, LACSPCORNetConfig
from .tcn_gru_baseline import TCNGRUBaselineConfig, TCNGRUCleanBaseline


def build_model(model_cfg: Dict[str, Any]):
    name = str(model_cfg.get("name", "dual_state"))
    if name == "dual_state":
        cfg = LACSPCORNetConfig(
            input_dim=int(model_cfg["input_dim"]),
            stem_dim=int(model_cfg.get("stem_dim", 64)),
            state_dim=int(model_cfg.get("state_dim", 64)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            norm_type=str(model_cfg.get("norm_type", "groupnorm")),
            groupnorm_groups=int(model_cfg.get("groupnorm_groups", 8)),
            residual_on_input=bool(model_cfg.get("residual_on_input", True)),
            switch_gate_mode=str(model_cfg.get("switch_gate_mode", "learned")),
            explicit_gate_switch_weight=float(model_cfg.get("explicit_gate_switch_weight", 0.35)),
            explicit_gate_turn_weight=float(model_cfg.get("explicit_gate_turn_weight", 0.35)),
            explicit_gate_lost_weight=float(model_cfg.get("explicit_gate_lost_weight", 0.15)),
            explicit_gate_coast_weight=float(model_cfg.get("explicit_gate_coast_weight", 0.10)),
            explicit_gate_zoom_weight=float(model_cfg.get("explicit_gate_zoom_weight", 0.05)),
            explicit_gate_jitter_weight=float(model_cfg.get("explicit_gate_jitter_weight", 0.25)),
            explicit_gate_jitter_scale=float(model_cfg.get("explicit_gate_jitter_scale", 1.50)),
            explicit_gate_threshold=float(model_cfg.get("explicit_gate_threshold", 0.45)),
            explicit_gate_temperature=float(model_cfg.get("explicit_gate_temperature", 0.12)),
            recover_residual_cap_enabled=bool(model_cfg.get("recover_residual_cap_enabled", True)),
            recover_residual_cap_value=float(model_cfg.get("recover_residual_cap_value", 8.0)),
            recover_residual_min_scale=float(model_cfg.get("recover_residual_min_scale", 0.25)),
            recover_residual_trigger_threshold=float(model_cfg.get("recover_residual_trigger_threshold", 0.55)),
            residual_base_gain=float(model_cfg.get("residual_base_gain", 0.10)),
            residual_gate_gain=float(model_cfg.get("residual_gate_gain", 0.90)),
            lead_enabled=bool(model_cfg.get("lead_enabled", False)),
            future_head=bool(model_cfg.get("future_head", False)),
            velocity_head=bool(model_cfg.get("velocity_head", False)),
        )
        return name, cfg, LACSPCORNet(cfg)
    if name == "tcn_gru":
        cfg = TCNGRUBaselineConfig(
            input_dim=int(model_cfg["input_dim"]),
            stem_dim=int(model_cfg.get("stem_dim", 64)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            norm_type=str(model_cfg.get("norm_type", "groupnorm")),
            groupnorm_groups=int(model_cfg.get("groupnorm_groups", 8)),
            residual_on_input=bool(model_cfg.get("residual_on_input", True)),
            velocity_head=bool(model_cfg.get("velocity_head", False)),
            future_head=bool(model_cfg.get("future_head", False)),
        )
        return name, cfg, TCNGRUCleanBaseline(cfg)
    raise ValueError(f"unsupported model name: {name}")
