#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh training readiness flags from total and scenario thresholds")
    p.add_argument("--registry_csv", required=True)
    p.add_argument("--dataset_audit_json", required=True)
    p.add_argument("--scenario_summary_csv", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry_csv)
    audit_path = Path(args.dataset_audit_json)
    scenario_path = Path(args.scenario_summary_csv)

    registry = pd.read_csv(registry_path)
    with audit_path.open("r", encoding="utf-8") as f:
        audit = json.load(f)
    scenario = pd.read_csv(scenario_path)

    full_logs_current = int(pd.to_numeric(scenario.get("num_runs", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not scenario.empty else 0

    total_ok = bool(
        full_logs_current >= 20
        and audit.get("total_pseudo_expert_valid_rows", 0) >= 3000
        and audit.get("usable_sequences_at_seq_len", 0) >= 500
    )
    scenario_ok = bool((scenario["enough_for_training_subtask"] == 1).all()) if not scenario.empty else False

    registry = registry.copy()
    registry["requires_training"] = registry["supervision_variant"].astype(str).ne("none").astype(int)
    registry["ready_to_train"] = 0
    registry["gate_status"] = "blocked_by_data_thresholds"

    baseline_mask = registry["supervision_variant"].astype(str).eq("none")
    registry.loc[baseline_mask, "ready_to_train"] = 1
    registry.loc[baseline_mask, "gate_status"] = "no_training_required"

    learned_mask = ~baseline_mask
    if total_ok and scenario_ok:
        registry.loc[learned_mask, "ready_to_train"] = 1
        registry.loc[learned_mask, "gate_status"] = "ready"

    registry.to_csv(registry_path, index=False, encoding="utf-8")
    print(
        {
            "registry_csv": str(registry_path),
            "full_logs_current": full_logs_current,
            "total_ok": total_ok,
            "scenario_ok": scenario_ok,
            "learned_ready_count": int(registry.loc[learned_mask, "ready_to_train"].sum()),
        }
    )


if __name__ == "__main__":
    main()
