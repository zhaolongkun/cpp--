#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize full collection progress by scenario bucket")
    p.add_argument("--progress_csv", required=True, help="Run-level progress CSV")
    p.add_argument("--output_csv", required=True, help="Scenario-level progress CSV")
    p.add_argument("--target_full_logs", type=int, default=20)
    p.add_argument("--target_pseudo_valid_rows", type=int, default=3000)
    p.add_argument("--target_usable_sequences", type=int, default=500)
    p.add_argument("--stable_target_valid_rows", type=int, default=800)
    p.add_argument("--scenario_target_valid_rows", type=int, default=800)
    p.add_argument("--per_scenario_target_usable_sequences", type=int, default=120)
    p.add_argument("--maneuver_success_target_runs", type=int, default=3)
    p.add_argument("--recovery_hold_target_runs", type=int, default=3)
    return p.parse_args()


def normalize_scenario(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("normal", "stable"):
        return "stable"
    if s == "maneuver":
        return "maneuver"
    if s in ("loss", "recovery"):
        return "recovery"
    return s or "unknown"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.progress_csv)
    if df.empty:
        raise RuntimeError("progress_csv is empty")

    df = df.copy()
    df["scenario_bucket"] = df["scenario"].map(normalize_scenario)

    scenario_rows: List[Dict[str, object]] = []
    for scenario in ["stable", "maneuver", "recovery"]:
        seg = df[df["scenario_bucket"] == scenario]
        num_runs = int(len(seg))
        target_positive_runs = int(seg["has_target_positive"].sum()) if num_runs else 0
        recovery_runs = int(seg["has_recovery_event"].sum()) if num_runs else 0
        recovery_meets_hold_runs = int(seg["recovery_meets_hold"].sum()) if num_runs else 0
        pseudo_valid_rows = int(seg["pseudo_valid_rows"].sum()) if num_runs else 0
        usable_sequences = int(seg["usable_sequences"].sum()) if num_runs else 0
        total_rows = int(seg["rows"].sum()) if num_runs else 0
        valid_ratio = float(pseudo_valid_rows / total_rows) if total_rows > 0 else 0.0

        if scenario == "stable":
            enough = bool(
                pseudo_valid_rows >= args.stable_target_valid_rows
                and usable_sequences >= args.per_scenario_target_usable_sequences
            )
            admission_reason = (
                f"stable_rows>={args.stable_target_valid_rows} and usable>={args.per_scenario_target_usable_sequences}"
            )
        elif scenario == "maneuver":
            enough = bool(
                target_positive_runs >= args.maneuver_success_target_runs
                and usable_sequences >= args.per_scenario_target_usable_sequences
            )
            admission_reason = (
                f"successful_runs>={args.maneuver_success_target_runs} and usable>={args.per_scenario_target_usable_sequences}"
            )
        else:
            enough = bool(
                recovery_meets_hold_runs >= args.recovery_hold_target_runs
                and usable_sequences >= args.per_scenario_target_usable_sequences
            )
            admission_reason = (
                f"hold_runs>={args.recovery_hold_target_runs} and usable>={args.per_scenario_target_usable_sequences}"
            )

        scenario_rows.append(
            {
                "scenario": scenario,
                "num_runs": num_runs,
                "target_positive_runs": target_positive_runs,
                "recovery_runs": recovery_runs,
                "recovery_meets_hold_runs": recovery_meets_hold_runs,
                "pseudo_valid_rows": pseudo_valid_rows,
                "usable_sequences": usable_sequences,
                "valid_ratio": valid_ratio,
                "enough_for_training_subtask": int(enough),
                "admission_rule": admission_reason,
                "gap_pseudo_valid_rows": max(
                    0,
                    (args.stable_target_valid_rows if scenario == "stable" else args.scenario_target_valid_rows) - pseudo_valid_rows,
                ),
                "gap_usable_sequences": max(0, args.per_scenario_target_usable_sequences - usable_sequences),
                "gap_successful_runs": max(0, args.maneuver_success_target_runs - target_positive_runs) if scenario == "maneuver" else 0,
                "gap_recovery_hold_runs": max(0, args.recovery_hold_target_runs - recovery_meets_hold_runs) if scenario == "recovery" else 0,
            }
        )

    out_df = pd.DataFrame(scenario_rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        {
            "output_csv": str(out_path),
            "rows": int(len(out_df)),
            "scenario_summary": out_df[["scenario", "pseudo_valid_rows", "usable_sequences"]].to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    main()
