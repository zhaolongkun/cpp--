#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


SCENARIO_ORDER = ["recovery", "maneuver", "stable"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build scenario repeat-coverage status dashboard")
    p.add_argument("--scenario_summary_csv", required=True)
    p.add_argument("--success_rate_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--valid_rows_target", type=int, default=800)
    p.add_argument("--usable_sequences_target", type=int, default=120)
    p.add_argument("--successful_runs_target", type=int, default=3)
    p.add_argument("--recovery_hold_target", type=int, default=3)
    return p.parse_args()


def as_int(row: pd.Series, key: str) -> int:
    if key not in row.index:
        return 0
    value = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
    return int(value) if pd.notna(value) else 0


def classify_stage(
    scenario: str,
    valid_rows_current: int,
    valid_rows_target: int,
    usable_sequences_current: int,
    usable_sequences_target: int,
    successful_runs_current: int,
    successful_runs_target: int,
    meets_hold_runs_current: int,
    meets_hold_runs_target: int,
) -> str:
    if (
        valid_rows_current >= valid_rows_target
        and usable_sequences_current >= usable_sequences_target
        and successful_runs_current >= successful_runs_target
        and meets_hold_runs_current >= meets_hold_runs_target
    ):
        return "training-ready"
    if successful_runs_current <= 0 or valid_rows_current <= 0:
        return "zero-sample"
    if scenario == "recovery" and meets_hold_runs_current < max(1, meets_hold_runs_target):
        if meets_hold_runs_current <= 1:
            return "first-breakthrough"
        return "repeat-coverage"
    if successful_runs_current <= 1:
        return "first-breakthrough"
    return "repeat-coverage"


def stage_reason(stage: str, scenario: str, hold_gap: int, valid_gap: int, success_gap: int) -> str:
    if stage == "training-ready":
        return "All repeat-coverage thresholds are satisfied for this scenario."
    if stage == "zero-sample":
        return "No successful supervised capture has been accumulated yet."
    if stage == "first-breakthrough":
        if scenario == "recovery" and hold_gap > 0:
            return "The first hold-satisfying recovery run exists, but repeated hold-success coverage is still missing."
        return "The first successful supervised run exists, but repeated successful coverage is still missing."
    if scenario == "recovery" and hold_gap > 0:
        return "Recovery has repeated evidence, but the hold-success count is still below the target."
    if valid_gap > 0 or success_gap > 0:
        return "Repeated successful capture exists, but scenario strength is still below the formal training target."
    return "Repeated successful capture exists and only residual balance gaps remain."


def build_markdown(df: pd.DataFrame) -> str:
    lines: List[str] = [
        "# Scenario Repeat Coverage Status",
        "",
        "The current stage is no longer zero-sample acquisition, but repeated scenario coverage enhancement for maneuver and recovery.",
        "",
    ]
    for _, row in df.iterrows():
        lines.extend(
            [
                f"## {row['scenario']}",
                "",
                f"- Coverage stage: `{row['coverage_stage']}`",
                f"- Valid rows: `{row['valid_rows_current']}/{row['valid_rows_target']}`",
                f"- Usable sequences: `{row['usable_sequences_current']}/{row['usable_sequences_target']}`",
                f"- Successful runs: `{row['successful_runs_current']}/{row['successful_runs_target']}`",
                f"- Hold-satisfying runs: `{row['meets_hold_runs_current']}/{row['meets_hold_runs_target']}`",
                f"- Remaining gaps: valid rows `{row['valid_rows_gap']}`, usable sequences `{row['usable_sequences_gap']}`, successful runs `{row['successful_runs_gap']}`, hold runs `{row['meets_hold_runs_gap']}`",
                f"- Reason: {row['coverage_stage_reason']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    scenario_summary = pd.read_csv(args.scenario_summary_csv)
    success_rate = pd.read_csv(args.success_rate_csv)

    rows: List[Dict[str, object]] = []
    for scenario in SCENARIO_ORDER:
        summary_row = scenario_summary[scenario_summary["scenario"].astype(str) == scenario]
        summary = summary_row.iloc[0] if not summary_row.empty else pd.Series(dtype=object)
        rate_row = success_rate[success_rate["scenario"].astype(str) == scenario]
        rate = rate_row.iloc[0] if not rate_row.empty else pd.Series(dtype=object)

        valid_rows_current = as_int(summary, "pseudo_valid_rows")
        usable_sequences_current = as_int(summary, "usable_sequences")
        successful_runs_current = as_int(rate, "successful_runs")
        meets_hold_runs_current = as_int(rate, "recovery_meets_hold_runs") if scenario == "recovery" else 0

        meets_hold_runs_target = args.recovery_hold_target if scenario == "recovery" else 0
        stage = classify_stage(
            scenario=scenario,
            valid_rows_current=valid_rows_current,
            valid_rows_target=args.valid_rows_target,
            usable_sequences_current=usable_sequences_current,
            usable_sequences_target=args.usable_sequences_target,
            successful_runs_current=successful_runs_current,
            successful_runs_target=args.successful_runs_target,
            meets_hold_runs_current=meets_hold_runs_current,
            meets_hold_runs_target=meets_hold_runs_target,
        )

        valid_rows_gap = max(0, args.valid_rows_target - valid_rows_current)
        usable_sequences_gap = max(0, args.usable_sequences_target - usable_sequences_current)
        successful_runs_gap = max(0, args.successful_runs_target - successful_runs_current)
        meets_hold_runs_gap = max(0, meets_hold_runs_target - meets_hold_runs_current)

        rows.append(
            {
                "scenario": scenario,
                "valid_rows_current": valid_rows_current,
                "valid_rows_target": args.valid_rows_target,
                "usable_sequences_current": usable_sequences_current,
                "usable_sequences_target": args.usable_sequences_target,
                "successful_runs_current": successful_runs_current,
                "successful_runs_target": args.successful_runs_target,
                "meets_hold_runs_current": meets_hold_runs_current,
                "meets_hold_runs_target": meets_hold_runs_target,
                "valid_rows_gap": valid_rows_gap,
                "usable_sequences_gap": usable_sequences_gap,
                "successful_runs_gap": successful_runs_gap,
                "meets_hold_runs_gap": meets_hold_runs_gap,
                "coverage_stage": stage,
                "coverage_stage_reason": stage_reason(
                    stage=stage,
                    scenario=scenario,
                    hold_gap=meets_hold_runs_gap,
                    valid_gap=valid_rows_gap,
                    success_gap=successful_runs_gap,
                ),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    out_md.write_text(build_markdown(out_df), encoding="utf-8")
    print({"output_csv": str(out_csv), "output_md": str(out_md), "rows": int(len(out_df))})


if __name__ == "__main__":
    main()
