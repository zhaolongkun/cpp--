#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


SCENARIO_ORDER = {
    "maneuver": 0,
    "stable": 1,
    "recovery": 2,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build scenario-wise capture guidance from current collection progress and success profiles"
    )
    p.add_argument("--progress_csv", required=True, help="full_collection_progress.csv")
    p.add_argument("--scenario_summary_csv", required=True, help="full_collection_progress_by_scenario.csv")
    p.add_argument("--profile_csv", required=True, help="target_presentation_success_profile.csv")
    p.add_argument("--success_rate_csv", required=True, help="scenario_capture_success_rate.csv")
    p.add_argument("--output_csv", required=True, help="Scenario-wise guidance CSV")
    p.add_argument("--output_md", required=True, help="Scenario-wise guidance Markdown summary")
    return p.parse_args()


def as_int(row: pd.Series, key: str) -> int:
    if key not in row.index:
        return 0
    v = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
    return int(v) if pd.notna(v) else 0


def as_float(row: pd.Series, key: str) -> float:
    if key not in row.index:
        return 0.0
    v = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else 0.0


def scenario_priority(scenario: str) -> int:
    return SCENARIO_ORDER.get(str(scenario), 99)


def build_next_target(scenario: str, gap_rows: int, gap_hold_runs: int) -> str:
    if scenario == "maneuver":
        if gap_rows > 0:
            return (
                f"Repeat the successful in-frame maneuver envelope to add {gap_rows} more valid rows "
                f"and raise successful maneuver runs toward repeated coverage."
            )
        return "Maneuver bucket meets current thresholds; keep collecting only if balanced coverage slips."
    if scenario == "stable":
        if gap_rows > 0:
            return (
                f"Add stable-scene supervision volume to close the remaining {gap_rows} valid-row gap "
                f"and the admission-level usable-sequence gap."
            )
        return "Stable bucket meets current thresholds; keep collecting only if balance gating remains active."
    if scenario == "recovery":
        if gap_hold_runs > 0:
            return (
                f"Collect {gap_hold_runs} more hold-satisfying loss-to-reacquire runs; "
                f"keep re-entry center-near and close {gap_rows} remaining valid-row gap."
            )
        if gap_rows > 0:
            return (
                f"Repeat the successful recovery envelope until the remaining {gap_rows} valid-row gap closes."
            )
        return "Recovery bucket meets current thresholds; keep collecting only if repetition quality degrades."
    return "No additional target defined."


def build_operator_note(scenario: str, success_rate: float, gap_rows: int, gap_hold_runs: int) -> str:
    if scenario == "maneuver":
        if gap_rows > 0:
            return (
                "Maneuver is now the weakest key scenario. Keep the target large and fully in frame through both horizontal "
                "and vertical bursts, and optimize for repeated successful runs rather than one-off positives."
            )
        return "Maneuver coverage is adequate; collect only if more repetition is needed for balance."
    if scenario == "stable":
        return (
            "Stable is no longer the key-scene evidence blocker, but it remains the largest admission-level volume deficit. "
            "Use stable captures to raise training readiness rather than to prove scenario existence."
        )
    if success_rate < 0.5:
        return (
            "Recovery is already criterion-satisfying, so it is no longer the first priority. "
            "Collect it only to close the remaining row gap or to strengthen recovery-side quantitative discussion."
        )
    return (
        "Recovery has cleared the hold-success criterion and now serves as a quantitative discussion scene; "
        "additional runs mainly improve row volume and repeated recovery statistics."
    )


def build_research_role(scenario: str) -> str:
    if scenario == "maneuver":
        return (
            "Support the claim that reliability-aware residual injection and bounded safety fusion remain effective "
            "under high-dynamic image-plane motion."
        )
    if scenario == "stable":
        return (
            "Provide the most reliable supervision basis for nominal-to-residual learning and improve the repeatability "
            "of later trained-method comparisons."
        )
    if scenario == "recovery":
        return (
            "Support quantitative discussion of conservative loss handling and structured command resumption after reacquisition."
        )
    return "No specific research role defined."


def pick_template_run(progress: pd.DataFrame, scenario: str) -> str:
    candidates = progress[progress["scenario"].astype(str) == str(scenario)].copy()
    candidates = candidates[pd.to_numeric(candidates["pseudo_valid_rows"], errors="coerce").fillna(0) > 0].copy()
    if candidates.empty:
        return ""
    candidates["pseudo_valid_rows_num"] = pd.to_numeric(candidates["pseudo_valid_rows"], errors="coerce").fillna(0)
    candidates["usable_sequences_num"] = pd.to_numeric(candidates["usable_sequences"], errors="coerce").fillna(0)
    candidates["rows_num"] = pd.to_numeric(candidates["rows"], errors="coerce").fillna(0)
    candidates = candidates.sort_values(
        ["pseudo_valid_rows_num", "usable_sequences_num", "rows_num"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return str(candidates.iloc[0]["run_id"])


def build_markdown(df: pd.DataFrame) -> str:
    lines: List[str] = [
        "# Scenario Capture Guidance",
        "",
        "This file is derived from the latest `training_full` statistics and should be treated as the current source of truth for live full-profile collection.",
        "The current stage is no longer zero-sample acquisition, but repeated scenario coverage enhancement for maneuver and recovery.",
        "",
    ]
    for _, row in df.iterrows():
        scenario = str(row["scenario"])
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"- Priority: `{as_int(row, 'priority_rank')}`",
                f"- Coverage stage: `{row['coverage_stage']}`",
                f"- Primary KPI: `{row['primary_kpi']}`",
                f"- Coverage now: `{as_int(row, 'current_pseudo_valid_rows')}` valid rows, `{as_int(row, 'current_usable_sequences')}` usable sequences, success `{as_int(row, 'successful_runs')}/{as_int(row, 'attempted_runs')}`",
                f"- Remaining gap: `{as_int(row, 'gap_pseudo_valid_rows')}` valid rows, `{as_int(row, 'gap_usable_sequences')}` usable sequences, `{as_int(row, 'gap_recovery_hold_runs')}` recovery-hold runs",
            ]
        )
        if str(row.get("template_run_id", "")):
            lines.extend(
                [
                    f"- Template run: `{row['template_run_id']}`",
                    f"- Observed successful envelope: `det_conf_mean={as_float(row, 'observed_det_conf_mean_success'):.3f}`, `det_conf_p90={as_float(row, 'observed_det_conf_p90_success'):.3f}`, `bbox_area_mean={as_float(row, 'observed_bbox_area_mean_success'):.1f}`, `bbox_area_p90={as_float(row, 'observed_bbox_area_p90_success'):.1f}`",
                    f"- Observed hold/tracked duration: `longest_tracked_segment_ms={as_float(row, 'observed_longest_tracked_segment_ms_success'):.1f}`, `reacquire_tracked_hold_ms={as_float(row, 'observed_reacquire_tracked_hold_ms_success'):.1f}`",
                ]
            )
        lines.extend(
            [
                f"- Next target: {row['next_capture_target']}",
                f"- Research role: {row['research_role']}",
                f"- Operator note: {row['operator_note']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    progress = pd.read_csv(args.progress_csv)
    scenario_summary = pd.read_csv(args.scenario_summary_csv)
    profile = pd.read_csv(args.profile_csv)
    success_rate = pd.read_csv(args.success_rate_csv)

    scenarios = sorted(
        set(scenario_summary.get("scenario", pd.Series(dtype=str)).astype(str).tolist())
        | set(success_rate.get("scenario", pd.Series(dtype=str)).astype(str).tolist()),
        key=scenario_priority,
    )

    profile_runs = profile[profile["profile_type"].astype(str) == "run"].copy()
    profile_success = profile_runs[profile_runs["success_group"].astype(str) == "success"].copy()

    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        summary_row = scenario_summary[scenario_summary["scenario"].astype(str) == scenario]
        summary = summary_row.iloc[0] if not summary_row.empty else pd.Series(dtype=object)

        rate_row = success_rate[success_rate["scenario"].astype(str) == scenario]
        rate = rate_row.iloc[0] if not rate_row.empty else pd.Series(dtype=object)

        template_run_id = pick_template_run(progress, scenario)
        profile_row = profile_success[
            (profile_success["scenario"].astype(str) == scenario)
            & (profile_success["run_id"].astype(str) == template_run_id)
        ]
        if profile_row.empty:
            profile_row = profile_success[profile_success["scenario"].astype(str) == scenario]
        profile_series = profile_row.iloc[0] if not profile_row.empty else pd.Series(dtype=object)

        gap_rows = as_int(summary, "gap_pseudo_valid_rows")
        gap_usable = as_int(summary, "gap_usable_sequences")
        gap_hold = as_int(summary, "gap_recovery_hold_runs")
        success_rate_value = as_float(rate, "success_rate")

        rows.append(
            {
                "scenario": scenario,
                "priority_rank": scenario_priority(scenario) + 1,
                "attempted_runs": as_int(rate, "attempted_runs") or as_int(summary, "num_runs"),
                "successful_runs": as_int(rate, "successful_runs") or as_int(summary, "target_positive_runs"),
                "success_rate": success_rate_value,
                "current_pseudo_valid_rows": as_int(summary, "pseudo_valid_rows"),
                "current_usable_sequences": as_int(summary, "usable_sequences"),
                "gap_pseudo_valid_rows": gap_rows,
                "gap_usable_sequences": gap_usable,
                "gap_recovery_hold_runs": gap_hold,
                "coverage_stage": (
                    "training-ready"
                    if gap_rows <= 0 and gap_usable <= 0 and gap_hold <= 0
                    else ("first-breakthrough" if as_int(rate, "successful_runs") <= 1 else "repeat-coverage")
                ),
                "primary_kpi": (
                    "recovery_meets_hold_runs"
                    if scenario == "recovery" and gap_hold > 0
                    else ("pseudo_valid_rows" if gap_rows > 0 else "repetition_quality")
                ),
                "template_run_id": template_run_id,
                "observed_det_positive_ratio_success": as_float(profile_series, "det_positive_ratio"),
                "observed_det_conf_mean_success": as_float(profile_series, "det_conf_mean"),
                "observed_det_conf_p90_success": as_float(profile_series, "det_conf_p90"),
                "observed_bbox_area_mean_success": as_float(profile_series, "bbox_area_mean"),
                "observed_bbox_area_p90_success": as_float(profile_series, "bbox_area_p90"),
                "observed_tracked_ratio_success": as_float(profile_series, "tracked_ratio"),
                "observed_longest_tracked_segment_ms_success": as_float(profile_series, "longest_tracked_segment_ms"),
                "observed_reacquire_tracked_hold_ms_success": as_float(profile_series, "reacquire_tracked_hold_ms"),
                "next_capture_target": build_next_target(scenario, gap_rows, gap_hold),
                "research_role": build_research_role(scenario),
                "operator_note": build_operator_note(scenario, success_rate_value, gap_rows, gap_hold),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["priority_rank", "scenario"], kind="mergesort")
    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    out_md.write_text(build_markdown(out_df), encoding="utf-8")
    print(
        {
            "output_csv": str(out_csv),
            "output_md": str(out_md),
            "num_scenarios": int(len(out_df)),
        }
    )


if __name__ == "__main__":
    main()
