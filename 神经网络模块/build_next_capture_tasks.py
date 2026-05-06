#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the next three concrete capture tasks")
    p.add_argument("--progress_csv", required=True)
    p.add_argument("--guidance_csv", required=True)
    p.add_argument("--repeat_status_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_md", required=True)
    return p.parse_args()


def as_int(value: object) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(numeric) if pd.notna(numeric) else 0


def min_valid_goal(template_rows: int, floor_value: int) -> int:
    if template_rows <= 0:
        return floor_value
    return max(floor_value, int(round(template_rows * 0.6)))


def build_markdown(df: pd.DataFrame) -> str:
    lines: List[str] = [
        "# Next Three Capture Tasks",
        "",
        "The current stage is no longer zero-sample acquisition, but repeated scenario coverage enhancement for maneuver and recovery.",
        "",
    ]
    for _, row in df.iterrows():
        lines.extend(
            [
                f"## {row['task_id']}",
                "",
                f"- Scenario: `{row['scenario']}`",
                f"- Objective: {row['objective']}",
                f"- Success criteria: {row['success_criteria']}",
                f"- Remaining gap after success: valid rows `{row['remaining_valid_rows_gap_after_success']}`, usable sequences `{row['remaining_usable_sequences_gap_after_success']}`, hold runs `{row['remaining_hold_runs_gap_after_success']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    progress = pd.read_csv(args.progress_csv)
    guidance = pd.read_csv(args.guidance_csv)
    repeat_status = pd.read_csv(args.repeat_status_csv)

    guidance_index = {str(row["scenario"]): row for _, row in guidance.iterrows()}
    repeat_index = {str(row["scenario"]): row for _, row in repeat_status.iterrows()}
    progress_index = {str(row["run_id"]): row for _, row in progress.iterrows()}

    recovery_guidance = guidance_index.get("recovery", pd.Series(dtype=object))
    maneuver_guidance = guidance_index.get("maneuver", pd.Series(dtype=object))
    stable_guidance = guidance_index.get("stable", pd.Series(dtype=object))
    recovery_status = repeat_index.get("recovery", pd.Series(dtype=object))
    maneuver_status = repeat_index.get("maneuver", pd.Series(dtype=object))
    stable_status = repeat_index.get("stable", pd.Series(dtype=object))

    recovery_template_run = str(recovery_guidance.get("template_run_id", ""))
    maneuver_template_run = str(maneuver_guidance.get("template_run_id", ""))
    stable_template_run = str(stable_guidance.get("template_run_id", ""))
    recovery_template_rows = as_int(progress_index.get(recovery_template_run, pd.Series(dtype=object)).get("pseudo_valid_rows", 0))
    maneuver_template_rows = as_int(progress_index.get(maneuver_template_run, pd.Series(dtype=object)).get("pseudo_valid_rows", 0))
    stable_template_rows = as_int(progress_index.get(stable_template_run, pd.Series(dtype=object)).get("pseudo_valid_rows", 0))
    recovery_template_usable = as_int(progress_index.get(recovery_template_run, pd.Series(dtype=object)).get("usable_sequences", 0))
    maneuver_template_usable = as_int(progress_index.get(maneuver_template_run, pd.Series(dtype=object)).get("usable_sequences", 0))
    stable_template_usable = as_int(progress_index.get(stable_template_run, pd.Series(dtype=object)).get("usable_sequences", 0))

    recovery_valid_goal = min_valid_goal(recovery_template_rows, 80)
    recovery_usable_goal = min_valid_goal(recovery_template_usable, 70)
    maneuver_valid_goal = min_valid_goal(maneuver_template_rows, 120)
    maneuver_usable_goal = min_valid_goal(maneuver_template_usable, 100)
    stable_valid_goal = min_valid_goal(stable_template_rows, 40)
    stable_usable_goal = min_valid_goal(stable_template_usable, 30)

    current_recovery_hold = as_int(recovery_status.get("meets_hold_runs_current", 0))
    recovery_hold_target = as_int(recovery_status.get("meets_hold_runs_target", 0))
    current_recovery_gap = as_int(recovery_status.get("valid_rows_gap", 0))
    current_maneuver_gap = as_int(maneuver_status.get("valid_rows_gap", 0))
    current_stable_gap = as_int(stable_status.get("valid_rows_gap", 0))
    current_stable_usable_gap = as_int(stable_status.get("usable_sequences_gap", 0))

    tasks: List[Dict[str, object]] = []
    next_priority = 1

    recovery_hold_gap = max(0, recovery_hold_target - current_recovery_hold)
    for offset in range(min(2, recovery_hold_gap)):
        tasks.append(
            {
                "task_id": f"Task R{offset + 1}",
                "priority_rank": next_priority,
                "scenario": "recovery",
                "objective": "Add one new hold-satisfying recovery run and increase repeated recovery coverage.",
                "success_criteria": (
                    f"`recovery_meets_hold = 1` for the new run, `pseudo_valid_rows >= {recovery_valid_goal}`, "
                    f"`usable_sequences >= {recovery_usable_goal}`."
                ),
                "remaining_valid_rows_gap_after_success": max(0, current_recovery_gap - recovery_valid_goal * (offset + 1)),
                "remaining_usable_sequences_gap_after_success": 0,
                "remaining_hold_runs_gap_after_success": max(0, recovery_hold_gap - (offset + 1)),
            }
        )
        next_priority += 1

    for offset in range(2):
        tasks.append(
            {
                "task_id": f"Task M{offset + 1}",
                "priority_rank": next_priority,
                "scenario": "maneuver",
                "objective": "Add one high-quality maneuver run to strengthen repeated valid coverage under the current success template.",
                "success_criteria": (
                    f"`pseudo_valid_rows >= {maneuver_valid_goal}`, `usable_sequences >= {maneuver_usable_goal}`, "
                    "and the maneuver phase remains in-frame without collapse to prolonged loss."
                ),
                "remaining_valid_rows_gap_after_success": max(0, current_maneuver_gap - maneuver_valid_goal * (offset + 1)),
                "remaining_usable_sequences_gap_after_success": 0,
                "remaining_hold_runs_gap_after_success": 0,
            }
        )
        next_priority += 1

    for offset in range(2):
        tasks.append(
            {
                "task_id": f"Task S{offset + 1}",
                "priority_rank": next_priority,
                "scenario": "stable",
                "objective": "Add one stable-scene supervision run to raise training-admission volume in the stable bucket.",
                "success_criteria": (
                    f"`pseudo_valid_rows >= {stable_valid_goal}`, `usable_sequences >= {stable_usable_goal}`, "
                    "and sustain tracked stable-center data long enough to preserve hindsight-valid windows."
                ),
                "remaining_valid_rows_gap_after_success": max(0, current_stable_gap - stable_valid_goal * (offset + 1)),
                "remaining_usable_sequences_gap_after_success": max(0, current_stable_usable_gap - stable_usable_goal * (offset + 1)),
                "remaining_hold_runs_gap_after_success": 0,
            }
        )
        next_priority += 1

    if len(tasks) < 5:
        tasks.append(
            {
                "task_id": "Task R3",
                "priority_rank": next_priority,
                "scenario": "recovery",
                "objective": "Add one more high-quality recovery run to close the remaining recovery valid-row gap after hold-success is satisfied.",
                "success_criteria": (
                    f"`pseudo_valid_rows >= {recovery_valid_goal}`, `usable_sequences >= {recovery_usable_goal}`, "
                    "with a clean loss-to-reacquire segment preserved."
                ),
                "remaining_valid_rows_gap_after_success": max(0, current_recovery_gap - recovery_valid_goal),
                "remaining_usable_sequences_gap_after_success": 0,
                "remaining_hold_runs_gap_after_success": 0,
            }
        )

    out_df = pd.DataFrame(tasks).sort_values(["priority_rank", "task_id"], kind="mergesort")
    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    out_md.write_text(build_markdown(out_df), encoding="utf-8")
    print({"output_csv": str(out_csv), "output_md": str(out_md), "rows": int(len(out_df))})


if __name__ == "__main__":
    main()
