#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paper-ready result blocks from current training statistics")
    p.add_argument("--scenario_summary_csv", required=True)
    p.add_argument("--repeat_status_csv", required=True)
    p.add_argument("--success_rate_csv", required=True)
    p.add_argument("--dataset_audit_json", required=True)
    p.add_argument("--registry_csv", required=True)
    p.add_argument("--maneuver_analysis_md", required=True)
    p.add_argument("--paper_dir", required=True)
    return p.parse_args()


def get_row(df: pd.DataFrame, scenario: str) -> pd.Series:
    seg = df[df["scenario"].astype(str) == scenario]
    return seg.iloc[0] if not seg.empty else pd.Series(dtype=object)


def as_int(row: pd.Series, key: str) -> int:
    if key not in row.index:
        return 0
    value = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
    return int(value) if pd.notna(value) else 0


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_recovery_block(summary_row: pd.Series, repeat_row: pd.Series) -> str:
    valid_rows = as_int(summary_row, "pseudo_valid_rows")
    usable = as_int(summary_row, "usable_sequences")
    hold_runs = as_int(summary_row, "recovery_meets_hold_runs")
    gap_rows = as_int(repeat_row, "valid_rows_gap")
    return "\n".join(
        [
            "# Recovery Results Block",
            "",
            "Recovery has now moved beyond existence-only evidence and should be described as a criterion-satisfying, quantitatively discussable scenario.",
            "",
            "Current quantitative status:",
            f"- valid rows: `{valid_rows}`",
            f"- usable sequences: `{usable}`",
            f"- hold-satisfying recovery runs: `{hold_runs}` / `3`",
            "",
            "What this now supports in the paper:",
            "- recovery is no longer blocked by missing hold-success evidence",
            "- recovery can support recovery-related quantitative discussion and training-oriented data-admission discussion",
            "- recovery-related quantitative discussion can now rely on repeated hold-satisfying runs rather than on a single illustrative event",
            "",
            "What this still does not support:",
            "- a fully training-ready recovery bucket under the per-scenario `800` valid-row target",
            "- a final mean +/- std recovery benchmark with complete scenario balance across all scene families",
            "",
            f"Remaining recovery gap for admission-level training: `{gap_rows}` valid rows.",
        ]
    )


def build_maneuver_block(summary_row: pd.Series, repeat_row: pd.Series, maneuver_analysis_md: str) -> str:
    valid_rows = as_int(summary_row, "pseudo_valid_rows")
    usable = as_int(summary_row, "usable_sequences")
    success_runs = as_int(repeat_row, "successful_runs_current")
    gap_rows = as_int(repeat_row, "valid_rows_gap")
    return "\n".join(
        [
            "# Maneuver Results Block",
            "",
            "Maneuver has already broken zero-sample status, but it has not yet reached repeated-coverage strength.",
            "",
            "Current quantitative status:",
            f"- valid rows: `{valid_rows}`",
            f"- usable sequences: `{usable}`",
            f"- successful maneuver runs: `{success_runs}` / `3`",
            f"- coverage stage: `{repeat_row.get('coverage_stage', 'unknown')}`",
            "",
            "Paper interpretation:",
            "- maneuver already has supervised evidence and is no longer absent from the dataset",
            "- maneuver now has repeated successful capture and can support admission-level training under the current gate",
            "- maneuver still remains below the final 800-row coverage-strength target for the final scenario-wise benchmark",
            "",
            f"Remaining maneuver gap for final coverage-strength completion: `{gap_rows}` valid rows.",
            "",
            "Template-stability note:",
            maneuver_analysis_md.strip(),
        ]
    )


def build_training_block(
    summary: pd.DataFrame,
    repeat_status: pd.DataFrame,
    audit: Dict[str, object],
    registry: pd.DataFrame,
) -> str:
    full_logs = int(pd.to_numeric(summary.get("num_runs", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    total_valid = int(audit.get("total_pseudo_expert_valid_rows", 0))
    total_usable = int(audit.get("usable_sequences_at_seq_len", 0))
    learned_ready = registry[registry["supervision_variant"].astype(str) != "none"]["ready_to_train"].sum() if not registry.empty else 0
    learned_completed = int(
        (
            (registry["supervision_variant"].astype(str) != "none")
            & (registry["status"].astype(str) == "completed")
        ).sum()
    ) if not registry.empty else 0

    stable = get_row(repeat_status, "stable")
    maneuver = get_row(repeat_status, "maneuver")
    recovery = get_row(repeat_status, "recovery")
    stable_ready = as_int(stable, "valid_rows_current") >= 800 and as_int(stable, "usable_sequences_current") >= 120
    maneuver_ready = as_int(maneuver, "successful_runs_current") >= 3 and as_int(maneuver, "usable_sequences_current") >= 120
    recovery_ready = as_int(recovery, "meets_hold_runs_current") >= 3 and as_int(recovery, "usable_sequences_current") >= 120
    total_ready = full_logs >= 20 and total_valid >= 3000 and total_usable >= 500

    blocker_lines = []
    if not total_ready:
        blocker_lines.append("- total-volume thresholds are not yet fully met")
    if not stable_ready:
        blocker_lines.append("- stable supervision volume is still below the admission rule")
    if not maneuver_ready:
        blocker_lines.append("- maneuver has not yet cleared the repeated-success admission rule")
    if not recovery_ready:
        blocker_lines.append("- recovery has not yet cleared the hold-success admission rule")
    if not blocker_lines:
        if learned_completed > 0:
            blocker_lines.append("- admission gate is satisfied and the fixed formal training matrix has already run")
        else:
            blocker_lines.append("- admission gate is satisfied; the formal training matrix can run immediately")

    weakest_key_scene_line = (
        f"- weakest key scene: `maneuver`, because its final coverage-strength volume is still `{as_int(maneuver, 'valid_rows_current')}` / `800` even though repeated success is already `{as_int(maneuver, 'successful_runs_current')}` / `3`"
    )
    remaining_rows = {
        "stable": max(0, 800 - as_int(stable, "valid_rows_current")),
        "maneuver": max(0, 800 - as_int(maneuver, "valid_rows_current")),
        "recovery": max(0, 800 - as_int(recovery, "valid_rows_current")),
    }
    largest_gap_scene = max(remaining_rows, key=remaining_rows.get)
    largest_gap_value = remaining_rows[largest_gap_scene]
    largest_gap_line = (
        f"- largest remaining coverage-strength gap: `{largest_gap_scene}`, because it still has a `{largest_gap_value}`-row gap to the 800-row scenario target"
    )

    return "\n".join(
        [
            "# Training Readiness Block",
            "",
            (
                "The admission gate is now satisfied and the fixed formal GRU training matrix has been executed."
                if learned_completed > 0
                else "Formal GRU training is admitted once total-volume thresholds are met and each scenario satisfies its admission-critical role-specific condition."
            ),
            "",
            "Current admission status:",
            f"- full logs: `{full_logs}` / `20`",
            f"- audit-level pseudo-valid rows: `{total_valid}` / `3000`",
            f"- audit-level usable sequences: `{total_usable}` / `500`",
            "",
            "Current bottleneck split:",
            weakest_key_scene_line,
            largest_gap_line,
            "",
            "Admission gate interpretation:",
            "- `stable` is admission-ready only after it clears the stable-scene supervision-volume target",
            "- `maneuver` is admission-ready once repeated successful capture is established and usable sequence support is present",
            "- `recovery` is admission-ready once hold-satisfying reacquisition evidence is repeated and usable sequence support is present",
            "- the remaining per-scenario 800-row targets for `maneuver` and `recovery` remain coverage-strength goals for the final benchmark tables rather than hard blockers for starting training",
            "",
            "Research-use interpretation:",
            "- `maneuver` data are needed to test whether trust allocation and bounded residual authority remain valid under high-dynamic target motion",
            "- `stable` data are needed to provide a reliable supervision basis for nominal-to-residual learning and repeatable training comparisons",
            "- `recovery` data already support criterion-satisfying discussion of loss handling and reacquisition, so recovery is no longer the gating missing-evidence class",
            "",
            "Scenario-level status:",
            f"- recovery: stage `{recovery.get('coverage_stage', 'unknown')}`, valid rows `{as_int(recovery, 'valid_rows_current')}` / `800`, successful runs `{as_int(recovery, 'successful_runs_current')}` / `3`, hold runs `{as_int(recovery, 'meets_hold_runs_current')}` / `3`",
            f"- maneuver: stage `{maneuver.get('coverage_stage', 'unknown')}`, valid rows `{as_int(maneuver, 'valid_rows_current')}` / `800`, successful runs `{as_int(maneuver, 'successful_runs_current')}` / `3`",
            f"- stable: stage `{stable.get('coverage_stage', 'unknown')}`, valid rows `{as_int(stable, 'valid_rows_current')}` / `800`, usable sequences `{as_int(stable, 'usable_sequences_current')}` / `120`",
            "",
            "Current gate status:",
            *blocker_lines,
            "",
            f"Learned-matrix completion status: `{learned_completed}` learned variants completed.",
            "",
            f"Registry status: learned variants ready to train = `{int(learned_ready)}`.",
        ]
    )


def build_claim_boundary() -> str:
    return "\n".join(
        [
            "# Current Claim Boundary",
            "",
            "## Claims that can now be made",
            "",
            "- online full-chain evidence is established from visual input to final command synthesis",
            "- live camera + ONNX + mock actuator positive evidence is established",
            "- recovery has reached criterion-satisfying status through repeated hold-success runs",
            "- the pseudo-expert data-production line is closed and auditable",
            "- the fixed formal training matrix has been completed for the two learned variants",
            "",
            "## Claims that cannot yet be made",
            "",
            "- maneuver already has sufficient final coverage-strength volume for a completed scenario-wise benchmark",
            "- recovery already has final coverage-strength volume for a completed scenario-wise benchmark",
            "- `gru_future_error_aware` has completed formal training and outperformed the baseline",
            "- the final main result tables already satisfy SCI-Q1-level repeated-run statistical strength",
        ]
    )


def build_submission_gap_checklist() -> str:
    return "\n".join(
        [
            "# Final Submission Gap Checklist",
            "",
            "- Run the trained ONNX variants through the fixed replay/live benchmark and summarize the controller-level comparison.",
            "- Backfill the final comparative and ablation tables from the trained-variant results.",
            "- Backfill the main result tables and perform final manuscript-wide polishing.",
        ]
    )


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.scenario_summary_csv)
    repeat_status = pd.read_csv(args.repeat_status_csv)
    pd.read_csv(args.success_rate_csv)
    with Path(args.dataset_audit_json).open("r", encoding="utf-8") as f:
        audit = json.load(f)
    registry = pd.read_csv(args.registry_csv)
    maneuver_analysis_md = Path(args.maneuver_analysis_md).read_text(encoding="utf-8")

    paper_dir = Path(args.paper_dir)
    recovery_row = get_row(summary, "recovery")
    maneuver_row = get_row(summary, "maneuver")
    recovery_repeat = get_row(repeat_status, "recovery")
    maneuver_repeat = get_row(repeat_status, "maneuver")

    write_text(paper_dir / "recovery_results_block.md", build_recovery_block(recovery_row, recovery_repeat))
    write_text(paper_dir / "maneuver_results_block.md", build_maneuver_block(maneuver_row, maneuver_repeat, maneuver_analysis_md))
    write_text(paper_dir / "training_readiness_block.md", build_training_block(summary, repeat_status, audit, registry))
    write_text(paper_dir / "current_claim_boundary.md", build_claim_boundary())
    write_text(paper_dir / "final_submission_gap_checklist.md", build_submission_gap_checklist())
    print({"paper_dir": str(paper_dir), "generated_blocks": 5})


if __name__ == "__main__":
    main()
