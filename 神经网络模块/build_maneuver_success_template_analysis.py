#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze maneuver success-template stability")
    p.add_argument("--progress_csv", required=True)
    p.add_argument("--profile_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_md", required=True)
    return p.parse_args()


def coefficient_of_variation(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    mean_value = float(values.mean())
    if abs(mean_value) < 1e-12:
        return 0.0
    return float(values.std(ddof=0) / mean_value)


def verdict(success_count: int, det_conf_cv: float, bbox_area_cv: float, tracked_cv: float) -> str:
    if success_count <= 1:
        return (
            "Only one successful maneuver run is currently available; the template is promising but not yet statistically stable "
            "for repeated-coverage claims."
        )
    if max(det_conf_cv, bbox_area_cv, tracked_cv) <= 0.15:
        return "The successful maneuver runs are starting to converge; the current template is suitable for repeated coverage collection."
    return "Successful maneuver capture exists, but the feature spread is still wide; keep collecting before treating the template as stable."


def build_markdown(runs_df: pd.DataFrame, summary: Dict[str, object]) -> str:
    lines: List[str] = [
        "# Maneuver Success Template Analysis",
        "",
        f"- Successful maneuver runs: `{summary['success_run_count']}`",
        f"- Verdict: {summary['template_stability_verdict']}",
        f"- Valid-row spread: mean `{summary['valid_rows_mean']:.1f}`, min `{summary['valid_rows_min']}`, max `{summary['valid_rows_max']}`",
        f"- Longest tracked spread: mean `{summary['longest_tracked_segment_ms_mean']:.1f}` ms, min `{summary['longest_tracked_segment_ms_min']:.1f}` ms, max `{summary['longest_tracked_segment_ms_max']:.1f}` ms",
        f"- CV(det_conf_mean) `{summary['det_conf_mean_cv']:.3f}`, CV(bbox_area_mean) `{summary['bbox_area_mean_cv']:.3f}`, CV(longest_tracked_segment_ms) `{summary['longest_tracked_segment_ms_cv']:.3f}`",
        "",
        "## Per-Run Rows",
        "",
    ]
    if runs_df.empty:
        lines.append("- No successful maneuver runs are available.")
        return "\n".join(lines)
    for _, row in runs_df.iterrows():
        lines.extend(
            [
                f"- `{row['run_id']}`: valid rows `{int(row['pseudo_valid_rows'])}`, usable sequences `{int(row['usable_sequences'])}`, longest tracked `{float(row['longest_tracked_segment_ms']):.1f}` ms, det_conf_mean `{float(row['det_conf_mean']):.3f}`, bbox_area_mean `{float(row['bbox_area_mean']):.1f}`, center_region_ratio `{float(row['center_region_ratio']):.3f}`",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    progress = pd.read_csv(args.progress_csv)
    profile = pd.read_csv(args.profile_csv)

    successful = progress[
        (progress["scenario"].astype(str) == "maneuver")
        & (pd.to_numeric(progress["pseudo_valid_rows"], errors="coerce").fillna(0) > 0)
    ].copy()
    profile_runs = profile[profile["profile_type"].astype(str) == "run"].copy()
    profile_maneuver = profile_runs[profile_runs["scenario"].astype(str) == "maneuver"].copy()

    merged = successful.merge(
        profile_maneuver,
        on="run_id",
        how="left",
        suffixes=("", "_profile"),
    )
    if "scenario_profile" in merged.columns:
        merged = merged.drop(columns=["scenario_profile"])
    merged = merged.sort_values(["pseudo_valid_rows", "usable_sequences"], ascending=[False, False], kind="mergesort")

    success_count = int(len(merged))
    summary = {
        "success_run_count": success_count,
        "valid_rows_mean": float(pd.to_numeric(merged.get("pseudo_valid_rows", pd.Series(dtype=float)), errors="coerce").mean()) if success_count else 0.0,
        "valid_rows_min": int(pd.to_numeric(merged.get("pseudo_valid_rows", pd.Series(dtype=float)), errors="coerce").min()) if success_count else 0,
        "valid_rows_max": int(pd.to_numeric(merged.get("pseudo_valid_rows", pd.Series(dtype=float)), errors="coerce").max()) if success_count else 0,
        "longest_tracked_segment_ms_mean": float(pd.to_numeric(merged.get("longest_tracked_segment_ms", pd.Series(dtype=float)), errors="coerce").mean()) if success_count else 0.0,
        "longest_tracked_segment_ms_min": float(pd.to_numeric(merged.get("longest_tracked_segment_ms", pd.Series(dtype=float)), errors="coerce").min()) if success_count else 0.0,
        "longest_tracked_segment_ms_max": float(pd.to_numeric(merged.get("longest_tracked_segment_ms", pd.Series(dtype=float)), errors="coerce").max()) if success_count else 0.0,
        "det_conf_mean_cv": coefficient_of_variation(merged.get("det_conf_mean", pd.Series(dtype=float))),
        "bbox_area_mean_cv": coefficient_of_variation(merged.get("bbox_area_mean", pd.Series(dtype=float))),
        "longest_tracked_segment_ms_cv": coefficient_of_variation(merged.get("longest_tracked_segment_ms", pd.Series(dtype=float))),
    }
    summary["template_stability_verdict"] = verdict(
        success_count=success_count,
        det_conf_cv=float(summary["det_conf_mean_cv"]),
        bbox_area_cv=float(summary["bbox_area_mean_cv"]),
        tracked_cv=float(summary["longest_tracked_segment_ms_cv"]),
    )

    out_rows = merged.copy()
    for key, value in summary.items():
        out_rows[key] = value

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_rows.to_csv(out_csv, index=False, encoding="utf-8")
    out_md.write_text(build_markdown(merged, summary), encoding="utf-8")
    print({"output_csv": str(out_csv), "output_md": str(out_md), "success_run_count": success_count})


if __name__ == "__main__":
    main()
