#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


METRICS = [
    "det_positive_ratio",
    "det_conf_mean",
    "det_conf_p90",
    "det_conf_max",
    "bbox_area_mean",
    "bbox_area_p90",
    "bbox_area_max",
    "center_region_ratio",
    "tracked_ratio",
    "longest_tracked_segment_ms",
    "reacquire_tracked_hold_ms",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build success-vs-failure contrast report for target presentation")
    p.add_argument("--profile_csv", required=True, help="target_presentation_success_profile.csv")
    p.add_argument("--output_csv", required=True, help="Contrast report output CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile = pd.read_csv(args.profile_csv)
    if profile.empty:
        raise RuntimeError("profile_csv is empty")

    runs = profile[profile["profile_type"] == "run"].copy()
    success = runs[runs["success_group"] == "success"].copy()
    failure = runs[runs["success_group"] == "failure"].copy()
    if success.empty or failure.empty:
        raise RuntimeError("Need both success and failure runs to build contrast report")

    rows: List[Dict[str, object]] = []
    for metric in METRICS:
        s = float(pd.to_numeric(success[metric], errors="coerce").mean())
        f = float(pd.to_numeric(failure[metric], errors="coerce").mean())
        abs_gap = s - f
        rel_ratio = (s / f) if abs(f) > 1e-12 else None
        rows.append(
            {
                "metric": metric,
                "success_mean": s,
                "failure_mean": f,
                "absolute_gap": abs_gap,
                "relative_ratio_success_over_failure": rel_ratio,
                "interpretation": "higher_in_success" if abs_gap > 0 else ("lower_in_success" if abs_gap < 0 else "equal"),
            }
        )

    # Add a short overall conclusion row based on the largest relative/absolute differences.
    contrast_df = pd.DataFrame(rows)
    top_abs = contrast_df.loc[contrast_df["absolute_gap"].abs().idxmax(), "metric"]
    ratio_df = contrast_df.dropna(subset=["relative_ratio_success_over_failure"]).copy()
    top_ratio = ratio_df.loc[ratio_df["relative_ratio_success_over_failure"].idxmax(), "metric"] if not ratio_df.empty else ""
    summary_row = {
        "metric": "summary",
        "success_mean": "",
        "failure_mean": "",
        "absolute_gap": "",
        "relative_ratio_success_over_failure": "",
        "interpretation": f"largest_abs_gap={top_abs}; largest_ratio_gap={top_ratio}",
    }
    contrast_df = pd.concat([contrast_df, pd.DataFrame([summary_row])], ignore_index=True)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    contrast_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        {
            "output_csv": str(out_path),
            "num_metrics": len(METRICS),
            "top_abs_gap_metric": top_abs,
            "top_ratio_gap_metric": top_ratio,
        }
    )


if __name__ == "__main__":
    main()
