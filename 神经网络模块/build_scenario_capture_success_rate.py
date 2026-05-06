#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build capture success-rate dashboard by scenario")
    p.add_argument("--failure_breakdown_csv", required=True)
    p.add_argument("--output_csv", required=True)
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
    df = pd.read_csv(args.failure_breakdown_csv)
    if df.empty:
        raise RuntimeError("failure_breakdown_csv is empty")
    df["scenario_bucket"] = df["scenario"].map(normalize_scenario)

    rows = []
    for scenario in ["stable", "maneuver", "recovery"]:
        seg = df[df["scenario_bucket"] == scenario]
        attempted = int(len(seg))
        successful = int(seg["has_pseudo_valid"].sum()) if attempted else 0
        detector_positive = int(seg["has_detection"].sum()) if attempted else 0
        tracking_positive = int(seg["has_tracking"].sum()) if attempted else 0
        pseudo_valid = int(seg["has_pseudo_valid"].sum()) if attempted else 0
        recovery_hold = int(seg["recovery_meets_hold"].sum()) if attempted else 0
        rows.append(
            {
                "scenario": scenario,
                "attempted_runs": attempted,
                "successful_runs": successful,
                "success_rate": float(successful / attempted) if attempted > 0 else 0.0,
                "detector_positive_runs": detector_positive,
                "tracking_positive_runs": tracking_positive,
                "pseudo_valid_runs": pseudo_valid,
                "recovery_meets_hold_runs": recovery_hold,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print({"output_csv": str(out_path), "rows": int(len(out_df))})


if __name__ == "__main__":
    main()
