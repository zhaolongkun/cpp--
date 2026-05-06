#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch pseudo-expert augmentation for full-profile logs")
    p.add_argument("--input_dir", required=True, help="Directory containing full-profile tracker logs")
    p.add_argument("--output_dir", required=True, help="Directory to store augmented CSVs")
    p.add_argument("--summary_csv", required=True, help="Summary CSV path")
    p.add_argument("--pattern", default="tracker_log*.csv", help="Glob pattern for input logs")
    p.add_argument(
        "--variants",
        default="future_smoothed_base,future_error_aware",
        help="Comma-separated pseudo-expert variants to build",
    )
    p.add_argument("--future_horizon", type=int, default=5)
    p.add_argument("--min_future_len", type=int, default=3)
    p.add_argument("--dedup_by_frame_id", action="store_true")
    return p.parse_args()


def infer_scenario(name: str) -> str:
    s = name.lower()
    if "recovery" in s or "loss" in s:
        return "loss"
    if "maneuver" in s:
        return "maneuver"
    return "normal"


def is_nonempty_csv(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def run_builder(script: Path, input_csv: Path, output_csv: Path, args: argparse.Namespace, variant: str) -> None:
    cmd = [
        sys.executable,
        str(script),
        "--input_csv",
        str(input_csv),
        "--output_csv",
        str(output_csv),
        "--future_horizon",
        str(args.future_horizon),
        "--min_future_len",
        str(args.min_future_len),
        "--variant",
        variant,
    ]
    if args.dedup_by_frame_id:
        cmd.append("--dedup_by_frame_id")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    summary_csv = Path(args.summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "build_pseudo_expert_from_full_log.py"
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants:
        raise ValueError("variants must not be empty")

    rows: List[Dict[str, object]] = []
    for csv_path in sorted(input_dir.glob(args.pattern)):
        if not is_nonempty_csv(csv_path):
            continue
        for variant in variants:
            out_name = f"{csv_path.stem}.{variant}.aug.csv"
            out_csv = output_dir / out_name
            try:
                run_builder(script, csv_path, out_csv, args, variant)
                df = pd.read_csv(out_csv)
                pseudo_valid = int((df["pseudo_expert_valid"] == 1).sum()) if "pseudo_expert_valid" in df.columns else 0
                valid_ratio = float(pseudo_valid / len(df)) if len(df) > 0 else 0.0
                run_id = str(df["run_id"].iloc[0]) if len(df) > 0 and "run_id" in df.columns else ""
                rows.append(
                    {
                        "source_csv": str(csv_path),
                        "augmented_csv": str(out_csv),
                        "variant": variant,
                        "rows": int(len(df)),
                        "pseudo_expert_valid_rows": pseudo_valid,
                        "valid_ratio": valid_ratio,
                        "run_id": run_id,
                        "scenario": infer_scenario(csv_path.name),
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "source_csv": str(csv_path),
                        "augmented_csv": str(out_csv),
                        "variant": variant,
                        "rows": 0,
                        "pseudo_expert_valid_rows": 0,
                        "valid_ratio": 0.0,
                        "run_id": "",
                        "scenario": infer_scenario(csv_path.name),
                        "status": "error",
                        "error": str(e),
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_csv, index=False, encoding="utf-8")
    json_path = summary_csv.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_logs": int(summary["source_csv"].nunique()) if len(summary) > 0 else 0,
                "num_augmented": int(len(summary)),
                "variants": variants,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(
        {
            "summary_csv": str(summary_csv),
            "num_augmented": int(len(summary)),
            "num_ok": int((summary["status"] == "ok").sum()) if len(summary) else 0,
            "num_error": int((summary["status"] == "error").sum()) if len(summary) else 0,
            "variants": variants,
        }
    )


if __name__ == "__main__":
    main()
