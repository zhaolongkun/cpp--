#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare pseudo-expert supervision variants on augmented CSVs")
    p.add_argument("--summary_csv", required=True, help="Batch summary CSV from build_pseudo_expert_batch.py")
    p.add_argument("--output_csv", required=True, help="Variant comparison CSV output")
    p.add_argument("--seq_len", type=int, default=8, help="Sequence length used by training")
    p.add_argument("--near_zero_eps", type=float, default=0.05, help="Absolute L1 threshold for near-zero label ratio")
    p.add_argument("--outlier_l1_thresh", type=float, default=0.5, help="Absolute L1 threshold for outlier ratio")
    return p.parse_args()


def contiguous_lengths(mask: np.ndarray) -> List[int]:
    lengths: List[int] = []
    current = 0
    for v in mask:
        if v:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def resolve_path(project_root: Path, path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "反无" and parts[1] == "cpp智能控制":
        repo_root = project_root.parent.parent
        return (repo_root / p).resolve()
    return (project_root / p).resolve()


def mean_step_abs(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr))))


def normalize_scenario(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("normal", "stable"):
        return "stable"
    if s == "maneuver":
        return "maneuver"
    if s in ("loss", "recovery"):
        return "recovery"
    return s or "unknown"


def direction_consistency(arr: np.ndarray, eps: float) -> float:
    if arr.size < 2:
        return 0.0
    prev = arr[:-1]
    cur = arr[1:]
    mask = (np.abs(prev) > eps) & (np.abs(cur) > eps)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.sign(prev[mask]) == np.sign(cur[mask])))


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    if summary.empty:
        raise RuntimeError("summary_csv is empty")
    project_root = Path(__file__).resolve().parent.parent

    rows: List[Dict[str, object]] = []
    for variant, group in summary.groupby("variant", sort=True):
        total_rows = 0
        total_valid = 0
        usable_sequences = 0
        dx_vals: List[np.ndarray] = []
        dy_vals: List[np.ndarray] = []
        step_l1_vals: List[float] = []
        scenario_valid = {"stable": 0, "maneuver": 0, "recovery": 0}
        scenario_usable = {"stable": 0, "maneuver": 0, "recovery": 0}
        direction_x_vals: List[float] = []
        direction_y_vals: List[float] = []
        near_zero_count = 0
        outlier_count = 0

        for _, row in group.iterrows():
            aug_path = resolve_path(project_root, str(row["augmented_csv"]))
            df = pd.read_csv(aug_path)
            total_rows += int(len(df))
            valid_mask = (df["pseudo_expert_valid"] == 1).to_numpy(dtype=bool) if "pseudo_expert_valid" in df.columns else np.zeros(len(df), dtype=bool)
            valid_rows = int(valid_mask.sum())
            total_valid += valid_rows
            scenario = normalize_scenario(str(row["scenario"]))
            if scenario in scenario_valid:
                scenario_valid[scenario] += valid_rows
            else:
                scenario_valid[scenario] = valid_rows

            lengths = contiguous_lengths(valid_mask)
            usable = int(sum(max(0, l - args.seq_len + 1) for l in lengths))
            usable_sequences += usable
            scenario_usable[scenario] = scenario_usable.get(scenario, 0) + usable

            if valid_rows > 0:
                dx = df.loc[valid_mask, "delta_cmd_target_x"].to_numpy(dtype=np.float64)
                dy = df.loc[valid_mask, "delta_cmd_target_y"].to_numpy(dtype=np.float64)
                dx_vals.append(dx)
                dy_vals.append(dy)
                step_l1_vals.append(mean_step_abs(np.abs(dx) + np.abs(dy)))
                direction_x_vals.append(direction_consistency(dx, args.near_zero_eps))
                direction_y_vals.append(direction_consistency(dy, args.near_zero_eps))
                l1 = np.abs(dx) + np.abs(dy)
                near_zero_count += int((l1 < args.near_zero_eps).sum())
                outlier_count += int((l1 > args.outlier_l1_thresh).sum())

        dx_all = np.concatenate(dx_vals) if dx_vals else np.zeros(0, dtype=np.float64)
        dy_all = np.concatenate(dy_vals) if dy_vals else np.zeros(0, dtype=np.float64)
        l1_all = np.abs(dx_all) + np.abs(dy_all)

        rows.append(
            {
                "variant": str(variant),
                "num_augmented_files": int(len(group)),
                "total_rows": int(total_rows),
                "pseudo_valid_rows": int(total_valid),
                "valid_ratio_weighted": float(total_valid / total_rows) if total_rows > 0 else 0.0,
                "usable_sequences": int(usable_sequences),
                "delta_abs_mean_x": float(np.mean(np.abs(dx_all))) if dx_all.size else 0.0,
                "delta_abs_mean_y": float(np.mean(np.abs(dy_all))) if dy_all.size else 0.0,
                "delta_abs_mean_l1": float(np.mean(l1_all)) if l1_all.size else 0.0,
                "delta_abs_p95_x": float(np.percentile(np.abs(dx_all), 95)) if dx_all.size else 0.0,
                "delta_abs_p95_y": float(np.percentile(np.abs(dy_all), 95)) if dy_all.size else 0.0,
                "label_smoothness_mean_step_l1": float(np.mean(step_l1_vals)) if step_l1_vals else 0.0,
                "direction_consistency_x": float(np.mean(direction_x_vals)) if direction_x_vals else 0.0,
                "direction_consistency_y": float(np.mean(direction_y_vals)) if direction_y_vals else 0.0,
                "near_zero_ratio": float(near_zero_count / total_valid) if total_valid > 0 else 0.0,
                "outlier_ratio_l1_gt_thresh": float(outlier_count / total_valid) if total_valid > 0 else 0.0,
                "scenario_valid_rows_stable": int(scenario_valid.get("stable", 0)),
                "scenario_valid_rows_maneuver": int(scenario_valid.get("maneuver", 0)),
                "scenario_valid_rows_recovery": int(scenario_valid.get("recovery", 0)),
                "scenario_usable_sequences_stable": int(scenario_usable.get("stable", 0)),
                "scenario_usable_sequences_maneuver": int(scenario_usable.get("maneuver", 0)),
                "scenario_usable_sequences_recovery": int(scenario_usable.get("recovery", 0)),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("variant", kind="mergesort")
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        {
            "output_csv": str(out_path),
            "variants": out_df["variant"].tolist(),
            "usable_sequences": out_df[["variant", "usable_sequences"]].to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    main()
