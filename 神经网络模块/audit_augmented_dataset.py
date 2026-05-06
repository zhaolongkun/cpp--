#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit batch augmented CSVs before formal training")
    p.add_argument("--summary_csv", required=True, help="Summary CSV from build_pseudo_expert_batch.py")
    p.add_argument("--report_json", required=True, help="Output audit JSON")
    p.add_argument("--report_txt", required=True, help="Output text report")
    p.add_argument("--seq_len", type=int, default=8, help="Sequence length used by training")
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


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    if len(summary) == 0:
        raise RuntimeError("summary_csv is empty")
    project_root = Path(__file__).resolve().parent.parent

    all_deltas_x: List[np.ndarray] = []
    all_deltas_y: List[np.ndarray] = []
    all_future_len: List[np.ndarray] = []
    scenario_counts: Dict[str, int] = {}
    usable_sequence_count = 0
    file_reports: List[Dict[str, object]] = []

    for _, row in summary.iterrows():
        aug_path = resolve_path(project_root, str(row["augmented_csv"]))
        df = pd.read_csv(aug_path)
        valid_mask = (df["pseudo_expert_valid"] == 1).to_numpy(dtype=bool)
        lengths = contiguous_lengths(valid_mask)
        usable = int(sum(max(0, l - args.seq_len + 1) for l in lengths))
        usable_sequence_count += usable
        scenario = str(row["scenario"])
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + int(valid_mask.sum())
        if valid_mask.any():
            all_deltas_x.append(df.loc[valid_mask, "delta_cmd_target_x"].to_numpy(dtype=np.float64))
            all_deltas_y.append(df.loc[valid_mask, "delta_cmd_target_y"].to_numpy(dtype=np.float64))
            all_future_len.append(df.loc[valid_mask, "future_window_len"].to_numpy(dtype=np.int64))

        file_reports.append(
            {
                "augmented_csv": str(aug_path),
                "variant": str(row["variant"]),
                "scenario": scenario,
                "rows": int(len(df)),
                "pseudo_valid_rows": int(valid_mask.sum()),
                "valid_ratio": float(valid_mask.mean()) if len(df) > 0 else 0.0,
                "usable_sequences_at_seq_len": usable,
            }
        )

    dx = np.concatenate(all_deltas_x) if all_deltas_x else np.zeros(0, dtype=np.float64)
    dy = np.concatenate(all_deltas_y) if all_deltas_y else np.zeros(0, dtype=np.float64)
    fw = np.concatenate(all_future_len) if all_future_len else np.zeros(0, dtype=np.int64)

    total_rows = int(summary["rows"].sum())
    total_valid = int(summary["pseudo_expert_valid_rows"].sum())
    report = {
        "num_augmented_files": int(len(summary)),
        "variants": sorted(summary["variant"].unique().tolist()),
        "total_rows": total_rows,
        "total_pseudo_expert_valid_rows": total_valid,
        "global_valid_ratio": float(total_valid / total_rows) if total_rows > 0 else 0.0,
        "scenario_valid_row_counts": scenario_counts,
        "delta_cmd_target_x_abs_mean": float(np.mean(np.abs(dx))) if dx.size else 0.0,
        "delta_cmd_target_y_abs_mean": float(np.mean(np.abs(dy))) if dy.size else 0.0,
        "delta_cmd_target_x_abs_p95": float(np.percentile(np.abs(dx), 95)) if dx.size else 0.0,
        "delta_cmd_target_y_abs_p95": float(np.percentile(np.abs(dy), 95)) if dy.size else 0.0,
        "future_window_len_distribution": {str(int(k)): int(v) for k, v in pd.Series(fw).value_counts().sort_index().items()} if fw.size else {},
        "usable_sequences_at_seq_len": usable_sequence_count,
        "enough_for_formal_gru_training": bool(usable_sequence_count >= 500),
        "file_reports": file_reports,
    }

    report_json = Path(args.report_json)
    report_txt = Path(args.report_txt)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_txt.parent.mkdir(parents=True, exist_ok=True)

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        f"num_augmented_files: {report['num_augmented_files']}",
        f"variants: {', '.join(report['variants'])}",
        f"total_rows: {report['total_rows']}",
        f"total_pseudo_expert_valid_rows: {report['total_pseudo_expert_valid_rows']}",
        f"global_valid_ratio: {report['global_valid_ratio']:.4f}",
        f"usable_sequences_at_seq_len: {report['usable_sequences_at_seq_len']}",
        f"enough_for_formal_gru_training: {report['enough_for_formal_gru_training']}",
        f"scenario_valid_row_counts: {report['scenario_valid_row_counts']}",
        f"delta_abs_mean: x={report['delta_cmd_target_x_abs_mean']:.4f}, y={report['delta_cmd_target_y_abs_mean']:.4f}",
        f"delta_abs_p95: x={report['delta_cmd_target_x_abs_p95']:.4f}, y={report['delta_cmd_target_y_abs_p95']:.4f}",
        f"future_window_len_distribution: {report['future_window_len_distribution']}",
    ]
    with report_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(report)


if __name__ == "__main__":
    main()
