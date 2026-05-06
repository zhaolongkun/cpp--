#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build deployment-oriented pseudo-expert supervision from a full-profile tracker log.

Supported supervision variants:

1) future_smoothed_base
   cmd_expert(t) = weighted average of future baseline commands.

2) future_error_aware
   cmd_expert(t) = hindsight controller applied to weighted future error and velocity:
       u_x = clip(kp_x * (dx_bar + tau_v * vx_bar), -cmd_limit, cmd_limit)
       u_y = clip(kp_y * (dy_bar + tau_v * vy_bar), -cmd_limit, cmd_limit)

Both variants avoid assuming a human teleoperation channel and remain aligned with deployment.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "run_id",
    "timestamp_ms",
    "frame_id",
    "dt_ms",
    "img_w",
    "img_h",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "det_conf",
    "dx_hat",
    "dy_hat",
    "vx_hat",
    "vy_hat",
    "lost_flag",
    "is_meas_update",
    "meas_age_ms",
    "cmd_base_x",
    "cmd_base_y",
    "cmd_sent_x",
    "cmd_sent_y",
    "det_count",
    "track_count",
    "controlled_id",
    "note",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pseudo-expert augmented CSV from full tracker log")
    p.add_argument("--input_csv", required=True, help="Path to full-profile tracker log CSV")
    p.add_argument("--output_csv", required=True, help="Path to output augmented CSV")
    p.add_argument("--future_horizon", type=int, default=5, help="Number of future valid tracked frames")
    p.add_argument("--min_future_len", type=int, default=3, help="Minimum future valid tracked frames")
    p.add_argument(
        "--variant",
        type=str,
        default="future_smoothed_base",
        choices=["future_smoothed_base", "future_error_aware"],
        help="Pseudo-expert supervision variant",
    )
    p.add_argument(
        "--valid_notes",
        type=str,
        default="tracked",
        help="Comma-separated note values accepted as valid tracked state",
    )
    p.add_argument(
        "--dedup_by_frame_id",
        action="store_true",
        help="Keep the last row for each run_id + frame_id before building pseudo expert",
    )
    p.add_argument("--kp_x", type=float, default=0.06, help="Hindsight controller kp for x when variant=future_error_aware")
    p.add_argument("--kp_y", type=float, default=0.03, help="Hindsight controller kp for y when variant=future_error_aware")
    p.add_argument("--tau_v_sec", type=float, default=0.08, help="Velocity lookahead time for future_error_aware")
    p.add_argument("--cmd_limit", type=float, default=80.0, help="Command clamp for future_error_aware")
    return p.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input full log missing required columns: {missing}")


def numeric_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def is_valid_row(row: pd.Series, valid_notes: set) -> bool:
    note = str(row.get("note", "")).strip()
    track_count = row.get("track_count", np.nan)
    controlled_id = row.get("controlled_id", np.nan)
    lost_flag = row.get("lost_flag", np.nan)
    cmd_base_x = row.get("cmd_base_x", np.nan)
    cmd_base_y = row.get("cmd_base_y", np.nan)
    if not (
        np.isfinite(track_count)
        and np.isfinite(controlled_id)
        and np.isfinite(lost_flag)
        and np.isfinite(cmd_base_x)
        and np.isfinite(cmd_base_y)
    ):
        return False
    return (
        int(track_count) > 0
        and int(controlled_id) != -1
        and int(lost_flag) == 0
        and note in valid_notes
    )


def clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def build_for_run(
    run_df: pd.DataFrame,
    future_horizon: int,
    min_future_len: int,
    valid_notes: set,
    variant: str,
    kp_x: float,
    kp_y: float,
    tau_v_sec: float,
    cmd_limit: float,
) -> pd.DataFrame:
    out = run_df.copy()
    n = len(out)

    base_x = out["cmd_base_x"].to_numpy(dtype=np.float64)
    base_y = out["cmd_base_y"].to_numpy(dtype=np.float64)
    dx_hat = out["dx_hat"].to_numpy(dtype=np.float64)
    dy_hat = out["dy_hat"].to_numpy(dtype=np.float64)
    vx_hat = out["vx_hat"].to_numpy(dtype=np.float64)
    vy_hat = out["vy_hat"].to_numpy(dtype=np.float64)
    frame_id = out["frame_id"].to_numpy(dtype=np.int64)
    controlled_id = out["controlled_id"].to_numpy(dtype=np.int64)
    valid_mask = np.array([is_valid_row(out.iloc[i], valid_notes) for i in range(n)], dtype=bool)

    expert_x = base_x.copy()
    expert_y = base_y.copy()
    delta_x = np.zeros(n, dtype=np.float64)
    delta_y = np.zeros(n, dtype=np.float64)
    pseudo_valid = np.zeros(n, dtype=np.int8)
    future_len = np.zeros(n, dtype=np.int32)
    hindsight_source = np.full(n, "invalid_window", dtype=object)

    for i in range(n):
        if not valid_mask[i]:
            continue

        target_id = controlled_id[i]
        prev_frame = frame_id[i]
        future_idx: List[int] = []
        for j in range(i + 1, n):
            if len(future_idx) >= future_horizon:
                break
            if frame_id[j] <= prev_frame:
                continue
            if not valid_mask[j]:
                break
            if controlled_id[j] != target_id:
                break
            future_idx.append(j)
            prev_frame = frame_id[j]

        future_len[i] = len(future_idx)
        if len(future_idx) < min_future_len:
            continue

        weights = np.arange(len(future_idx), 0, -1, dtype=np.float64)
        weights = weights / weights.sum()
        idx = np.asarray(future_idx, dtype=np.int64)

        if variant == "future_smoothed_base":
            expert_x[i] = float(np.sum(weights * base_x[idx]))
            expert_y[i] = float(np.sum(weights * base_y[idx]))
            hindsight_source[i] = f"future_smoothed_base_h{future_horizon}_min{min_future_len}_triangular"
        else:
            dx_bar = float(np.sum(weights * dx_hat[idx]))
            dy_bar = float(np.sum(weights * dy_hat[idx]))
            vx_bar = float(np.sum(weights * vx_hat[idx]))
            vy_bar = float(np.sum(weights * vy_hat[idx]))
            expert_x[i] = clamp(kp_x * (dx_bar + tau_v_sec * vx_bar), -cmd_limit, cmd_limit)
            expert_y[i] = clamp(kp_y * (dy_bar + tau_v_sec * vy_bar), -cmd_limit, cmd_limit)
            hindsight_source[i] = (
                f"future_error_aware_h{future_horizon}_min{min_future_len}_"
                f"kpx{kp_x:.3f}_kpy{kp_y:.3f}_tau{tau_v_sec:.3f}"
            )
        delta_x[i] = expert_x[i] - base_x[i]
        delta_y[i] = expert_y[i] - base_y[i]
        pseudo_valid[i] = 1

    out["cmd_expert_x"] = expert_x
    out["cmd_expert_y"] = expert_y
    out["delta_cmd_target_x"] = delta_x
    out["delta_cmd_target_y"] = delta_y
    out["pseudo_expert_valid"] = pseudo_valid
    out["future_window_len"] = future_len
    out["hindsight_source"] = hindsight_source
    return out


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.is_file():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")
    if args.future_horizon <= 0:
        raise ValueError("future_horizon must be > 0")
    if args.min_future_len <= 0 or args.min_future_len > args.future_horizon:
        raise ValueError("min_future_len must be in [1, future_horizon]")
    if args.cmd_limit <= 0:
        raise ValueError("cmd_limit must be > 0")
    if args.tau_v_sec < 0:
        raise ValueError("tau_v_sec must be >= 0")

    df = pd.read_csv(input_csv)
    validate_columns(df)
    df = df.sort_values(["run_id", "timestamp_ms", "frame_id"], ascending=[True, True, True]).reset_index(drop=True)
    if args.dedup_by_frame_id:
        df = df.drop_duplicates(subset=["run_id", "frame_id"], keep="last").reset_index(drop=True)

    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ("run_id", "note")]
    numeric_inplace(df, numeric_cols)

    valid_notes = {s.strip() for s in args.valid_notes.split(",") if s.strip()}
    if not valid_notes:
        raise ValueError("valid_notes must not be empty")

    runs = []
    for _, run_df in df.groupby("run_id", sort=False):
        runs.append(
            build_for_run(
                run_df.reset_index(drop=True),
                args.future_horizon,
                args.min_future_len,
                valid_notes,
                args.variant,
                args.kp_x,
                args.kp_y,
                args.tau_v_sec,
                args.cmd_limit,
            )
        )
    aug = pd.concat(runs, axis=0, ignore_index=True)
    aug.to_csv(output_csv, index=False, encoding="utf-8")

    print(
        {
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "rows": int(len(aug)),
            "pseudo_expert_valid_rows": int(aug["pseudo_expert_valid"].sum()),
            "dedup_by_frame_id": bool(args.dedup_by_frame_id),
            "variant": args.variant,
            "future_horizon": int(args.future_horizon),
            "min_future_len": int(args.min_future_len),
            "valid_notes": sorted(valid_notes),
        }
    )


if __name__ == "__main__":
    main()
