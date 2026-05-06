#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantify target-presentation success profile from full-profile logs")
    p.add_argument("--progress_csv", required=True, help="Run-level progress CSV")
    p.add_argument("--output_csv", required=True, help="Output CSV path")
    p.add_argument("--center_margin_x", type=float, default=60.0)
    p.add_argument("--center_margin_y", type=float, default=40.0)
    return p.parse_args()


def resolve(project_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (project_root / p).resolve()


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [
        "timestamp_ms",
        "frame_id",
        "img_w",
        "img_h",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "det_count",
        "track_count",
        "controlled_id",
        "det_conf",
        "lost_flag",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "note" in df.columns:
        df["note"] = df["note"].fillna("").astype(str).str.replace("\x00", "", regex=False).str.strip()
    return df.sort_values(["timestamp_ms", "frame_id"], kind="mergesort").reset_index(drop=True)


def longest_segment_ms(mask: np.ndarray, ts: np.ndarray) -> Tuple[int, float]:
    best_len = 0
    best_ms = 0.0
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            end = i - 1
            length = end - start + 1
            duration = float(ts[end] - ts[start]) if np.isfinite(ts[end]) and np.isfinite(ts[start]) else 0.0
            if length > best_len:
                best_len = length
                best_ms = duration
            start = None
    if start is not None:
        end = len(mask) - 1
        length = end - start + 1
        duration = float(ts[end] - ts[start]) if np.isfinite(ts[end]) and np.isfinite(ts[start]) else 0.0
        if length > best_len:
            best_len = length
            best_ms = duration
    return best_len, best_ms


def detect_reacquire_hold_ms(df: pd.DataFrame) -> float:
    if df.empty or "note" not in df.columns or "timestamp_ms" not in df.columns:
        return 0.0
    notes = df["note"].tolist()
    ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(df)
    i = 0
    best_hold = 0.0
    while i < n:
        if notes[i] not in ("lost", "coasting"):
            i += 1
            continue
        while i < n and notes[i] in ("lost", "coasting"):
            i += 1
        tracked_start = i
        while i < n and notes[i] == "tracked":
            i += 1
        tracked_end = i - 1
        if tracked_start < n and tracked_end >= tracked_start:
            hold = float(ts[tracked_end] - ts[tracked_start]) if np.isfinite(ts[tracked_end]) and np.isfinite(ts[tracked_start]) else 0.0
            best_hold = max(best_hold, hold)
    return best_hold


def summarize_one(df: pd.DataFrame, center_margin_x: float, center_margin_y: float) -> Dict[str, float]:
    rows = len(df)
    det_mask = (
        (df.get("det_count", pd.Series(dtype=float)).fillna(0) > 0)
        | (df.get("det_conf", pd.Series(dtype=float)).fillna(0) > 0)
    ).to_numpy(dtype=bool)
    tracked_mask = (
        (df.get("track_count", pd.Series(dtype=float)).fillna(0) > 0)
        & (df.get("controlled_id", pd.Series(dtype=float)).fillna(-1) != -1)
        & (df.get("lost_flag", pd.Series(dtype=float)).fillna(1) == 0)
    ).to_numpy(dtype=bool)
    ts = pd.to_numeric(df.get("timestamp_ms", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)

    det_conf = pd.to_numeric(df.get("det_conf", pd.Series(dtype=float)), errors="coerce")
    det_conf_pos = det_conf[det_mask]

    bbox_area = None
    center_region_ratio = 0.0
    if all(col in df.columns for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "img_w", "img_h"]):
        w = (df["bbox_x2"] - df["bbox_x1"]).clip(lower=0)
        h = (df["bbox_y2"] - df["bbox_y1"]).clip(lower=0)
        bbox_area = w * h
        cx = 0.5 * (df["bbox_x1"] + df["bbox_x2"])
        cy = 0.5 * (df["bbox_y1"] + df["bbox_y2"])
        center_x = 0.5 * df["img_w"]
        center_y = 0.5 * df["img_h"]
        center_mask = (
            (cx - center_x).abs() <= center_margin_x
        ) & (
            (cy - center_y).abs() <= center_margin_y
        )
        if det_mask.any():
            center_region_ratio = float(center_mask[det_mask].mean())
    else:
        bbox_area = pd.Series(dtype=float)

    bbox_pos = bbox_area[det_mask] if bbox_area is not None and len(bbox_area) == len(df) else pd.Series(dtype=float)
    longest_frames, longest_ms = longest_segment_ms(tracked_mask, ts)

    return {
        "rows": int(rows),
        "det_positive_ratio": float(det_mask.mean()) if rows > 0 else 0.0,
        "det_conf_mean": float(det_conf_pos.mean()) if len(det_conf_pos) > 0 else 0.0,
        "det_conf_p90": float(np.percentile(det_conf_pos, 90)) if len(det_conf_pos) > 0 else 0.0,
        "det_conf_max": float(det_conf_pos.max()) if len(det_conf_pos) > 0 else 0.0,
        "bbox_area_mean": float(bbox_pos.mean()) if len(bbox_pos) > 0 else 0.0,
        "bbox_area_p90": float(np.percentile(bbox_pos, 90)) if len(bbox_pos) > 0 else 0.0,
        "bbox_area_max": float(bbox_pos.max()) if len(bbox_pos) > 0 else 0.0,
        "center_region_ratio": float(center_region_ratio),
        "tracked_ratio": float(tracked_mask.mean()) if rows > 0 else 0.0,
        "longest_tracked_segment_frames": int(longest_frames),
        "longest_tracked_segment_ms": float(longest_ms),
        "reacquire_tracked_hold_ms": float(detect_reacquire_hold_ms(df)),
    }


def main() -> None:
    args = parse_args()
    progress = pd.read_csv(args.progress_csv)
    if progress.empty:
        raise RuntimeError("progress_csv is empty")
    project_root = Path(__file__).resolve().parent.parent

    rows: List[Dict[str, object]] = []
    success_rows: List[Dict[str, float]] = []
    failure_rows: List[Dict[str, float]] = []

    for _, row in progress.iterrows():
        log_path = resolve(project_root, str(row["log_path"]))
        df = load_log(log_path)
        metrics = summarize_one(df, args.center_margin_x, args.center_margin_y)
        success_group = "success" if int(row["pseudo_valid_rows"]) > 0 else "failure"
        out = {
            "profile_type": "run",
            "run_id": str(row["run_id"]),
            "scenario": str(row["scenario"]),
            "success_group": success_group,
            **metrics,
        }
        rows.append(out)
        if success_group == "success":
            success_rows.append(metrics)
        else:
            failure_rows.append(metrics)

    def agg(name: str, bucket: List[Dict[str, float]]) -> None:
        if not bucket:
            return
        keys = list(bucket[0].keys())
        agg_row: Dict[str, object] = {
            "profile_type": "aggregate",
            "run_id": name,
            "scenario": "all",
            "success_group": name,
        }
        for k in keys:
            agg_row[k] = float(np.mean([float(x[k]) for x in bucket]))
        rows.append(agg_row)

    agg("success", success_rows)
    agg("failure", failure_rows)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        {
            "output_csv": str(out_path),
            "num_runs": int((out_df["profile_type"] == "run").sum()),
            "success_runs": int(len(success_rows)),
        }
    )


if __name__ == "__main__":
    main()
