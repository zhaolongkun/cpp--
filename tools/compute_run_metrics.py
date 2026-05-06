#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从多个 tracker_log.csv 计算 run-level 指标。

输出:
    outputs/run_level_metrics.csv

统计口径:
    - RMSE / MAE: 仅在 lost_flag == 0 且 dx_hat/dy_hat 有效时计算
    - TV(cmd): 基于 cmd_sent_x/y 的归一化总变差, 单位近似为 cmd units / s
    - Safety Trigger Ratio: 任一 safety flag 触发的帧占比, 百分数
    - infer_used_model ratio: infer_used_model == 1 的帧占比
    - fallback ratio: fallback_delta_zero == 1 的帧占比
    - meanDelta: abs(delta_cmd_x) + abs(delta_cmd_y) 的均值
    - meanCmdDiff: abs(cmd_sent - cmd_base) 两轴和的均值
    - Inference Latency: infer_used_model == 1 帧上的 infer_latency_ms 均值
    - Control-Cycle Duration:
        优先使用 control_cycle_exec_ms 均值, 否则退化为 dt_ms 的正值均值
"""


import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "run_id",
    "timestamp_ms",
    "frame_id",
    "dt_ms",
    "img_w",
    "img_h",
    "dx_hat",
    "dy_hat",
    "lost_flag",
    "cmd_base_x",
    "cmd_base_y",
    "cmd_sent_x",
    "cmd_sent_y",
    "delta_cmd_x",
    "delta_cmd_y",
    "infer_used_model",
    "fallback_delta_zero",
    "residual_clip_flag",
    "slew_limit_flag",
    "final_sat_flag",
    "infer_status",
]

OPTIONAL_NUMERIC_COLUMNS = [
    "infer_latency_ms",
    "control_cycle_exec_ms",
]

OPTIONAL_EVENT_MANIFEST_COLUMNS = [
    "run_id",
    "scenario",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute run-level metrics from tracker_log.csv files.")
    parser.add_argument(
        "--logs_root",
        type=str,
        required=True,
        help="日志根目录, 脚本会递归搜索 tracker_log.csv。",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="tracker_log*.csv",
        help="日志文件匹配模式, 默认: tracker_log*.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/run_level_metrics.csv",
        help="输出 CSV 路径, 默认: outputs/run_level_metrics.csv",
    )
    parser.add_argument(
        "--event_manifest_csv",
        type=str,
        default="",
        help="可选, event_manifest.csv, 仅用于统计每个 run 的事件数。",
    )
    parser.add_argument(
        "--method_from",
        type=str,
        default="path",
        choices=["path", "csv", "constant"],
        help="方法名来源: path / csv / constant",
    )
    parser.add_argument(
        "--constant_method",
        type=str,
        default="unknown",
        help="当 method_from=constant 时使用。",
    )
    parser.add_argument(
        "--min_valid_err_frames",
        type=int,
        default=10,
        help="误差有效帧数少于该值时, RMSE/MAE 记为 NaN。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细日志。",
    )
    return parser.parse_args()


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def discover_log_files(logs_root: Path, pattern: str) -> List[Path]:
    return sorted(p for p in logs_root.rglob(pattern) if p.is_file())


def load_event_counts(manifest_path: Optional[Path], verbose: bool) -> Dict[str, Dict[str, int]]:
    if manifest_path is None or not manifest_path.exists():
        return {}

    df = pd.read_csv(manifest_path)
    missing = [c for c in OPTIONAL_EVENT_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"event_manifest.csv 缺少必要列: {missing}")

    df["run_id"] = df["run_id"].astype(str)
    df["scenario"] = df["scenario"].astype(str).str.strip().str.lower()
    valid_scenarios = {"normal", "loss", "maneuver"}
    bad = sorted(set(df["scenario"].unique()) - valid_scenarios)
    if bad:
        raise ValueError(f"event_manifest.csv 存在非法 scenario: {bad}")

    result: Dict[str, Dict[str, int]] = {}
    for run_id, group in df.groupby("run_id"):
        result[run_id] = {
            "normal_event_count": int((group["scenario"] == "normal").sum()),
            "loss_event_count": int((group["scenario"] == "loss").sum()),
            "maneuver_event_count": int((group["scenario"] == "maneuver").sum()),
        }

    log(f"[Info] 已加载 event manifest: {manifest_path}", verbose)
    return result


def ensure_required_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少必要列: {missing}")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    numeric_cols = [
        "timestamp_ms",
        "frame_id",
        "dt_ms",
        "img_w",
        "img_h",
        "dx_hat",
        "dy_hat",
        "lost_flag",
        "cmd_base_x",
        "cmd_base_y",
        "cmd_sent_x",
        "cmd_sent_y",
        "delta_cmd_x",
        "delta_cmd_y",
        "infer_used_model",
        "fallback_delta_zero",
        "residual_clip_flag",
        "slew_limit_flag",
        "final_sat_flag",
        "infer_latency_ms",
        "control_cycle_exec_ms",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    sort_cols = ["timestamp_ms"]
    if "frame_id" in out.columns:
        sort_cols.append("frame_id")
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out


def infer_run_id(df: pd.DataFrame, csv_path: Path) -> str:
    if "run_id" in df.columns:
        vals = df["run_id"].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if not vals.empty:
            return vals.iloc[0]
    return csv_path.stem


def infer_method(csv_path: Path, logs_root: Path, df: pd.DataFrame, method_from: str, constant_method: str) -> str:
    if method_from == "constant":
        return constant_method

    if method_from == "csv" and "method" in df.columns:
        vals = df["method"].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if not vals.empty:
            return vals.iloc[0]
        return constant_method

    try:
        rel_parts = csv_path.relative_to(logs_root).parts
        if len(rel_parts) >= 2:
            return rel_parts[0]
    except Exception:
        pass

    if csv_path.parent != logs_root:
        return csv_path.parent.name
    return constant_method


def safe_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return float("nan")
    return float(mask.mean())


def safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def compute_error_norm(df: pd.DataFrame) -> np.ndarray:
    dx = df["dx_hat"].to_numpy(dtype=float)
    dy = df["dy_hat"].to_numpy(dtype=float)
    return np.sqrt(dx * dx + dy * dy)


def compute_tv_normalized(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return float("nan")

    cmd_x = df["cmd_sent_x"].to_numpy(dtype=float)
    cmd_y = df["cmd_sent_y"].to_numpy(dtype=float)
    ts = df["timestamp_ms"].to_numpy(dtype=float)

    valid_curr = np.isfinite(cmd_x[1:]) & np.isfinite(cmd_y[1:])
    valid_prev = np.isfinite(cmd_x[:-1]) & np.isfinite(cmd_y[:-1])
    valid_pair = valid_curr & valid_prev
    if valid_pair.sum() == 0:
        return float("nan")

    dcmd = np.abs(cmd_x[1:] - cmd_x[:-1]) + np.abs(cmd_y[1:] - cmd_y[:-1])
    tv_raw = dcmd[valid_pair].sum()

    t_span_ms = float(ts[-1] - ts[0])
    if not np.isfinite(t_span_ms) or t_span_ms <= 0:
        return float("nan")

    return float(tv_raw / (t_span_ms / 1000.0))


def compute_run_metrics_for_file(
    csv_path: Path,
    logs_root: Path,
    event_counts_map: Dict[str, Dict[str, int]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    raw = pd.read_csv(csv_path)
    ensure_required_columns(raw, csv_path)
    df = prepare_dataframe(raw)

    run_id = infer_run_id(df, csv_path)
    method = infer_method(csv_path, logs_root, df, args.method_from, args.constant_method)

    err = compute_error_norm(df)
    lost = df["lost_flag"].to_numpy(dtype=float)
    valid_err_mask = np.isfinite(err) & np.isfinite(lost) & (lost == 0)
    valid_err_frames = int(valid_err_mask.sum())

    if valid_err_frames >= args.min_valid_err_frames:
        rmse = float(np.sqrt(np.mean(np.square(err[valid_err_mask]))))
        mae = float(np.mean(err[valid_err_mask]))
    else:
        rmse = float("nan")
        mae = float("nan")

    tv_cmd = compute_tv_normalized(df)

    clip_flag = pd.to_numeric(df["residual_clip_flag"], errors="coerce").fillna(0).to_numpy(dtype=float) == 1
    slew_flag = pd.to_numeric(df["slew_limit_flag"], errors="coerce").fillna(0).to_numpy(dtype=float) == 1
    sat_flag = pd.to_numeric(df["final_sat_flag"], errors="coerce").fillna(0).to_numpy(dtype=float) == 1
    safety_any = clip_flag | slew_flag | sat_flag

    infer_used = pd.to_numeric(df["infer_used_model"], errors="coerce").fillna(0).to_numpy(dtype=float) == 1
    fallback = pd.to_numeric(df["fallback_delta_zero"], errors="coerce").fillna(0).to_numpy(dtype=float) == 1

    delta_x = df["delta_cmd_x"].to_numpy(dtype=float)
    delta_y = df["delta_cmd_y"].to_numpy(dtype=float)
    delta_mag = np.abs(delta_x) + np.abs(delta_y)

    cmd_base_x = df["cmd_base_x"].to_numpy(dtype=float)
    cmd_base_y = df["cmd_base_y"].to_numpy(dtype=float)
    cmd_sent_x = df["cmd_sent_x"].to_numpy(dtype=float)
    cmd_sent_y = df["cmd_sent_y"].to_numpy(dtype=float)
    cmd_diff = np.abs(cmd_sent_x - cmd_base_x) + np.abs(cmd_sent_y - cmd_base_y)

    lat = df["infer_latency_ms"].to_numpy(dtype=float)
    lat_mask = infer_used & np.isfinite(lat)
    inference_latency_ms = float(lat[lat_mask].mean()) if lat_mask.sum() > 0 else float("nan")
    inference_latency_source = "infer_latency_ms" if lat_mask.sum() > 0 else "missing"

    cyc = df["control_cycle_exec_ms"].to_numpy(dtype=float)
    cyc_mask = np.isfinite(cyc) & (cyc > 0)
    if cyc_mask.sum() > 0:
        control_cycle_duration_ms = float(cyc[cyc_mask].mean())
        control_cycle_source = "control_cycle_exec_ms"
    else:
        dt = df["dt_ms"].to_numpy(dtype=float)
        dt_mask = np.isfinite(dt) & (dt > 0)
        control_cycle_duration_ms = float(dt[dt_mask].mean()) if dt_mask.sum() > 0 else float("nan")
        control_cycle_source = "dt_ms"

    ts = df["timestamp_ms"].to_numpy(dtype=float)
    frame_ids = df["frame_id"].to_numpy(dtype=float)

    event_counts = event_counts_map.get(
        run_id,
        {"normal_event_count": 0, "loss_event_count": 0, "maneuver_event_count": 0},
    )

    metrics: Dict[str, object] = {
        "run_id": run_id,
        "method": method,
        "log_path": str(csv_path),
        "num_frames": int(len(df)),
        "valid_err_frames": valid_err_frames,
        "rmse": rmse,
        "mae": mae,
        "tv_cmd": tv_cmd,
        "safety_trigger_ratio_pct": float(safe_ratio(safety_any.astype(float)) * 100.0),
        "clip_ratio_pct": float(safe_ratio(clip_flag.astype(float)) * 100.0),
        "slew_ratio_pct": float(safe_ratio(slew_flag.astype(float)) * 100.0),
        "sat_ratio_pct": float(safe_ratio(sat_flag.astype(float)) * 100.0),
        "infer_used_model_ratio": float(safe_ratio(infer_used.astype(float))),
        "fallback_ratio": float(safe_ratio(fallback.astype(float))),
        "mean_delta": float(safe_mean(delta_mag)),
        "mean_cmd_diff": float(safe_mean(cmd_diff)),
        "inference_latency_ms": inference_latency_ms,
        "control_cycle_duration_ms": control_cycle_duration_ms,
        "inference_latency_source": inference_latency_source,
        "control_cycle_source": control_cycle_source,
        "timestamp_start_ms": float(ts[0]) if len(ts) > 0 and np.isfinite(ts[0]) else float("nan"),
        "timestamp_end_ms": float(ts[-1]) if len(ts) > 0 and np.isfinite(ts[-1]) else float("nan"),
        "run_duration_ms": float(ts[-1] - ts[0]) if len(ts) >= 2 and np.isfinite(ts[-1] - ts[0]) else float("nan"),
        "frame_id_min": float(np.nanmin(frame_ids)) if len(frame_ids) > 0 else float("nan"),
        "frame_id_max": float(np.nanmax(frame_ids)) if len(frame_ids) > 0 else float("nan"),
        "normal_event_count": int(event_counts["normal_event_count"]),
        "loss_event_count": int(event_counts["loss_event_count"]),
        "maneuver_event_count": int(event_counts["maneuver_event_count"]),
        "run_valid": int(valid_err_frames >= args.min_valid_err_frames),
    }
    return metrics


def main() -> int:
    args = parse_args()

    logs_root = Path(args.logs_root).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    manifest_path = Path(args.event_manifest_csv).expanduser().resolve() if args.event_manifest_csv else None

    if not logs_root.exists() or not logs_root.is_dir():
        print(f"[Error] logs_root 不存在或不是目录: {logs_root}", file=sys.stderr)
        return 1

    files = discover_log_files(logs_root, args.glob)
    if not files:
        print(f"[Error] 未找到日志文件: root={logs_root}, glob={args.glob}", file=sys.stderr)
        return 1

    event_counts_map = load_event_counts(manifest_path, args.verbose)
    all_metrics: List[Dict[str, object]] = []

    log(f"[Info] 共发现 {len(files)} 个日志文件。", args.verbose)
    for csv_path in files:
        try:
            metrics = compute_run_metrics_for_file(csv_path, logs_root, event_counts_map, args)
            all_metrics.append(metrics)
            log(
                f"[OK] {csv_path.name} -> run_id={metrics['run_id']}, method={metrics['method']}, rmse={metrics['rmse']}, tv={metrics['tv_cmd']}",
                args.verbose,
            )
        except Exception as exc:
            print(f"[Warn] 跳过文件 {csv_path}: {exc}", file=sys.stderr)

    if not all_metrics:
        print("[Error] 没有任何日志成功计算 run-level 指标。", file=sys.stderr)
        return 2

    out_df = pd.DataFrame(all_metrics)
    preferred_order = [
        "run_id",
        "method",
        "log_path",
        "num_frames",
        "valid_err_frames",
        "rmse",
        "mae",
        "tv_cmd",
        "safety_trigger_ratio_pct",
        "clip_ratio_pct",
        "slew_ratio_pct",
        "sat_ratio_pct",
        "infer_used_model_ratio",
        "fallback_ratio",
        "mean_delta",
        "mean_cmd_diff",
        "inference_latency_ms",
        "control_cycle_duration_ms",
        "inference_latency_source",
        "control_cycle_source",
        "timestamp_start_ms",
        "timestamp_end_ms",
        "run_duration_ms",
        "frame_id_min",
        "frame_id_max",
        "normal_event_count",
        "loss_event_count",
        "maneuver_event_count",
        "run_valid",
    ]
    cols = [c for c in preferred_order if c in out_df.columns] + [c for c in out_df.columns if c not in preferred_order]
    out_df = out_df[cols]
    out_df = out_df.sort_values(["method", "run_id"], kind="mergesort").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[Done] run-level metrics saved to: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
