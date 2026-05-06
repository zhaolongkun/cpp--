#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结合 tracker_log.csv 和 event_manifest.csv 计算 event-level 指标。

输出:
    outputs/event_level_metrics.csv

支持事件:
    - normal
    - loss
    - maneuver
"""


import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TRACKER_REQUIRED_COLUMNS = [
    "run_id",
    "timestamp_ms",
    "frame_id",
    "dt_ms",
    "img_w",
    "img_h",
    "dx_hat",
    "dy_hat",
    "lost_flag",
    "cmd_sent_x",
    "cmd_sent_y",
    "residual_clip_flag",
    "slew_limit_flag",
    "final_sat_flag",
]

MANIFEST_REQUIRED_COLUMNS = [
    "run_id",
    "scenario",
    "event_id",
]

MANIFEST_OPTIONAL_COLUMNS = [
    "t0_ms",
    "t1_ms",
    "notes",
]


@dataclass
class RunLog:
    run_id: str
    method: str
    log_path: Path
    df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute event-level metrics from tracker logs and event manifest.")
    parser.add_argument("--logs_root", type=str, required=True, help="日志根目录。")
    parser.add_argument("--event_manifest_csv", type=str, required=True, help="事件清单 CSV。")
    parser.add_argument("--glob", type=str, default="tracker_log*.csv", help="日志匹配模式。")
    parser.add_argument("--output_csv", type=str, default="outputs/event_level_metrics.csv", help="输出路径。")
    parser.add_argument("--method_from", type=str, default="path", choices=["path", "csv", "constant"])
    parser.add_argument("--constant_method", type=str, default="unknown")
    parser.add_argument("--settle_px_abs", type=float, default=10.0, help="settle band 绝对像素下界。")
    parser.add_argument("--settle_px_rel", type=float, default=0.01, help="settle band 相对比例。")
    parser.add_argument("--hold_ms", type=float, default=200.0, help="稳态保持时间。")
    parser.add_argument("--recovery_max_ms", type=float, default=2000.0, help="loss 事件最大恢复窗口。")
    parser.add_argument("--min_loss_frames", type=int, default=2, help="loss 事件最小丢失帧数。")
    parser.add_argument("--min_loss_duration_ms", type=float, default=60.0, help="loss 事件最短持续时间。")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def discover_log_files(logs_root: Path, pattern: str) -> List[Path]:
    return sorted(p for p in logs_root.rglob(pattern) if p.is_file())


def infer_run_id(df: pd.DataFrame, csv_path: Path) -> str:
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


def ensure_tracker_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [c for c in TRACKER_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少必要列: {missing}")


def prepare_tracker_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "timestamp_ms",
        "frame_id",
        "dt_ms",
        "img_w",
        "img_h",
        "dx_hat",
        "dy_hat",
        "lost_flag",
        "cmd_sent_x",
        "cmd_sent_y",
        "residual_clip_flag",
        "slew_limit_flag",
        "final_sat_flag",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "run_id" in out.columns:
        out["run_id"] = out["run_id"].astype(str).str.strip()
    out = out.sort_values(["timestamp_ms", "frame_id"], kind="mergesort").reset_index(drop=True)
    return out


def build_run_index(args: argparse.Namespace) -> Dict[str, RunLog]:
    logs_root = Path(args.logs_root).expanduser().resolve()
    files = discover_log_files(logs_root, args.glob)
    if not files:
        raise FileNotFoundError(f"未找到日志文件: root={logs_root}, glob={args.glob}")

    run_index: Dict[str, RunLog] = {}
    for csv_path in files:
        df = pd.read_csv(csv_path)
        ensure_tracker_columns(df, csv_path)
        df = prepare_tracker_df(df)
        run_id = infer_run_id(df, csv_path)
        method = infer_method(csv_path, logs_root, df, args.method_from, args.constant_method)
        if run_id in run_index:
            raise ValueError(f"重复 run_id: {run_id}, 冲突文件: {run_index[run_id].log_path} vs {csv_path}")
        run_index[run_id] = RunLog(run_id=run_id, method=method, log_path=csv_path, df=df)
    return run_index


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"event_manifest.csv 不存在: {manifest_path}")

    df = pd.read_csv(manifest_path)
    missing = [c for c in MANIFEST_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"event_manifest.csv 缺少必要列: {missing}")

    for col in MANIFEST_OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["run_id"] = df["run_id"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip().str.lower()
    df["event_id"] = df["event_id"].astype(str).str.strip()
    df["notes"] = df["notes"].fillna("").astype(str)
    for col in ["t0_ms", "t1_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = {"normal", "loss", "maneuver"}
    bad = sorted(set(df["scenario"].unique()) - valid)
    if bad:
        raise ValueError(f"event_manifest.csv 存在非法 scenario: {bad}")

    return df


def compute_error(df: pd.DataFrame) -> np.ndarray:
    dx = df["dx_hat"].to_numpy(dtype=float)
    dy = df["dy_hat"].to_numpy(dtype=float)
    return np.sqrt(dx * dx + dy * dy)


def subset_by_time(df: pd.DataFrame, t0_ms: float, t1_ms: float) -> pd.DataFrame:
    mask = (df["timestamp_ms"] >= t0_ms) & (df["timestamp_ms"] <= t1_ms)
    return df.loc[mask].copy().reset_index(drop=True)


def valid_err_mask(df: pd.DataFrame) -> np.ndarray:
    err = compute_error(df)
    lost = df["lost_flag"].to_numpy(dtype=float)
    return np.isfinite(err) & np.isfinite(lost) & (lost == 0)


def compute_rmse_mae(df: pd.DataFrame) -> Tuple[float, float, int]:
    err = compute_error(df)
    mask = valid_err_mask(df)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(np.sqrt(np.mean(np.square(err[mask])))), float(np.mean(err[mask])), n


def compute_tv_normalized(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return float("nan")

    cmd_x = df["cmd_sent_x"].to_numpy(dtype=float)
    cmd_y = df["cmd_sent_y"].to_numpy(dtype=float)
    ts = df["timestamp_ms"].to_numpy(dtype=float)
    valid_pair = (
        np.isfinite(cmd_x[1:])
        & np.isfinite(cmd_y[1:])
        & np.isfinite(cmd_x[:-1])
        & np.isfinite(cmd_y[:-1])
    )
    if valid_pair.sum() == 0:
        return float("nan")

    dcmd = np.abs(cmd_x[1:] - cmd_x[:-1]) + np.abs(cmd_y[1:] - cmd_y[:-1])
    tv_raw = dcmd[valid_pair].sum()
    span_ms = float(ts[-1] - ts[0])
    if not np.isfinite(span_ms) or span_ms <= 0:
        return float("nan")
    return float(tv_raw / (span_ms / 1000.0))


def compute_safety_ratios(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    clip_flag = pd.to_numeric(df["residual_clip_flag"], errors="coerce").fillna(0).to_numpy(dtype=float) == 1
    slew_flag = pd.to_numeric(df["slew_limit_flag"], errors="coerce").fillna(0).to_numpy(dtype=float) == 1
    sat_flag = pd.to_numeric(df["final_sat_flag"], errors="coerce").fillna(0).to_numpy(dtype=float) == 1
    any_flag = clip_flag | slew_flag | sat_flag

    return (
        float(any_flag.mean() * 100.0),
        float(clip_flag.mean() * 100.0),
        float(slew_flag.mean() * 100.0),
        float(sat_flag.mean() * 100.0),
    )


def compute_settle_band_px(df: pd.DataFrame, settle_px_abs: float, settle_px_rel: float) -> float:
    img_w = df["img_w"].to_numpy(dtype=float)
    img_h = df["img_h"].to_numpy(dtype=float)
    valid = np.isfinite(img_w) & np.isfinite(img_h) & (img_w > 0) & (img_h > 0)
    if valid.sum() == 0:
        return float(settle_px_abs)
    min_dim = float(np.median(np.minimum(img_w[valid], img_h[valid])))
    return float(max(settle_px_abs, settle_px_rel * min_dim))


def find_first_stable_time(df: pd.DataFrame, start_ms: float, settle_band_px: float, hold_ms: float) -> float:
    if len(df) == 0:
        return float("nan")

    ts = df["timestamp_ms"].to_numpy(dtype=float)
    err = compute_error(df)
    lost = df["lost_flag"].to_numpy(dtype=float)

    for i in range(len(df)):
        t0 = ts[i]
        if not np.isfinite(t0) or t0 < start_ms:
            continue
        if not (np.isfinite(err[i]) and np.isfinite(lost[i]) and lost[i] == 0 and err[i] <= settle_band_px):
            continue

        t_end = t0 + hold_ms
        mask = (ts >= t0) & (ts <= t_end)
        if mask.sum() == 0:
            continue

        err_seg = err[mask]
        lost_seg = lost[mask]
        if (
            np.all(np.isfinite(err_seg))
            and np.all(np.isfinite(lost_seg))
            and np.all(lost_seg == 0)
            and np.all(err_seg <= settle_band_px)
        ):
            return float(t0)

    return float("nan")


def detect_loss_segment(
    df: pd.DataFrame,
    t0_ms: float,
    min_loss_frames: int,
    min_loss_duration_ms: float,
) -> Tuple[float, float, float]:
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan")

    ts = df["timestamp_ms"].to_numpy(dtype=float)
    lost = df["lost_flag"].to_numpy(dtype=float)
    start_idx_candidates = np.where((ts >= t0_ms) & (lost == 1))[0]
    if start_idx_candidates.size == 0:
        return float("nan"), float("nan"), float("nan")

    i = int(start_idx_candidates[0])
    j = i
    while j + 1 < len(df) and lost[j + 1] == 1:
        j += 1

    n_frames = j - i + 1
    duration_ms = float(ts[j] - ts[i]) if j > i else 0.0
    if n_frames < min_loss_frames and duration_ms < min_loss_duration_ms:
        return float("nan"), float("nan"), float("nan")

    reacquire_idx = j + 1
    while reacquire_idx < len(df) and lost[reacquire_idx] == 1:
        reacquire_idx += 1
    reacquire_start_ms = float(ts[reacquire_idx]) if reacquire_idx < len(df) else float("nan")
    return float(ts[i]), float(ts[j]), reacquire_start_ms


def make_base_event_record(run_log: RunLog, event_row: pd.Series, scenario: str) -> Dict[str, object]:
    return {
        "run_id": run_log.run_id,
        "method": run_log.method,
        "log_path": str(run_log.log_path),
        "scenario": scenario,
        "event_id": str(event_row["event_id"]),
        "notes": str(event_row.get("notes", "")),
        "t0_ms": event_row["t0_ms"],
        "t1_ms": event_row["t1_ms"],
    }


def process_normal_event(run_log: RunLog, event_row: pd.Series, args: argparse.Namespace) -> Dict[str, object]:
    if np.isfinite(event_row["t0_ms"]) and np.isfinite(event_row["t1_ms"]):
        used_t0 = float(event_row["t0_ms"])
        used_t1 = float(event_row["t1_ms"])
        event_df = subset_by_time(run_log.df, used_t0, used_t1)
    else:
        used_t0 = float(run_log.df["timestamp_ms"].iloc[0])
        used_t1 = float(run_log.df["timestamp_ms"].iloc[-1])
        event_df = run_log.df.copy()

    rmse, mae, valid_frames = compute_rmse_mae(event_df)
    tv_cmd = compute_tv_normalized(event_df)
    safety_ratio, clip_ratio, slew_ratio, sat_ratio = compute_safety_ratios(event_df)
    settle_band = compute_settle_band_px(event_df, args.settle_px_abs, args.settle_px_rel)

    err = compute_error(event_df)
    mask = valid_err_mask(event_df)
    overshoot = float(np.nanmax(err[mask])) if mask.sum() > 0 else float("nan")
    stable_time = find_first_stable_time(event_df, used_t0, settle_band, args.hold_ms)
    settling_time_ms = float(stable_time - used_t0) if np.isfinite(stable_time) else float("nan")

    row = make_base_event_record(run_log, event_row, "normal")
    row.update(
        {
            "used_t0_ms": used_t0,
            "used_t1_ms": used_t1,
            "event_duration_ms": float(used_t1 - used_t0),
            "settle_band_px": settle_band,
            "hold_ms": float(args.hold_ms),
            "valid_err_frames": valid_frames,
            "rmse_event": rmse,
            "mae_event": mae,
            "overshoot": overshoot,
            "settling_time_ms": settling_time_ms,
            "recovery_time_ms": float("nan"),
            "recovery_success": np.nan,
            "reacquire_start_ms": float("nan"),
            "tv_cmd_event": tv_cmd,
            "safety_trigger_ratio_pct": safety_ratio,
            "clip_ratio_pct": clip_ratio,
            "slew_ratio_pct": slew_ratio,
            "sat_ratio_pct": sat_ratio,
            "event_valid": int(len(event_df) >= 3),
        }
    )
    return row


def process_maneuver_event(run_log: RunLog, event_row: pd.Series, args: argparse.Namespace) -> Dict[str, object]:
    if not np.isfinite(event_row["t0_ms"]) or not np.isfinite(event_row["t1_ms"]):
        raise ValueError("maneuver 事件必须提供 t0_ms 和 t1_ms")

    used_t0 = float(event_row["t0_ms"])
    used_t1 = float(event_row["t1_ms"])
    event_df = subset_by_time(run_log.df, used_t0, used_t1)

    rmse, mae, valid_frames = compute_rmse_mae(event_df)
    tv_cmd = compute_tv_normalized(event_df)
    safety_ratio, clip_ratio, slew_ratio, sat_ratio = compute_safety_ratios(event_df)
    settle_band = compute_settle_band_px(event_df, args.settle_px_abs, args.settle_px_rel)

    err = compute_error(event_df)
    mask = valid_err_mask(event_df)
    overshoot = float(np.nanmax(err[mask])) if mask.sum() > 0 else float("nan")
    stable_time = find_first_stable_time(event_df, used_t0, settle_band, args.hold_ms)
    settling_time_ms = float(stable_time - used_t0) if np.isfinite(stable_time) else float("nan")

    row = make_base_event_record(run_log, event_row, "maneuver")
    row.update(
        {
            "used_t0_ms": used_t0,
            "used_t1_ms": used_t1,
            "event_duration_ms": float(used_t1 - used_t0),
            "settle_band_px": settle_band,
            "hold_ms": float(args.hold_ms),
            "valid_err_frames": valid_frames,
            "rmse_event": rmse,
            "mae_event": mae,
            "overshoot": overshoot,
            "settling_time_ms": settling_time_ms,
            "recovery_time_ms": float("nan"),
            "recovery_success": np.nan,
            "reacquire_start_ms": float("nan"),
            "tv_cmd_event": tv_cmd,
            "safety_trigger_ratio_pct": safety_ratio,
            "clip_ratio_pct": clip_ratio,
            "slew_ratio_pct": slew_ratio,
            "sat_ratio_pct": sat_ratio,
            "event_valid": int(len(event_df) >= 3),
        }
    )
    return row


def process_loss_event(run_log: RunLog, event_row: pd.Series, args: argparse.Namespace) -> Dict[str, object]:
    if not np.isfinite(event_row["t0_ms"]):
        raise ValueError("loss 事件必须至少提供 t0_ms")

    nominal_t0 = float(event_row["t0_ms"])
    nominal_t1 = float(event_row["t1_ms"]) if np.isfinite(event_row["t1_ms"]) else float(nominal_t0 + args.recovery_max_ms)

    loss_start, loss_end, reacquire_start = detect_loss_segment(
        run_log.df,
        t0_ms=nominal_t0,
        min_loss_frames=args.min_loss_frames,
        min_loss_duration_ms=args.min_loss_duration_ms,
    )
    used_t0 = loss_start if np.isfinite(loss_start) else nominal_t0
    used_t1 = nominal_t1
    event_df = subset_by_time(run_log.df, used_t0, used_t1)

    rmse, mae, valid_frames = compute_rmse_mae(event_df)
    tv_cmd = compute_tv_normalized(event_df)
    safety_ratio, clip_ratio, slew_ratio, sat_ratio = compute_safety_ratios(event_df)
    settle_band = compute_settle_band_px(event_df, args.settle_px_abs, args.settle_px_rel)

    if np.isfinite(reacquire_start):
        post_reacq_df = subset_by_time(event_df, reacquire_start, used_t1)
        err = compute_error(post_reacq_df)
        mask = valid_err_mask(post_reacq_df)
        overshoot = float(np.nanmax(err[mask])) if mask.sum() > 0 else float("nan")

        stable_from_reacq = find_first_stable_time(post_reacq_df, reacquire_start, settle_band, args.hold_ms)
        settling_time_ms = float(stable_from_reacq - reacquire_start) if np.isfinite(stable_from_reacq) else float("nan")

        stable_from_loss = find_first_stable_time(event_df, reacquire_start, settle_band, args.hold_ms)
        recovery_time_ms = float(stable_from_loss - used_t0) if np.isfinite(stable_from_loss) else float("nan")
        recovery_success = int(np.isfinite(recovery_time_ms))
    else:
        overshoot = float("nan")
        settling_time_ms = float("nan")
        recovery_time_ms = float("nan")
        recovery_success = 0

    row = make_base_event_record(run_log, event_row, "loss")
    row.update(
        {
            "used_t0_ms": used_t0,
            "used_t1_ms": used_t1,
            "event_duration_ms": float(used_t1 - used_t0) if np.isfinite(used_t1 - used_t0) else float("nan"),
            "settle_band_px": settle_band,
            "hold_ms": float(args.hold_ms),
            "valid_err_frames": valid_frames,
            "rmse_event": rmse,
            "mae_event": mae,
            "overshoot": overshoot,
            "settling_time_ms": settling_time_ms,
            "recovery_time_ms": recovery_time_ms,
            "recovery_success": recovery_success,
            "reacquire_start_ms": reacquire_start,
            "tv_cmd_event": tv_cmd,
            "safety_trigger_ratio_pct": safety_ratio,
            "clip_ratio_pct": clip_ratio,
            "slew_ratio_pct": slew_ratio,
            "sat_ratio_pct": sat_ratio,
            "event_valid": int(len(event_df) >= 3),
        }
    )
    return row


def main() -> int:
    args = parse_args()
    logs_root = Path(args.logs_root).expanduser().resolve()
    manifest_path = Path(args.event_manifest_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not logs_root.exists() or not logs_root.is_dir():
        print(f"[Error] logs_root 不存在或不是目录: {logs_root}", file=sys.stderr)
        return 1

    try:
        run_index = build_run_index(args)
        manifest_df = load_manifest(manifest_path)
    except Exception as exc:
        print(f"[Error] 输入校验失败: {exc}", file=sys.stderr)
        return 2

    rows: List[Dict[str, object]] = []
    for _, event_row in manifest_df.iterrows():
        run_id = str(event_row["run_id"])
        if run_id not in run_index:
            print(f"[Warn] event_manifest 中的 run_id 未找到对应日志, 跳过: {run_id}", file=sys.stderr)
            continue

        run_log = run_index[run_id]
        scenario = str(event_row["scenario"]).strip().lower()
        try:
            if scenario == "normal":
                row = process_normal_event(run_log, event_row, args)
            elif scenario == "maneuver":
                row = process_maneuver_event(run_log, event_row, args)
            elif scenario == "loss":
                row = process_loss_event(run_log, event_row, args)
            else:
                raise ValueError(f"未知 scenario: {scenario}")

            rows.append(row)
            log(
                f"[OK] run_id={run_id}, scenario={scenario}, event_id={event_row['event_id']}, rmse={row['rmse_event']}, os={row['overshoot']}",
                args.verbose,
            )
        except Exception as exc:
            print(
                f"[Warn] 事件计算失败, run_id={run_id}, scenario={scenario}, event_id={event_row['event_id']}: {exc}",
                file=sys.stderr,
            )

    if not rows:
        print("[Error] 没有任何事件成功计算 event-level 指标。", file=sys.stderr)
        return 3

    out_df = pd.DataFrame(rows)
    preferred_cols = [
        "run_id",
        "method",
        "log_path",
        "scenario",
        "event_id",
        "notes",
        "t0_ms",
        "t1_ms",
        "used_t0_ms",
        "used_t1_ms",
        "event_duration_ms",
        "settle_band_px",
        "hold_ms",
        "valid_err_frames",
        "rmse_event",
        "mae_event",
        "overshoot",
        "settling_time_ms",
        "recovery_time_ms",
        "recovery_success",
        "reacquire_start_ms",
        "tv_cmd_event",
        "safety_trigger_ratio_pct",
        "clip_ratio_pct",
        "slew_ratio_pct",
        "sat_ratio_pct",
        "event_valid",
    ]
    cols = [c for c in preferred_cols if c in out_df.columns] + [c for c in out_df.columns if c not in preferred_cols]
    out_df = out_df[cols]
    out_df = out_df.sort_values(["method", "run_id", "scenario", "event_id"], kind="mergesort").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[Done] event-level metrics saved to: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
