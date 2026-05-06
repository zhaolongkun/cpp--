from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _sorted_log(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("timestamp_ms", "frame_id") if c in df.columns]
    if keys:
        return df.sort_values(keys).reset_index(drop=True)
    return df.reset_index(drop=True)


def _dt_ms(df: pd.DataFrame) -> np.ndarray:
    if "dt_ms" in df.columns:
        dt = df["dt_ms"].to_numpy(dtype=np.float64)
        dt = np.where(np.isfinite(dt) & (dt > 0.0), dt, np.nan)
        if np.isnan(dt).all():
            return np.full(len(df), 33.0, dtype=np.float64)
        fill = float(np.nanmedian(dt))
        return np.where(np.isnan(dt), fill, dt)
    if "timestamp_ms" in df.columns and len(df) > 1:
        ts = df["timestamp_ms"].to_numpy(dtype=np.float64)
        diff = np.diff(ts, prepend=ts[0])
        diff[0] = np.median(diff[1:]) if len(diff) > 1 else 33.0
        diff = np.where(np.isfinite(diff) & (diff > 0.0), diff, np.nan)
        fill = float(np.nanmedian(diff)) if not np.isnan(diff).all() else 33.0
        return np.where(np.isnan(diff), fill, diff)
    return np.full(len(df), 33.0, dtype=np.float64)


def _active_mask(df: pd.DataFrame, use_model_active_mask: bool) -> np.ndarray:
    if not use_model_active_mask:
        return np.ones(len(df), dtype=bool)
    if "infer_used_model" in df.columns:
        mask = df["infer_used_model"].to_numpy(dtype=np.float64) > 0.5
        if int(mask.sum()) > 0:
            return mask
    if "infer_status" in df.columns:
        mask = df["infer_status"].astype(str).str.lower().eq("ok").to_numpy(dtype=bool)
        if int(mask.sum()) > 0:
            return mask
    return np.ones(len(df), dtype=bool)


def _tracked_mask(df: pd.DataFrame) -> np.ndarray:
    if "track_count" in df.columns:
        tracked = df["track_count"].to_numpy(dtype=np.float64) > 0.5
    else:
        tracked = np.ones(len(df), dtype=bool)
    if "lost_flag" in df.columns:
        tracked &= df["lost_flag"].to_numpy(dtype=np.float64) <= 0.5
    if "note" in df.columns:
        notes = df["note"].astype(str).str.lower()
        tracked &= ~notes.isin(["lost", "coasting"]).to_numpy(dtype=bool)
    return tracked


def _residual_norm(df: pd.DataFrame) -> np.ndarray:
    if "dx_hat" not in df.columns or "dy_hat" not in df.columns:
        raise ValueError("control metrics require dx_hat and dy_hat columns in tracker log")
    dx = df["dx_hat"].to_numpy(dtype=np.float64)
    dy = df["dy_hat"].to_numpy(dtype=np.float64)
    return np.sqrt(dx * dx + dy * dy)


def _window_holds(mask: np.ndarray, dt_ms: np.ndarray, start_idx: int, hold_ms: float) -> bool:
    acc = 0.0
    i = start_idx
    while i < len(mask) and mask[i]:
        acc += float(dt_ms[i])
        if acc >= hold_ms:
            return True
        i += 1
    return False


def _episode_starts(err: np.ndarray, active_tracked: np.ndarray, start_threshold_px: float) -> List[int]:
    starts: List[int] = []
    prev_high = False
    prev_valid = False
    for i, (high, valid) in enumerate(zip(err >= start_threshold_px, active_tracked)):
        current = bool(high and valid)
        if current and (not prev_high or not prev_valid):
            starts.append(i)
        prev_high = current
        prev_valid = bool(valid)
    return starts


def _settling_metrics(
    df: pd.DataFrame,
    active_mask: np.ndarray,
    tracked_mask: np.ndarray,
    err: np.ndarray,
    dt_ms: np.ndarray,
    start_threshold_px: float,
    settle_band_px: float,
    settle_hold_ms: float,
    overshoot_min_initial_px: float,
) -> Dict[str, object]:
    active_tracked = active_mask & tracked_mask
    starts = _episode_starts(err, active_tracked, start_threshold_px)
    settle_times: List[float] = []
    overshoots: List[float] = []
    episodes: List[Dict[str, float | int | None]] = []

    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(df)
        settle_idx = None
        settle_mask = active_tracked & (err <= settle_band_px)
        for i in range(start, end):
            if settle_mask[i] and _window_holds(settle_mask, dt_ms, i, settle_hold_ms):
                settle_idx = i
                break
        if settle_idx is not None:
            settle_ms = float(df.iloc[settle_idx]["timestamp_ms"] - df.iloc[start]["timestamp_ms"]) if "timestamp_ms" in df.columns else float(np.sum(dt_ms[start:settle_idx + 1]))
            settle_times.append(settle_ms)
        else:
            settle_ms = None

        seg_end = settle_idx + 1 if settle_idx is not None else end
        seg = df.iloc[start:seg_end]
        init_dx = float(df.iloc[start]["dx_hat"])
        init_dy = float(df.iloc[start]["dy_hat"])
        peak_ratio = 0.0
        for col, init_v in (("dx_hat", init_dx), ("dy_hat", init_dy)):
            init_abs = abs(init_v)
            if init_abs < overshoot_min_initial_px:
                continue
            series = seg[col].to_numpy(dtype=np.float64)
            opp = series * np.sign(init_v) < 0.0
            if np.any(opp):
                peak_ratio = max(peak_ratio, float(np.max(np.abs(series[opp])) / init_abs))
        overshoots.append(peak_ratio)
        episodes.append(
            {
                "start_frame_id": int(df.iloc[start]["frame_id"]) if "frame_id" in df.columns else int(start),
                "settle_frame_id": int(df.iloc[settle_idx]["frame_id"]) if settle_idx is not None and "frame_id" in df.columns else None,
                "initial_error_px": float(err[start]),
                "settling_time_ms": settle_ms,
                "overshoot_ratio": peak_ratio,
            }
        )

    return {
        "count": len(starts),
        "mean_ms": float(np.mean(settle_times)) if settle_times else None,
        "median_ms": float(np.median(settle_times)) if settle_times else None,
        "p90_ms": float(np.percentile(settle_times, 90)) if settle_times else None,
        "overshoot_ratio_mean": float(np.mean(overshoots)) if overshoots else None,
        "overshoot_ratio_p90": float(np.percentile(overshoots, 90)) if overshoots else None,
        "episodes": episodes,
    }


def _lost_recovery_metrics(
    df: pd.DataFrame,
    tracked_mask: np.ndarray,
    dt_ms: np.ndarray,
    recovery_hold_ms: float,
) -> Dict[str, object]:
    if "timestamp_ms" in df.columns:
        ts = df["timestamp_ms"].to_numpy(dtype=np.float64)
    else:
        ts = np.cumsum(dt_ms)

    lost_state = ~tracked_mask
    starts: List[int] = []
    prev = False
    for i, is_lost in enumerate(lost_state):
        if is_lost and not prev:
            starts.append(i)
        prev = bool(is_lost)

    rec_times: List[float] = []
    events: List[Dict[str, float | int | None]] = []
    for start in starts:
        recovery_idx = None
        for i in range(start + 1, len(df)):
            if tracked_mask[i] and _window_holds(tracked_mask, dt_ms, i, recovery_hold_ms):
                recovery_idx = i
                break
        if recovery_idx is not None:
            rec_ms = float(ts[recovery_idx] - ts[start])
            rec_times.append(rec_ms)
        else:
            rec_ms = None
        events.append(
            {
                "lost_start_frame_id": int(df.iloc[start]["frame_id"]) if "frame_id" in df.columns else int(start),
                "recovery_frame_id": int(df.iloc[recovery_idx]["frame_id"]) if recovery_idx is not None and "frame_id" in df.columns else None,
                "recovery_time_ms": rec_ms,
            }
        )
    return {
        "count": len(starts),
        "mean_ms": float(np.mean(rec_times)) if rec_times else None,
        "median_ms": float(np.median(rec_times)) if rec_times else None,
        "p90_ms": float(np.percentile(rec_times, 90)) if rec_times else None,
        "success_rate": float(len(rec_times) / len(starts)) if starts else None,
        "events": events,
    }


def evaluate_control_metrics(
    df: pd.DataFrame,
    start_threshold_px: float,
    settle_band_px: float,
    settle_hold_ms: float,
    overshoot_min_initial_px: float,
    recovery_hold_ms: float,
    use_model_active_mask: bool,
) -> Dict[str, object]:
    df = _sorted_log(df)
    dt_ms = _dt_ms(df)
    active_mask = _active_mask(df, use_model_active_mask=use_model_active_mask)
    tracked_mask = _tracked_mask(df)
    err = _residual_norm(df)

    active_rows = int(active_mask.sum())
    tracked_active = active_mask & tracked_mask
    active_time_ms = float(np.sum(dt_ms[active_mask]))
    tracked_time_ms = float(np.sum(dt_ms[tracked_active]))

    residual_values = err[tracked_active]
    residual_summary = {
        "mean_px": float(np.mean(residual_values)) if residual_values.size else None,
        "median_px": float(np.median(residual_values)) if residual_values.size else None,
        "p95_px": float(np.percentile(residual_values, 95)) if residual_values.size else None,
        "max_px": float(np.max(residual_values)) if residual_values.size else None,
    }

    settling = _settling_metrics(
        df=df,
        active_mask=active_mask,
        tracked_mask=tracked_mask,
        err=err,
        dt_ms=dt_ms,
        start_threshold_px=start_threshold_px,
        settle_band_px=settle_band_px,
        settle_hold_ms=settle_hold_ms,
        overshoot_min_initial_px=overshoot_min_initial_px,
    )
    recovery = _lost_recovery_metrics(df=df, tracked_mask=tracked_mask, dt_ms=dt_ms, recovery_hold_ms=recovery_hold_ms)

    return {
        "row_count": int(len(df)),
        "active_rows": active_rows,
        "tracked_active_rows": int(tracked_active.sum()),
        "tracking_retention_rate": float(tracked_time_ms / active_time_ms) if active_time_ms > 0.0 else None,
        "post_control_residual": residual_summary,
        "settling_time": {
            "count": settling["count"],
            "mean_ms": settling["mean_ms"],
            "median_ms": settling["median_ms"],
            "p90_ms": settling["p90_ms"],
        },
        "overshoot": {
            "count": settling["count"],
            "ratio_mean": settling["overshoot_ratio_mean"],
            "ratio_p90": settling["overshoot_ratio_p90"],
        },
        "lost_recovery_time": {
            "count": recovery["count"],
            "mean_ms": recovery["mean_ms"],
            "median_ms": recovery["median_ms"],
            "p90_ms": recovery["p90_ms"],
            "success_rate": recovery["success_rate"],
        },
        "definitions": {
            "post_control_residual": "tracked+active frames on image-plane residual norm sqrt(dx_hat^2 + dy_hat^2)",
            "settling_time": "time from disturbance start (error_norm >= start_threshold_px) to first sustained entry within settle_band_px for settle_hold_ms",
            "overshoot": "ratio of opposite-sign residual peak to initial axis error during each disturbance episode",
            "lost_recovery_time": "time from lost/coasting start to first sustained tracked interval for recovery_hold_ms",
            "tracking_retention_rate": "active-time ratio spent in tracked state",
        },
        "thresholds": {
            "start_threshold_px": float(start_threshold_px),
            "settle_band_px": float(settle_band_px),
            "settle_hold_ms": float(settle_hold_ms),
            "overshoot_min_initial_px": float(overshoot_min_initial_px),
            "recovery_hold_ms": float(recovery_hold_ms),
            "use_model_active_mask": bool(use_model_active_mask),
        },
        "episodes": {
            "settling": settling["episodes"],
            "recovery": recovery["events"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate control-side metrics from tracker log CSV")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--start_threshold_px", type=float, default=20.0)
    ap.add_argument("--settle_band_px", type=float, default=10.0)
    ap.add_argument("--settle_hold_ms", type=float, default=300.0)
    ap.add_argument("--overshoot_min_initial_px", type=float, default=15.0)
    ap.add_argument("--recovery_hold_ms", type=float, default=200.0)
    ap.add_argument("--use_model_active_mask", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    payload = evaluate_control_metrics(
        df=df,
        start_threshold_px=args.start_threshold_px,
        settle_band_px=args.settle_band_px,
        settle_hold_ms=args.settle_hold_ms,
        overshoot_min_initial_px=args.overshoot_min_initial_px,
        recovery_hold_ms=args.recovery_hold_ms,
        use_model_active_mask=bool(args.use_model_active_mask),
    )
    payload["input_csv"] = str(Path(args.input_csv))
    if args.label:
        payload["label"] = args.label

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
