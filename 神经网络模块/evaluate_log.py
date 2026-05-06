import argparse
import csv
import json
import math
import os
from typing import Dict, List

import numpy as np


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _i(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return int(default)


def load_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def sign_flip_rate(values: np.ndarray, t_sec: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    s = np.sign(values)
    flips = 0
    for i in range(1, s.shape[0]):
        if s[i] == 0 or s[i - 1] == 0:
            continue
        if s[i] != s[i - 1]:
            flips += 1
    dt = max(1e-9, float(t_sec[-1] - t_sec[0]))
    return float(flips / dt)


def main() -> None:
    p = argparse.ArgumentParser("Evaluate tracker log for paper metrics")
    p.add_argument("--csv", required=True)
    p.add_argument("--out_json", default="")
    args = p.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        raise RuntimeError("empty csv")

    n = len(rows)
    time_ns = np.array([_i(r, "time_ns", 0) for r in rows], dtype=np.int64)
    t_sec = (time_ns - time_ns[0]).astype(np.float64) * 1e-9
    duration = max(1e-9, float(t_sec[-1]))

    note = [str(r.get("note", "lost")).strip().lower() for r in rows]
    tracked_mask = np.array([1 if x == "tracked" else 0 for x in note], dtype=np.int32)
    coast_mask = np.array([1 if x == "coasting" else 0 for x in note], dtype=np.int32)
    valid_mask = np.array([1 if (tracked_mask[i] or coast_mask[i]) else 0 for i in range(n)], dtype=np.int32)

    x_error = np.array([_f(r, "x_error", 0.0) for r in rows], dtype=np.float64)
    y_error = np.array([_f(r, "y_error", 0.0) for r in rows], dtype=np.float64)
    cmd_x = np.array([_f(r, "cmd_x", 0.0) for r in rows], dtype=np.float64)
    cmd_y = np.array([_f(r, "cmd_y", 0.0) for r in rows], dtype=np.float64)
    is_meas = np.array([_i(r, "is_meas_update", 0) for r in rows], dtype=np.int32)
    meas_age_ms = np.array([_f(r, "meas_age_ms", 0.0) for r in rows], dtype=np.float64)
    pnr_gate_d2 = np.array([_f(r, "pnr_gate_d2", 0.0) for r in rows], dtype=np.float64)
    pnr_alpha_q = np.array([_f(r, "pnr_alpha_q", 1.0) for r in rows], dtype=np.float64)
    pnr_alpha_r = np.array([_f(r, "pnr_alpha_r", 1.0) for r in rows], dtype=np.float64)
    pnr_outlier = np.array([_f(r, "pnr_outlier_prob", 0.0) for r in rows], dtype=np.float64)

    valid_idx = np.where(valid_mask > 0)[0]
    if valid_idx.shape[0] > 0:
        xv = x_error[valid_idx]
        yv = y_error[valid_idx]
        tv = t_sec[valid_idx]
        rmse_x = float(np.sqrt(np.mean(xv * xv)))
        rmse_y = float(np.sqrt(np.mean(yv * yv)))
        mae_x = float(np.mean(np.abs(xv)))
        mae_y = float(np.mean(np.abs(yv)))
        flip_x_hz = sign_flip_rate(xv, tv)
        flip_y_hz = sign_flip_rate(yv, tv)
    else:
        rmse_x = rmse_y = mae_x = mae_y = 0.0
        flip_x_hz = flip_y_hz = 0.0

    if cmd_x.shape[0] > 1:
        dcmd_x = np.diff(cmd_x)
        dcmd_y = np.diff(cmd_y)
        cmd_jitter_x = float(np.std(dcmd_x))
        cmd_jitter_y = float(np.std(dcmd_y))
    else:
        cmd_jitter_x = cmd_jitter_y = 0.0

    metrics = {
        "num_rows": int(n),
        "duration_sec": float(duration),
        "tracked_ratio": float(np.mean(tracked_mask)),
        "valid_track_ratio": float(np.mean(valid_mask)),
        "coasting_ratio": float(np.mean(coast_mask)),
        "rmse_x_px": rmse_x,
        "rmse_y_px": rmse_y,
        "mae_x_px": mae_x,
        "mae_y_px": mae_y,
        "sign_flip_x_hz": flip_x_hz,
        "sign_flip_y_hz": flip_y_hz,
        "mean_abs_cmd_x": float(np.mean(np.abs(cmd_x))),
        "mean_abs_cmd_y": float(np.mean(np.abs(cmd_y))),
        "cmd_jitter_x": cmd_jitter_x,
        "cmd_jitter_y": cmd_jitter_y,
        "meas_update_rate_hz": float(np.sum(is_meas) / duration),
        "mean_meas_age_ms": float(np.mean(meas_age_ms)),
        "p95_meas_age_ms": float(np.percentile(meas_age_ms, 95)),
        "mean_gate_d2": float(np.mean(pnr_gate_d2)),
        "mean_alpha_q": float(np.mean(pnr_alpha_q)),
        "mean_alpha_r": float(np.mean(pnr_alpha_r)),
        "mean_outlier_prob": float(np.mean(pnr_outlier)),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

