import csv
import math
from typing import Dict, List, Tuple

import numpy as np


FEATURE_NAMES = [
    "x_error",
    "y_error",
    "dx_hat",
    "dy_hat",
    "cmd_x",
    "cmd_y",
    "bbox_w",
    "bbox_h",
    "bbox_log_area",
    "det_count",
    "track_count",
    "coast_count",
    "is_meas_update",
    "meas_age_ms",
    "pnr_gate_d2",
    "pnr_model_cv",
    "pnr_model_ca",
    "valid_track",
]

TARGET_NAMES = ["bias_x", "bias_y", "alpha_q", "alpha_r", "outlier_prob"]


class Bounds:
    def __init__(
        self,
        alpha_q_min: float = 0.35,
        alpha_q_max: float = 3.00,
        alpha_r_min: float = 0.35,
        alpha_r_max: float = 4.50,
        bias_limit_px: float = 4.0,
        outlier_prob_min: float = 0.05,
        outlier_prob_max: float = 0.95,
    ) -> None:
        self.alpha_q_min = alpha_q_min
        self.alpha_q_max = alpha_q_max
        self.alpha_r_min = alpha_r_min
        self.alpha_r_max = alpha_r_max
        self.bias_limit_px = bias_limit_px
        self.outlier_prob_min = outlier_prob_min
        self.outlier_prob_max = outlier_prob_max

    def to_dict(self) -> Dict[str, float]:
        return {
            "alpha_q_min": float(self.alpha_q_min),
            "alpha_q_max": float(self.alpha_q_max),
            "alpha_r_min": float(self.alpha_r_min),
            "alpha_r_max": float(self.alpha_r_max),
            "bias_limit_px": float(self.bias_limit_px),
            "outlier_prob_min": float(self.outlier_prob_min),
            "outlier_prob_max": float(self.outlier_prob_max),
        }


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _diff_keep_len(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    if x.shape[0] <= 1:
        return out
    out[1:] = x[1:] - x[:-1]
    return out


def _rolling_mad(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = x[lo : i + 1]
        med = np.median(seg)
        out[i] = np.median(np.abs(seg - med))
    return out


def load_tracker_csv(path: str) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n = len(rows)
    if n == 0:
        raise RuntimeError(f"empty csv: {path}")

    time_ns = np.zeros(n, dtype=np.int64)
    x_error = np.zeros(n, dtype=np.float64)
    y_error = np.zeros(n, dtype=np.float64)
    dx_hat = np.zeros(n, dtype=np.float64)
    dy_hat = np.zeros(n, dtype=np.float64)
    cmd_x = np.zeros(n, dtype=np.float64)
    cmd_y = np.zeros(n, dtype=np.float64)
    bbox_w = np.zeros(n, dtype=np.float64)
    bbox_h = np.zeros(n, dtype=np.float64)
    det_count = np.zeros(n, dtype=np.float64)
    track_count = np.zeros(n, dtype=np.float64)
    coast_count = np.zeros(n, dtype=np.float64)
    is_meas_update = np.zeros(n, dtype=np.float64)
    meas_age_ms = np.zeros(n, dtype=np.float64)
    pnr_gate_d2 = np.zeros(n, dtype=np.float64)
    pnr_model_cv = np.zeros(n, dtype=np.float64)
    pnr_model_ca = np.zeros(n, dtype=np.float64)
    valid_track = np.zeros(n, dtype=np.float64)

    for i, row in enumerate(rows):
        time_ns[i] = _i(row, "time_ns", 0)
        x_error[i] = _f(row, "x_error", 0.0)
        y_error[i] = _f(row, "y_error", 0.0)
        dx_hat[i] = _f(row, "dx_hat", 0.0)
        dy_hat[i] = _f(row, "dy_hat", 0.0)
        cmd_x[i] = _f(row, "cmd_x", 0.0)
        cmd_y[i] = _f(row, "cmd_y", 0.0)
        x1 = _f(row, "bbox_x1", 0.0)
        y1 = _f(row, "bbox_y1", 0.0)
        x2 = _f(row, "bbox_x2", 0.0)
        y2 = _f(row, "bbox_y2", 0.0)
        bbox_w[i] = max(0.0, x2 - x1)
        bbox_h[i] = max(0.0, y2 - y1)
        det_count[i] = float(_i(row, "det_count", 0))
        track_count[i] = float(_i(row, "track_count", 0))
        coast_count[i] = float(_i(row, "coast_count", 0))
        is_meas_update[i] = float(_i(row, "is_meas_update", 0))
        meas_age_ms[i] = _f(row, "meas_age_ms", 0.0)
        pnr_gate_d2[i] = _f(row, "pnr_gate_d2", 0.0)
        pnr_model_cv[i] = _f(row, "pnr_model_cv", 0.5)
        pnr_model_ca[i] = _f(row, "pnr_model_ca", 0.5)
        note = str(row.get("note", "lost")).strip().lower()
        valid_track[i] = 1.0 if (note in ("tracked", "coasting") and det_count[i] > 0.0 and track_count[i] > 0.0) else 0.0

    bbox_log_area = np.log1p(np.maximum(1.0, bbox_w * bbox_h))

    features = np.stack(
        [
            x_error,
            y_error,
            dx_hat,
            dy_hat,
            cmd_x,
            cmd_y,
            bbox_w,
            bbox_h,
            bbox_log_area,
            det_count,
            track_count,
            coast_count,
            is_meas_update,
            meas_age_ms,
            pnr_gate_d2,
            pnr_model_cv,
            pnr_model_ca,
            valid_track,
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "time_ns": time_ns,
        "features": features,
        "x_error": x_error,
        "y_error": y_error,
        "valid_track": valid_track,
    }


def build_pseudo_targets(series: Dict[str, np.ndarray], bounds: Bounds, trend_alpha: float = 0.08, mad_window: int = 21) -> np.ndarray:
    x_error = series["x_error"].astype(np.float64)
    y_error = series["y_error"].astype(np.float64)
    valid = series["valid_track"].astype(np.float64)
    n = x_error.shape[0]

    trend_x = np.zeros(n, dtype=np.float64)
    trend_y = np.zeros(n, dtype=np.float64)
    trend_x[0] = x_error[0]
    trend_y[0] = y_error[0]

    for i in range(1, n):
        if valid[i] > 0.5:
            trend_x[i] = trend_alpha * x_error[i] + (1.0 - trend_alpha) * trend_x[i - 1]
            trend_y[i] = trend_alpha * y_error[i] + (1.0 - trend_alpha) * trend_y[i - 1]
        else:
            trend_x[i] = trend_x[i - 1]
            trend_y[i] = trend_y[i - 1]

    resid_x = x_error - trend_x
    resid_y = y_error - trend_y
    resid_norm = np.sqrt(resid_x * resid_x + resid_y * resid_y)
    resid_mad = _rolling_mad(resid_norm, mad_window)
    noise_score = resid_norm / (resid_mad + 1e-3)

    vx = _diff_keep_len(trend_x)
    vy = _diff_keep_len(trend_y)
    ax = _diff_keep_len(vx)
    ay = _diff_keep_len(vy)
    acc_norm = np.sqrt(ax * ax + ay * ay)
    acc_mad = _rolling_mad(acc_norm, mad_window)
    acc_score = acc_norm / (acc_mad + 1e-3)

    bias_x = np.clip(resid_x, -bounds.bias_limit_px, bounds.bias_limit_px)
    bias_y = np.clip(resid_y, -bounds.bias_limit_px, bounds.bias_limit_px)

    alpha_q = np.clip(0.70 + 0.25 * acc_score + 0.30 * (1.0 - valid), bounds.alpha_q_min, bounds.alpha_q_max)
    alpha_r = np.clip(0.75 + 0.35 * noise_score + 1.10 * (1.0 - valid), bounds.alpha_r_min, bounds.alpha_r_max)

    outlier_logits = 1.20 * (noise_score - 1.0) + 2.20 * (1.0 - valid)
    outlier_prob = np.clip(_sigmoid(outlier_logits), bounds.outlier_prob_min, bounds.outlier_prob_max)

    targets = np.stack([bias_x, bias_y, alpha_q, alpha_r, outlier_prob], axis=1).astype(np.float32)
    return targets


def build_sequences(features: np.ndarray, targets: np.ndarray, valid_track: np.ndarray, seq_len: int = 12, min_valid_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    n, f = features.shape
    if n < seq_len:
        raise RuntimeError(f"not enough rows ({n}) for seq_len={seq_len}")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for i in range(seq_len - 1, n):
        lo = i - seq_len + 1
        hi = i + 1
        vr = float(np.mean(valid_track[lo:hi]))
        if vr < min_valid_ratio:
            continue
        xs.append(features[lo:hi, :])
        ys.append(targets[i, :])

    if not xs:
        raise RuntimeError("no sequence samples after filtering; reduce min_valid_ratio")

    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.stack(ys, axis=0).astype(np.float32)
    assert x.shape[1] == seq_len and x.shape[2] == f
    return x, y
