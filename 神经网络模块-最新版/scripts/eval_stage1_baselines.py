from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.features import ALL_FEATURE_COLUMNS, FEATURE_INDEX, check_feature_index_alignment
from la_cspc_ornet.stage1_common import (
    build_segmented_pseudo_targets,
    build_window_segment_masks,
    compute_metrics,
    ema,
    ensure_segment_columns,
    prefilter_cfg_from_dict,
    rolling_median,
    teacher_cfg_from_dict,
)


def diff_keep(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    if len(x) > 1:
        out[1:] = x[1:] - x[:-1]
    return out


def moving_average(x: np.ndarray, win: int = 5) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        lo = max(0, i - win + 1)
        out[i] = float(np.mean(x[lo : i + 1]))
    return out


def kalman_cv_baseline(zx: np.ndarray, zy: np.ndarray, dt_ms: np.ndarray, conf: np.ndarray, lost: np.ndarray, q_scale: float = 0.5, r_scale: float = 1.0) -> np.ndarray:
    n = len(zx)
    x = np.array([zx[0], zy[0], 0.0, 0.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64) * 10.0
    out = np.zeros((n, 2), dtype=np.float64)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    I = np.eye(4, dtype=np.float64)
    for i in range(n):
        dt = max(float(dt_ms[i]), 1e-3)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        q = q_scale * np.array([[dt * dt, 0, dt, 0], [0, dt * dt, 0, dt], [dt, 0, 1, 0], [0, dt, 0, 1]], dtype=np.float64)
        x = F @ x
        P = F @ P @ F.T + q
        if lost[i] < 0.5 and conf[i] > 0.0:
            z = np.array([zx[i], zy[i]], dtype=np.float64)
            R = np.eye(2, dtype=np.float64) * max(1e-3, r_scale / max(conf[i], 0.05))
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - K @ H) @ P
        out[i] = x[:2]
    return out


def imm_baseline(zx: np.ndarray, zy: np.ndarray, dt_ms: np.ndarray, conf: np.ndarray, lost: np.ndarray) -> np.ndarray:
    n = len(zx)
    x1 = np.array([zx[0], zy[0], 0.0, 0.0], dtype=np.float64)
    x2 = x1.copy()
    P1 = np.eye(4, dtype=np.float64) * 10.0
    P2 = np.eye(4, dtype=np.float64) * 10.0
    mu = np.array([0.5, 0.5], dtype=np.float64)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    I = np.eye(4, dtype=np.float64)
    out = np.zeros((n, 2), dtype=np.float64)
    p_switch = 0.05
    for i in range(n):
        dt = max(float(dt_ms[i]), 1e-3)
        trans = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]], dtype=np.float64)
        c = trans.T @ mu
        mix_mu = np.zeros((2, 2), dtype=np.float64)
        for j in range(2):
            denom = max(c[j], 1e-9)
            for k in range(2):
                mix_mu[k, j] = trans[k, j] * mu[k] / denom
        states = [x1, x2]
        covs = [P1, P2]
        mixed_states = []
        mixed_covs = []
        for j in range(2):
            xj = mix_mu[0, j] * states[0] + mix_mu[1, j] * states[1]
            Pj = np.zeros((4, 4), dtype=np.float64)
            for k in range(2):
                dx = states[k] - xj
                Pj += mix_mu[k, j] * (covs[k] + np.outer(dx, dx))
            mixed_states.append(xj)
            mixed_covs.append(Pj)
        xs = []
        Ps = []
        likelihoods = []
        for j, q_scale in enumerate((0.5, 4.0)):
            x = mixed_states[j]
            P = mixed_covs[j]
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
            q = q_scale * np.array([[dt * dt, 0, dt, 0], [0, dt * dt, 0, dt], [dt, 0, 1, 0], [0, dt, 0, 1]], dtype=np.float64)
            x = F @ x
            P = F @ P @ F.T + q
            likelihood = 1.0
            if lost[i] < 0.5 and conf[i] > 0.0:
                z = np.array([zx[i], zy[i]], dtype=np.float64)
                R = np.eye(2, dtype=np.float64) * max(1e-3, 1.0 / max(conf[i], 0.05))
                y = z - H @ x
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (I - K @ H) @ P
                likelihood = float(np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y))
            xs.append(x)
            Ps.append(P)
            likelihoods.append(likelihood)
        mu = c * np.asarray(likelihoods)
        mu = mu / max(np.sum(mu), 1e-9)
        x1, x2 = xs
        P1, P2 = Ps
        fused = mu[0] * x1 + mu[1] * x2
        out[i] = fused[:2]
    return out


def build_window_arrays(df: pd.DataFrame, tags: list[str], seq_len: int, stride: int, feature_stats: Dict[str, Dict[str, float]]):
    windows_x = []
    windows_y = []
    preds: Dict[str, list[np.ndarray]] = {
        "raw": [],
        "ema": [],
        "sma": [],
        "median": [],
        "plain_kf": [],
        "plain_imm": [],
        "robust_imm_kf": [],
    }
    feature_names = ALL_FEATURE_COLUMNS
    med = np.asarray([feature_stats[name]["median"] for name in feature_names], dtype=np.float32)
    iqr = np.asarray([feature_stats[name]["iqr"] for name in feature_names], dtype=np.float32)

    for _, g in df[df["segment_tag"].astype(str).isin(tags)].groupby("segment_tag"):
        g = g.sort_values(["timestamp_ms", "frame_id"]).reset_index(drop=True)
        feat = g[feature_names].to_numpy(dtype=np.float32)
        feat = np.clip((feat - med.reshape(1, -1)) / iqr.reshape(1, -1), -8.0, 8.0)
        target = g[["pseudo_clean_dx", "pseudo_clean_dy"]].to_numpy(dtype=np.float32)
        sensor_dx = g["sensor_dx_raw"].to_numpy(dtype=np.float64) if "sensor_dx_raw" in g.columns else g["dx_raw"].to_numpy(dtype=np.float64)
        sensor_dy = g["sensor_dy_raw"].to_numpy(dtype=np.float64) if "sensor_dy_raw" in g.columns else g["dy_raw"].to_numpy(dtype=np.float64)
        raw_xy = np.column_stack([sensor_dx, sensor_dy])
        ema_xy = np.column_stack([ema(sensor_dx, alpha=0.20), ema(sensor_dy, alpha=0.20)])
        sma_xy = np.column_stack([moving_average(sensor_dx, win=5), moving_average(sensor_dy, win=5)])
        med_xy = np.column_stack([rolling_median(sensor_dx, win=5), rolling_median(sensor_dy, win=5)])
        dt = np.maximum(g["dt_ms"].to_numpy(dtype=np.float64), 1e-3)
        conf = np.clip(g["det_conf"].to_numpy(dtype=np.float64), 0.0, 1.0)
        lost = g["lost_flag"].to_numpy(dtype=np.float64)
        kf_xy = kalman_cv_baseline(sensor_dx, sensor_dy, dt, conf, lost)
        imm_xy = imm_baseline(sensor_dx, sensor_dy, dt, conf, lost)
        robust_xy = g[["dx_hat", "dy_hat"]].to_numpy(dtype=np.float64)
        for end in range(seq_len - 1, len(g), stride):
            start = end - seq_len + 1
            windows_x.append(feat[start : end + 1])
            windows_y.append(target[end])
            preds["raw"].append(raw_xy[end])
            preds["ema"].append(ema_xy[end])
            preds["sma"].append(sma_xy[end])
            preds["median"].append(med_xy[end])
            preds["plain_kf"].append(kf_xy[end])
            preds["plain_imm"].append(imm_xy[end])
            preds["robust_imm_kf"].append(robust_xy[end])

    X = np.stack(windows_x).astype(np.float32)
    y = np.stack(windows_y).astype(np.float32)
    pred_arrays = {name: np.stack(values).astype(np.float32) for name, values in preds.items()}
    return X, y, pred_arrays


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate stage1 baselines with the same split and segmentation as the model")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    if not check_feature_index_alignment():
        raise RuntimeError("feature index alignment check failed; aborting stage1 baseline eval")

    meta = json.loads(Path(args.meta_json).read_text(encoding="utf-8"))
    df = pd.read_csv(args.csv)
    df = ensure_segment_columns(df)
    teacher_cfg = teacher_cfg_from_dict(meta.get("teacher"))
    prefilter_cfg = prefilter_cfg_from_dict(meta.get("prefilter"))
    df = build_segmented_pseudo_targets(df, teacher_cfg=teacher_cfg, prefilter_cfg=prefilter_cfg)

    X_test, y_test, preds = build_window_arrays(
        df=df,
        tags=[str(x) for x in meta["test_tags"]],
        seq_len=int(meta["seq_len"]),
        stride=int(meta["stride"]),
        feature_stats=meta["normalization"],
    )
    masks = build_window_segment_masks(X_test)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, pred_xy in preds.items():
        out[name] = {seg_name: compute_metrics(pred_xy, y_test, X_test, seg_mask) for seg_name, seg_mask in masks.items()}
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
