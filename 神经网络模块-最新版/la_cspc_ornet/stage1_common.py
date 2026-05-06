from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import ALL_FEATURE_COLUMNS, FEATURE_INDEX


SEGMENT_TYPES: Tuple[str, ...] = ("stable", "jitter", "zoom", "recover", "turn")


@dataclass(frozen=True)
class Stage1NormalizationStats:
    median: np.ndarray
    iqr: np.ndarray

    def to_json_dict(self, feature_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
        return {
            name: {"median": float(self.median[idx]), "iqr": float(self.iqr[idx])}
            for idx, name in enumerate(feature_names)
        }


@dataclass(frozen=True)
class Stage1TeacherConfig:
    trend_alpha: float = 0.20
    jitter_ema_alpha: float = 0.45
    rolling_median_win: int = 5
    stable_dx_hat_weight: float = 0.55
    stable_trend_weight: float = 0.45
    stable_event_threshold: float = 0.25
    stable_zoom_threshold: float = 0.20
    jitter_raw_weight: float = 0.35
    jitter_dx_hat_weight: float = 0.10
    jitter_ema_weight: float = 0.35
    jitter_median_weight: float = 0.20
    recover_lost_last_good_weight: float = 0.92
    recover_lost_trend_weight: float = 0.08
    recover_target_dx_hat_weight: float = 0.10
    recover_target_ema_weight: float = 0.45
    recover_target_raw_weight: float = 0.45
    recover_ramp_step: float = 0.18
    recover_lost_threshold: float = 0.50
    recover_coast_threshold: float = 0.00
    reliable_dx_hat_weight: float = 0.25
    reliable_ema_weight: float = 0.40
    reliable_median_weight: float = 0.35
    turn_raw_weight: float = 0.65
    turn_dx_hat_weight: float = 0.05
    turn_ema_weight: float = 0.30
    turn_threshold: float = 0.60
    switch_threshold: float = 0.60

    def to_json_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in asdict(self).items():
            if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
                out[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                out[key] = float(value)
            else:
                out[key] = value
        return out


@dataclass(frozen=True)
class Stage1PrefilterConfig:
    enabled: bool = True
    mode: str = "sma"
    window: int = 5

    def to_json_dict(self) -> Dict[str, float | str | bool]:
        return {
            "enabled": bool(self.enabled),
            "mode": str(self.mode),
            "window": int(self.window),
        }


def teacher_cfg_from_dict(cfg: Dict[str, float] | None) -> Stage1TeacherConfig:
    if not cfg:
        return Stage1TeacherConfig()
    defaults = asdict(Stage1TeacherConfig())
    coerced: Dict[str, float] = {}
    for key in list(cfg.keys()):
        if key not in defaults:
            raise ValueError(f"unsupported teacher config key: {key}")
        default_val = defaults[key]
        value = cfg[key]
        if isinstance(default_val, int) and not isinstance(default_val, bool):
            coerced[key] = int(value)
        elif isinstance(default_val, float):
            coerced[key] = float(value)
        else:
            coerced[key] = value
    merged = {**defaults, **coerced}
    return Stage1TeacherConfig(**merged)


def prefilter_cfg_from_dict(cfg: Dict[str, float] | None) -> Stage1PrefilterConfig:
    if not cfg:
        return Stage1PrefilterConfig()
    defaults = asdict(Stage1PrefilterConfig())
    coerced: Dict[str, float | str | bool] = {}
    for key in list(cfg.keys()):
        if key not in defaults:
            raise ValueError(f"unsupported prefilter config key: {key}")
        default_val = defaults[key]
        value = cfg[key]
        if isinstance(default_val, bool):
            coerced[key] = bool(value)
        elif isinstance(default_val, int):
            coerced[key] = int(value)
        elif isinstance(default_val, float):
            coerced[key] = float(value)
        else:
            coerced[key] = str(value)
    merged = {**defaults, **coerced}
    return Stage1PrefilterConfig(**merged)


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def rolling_median(x: np.ndarray, win: int = 5) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        lo = max(0, i - win + 1)
        out[i] = np.median(x[lo : i + 1])
    return out


def moving_average(x: np.ndarray, win: int = 5) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        lo = max(0, i - win + 1)
        out[i] = float(np.mean(x[lo : i + 1]))
    return out


def diff_keep_len(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    if x.shape[0] > 1:
        out[1:] = x[1:] - x[:-1]
    return out


def clip_quantile(x: np.ndarray, q: float = 0.99) -> np.ndarray:
    lim = float(np.quantile(np.abs(x), q))
    lim = max(lim, 1e-6)
    return np.clip(x, -lim, lim)


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(list(weights), dtype=np.float64)
    w = np.clip(w, 1e-6, None)
    return w / np.sum(w)


def parse_segment_tag_from_path(path: Path) -> str:
    stem = path.stem
    match = re.match(r"tracker_log_(.+)_\d{8}_\d{6}$", stem)
    if match:
        return match.group(1)
    return stem.replace("tracker_log_", "", 1)


def parse_segment_type(tag: str) -> str:
    prefix = str(tag).split("_", 1)[0].lower()
    return prefix if prefix in SEGMENT_TYPES else "unknown"


def ensure_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "segment_tag" not in out.columns:
        out["segment_tag"] = out["run_id"].astype(str)
    if "segment_type" not in out.columns:
        out["segment_type"] = out["segment_tag"].astype(str).map(parse_segment_type)
    if "source_file" not in out.columns:
        out["source_file"] = out["segment_tag"].astype(str)
    if "source_origin_file" not in out.columns:
        out["source_origin_file"] = out["source_file"].astype(str)
    if "source_origin_tag" not in out.columns:
        out["source_origin_tag"] = out["segment_tag"].astype(str)
    if "source_group" not in out.columns:
        out["source_group"] = out["source_origin_file"].astype(str)
    return out


def fill_stage1_numeric_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    zero_default_cols = [
        "dx_raw",
        "dy_raw",
        "dx_hat",
        "dy_hat",
        "vx_hat",
        "vy_hat",
        "det_conf",
        "lost_flag",
        "is_meas_update",
        "meas_age_ms",
        "zoom_delta",
        "coast_count",
        "bbox_area_px",
        "dt_ms",
        "img_w",
        "img_h",
    ]
    for col in zero_default_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def build_switch_turn_scores(dx: np.ndarray, dy: np.ndarray, dt_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dt = np.maximum(dt_ms, 1.0)
    vx = diff_keep_len(dx) / dt
    vy = diff_keep_len(dy) / dt
    ax = diff_keep_len(vx) / dt
    ay = diff_keep_len(vy) / dt
    speed = np.sqrt(vx * vx + vy * vy)
    accel = np.sqrt(ax * ax + ay * ay)
    speed_scale = max(np.percentile(speed, 95), 1e-6)
    accel_scale = max(np.percentile(accel, 95), 1e-6)
    flip = np.zeros_like(speed)
    if speed.shape[0] > 1:
        flip_x = (np.sign(vx[1:]) * np.sign(vx[:-1]) < 0).astype(np.float64)
        flip_y = (np.sign(vy[1:]) * np.sign(vy[:-1]) < 0).astype(np.float64)
        flip[1:] = np.maximum(flip_x, flip_y)
    switch = np.clip(0.5 * (speed / speed_scale) + 0.5 * flip, 0.0, 1.0)
    turn = np.clip(accel / accel_scale, 0.0, 1.0)
    return switch, turn


def segment_event_score(df: pd.DataFrame) -> np.ndarray:
    switch = df["switch_score"].to_numpy(dtype=np.float64)
    turn = df["turn_score"].to_numpy(dtype=np.float64)
    zoom = np.clip(np.abs(df["zoom_delta"].to_numpy(dtype=np.float64)) * 4.0, 0.0, 1.0)
    lost = np.clip(df["lost_flag"].to_numpy(dtype=np.float64), 0.0, 1.0)
    coast = np.clip(df["coast_count"].to_numpy(dtype=np.float64) / 3.0, 0.0, 1.0)
    return np.clip(np.maximum.reduce([switch, turn, zoom, lost, coast]), 0.0, 1.0)


def apply_prefilter(x: np.ndarray, cfg: Stage1PrefilterConfig) -> np.ndarray:
    if not cfg.enabled:
        return x.astype(np.float64, copy=True)
    mode = str(cfg.mode).lower()
    if mode == "sma":
        return moving_average(x, win=max(int(cfg.window), 1))
    if mode == "median":
        return rolling_median(x, win=max(int(cfg.window), 1))
    if mode == "none":
        return x.astype(np.float64, copy=True)
    raise ValueError(f"unsupported prefilter mode: {cfg.mode}")


def apply_stage1_feature_transforms(
    df: pd.DataFrame,
    teacher_cfg: Stage1TeacherConfig | None = None,
    prefilter_cfg: Stage1PrefilterConfig | None = None,
) -> pd.DataFrame:
    teacher_cfg = teacher_cfg or Stage1TeacherConfig()
    prefilter_cfg = prefilter_cfg or Stage1PrefilterConfig()
    out = ensure_segment_columns(df)
    out = fill_stage1_numeric_defaults(out)
    out = out.copy()
    sensor_dx = out["dx_raw"].to_numpy(dtype=np.float64)
    sensor_dy = out["dy_raw"].to_numpy(dtype=np.float64)
    out["sensor_dx_raw"] = sensor_dx
    out["sensor_dy_raw"] = sensor_dy
    out["stable_baseline_dx"] = ema(sensor_dx, alpha=teacher_cfg.trend_alpha)
    out["stable_baseline_dy"] = ema(sensor_dy, alpha=teacher_cfg.trend_alpha)
    filtered_dx = apply_prefilter(sensor_dx, prefilter_cfg)
    filtered_dy = apply_prefilter(sensor_dy, prefilter_cfg)
    out["input_dx"] = filtered_dx
    out["input_dy"] = filtered_dy
    out["dx_raw"] = filtered_dx
    out["dy_raw"] = filtered_dy
    out["bbox_area_norm"] = out["bbox_area_px"] / np.maximum(out["img_w"] * out["img_h"], 1.0)
    out["bbox_area_norm"] = np.log1p(np.clip(out["bbox_area_norm"], 0.0, None))
    out["det_conf"] = np.clip(out["det_conf"].to_numpy(dtype=np.float64), 0.0, 1.0)
    out["lost_flag"] = (out["lost_flag"].to_numpy(dtype=np.float64) > 0.5).astype(np.float64)
    out["is_meas_update"] = (out["is_meas_update"].to_numpy(dtype=np.float64) > 0.5).astype(np.float64)
    out["zoom_delta"] = clip_quantile(out["zoom_delta"].to_numpy(dtype=np.float64), q=0.98)
    out["meas_age_ms"] = np.clip(np.log1p(np.maximum(out["meas_age_ms"].to_numpy(dtype=np.float64), 0.0)), 0.0, np.log1p(1000.0))
    dt_ms = np.maximum(out["dt_ms"].to_numpy(dtype=np.float64), 1.0)
    dt_med = max(float(np.median(dt_ms)), 1.0)
    out["dt_ms"] = dt_ms / dt_med
    out["coast_count"] = np.clip(out["coast_count"].to_numpy(dtype=np.float64), 0.0, 10.0) / 10.0

    out["trend_dx"] = ema(filtered_dx, alpha=teacher_cfg.trend_alpha)
    out["trend_dy"] = ema(filtered_dy, alpha=teacher_cfg.trend_alpha)
    out["d1_dx"] = diff_keep_len(out["dx_raw"].to_numpy(dtype=np.float64))
    out["d1_dy"] = diff_keep_len(out["dy_raw"].to_numpy(dtype=np.float64))
    out["d2_dx"] = diff_keep_len(out["d1_dx"].to_numpy(dtype=np.float64))
    out["d2_dy"] = diff_keep_len(out["d1_dy"].to_numpy(dtype=np.float64))
    out["trend_vx"] = diff_keep_len(out["trend_dx"].to_numpy(dtype=np.float64)) / np.maximum(dt_ms, 1.0)
    out["trend_vy"] = diff_keep_len(out["trend_dy"].to_numpy(dtype=np.float64)) / np.maximum(dt_ms, 1.0)
    switch_score, turn_score = build_switch_turn_scores(
        sensor_dx,
        sensor_dy,
        np.maximum(out["dt_ms"].to_numpy(dtype=np.float64), 1e-3),
    )
    out["switch_score"] = switch_score
    out["turn_score"] = turn_score
    out["event_score"] = segment_event_score(out)
    return out


def robust_blend(values: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    arr = np.stack(list(values), axis=0)
    w = normalize_weights(weights)
    lo = np.quantile(arr, 0.20, axis=0)
    hi = np.quantile(arr, 0.80, axis=0)
    trimmed = np.clip(arr, lo, hi)
    return np.tensordot(w, trimmed, axes=(0, 0))


def build_segmented_pseudo_targets(
    df: pd.DataFrame,
    teacher_cfg: Stage1TeacherConfig | None = None,
    prefilter_cfg: Stage1PrefilterConfig | None = None,
) -> pd.DataFrame:
    teacher_cfg = teacher_cfg or Stage1TeacherConfig()
    prefilter_cfg = prefilter_cfg or Stage1PrefilterConfig()
    out = apply_stage1_feature_transforms(df, teacher_cfg=teacher_cfg, prefilter_cfg=prefilter_cfg)
    dx_input = out["dx_raw"].to_numpy(dtype=np.float64)
    dy_input = out["dy_raw"].to_numpy(dtype=np.float64)
    dx_raw = out["sensor_dx_raw"].to_numpy(dtype=np.float64)
    dy_raw = out["sensor_dy_raw"].to_numpy(dtype=np.float64)
    dx_hat = out["dx_hat"].to_numpy(dtype=np.float64)
    dy_hat = out["dy_hat"].to_numpy(dtype=np.float64)
    trend_dx = out["trend_dx"].to_numpy(dtype=np.float64)
    trend_dy = out["trend_dy"].to_numpy(dtype=np.float64)
    ema_short_dx = ema(dx_raw, alpha=teacher_cfg.jitter_ema_alpha)
    ema_short_dy = ema(dy_raw, alpha=teacher_cfg.jitter_ema_alpha)
    med_dx = rolling_median(dx_raw, win=teacher_cfg.rolling_median_win)
    med_dy = rolling_median(dy_raw, win=teacher_cfg.rolling_median_win)
    conf = out["det_conf"].to_numpy(dtype=np.float64)
    lost = out["lost_flag"].to_numpy(dtype=np.float64)
    meas = out["is_meas_update"].to_numpy(dtype=np.float64)
    zoom_mag = np.clip(np.abs(out["zoom_delta"].to_numpy(dtype=np.float64)) * 4.0, 0.0, 1.0)
    turn = out["turn_score"].to_numpy(dtype=np.float64)
    switch = out["switch_score"].to_numpy(dtype=np.float64)
    coast = out["coast_count"].to_numpy(dtype=np.float64)
    event = out["event_score"].to_numpy(dtype=np.float64)

    stable_mask = (event < teacher_cfg.stable_event_threshold) & (lost < teacher_cfg.recover_lost_threshold) & (zoom_mag < teacher_cfg.stable_zoom_threshold)
    recover_mask = (lost >= teacher_cfg.recover_lost_threshold) | (coast > teacher_cfg.recover_coast_threshold)
    turn_mask = (turn >= teacher_cfg.turn_threshold) | (switch >= teacher_cfg.switch_threshold)
    jitter_mask = (~stable_mask) & (~recover_mask) & (~turn_mask)

    teacher_dx = np.zeros_like(dx_raw, dtype=np.float64)
    teacher_dy = np.zeros_like(dy_raw, dtype=np.float64)
    weight = np.zeros_like(dx_raw, dtype=np.float64)

    reliable_dx = robust_blend(
        (dx_hat, ema_short_dx, med_dx),
        (teacher_cfg.reliable_dx_hat_weight, teacher_cfg.reliable_ema_weight, teacher_cfg.reliable_median_weight),
    )
    reliable_dy = robust_blend(
        (dy_hat, ema_short_dy, med_dy),
        (teacher_cfg.reliable_dx_hat_weight, teacher_cfg.reliable_ema_weight, teacher_cfg.reliable_median_weight),
    )
    last_good_dx = float(reliable_dx[0])
    last_good_dy = float(reliable_dy[0])
    recovery_ramp = 0.0

    for i in range(len(out)):
        if stable_mask[i]:
            stable_w = normalize_weights((teacher_cfg.stable_dx_hat_weight, teacher_cfg.stable_trend_weight))
            teacher_dx[i] = stable_w[0] * dx_hat[i] + stable_w[1] * trend_dx[i]
            teacher_dy[i] = stable_w[0] * dy_hat[i] + stable_w[1] * trend_dy[i]
            last_good_dx = teacher_dx[i]
            last_good_dy = teacher_dy[i]
            recovery_ramp = 0.0
            weight[i] = 0.80 + 0.20 * conf[i]
            continue

        if recover_mask[i]:
            if lost[i] >= 0.5:
                recover_lost_w = normalize_weights((teacher_cfg.recover_lost_last_good_weight, teacher_cfg.recover_lost_trend_weight))
                teacher_dx[i] = recover_lost_w[0] * last_good_dx + recover_lost_w[1] * trend_dx[i]
                teacher_dy[i] = recover_lost_w[0] * last_good_dy + recover_lost_w[1] * trend_dy[i]
                recovery_ramp = 0.0
            else:
                recover_lost_w = normalize_weights((teacher_cfg.recover_lost_last_good_weight, teacher_cfg.recover_lost_trend_weight))
                recovery_ramp = min(recovery_ramp + teacher_cfg.recover_ramp_step, 1.0)
                target_dx = robust_blend(
                    (dx_hat[i : i + 1], ema_short_dx[i : i + 1], dx_raw[i : i + 1]),
                    (
                        teacher_cfg.recover_target_dx_hat_weight,
                        teacher_cfg.recover_target_ema_weight,
                        teacher_cfg.recover_target_raw_weight,
                    ),
                )[0]
                target_dy = robust_blend(
                    (dy_hat[i : i + 1], ema_short_dy[i : i + 1], dy_raw[i : i + 1]),
                    (
                        teacher_cfg.recover_target_dx_hat_weight,
                        teacher_cfg.recover_target_ema_weight,
                        teacher_cfg.recover_target_raw_weight,
                    ),
                )[0]
                teacher_dx[i] = (1.0 - recovery_ramp) * (recover_lost_w[0] * last_good_dx + recover_lost_w[1] * trend_dx[i]) + recovery_ramp * target_dx
                teacher_dy[i] = (1.0 - recovery_ramp) * (recover_lost_w[0] * last_good_dy + recover_lost_w[1] * trend_dy[i]) + recovery_ramp * target_dy
                last_good_dx = teacher_dx[i]
                last_good_dy = teacher_dy[i]
            weight[i] = 0.25 + 0.30 * conf[i] + 0.15 * meas[i] + 0.15 * (1.0 - zoom_mag[i]) + 0.15 * event[i]
            continue

        if turn_mask[i]:
            turn_w = normalize_weights((teacher_cfg.turn_raw_weight, teacher_cfg.turn_dx_hat_weight, teacher_cfg.turn_ema_weight))
            teacher_dx[i] = turn_w[0] * dx_raw[i] + turn_w[1] * dx_hat[i] + turn_w[2] * ema_short_dx[i]
            teacher_dy[i] = turn_w[0] * dy_raw[i] + turn_w[1] * dy_hat[i] + turn_w[2] * ema_short_dy[i]
            last_good_dx = teacher_dx[i]
            last_good_dy = teacher_dy[i]
            weight[i] = 0.50 + 0.25 * conf[i] + 0.25 * event[i]
            continue

        if jitter_mask[i]:
            teacher_dx[i] = robust_blend(
                (dx_raw[i : i + 1], dx_hat[i : i + 1], ema_short_dx[i : i + 1], med_dx[i : i + 1]),
                (
                    teacher_cfg.jitter_raw_weight,
                    teacher_cfg.jitter_dx_hat_weight,
                    teacher_cfg.jitter_ema_weight,
                    teacher_cfg.jitter_median_weight,
                ),
            )[0]
            teacher_dy[i] = robust_blend(
                (dy_raw[i : i + 1], dy_hat[i : i + 1], ema_short_dy[i : i + 1], med_dy[i : i + 1]),
                (
                    teacher_cfg.jitter_raw_weight,
                    teacher_cfg.jitter_dx_hat_weight,
                    teacher_cfg.jitter_ema_weight,
                    teacher_cfg.jitter_median_weight,
                ),
            )[0]
            last_good_dx = teacher_dx[i]
            last_good_dy = teacher_dy[i]
            weight[i] = 0.45 + 0.25 * conf[i] + 0.15 * (1.0 - zoom_mag[i]) + 0.15 * event[i]
            continue

        teacher_dx[i] = 0.60 * dx_hat[i] + 0.40 * trend_dx[i]
        teacher_dy[i] = 0.60 * dy_hat[i] + 0.40 * trend_dy[i]
        weight[i] = 0.50
    out["pseudo_clean_dx"] = teacher_dx
    out["pseudo_clean_dy"] = teacher_dy
    out["pseudo_weight"] = np.clip(weight, 0.05, 1.25)
    return out


def build_stratified_segment_split(segment_info: pd.DataFrame) -> Dict[str, List[str]]:
    split = {"train": [], "val": [], "test": []}
    for seg_type, g in segment_info.groupby("segment_type"):
        if "source_group" in g.columns:
            group_to_tags = (
                g.groupby("source_group")["segment_tag"]
                .apply(lambda s: sorted({str(v) for v in s.tolist()}))
                .to_dict()
            )
            groups = sorted(group_to_tags.keys())
        else:
            group_to_tags = {tag: [tag] for tag in sorted(g["segment_tag"].astype(str).tolist())}
            groups = sorted(group_to_tags.keys())
        n = len(groups)
        if n == 1:
            split["train"].extend(group_to_tags[groups[0]])
            continue
        if n == 2:
            split["train"].extend(group_to_tags[groups[0]])
            split["test"].extend(group_to_tags[groups[1]])
            continue
        n_val = 1
        n_test = 1
        n_train = max(1, n - n_val - n_test)
        train_groups = groups[:n_train]
        val_groups = groups[n_train : n_train + n_val]
        test_groups = groups[n_train + n_val : n_train + n_val + n_test]
        for group_key in train_groups:
            split["train"].extend(group_to_tags[group_key])
        for group_key in val_groups:
            split["val"].extend(group_to_tags[group_key])
        for group_key in test_groups:
            split["test"].extend(group_to_tags[group_key])
    return split


def compute_feature_stats_from_train(X_train: np.ndarray) -> Stage1NormalizationStats:
    flat = X_train.reshape(-1, X_train.shape[-1]).astype(np.float64)
    med = np.median(flat, axis=0)
    q1 = np.percentile(flat, 25, axis=0)
    q3 = np.percentile(flat, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    passthrough_idx = {
        FEATURE_INDEX.det_conf,
        FEATURE_INDEX.lost_flag,
        FEATURE_INDEX.is_meas_update,
        FEATURE_INDEX.zoom_delta,
        FEATURE_INDEX.dt_ms,
        FEATURE_INDEX.switch_score,
        FEATURE_INDEX.turn_score,
        FEATURE_INDEX.coast_count,
    }
    for idx in passthrough_idx:
        med[idx] = 0.0
        iqr[idx] = 1.0
    return Stage1NormalizationStats(median=med.astype(np.float32), iqr=iqr.astype(np.float32))


def apply_feature_stats(X: np.ndarray, stats: Stage1NormalizationStats) -> np.ndarray:
    out = X.astype(np.float32).copy()
    out = (out - stats.median.reshape(1, 1, -1)) / stats.iqr.reshape(1, 1, -1)
    return np.clip(out, -8.0, 8.0)


def build_window_segment_masks(X: np.ndarray) -> Dict[str, np.ndarray]:
    last = X[:, -1, :]
    switch = last[:, FEATURE_INDEX.switch_score]
    turn = last[:, FEATURE_INDEX.turn_score]
    zoom = np.abs(last[:, FEATURE_INDEX.zoom_delta]) > 0.10
    lost = (last[:, FEATURE_INDEX.lost_flag] > 0.5) | (last[:, FEATURE_INDEX.coast_count] > 0.0)
    jitter_signal = np.sqrt(last[:, FEATURE_INDEX.d1_dx] ** 2 + last[:, FEATURE_INDEX.d1_dy] ** 2)
    jitter_mask = jitter_signal >= np.percentile(jitter_signal, 75) if len(jitter_signal) else np.zeros(0, dtype=bool)
    fast_turn = (switch >= 0.5) | (turn >= 0.5)
    normal = ~(zoom | lost | fast_turn | jitter_mask)
    return {
        "overall": np.ones(len(last), dtype=bool),
        "normal": normal,
        "high_jitter": jitter_mask,
        "zoom_transition": zoom,
        "lost_coasting": lost,
        "fast_turn": fast_turn,
    }


def compute_metrics(pred_xy: np.ndarray, target_xy: np.ndarray, X: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float]:
    if mask is None:
        mask = np.ones(len(pred_xy), dtype=bool)
    if int(mask.sum()) == 0:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "switch_segment_lag": float("nan"),
            "turn_point_retention_error": float("nan"),
            "jitter_energy": float("nan"),
        }
    pred_xy = pred_xy[mask]
    target_xy = target_xy[mask]
    X = X[mask]
    err = pred_xy - target_xy
    pred_v = np.vstack([np.zeros((1, 2)), np.diff(pred_xy, axis=0)])
    target_v = np.vstack([np.zeros((1, 2)), np.diff(target_xy, axis=0)])
    switch = X[:, -1, FEATURE_INDEX.switch_score : FEATURE_INDEX.switch_score + 1]
    turn = X[:, -1, FEATURE_INDEX.turn_score : FEATURE_INDEX.turn_score + 1]
    lag = float(np.mean(np.abs(pred_v - target_v) * switch))
    turn_err = float(np.mean(np.abs(pred_v - target_v) * turn))
    jitter = float(np.mean(np.square(np.diff(pred_v, axis=0)))) if len(pred_v) > 2 else 0.0
    return {
        "count": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "switch_segment_lag": lag,
        "turn_point_retention_error": turn_err,
        "jitter_energy": jitter,
    }


def split_type_distribution(segment_info: pd.DataFrame, tags: Sequence[str]) -> Dict[str, int]:
    tag_set = set(tags)
    sub = segment_info[segment_info["segment_tag"].astype(str).isin(tag_set)]
    counts = sub["segment_type"].value_counts().to_dict()
    return {seg_type: int(counts.get(seg_type, 0)) for seg_type in SEGMENT_TYPES}


def ensure_all_feature_columns(df: pd.DataFrame) -> None:
    missing = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing derived feature columns: {missing}")
