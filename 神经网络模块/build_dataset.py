#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2S-LAVS-Lite 数据集构造脚本

功能：
1) 读取逐帧 CSV，按 run_id + timestamp_ms 排序
2) 离线计算 bbox_area / delta_cmd_target / lost_streak
3) 按规则过滤异常样本并构造 T=8 的时序样本
4) 按 run_id 划分 train/val/test（防泄漏）
5) 仅用训练集拟合 z-score 标准化器，并应用到 val/test
6) 保存 dataset.npz / scaler.json / feature_spec.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 固定列定义（与工程协议一致）
# -----------------------------
CSV_REQUIRED_COLUMNS: List[str] = [
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
    "cmd_expert_x",
    "cmd_expert_y",
    "cmd_sent_x",
    "cmd_sent_y",
    "act_pos_x",
    "act_pos_y",
    "act_vel_x",
    "act_vel_y",
]

# 训练输入特征顺序必须锁死
FEATURE_COLUMNS: List[str] = [
    "dx_hat",
    "dy_hat",
    "vx_hat",
    "vy_hat",
    "bbox_area",
    "det_conf",
    "lost_flag",
    "is_meas_update",
    "meas_age_ms",
    "cmd_base_x",
    "cmd_base_y",
    "dt_ms",
]

# 标签顺序固定
TARGET_COLUMNS: List[str] = [
    "delta_cmd_target_x",
    "delta_cmd_target_y",
]

# 连续特征（做 z-score）
CONTINUOUS_FEATURES: List[str] = [
    "dx_hat",
    "dy_hat",
    "vx_hat",
    "vy_hat",
    "bbox_area",
    "meas_age_ms",
    "cmd_base_x",
    "cmd_base_y",
    "dt_ms",
]

# 保持原值/二值特征
RAW_KEEP_FEATURES: List[str] = ["det_conf"]
BINARY_FEATURES: List[str] = ["lost_flag", "is_meas_update"]

# 旧版 tracker_log.csv 最小字段集合（兼容模式）
LEGACY_MIN_COLUMNS: List[str] = [
    "time_ns",
    "dx_hat",
    "dy_hat",
    "cmd_x",
    "cmd_y",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "is_meas_update",
    "meas_age_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R2S-LAVS-Lite sequence dataset from frame-wise CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="输入逐帧 CSV 文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    parser.add_argument("--seq_len", type=int, default=8, help="时序窗口长度，默认 8")
    parser.add_argument("--stride", type=int, default=1, help="滑窗步长，默认 1")

    parser.add_argument("--train_ratio", type=float, default=0.7, help="run_id 划分比例 train")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="run_id 划分比例 val")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="run_id 划分比例 test")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--meas_age_max", type=float, default=300.0, help="标签帧最大 meas_age_ms")
    parser.add_argument("--dt_break_scale", type=float, default=1.8, help="dt 断点阈值系数")
    parser.add_argument("--lost_streak_break", type=int, default=5, help="lost_streak 断点阈值")

    parser.add_argument(
        "--label_outlier_mode",
        type=str,
        default="drop",
        choices=["drop", "clip"],
        help="标签异常处理方式: drop 或 clip",
    )
    parser.add_argument(
        "--label_abs_max",
        type=float,
        default=0.0,
        help="标签绝对值阈值，<=0 表示不启用。>0 时按 label_outlier_mode 处理",
    )

    parser.add_argument("--dataset_name", type=str, default="dataset_t8.npz", help="输出 npz 文件名")
    parser.add_argument("--verbose", action="store_true", help="打印更详细日志")

    # 旧版日志兼容（仅用于过渡/联调）
    parser.add_argument("--legacy_compat", action="store_true", help="启用旧版 tracker_log.csv 兼容转换")
    parser.add_argument("--legacy_img_w", type=float, default=1920.0, help="旧版日志默认图像宽（兼容模式）")
    parser.add_argument("--legacy_img_h", type=float, default=1080.0, help="旧版日志默认图像高（兼容模式）")
    parser.add_argument(
        "--legacy_cmd_base_mode",
        type=str,
        default="same_as_cmd",
        choices=["same_as_cmd", "zero"],
        help="兼容模式下 cmd_base 生成方式: same_as_cmd 或 zero",
    )
    parser.add_argument(
        "--legacy_run_id",
        type=str,
        default="",
        help="兼容模式下 run_id（默认使用文件名）",
    )

    args = parser.parse_args()

    # 基础参数校验
    if args.seq_len <= 0:
        raise ValueError("seq_len 必须 > 0")
    if args.stride <= 0:
        raise ValueError("stride 必须 > 0")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"train/val/test 比例之和必须为 1.0，当前为 {ratio_sum}")

    return args


def _assert_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")


def _is_legacy_tracker_schema(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in LEGACY_MIN_COLUMNS)


def _convert_legacy_tracker_to_current(df: pd.DataFrame, csv_path: str, args: argparse.Namespace) -> pd.DataFrame:
    """
    将旧版 tracker_log.csv 转为当前 R2S-LAVS-Lite 所需字段。
    注意: 该模式仅用于过渡联调，不建议作为正式论文训练数据来源。
    """
    print("[Warn] 检测到旧版日志表头，已启用 legacy 兼容转换。")
    print("[Warn] 建议尽快切换到新CSV字段在线记录（含 cmd_base_x/cmd_expert_x 等）。")

    out = pd.DataFrame()
    n = len(df)
    if n == 0:
        return out

    run_id = args.legacy_run_id.strip() if args.legacy_run_id else f"legacy_{Path(csv_path).stem}"
    out["run_id"] = [run_id] * n

    # 时间与帧号
    if "timestamp_ms" in df.columns:
        out["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    else:
        out["timestamp_ms"] = pd.to_numeric(df["time_ns"], errors="coerce") / 1e6

    if "frame_id" in df.columns:
        out["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    elif "frame_seq" in df.columns:
        out["frame_id"] = pd.to_numeric(df["frame_seq"], errors="coerce")
    else:
        out["frame_id"] = np.arange(n, dtype=np.int64)

    # dt_ms 由时间差计算
    out["dt_ms"] = out["timestamp_ms"].diff()
    dt_pos = out.loc[out["dt_ms"] > 0, "dt_ms"]
    dt_med = float(dt_pos.median()) if len(dt_pos) > 0 else 33.33
    if not np.isfinite(dt_med) or dt_med <= 0:
        dt_med = 33.33
    out.loc[out["dt_ms"].isna(), "dt_ms"] = dt_med
    out["dt_ms"] = out["dt_ms"].fillna(dt_med)

    # 图像尺寸
    if "img_w" in df.columns and "img_h" in df.columns:
        out["img_w"] = pd.to_numeric(df["img_w"], errors="coerce").fillna(float(args.legacy_img_w))
        out["img_h"] = pd.to_numeric(df["img_h"], errors="coerce").fillna(float(args.legacy_img_h))
    else:
        out["img_w"] = float(args.legacy_img_w)
        out["img_h"] = float(args.legacy_img_h)

    # 检测框
    for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    # 检测置信度
    if "det_conf" in df.columns:
        out["det_conf"] = pd.to_numeric(df["det_conf"], errors="coerce")
    elif "det_count" in df.columns:
        det_count = pd.to_numeric(df["det_count"], errors="coerce").fillna(0.0)
        out["det_conf"] = (det_count > 0).astype(np.float32)
    else:
        out["det_conf"] = 1.0

    # 状态估计
    out["dx_hat"] = pd.to_numeric(df["dx_hat"], errors="coerce")
    out["dy_hat"] = pd.to_numeric(df["dy_hat"], errors="coerce")

    dt_s = (out["dt_ms"].to_numpy(dtype=np.float64) / 1000.0).astype(np.float64)
    dx = out["dx_hat"].to_numpy(dtype=np.float64)
    dy = out["dy_hat"].to_numpy(dtype=np.float64)
    vx = np.zeros_like(dx, dtype=np.float64)
    vy = np.zeros_like(dy, dtype=np.float64)
    if len(dx) > 1:
        denom = np.where(dt_s[1:] > 1e-6, dt_s[1:], np.nan)
        vx[1:] = np.diff(dx) / denom
        vy[1:] = np.diff(dy) / denom
    vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
    vy = np.nan_to_num(vy, nan=0.0, posinf=0.0, neginf=0.0)
    out["vx_hat"] = vx
    out["vy_hat"] = vy

    # 失锁与测量状态
    if "lost_flag" in df.columns:
        out["lost_flag"] = (pd.to_numeric(df["lost_flag"], errors="coerce").fillna(0) >= 0.5).astype(np.int8)
    elif "coast_count" in df.columns:
        coast = pd.to_numeric(df["coast_count"], errors="coerce").fillna(0.0)
        out["lost_flag"] = (coast > 0).astype(np.int8)
    elif "det_count" in df.columns:
        det_count = pd.to_numeric(df["det_count"], errors="coerce").fillna(0.0)
        out["lost_flag"] = (det_count <= 0).astype(np.int8)
    else:
        out["lost_flag"] = 0

    out["is_meas_update"] = (pd.to_numeric(df["is_meas_update"], errors="coerce").fillna(0) >= 0.5).astype(np.int8)
    out["meas_age_ms"] = pd.to_numeric(df["meas_age_ms"], errors="coerce").fillna(0.0)

    # 命令字段
    cmd_x = pd.to_numeric(df["cmd_x"], errors="coerce")
    cmd_y = pd.to_numeric(df["cmd_y"], errors="coerce")

    if "cmd_base_x" in df.columns and "cmd_base_y" in df.columns:
        out["cmd_base_x"] = pd.to_numeric(df["cmd_base_x"], errors="coerce")
        out["cmd_base_y"] = pd.to_numeric(df["cmd_base_y"], errors="coerce")
    else:
        if args.legacy_cmd_base_mode == "zero":
            out["cmd_base_x"] = 0.0
            out["cmd_base_y"] = 0.0
            print("[Warn] legacy_cmd_base_mode=zero：cmd_base 由0填充，仅用于联调。")
        else:
            out["cmd_base_x"] = cmd_x
            out["cmd_base_y"] = cmd_y
            print("[Warn] legacy_cmd_base_mode=same_as_cmd：可能导致标签接近0。")

    if "cmd_expert_x" in df.columns and "cmd_expert_y" in df.columns:
        out["cmd_expert_x"] = pd.to_numeric(df["cmd_expert_x"], errors="coerce")
        out["cmd_expert_y"] = pd.to_numeric(df["cmd_expert_y"], errors="coerce")
    else:
        out["cmd_expert_x"] = cmd_x
        out["cmd_expert_y"] = cmd_y
        print("[Warn] 旧日志无 cmd_expert_*，已使用 cmd_x/cmd_y 作为伪 expert。")

    if "cmd_sent_x" in df.columns and "cmd_sent_y" in df.columns:
        out["cmd_sent_x"] = pd.to_numeric(df["cmd_sent_x"], errors="coerce")
        out["cmd_sent_y"] = pd.to_numeric(df["cmd_sent_y"], errors="coerce")
    else:
        out["cmd_sent_x"] = cmd_x
        out["cmd_sent_y"] = cmd_y

    # 执行器反馈（旧日志无则置0）
    for c in ["act_pos_x", "act_pos_y", "act_vel_x", "act_vel_y"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = 0.0

    # 最后按当前协议补齐检查
    _assert_required_columns(out, CSV_REQUIRED_COLUMNS)
    return out


def _compute_lost_streak(flags: Sequence[int]) -> np.ndarray:
    streak = np.zeros(len(flags), dtype=np.int32)
    cnt = 0
    for i, v in enumerate(flags):
        if int(v) == 1:
            cnt += 1
        else:
            cnt = 0
        streak[i] = cnt
    return streak


def _safe_to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_and_preprocess(csv_path: str, args: argparse.Namespace) -> pd.DataFrame:
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    # 若缺少新协议字段，允许兼容旧版 tracker_log.csv
    missing = [c for c in CSV_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        if args.legacy_compat and _is_legacy_tracker_schema(df):
            df = _convert_legacy_tracker_to_current(df, csv_path=csv_path, args=args)
        else:
            raise ValueError(
                "CSV 表头与当前协议不匹配，缺少列: {}\n"
                "当前脚本需要新协议字段；如你使用旧版 tracker_log.csv，可加 --legacy_compat 先联调。".format(missing)
            )

    _assert_required_columns(df, CSV_REQUIRED_COLUMNS)

    # run_id 强制字符串，防止后续切分混乱
    df["run_id"] = df["run_id"].astype(str)

    # 数值列转换
    numeric_cols = [c for c in CSV_REQUIRED_COLUMNS if c != "run_id"]
    df = _safe_to_numeric(df, numeric_cols)

    # 关键字段缺失行丢弃
    critical_cols = [
        "run_id",
        "timestamp_ms",
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
        "cmd_expert_x",
        "cmd_expert_y",
    ]
    before_drop = len(df)
    df = df.dropna(subset=critical_cols)
    dropped_na = before_drop - len(df)

    # 排序
    df = df.sort_values(["run_id", "timestamp_ms"], ascending=[True, True]).reset_index(drop=True)

    # 清理无效图像尺寸
    before_size = len(df)
    df = df[(df["img_w"] > 0) & (df["img_h"] > 0)].copy()
    dropped_bad_size = before_size - len(df)

    # 二值列标准化为 0/1
    for col in BINARY_FEATURES:
        df[col] = (df[col] >= 0.5).astype(np.int8)

    # det_conf 限制到 [0, 1]
    df["det_conf"] = df["det_conf"].clip(lower=0.0, upper=1.0)

    # 离线字段：bbox_area
    bbox_w = (df["bbox_x2"] - df["bbox_x1"]).clip(lower=0.0)
    bbox_h = (df["bbox_y2"] - df["bbox_y1"]).clip(lower=0.0)
    img_area = df["img_w"] * df["img_h"]
    df["bbox_area"] = (bbox_w * bbox_h) / img_area
    df["bbox_area"] = df["bbox_area"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    # 离线字段：标签
    df["delta_cmd_target_x"] = df["cmd_expert_x"] - df["cmd_base_x"]
    df["delta_cmd_target_y"] = df["cmd_expert_y"] - df["cmd_base_y"]

    # lost_streak（run 内连续失锁计数）
    df["lost_streak"] = (
        df.groupby("run_id", sort=False)["lost_flag"]
        .transform(lambda s: pd.Series(_compute_lost_streak(s.to_numpy()), index=s.index))
        .astype(np.int32)
    )

    # dt 断点阈值（按 run 内正 dt 中位数）
    run_median_dt = (
        df[df["dt_ms"] > 0]
        .groupby("run_id")["dt_ms"]
        .median()
        .to_dict()
    )
    global_dt = float(df.loc[df["dt_ms"] > 0, "dt_ms"].median()) if (df["dt_ms"] > 0).any() else 1.0
    if not np.isfinite(global_dt) or global_dt <= 0:
        global_dt = 1.0

    df["run_median_dt"] = df["run_id"].map(run_median_dt).fillna(global_dt)
    df["dt_break_flag"] = (df["dt_ms"] <= 0) | (df["dt_ms"] > args.dt_break_scale * df["run_median_dt"])
    df["lost_break_flag"] = df["lost_streak"] >= int(args.lost_streak_break)
    df["break_flag"] = df["dt_break_flag"] | df["lost_break_flag"]

    # 再次清理 feature / target 中的 inf
    need_finite_cols = list(set(FEATURE_COLUMNS + TARGET_COLUMNS + ["timestamp_ms"]))
    before_inf = len(df)
    finite_mask = np.isfinite(df[need_finite_cols].to_numpy(dtype=np.float64)).all(axis=1)
    df = df.loc[finite_mask].copy()
    dropped_inf = before_inf - len(df)

    print(f"[Info] 原始行数: {before_drop}")
    print(f"[Info] 因关键字段缺失丢弃: {dropped_na}")
    print(f"[Info] 因无效图像尺寸丢弃: {dropped_bad_size}")
    print(f"[Info] 因非有限数值丢弃: {dropped_inf}")
    print(f"[Info] 预处理后剩余行数: {len(df)}")
    print(f"[Info] run_id 数量: {df['run_id'].nunique()}")

    return df.reset_index(drop=True)


def split_run_ids(
    run_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    run_ids = list(run_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(run_ids)

    n = len(run_ids)
    if n == 0:
        return [], [], []

    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()
    raw = ratios * n
    counts = np.floor(raw).astype(int)

    # 分配余数给小数部分大的集合
    remain = n - int(counts.sum())
    frac_order = np.argsort(-(raw - counts))
    for i in range(remain):
        counts[frac_order[i % 3]] += 1

    # 至少保证 train 非空
    if counts[0] == 0:
        donor = int(np.argmax(counts[1:]) + 1) if counts[1:].sum() > 0 else -1
        if donor >= 1 and counts[donor] > 0:
            counts[donor] -= 1
            counts[0] += 1

    train_n, val_n, test_n = int(counts[0]), int(counts[1]), int(counts[2])
    if train_n + val_n + test_n != n:
        # 理论不会进入，兜底修正
        test_n = n - train_n - val_n

    train_ids = run_ids[:train_n]
    val_ids = run_ids[train_n : train_n + val_n]
    test_ids = run_ids[train_n + val_n : train_n + val_n + test_n]

    return train_ids, val_ids, test_ids


def _extract_valid_segments(run_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    根据 break_flag 将单个 run 切分为多个连续有效段。
    break_flag=True 的行本身不进入任何段，也不允许窗口跨越。
    """
    segments: List[pd.DataFrame] = []
    mask = (~run_df["break_flag"]).to_numpy()

    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if (not ok) and (start is not None):
            seg = run_df.iloc[start:i]
            if len(seg) > 0:
                segments.append(seg)
            start = None
    if start is not None:
        seg = run_df.iloc[start:]
        if len(seg) > 0:
            segments.append(seg)

    return segments


def build_sequence_samples(
    df: pd.DataFrame,
    run_ids: Sequence[str],
    seq_len: int,
    stride: int,
    meas_age_max: float,
    label_outlier_mode: str,
    label_abs_max: float,
) -> Dict[str, np.ndarray]:
    """
    对给定 run_id 子集构造序列样本。
    返回:
      X: [N, T, F], y: [N, 2], run_id: [N], timestamp_ms: [N]
    """
    sub = df[df["run_id"].isin(run_ids)].copy()
    sub = sub.sort_values(["run_id", "timestamp_ms"], ascending=[True, True]).reset_index(drop=True)

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    rid_list: List[str] = []
    ts_list: List[int] = []

    stats = {
        "total_candidate": 0,
        "skip_lost_label": 0,
        "skip_meas_age": 0,
        "skip_invalid_pseudo_label": 0,
        "skip_label_outlier": 0,
        "skip_non_finite": 0,
        "accepted": 0,
    }

    for rid, run_df in sub.groupby("run_id", sort=False):
        run_df = run_df.sort_values("timestamp_ms").reset_index(drop=True)
        segments = _extract_valid_segments(run_df)

        for seg in segments:
            if len(seg) < seq_len:
                continue

            feat_arr = seg[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            target_arr = seg[TARGET_COLUMNS].to_numpy(dtype=np.float32)
            lost_arr = seg["lost_flag"].to_numpy(dtype=np.int8)
            meas_arr = seg["meas_age_ms"].to_numpy(dtype=np.float32)
            ts_arr = seg["timestamp_ms"].to_numpy(dtype=np.int64)
            if "pseudo_expert_valid" in seg.columns:
                pseudo_valid_arr = seg["pseudo_expert_valid"].to_numpy(dtype=np.int8)
            else:
                pseudo_valid_arr = np.ones(len(seg), dtype=np.int8)

            for end in range(seq_len - 1, len(seg), stride):
                stats["total_candidate"] += 1

                # 规则1: 当前帧 lost_flag=1 不作为标签样本
                if int(lost_arr[end]) == 1:
                    stats["skip_lost_label"] += 1
                    continue

                # 规则2: meas_age 超阈值跳过
                if float(meas_arr[end]) > float(meas_age_max):
                    stats["skip_meas_age"] += 1
                    continue

                # 规则3: 若存在 pseudo_expert_valid 列，则标签帧必须是有效伪专家监督
                if int(pseudo_valid_arr[end]) != 1:
                    stats["skip_invalid_pseudo_label"] += 1
                    continue

                x_win = feat_arr[end - seq_len + 1 : end + 1]
                y = target_arr[end].copy()

                if not np.isfinite(x_win).all() or not np.isfinite(y).all():
                    stats["skip_non_finite"] += 1
                    continue

                # 规则4: 标签异常值（drop 或 clip）
                if label_abs_max > 0:
                    outlier = np.any(np.abs(y) > label_abs_max)
                    if outlier:
                        if label_outlier_mode == "drop":
                            stats["skip_label_outlier"] += 1
                            continue
                        # clip 模式
                        y = np.clip(y, -label_abs_max, label_abs_max)

                x_list.append(x_win)
                y_list.append(y)
                rid_list.append(str(rid))
                ts_list.append(int(ts_arr[end]))
                stats["accepted"] += 1

    if len(x_list) == 0:
        X = np.empty((0, seq_len, len(FEATURE_COLUMNS)), dtype=np.float32)
        y = np.empty((0, len(TARGET_COLUMNS)), dtype=np.float32)
        run_idx = np.empty((0,), dtype="<U1")
        ts_idx = np.empty((0,), dtype=np.int64)
    else:
        X = np.stack(x_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)
        run_idx = np.asarray(rid_list, dtype=str)
        ts_idx = np.asarray(ts_list, dtype=np.int64)

    return {
        "X": X,
        "y": y,
        "run_id": run_idx,
        "timestamp_ms": ts_idx,
        "stats": np.array([stats], dtype=object),
    }


def fit_scaler_from_train(X_train: np.ndarray) -> Dict[str, List[float]]:
    if X_train.ndim != 3:
        raise ValueError(f"X_train 维度应为 [N,T,F]，当前 {X_train.shape}")
    if X_train.shape[0] == 0:
        raise ValueError("训练集样本数为 0，无法拟合标准化器。请检查数据与过滤规则。")

    cont_idx = [FEATURE_COLUMNS.index(c) for c in CONTINUOUS_FEATURES]
    flat = X_train.reshape(-1, X_train.shape[-1]).astype(np.float64)

    mean = flat[:, cont_idx].mean(axis=0)
    std = flat[:, cont_idx].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    scaler = {
        "feature_order": FEATURE_COLUMNS,
        "continuous_features": CONTINUOUS_FEATURES,
        "continuous_indices": cont_idx,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "binary_features": BINARY_FEATURES,
        "raw_keep_features": RAW_KEEP_FEATURES,
    }
    return scaler


def apply_scaler(X: np.ndarray, scaler: Dict[str, List[float]]) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32)

    out = X.astype(np.float32).copy()
    idx = np.asarray(scaler["continuous_indices"], dtype=np.int64)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)

    out[:, :, idx] = (out[:, :, idx] - mean[None, None, :]) / std[None, None, :]
    return out


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取并预处理
    df = load_and_preprocess(args.input_csv, args)
    if len(df) == 0:
        raise RuntimeError("预处理后无可用数据。")

    # 2) run_id 划分
    unique_runs = sorted(df["run_id"].unique().tolist())
    train_ids, val_ids, test_ids = split_run_ids(
        unique_runs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"[Split] train runs: {len(train_ids)}, val runs: {len(val_ids)}, test runs: {len(test_ids)}")
    if args.verbose:
        print(f"[Split] train_ids={train_ids}")
        print(f"[Split] val_ids={val_ids}")
        print(f"[Split] test_ids={test_ids}")

    # 3) 构造时序样本
    train_data = build_sequence_samples(
        df,
        run_ids=train_ids,
        seq_len=args.seq_len,
        stride=args.stride,
        meas_age_max=args.meas_age_max,
        label_outlier_mode=args.label_outlier_mode,
        label_abs_max=args.label_abs_max,
    )
    val_data = build_sequence_samples(
        df,
        run_ids=val_ids,
        seq_len=args.seq_len,
        stride=args.stride,
        meas_age_max=args.meas_age_max,
        label_outlier_mode=args.label_outlier_mode,
        label_abs_max=args.label_abs_max,
    )
    test_data = build_sequence_samples(
        df,
        run_ids=test_ids,
        seq_len=args.seq_len,
        stride=args.stride,
        meas_age_max=args.meas_age_max,
        label_outlier_mode=args.label_outlier_mode,
        label_abs_max=args.label_abs_max,
    )

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    print(f"[Shape] X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[Shape] X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"[Shape] X_test={X_test.shape}, y_test={y_test.shape}")

    if X_train.shape[0] == 0:
        raise RuntimeError("训练集样本为空，无法继续。请检查采集数据或放宽过滤阈值。")

    # 4) 标准化（仅训练集拟合）
    scaler = fit_scaler_from_train(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val = apply_scaler(X_val, scaler)
    X_test = apply_scaler(X_test, scaler)

    # 5) 保存数据与元信息
    npz_path = output_dir / args.dataset_name
    np.savez_compressed(
        npz_path,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        run_id_train=train_data["run_id"].astype(str),
        timestamp_train=train_data["timestamp_ms"].astype(np.int64),
        run_id_val=val_data["run_id"].astype(str),
        timestamp_val=val_data["timestamp_ms"].astype(np.int64),
        run_id_test=test_data["run_id"].astype(str),
        timestamp_test=test_data["timestamp_ms"].astype(np.int64),
        feature_names=np.asarray(FEATURE_COLUMNS, dtype=str),
        target_names=np.asarray(TARGET_COLUMNS, dtype=str),
    )

    scaler_path = output_dir / "scaler.json"
    save_json(scaler, scaler_path)

    feature_spec = {
        "sequence_length": int(args.seq_len),
        "input_dim": int(len(FEATURE_COLUMNS)),
        "output_dim": int(len(TARGET_COLUMNS)),
        "feature_order": FEATURE_COLUMNS,
        "target_order": TARGET_COLUMNS,
        "continuous_features": CONTINUOUS_FEATURES,
        "binary_features": BINARY_FEATURES,
        "raw_keep_features": RAW_KEEP_FEATURES,
        "stride": int(args.stride),
        "meas_age_max": float(args.meas_age_max),
        "dt_break_scale": float(args.dt_break_scale),
        "lost_streak_break": int(args.lost_streak_break),
        "label_outlier_mode": args.label_outlier_mode,
        "label_abs_max": float(args.label_abs_max),
    }
    feature_spec_path = output_dir / "feature_spec.json"
    save_json(feature_spec, feature_spec_path)

    # 保存 split run_id，便于复现
    split_info = {
        "train_runs": train_ids,
        "val_runs": val_ids,
        "test_runs": test_ids,
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
    }
    split_path = output_dir / "split_run_ids.json"
    save_json(split_info, split_path)

    # 打印过滤统计
    train_stats = train_data["stats"][0]
    val_stats = val_data["stats"][0]
    test_stats = test_data["stats"][0]
    print(f"[Stats][train] {train_stats}")
    print(f"[Stats][val]   {val_stats}")
    print(f"[Stats][test]  {test_stats}")
    print(f"[Done] npz saved: {npz_path}")
    print(f"[Done] scaler saved: {scaler_path}")
    print(f"[Done] feature spec saved: {feature_spec_path}")
    print(f"[Done] split info saved: {split_path}")


if __name__ == "__main__":
    main()
