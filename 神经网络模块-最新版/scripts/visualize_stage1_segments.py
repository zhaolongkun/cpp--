from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.features import ALL_FEATURE_COLUMNS
from la_cspc_ornet.models.factory import build_model
from la_cspc_ornet.stage1_common import (
    apply_feature_stats,
    build_segmented_pseudo_targets,
    ema,
    ensure_segment_columns,
    prefilter_cfg_from_dict,
    rolling_median,
    teacher_cfg_from_dict,
)


SEGMENT_ORDER = ["stable", "jitter", "recover", "turn", "zoom"]


def moving_average(x: np.ndarray, win: int = 5) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        lo = max(0, i - win + 1)
        out[i] = float(np.mean(x[lo : i + 1]))
    return out


def load_model_from_config(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    ckpt = torch.load(Path(cfg["paths"]["out_dir"]) / "best_stage1_clean.pt", map_location="cpu")
    _, _, model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, model


def infer_segment(model, seg_df: pd.DataFrame, seq_len: int, feature_stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    feat = seg_df[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    baseline_seq = seg_df[["dx_raw", "dy_raw"]].to_numpy(dtype=np.float32)
    stable_baseline_seq = seg_df[["stable_baseline_dx", "stable_baseline_dy"]].to_numpy(dtype=np.float32)
    med = np.asarray([feature_stats[name]["median"] for name in ALL_FEATURE_COLUMNS], dtype=np.float32)
    iqr = np.asarray([feature_stats[name]["iqr"] for name in ALL_FEATURE_COLUMNS], dtype=np.float32)
    feat = np.clip((feat - med.reshape(1, -1)) / iqr.reshape(1, -1), -8.0, 8.0)
    preds = np.full((len(seg_df), 2), np.nan, dtype=np.float32)
    for end in range(seq_len - 1, len(seg_df)):
        window = torch.from_numpy(feat[end - seq_len + 1 : end + 1]).unsqueeze(0)
        baseline_window = torch.from_numpy(baseline_seq[end - seq_len + 1 : end + 1]).unsqueeze(0)
        stable_baseline_window = torch.from_numpy(stable_baseline_seq[end - seq_len + 1 : end + 1]).unsqueeze(0)
        with torch.no_grad():
            pred = model(window, baseline_seq=baseline_window, stable_baseline_seq=stable_baseline_window)["clean"][0].cpu().numpy()
        preds[end] = pred
    return preds


def select_usable_segments(meta: Dict[str, object]) -> Dict[str, str]:
    selected = meta.get("data_selection", {}).get("selected_tag_decisions", {}) if isinstance(meta.get("data_selection"), dict) else {}
    return {str(tag): str(decision) for tag, decision in selected.items()}


def tag_split_name(meta: Dict[str, object], seg_tag: str) -> str:
    if seg_tag in {str(x) for x in meta.get("train_tags", [])}:
        return "train"
    if seg_tag in {str(x) for x in meta.get("val_tags", [])}:
        return "val"
    if seg_tag in {str(x) for x in meta.get("test_tags", [])}:
        return "test"
    return "unassigned"


def choose_segment_per_type(df: pd.DataFrame, usable_tag_decisions: Dict[str, str], meta: Dict[str, object]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    if usable_tag_decisions:
        cand = df[df["segment_tag"].astype(str).isin(set(usable_tag_decisions.keys()))].copy()
        cand["decision_rank"] = cand["segment_tag"].astype(str).map(lambda x: 0 if usable_tag_decisions.get(x) == "keep" else 1)
    else:
        cand = df.copy()
        cand["decision_rank"] = 1
    for seg_type in SEGMENT_ORDER:
        sub = cand[cand["segment_type"].astype(str) == seg_type]
        if len(sub) == 0:
            continue
        stats = (
            sub.groupby("segment_tag")
            .agg(length=("frame_id", "count"), decision_rank=("decision_rank", "min"))
            .sort_values(["decision_rank", "length"], ascending=[True, False])
        )
        seg_tag = str(stats.index[0])
        out[seg_type] = (seg_tag, tag_split_name(meta, seg_tag))
    return out


def build_baseline_curves(seg_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    dx = seg_df["sensor_dx_raw"].to_numpy(dtype=np.float64) if "sensor_dx_raw" in seg_df.columns else seg_df["dx_raw"].to_numpy(dtype=np.float64)
    dy = seg_df["sensor_dy_raw"].to_numpy(dtype=np.float64) if "sensor_dy_raw" in seg_df.columns else seg_df["dy_raw"].to_numpy(dtype=np.float64)
    return {
        "raw": np.column_stack([dx, dy]).astype(np.float32),
        "ema": np.column_stack([ema(dx, alpha=0.20), ema(dy, alpha=0.20)]).astype(np.float32),
        "sma": np.column_stack([moving_average(dx, win=5), moving_average(dy, win=5)]).astype(np.float32),
        "median": np.column_stack([rolling_median(dx, win=5), rolling_median(dy, win=5)]).astype(np.float32),
    }


def plot_segment(seg_df: pd.DataFrame, baselines: Dict[str, np.ndarray], pred_tcn: np.ndarray, pred_dual: np.ndarray, output_png: Path) -> None:
    t = np.arange(len(seg_df))
    teacher = seg_df[["pseudo_clean_dx", "pseudo_clean_dy"]].to_numpy(dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    labels = ["dx", "dy"]
    for axis_idx, ax in enumerate(axes):
        ax.plot(t, baselines["raw"][:, axis_idx], label="raw", linewidth=1.0, alpha=0.8)
        ax.plot(t, baselines["ema"][:, axis_idx], label="ema", linewidth=1.2)
        ax.plot(t, baselines["sma"][:, axis_idx], label="sma", linewidth=1.2)
        ax.plot(t, teacher[:, axis_idx], label="teacher", linewidth=1.4)
        ax.plot(t, pred_tcn[:, axis_idx], label="tcn_gru", linewidth=1.4)
        ax.plot(t, pred_dual[:, axis_idx], label="dual_state", linewidth=1.4)
        ax.set_ylabel(labels[axis_idx])
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="upper right", ncol=6)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize stage1 segment curves for raw/teacher/model outputs")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--tcn_config", required=True)
    ap.add_argument("--dual_config", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = ensure_segment_columns(df)
    meta = json.loads(Path(args.meta_json).read_text(encoding="utf-8"))
    teacher_cfg = teacher_cfg_from_dict(meta.get("teacher"))
    prefilter_cfg = prefilter_cfg_from_dict(meta.get("prefilter"))
    df = build_segmented_pseudo_targets(df, teacher_cfg=teacher_cfg, prefilter_cfg=prefilter_cfg)

    _, tcn_model = load_model_from_config(args.tcn_config)
    _, dual_model = load_model_from_config(args.dual_config)
    stats = meta["normalization"]
    seq_len = int(meta["seq_len"])
    usable_tag_decisions = select_usable_segments(meta)
    selected = choose_segment_per_type(df, usable_tag_decisions, meta)

    out_dir = Path(args.output_dir)
    summary = {}
    for seg_type, seg_info in selected.items():
        seg_tag, split_name = seg_info
        seg_df = df[df["segment_tag"].astype(str) == seg_tag].sort_values(["timestamp_ms", "frame_id"]).reset_index(drop=True)
        baselines = build_baseline_curves(seg_df)
        pred_tcn = infer_segment(tcn_model, seg_df, seq_len=seq_len, feature_stats=stats)
        pred_dual = infer_segment(dual_model, seg_df, seq_len=seq_len, feature_stats=stats)
        png_path = out_dir / f"{seg_type}_{seg_tag}.png"
        plot_segment(seg_df, baselines, pred_tcn, pred_dual, png_path)
        summary[seg_type] = {
            "segment_tag": seg_tag,
            "split": split_name,
            "decision": usable_tag_decisions.get(seg_tag, "unknown"),
            "plot": str(png_path),
            "baselines": ["raw", "ema", "sma", "teacher", "tcn_gru", "dual_state"],
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
