from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.features import ALL_FEATURE_COLUMNS, REQUIRED_LOG_COLUMNS_STAGE1, check_feature_index_alignment
from la_cspc_ornet.stage1_common import (
    Stage1NormalizationStats,
    apply_feature_stats,
    build_segmented_pseudo_targets,
    build_stratified_segment_split,
    compute_feature_stats_from_train,
    ensure_all_feature_columns,
    ensure_segment_columns,
    prefilter_cfg_from_dict,
    split_type_distribution,
    teacher_cfg_from_dict,
)


def ensure_required(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_LOG_COLUMNS_STAGE1 if c not in df.columns]
    if missing:
        raise ValueError(f"tracker log missing required columns: {missing}")


def build_windows_for_segment(
    g: pd.DataFrame, seq_len: int, stride: int
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[str],
    List[str],
    List[str],
]:
    Xs: List[np.ndarray] = []
    baseline_seqs: List[np.ndarray] = []
    stable_baseline_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []
    baselines: List[np.ndarray] = []
    stable_baselines: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    tags: List[str] = []
    types: List[str] = []
    groups: List[str] = []

    feat = g[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    baseline_seq = g[["dx_raw", "dy_raw"]].to_numpy(dtype=np.float32)
    stable_baseline_seq = g[["stable_baseline_dx", "stable_baseline_dy"]].to_numpy(dtype=np.float32)
    clean_seq = g[["pseudo_clean_dx", "pseudo_clean_dy"]].to_numpy(dtype=np.float32)
    weight = g[["pseudo_weight"]].to_numpy(dtype=np.float32)
    seg_tag = str(g.iloc[0]["segment_tag"])
    seg_type = str(g.iloc[0]["segment_type"])
    seg_group = str(g.iloc[0]["source_group"])

    for end in range(seq_len - 1, len(g), stride):
        start = end - seq_len + 1
        Xs.append(feat[start : end + 1])
        baseline_seqs.append(baseline_seq[start : end + 1])
        stable_baseline_seqs.append(stable_baseline_seq[start : end + 1])
        y_seqs.append(clean_seq[start : end + 1])
        baselines.append(baseline_seq[end])
        stable_baselines.append(stable_baseline_seq[end])
        ys.append(clean_seq[end])
        ws.append(weight[end])
        tags.append(seg_tag)
        types.append(seg_type)
        groups.append(seg_group)
    return Xs, baseline_seqs, stable_baseline_seqs, y_seqs, baselines, stable_baselines, ys, ws, tags, types, groups


def build_all_windows(df: pd.DataFrame, seq_len: int, stride: int) -> Dict[str, object]:
    Xs: List[np.ndarray] = []
    baseline_seqs: List[np.ndarray] = []
    stable_baseline_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []
    baselines: List[np.ndarray] = []
    stable_baselines: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    tags: List[str] = []
    types: List[str] = []
    groups: List[str] = []
    for _, g in df.groupby("segment_tag"):
        g = g.sort_values(["timestamp_ms", "frame_id"]).reset_index(drop=True)
        if len(g) < seq_len:
            continue
        (
            seg_X,
            seg_baseline_seq,
            seg_stable_baseline_seq,
            seg_y_seq,
            seg_baseline,
            seg_stable_baseline,
            seg_y,
            seg_w,
            seg_tags,
            seg_types,
            seg_groups,
        ) = build_windows_for_segment(
            g, seq_len, stride
        )
        Xs.extend(seg_X)
        baseline_seqs.extend(seg_baseline_seq)
        stable_baseline_seqs.extend(seg_stable_baseline_seq)
        y_seqs.extend(seg_y_seq)
        baselines.extend(seg_baseline)
        stable_baselines.extend(seg_stable_baseline)
        ys.extend(seg_y)
        ws.extend(seg_w)
        tags.extend(seg_tags)
        types.extend(seg_types)
        groups.extend(seg_groups)
    if not Xs:
        raise RuntimeError("no stage1 windows built; check segment lengths and seq_len")
    return {
        "X": np.stack(Xs),
        "baseline_seq": np.stack(baseline_seqs),
        "stable_baseline_seq": np.stack(stable_baseline_seqs),
        "y_clean_seq": np.stack(y_seqs),
        "baseline": np.stack(baselines),
        "stable_baseline": np.stack(stable_baselines),
        "y_clean": np.stack(ys),
        "weight": np.stack(ws),
        "segment_tag": np.asarray(tags),
        "segment_type": np.asarray(types),
        "source_group": np.asarray(groups),
    }


def pack_split(arrays: Dict[str, np.ndarray], keys: List[str]) -> Dict[str, np.ndarray]:
    mask = np.asarray([tag in set(keys) for tag in arrays["segment_tag"]], dtype=bool)
    return {
        "X": arrays["X"][mask],
        "baseline_seq": arrays["baseline_seq"][mask],
        "stable_baseline_seq": arrays["stable_baseline_seq"][mask],
        "y_clean_seq": arrays["y_clean_seq"][mask],
        "baseline": arrays["baseline"][mask],
        "stable_baseline": arrays["stable_baseline"][mask],
        "y_clean": arrays["y_clean"][mask],
        "weight": arrays["weight"][mask],
        "segment_tag": arrays["segment_tag"][mask],
        "segment_type": arrays["segment_type"][mask],
        "source_group": arrays["source_group"][mask],
    }


def load_allowed_segment_tags(validation_summary_json: str, allowed_decisions: List[str]) -> Dict[str, str]:
    obj = json.loads(Path(validation_summary_json).read_text(encoding="utf-8"))
    allowed = {}
    for seg in obj.get("segments", []):
        decision = str(seg.get("decision", ""))
        if decision in set(allowed_decisions):
            allowed[str(seg["manual_tag"])] = decision
    return allowed


def main() -> None:
    ap = argparse.ArgumentParser(description="Build stage1 clean-only dataset from merged tracker full log")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--algorithm_latency_ms", type=float, default=0.0)
    ap.add_argument("--control_latency_ms", type=float, default=0.0)
    ap.add_argument("--actuation_latency_ms", type=float, default=0.0)
    args = ap.parse_args()
    if not check_feature_index_alignment():
        raise RuntimeError("feature index alignment check failed; aborting stage1 dataset build")

    cfg_dict = {}
    cfg_path = Path(args.config) if args.config else (ROOT / "configs" / "stage1_clean.yaml")
    if cfg_path.exists():
        cfg_dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    teacher_cfg = teacher_cfg_from_dict(cfg_dict.get("teacher"))
    prefilter_cfg = prefilter_cfg_from_dict(cfg_dict.get("prefilter"))
    data_selection_cfg = cfg_dict.get("data_selection", {}) or {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    ensure_required(df)
    df = ensure_segment_columns(df)
    allowed_decisions = list(data_selection_cfg.get("allowed_decisions", []))
    validation_summary_json = str(data_selection_cfg.get("validation_summary_json", "")).strip()
    selected_tag_decisions: Dict[str, str] = {}
    if validation_summary_json and allowed_decisions:
        selected_tag_decisions = load_allowed_segment_tags(validation_summary_json, allowed_decisions)
        if not selected_tag_decisions:
            raise RuntimeError("no segments left after applying validation-summary decision filter")
        df = df[df["segment_tag"].astype(str).isin(set(selected_tag_decisions.keys()))].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("all rows filtered out by validation-summary decision filter")
    df = build_segmented_pseudo_targets(df, teacher_cfg=teacher_cfg, prefilter_cfg=prefilter_cfg)
    ensure_all_feature_columns(df)

    segment_info = (
        df[["segment_tag", "segment_type", "source_group"]]
        .drop_duplicates()
        .sort_values(["segment_type", "source_group", "segment_tag"])
        .reset_index(drop=True)
    )
    split = build_stratified_segment_split(segment_info)
    arrays = build_all_windows(df, seq_len=args.seq_len, stride=args.stride)
    train_pack = pack_split(arrays, split["train"])
    val_pack = pack_split(arrays, split["val"])
    test_pack = pack_split(arrays, split["test"])

    stats: Stage1NormalizationStats = compute_feature_stats_from_train(train_pack["X"])
    X_train = apply_feature_stats(train_pack["X"], stats)
    X_val = apply_feature_stats(val_pack["X"], stats) if len(val_pack["X"]) else val_pack["X"]
    X_test = apply_feature_stats(test_pack["X"], stats) if len(test_pack["X"]) else test_pack["X"]

    np.savez_compressed(
        out_dir / "stage1_clean_dataset.npz",
        X_train=X_train,
        baseline_seq_train=train_pack["baseline_seq"],
        baseline_train=train_pack["baseline"],
        stable_baseline_seq_train=train_pack["stable_baseline_seq"],
        stable_baseline_train=train_pack["stable_baseline"],
        y_clean_seq_train=train_pack["y_clean_seq"],
        y_clean_train=train_pack["y_clean"],
        w_train=train_pack["weight"],
        X_val=X_val,
        baseline_seq_val=val_pack["baseline_seq"],
        baseline_val=val_pack["baseline"],
        stable_baseline_seq_val=val_pack["stable_baseline_seq"],
        stable_baseline_val=val_pack["stable_baseline"],
        y_clean_seq_val=val_pack["y_clean_seq"],
        y_clean_val=val_pack["y_clean"],
        w_val=val_pack["weight"],
        X_test=X_test,
        baseline_seq_test=test_pack["baseline_seq"],
        baseline_test=test_pack["baseline"],
        stable_baseline_seq_test=test_pack["stable_baseline_seq"],
        stable_baseline_test=test_pack["stable_baseline"],
        y_clean_seq_test=test_pack["y_clean_seq"],
        y_clean_test=test_pack["y_clean"],
        w_test=test_pack["weight"],
        segment_tag_train=train_pack["segment_tag"],
        segment_type_train=train_pack["segment_type"],
        source_group_train=train_pack["source_group"],
        segment_tag_val=val_pack["segment_tag"],
        segment_type_val=val_pack["segment_type"],
        source_group_val=val_pack["source_group"],
        segment_tag_test=test_pack["segment_tag"],
        segment_type_test=test_pack["segment_type"],
        source_group_test=test_pack["source_group"],
    )

    meta = {
        "seq_len": args.seq_len,
        "stride": args.stride,
        "feature_columns": ALL_FEATURE_COLUMNS,
        "input_csv": args.input_csv,
        "train_tags": split["train"],
        "val_tags": split["val"],
        "test_tags": split["test"],
        "train_source_groups": sorted({str(v) for v in train_pack["source_group"].tolist()}),
        "val_source_groups": sorted({str(v) for v in val_pack["source_group"].tolist()}),
        "test_source_groups": sorted({str(v) for v in test_pack["source_group"].tolist()}),
        "train_type_distribution": split_type_distribution(segment_info, split["train"]),
        "val_type_distribution": split_type_distribution(segment_info, split["val"]),
        "test_type_distribution": split_type_distribution(segment_info, split["test"]),
        "sample_count": int(arrays["X"].shape[0]),
        "split_counts": {
            "train": int(X_train.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "residual_baseline_source": "prefiltered_dx_raw_dy_raw_unscaled",
        "normalization": stats.to_json_dict(ALL_FEATURE_COLUMNS),
        "delay_placeholders_ms": {
            "algorithm_latency_ms": float(args.algorithm_latency_ms),
            "control_latency_ms": float(args.control_latency_ms),
            "actuation_latency_ms": float(args.actuation_latency_ms),
        },
        "teacher": teacher_cfg.to_json_dict(),
        "prefilter": prefilter_cfg.to_json_dict(),
        "data_selection": {
            "validation_summary_json": validation_summary_json,
            "allowed_decisions": allowed_decisions,
            "selected_tag_decisions": selected_tag_decisions,
        },
    }
    (out_dir / "stage1_clean_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "dataset": str(out_dir / "stage1_clean_dataset.npz"),
        "meta": str(out_dir / "stage1_clean_meta.json"),
        "sample_count": meta["sample_count"],
        "split_counts": meta["split_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
