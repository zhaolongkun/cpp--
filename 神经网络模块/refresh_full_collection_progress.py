#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh full-profile collection progress table from raw logs and batch augmentation outputs.")
    p.add_argument("--summary_csv", required=True, help="Batch summary CSV from build_pseudo_expert_batch.py")
    p.add_argument("--output_csv", required=True, help="Progress CSV output path")
    p.add_argument("--seq_len", type=int, default=8, help="Sequence length used by training")
    p.add_argument("--hold_ms", type=float, default=200.0, help="Hold-time criterion for recovery")
    return p.parse_args()


def contiguous_lengths(mask: np.ndarray) -> List[int]:
    lengths: List[int] = []
    current = 0
    for v in mask:
        if v:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def normalize_scenario(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("normal", "stable"):
        return "stable"
    if s == "maneuver":
        return "maneuver"
    if s in ("loss", "recovery"):
        return "recovery"
    return s or "unknown"


def resolve_path(project_root: Path, path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "反无" and parts[1] == "cpp智能控制":
        repo_root = project_root.parent.parent
        return (repo_root / p).resolve()
    return (project_root / p).resolve()


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in [
        "timestamp_ms",
        "frame_id",
        "det_count",
        "track_count",
        "controlled_id",
        "lost_flag",
        "cmd_base_x",
        "cmd_base_y",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "note" in df.columns:
        df["note"] = df["note"].fillna("").astype(str).str.replace("\x00", "", regex=False).str.strip()
    if "run_id" in df.columns:
        df["run_id"] = df["run_id"].fillna("").astype(str).str.replace("\x00", "", regex=False).str.strip()
    return df.sort_values(["timestamp_ms", "frame_id"], kind="mergesort").reset_index(drop=True)


def infer_run_id(df: pd.DataFrame) -> str:
    if "run_id" in df.columns:
        vals = df["run_id"][df["run_id"].astype(str).str.strip() != ""]
        if not vals.empty:
            return str(vals.iloc[0])
    return ""


def detect_recovery(df: pd.DataFrame, hold_ms: float) -> Tuple[bool, bool]:
    if df.empty or "note" not in df.columns or "timestamp_ms" not in df.columns:
        return False, False
    notes = df["note"].tolist()
    ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(df)
    i = 0
    has_recovery = False
    meets_hold = False
    while i < n:
        if notes[i] not in ("lost", "coasting"):
            i += 1
            continue
        start = i
        while i < n and notes[i] in ("lost", "coasting"):
            i += 1
        lost_end = i - 1
        tracked_start = i
        while i < n and notes[i] == "tracked":
            i += 1
        tracked_end = i - 1
        if tracked_start < n and tracked_end >= tracked_start:
            has_recovery = True
            hold = float(ts[tracked_end] - ts[tracked_start]) if np.isfinite(ts[tracked_end]) and np.isfinite(ts[tracked_start]) else 0.0
            if hold >= hold_ms:
                meets_hold = True
                break
    return has_recovery, meets_hold


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    if summary.empty:
        raise RuntimeError("summary_csv is empty")
    project_root = Path(__file__).resolve().parent.parent

    rows: List[Dict[str, object]] = []
    for source_csv, group in summary.groupby("source_csv", sort=True):
        src_path = resolve_path(project_root, str(source_csv))
        raw_df = load_log(src_path)

        run_id = infer_run_id(raw_df)
        scenario = normalize_scenario(str(group["scenario"].iloc[0]) if "scenario" in group.columns else "unknown")
        row_count = int(len(raw_df))
        has_target_positive = False
        if not raw_df.empty:
            has_target_positive = bool(
                (
                    (raw_df["track_count"] > 0)
                    & (raw_df["controlled_id"] != -1)
                    & (raw_df["lost_flag"] == 0)
                    & ((raw_df["cmd_base_x"].abs() + raw_df["cmd_base_y"].abs()) > 1e-9)
                ).any()
            )
        has_recovery_event, recovery_meets_hold = detect_recovery(raw_df, args.hold_ms)

        best_variant = ""
        best_valid = -1
        best_usable = -1
        per_variant_valid: Dict[str, int] = {}
        per_variant_usable: Dict[str, int] = {}
        for _, vrow in group.iterrows():
            variant = str(vrow["variant"])
            aug_path = resolve_path(project_root, str(vrow["augmented_csv"]))
            if not aug_path.exists():
                continue
            aug_df = pd.read_csv(aug_path)
            valid_mask = (aug_df["pseudo_expert_valid"] == 1).to_numpy(dtype=bool) if "pseudo_expert_valid" in aug_df.columns else np.zeros(len(aug_df), dtype=bool)
            valid_rows = int(valid_mask.sum())
            usable = int(sum(max(0, l - args.seq_len + 1) for l in contiguous_lengths(valid_mask)))
            per_variant_valid[variant] = valid_rows
            per_variant_usable[variant] = usable
            if (valid_rows > best_valid) or (valid_rows == best_valid and usable > best_usable):
                best_variant = variant
                best_valid = valid_rows
                best_usable = usable

        note_parts: List[str] = []
        if has_target_positive:
            note_parts.append("target_positive")
        else:
            note_parts.append("all_lost_or_no_control")
        if has_recovery_event:
            note_parts.append("recovery_event")
        if recovery_meets_hold:
            note_parts.append("recovery_hold")
        if best_valid <= 0:
            note_parts.append("no_valid_supervision")

        try:
            log_path_for_csv = str(src_path.resolve().relative_to(project_root))
        except ValueError:
            log_path_for_csv = str(src_path)

        rows.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "log_path": log_path_for_csv,
                "rows": row_count,
                "best_variant": best_variant,
                "pseudo_valid_rows": max(0, best_valid),
                "usable_sequences": max(0, best_usable),
                "has_target_positive": int(has_target_positive),
                "has_recovery_event": int(has_recovery_event),
                "recovery_meets_hold": int(recovery_meets_hold),
                "pseudo_valid_rows_future_smoothed_base": per_variant_valid.get("future_smoothed_base", 0),
                "pseudo_valid_rows_future_error_aware": per_variant_valid.get("future_error_aware", 0),
                "usable_sequences_future_smoothed_base": per_variant_usable.get("future_smoothed_base", 0),
                "usable_sequences_future_error_aware": per_variant_usable.get("future_error_aware", 0),
                "notes": ",".join(note_parts),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["scenario", "log_path"], kind="mergesort")
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        {
            "output_csv": str(out_path),
            "num_runs": int(len(out_df)),
            "target_positive_runs": int(out_df["has_target_positive"].sum()) if len(out_df) else 0,
            "recovery_runs": int(out_df["has_recovery_event"].sum()) if len(out_df) else 0,
            "recovery_meets_hold_runs": int(out_df["recovery_meets_hold"].sum()) if len(out_df) else 0,
            "total_pseudo_valid_rows": int(out_df["pseudo_valid_rows"].sum()) if len(out_df) else 0,
            "total_usable_sequences": int(out_df["usable_sequences"].sum()) if len(out_df) else 0,
        }
    )


if __name__ == "__main__":
    main()
