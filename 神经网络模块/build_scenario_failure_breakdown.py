#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose why maneuver/recovery scenarios currently produce zero valid supervision")
    p.add_argument("--summary_csv", required=True, help="Batch summary CSV from build_pseudo_expert_batch.py")
    p.add_argument("--output_csv", required=True, help="Failure breakdown CSV output")
    p.add_argument("--hold_ms", type=float, default=200.0, help="Hold-time criterion for recovery")
    return p.parse_args()


def normalize_scenario(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("normal", "stable"):
        return "stable"
    if s == "maneuver":
        return "maneuver"
    if s in ("loss", "recovery"):
        return "recovery"
    return s or "unknown"


def resolve(project_root: Path, path_str: str) -> Path:
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
    for col in ["timestamp_ms", "det_count", "track_count", "controlled_id", "det_conf", "lost_flag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "note" in df.columns:
        df["note"] = df["note"].fillna("").astype(str).str.replace("\x00", "", regex=False).str.strip()
    if "run_id" in df.columns:
        df["run_id"] = df["run_id"].fillna("").astype(str).str.replace("\x00", "", regex=False).str.strip()
    return df.sort_values(["timestamp_ms", "frame_id"], kind="mergesort").reset_index(drop=True)


def infer_run_id(df: pd.DataFrame, fallback: str) -> str:
    if "run_id" in df.columns:
        vals = df["run_id"][df["run_id"].astype(str).str.strip() != ""]
        if not vals.empty:
            return str(vals.iloc[0])
    return fallback


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
        while i < n and notes[i] in ("lost", "coasting"):
            i += 1
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


def classify_failure(
    scenario: str,
    has_detection: bool,
    has_tracking: bool,
    has_controlled_id: bool,
    has_pseudo_valid: bool,
    has_recovery_event: bool,
    recovery_meets_hold: bool,
) -> str:
    if has_pseudo_valid:
        return "success"
    if not has_detection:
        return "detector_no_target"
    if has_detection and not has_tracking:
        return "tracking_not_established"
    if has_tracking and not has_controlled_id:
        return "controlled_id_not_assigned"
    if scenario == "recovery":
        if has_recovery_event and not recovery_meets_hold:
            return "recovery_hold_not_met"
        if has_tracking and not has_pseudo_valid:
            return "tracked_but_no_valid_supervision"
        return "recovery_not_observed"
    if scenario == "maneuver":
        if has_tracking and not has_pseudo_valid:
            return "maneuver_no_valid_supervision_window"
    if has_tracking and not has_pseudo_valid:
        return "tracked_but_no_valid_supervision"
    return "unknown"


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    if summary.empty:
        raise RuntimeError("summary_csv is empty")
    project_root = Path(__file__).resolve().parent.parent

    rows: List[Dict[str, object]] = []
    for source_csv, group in summary.groupby("source_csv", sort=True):
        src_path = resolve(project_root, str(source_csv))
        raw_df = load_log(src_path)
        scenario = normalize_scenario(str(group["scenario"].iloc[0]))
        run_id = infer_run_id(raw_df, src_path.stem)
        has_detection = bool(((raw_df.get("det_count", pd.Series(dtype=float)).fillna(0) > 0) | (raw_df.get("det_conf", pd.Series(dtype=float)).fillna(0) > 0)).any())
        has_tracking = bool((raw_df.get("track_count", pd.Series(dtype=float)).fillna(0) > 0).any())
        has_controlled_id = bool((raw_df.get("controlled_id", pd.Series(dtype=float)).fillna(-1) != -1).any())
        has_pseudo_valid = bool(group["pseudo_expert_valid_rows"].max() > 0)
        has_recovery_event, recovery_meets_hold = detect_recovery(raw_df, args.hold_ms)
        failure_stage = classify_failure(
            scenario=scenario,
            has_detection=has_detection,
            has_tracking=has_tracking,
            has_controlled_id=has_controlled_id,
            has_pseudo_valid=has_pseudo_valid,
            has_recovery_event=has_recovery_event,
            recovery_meets_hold=recovery_meets_hold,
        )

        note_dist = ""
        if "note" in raw_df.columns and not raw_df.empty:
            vc = raw_df["note"].value_counts(dropna=False).to_dict()
            note_dist = ",".join(f"{k}:{v}" for k, v in vc.items())

        rows.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "has_detection": int(has_detection),
                "has_tracking": int(has_tracking),
                "has_controlled_id": int(has_controlled_id),
                "has_pseudo_valid": int(has_pseudo_valid),
                "has_recovery_event": int(has_recovery_event),
                "recovery_meets_hold": int(recovery_meets_hold),
                "failure_stage": failure_stage,
                "notes": note_dist,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["scenario", "run_id"], kind="mergesort")
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        {
            "output_csv": str(out_path),
            "rows": int(len(out_df)),
            "failure_stage_counts": out_df["failure_stage"].value_counts().to_dict(),
        }
    )


if __name__ == "__main__":
    main()
