#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract high-quality positive segments from tracker_log.csv")
    p.add_argument("--input_csv", required=True, help="Path to brief-profile tracker_log CSV")
    p.add_argument("--output_csv", required=True, help="Output segment summary CSV")
    p.add_argument("--min_duration_ms", type=float, default=1000.0, help="Paper/training preference threshold")
    p.add_argument("--min_cmd_mag", type=float, default=1e-3, help="Minimum baseline command magnitude")
    return p.parse_args()


def to_int(row: Dict[str, str], key: str) -> int:
    try:
        return int(float((row.get(key, "") or "0").strip()))
    except Exception:
        return 0


def to_float(row: Dict[str, str], key: str) -> float:
    try:
        return float((row.get(key, "") or "0").strip())
    except Exception:
        return 0.0


def cmd_mag(row: Dict[str, str]) -> float:
    x = to_float(row, "cmd_base_x")
    y = to_float(row, "cmd_base_y")
    return math.hypot(x, y)


def is_positive_frame(row: Dict[str, str], min_cmd_mag: float) -> bool:
    return (
        to_int(row, "track_count") > 0
        and to_int(row, "controlled_id") != -1
        and to_int(row, "lost_flag") == 0
        and cmd_mag(row) > min_cmd_mag
        and (row.get("note", "") or "").strip() == "tracked"
    )


def classify_segment(seg: Dict[str, float], min_duration_ms: float) -> str:
    duration_ms = seg["duration_ms"]
    mean_det_conf = seg["mean_det_conf"]
    mean_abs_dx = seg["mean_abs_dx"]
    mean_abs_dy = seg["mean_abs_dy"]
    mean_cmd_base = seg["mean_cmd_base"]

    stable_centered = mean_abs_dx <= 120.0 and mean_abs_dy <= 90.0
    confident = mean_det_conf >= 0.65
    command_clear = mean_cmd_base >= 1.0

    if duration_ms >= min_duration_ms and confident and stable_centered and command_clear:
        return "paper_case"
    if duration_ms >= min_duration_ms and mean_det_conf >= 0.50:
        return "future_training_candidate"
    return "hard_case"


def summarize_segment(segment_id: int, rows: List[Dict[str, str]], min_duration_ms: float) -> Dict[str, object]:
    start_frame = to_int(rows[0], "frame_id")
    end_frame = to_int(rows[-1], "frame_id")
    timestamps = [to_float(r, "timestamp_ms") for r in rows]
    dts = [to_float(r, "dt_ms") for r in rows]
    det_confs = [to_float(r, "det_conf") for r in rows]
    dxs = [abs(to_float(r, "dx_hat")) for r in rows]
    dys = [abs(to_float(r, "dy_hat")) for r in rows]
    cmds = [cmd_mag(r) for r in rows]
    losts = [to_int(r, "lost_flag") for r in rows]
    tracked = [1 if (r.get("note", "") or "").strip() == "tracked" else 0 for r in rows]
    note_mode = Counter((r.get("note", "") or "").strip() for r in rows).most_common(1)[0][0]

    duration_ms = max(0.0, timestamps[-1] - timestamps[0])
    if len(rows) == 1:
        duration_ms = dts[0]

    seg = {
        "segment_id": f"seg_{segment_id:03d}",
        "start_frame": start_frame,
        "end_frame": end_frame,
        "length": len(rows),
        "duration_ms": round(duration_ms, 3),
        "mean_det_conf": round(sum(det_confs) / len(det_confs), 6),
        "mean_bbox_area": "",
        "mean_abs_dx": round(sum(dxs) / len(dxs), 6),
        "mean_abs_dy": round(sum(dys) / len(dys), 6),
        "mean_cmd_base": round(sum(cmds) / len(cmds), 6),
        "tracked_ratio": round(sum(tracked) / len(tracked), 6),
        "lost_zero_ratio": round(sum(1 for v in losts if v == 0) / len(losts), 6),
        "note_mode": note_mode,
    }
    seg["recommended_use"] = classify_segment(seg, min_duration_ms)
    return seg


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    segments: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    prev_frame = None

    for row in rows:
        frame_id = to_int(row, "frame_id")
        positive = is_positive_frame(row, args.min_cmd_mag)
        contiguous = prev_frame is not None and frame_id == prev_frame + 1

        if positive:
            if current and contiguous:
                current.append(row)
            else:
                if current:
                    segments.append(current)
                current = [row]
        else:
            if current:
                segments.append(current)
                current = []
        prev_frame = frame_id

    if current:
        segments.append(current)

    summaries = [summarize_segment(i + 1, seg, args.min_duration_ms) for i, seg in enumerate(segments)]
    summaries.sort(key=lambda x: (-x["duration_ms"], -x["mean_det_conf"], -x["mean_cmd_base"], x["start_frame"]))

    fieldnames = [
        "segment_id",
        "start_frame",
        "end_frame",
        "length",
        "duration_ms",
        "mean_det_conf",
        "mean_bbox_area",
        "mean_abs_dx",
        "mean_abs_dy",
        "mean_cmd_base",
        "tracked_ratio",
        "lost_zero_ratio",
        "note_mode",
        "recommended_use",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    counts = Counter(s["recommended_use"] for s in summaries)
    print(
        {
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "num_segments": len(summaries),
            "paper_case": counts.get("paper_case", 0),
            "future_training_candidate": counts.get("future_training_candidate", 0),
            "hard_case": counts.get("hard_case", 0),
        }
    )


if __name__ == "__main__":
    main()
