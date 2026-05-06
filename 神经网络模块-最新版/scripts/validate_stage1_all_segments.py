from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from la_cspc_ornet.stage1_common import parse_segment_tag_from_path
from la_cspc_ornet.features import check_feature_index_alignment
from validate_stage1_segment_tag import (
    _compute_segment_stats,
    _credibility,
    _score_types,
    apply_stage1_feature_transforms,
    ensure_segment_columns,
    parse_segment_type,
)

REQUIRED_STAGE1_TYPES = ("stable", "jitter", "recover", "turn")
ALL_STAGE1_TYPES = ("stable", "jitter", "zoom", "recover", "turn")


def _decision(tag_trust: str, reshoot_recommended: bool) -> str:
    if reshoot_recommended or tag_trust == "low":
        return "reshoot"
    if tag_trust == "medium":
        return "keep_but_review"
    return "keep"


def _type_counts(rows: List[Dict[str, object]], usable_only: bool) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in ALL_STAGE1_TYPES}
    for row in rows:
        if usable_only and row["decision"] == "reshoot":
            continue
        t = str(row["manual_type"])
        if t in counts:
            counts[t] += 1
    return counts


def _decision_counts(rows: List[Dict[str, object]]) -> Dict[str, int]:
    out = {"keep": 0, "keep_but_review": 0, "reshoot": 0}
    for row in rows:
        out[str(row["decision"])] += 1
    return out


def _ready_for_merge(usable_counts_by_type: Dict[str, int]) -> bool:
    return all(usable_counts_by_type.get(k, 0) >= 1 for k in REQUIRED_STAGE1_TYPES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch validate all stage1 raw log tags")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()
    if not check_feature_index_alignment():
        raise RuntimeError("feature index alignment check failed; aborting batch tag validation")

    input_dir = Path(args.input_dir)
    rows: List[Dict[str, object]] = []

    for csv_path in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        raw_dt_ms = df["dt_ms"].to_numpy(dtype=float) if "dt_ms" in df.columns else None
        df = ensure_segment_columns(df)
        df = apply_stage1_feature_transforms(df)

        manual_tag = parse_segment_tag_from_path(csv_path)
        manual_type = parse_segment_type(manual_tag)
        stats = _compute_segment_stats(df, raw_dt_ms if raw_dt_ms is not None else pd.Series([33.0] * len(df)).to_numpy())
        scores, rules = _score_types(stats)
        predicted_type = max(scores.items(), key=lambda kv: kv[1])[0]
        tag_trust, reshoot_recommended, reasons = _credibility(manual_type, predicted_type, scores, rules, stats)
        decision = _decision(tag_trust, reshoot_recommended)

        rows.append(
            {
                "file_name": csv_path.name,
                "manual_tag": manual_tag,
                "manual_type": manual_type,
                "predicted_type": predicted_type,
                "tag_trust": tag_trust,
                "reshoot_recommended": bool(reshoot_recommended),
                "decision": decision,
                "valid_target_ratio": float(stats["valid_target_ratio"]),
                "lost_or_coasting_ratio": float(stats["lost_or_coasting_ratio"]),
                "zoom_transition_ratio": float(stats["zoom_transition_ratio"]),
                "switch_active_ratio": float(stats["switch_active_ratio"]),
                "turn_active_ratio": float(stats["turn_active_ratio"]),
                "duration_s": float(stats["duration_s"]),
                "reasons": reasons,
            }
        )

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "input_dir": str(input_dir),
        "file_count": int(len(rows)),
        "segments": rows,
        "required_stage1_types": list(REQUIRED_STAGE1_TYPES),
        "usable_counts_by_type": _type_counts(rows, usable_only=True),
        "all_counts_by_type": _type_counts(rows, usable_only=False),
        "decision_counts": _decision_counts(rows),
        "ready_for_merge": _ready_for_merge(_type_counts(rows, usable_only=True)),
    }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
