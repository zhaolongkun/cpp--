from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.stage1_common import (
    apply_stage1_feature_transforms,
    ensure_segment_columns,
    parse_segment_tag_from_path,
    parse_segment_type,
)
from la_cspc_ornet.features import check_feature_index_alignment


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def _safe_sum(x: np.ndarray) -> int:
    if x.size == 0:
        return 0
    return int(np.sum(x))


def _compute_segment_stats(df: pd.DataFrame, raw_dt_ms: np.ndarray) -> Dict[str, float]:
    valid = (
        (df["lost_flag"].to_numpy(dtype=np.float64) < 0.5)
        & (df["det_conf"].to_numpy(dtype=np.float64) > 0.0)
    )
    lost_or_coasting = (
        (df["lost_flag"].to_numpy(dtype=np.float64) > 0.5)
        | (df["coast_count"].to_numpy(dtype=np.float64) > 0.0)
    )
    zoom_active = np.abs(df["zoom_delta"].to_numpy(dtype=np.float64)) > 1e-4
    switch_active = df["switch_score"].to_numpy(dtype=np.float64) >= 0.5
    turn_active = df["turn_score"].to_numpy(dtype=np.float64) >= 0.5
    dt_ms = np.maximum(raw_dt_ms.astype(np.float64), 1e-6)

    dx = df["dx_raw"].to_numpy(dtype=np.float64)
    dy = df["dy_raw"].to_numpy(dtype=np.float64)
    trend_dx = df["trend_dx"].to_numpy(dtype=np.float64)
    trend_dy = df["trend_dy"].to_numpy(dtype=np.float64)
    img_w = np.maximum(df["img_w"].to_numpy(dtype=np.float64), 1.0)
    img_h = np.maximum(df["img_h"].to_numpy(dtype=np.float64), 1.0)

    resid_norm = np.abs(dx - trend_dx) / img_w + np.abs(dy - trend_dy) / img_h
    jitter_strength = float(np.clip(np.median(resid_norm) / 0.02, 0.0, 1.0))

    vx = np.zeros_like(dx)
    vy = np.zeros_like(dy)
    if len(dx) > 1:
        vx[1:] = (dx[1:] - dx[:-1]) / dt_ms[1:]
        vy[1:] = (dy[1:] - dy[:-1]) / dt_ms[1:]
    flip_x = np.zeros_like(dx, dtype=np.float64)
    flip_y = np.zeros_like(dy, dtype=np.float64)
    if len(dx) > 2:
        flip_x[2:] = (np.sign(vx[2:]) * np.sign(vx[1:-1]) < 0).astype(np.float64)
        flip_y[2:] = (np.sign(vy[2:]) * np.sign(vy[1:-1]) < 0).astype(np.float64)
    sign_flip_rate = float(np.mean(np.maximum(flip_x, flip_y))) if len(dx) else 0.0

    recovery_transitions = 0
    if len(df) > 1:
        prev_lost = lost_or_coasting[:-1]
        now_valid = valid[1:]
        recovery_transitions = int(np.sum(prev_lost & now_valid))

    return {
        "frame_count": int(len(df)),
        "duration_s": float(np.sum(dt_ms) / 1000.0),
        "valid_target_ratio": _safe_mean(valid.astype(np.float64)),
        "lost_or_coasting_ratio": _safe_mean(lost_or_coasting.astype(np.float64)),
        "zoom_transition_ratio": _safe_mean(zoom_active.astype(np.float64)),
        "switch_active_ratio": _safe_mean(switch_active.astype(np.float64)),
        "turn_active_ratio": _safe_mean(turn_active.astype(np.float64)),
        "high_event_ratio": _safe_mean((df["event_score"].to_numpy(dtype=np.float64) >= 0.5).astype(np.float64)),
        "mean_det_conf": float(np.mean(df["det_conf"].to_numpy(dtype=np.float64))) if len(df) else 0.0,
        "mean_abs_zoom_delta": float(np.mean(np.abs(df["zoom_delta"].to_numpy(dtype=np.float64)))) if len(df) else 0.0,
        "mean_coast_count": float(np.mean(df["coast_count"].to_numpy(dtype=np.float64))) if len(df) else 0.0,
        "recovery_transitions": recovery_transitions,
        "sign_flip_rate": sign_flip_rate,
        "jitter_strength": jitter_strength,
    }


def _score_types(stats: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, Dict[str, bool]]]:
    valid = stats["valid_target_ratio"]
    lost = stats["lost_or_coasting_ratio"]
    zoom = min(stats["zoom_transition_ratio"] / 0.10, 1.0)
    turn = min(max(stats["turn_active_ratio"] / 0.20, stats["switch_active_ratio"] / 0.25, stats["sign_flip_rate"] / 0.10), 1.0)
    jitter = stats["jitter_strength"]
    recover = min(stats["lost_or_coasting_ratio"] / 0.15 + min(stats["recovery_transitions"], 2) / 2.0, 1.5)
    recover = min(recover / 1.5, 1.0)

    rules = {
        "stable": {
            "valid_high": valid >= 0.60,
            "lost_low": lost <= 0.10,
            "zoom_low": stats["zoom_transition_ratio"] <= 0.05,
            "turn_low": max(stats["turn_active_ratio"], stats["switch_active_ratio"]) <= 0.15,
        },
        "jitter": {
            "valid_ok": valid >= 0.50,
            "lost_low": lost <= 0.20,
            "zoom_low": stats["zoom_transition_ratio"] <= 0.10,
            "jitter_present": jitter >= 0.45,
        },
        "zoom": {
            "valid_ok": valid >= 0.40,
            "zoom_present": stats["zoom_transition_ratio"] >= 0.08 or stats["mean_abs_zoom_delta"] >= 0.03,
        },
        "recover": {
            "lost_present": stats["lost_or_coasting_ratio"] >= 0.08,
            "recovery_present": stats["recovery_transitions"] >= 1,
            "valid_after_recovery": valid >= 0.20,
        },
        "turn": {
            "valid_ok": valid >= 0.40,
            "turn_present": max(stats["turn_active_ratio"], stats["switch_active_ratio"]) >= 0.18 or stats["sign_flip_rate"] >= 0.08,
            "zoom_not_dominant": stats["zoom_transition_ratio"] <= 0.20,
        },
    }

    scores = {
        # Stable should lose quickly once dynamic event indicators become strong.
        "stable": 1.20 * valid + 0.60 * (1.0 - lost) + 0.25 * (1.0 - zoom) + 0.20 * (1.0 - turn) + 0.15 * (1.0 - jitter),
        # Jitter prefers visible target + local residual energy, but should lose to true turn/recover.
        "jitter": 0.85 * valid + 1.35 * jitter + 0.20 * stats["switch_active_ratio"] + 0.15 * (1.0 - zoom) - 0.35 * turn - 0.25 * lost,
        # Zoom should dominate only when zoom proxy is clearly active.
        "zoom": 0.70 * valid + 1.80 * zoom + 0.10 * jitter - 0.15 * lost,
        # Recover needs both lost/coast and actual recovery transition.
        "recover": 0.45 * valid + 1.60 * recover + 0.25 * lost + 0.10 * (1.0 - zoom),
        # Turn should dominate when switch/turn/sign-flip are strong and zoom is not dominant.
        "turn": 0.90 * valid + 1.70 * turn + 0.15 * stats["switch_active_ratio"] + 0.15 * (1.0 - zoom) - 0.10 * lost,
    }
    return scores, rules


def _credibility(manual_type: str, predicted_type: str, scores: Dict[str, float], rules: Dict[str, Dict[str, bool]], stats: Dict[str, float]) -> Tuple[str, bool, List[str]]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
    margin = best_score - second_score
    reasons: List[str] = []

    if stats["valid_target_ratio"] < 0.30:
        reasons.append("valid_target_ratio too low")
    if stats["duration_s"] < 20.0:
        reasons.append("segment too short")
    if manual_type == "unknown":
        reasons.append("manual tag prefix unknown")

    if manual_type != predicted_type:
        reasons.append(f"predicted_type={predicted_type} differs from manual_type={manual_type}")

    manual_rules = rules.get(manual_type, {})
    passed_manual_rules = sum(bool(v) for v in manual_rules.values())
    total_manual_rules = max(len(manual_rules), 1)
    rule_ratio = passed_manual_rules / total_manual_rules

    # If the manual tag's own rule set passes strongly, keep the segment for review
    # instead of throwing it away on a close class mismatch. This is important for
    # recover/zoom/turn segments whose dynamic signatures can overlap.
    if manual_type != predicted_type and rule_ratio >= 0.67 and stats["valid_target_ratio"] >= 0.20 and margin < 0.45:
        reasons.append("manual tag rules pass strongly despite class mismatch")
        return "medium", False, reasons

    if manual_type == predicted_type and margin >= 0.20 and rule_ratio >= 0.75 and stats["valid_target_ratio"] >= 0.30:
        return "high", False, reasons
    if manual_type == predicted_type and rule_ratio >= 0.50 and stats["valid_target_ratio"] >= 0.20:
        return "medium", False, reasons
    if margin < 0.10 and stats["valid_target_ratio"] >= 0.30:
        reasons.append("class margin too small")
        return "medium", False, reasons
    reasons.append("tag confidence low")
    return "low", True, reasons


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate whether a collected tracker log matches its manual segment tag")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--expected_tag", default="")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()
    if not check_feature_index_alignment():
        raise RuntimeError("feature index alignment check failed; aborting segment tag validation")

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    raw_dt_ms = df["dt_ms"].to_numpy(dtype=np.float64) if "dt_ms" in df.columns else np.ones(len(df), dtype=np.float64) * 33.0
    df = ensure_segment_columns(df)
    df = apply_stage1_feature_transforms(df)

    expected_tag = args.expected_tag or parse_segment_tag_from_path(csv_path)
    manual_type = parse_segment_type(expected_tag)
    stats = _compute_segment_stats(df, raw_dt_ms)
    scores, rules = _score_types(stats)
    predicted_type = max(scores.items(), key=lambda kv: kv[1])[0]
    trust, reshoot, reasons = _credibility(manual_type, predicted_type, scores, rules, stats)

    result = {
        "input_csv": str(csv_path),
        "manual_tag": expected_tag,
        "manual_type": manual_type,
        "predicted_type": predicted_type,
        "tag_trust": trust,
        "reshoot_recommended": reshoot,
        "stats": stats,
        "score_breakdown": {k: float(v) for k, v in scores.items()},
        "rule_checks": rules,
        "reasons": reasons,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
