#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
按场景汇总 Table VII。

输入:
    - run_level_metrics.csv
    - event_level_metrics.csv

输出:
    - table_vii_summary.csv

固定口径:
    - normal: run-level mean ± std
    - loss: event-level mean ± std
    - maneuver: event-level mean ± std
    - 默认支持 Base only 和 Full method
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


RUN_REQUIRED = ["run_id", "method", "rmse", "mae", "tv_cmd", "safety_trigger_ratio_pct"]
EVENT_REQUIRED = [
    "run_id",
    "method",
    "scenario",
    "rmse_event",
    "mae_event",
    "overshoot",
    "settling_time_ms",
    "tv_cmd_event",
    "safety_trigger_ratio_pct",
    "recovery_time_ms",
    "event_valid",
]

DEFAULT_METHODS = [
    "Base only",
    "Full method",
]

DEFAULT_SCENARIOS = [
    "normal",
    "loss",
    "maneuver",
]

BUILTIN_METHOD_ALIASES = {
    "baseonly": "Base only",
    "baselineonly": "Base only",
    "base": "Base only",
    "fullmethod": "Full method",
    "full": "Full method",
    "r2slavslite": "Full method",
}

for _label in DEFAULT_METHODS:
    BUILTIN_METHOD_ALIASES[re.sub(r"[^a-z0-9]+", "", _label.lower())] = _label


def canonicalize_name(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def parse_csv_list(spec: str) -> List[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Table VII by scenario.")
    parser.add_argument("--run_csv", type=str, required=True, help="run_level_metrics.csv")
    parser.add_argument("--event_csv", type=str, required=True, help="event_level_metrics.csv")
    parser.add_argument("--output_csv", type=str, default="outputs/table_vii_summary.csv", help="输出 summary CSV")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="逗号分隔的方法名列表。默认 Base only 和 Full method。",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="逗号分隔的场景列表。默认 normal,loss,maneuver。",
    )
    parser.add_argument(
        "--method_map",
        type=str,
        default="",
        help="可选的方法名映射, 例如: base_only=Base only;full_method=Full method",
    )
    parser.add_argument("--precision", type=int, default=2, help="fmt 保留小数位数。")
    return parser.parse_args()


def parse_method_map(spec: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    spec = spec.strip()
    if not spec:
        return mapping
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"非法 method_map 项: {item}")
        raw, display = item.split("=", 1)
        raw = raw.strip()
        display = display.strip()
        if not raw or not display:
            raise ValueError(f"非法 method_map 项: {item}")
        mapping[raw] = display
    return mapping


def build_method_aliases(method_map: Dict[str, str]) -> Dict[str, str]:
    aliases = dict(BUILTIN_METHOD_ALIASES)
    for raw, display in method_map.items():
        aliases[canonicalize_name(raw)] = display
        aliases[canonicalize_name(display)] = display
    return aliases


def normalize_method_name(value: object, aliases: Dict[str, str]) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return text
    return aliases.get(canonicalize_name(text), text)


def normalize_method_list(methods: List[str], aliases: Dict[str, str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for item in methods:
        name = normalize_method_name(item, aliases)
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def apply_method_aliases(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["method"] = out["method"].map(lambda x: normalize_method_name(x, aliases))
    return out


def ensure_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要列: {missing}")


def read_csv_checked(csv_path: Path, name: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"{name} 不存在: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"读取 {name} 失败: {exc}") from exc


def clean_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scenario"] = out["scenario"].astype(str).str.strip().str.lower()
    return out


def fmt_mean_std(mean: float, std: float, precision: int) -> str:
    if not np.isfinite(mean):
        return "N/A"
    if not np.isfinite(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def aggregate_series(values: pd.Series) -> Dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.shape[0])
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "n": 0}
    if n == 1:
        return {"mean": float(vals.mean()), "std": 0.0, "n": 1}
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)), "n": n}


def empty_stats() -> Dict[str, float]:
    return {"mean": np.nan, "std": np.nan, "n": 0}


def parse_scenarios(spec: str) -> List[str]:
    scenarios = [item.strip().lower() for item in str(spec).split(",") if item.strip()]
    if not scenarios:
        raise ValueError("--scenarios 不能为空")
    valid = {"normal", "loss", "maneuver"}
    bad = [item for item in scenarios if item not in valid]
    if bad:
        raise ValueError(f"--scenarios 存在非法 scenario: {bad}")
    deduped: List[str] = []
    seen = set()
    for item in scenarios:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def warn_missing_methods(methods: List[str], run_df: pd.DataFrame, event_df: pd.DataFrame) -> None:
    available = set(run_df["method"].dropna().astype(str)) | set(event_df["method"].dropna().astype(str))
    missing = [method for method in methods if method not in available]
    if missing:
        print(f"[Warn] 以下方法未在输入 CSV 中找到, 对应输出行将为 N/A: {missing}", file=sys.stderr)


def scenario_run_ids(event_df: pd.DataFrame, scenario: str) -> Set[str]:
    sub = event_df[event_df["scenario"] == scenario]
    return set(sub["run_id"].dropna().astype(str).str.strip())


def select_normal_runs(run_df: pd.DataFrame, event_df: pd.DataFrame, method: str) -> pd.DataFrame:
    sub = run_df[run_df["method"] == method].copy()
    if sub.empty:
        return sub

    count_col = "normal_event_count"
    if count_col in sub.columns:
        counts = pd.to_numeric(sub[count_col], errors="coerce").fillna(0)
        return sub[counts > 0].copy()

    run_ids = scenario_run_ids(event_df, "normal")
    if not run_ids:
        return sub.iloc[0:0].copy()
    return sub[sub["run_id"].astype(str).isin(run_ids)].copy()


def select_event_rows(event_df: pd.DataFrame, method: str, scenario: str) -> pd.DataFrame:
    mask = (event_df["method"] == method) & (event_df["scenario"] == scenario) & (event_df["event_valid"] == 1)
    return event_df.loc[mask].copy()


def build_normal_row(method: str, run_sub: pd.DataFrame, precision: int) -> Dict[str, object]:
    rmse_stat = aggregate_series(run_sub["rmse"]) if not run_sub.empty else empty_stats()
    mae_stat = aggregate_series(run_sub["mae"]) if not run_sub.empty else empty_stats()
    tv_stat = aggregate_series(run_sub["tv_cmd"]) if not run_sub.empty else empty_stats()
    safe_stat = aggregate_series(run_sub["safety_trigger_ratio_pct"]) if not run_sub.empty else empty_stats()

    return {
        "scenario": "normal",
        "method": method,
        "aggregation_level": "run",
        "data_source": "run_level_metrics.csv",
        "num_runs": int(run_sub["run_id"].astype(str).nunique()) if not run_sub.empty else 0,
        "num_events": 0,
        "rmse_mean": rmse_stat["mean"],
        "rmse_std": rmse_stat["std"],
        "rmse_fmt": fmt_mean_std(rmse_stat["mean"], rmse_stat["std"], precision),
        "mae_mean": mae_stat["mean"],
        "mae_std": mae_stat["std"],
        "mae_fmt": fmt_mean_std(mae_stat["mean"], mae_stat["std"], precision),
        "tv_cmd_mean": tv_stat["mean"],
        "tv_cmd_std": tv_stat["std"],
        "tv_cmd_fmt": fmt_mean_std(tv_stat["mean"], tv_stat["std"], precision),
        "safety_trigger_ratio_mean": safe_stat["mean"],
        "safety_trigger_ratio_std": safe_stat["std"],
        "safety_trigger_ratio_fmt": fmt_mean_std(safe_stat["mean"], safe_stat["std"], precision),
        "overshoot_mean": np.nan,
        "overshoot_std": np.nan,
        "overshoot_fmt": "N/A",
        "settling_time_mean": np.nan,
        "settling_time_std": np.nan,
        "settling_time_fmt": "N/A",
        "recovery_time_mean": np.nan,
        "recovery_time_std": np.nan,
        "recovery_time_fmt": "N/A",
    }


def build_event_row(method: str, scenario: str, event_sub: pd.DataFrame, precision: int) -> Dict[str, object]:
    rmse_stat = aggregate_series(event_sub["rmse_event"]) if not event_sub.empty else empty_stats()
    mae_stat = aggregate_series(event_sub["mae_event"]) if not event_sub.empty else empty_stats()
    tv_stat = aggregate_series(event_sub["tv_cmd_event"]) if not event_sub.empty else empty_stats()
    safe_stat = (
        aggregate_series(event_sub["safety_trigger_ratio_pct"]) if not event_sub.empty else empty_stats()
    )
    overshoot_stat = aggregate_series(event_sub["overshoot"]) if not event_sub.empty else empty_stats()
    settling_stat = aggregate_series(event_sub["settling_time_ms"]) if not event_sub.empty else empty_stats()

    if scenario == "loss":
        rec_sub = event_sub.copy()
        if "recovery_success" in rec_sub.columns:
            rec_sub = rec_sub[rec_sub["recovery_success"] == 1]
        recovery_stat = aggregate_series(rec_sub["recovery_time_ms"]) if not rec_sub.empty else empty_stats()
    else:
        recovery_stat = empty_stats()

    return {
        "scenario": scenario,
        "method": method,
        "aggregation_level": "event",
        "data_source": "event_level_metrics.csv",
        "num_runs": int(event_sub["run_id"].astype(str).nunique()) if not event_sub.empty else 0,
        "num_events": int(event_sub.shape[0]),
        "rmse_mean": rmse_stat["mean"],
        "rmse_std": rmse_stat["std"],
        "rmse_fmt": fmt_mean_std(rmse_stat["mean"], rmse_stat["std"], precision),
        "mae_mean": mae_stat["mean"],
        "mae_std": mae_stat["std"],
        "mae_fmt": fmt_mean_std(mae_stat["mean"], mae_stat["std"], precision),
        "tv_cmd_mean": tv_stat["mean"],
        "tv_cmd_std": tv_stat["std"],
        "tv_cmd_fmt": fmt_mean_std(tv_stat["mean"], tv_stat["std"], precision),
        "safety_trigger_ratio_mean": safe_stat["mean"],
        "safety_trigger_ratio_std": safe_stat["std"],
        "safety_trigger_ratio_fmt": fmt_mean_std(safe_stat["mean"], safe_stat["std"], precision),
        "overshoot_mean": overshoot_stat["mean"],
        "overshoot_std": overshoot_stat["std"],
        "overshoot_fmt": fmt_mean_std(overshoot_stat["mean"], overshoot_stat["std"], precision),
        "settling_time_mean": settling_stat["mean"],
        "settling_time_std": settling_stat["std"],
        "settling_time_fmt": fmt_mean_std(settling_stat["mean"], settling_stat["std"], precision),
        "recovery_time_mean": recovery_stat["mean"],
        "recovery_time_std": recovery_stat["std"],
        "recovery_time_fmt": fmt_mean_std(recovery_stat["mean"], recovery_stat["std"], precision),
    }


def main() -> int:
    args = parse_args()
    if args.precision < 0:
        print("[Error] precision 不能为负数。", file=sys.stderr)
        return 1

    try:
        method_map = parse_method_map(args.method_map)
        aliases = build_method_aliases(method_map)
        methods = normalize_method_list(parse_csv_list(args.methods), aliases)
        scenarios = parse_scenarios(args.scenarios)
        if not methods:
            raise ValueError("--methods 不能为空")

        run_csv = Path(args.run_csv).expanduser().resolve()
        event_csv = Path(args.event_csv).expanduser().resolve()
        output_csv = Path(args.output_csv).expanduser().resolve()

        run_df = read_csv_checked(run_csv, "run_level_metrics.csv")
        event_df = read_csv_checked(event_csv, "event_level_metrics.csv")

        ensure_columns(run_df, RUN_REQUIRED, "run_level_metrics.csv")
        ensure_columns(event_df, EVENT_REQUIRED, "event_level_metrics.csv")

        run_df = apply_method_aliases(run_df, aliases)
        event_df = apply_method_aliases(event_df, aliases)

        run_df = clean_numeric(
            run_df,
            ["rmse", "mae", "tv_cmd", "safety_trigger_ratio_pct", "run_valid", "normal_event_count"],
        )
        event_df = clean_numeric(
            event_df,
            [
                "rmse_event",
                "mae_event",
                "overshoot",
                "settling_time_ms",
                "tv_cmd_event",
                "safety_trigger_ratio_pct",
                "recovery_time_ms",
                "event_valid",
                "recovery_success",
            ],
        )
        event_df = normalize_scenarios(event_df)

        if "run_valid" in run_df.columns:
            run_df = run_df[(run_df["run_valid"] == 1) | run_df["run_valid"].isna()].copy()

        warn_missing_methods(methods, run_df, event_df)

        rows = []
        for scenario in scenarios:
            for method in methods:
                if scenario == "normal":
                    run_sub = select_normal_runs(run_df, event_df, method)
                    rows.append(build_normal_row(method, run_sub, args.precision))
                else:
                    event_sub = select_event_rows(event_df, method, scenario)
                    rows.append(build_event_row(method, scenario, event_sub, args.precision))

        out_df = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[Done] Table VII summary saved to: {output_csv}")
        return 0
    except Exception as exc:
        print(f"[Error] Table VII 汇总失败: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
