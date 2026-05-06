#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
汇总主结果表 Table IV。

输入:
    - run_level_metrics.csv
    - event_level_metrics.csv

输出:
    - table_iv_summary.csv

默认口径:
    - run-level: RMSE, MAE, TV(cmd)
    - event-level: Overshoot, Settling Time, Recovery Time
    - Overshoot / Settling Time 默认汇总 loss + maneuver
    - Recovery Time 默认仅汇总 loss
    - 输出数值列 + mean ± std 格式列
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


RUN_REQUIRED = ["method", "rmse", "mae", "tv_cmd"]
EVENT_REQUIRED = ["method", "scenario", "overshoot", "settling_time_ms", "recovery_time_ms", "event_valid"]

DEFAULT_METHODS = [
    "Base only",
    "Base + Residual",
    "Base + Residual + Gate",
    "Full method",
]

BUILTIN_METHOD_ALIASES = {
    "baseonly": "Base only",
    "baselineonly": "Base only",
    "base": "Base only",
    "baseresidual": "Base + Residual",
    "residualonly": "Base + Residual",
    "residual": "Base + Residual",
    "baseresidualgate": "Base + Residual + Gate",
    "residualgate": "Base + Residual + Gate",
    "gateaware": "Base + Residual + Gate",
    "fullmethod": "Full method",
    "full": "Full method",
    "fullmodel": "Full method",
    "r2slavslite": "Full method",
}

for _label in DEFAULT_METHODS:
    BUILTIN_METHOD_ALIASES[re.sub(r"[^a-z0-9]+", "", _label.lower())] = _label


def canonicalize_name(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def parse_csv_list(spec: str) -> List[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Table IV from run-level and event-level metrics.")
    parser.add_argument("--run_csv", type=str, required=True, help="run_level_metrics.csv")
    parser.add_argument("--event_csv", type=str, required=True, help="event_level_metrics.csv")
    parser.add_argument("--output_csv", type=str, default="outputs/table_iv_summary.csv", help="输出 summary CSV")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="逗号分隔的方法名列表。默认使用论文主结果四个方法。",
    )
    parser.add_argument(
        "--method_map",
        type=str,
        default="",
        help="可选的方法名映射, 例如: base=Base only;full_method=Full method",
    )
    parser.add_argument(
        "--overshoot_scenarios",
        type=str,
        default="loss,maneuver",
        help="用于 Overshoot 聚合的 scenario 列表。",
    )
    parser.add_argument(
        "--settling_scenarios",
        type=str,
        default="loss,maneuver",
        help="用于 Settling Time 聚合的 scenario 列表。",
    )
    parser.add_argument(
        "--recovery_scenarios",
        type=str,
        default="loss",
        help="用于 Recovery Time 聚合的 scenario 列表。",
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


def parse_scenarios(spec: str, arg_name: str) -> List[str]:
    scenarios = [item.strip().lower() for item in str(spec).split(",") if item.strip()]
    if not scenarios:
        raise ValueError(f"{arg_name} 不能为空")
    valid = {"normal", "loss", "maneuver"}
    bad = [item for item in scenarios if item not in valid]
    if bad:
        raise ValueError(f"{arg_name} 存在非法 scenario: {bad}")
    return scenarios


def select_event_rows(event_df: pd.DataFrame, method: str, scenarios: List[str], metric_col: str) -> pd.DataFrame:
    mask = (event_df["method"] == method) & (event_df["event_valid"] == 1) & (event_df["scenario"].isin(scenarios))
    sub = event_df.loc[mask].copy()
    if metric_col == "recovery_time_ms" and "recovery_success" in sub.columns:
        sub = sub[sub["recovery_success"] == 1]
    return sub


def warn_missing_methods(methods: List[str], run_df: pd.DataFrame, event_df: pd.DataFrame) -> None:
    available = set(run_df["method"].dropna().astype(str)) | set(event_df["method"].dropna().astype(str))
    missing = [method for method in methods if method not in available]
    if missing:
        print(f"[Warn] 以下方法未在输入 CSV 中找到, 对应输出行将为 N/A: {missing}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if args.precision < 0:
        print("[Error] precision 不能为负数。", file=sys.stderr)
        return 1

    try:
        method_map = parse_method_map(args.method_map)
        aliases = build_method_aliases(method_map)
        methods = normalize_method_list(parse_csv_list(args.methods), aliases)
        overshoot_scenarios = parse_scenarios(args.overshoot_scenarios, "--overshoot_scenarios")
        settling_scenarios = parse_scenarios(args.settling_scenarios, "--settling_scenarios")
        recovery_scenarios = parse_scenarios(args.recovery_scenarios, "--recovery_scenarios")
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

        run_df = clean_numeric(run_df, ["rmse", "mae", "tv_cmd", "run_valid"])
        event_df = clean_numeric(
            event_df,
            ["overshoot", "settling_time_ms", "recovery_time_ms", "event_valid", "recovery_success"],
        )
        event_df = normalize_scenarios(event_df)

        if "run_valid" in run_df.columns:
            run_df = run_df[(run_df["run_valid"] == 1) | run_df["run_valid"].isna()].copy()

        warn_missing_methods(methods, run_df, event_df)

        rows = []
        for method in methods:
            run_sub = run_df[run_df["method"] == method].copy()

            rmse_stat = aggregate_series(run_sub["rmse"]) if not run_sub.empty else empty_stats()
            mae_stat = aggregate_series(run_sub["mae"]) if not run_sub.empty else empty_stats()
            tv_stat = aggregate_series(run_sub["tv_cmd"]) if not run_sub.empty else empty_stats()

            os_sub = select_event_rows(event_df, method, overshoot_scenarios, "overshoot")
            st_sub = select_event_rows(event_df, method, settling_scenarios, "settling_time_ms")
            rt_sub = select_event_rows(event_df, method, recovery_scenarios, "recovery_time_ms")

            os_stat = aggregate_series(os_sub["overshoot"]) if not os_sub.empty else empty_stats()
            st_stat = aggregate_series(st_sub["settling_time_ms"]) if not st_sub.empty else empty_stats()
            rt_stat = aggregate_series(rt_sub["recovery_time_ms"]) if not rt_sub.empty else empty_stats()

            rows.append(
                {
                    "method": method,
                    "num_runs": int(run_sub.shape[0]),
                    "num_overshoot_events": int(os_stat["n"]),
                    "num_settling_events": int(st_stat["n"]),
                    "num_recovery_events": int(rt_stat["n"]),
                    "rmse_mean": rmse_stat["mean"],
                    "rmse_std": rmse_stat["std"],
                    "rmse_fmt": fmt_mean_std(rmse_stat["mean"], rmse_stat["std"], args.precision),
                    "mae_mean": mae_stat["mean"],
                    "mae_std": mae_stat["std"],
                    "mae_fmt": fmt_mean_std(mae_stat["mean"], mae_stat["std"], args.precision),
                    "overshoot_mean": os_stat["mean"],
                    "overshoot_std": os_stat["std"],
                    "overshoot_fmt": fmt_mean_std(os_stat["mean"], os_stat["std"], args.precision),
                    "settling_time_mean": st_stat["mean"],
                    "settling_time_std": st_stat["std"],
                    "settling_time_fmt": fmt_mean_std(st_stat["mean"], st_stat["std"], args.precision),
                    "tv_cmd_mean": tv_stat["mean"],
                    "tv_cmd_std": tv_stat["std"],
                    "tv_cmd_fmt": fmt_mean_std(tv_stat["mean"], tv_stat["std"], args.precision),
                    "recovery_time_mean": rt_stat["mean"],
                    "recovery_time_std": rt_stat["std"],
                    "recovery_time_fmt": fmt_mean_std(rt_stat["mean"], rt_stat["std"], args.precision),
                }
            )

        out_df = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[Done] Table IV summary saved to: {output_csv}")
        return 0
    except Exception as exc:
        print(f"[Error] Table IV 汇总失败: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
