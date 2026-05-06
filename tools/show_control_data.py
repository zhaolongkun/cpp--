#!/usr/bin/env python3
import argparse
import csv
import math
import os
from typing import Dict, List


def to_float(v: str) -> float:
    if v is None:
        return 0.0
    s = str(v).strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def quantile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * p
    i = int(math.floor(k))
    j = min(i + 1, len(arr) - 1)
    t = k - i
    return arr[i] * (1.0 - t) + arr[j] * t


def stat_line(name: str, values: List[float]) -> str:
    if not values:
        return f"{name}: empty"
    abs_vals = [abs(v) for v in values]
    return (
        f"{name}: min={min(values):.6f} max={max(values):.6f} "
        f"mean={sum(values)/len(values):.6f} "
        f"p95_abs={quantile(abs_vals, 0.95):.6f} "
        f"p99_abs={quantile(abs_vals, 0.99):.6f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Show control-data statistics from tracker_log.csv before enabling hardware output."
    )
    ap.add_argument("--csv", default="logs/tracker_log.csv", help="Path to tracker_log.csv")
    ap.add_argument("--tail", type=int, default=10, help="Print last N non-zero command rows")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"[error] csv not found: {csv_path}")
        return 1

    with open(csv_path, "r", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))

    print(f"using_log={csv_path}")
    print(f"rows={len(rows)}")
    if not rows:
        return 0

    cmd_x = [to_float(r.get("cmd_x")) for r in rows]
    cmd_y = [to_float(r.get("cmd_y")) for r in rows]
    x_error = [to_float(r.get("x_error")) for r in rows]
    y_error = [to_float(r.get("y_error")) for r in rows]

    nonzero_idx = [i for i, (x, y) in enumerate(zip(cmd_x, cmd_y)) if abs(x) > 1e-9 or abs(y) > 1e-9]
    print(f"nonzero_cmd_rows={len(nonzero_idx)}")

    if nonzero_idx:
        i = nonzero_idx[0]
        r = rows[i]
        print(
            "first_nonzero "
            f"idx={i} frame={r.get('frame_seq','')} "
            f"cmd=({cmd_x[i]:.6f},{cmd_y[i]:.6f}) "
            f"err=({x_error[i]:.6f},{y_error[i]:.6f}) "
            f"note={r.get('note','')}"
        )

    print(stat_line("cmd_x", cmd_x))
    print(stat_line("cmd_y", cmd_y))
    print(stat_line("x_error", x_error))
    print(stat_line("y_error", y_error))

    print("last_nonzero_cmd_rows:")
    tail = max(0, args.tail)
    for i in nonzero_idx[-tail:]:
        r = rows[i]
        print(
            f"idx={i} frame={r.get('frame_seq','')} "
            f"cmd=({cmd_x[i]:.6f},{cmd_y[i]:.6f}) "
            f"err=({x_error[i]:.6f},{y_error[i]:.6f}) "
            f"note={r.get('note','')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
