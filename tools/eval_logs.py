#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from pathlib import Path


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
      return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def rms(vals):
    if not vals:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def main():
    if len(sys.argv) < 2:
        print("Usage: eval_logs.py <logs/tracker_log.csv>")
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Log file not found: {path}")
        return 1

    dxs, dys, errs = [], [], []
    cmd_xs, cmd_ys = [], []
    controlled_ids = []
    coast_counts = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dx = float(row["dx_hat"])
                dy = float(row["dy_hat"])
                cmd_x = float(row["cmd_x"])
                cmd_y = float(row["cmd_y"])
                cid = int(row["controlled_id"])
                coast = int(row["coast_count"])
            except Exception:
                continue

            dxs.append(dx)
            dys.append(dy)
            errs.append(math.hypot(dx, dy))
            cmd_xs.append(cmd_x)
            cmd_ys.append(cmd_y)
            controlled_ids.append(cid)
            coast_counts.append(coast)

    if not errs:
        print("No valid rows parsed.")
        return 1

    mae = sum(abs(e) for e in errs) / len(errs)
    rmse = rms(errs)
    p95 = percentile(errs, 0.95)

    idsw = 0
    prev = -1
    for cid in controlled_ids:
        if cid < 0:
            continue
        if prev >= 0 and cid != prev:
            idsw += 1
        prev = cid

    coast_ticks = sum(1 for c in coast_counts if c > 0)
    coast_ratio = coast_ticks / len(coast_counts)

    max_coast = 0
    run = 0
    for c in coast_counts:
        if c > 0:
            run += 1
            max_coast = max(max_coast, run)
        else:
            run = 0

    delta_cmd = []
    for i in range(1, len(cmd_xs)):
        delta_cmd.append(math.hypot(cmd_xs[i] - cmd_xs[i - 1], cmd_ys[i] - cmd_ys[i - 1]))

    print("=== tracker log evaluation ===")
    print(f"rows: {len(errs)}")
    print(f"MAE(|e|): {mae:.4f}")
    print(f"RMSE(|e|): {rmse:.4f}")
    print(f"P95(|e|): {p95:.4f}")
    print(f"IDSW: {idsw}")
    print(f"coast_ticks: {coast_ticks} ({coast_ratio:.2%})")
    print(f"max_consecutive_coast_ticks: {max_coast}")
    print(f"RMS(螖cmd): {rms(delta_cmd):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
