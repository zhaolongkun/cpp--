#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_logs.py <logs/tracker_log.csv>")
        return 1

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}")
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Log file not found: {path}")
        return 1

    t, dx, dy, cx, cy = [], [], [], [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                t.append(i)
                dx.append(float(row["dx_hat"]))
                dy.append(float(row["dy_hat"]))
                cx.append(float(row["cmd_x"]))
                cy.append(float(row["cmd_y"]))
            except Exception:
                continue

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(t, dx, label="dx_hat")
    axs[0].plot(t, dy, label="dy_hat")
    axs[0].set_ylabel("pixel error")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, cx, label="cmd_x")
    axs[1].plot(t, cy, label="cmd_y")
    axs[1].set_ylabel("motor cmd")
    axs[1].set_xlabel("tick")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
