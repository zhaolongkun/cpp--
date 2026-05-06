import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# Edit only these two globals if you want to plot other columns.
PLOT_Y1_COLUMN = "det_dx"
PLOT_Y2_COLUMN = "first_filter_dx"
MAX_SANITIZED_LINE_LENGTH = 1_000_000
COLUMN_ALIASES = {
    "det_x": "fused_dx",
    "det_y": "det_dy",
    "track_x": "track_dx",
    "track_y": "track_dy",
    "median_filter_x": "median_filter_dx",
    "median_filter_y": "median_filter_dy",
    "first_filter_x": "first_filter_dx",
    "first_filter_y": "first_filter_dy",
}


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sanitized_csv_lines(csv_path: Path):
    with csv_path.open("rb") as handle:
        for raw_line in handle:
            clean_line = raw_line.replace(b"\x00", b"")
            if not clean_line.strip():
                continue
            if len(clean_line) > MAX_SANITIZED_LINE_LENGTH:
                continue
            yield clean_line.decode("utf-8-sig", errors="ignore")


def resolve_column(requested: str, fieldnames) -> str:
    available = set(fieldnames or [])
    if requested in available:
        return requested

    alias = COLUMN_ALIASES.get(requested)
    if alias in available:
        return alias

    return requested


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_csv = base_dir / "track-fusion-move-baseline_new.csv"

    parser = argparse.ArgumentParser(
        description="Live plot for two columns from test.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help="CSV file path.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["frame", "time"],
        default="frame",
        help="Use frame_id or timestamp_ms as x-axis.",
    )
    parser.add_argument(
        "--raw-rows",
        action="store_true",
        help="Plot every row directly. Default behavior deduplicates by frame_id.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=300,
        help="Refresh interval in milliseconds.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []

    reader = csv.DictReader(sanitized_csv_lines(csv_path))
    y1_column = resolve_column(PLOT_Y1_COLUMN, reader.fieldnames)
    y2_column = resolve_column(PLOT_Y2_COLUMN, reader.fieldnames)
    required = {"frame_id", "timestamp_ms", y1_column, y2_column}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError("CSV missing columns: {}".format(sorted(missing)))

    rows = []
    for row in reader:
        row[PLOT_Y1_COLUMN] = row[y1_column]
        row[PLOT_Y2_COLUMN] = row[y2_column]
        try:
            float(row["frame_id"])
            float(row["timestamp_ms"])
            float(row[PLOT_Y1_COLUMN])
            float(row[PLOT_Y2_COLUMN])
        except (TypeError, ValueError):
            continue
        rows.append(row)
    return rows


def dedup_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    latest_by_frame = {}  # type: Dict[int, Dict[str, str]]
    for row in rows:
        frame_id = int(float(row["frame_id"]))
        latest_by_frame[frame_id] = row
    return [latest_by_frame[k] for k in sorted(latest_by_frame)]


def make_series(
    rows: List[Dict[str, str]],
    x_axis: str,
) -> Tuple[List[float], List[float], List[float]]:
    x_values = []  # type: List[float]
    y1_values = []  # type: List[float]
    y2_values = []  # type: List[float]

    t0 = float(rows[0]["timestamp_ms"]) if (x_axis == "time" and rows) else 0.0

    for row in rows:
        if x_axis == "frame":
            x_values.append(float(row["frame_id"]))
        else:
            x_values.append((float(row["timestamp_ms"]) - t0) / 1000.0)
        y1_values.append(float(row[PLOT_Y1_COLUMN]))
        y2_values.append(float(row[PLOT_Y2_COLUMN]))

    return x_values, y1_values, y2_values


def live_plot(args: argparse.Namespace) -> None:
    configure_csv_field_limit()
    plt.ion()
    figure, axes = plt.subplots(figsize=(12, 6))
    manager = plt.get_current_fig_manager()
    try:
        manager.set_window_title(
            "Live Plot: {} / {}".format(PLOT_Y1_COLUMN, PLOT_Y2_COLUMN)
        )
    except Exception:
        pass

    line1, = axes.plot([], [], label=PLOT_Y1_COLUMN, linewidth=1.4, color="#1f77b4")
    line2, = axes.plot([], [], label=PLOT_Y2_COLUMN, linewidth=1.4, color="#d62728")
    status_text = axes.text(
        0.01,
        0.99,
        "",
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "#999999",
        },
    )

    axes.set_title("{} / {}".format(PLOT_Y1_COLUMN, PLOT_Y2_COLUMN))
    axes.set_xlabel("frame_id" if args.x_axis == "frame" else "time (s)")
    axes.set_ylabel("value")
    axes.grid(True, linestyle="--", alpha=0.35)
    axes.legend()

    print("Live window started.")
    print("Edit globals at the top of this file:")
    print("PLOT_Y1_COLUMN = {}".format(PLOT_Y1_COLUMN))
    print("PLOT_Y2_COLUMN = {}".format(PLOT_Y2_COLUMN))
    print("Use toolbar Zoom or Pan to inspect the curve.")
    print("Close the window to exit.")

    last_signature = None
    while plt.fignum_exists(figure.number):
        try:
            rows = load_rows(args.csv)
            plot_rows = rows if args.raw_rows else dedup_rows(rows)

            if plot_rows:
                signature = (
                    len(plot_rows),
                    plot_rows[-1]["frame_id"],
                    plot_rows[-1]["timestamp_ms"],
                    plot_rows[-1][PLOT_Y1_COLUMN],
                    plot_rows[-1][PLOT_Y2_COLUMN],
                )
                if signature != last_signature:
                    x_values, y1_values, y2_values = make_series(plot_rows, args.x_axis)
                    line1.set_data(x_values, y1_values)
                    line2.set_data(x_values, y2_values)
                    axes.relim()
                    axes.autoscale_view()
                    axes.margins(x=0.02, y=0.12)

                    status_text.set_text(
                        "points: {}\nlast frame: {}\nlast time: {}\ny: [{:.2f}, {:.2f}]".format(
                            len(plot_rows),
                            plot_rows[-1]["frame_id"],
                            plot_rows[-1]["timestamp_ms"],
                            min(min(y1_values), min(y2_values)),
                            max(max(y1_values), max(y2_values)),
                        )
                    )
                    figure.canvas.draw_idle()
                    last_signature = signature
            else:
                status_text.set_text("waiting for data...")
                figure.canvas.draw_idle()
        except Exception as exc:
            status_text.set_text("read error: {}".format(exc))
            figure.canvas.draw_idle()

        plt.pause(max(args.interval_ms, 50) / 1000.0)


def main() -> None:
    args = parse_args()
    live_plot(args)


if __name__ == "__main__":
    main()
