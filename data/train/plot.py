import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


MAX_SANITIZED_LINE_LENGTH = 1_000_000


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


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_csv = base_dir / "dscgnet_predictions.csv"

    parser = argparse.ArgumentParser(
        description="Plot current / true / predicted legacy control with prediction error."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help="CSV file path.",
    )
    parser.add_argument(
        "--component",
        choices=["x", "y"],
        default="x",
        help="Plot x or y control channel.",
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


def _required_columns(component: str) -> Dict[str, str]:
    suffix = component.lower()
    return {
        "current": f"current_legacy_u_{suffix}",
        "true": f"true_legacy_u_{suffix}",
        "pred": f"pred_legacy_u_{suffix}",
        "delta": f"delta_pred_{suffix}",
    }


def load_rows(csv_path: Path, component: str) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []

    required = _required_columns(component)
    reader = csv.DictReader(sanitized_csv_lines(csv_path))
    needed = {
        "frame_id",
        "timestamp_ms",
        "model_ready",
        required["current"],
        required["true"],
        required["pred"],
        required["delta"],
    }
    missing = needed.difference(reader.fieldnames or [])
    if missing:
        raise ValueError("CSV missing columns: {}".format(sorted(missing)))

    rows = []
    for row in reader:
        if row.get("model_ready") != "1":
            continue
        try:
            float(row["frame_id"])
            float(row["timestamp_ms"])
            float(row[required["current"]])
            float(row[required["true"]])
            float(row[required["pred"]])
            float(row[required["delta"]])
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
    component: str,
    x_axis: str,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    columns = _required_columns(component)
    x_values: List[float] = []
    current_values: List[float] = []
    true_values: List[float] = []
    pred_values: List[float] = []
    error_values: List[float] = []
    delta_values: List[float] = []

    t0 = float(rows[0]["timestamp_ms"]) if (x_axis == "time" and rows) else 0.0

    for row in rows:
        if x_axis == "frame":
            x_values.append(float(row["frame_id"]))
        else:
            x_values.append((float(row["timestamp_ms"]) - t0) / 1000.0)

        current_value = float(row[columns["current"]])
        true_value = float(row[columns["true"]])
        pred_value = float(row[columns["pred"]])
        delta_value = float(row[columns["delta"]])

        current_values.append(current_value)
        true_values.append(true_value)
        pred_values.append(pred_value)
        error_values.append(pred_value - true_value)
        delta_values.append(delta_value)

    return x_values, current_values, true_values, pred_values, error_values, delta_values


def update_axes_style(signal_ax, error_ax, x_axis: str, component: str) -> None:
    signal_ax.set_title(
        "Legacy Control Compare ({})".format(component.upper())
    )
    signal_ax.set_ylabel("control value")
    signal_ax.grid(True, linestyle="--", alpha=0.35)
    signal_ax.legend(loc="upper right")

    error_ax.set_title("Prediction Error / Delta ({})".format(component.upper()))
    error_ax.set_xlabel("frame_id" if x_axis == "frame" else "time (s)")
    error_ax.set_ylabel("value")
    error_ax.grid(True, linestyle="--", alpha=0.35)
    error_ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":")
    error_ax.legend(loc="upper right")


def live_plot(args: argparse.Namespace) -> None:
    configure_csv_field_limit()
    plt.ion()
    figure, (signal_ax, error_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    manager = plt.get_current_fig_manager()
    try:
        manager.set_window_title(
            "Legacy Control Compare ({})".format(args.component.upper())
        )
    except Exception:
        pass

    current_line, = signal_ax.plot(
        [],
        [],
        label="current_legacy_u_{}".format(args.component),
        linewidth=1.2,
        color="#444444",
        linestyle="--",
    )
    true_line, = signal_ax.plot(
        [],
        [],
        label="true_legacy_u_{}".format(args.component),
        linewidth=1.5,
        color="#1f77b4",
    )
    pred_line, = signal_ax.plot(
        [],
        [],
        label="pred_legacy_u_{}".format(args.component),
        linewidth=1.2,
        color="#d62728",
    )
    error_line, = error_ax.plot(
        [],
        [],
        label="pred - true",
        linewidth=1.2,
        color="#9467bd",
    )
    delta_line, = error_ax.plot(
        [],
        [],
        label="delta_pred_{}".format(args.component),
        linewidth=1.2,
        color="#2ca02c",
        linestyle="--",
    )

    status_text = signal_ax.text(
        0.01,
        0.99,
        "",
        transform=signal_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#999999",
        },
    )

    update_axes_style(signal_ax, error_ax, args.x_axis, args.component)

    print("Live window started.")
    print("Component: {}".format(args.component))
    print("Top subplot: current / true / predicted legacy control")
    print("Bottom subplot: prediction error and delta_pred")
    print("Only rows with model_ready = 1 are plotted.")
    print("Close the window to exit.")

    last_signature = None
    while plt.fignum_exists(figure.number):
        try:
            rows = load_rows(args.csv, args.component)
            plot_rows = rows if args.raw_rows else dedup_rows(rows)

            if plot_rows:
                signature = (
                    len(plot_rows),
                    plot_rows[-1]["frame_id"],
                    plot_rows[-1]["timestamp_ms"],
                    plot_rows[-1][_required_columns(args.component)["true"]],
                    plot_rows[-1][_required_columns(args.component)["pred"]],
                )
                if signature != last_signature:
                    (
                        x_values,
                        current_values,
                        true_values,
                        pred_values,
                        error_values,
                        delta_values,
                    ) = make_series(plot_rows, args.component, args.x_axis)

                    current_line.set_data(x_values, current_values)
                    true_line.set_data(x_values, true_values)
                    pred_line.set_data(x_values, pred_values)
                    error_line.set_data(x_values, error_values)
                    delta_line.set_data(x_values, delta_values)

                    signal_ax.relim()
                    signal_ax.autoscale_view()
                    signal_ax.margins(x=0.02, y=0.12)

                    error_ax.relim()
                    error_ax.autoscale_view()
                    error_ax.margins(x=0.02, y=0.18)

                    mae = sum(abs(value) for value in error_values) / max(len(error_values), 1)
                    max_abs = max(abs(value) for value in error_values)
                    status_text.set_text(
                        "points: {}\nlast frame: {}\nlast time: {}\ncurrent / true / pred: [{:.3f}, {:.3f}, {:.3f}]\nerror mae: {:.4f}\nerror max_abs: {:.4f}".format(
                            len(plot_rows),
                            plot_rows[-1]["frame_id"],
                            plot_rows[-1]["timestamp_ms"],
                            current_values[-1],
                            true_values[-1],
                            pred_values[-1],
                            mae,
                            max_abs,
                        )
                    )

                    figure.canvas.draw_idle()
                    last_signature = signature
            else:
                status_text.set_text("waiting for model_ready=1 rows...")
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
