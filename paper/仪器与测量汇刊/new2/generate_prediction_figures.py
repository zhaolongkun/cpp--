import csv
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


THIN_LINEWIDTH = 0.9


def setup_matplotlib() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_test_rows(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("model_ready") != "1":
                continue
            if row.get("prediction_split") != "test":
                continue
            try:
                rows.append(
                    {
                        "frame_id": float(row["frame_id"]),
                        "timestamp_ms": float(row["timestamp_ms"]),
                        "current_x": float(row["current_legacy_u_x"]),
                        "current_y": float(row["current_legacy_u_y"]),
                        "true_x": float(row["true_legacy_u_x"]),
                        "true_y": float(row["true_legacy_u_y"]),
                        "pred_x": float(row["pred_legacy_u_x"]),
                        "pred_y": float(row["pred_legacy_u_y"]),
                    }
                )
            except (TypeError, ValueError):
                continue
    return rows


def select_dynamic_window(rows: List[Dict[str, float]], window_size: int = 220) -> List[Dict[str, float]]:
    if len(rows) <= window_size:
        return rows

    best_start = 0
    best_score = float("-inf")
    for start in range(0, len(rows) - window_size + 1):
        window = rows[start : start + window_size]
        frame_ids = [row["frame_id"] for row in window]
        contiguous = all(abs(frame_ids[idx] - frame_ids[idx - 1] - 1.0) < 1e-6 for idx in range(1, len(frame_ids)))
        if not contiguous:
            continue

        current_x = np.asarray([row["current_x"] for row in window], dtype=np.float64)
        current_y = np.asarray([row["current_y"] for row in window], dtype=np.float64)
        true_x = np.asarray([row["true_x"] for row in window], dtype=np.float64)
        true_y = np.asarray([row["true_y"] for row in window], dtype=np.float64)
        score = float(np.mean(np.abs(true_x - current_x) + np.abs(true_y - current_y)))
        if score > best_score:
            best_score = score
            best_start = start

    return rows[best_start : best_start + window_size]


def add_series(ax, x_values, y_values, label: str, color: str, dashes) -> None:
    (line,) = ax.plot(
        x_values,
        y_values,
        label=label,
        color=color,
        linewidth=THIN_LINEWIDTH,
        linestyle="--",
    )
    line.set_dashes(dashes)


def save_true_vs_pred_figure_en(rows: List[Dict[str, float]], output_path: Path) -> None:
    frames = [row["frame_id"] for row in rows]
    true_x = [row["true_x"] for row in rows]
    pred_x = [row["pred_x"] for row in rows]
    true_y = [row["true_y"] for row in rows]
    pred_y = [row["pred_y"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)

    add_series(axes[0], frames, true_x, "True next-step legacy control", "#1f77b4", (5, 2))
    add_series(axes[0], frames, pred_x, "Predicted next-step legacy control", "#d62728", (2, 2))
    axes[0].set_ylabel("X-axis control signal")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Test Segment: True vs. Predicted Next-Step Legacy Control")

    add_series(axes[1], frames, true_y, "True next-step legacy control", "#1f77b4", (5, 2))
    add_series(axes[1], frames, pred_y, "Predicted next-step legacy control", "#d62728", (2, 2))
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Y-axis control signal")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_current_true_pred_figure_en(rows: List[Dict[str, float]], output_path: Path) -> None:
    frames = [row["frame_id"] for row in rows]
    current_x = [row["current_x"] for row in rows]
    true_x = [row["true_x"] for row in rows]
    pred_x = [row["pred_x"] for row in rows]
    current_y = [row["current_y"] for row in rows]
    true_y = [row["true_y"] for row in rows]
    pred_y = [row["pred_y"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)

    add_series(axes[0], frames, current_x, "Current legacy control state", "#444444", (1.5, 1.5))
    add_series(axes[0], frames, true_x, "True next-step legacy control", "#1f77b4", (5, 2))
    add_series(axes[0], frames, pred_x, "Predicted next-step legacy control", "#d62728", (2, 2))
    axes[0].set_ylabel("X-axis control signal")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Test Segment: Current State, True Next-Step, and Prediction")

    add_series(axes[1], frames, current_y, "Current legacy control state", "#444444", (1.5, 1.5))
    add_series(axes[1], frames, true_y, "True next-step legacy control", "#1f77b4", (5, 2))
    add_series(axes[1], frames, pred_y, "Predicted next-step legacy control", "#d62728", (2, 2))
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Y-axis control signal")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()
    parser = argparse.ArgumentParser(description="Generate prediction figures for legacy-control one-step-ahead results.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Prediction CSV path. Defaults to data/train/dscgnet_predictions.csv.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="fig_test",
        help="Output filename prefix inside the picture directory.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=220,
        help="Length of the dynamic segment to render.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = args.csv_path if args.csv_path is not None else project_root() / "data" / "train" / "dscgnet_predictions.csv"
    picture_dir = base_dir / "picture"
    picture_dir.mkdir(parents=True, exist_ok=True)

    rows = load_test_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No valid test samples were loaded from {csv_path}.")

    window = select_dynamic_window(rows, window_size=args.window_size)
    start_frame = int(window[0]["frame_id"])
    end_frame = int(window[-1]["frame_id"])

    true_pred_path = picture_dir / f"{args.prefix}_prediction_en.png"
    transition_path = picture_dir / f"{args.prefix}_current_true_pred_en.png"

    save_true_vs_pred_figure_en(window, true_pred_path)
    save_current_true_pred_figure_en(window, transition_path)

    print(f"English figures generated for frame range: {start_frame} -> {end_frame}")
    print(true_pred_path)
    print(transition_path)


if __name__ == "__main__":
    main()
