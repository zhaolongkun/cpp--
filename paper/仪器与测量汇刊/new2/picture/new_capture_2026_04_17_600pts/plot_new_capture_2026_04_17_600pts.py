import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Raw data file used for this figure set.
RAW_DATA_CSV = Path(r"D:\kun-data\kun-code-data\反无\cpp智能控制\data\track-fusion-move_2026-4-17_new.csv")

# Prediction file generated from the raw data file with the trained DSCGNet checkpoint.
PREDICTION_CSV = Path(
    r"D:\kun-data\kun-code-data\反无\cpp智能控制\data\train\track-fusion-move_2026-4-17_new_dscgnet_predictions.csv"
)

# Selected contiguous segment for plotting.
# The segment is chosen from model_ready == 1 rows and contains 600 points.
# Start/end are written explicitly for traceability.
SEGMENT_START_CURRENT_ROW_INDEX = 900
SEGMENT_END_CURRENT_ROW_INDEX = 1555
SEGMENT_START_FRAME_ID = 960
SEGMENT_END_FRAME_ID = 1615
SEGMENT_POINT_COUNT = 656

OUTPUT_DIR = Path(__file__).resolve().parent
TRUE_PRED_FIGURE = OUTPUT_DIR / "fig_new_capture_2026_04_17_prediction_en_600pts.png"

LINE_WIDTH = 0.8


def setup_matplotlib() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def add_series(ax, x_values, y_values, label: str, color: str, dashes) -> None:
    (line,) = ax.plot(
        x_values,
        y_values,
        label=label,
        color=color,
        linewidth=LINE_WIDTH,
        linestyle="--",
    )
    line.set_dashes(dashes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a selected legacy-control prediction segment.")
    parser.add_argument("--start-row", type=int, default=None, help="Inclusive start current_row_index.")
    parser.add_argument("--end-row", type=int, default=None, help="Inclusive end current_row_index.")
    parser.add_argument("--start-frame", type=int, default=None, help="Inclusive start frame_id.")
    parser.add_argument("--end-frame", type=int, default=None, help="Inclusive end frame_id.")
    parser.add_argument("--tag", type=str, default=None, help="Optional suffix for output figure filenames.")
    return parser.parse_args()


def resolve_selection(args: argparse.Namespace) -> tuple[str, int, int, str]:
    has_row_range = args.start_row is not None or args.end_row is not None
    has_frame_range = args.start_frame is not None or args.end_frame is not None

    if has_row_range and has_frame_range:
        raise ValueError("Use either row range or frame range, not both.")

    if has_row_range:
        if args.start_row is None or args.end_row is None:
            raise ValueError("Both --start-row and --end-row are required.")
        if args.end_row < args.start_row:
            raise ValueError("--end-row must be >= --start-row.")
        return "row", int(args.start_row), int(args.end_row), f"rows_{args.start_row}_{args.end_row}"

    if has_frame_range:
        if args.start_frame is None or args.end_frame is None:
            raise ValueError("Both --start-frame and --end-frame are required.")
        if args.end_frame < args.start_frame:
            raise ValueError("--end-frame must be >= --start-frame.")
        return "frame", int(args.start_frame), int(args.end_frame), f"frames_{args.start_frame}_{args.end_frame}"

    return (
        "row",
        SEGMENT_START_CURRENT_ROW_INDEX,
        SEGMENT_END_CURRENT_ROW_INDEX,
        f"rows_{SEGMENT_START_CURRENT_ROW_INDEX}_{SEGMENT_END_CURRENT_ROW_INDEX}",
    )


def load_segment(selection_mode: str, start_value: int, end_value: int) -> pd.DataFrame:
    df = pd.read_csv(PREDICTION_CSV)
    df = df[df["model_ready"] == 1].copy()
    if selection_mode == "row":
        df = df[(df["current_row_index"] >= start_value) & (df["current_row_index"] <= end_value)].copy()
        df = df.sort_values("current_row_index").reset_index(drop=True)
    else:
        df = df[(df["frame_id"] >= start_value) & (df["frame_id"] <= end_value)].copy()
        df = df.sort_values("frame_id").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("The selected segment is empty.")

    return df


def save_true_vs_pred_figure(df: pd.DataFrame) -> None:
    frames = df["frame_id"].to_numpy()
    true_x = df["true_legacy_u_x"].to_numpy()
    pred_x = df["pred_legacy_u_x"].to_numpy()
    true_y = df["true_legacy_u_y"].to_numpy()
    pred_y = df["pred_legacy_u_y"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True)

    add_series(axes[0], frames, true_x, "True next-step legacy control", "#1f77b4", (5, 2))
    add_series(axes[0], frames, pred_x, "Predicted next-step legacy control", "#d62728", (2, 2))
    axes[0].set_ylabel("X-axis control signal")
    axes[0].set_title("New Capture Segment: True vs. Predicted Next-Step Legacy Control")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="upper right", fontsize=8)

    add_series(axes[1], frames, true_y, "True next-step legacy control", "#1f77b4", (5, 2))
    add_series(axes[1], frames, pred_y, "Predicted next-step legacy control", "#d62728", (2, 2))
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Y-axis control signal")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(TRUE_PRED_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()
    args = parse_args()
    selection_mode, start_value, end_value, default_tag = resolve_selection(args)
    df = load_segment(selection_mode, start_value, end_value)
    output_tag = args.tag or default_tag

    global TRUE_PRED_FIGURE
    TRUE_PRED_FIGURE = OUTPUT_DIR / f"fig_new_capture_prediction_en_{output_tag}.png"

    save_true_vs_pred_figure(df)

    row_start = int(df.iloc[0]["current_row_index"])
    row_end = int(df.iloc[-1]["current_row_index"])
    frame_start = int(df.iloc[0]["frame_id"])
    frame_end = int(df.iloc[-1]["frame_id"])
    point_count = len(df)

    print(f"Raw data file: {RAW_DATA_CSV}")
    print(f"Prediction file: {PREDICTION_CSV}")
    print(
        "Selected segment: "
        f"selection_mode={selection_mode}, "
        f"current_row_index {row_start}->{row_end}, "
        f"frame_id {frame_start}->{frame_end}, "
        f"points={point_count}"
    )
    print(TRUE_PRED_FIGURE)


if __name__ == "__main__":
    main()
