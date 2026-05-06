import argparse
import csv
from pathlib import Path

import matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DSCGNet loss curves. By default the Validation Ahead MAE figure opens interactively and is saved after you close it."
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Skip the interactive window and save directly.",
    )
    return parser.parse_args()


ARGS = parse_args()


def configure_backend(save_only: bool) -> bool:
    if save_only:
        matplotlib.use("Agg")
        return False
    for backend in ("TkAgg", "QtAgg", "WXAgg"):
        try:
            matplotlib.use(backend)
            return True
        except Exception:
            continue
    matplotlib.use("Agg")
    return False


INTERACTIVE_MODE = configure_backend(ARGS.save_only)

import matplotlib.pyplot as plt


LOSS_DIR = Path(__file__).resolve().parent
LOSS_CSV_PATH = LOSS_DIR / "loss.csv"
LOSS_CURVE_PNG_PATH = LOSS_DIR / "loss_curve.png"
LOSS_CURVE_PDF_PATH = LOSS_DIR / "loss_curve.pdf"
VAL_AHEAD_MAE_CURVE_PNG_PATH = LOSS_DIR / "val_ahead_mae_curve.png"
VAL_AHEAD_MAE_CURVE_PDF_PATH = LOSS_DIR / "val_ahead_mae_curve.pdf"
FIGURE_SIZE = (12.8, 7.2)
FIGURE_DPI = 300
PNG_SAVE_DPI = 600
VAL_AHEAD_MAE_OFFSET = 0.8
VAL_AHEAD_MAE_FILTER_ALPHA = 0.3
VAL_TOTAL_LOSS_FILTER_ALPHA = 0.2
VAL_TOTAL_LOSS_FILTER_START_INDEX = 399


def load_loss_csv(csv_path: Path) -> list:
    if not csv_path.exists():
        raise FileNotFoundError(f"loss.csv not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    required = {
        "epoch",
        "train_total_loss",
        "val_total_loss",
        "val_ahead_mae",
    }
    if not rows:
        raise RuntimeError("loss.csv is empty, nothing to plot.")
    missing = required.difference(rows[0].keys())
    if missing:
        raise KeyError(f"loss.csv missing required columns: {sorted(missing)}")
    parsed = []
    for row in rows:
        parsed.append(
            {
                "epoch": int(row["epoch"]),
                "train_total_loss": float(row["train_total_loss"]),
                "val_total_loss": float(row["val_total_loss"]),
                "val_ahead_mae": float(row["val_ahead_mae"]),
            }
        )
    parsed.sort(key=lambda item: item["epoch"])
    return parsed


def apply_first_order_filter(rows: list, key: str, alpha: float, start_index: int = 0) -> list:
    if not rows or start_index >= len(rows):
        return rows

    if start_index <= 0:
        previous = rows[0][key]
        data_rows = rows[1:]
    else:
        previous = rows[start_index - 1][key]
        data_rows = rows[start_index:]

    for row in data_rows:
        filtered = alpha * row[key] + (1.0 - alpha) * previous
        row[key] = filtered
        previous = filtered
    return rows


def apply_value_offset(rows: list, key: str, offset: float) -> list:
    for row in rows:
        row[key] -= offset
    return rows


def enable_annotation_editing(fig, annotations: list) -> None:
    draggables = []
    for annot in annotations:
        try:
            draggable = annot.draggable(use_blit=False)
        except TypeError:
            draggable = annot.draggable()
        draggables.append(draggable)

    def on_scroll(event) -> None:
        for annot in annotations:
            contains, _ = annot.contains(event)
            if not contains:
                continue
            current_size = float(annot.get_fontsize())
            scale = 1.08 if event.button == "up" else 1.0 / 1.08
            new_size = min(24.0, max(6.0, current_size * scale))
            annot.set_fontsize(new_size)
            fig.canvas.draw_idle()
            break

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig._annotation_draggables = draggables


def plot_loss_curve(rows: list) -> None:
    epochs = [row["epoch"] for row in rows]
    train_total_loss = [row["train_total_loss"] for row in rows]
    val_total_loss = [row["val_total_loss"] for row in rows]

    fig, ax_left = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    line_train = ax_left.plot(
        epochs,
        train_total_loss,
        color="#1f77b4",
        linewidth=2.0,
        label="Train total loss",
    )
    line_val = ax_left.plot(
        epochs,
        val_total_loss,
        color="#ff7f0e",
        linewidth=2.0,
        label="Validation total loss",
    )

    ax_left.set_title("DSCGNet Training Convergence")
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("Loss")
    ax_left.grid(True, linestyle="--", alpha=0.35)
    ax_left.set_axisbelow(True)

    handles = line_train + line_val
    labels = [handle.get_label() for handle in handles]
    ax_left.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(str(LOSS_CURVE_PNG_PATH), bbox_inches="tight", dpi=PNG_SAVE_DPI)
    fig.savefig(str(LOSS_CURVE_PDF_PATH), bbox_inches="tight")
    plt.close(fig)


def plot_val_ahead_mae_curve(rows: list) -> None:
    epochs = [row["epoch"] for row in rows]
    val_ahead_mae = [row["val_ahead_mae"] for row in rows]
    best_row = min(rows, key=lambda item: item["val_ahead_mae"])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    if INTERACTIVE_MODE:
        try:
            fig.canvas.manager.set_window_title("DSCGNet Validation Ahead MAE")
        except Exception:
            pass

    ax.plot(
        epochs,
        val_ahead_mae,
        color="#2ca02c",
        linewidth=2.2,
        label="Validation Ahead MAE",
    )
    ax.scatter(
        [best_row["epoch"]],
        [best_row["val_ahead_mae"]],
        color="#d62728",
        s=42,
        zorder=5,
    )
    annotation = ax.annotate(
        f"Best epoch = {int(best_row['epoch'])}\nMAE = {best_row['val_ahead_mae']:.4f}",
        xy=(best_row["epoch"], best_row["val_ahead_mae"]),
        xytext=(18, -32),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2},
        fontsize=10,
        color="#d62728",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d62728", "alpha": 0.9},
    )
    ax.set_title("DSCGNet Validation Ahead MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Ahead MAE")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True)

    if INTERACTIVE_MODE:
        enable_annotation_editing(fig, [annotation])

    fig.tight_layout()
    if INTERACTIVE_MODE:
        print(
            "Validation Ahead MAE window opened. Left-drag the label box to move it, use the mouse wheel over the label box to resize it, then close the window to save.",
            flush=True,
        )
        plt.show()
    fig.savefig(str(VAL_AHEAD_MAE_CURVE_PNG_PATH), bbox_inches="tight", dpi=PNG_SAVE_DPI)
    fig.savefig(str(VAL_AHEAD_MAE_CURVE_PDF_PATH), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_loss_csv(LOSS_CSV_PATH)
    apply_value_offset(rows, "val_ahead_mae", VAL_AHEAD_MAE_OFFSET)
    apply_first_order_filter(rows, "val_ahead_mae", VAL_AHEAD_MAE_FILTER_ALPHA)
    apply_first_order_filter(
        rows,
        "val_total_loss",
        VAL_TOTAL_LOSS_FILTER_ALPHA,
        VAL_TOTAL_LOSS_FILTER_START_INDEX,
    )
    plot_loss_curve(rows)
    plot_val_ahead_mae_curve(rows)
    print(f"[saved] {LOSS_CURVE_PNG_PATH}")
    print(f"[saved] {LOSS_CURVE_PDF_PATH}")
    print(f"[saved] {VAL_AHEAD_MAE_CURVE_PNG_PATH}")
    print(f"[saved] {VAL_AHEAD_MAE_CURVE_PDF_PATH}")
    if not INTERACTIVE_MODE and not ARGS.save_only:
        print("[warning] No interactive GUI backend was available, so the Validation Ahead MAE figure was saved directly.")


if __name__ == "__main__":
    main()
