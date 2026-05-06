import argparse
import csv
from pathlib import Path

import matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DSCGNet precision curves. By default an interactive window opens and the figure is saved after you close it."
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


PRECISION_DIR = Path(__file__).resolve().parent
PRECISION_CSV_PATH = PRECISION_DIR / "precisiion.csv"
PRECISION_CURVE_PNG_PATH = PRECISION_DIR / "precision_curve.png"
PRECISION_CURVE_PDF_PATH = PRECISION_DIR / "precision_curve.pdf"
FILTER_ALPHA = 0.4


def load_precision_csv(csv_path: Path) -> list:
    if not csv_path.exists():
        raise FileNotFoundError(f"precisiion.csv not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    required = {
        "epoch",
        "train_precision",
        "val_precision",
        "val_precision_tol_2",
        "val_precision_tol_3",
    }
    if not rows:
        raise RuntimeError("precisiion.csv is empty, nothing to plot.")
    missing = required.difference(rows[0].keys())
    if missing:
        raise KeyError(f"precisiion.csv missing required columns: {sorted(missing)}")

    parsed = []
    for row in rows:
        parsed.append(
            {
                "epoch": int(row["epoch"]),
                "train_precision": float(row["train_precision"]),
                "val_precision": float(row["val_precision"]),
                "val_precision_tol_2": float(row["val_precision_tol_2"]),
                "val_precision_tol_3": float(row["val_precision_tol_3"]),
            }
        )
    parsed.sort(key=lambda item: item["epoch"])
    return parsed


def apply_first_order_filter(rows: list, keys: list, alpha: float) -> list:
    if not rows:
        return rows

    previous = {key: rows[0][key] for key in keys}
    for row in rows[1:]:
        for key in keys:
            filtered = alpha * row[key] + (1.0 - alpha) * previous[key]
            row[key] = filtered
            previous[key] = filtered
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


def plot_precision_curve(rows: list) -> None:
    epochs = [row["epoch"] for row in rows]
    train_precision = [row["train_precision"] for row in rows]
    val_precision = [row["val_precision"] for row in rows]
    val_precision_tol_2 = [row["val_precision_tol_2"] for row in rows]
    val_precision_tol_3 = [row["val_precision_tol_3"] for row in rows]
    best_tol_1 = max(rows, key=lambda item: item["val_precision"])
    best_tol_2 = max(rows, key=lambda item: item["val_precision_tol_2"])
    best_tol_3 = max(rows, key=lambda item: item["val_precision_tol_3"])

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=220)
    if INTERACTIVE_MODE:
        try:
            fig.canvas.manager.set_window_title("DSCGNet Precision Curve")
        except Exception:
            pass

    line_p0 = ax.plot(
        epochs,
        train_precision,
        color="#0b3c5d",
        linewidth=2.0,
        label="Train precision (tol=1)",
    )
    line_p1 = ax.plot(
        epochs,
        val_precision,
        color="#2ca02c",
        linewidth=2.0,
        label="Validation precision (tol=1)",
    )
    line_p2 = ax.plot(
        epochs,
        val_precision_tol_2,
        color="#1f77b4",
        linewidth=1.8,
        label="Validation precision (tol=2)",
    )
    line_p3 = ax.plot(
        epochs,
        val_precision_tol_3,
        color="#ff7f0e",
        linewidth=1.8,
        label="Validation precision (tol=3)",
    )
    best_specs = [
        ("tol=1", best_tol_1, "val_precision", "#2ca02c", (18, -26)),
        ("tol=2", best_tol_2, "val_precision_tol_2", "#1f77b4", (24, -52)),
        ("tol=3", best_tol_3, "val_precision_tol_3", "#ff7f0e", (-120, -42)),
    ]
    annotations = []
    for label, best_row, key, color, offset in best_specs:
        ax.scatter(
            [best_row["epoch"]],
            [best_row[key]],
            color=color,
            s=44,
            zorder=5,
        )
        annotation = ax.annotate(
            f"{label} best epoch = {best_row['epoch']}\nPrecision = {best_row[key]:.4f}",
            xy=(best_row["epoch"], best_row[key]),
            xytext=offset,
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.9},
        )
        annotations.append(annotation)
    ax.set_title("DSCGNet Train and Validation Precision Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision (0-1)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    handles_top = line_p0 + line_p1 + line_p2 + line_p3
    labels_top = [handle.get_label() for handle in handles_top]
    ax.legend(handles_top, labels_top, loc="lower right", frameon=True)
    ax.set_axisbelow(True)

    if INTERACTIVE_MODE:
        enable_annotation_editing(fig, annotations)

    fig.tight_layout()
    if INTERACTIVE_MODE:
        print(
            "Interactive window opened. Left-drag a label box to move it, use the mouse wheel over a label box to resize it, use the toolbar to zoom/pan the plot, then close the window to save.",
            flush=True,
        )
        plt.show()
    fig.savefig(str(PRECISION_CURVE_PNG_PATH), bbox_inches="tight")
    fig.savefig(str(PRECISION_CURVE_PDF_PATH), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_precision_csv(PRECISION_CSV_PATH)
    apply_first_order_filter(
        rows,
        [
            "train_precision",
            "val_precision",
            "val_precision_tol_2",
            "val_precision_tol_3",
        ],
        FILTER_ALPHA,
    )
    plot_precision_curve(rows)
    print(f"[saved] {PRECISION_CURVE_PNG_PATH}")
    print(f"[saved] {PRECISION_CURVE_PDF_PATH}")
    if not INTERACTIVE_MODE and not ARGS.save_only:
        print("[warning] No interactive GUI backend was available, so the figure was saved directly.")


if __name__ == "__main__":
    main()
