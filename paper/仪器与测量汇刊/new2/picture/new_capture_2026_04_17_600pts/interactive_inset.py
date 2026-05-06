"""
Interactive inset adjuster.
- Drag inset boxes to reposition them
- Ctrl+S to save the figure to plot_comparison_experiments.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg" if "--save" in sys.argv else "TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from plot_comparison_experiments_2026_04_17 import (
    resolve_paths, setup_matplotlib,
    build_comparison_table, load_split_reference,
    tune_kalman_on_val, build_kalman_predictions,
    add_series,
)

ZOOM_X_START = 998
ZOOM_X_END   = 1022
ZOOM_Y_START = 968
ZOOM_Y_END   = 1025

OUTPUT_PNG = Path(__file__).parent / "interactive_inset.png"

COLS_X = [
    ("true_legacy_u_x", "#111111", (6, 2)),
    ("dscg_pred_x",     "#d62728", (2, 2)),
    ("linear_pred_x",   "#ff7f0e", (5, 1.6)),
    ("kalman_pred_x",   "#9467bd", (4, 1.5)),
    ("gru_pred_x",      "#2ca02c", (4, 1.5, 1.2, 1.5)),
]
COLS_Y = [
    ("true_legacy_u_y", "#111111", (6, 2)),
    ("dscg_pred_y",     "#d62728", (2, 2)),
    ("linear_pred_y",   "#ff7f0e", (5, 1.6)),
    ("kalman_pred_y",   "#9467bd", (4, 1.5)),
    ("gru_pred_y",      "#2ca02c", (4, 1.5, 1.2, 1.5)),
]


class DraggableInset:
    """Allow dragging an inset axes by clicking and dragging its background."""
    def __init__(self, axins):
        self.axins = axins
        self._press = None
        self.axins.patch.set_picker(True)
        self.axins.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.axins.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self.axins.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.axins.figure.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_press(self, event):
        if event.inaxes != self.axins:
            return
        pos = self.axins.get_position()
        self._press = (event.x, event.y, pos.x0, pos.y0, pos.width, pos.height)

    def _on_release(self, event):
        self._press = None

    def _on_motion(self, event):
        if self._press is None:
            return
        fig = self.axins.figure
        fw, fh = fig.get_size_inches() * fig.dpi
        x0, y0, ax_x0, ax_y0, ax_w, ax_h = self._press
        dx = (event.x - x0) / fw
        dy = (event.y - y0) / fh
        self.axins.set_position([ax_x0 + dx, ax_y0 + dy, ax_w, ax_h])
        fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.axins:
            return
        pos = self.axins.get_position()
        scale = 0.9 if event.button == "up" else 1.1
        cx = pos.x0 + pos.width / 2
        cy = pos.y0 + pos.height / 2
        nw = pos.width * scale
        nh = pos.height * scale
        self.axins.set_position([cx - nw/2, cy - nh/2, nw, nh])
        self.axins.figure.canvas.draw_idle()


def build_inset(parent_ax, frames, segment_df, cols, zoom_start, zoom_end, pos):
    fig = parent_ax.get_figure()
    axins = fig.add_axes(pos)
    mask = (frames >= zoom_start) & (frames <= zoom_end)
    zf = frames[mask]
    for col, color, dashes in cols:
        vals = segment_df[col].to_numpy(dtype=np.float64)[mask]
        (line,) = axins.plot(zf, vals, color=color, linewidth=0.7, linestyle="--")
        line.set_dashes(dashes)
    axins.set_xlim(zoom_start, zoom_end)
    axins.tick_params(labelsize=5)
    axins.grid(True, linestyle="--", alpha=0.3)
    axins.set_title(f"Zoom [{zoom_start}–{zoom_end}]", fontsize=5.5, pad=2)
    mark_inset(parent_ax, axins, loc1=2, loc2=3, fc="none", ec="0.4", lw=0.6)
    return axins


def main():
    setup_matplotlib()
    paths = resolve_paths()
    raw_df = pd.read_csv(paths["raw_data_csv"]).reset_index(drop=True)
    split_ref = load_split_reference(paths["dscg_csv"])
    best_q, best_r, _ = tune_kalman_on_val(raw_df, split_ref)
    kalman_df = build_kalman_predictions(raw_df, q_scale=best_q, r_scale=best_r)
    segment_df = build_comparison_table(paths).merge(
        kalman_df, on="current_row_index", how="left", validate="one_to_one"
    )

    frames = segment_df["frame_id"].to_numpy(dtype=np.int64)
    true_x = segment_df["true_legacy_u_x"].to_numpy(dtype=np.float64)
    true_y = segment_df["true_legacy_u_y"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.6), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    add_series(axes[0], frames, true_x, "True next-step legacy control", "#111111", (6, 2), linewidth=0.9)
    add_series(axes[0], frames, segment_df["dscg_pred_x"], "DSCGNet", "#d62728", (2, 2))
    add_series(axes[0], frames, segment_df["linear_pred_x"], "Linear extrapolation", "#ff7f0e", (5, 1.6))
    add_series(axes[0], frames, segment_df["kalman_pred_x"], "Kalman CV", "#9467bd", (4, 1.5))
    add_series(axes[0], frames, segment_df["gru_pred_x"], "GRU direct", "#2ca02c", (4, 1.5, 1.2, 1.5))
    axes[0].set_ylabel("X-axis control signal")
    axes[0].set_title("Comparison: One-Step-Ahead Legacy Control Prediction")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="lower right", fontsize=8, ncol=1)
    axes[0].margins(x=0.01)

    add_series(axes[1], frames, true_y, "True next-step legacy control", "#111111", (6, 2), linewidth=0.9)
    add_series(axes[1], frames, segment_df["dscg_pred_y"], "DSCGNet", "#d62728", (2, 2))
    add_series(axes[1], frames, segment_df["linear_pred_y"], "Linear extrapolation", "#ff7f0e", (5, 1.6))
    add_series(axes[1], frames, segment_df["kalman_pred_y"], "Kalman CV", "#9467bd", (4, 1.5))
    add_series(axes[1], frames, segment_df["gru_pred_y"], "GRU direct", "#2ca02c", (4, 1.5, 1.2, 1.5))
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Y-axis control signal")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="lower right", fontsize=8, ncol=1)
    axes[1].margins(x=0.01)

    # build insets — initial positions in figure fraction
    p0 = axes[0].get_position()
    p1 = axes[1].get_position()
    axins0 = build_inset(axes[0], frames, segment_df, COLS_X,
                         ZOOM_X_START, ZOOM_X_END,
                         [p0.x0 + p0.width*0.02, p0.y0 + p0.height*0.38,
                          p0.width*0.28, p0.height*0.52])
    axins1 = build_inset(axes[1], frames, segment_df, COLS_Y,
                         ZOOM_Y_START, ZOOM_Y_END,
                         [p1.x0 + p1.width*0.02, p1.y0 + p1.height*0.38,
                          p1.width*0.28, p1.height*0.52])

    drag0 = DraggableInset(axins0)  # noqa: F841
    drag1 = DraggableInset(axins1)  # noqa: F841

    def on_key(event):
        if event.key == "ctrl+s":
            fig.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches="tight")
            print(f"Saved: {OUTPUT_PNG}")

    fig.canvas.mpl_connect("key_press_event", on_key)
    if "--save" in sys.argv:
        fig.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_PNG}")
        plt.close(fig)
        return
    #fig.suptitle("Drag inset to reposition  |  Scroll on inset to resize  |  Ctrl+S to save",
                 #fontsize=7, color="gray", y=0.995)
    plt.show()


if __name__ == "__main__":
    main()
