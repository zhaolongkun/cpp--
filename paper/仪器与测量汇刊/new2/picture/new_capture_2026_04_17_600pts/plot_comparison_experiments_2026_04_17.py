from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle


SEGMENT_START_CURRENT_ROW_INDEX = 800
SEGMENT_END_CURRENT_ROW_INDEX = 1555
# The frame-id range is derived from the exported prediction rows at runtime.
SEGMENT_POINT_COUNT = 756

LINE_WIDTH = 0.8


def project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def output_dir() -> Path:
    return Path(__file__).resolve().parent


def setup_matplotlib() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def add_series(ax, x_values, y_values, label: str, color: str, dashes, linewidth: float = LINE_WIDTH) -> None:
    (line,) = ax.plot(
        x_values,
        y_values,
        label=label,
        color=color,
        linewidth=linewidth,
        linestyle="--",
        alpha=0.95,
    )
    line.set_dashes(dashes)


def resolve_paths() -> dict[str, Path]:
    root = project_root()
    return {
        "raw_data_csv": root / "data" / "track-fusion-move_2026-4-17_new.csv",
        "dscg_csv": root / "data" / "train" / "track-fusion-move_2026-4-17_new_dscgnet_predictions.csv",
        "gru_csv": root / "data" / "train" / "track-fusion-move_2026-4-17_new_gru_direct_predictions.csv",
        "figure_png": output_dir() / "plot_comparison_experiments.png",
        "metrics_csv": output_dir() / "comparison_metrics_rows_500_1555.csv",
        "kalman_params_txt": output_dir() / "kalman_params_rows_500_1555.txt",
    }


def load_prediction_segment(csv_path: Path, pred_x_col: str, pred_y_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["model_ready"] == 1].copy()
    df = df[
        (df["current_row_index"] >= SEGMENT_START_CURRENT_ROW_INDEX)
        & (df["current_row_index"] <= SEGMENT_END_CURRENT_ROW_INDEX)
    ].copy()
    df = df.sort_values("current_row_index").reset_index(drop=True)

    if len(df) != SEGMENT_POINT_COUNT:
        raise RuntimeError(f"{csv_path.name}: expected {SEGMENT_POINT_COUNT} points, got {len(df)}.")

    return df[
        [
            "current_row_index",
            "frame_id",
            "true_legacy_u_x",
            "true_legacy_u_y",
            pred_x_col,
            pred_y_col,
            "current_legacy_u_x",
            "current_legacy_u_y",
        ]
    ].copy()


def build_comparison_table(paths: dict[str, Path]) -> pd.DataFrame:
    dscg = load_prediction_segment(paths["dscg_csv"], "pred_legacy_u_x", "pred_legacy_u_y").rename(
        columns={
            "pred_legacy_u_x": "dscg_pred_x",
            "pred_legacy_u_y": "dscg_pred_y",
        }
    )
    gru = load_prediction_segment(paths["gru_csv"], "pred_legacy_u_x", "pred_legacy_u_y").rename(
        columns={
            "pred_legacy_u_x": "gru_pred_x",
            "pred_legacy_u_y": "gru_pred_y",
        }
    )

    merged = dscg.merge(
        gru[["current_row_index", "gru_pred_x", "gru_pred_y"]],
        on="current_row_index",
        how="inner",
        validate="one_to_one",
    )

    raw_df = pd.read_csv(paths["raw_data_csv"]).reset_index(drop=True)
    prev_df = raw_df[["first_filter_dx", "first_filter_dy"]].copy()
    prev_df["current_row_index"] = prev_df.index + 1
    prev_df = prev_df.rename(
        columns={
            "first_filter_dx": "prev_legacy_u_x",
            "first_filter_dy": "prev_legacy_u_y",
        }
    )
    merged = merged.merge(prev_df, on="current_row_index", how="left", validate="one_to_one")

    if merged["prev_legacy_u_x"].isna().any() or merged["prev_legacy_u_y"].isna().any():
        raise RuntimeError("Previous-state lookup failed for linear extrapolation baseline.")

    merged["last_value_pred_x"] = merged["current_legacy_u_x"]
    merged["last_value_pred_y"] = merged["current_legacy_u_y"]
    merged["linear_pred_x"] = 2.0 * merged["current_legacy_u_x"] - merged["prev_legacy_u_x"]
    merged["linear_pred_y"] = 2.0 * merged["current_legacy_u_y"] - merged["prev_legacy_u_y"]

    return merged


def load_split_reference(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["model_ready"] == 1].copy()
    return df[["current_row_index", "prediction_split", "true_legacy_u_x", "true_legacy_u_y"]].copy()


def _cv_kalman_predict_next(
    values_x: np.ndarray,
    values_y: np.ndarray,
    timestamps: np.ndarray,
    q_scale: float,
    r_scale: float,
) -> np.ndarray:
    n = int(values_x.shape[0])
    if n < 2:
        raise ValueError("Need at least two points for Kalman one-step prediction.")

    diffs = np.diff(timestamps)
    positive_diffs = diffs[diffs > 0]
    median_dt = float(np.median(positive_diffs)) if positive_diffs.size > 0 else 1.0
    dt_steps = np.maximum(diffs / max(median_dt, 1e-6), 1e-3)

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    I = np.eye(4, dtype=np.float64)

    vx0 = float(values_x[1] - values_x[0])
    vy0 = float(values_y[1] - values_y[0])
    x_post = np.array([values_x[0], vx0, values_y[0], vy0], dtype=np.float64)
    P_post = np.diag([10.0, 25.0, 10.0, 25.0]).astype(np.float64)
    R = np.eye(2, dtype=np.float64) * float(r_scale)

    preds = np.full((n - 1, 2), np.nan, dtype=np.float64)
    for i in range(n - 1):
        dt = float(dt_steps[i])
        F = np.array(
            [
                [1.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q_block = float(q_scale) * np.array(
            [
                [0.25 * dt**4, 0.5 * dt**3],
                [0.5 * dt**3, dt**2],
            ],
            dtype=np.float64,
        )
        Q = np.zeros((4, 4), dtype=np.float64)
        Q[:2, :2] = q_block
        Q[2:, 2:] = q_block

        x_pred = F @ x_post
        P_pred = F @ P_post @ F.T + Q
        preds[i] = np.array([x_pred[0], x_pred[2]], dtype=np.float64)

        z_next = np.array([values_x[i + 1], values_y[i + 1]], dtype=np.float64)
        innovation = z_next - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_post = x_pred + K @ innovation
        P_post = (I - K @ H) @ P_pred

    return preds


def tune_kalman_on_val(raw_df: pd.DataFrame, split_reference: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    timestamps = raw_df["timestamp_ms"].to_numpy(dtype=np.float64)
    values_x = raw_df["first_filter_dx"].to_numpy(dtype=np.float64)
    values_y = raw_df["first_filter_dy"].to_numpy(dtype=np.float64)

    val_ref = split_reference[split_reference["prediction_split"] == "val"].copy()
    if val_ref.empty:
        raise RuntimeError("Validation split is empty; cannot tune Kalman baseline.")

    q_candidates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    r_candidates = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

    best_q = None
    best_r = None
    best_score = float("inf")
    search_rows: list[dict[str, float]] = []

    for q_scale in q_candidates:
        for r_scale in r_candidates:
            preds = _cv_kalman_predict_next(values_x, values_y, timestamps, q_scale=q_scale, r_scale=r_scale)
            pred_df = pd.DataFrame(
                {
                    "current_row_index": np.arange(preds.shape[0], dtype=np.int64),
                    "kalman_pred_x": preds[:, 0],
                    "kalman_pred_y": preds[:, 1],
                }
            )
            joined = val_ref.merge(pred_df, on="current_row_index", how="inner", validate="one_to_one")
            err_x = joined["kalman_pred_x"].to_numpy(dtype=np.float64) - joined["true_legacy_u_x"].to_numpy(dtype=np.float64)
            err_y = joined["kalman_pred_y"].to_numpy(dtype=np.float64) - joined["true_legacy_u_y"].to_numpy(dtype=np.float64)
            score = float((np.mean(np.abs(err_x)) + np.mean(np.abs(err_y))) / 2.0)
            search_rows.append({"q_scale": q_scale, "r_scale": r_scale, "val_avg_mae_xy": score})
            if score < best_score:
                best_score = score
                best_q = q_scale
                best_r = r_scale

    if best_q is None or best_r is None:
        raise RuntimeError("Kalman parameter search failed.")

    return float(best_q), float(best_r), pd.DataFrame(search_rows).sort_values("val_avg_mae_xy").reset_index(drop=True)


def build_kalman_predictions(raw_df: pd.DataFrame, q_scale: float, r_scale: float) -> pd.DataFrame:
    timestamps = raw_df["timestamp_ms"].to_numpy(dtype=np.float64)
    values_x = raw_df["first_filter_dx"].to_numpy(dtype=np.float64)
    values_y = raw_df["first_filter_dy"].to_numpy(dtype=np.float64)
    preds = _cv_kalman_predict_next(values_x, values_y, timestamps, q_scale=q_scale, r_scale=r_scale)
    return pd.DataFrame(
        {
            "current_row_index": np.arange(preds.shape[0], dtype=np.int64),
            "kalman_pred_x": preds[:, 0],
            "kalman_pred_y": preds[:, 1],
        }
    )


def build_metrics_table(segment_df: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "LastValueBaseline": ("last_value_pred_x", "last_value_pred_y"),
        "LinearExtrapolationBaseline": ("linear_pred_x", "linear_pred_y"),
        "KalmanCVBaseline": ("kalman_pred_x", "kalman_pred_y"),
        "GRUDirectBaseline": ("gru_pred_x", "gru_pred_y"),
        "DSCGNet": ("dscg_pred_x", "dscg_pred_y"),
    }
    rows = []
    true_x = segment_df["true_legacy_u_x"].to_numpy(dtype=np.float64)
    true_y = segment_df["true_legacy_u_y"].to_numpy(dtype=np.float64)
    for method_name, (pred_x_col, pred_y_col) in methods.items():
        pred_x = segment_df[pred_x_col].to_numpy(dtype=np.float64)
        pred_y = segment_df[pred_y_col].to_numpy(dtype=np.float64)
        err_x = pred_x - true_x
        err_y = pred_y - true_y
        rows.append(
            {
                "method": method_name,
                "mae_x": float(np.mean(np.abs(err_x))),
                "mae_y": float(np.mean(np.abs(err_y))),
                "rmse_x": float(np.sqrt(np.mean(np.square(err_x)))),
                "rmse_y": float(np.sqrt(np.mean(np.square(err_y)))),
                "avg_mae_xy": float((np.mean(np.abs(err_x)) + np.mean(np.abs(err_y))) / 2.0),
            }
        )
    return pd.DataFrame(rows)


ZOOM_ROW_START = 820
ZOOM_ROW_END = 900


def _add_inset(parent_ax, segment_df: pd.DataFrame, cols: list[tuple[str, str, tuple]], inset_slot: int) -> None:
    """Add a zoomed inset axes in the left figure margin to avoid covering the main curves."""
    fig = parent_ax.get_figure()
    pos = parent_ax.get_position()
    rows = segment_df["current_row_index"].to_numpy(dtype=np.int64)
    frames = segment_df["frame_id"].to_numpy(dtype=np.int64)
    mask = (rows >= ZOOM_ROW_START) & (rows <= ZOOM_ROW_END)
    if not np.any(mask):
        return

    zoom_frames = frames[mask]
    zoom_x0 = float(zoom_frames.min())
    zoom_x1 = float(zoom_frames.max())

    zoom_stack = np.vstack([segment_df[col].to_numpy(dtype=np.float64)[mask] for col, _, _ in cols])
    zoom_y0 = float(np.min(zoom_stack))
    zoom_y1 = float(np.max(zoom_stack))
    y_pad = max((zoom_y1 - zoom_y0) * 0.12, 1.0)
    zoom_y0 -= y_pad
    zoom_y1 += y_pad

    rect = Rectangle(
        (zoom_x0, zoom_y0),
        zoom_x1 - zoom_x0,
        zoom_y1 - zoom_y0,
        fill=False,
        edgecolor="0.45",
        linewidth=0.8,
        linestyle="--",
        zorder=3,
    )
    parent_ax.add_patch(rect)

    inset_margin_x = 0.03
    inset_width = 0.15
    inset_height = pos.height * 0.36
    inset_y = pos.y0 + pos.height * (0.57 if inset_slot == 0 else 0.07)
    axins = fig.add_axes([inset_margin_x, inset_y, inset_width, inset_height], facecolor="white")

    for col, color, dashes in cols:
        vals = segment_df[col].to_numpy(dtype=np.float64)[mask]
        (line,) = axins.plot(zoom_frames, vals, color=color, linewidth=0.7, linestyle="--", alpha=0.95)
        line.set_dashes(dashes)

    axins.set_xlim(zoom_x0, zoom_x1)
    axins.set_ylim(zoom_y0, zoom_y1)
    axins.grid(True, linestyle="--", alpha=0.3)
    axins.tick_params(labelsize=5)
    axins.set_title(f"Zoom rows {ZOOM_ROW_START}-{ZOOM_ROW_END}", fontsize=5.5, pad=2)

    fig.add_artist(
        ConnectionPatch(
            xyA=(zoom_x0, zoom_y1),
            coordsA=parent_ax.transData,
            xyB=(1.0, 1.0),
            coordsB=axins.transAxes,
            color="0.45",
            linewidth=0.7,
        )
    )
    fig.add_artist(
        ConnectionPatch(
            xyA=(zoom_x1, zoom_y0),
            coordsA=parent_ax.transData,
            xyB=(1.0, 0.0),
            coordsB=axins.transAxes,
            color="0.45",
            linewidth=0.7,
        )
    )


def save_comparison_figure(segment_df: pd.DataFrame, output_png: Path) -> None:
    frames = segment_df["frame_id"].to_numpy(dtype=np.int64)
    true_x = segment_df["true_legacy_u_x"].to_numpy(dtype=np.float64)
    true_y = segment_df["true_legacy_u_y"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.6), sharex=True)

    add_series(axes[0], frames, true_x, "True next-step legacy control", "#111111", (6, 2), linewidth=0.9)
    add_series(axes[0], frames, segment_df["dscg_pred_x"], "DSCGNet", "#d62728", (2, 2))
    add_series(axes[0], frames, segment_df["linear_pred_x"], "Linear extrapolation", "#ff7f0e", (5, 1.6))
    add_series(axes[0], frames, segment_df["kalman_pred_x"], "Kalman CV", "#9467bd", (4, 1.5))
    add_series(axes[0], frames, segment_df["gru_pred_x"], "GRU direct", "#2ca02c", (4, 1.5, 1.2, 1.5))
    axes[0].set_ylabel("X-axis control signal")
    axes[0].set_title("Comparison on New Capture Segment: One-Step-Ahead Legacy Control Prediction")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="lower right", fontsize=8, ncol=1)
    axes[0].margins(x=0.01)

    _add_inset(axes[0], segment_df,
               [("true_legacy_u_x", "#111111", (6, 2)),
                ("dscg_pred_x",     "#d62728", (2, 2)),
                ("linear_pred_x",   "#ff7f0e", (5, 1.6)),
                ("kalman_pred_x",   "#9467bd", (4, 1.5)),
                ("gru_pred_x",      "#2ca02c", (4, 1.5, 1.2, 1.5))],
               inset_slot=0)

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

    _add_inset(axes[1], segment_df,
               [("true_legacy_u_y", "#111111", (6, 2)),
                ("dscg_pred_y",     "#d62728", (2, 2)),
                ("linear_pred_y",   "#ff7f0e", (5, 1.6)),
                ("kalman_pred_y",   "#9467bd", (4, 1.5)),
                ("gru_pred_y",      "#2ca02c", (4, 1.5, 1.2, 1.5))],
               inset_slot=1)

    fig.subplots_adjust(left=0.22, right=0.98, hspace=0.15)
    fig.savefig(str(output_png), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()
    paths = resolve_paths()
    raw_df = pd.read_csv(paths["raw_data_csv"]).reset_index(drop=True)
    split_reference = load_split_reference(paths["dscg_csv"])
    best_q, best_r, search_df = tune_kalman_on_val(raw_df, split_reference)
    kalman_df = build_kalman_predictions(raw_df, q_scale=best_q, r_scale=best_r)
    segment_df = build_comparison_table(paths).merge(kalman_df, on="current_row_index", how="left", validate="one_to_one")
    if segment_df["kalman_pred_x"].isna().any() or segment_df["kalman_pred_y"].isna().any():
        raise RuntimeError("Kalman predictions are missing for part of the selected segment.")
    metrics_df = build_metrics_table(segment_df)
    metrics_df.to_csv(paths["metrics_csv"], index=False, encoding="utf-8-sig")
    save_comparison_figure(segment_df, paths["figure_png"])
    top_search = search_df.head(10).to_string(index=False)
    paths["kalman_params_txt"].write_text(
        "\n".join(
            [
                "Kalman CV baseline for one-step-ahead legacy control prediction",
                f"raw_data_csv={paths['raw_data_csv']}",
                f"reference_split_csv={paths['dscg_csv']}",
                f"selected_q_scale={best_q}",
                f"selected_r_scale={best_r}",
                "selection_criterion=lowest avg MAE over validation split",
                "",
                "Top validation candidates:",
                top_search,
            ]
        ),
        encoding="utf-8",
    )

    print(f"Raw data file: {paths['raw_data_csv']}")
    print(f"DSCGNet prediction file: {paths['dscg_csv']}")
    print(f"GRU prediction file: {paths['gru_csv']}")
    print(f"Kalman params file: {paths['kalman_params_txt']}")
    print(f"Selected Kalman params: q_scale={best_q} r_scale={best_r}")
    print(
        "Selected segment: "
        f"current_row_index {SEGMENT_START_CURRENT_ROW_INDEX}->{SEGMENT_END_CURRENT_ROW_INDEX}, "
        f"frame_id {int(segment_df.iloc[0]['frame_id'])}->{int(segment_df.iloc[-1]['frame_id'])}, "
        f"points={SEGMENT_POINT_COUNT}"
    )
    print(f"Comparison figure: {paths['figure_png']}")
    print(f"Comparison metrics: {paths['metrics_csv']}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
