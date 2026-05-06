import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.causal_cnn_gru import CausalCNNGRU
from run_paper_experiments import (
    align_predictions,
    build_data,
    build_eval_sequences,
    compute_metrics,
    infer_delta,
    kalman_predict_next,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results_actual"
PAPER_DIR = ROOT_DIR / "paper" / "仪器与测量汇刊" / "new2"
PICTURE_DIR = PAPER_DIR / "picture"


def load_summary():
    summary_path = RESULTS_DIR / "final_paper_results.json"
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_args():
    return SimpleNamespace(
        csv_path=str(ROOT_DIR / "data" / "track-fusion-move-baseline.csv"),
        signal_x_col="window8_10_no.3_first_filter_x",
        signal_y_col="window8_10_no.3_first_filter_y",
        frame_col="frame_id",
        track_col="track_id",
        history_len=8,
        window_size=1,
        min_segment_len=12,
        batch_size=128,
        epochs=220,
        patience=60,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=64,
        sign_loss_weight=0.25,
        bias_loss_weight=0.10,
        mag_loss_weight=0.05,
        alpha=1.0,
        delta_max=10.0,
    )


def simple_smith_predict_next(signal, a):
    signal = signal.astype(np.float32)
    pred = np.empty_like(signal)
    pred[0] = signal[0]
    if len(signal) > 1:
        pred[1:] = signal[1:] + a * (signal[1:] - signal[:-1])
    return pred


def tune_alpha(base_ref, delta_pred, future_ref, delta_max):
    best = None
    for alpha in np.linspace(0.1, 1.0, 10):
        fused = base_ref + alpha * np.clip(delta_pred, -delta_max, delta_max)
        metrics = compute_metrics(fused, future_ref)
        key = (metrics["mae_future"], metrics["rmse_future"])
        if best is None or key < best["key"]:
            best = {"alpha": float(alpha), "metrics": metrics, "key": key}
    return best


def tune_simple_smith(signal, future_ref):
    best = None
    for a in np.linspace(0.0, 1.0, 21):
        pred = align_predictions(simple_smith_predict_next(signal, float(a)), future_ref)
        metrics = compute_metrics(pred, future_ref)
        key = (metrics["mae_future"], metrics["rmse_future"])
        if best is None or key < best["key"]:
            best = {"a": float(a), "metrics": metrics, "key": key}
    return best


def tune_kalman_pair(signal, future_ref):
    best = None
    for q in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        for r in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            pred = align_predictions(kalman_predict_next(signal, q, r), future_ref)
            metrics = compute_metrics(pred, future_ref)
            key = (metrics["mae_future"], metrics["rmse_future"])
            if best is None or key < best["key"]:
                best = {"q": float(q), "r": float(r), "metrics": metrics, "key": key}
    return best


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path, content):
    path.write_text(content, encoding="utf-8")


def save_actual_case_figure(target_path, curves):
    future_ref = curves["future_ref"]
    velocity = np.abs(np.diff(future_ref, axis=0)).sum(axis=1)
    center = int(np.argmax(velocity))
    start = max(0, center - 35)
    stop = min(len(future_ref), start + 80)
    t = np.arange(start, stop)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for axis_idx, axis_name in enumerate(["x", "y"]):
        axes[axis_idx].plot(t, future_ref[start:stop, axis_idx], label="Future Ref", linewidth=2.0, color="#111111")
        axes[axis_idx].plot(t, curves["Ref Only"][start:stop, axis_idx], label="Ref Only", linewidth=1.5)
        axes[axis_idx].plot(t, curves["Kalman"][start:stop, axis_idx], label="Kalman", linewidth=1.2)
        axes[axis_idx].plot(t, curves["Smith"][start:stop, axis_idx], label="Smith", linewidth=1.2)
        axes[axis_idx].plot(t, curves["Ours"][start:stop, axis_idx], label="Ours", linewidth=1.6)
        axes[axis_idx].set_ylabel(f"{axis_name}-axis")
        axes[axis_idx].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_xlabel("frame index")
    axes[0].legend(ncol=5, fontsize=9, loc="upper center")
    plt.tight_layout()
    plt.savefig(target_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"start": int(start), "stop": int(stop), "center": int(center)}


def save_main_comparison_figure(target_path, summary):
    methods = ["Ref Only", "Kalman", "Smith", "Ours"]
    mae = [summary["main_comparison"][name]["mae_future"] for name in methods]
    rmse = [summary["main_comparison"][name]["rmse_future"] for name in methods]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - width / 2, mae, width, label="MAE", color="#0f766e")
    ax.bar(x + width / 2, rmse, width, label="RMSE", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=10)
    ax.set_ylabel("future-frame error")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(target_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_module_ablation_figure(target_path, summary):
    names = ["Full model", "w/o temporal attention", "w/o causal conv", "w/o branch split"]
    mae = [summary["module_ablation"][name]["mae_future"] for name in names]
    params = [summary["module_ablation"][name]["params"] / 1000.0 for name in names]

    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(9.0, 4.6))
    ax1.bar(x, mae, color="#1d4ed8", width=0.55)
    ax1.set_ylabel("future-frame MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=10)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, params, color="#dc2626", marker="o", linewidth=2.0)
    ax2.set_ylabel("parameters (K)")

    plt.tight_layout()
    plt.savefig(target_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_actual_case_assets():
    args = build_args()
    _, _, split_map, _, data_dict, stats = build_data(args)
    val_signal = split_map["val"][0]["signal"]
    test_signal = split_map["test"][0]["signal"]
    val_eval = build_eval_sequences(val_signal, stats, args.history_len, args.window_size)
    test_eval = build_eval_sequences(test_signal, stats, args.history_len, args.window_size)

    model = CausalCNNGRU(input_dim=4, hidden_dim=args.hidden_dim, output_dim=2)
    ckpt_path = RESULTS_DIR / "full_model_retrained.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    val_delta = infer_delta(model, val_eval["x_norm"], torch.device("cpu"))
    alpha_cfg = tune_alpha(val_eval["base_ref"], val_delta, val_eval["future_ref"], args.delta_max)
    alpha = alpha_cfg["alpha"]

    kalman_cfg = tune_kalman_pair(val_eval["signal"], val_eval["future_ref"])
    smith_cfg = tune_simple_smith(val_eval["signal"], val_eval["future_ref"])

    test_delta = infer_delta(model, test_eval["x_norm"], torch.device("cpu"))
    ours_curve = test_eval["base_ref"] + alpha * np.clip(test_delta, -args.delta_max, args.delta_max)
    kalman_curve = align_predictions(kalman_predict_next(test_eval["signal"], kalman_cfg["q"], kalman_cfg["r"]), test_eval["future_ref"])
    smith_curve = align_predictions(simple_smith_predict_next(test_eval["signal"], smith_cfg["a"]), test_eval["future_ref"])

    curves = {
        "future_ref": test_eval["future_ref"],
        "Ref Only": test_eval["base_ref"],
        "Kalman": kalman_curve,
        "Smith": smith_curve,
        "Ours": ours_curve,
    }

    figure_path = PICTURE_DIR / "fig_actual_case.png"
    window_meta = save_actual_case_figure(figure_path, curves)

    folder = ensure_dir(PICTURE_DIR / "fig_actual_case")
    np.savez(
        folder / "fig_actual_case_data.npz",
        future_ref=curves["future_ref"],
        ref_only=curves["Ref Only"],
        kalman=curves["Kalman"],
        smith=curves["Smith"],
        ours=curves["Ours"],
        start=np.array(window_meta["start"], dtype=np.int32),
        stop=np.array(window_meta["stop"], dtype=np.int32),
    )
    write_text(
        folder / "fig_actual_case_meta.json",
        json.dumps(
            {
                "alpha": alpha,
                "delta_max": args.delta_max,
                "kalman_q": kalman_cfg["q"],
                "kalman_r": kalman_cfg["r"],
                "smith_a": smith_cfg["a"],
                "window": window_meta,
            },
            indent=2,
        ),
    )
    write_text(
        folder / "fig_actual_case.py",
        textwrap.dedent(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np


            base_dir = Path(__file__).resolve().parent
            data = np.load(base_dir / "fig_actual_case_data.npz")

            start = int(data["start"])
            stop = int(data["stop"])
            t = np.arange(start, stop)

            curves = {
                "Future Ref": data["future_ref"],
                "Ref Only": data["ref_only"],
                "Kalman": data["kalman"],
                "Smith": data["smith"],
                "Ours": data["ours"],
            }

            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            for axis_idx, axis_name in enumerate(["x", "y"]):
                axes[axis_idx].plot(t, curves["Future Ref"][start:stop, axis_idx], label="Future Ref", linewidth=2.0, color="#111111")
                axes[axis_idx].plot(t, curves["Ref Only"][start:stop, axis_idx], label="Ref Only", linewidth=1.5)
                axes[axis_idx].plot(t, curves["Kalman"][start:stop, axis_idx], label="Kalman", linewidth=1.2)
                axes[axis_idx].plot(t, curves["Smith"][start:stop, axis_idx], label="Smith", linewidth=1.2)
                axes[axis_idx].plot(t, curves["Ours"][start:stop, axis_idx], label="Ours", linewidth=1.6)
                axes[axis_idx].set_ylabel(f"{axis_name}-axis")
                axes[axis_idx].grid(True, linestyle="--", alpha=0.35)
            axes[1].set_xlabel("frame index")
            axes[0].legend(ncol=5, fontsize=9, loc="upper center")
            plt.tight_layout()
            plt.savefig(base_dir.parent / "fig_actual_case.png", dpi=220, bbox_inches="tight")
            """
        ).strip()
        + "\n",
    )


def export_summary_figure_assets(summary, figure_name, data_payload, figure_fn, script_body):
    figure_path = PICTURE_DIR / f"{figure_name}.png"
    figure_fn(figure_path, summary)
    folder = ensure_dir(PICTURE_DIR / figure_name)
    write_text(folder / f"{figure_name}_data.json", json.dumps(data_payload, indent=2))
    write_text(folder / f"{figure_name}.py", script_body.strip() + "\n")


def export_main_comparison_assets(summary):
    data_payload = {
        "methods": ["Ref Only", "Kalman", "Smith", "Ours"],
        "main_comparison": {k: summary["main_comparison"][k] for k in ["Ref Only", "Kalman", "Smith", "Ours"]},
    }
    script = textwrap.dedent(
        """
        import json
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np


        base_dir = Path(__file__).resolve().parent
        data = json.loads((base_dir / "fig_main_comparison_data.json").read_text(encoding="utf-8"))
        methods = data["methods"]
        mae = [data["main_comparison"][name]["mae_future"] for name in methods]
        rmse = [data["main_comparison"][name]["rmse_future"] for name in methods]

        x = np.arange(len(methods))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.bar(x - width / 2, mae, width, label="MAE", color="#0f766e")
        ax.bar(x + width / 2, rmse, width, label="RMSE", color="#f59e0b")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=10)
        ax.set_ylabel("future-frame error")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(base_dir.parent / "fig_main_comparison.png", dpi=220, bbox_inches="tight")
        """
    )
    export_summary_figure_assets(summary, "fig_main_comparison", data_payload, save_main_comparison_figure, script)


def export_module_ablation_assets(summary):
    names = ["Full model", "w/o temporal attention", "w/o causal conv", "w/o branch split"]
    data_payload = {
        "names": names,
        "module_ablation": {k: summary["module_ablation"][k] for k in names},
    }
    script = textwrap.dedent(
        """
        import json
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np


        base_dir = Path(__file__).resolve().parent
        data = json.loads((base_dir / "fig_module_ablation_data.json").read_text(encoding="utf-8"))
        names = data["names"]
        mae = [data["module_ablation"][name]["mae_future"] for name in names]
        params = [data["module_ablation"][name]["params"] / 1000.0 for name in names]

        x = np.arange(len(names))
        fig, ax1 = plt.subplots(figsize=(9.0, 4.6))
        ax1.bar(x, mae, color="#1d4ed8", width=0.55)
        ax1.set_ylabel("future-frame MAE")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=10)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x, params, color="#dc2626", marker="o", linewidth=2.0)
        ax2.set_ylabel("parameters (K)")

        plt.tight_layout()
        plt.savefig(base_dir.parent / "fig_module_ablation.png", dpi=220, bbox_inches="tight")
        """
    )
    export_summary_figure_assets(summary, "fig_module_ablation", data_payload, save_module_ablation_figure, script)


def main():
    ensure_dir(PICTURE_DIR)
    summary = load_summary()
    export_actual_case_assets()
    export_main_comparison_assets(summary)
    export_module_ablation_assets(summary)


if __name__ == "__main__":
    main()
