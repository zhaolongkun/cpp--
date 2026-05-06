import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.ablation_variants import NoAttnCausalCNNGRU, NoConvGRUAttn, SingleStreamCausalCNNGRUAttn
from models.causal_cnn_gru import CausalCNNGRU
from prepare_data import prepare_sequences
from train import evaluate_predictions, load_signal_segments, normalize_data, prepare_split_records, split_segments, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper experiments on the baseline CSV dataset.")
    parser.add_argument(
        "--csv-path",
        default="../data/track-fusion-move-baseline.csv",
        help="Path to the collected baseline CSV.",
    )
    parser.add_argument("--signal-x-col", default="window8_10_no.3_first_filter_x")
    parser.add_argument("--signal-y-col", default="window8_10_no.3_first_filter_y")
    parser.add_argument("--frame-col", default="frame_id")
    parser.add_argument("--track-col", default="track_id")
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--min-segment-len", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--sign-loss-weight", type=float, default=0.25)
    parser.add_argument("--bias-loss-weight", type=float, default=0.10)
    parser.add_argument("--mag-loss-weight", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.7, help="Bounded fusion weight used by the deployed runtime.")
    parser.add_argument("--delta-max", type=float, default=10.0, help="Bounded fusion clipping threshold.")
    parser.add_argument(
        "--output-dir",
        default="../paper/仪器与测量汇刊/new2/results_actual",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--main-checkpoint",
        default="checkpoints/causal_cnn_gru.pth",
        help="Existing main model checkpoint. Retrained only when missing.",
    )
    return parser.parse_args()


def compute_metrics(u, future_ref, threshold=10.0):
    diff = u - future_ref
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    if len(u) > 1:
        delta = u[1:] - u[:-1]
        jitter = float(np.mean(np.abs(delta)))
        spike_rate = float(np.mean(np.abs(delta) > threshold))
        sign_flip_x = float(np.mean((u[1:, 0] * u[:-1, 0]) < 0))
        sign_flip_y = float(np.mean((u[1:, 1] * u[:-1, 1]) < 0))
        sign_flip = 0.5 * (sign_flip_x + sign_flip_y)
    else:
        jitter = 0.0
        spike_rate = 0.0
        sign_flip = 0.0
    return {
        "mae_future": mae,
        "rmse_future": rmse,
        "jitter": jitter,
        "spike_rate": spike_rate,
        "sign_flip": float(sign_flip),
    }


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def build_data(args):
    csv_path = Path(args.csv_path).resolve()
    segments = load_signal_segments(
        csv_paths=[csv_path],
        signal_x_col=args.signal_x_col,
        signal_y_col=args.signal_y_col,
        frame_col=args.frame_col,
        track_col=args.track_col,
        keep_frame="last",
        min_segment_len=args.min_segment_len,
    )
    split_map = split_segments(segments, train_ratio=0.7, val_ratio=0.15)
    split_arrays, split_meta = prepare_split_records(
        segments=split_map,
        window_size=args.window_size,
        history_len=args.history_len,
        augment_train=True,
    )
    data_dict = {
        "X_train": split_arrays["train"]["X"].copy(),
        "Y_train": split_arrays["train"]["Y"].copy(),
        "X_val": split_arrays["val"]["X"].copy(),
        "Y_val": split_arrays["val"]["Y"].copy(),
        "X_test": split_arrays["test"]["X"].copy(),
        "Y_test": split_arrays["test"]["Y"].copy(),
    }
    data_dict, stats = normalize_data(data_dict)
    return csv_path, segments, split_map, split_meta, data_dict, stats


def build_eval_sequences(signal, stats, history_len, window_size):
    x_raw, _, e_ref = prepare_sequences(signal, window_size=window_size, history_len=history_len)
    mean = stats["mean"].reshape(1, 1, -1)
    std = stats["std"].reshape(1, 1, -1)
    x_norm = (x_raw - mean) / std
    base_ref = e_ref[history_len - 1 : -1]
    future_ref = e_ref[history_len:]
    raw_curr = signal[history_len - 1 : -1]
    return {
        "x_norm": x_norm.astype(np.float32),
        "base_ref": base_ref.astype(np.float32),
        "future_ref": future_ref.astype(np.float32),
        "raw_curr": raw_curr.astype(np.float32),
        "signal": signal.astype(np.float32),
    }


def fuse_delta(base_ref, delta_pred, alpha, delta_max):
    return base_ref + alpha * np.clip(delta_pred, -delta_max, delta_max)


def kalman_predict_next(signal, q_scale, r_scale):
    dt = 1.0
    f = np.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    q = q_scale * np.array(
        [[0.25, 0.0, 0.5, 0.0], [0.0, 0.25, 0.0, 0.5], [0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]],
        dtype=np.float64,
    )
    r = r_scale * np.eye(2, dtype=np.float64)
    x = np.zeros(4, dtype=np.float64)
    x[:2] = signal[0]
    if len(signal) > 1:
        x[2:] = signal[1] - signal[0]
    p = np.eye(4, dtype=np.float64) * 10.0
    out = np.zeros_like(signal, dtype=np.float64)
    for i, z in enumerate(signal):
        x = f @ x
        p = f @ p @ f.T + q
        y = z - (h @ x)
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)
        x = x + k @ y
        p = (np.eye(4) - k @ h) @ p
        out[i] = (h @ (f @ x))
    return out.astype(np.float32)


def smith_predict_next(signal, a, b, beta):
    signal = signal.astype(np.float64)
    out = np.zeros_like(signal, dtype=np.float64)
    model = signal[0].copy()
    prev = signal[0].copy()
    for i in range(len(signal)):
        vel = signal[i] - prev if i > 0 else np.zeros(2, dtype=np.float64)
        model_next = a * model + (1.0 - a) * signal[i] + b * vel
        out[i] = model_next + beta * (signal[i] - model)
        prev = signal[i]
        model = model_next
    return out.astype(np.float32)


def tune_kalman(val_eval):
    best = None
    for q in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        for r in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            pred = kalman_predict_next(val_eval["signal"], q, r)[len(val_eval["signal"]) - len(val_eval["future_ref"]) - 1 : -1]
            metrics = compute_metrics(pred, val_eval["future_ref"])
            key = (metrics["mae_future"], metrics["jitter"])
            if best is None or key < best["key"]:
                best = {"q": q, "r": r, "metrics": metrics, "key": key}
    return best


def tune_smith(val_eval):
    best = None
    for a in [0.2, 0.4, 0.6, 0.8]:
        for b in [0.0, 0.3, 0.6, 1.0]:
            for beta in [0.0, 0.3, 0.6, 1.0]:
                pred = smith_predict_next(val_eval["signal"], a, b, beta)[
                    len(val_eval["signal"]) - len(val_eval["future_ref"]) - 1 : -1
                ]
                metrics = compute_metrics(pred, val_eval["future_ref"])
                key = (metrics["mae_future"], metrics["jitter"])
                if best is None or key < best["key"]:
                    best = {"a": a, "b": b, "beta": beta, "metrics": metrics, "key": key}
    return best


def align_predictions(full_prediction, future_ref):
    start = len(full_prediction) - len(future_ref) - 1
    return full_prediction[start:-1]


def make_dataloaders(data_dict, batch_size):
    train_dataset = TensorDataset(torch.from_numpy(data_dict["X_train"]).float(), torch.from_numpy(data_dict["Y_train"]).float())
    val_dataset = TensorDataset(torch.from_numpy(data_dict["X_val"]).float(), torch.from_numpy(data_dict["Y_val"]).float())
    test_dataset = TensorDataset(torch.from_numpy(data_dict["X_test"]).float(), torch.from_numpy(data_dict["Y_test"]).float())
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, max(1, len(train_dataset))), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, max(1, len(val_dataset))))
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, max(1, len(test_dataset))))
    return train_loader, val_loader, test_loader


def infer_delta(model, x_norm, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_norm).float().to(device)
        return model(x_tensor).cpu().numpy()


def train_variant(name, model, data_dict, args, output_dir):
    train_loader, val_loader, test_loader = make_dataloaders(data_dict, args.batch_size)
    model, history, best_eval = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        sign_loss_weight=args.sign_loss_weight,
        bias_loss_weight=args.bias_loss_weight,
        mag_loss_weight=args.mag_loss_weight,
    )
    device = torch.device("cpu")
    test_eval = evaluate_predictions(model.to(device), test_loader, device)
    ckpt_path = output_dir / f"{name}.pth"
    torch.save(model.cpu().state_dict(), ckpt_path)
    return {
        "model": model.cpu(),
        "history": history,
        "best_val": best_eval,
        "test_eval": test_eval,
        "checkpoint": str(ckpt_path),
    }


def maybe_load_main_model(args, data_dict, output_dir):
    ckpt_path = Path(args.main_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (Path(__file__).resolve().parent / ckpt_path).resolve()
    model = CausalCNNGRU(input_dim=4, hidden_dim=args.hidden_dim, output_dim=2)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        return {"model": model, "checkpoint": str(ckpt_path), "retrained": False}
    trained = train_variant("causal_cnn_gru", model, data_dict, args, output_dir)
    return {"model": trained["model"], "checkpoint": trained["checkpoint"], "retrained": True}


def save_case_figure(method_curves, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    future_ref = method_curves["future_ref"]
    velocity = np.abs(np.diff(future_ref, axis=0)).sum(axis=1)
    center = int(np.argmax(velocity))
    start = max(0, center - 35)
    stop = min(len(future_ref), start + 80)
    t = np.arange(start, stop)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for axis_idx, axis_name in enumerate(["x", "y"]):
        axes[axis_idx].plot(t, future_ref[start:stop, axis_idx], label="Future Ref", linewidth=2.0, color="#111111")
        axes[axis_idx].plot(t, method_curves["Ref Only"][start:stop, axis_idx], label="Ref Only", linewidth=1.5)
        axes[axis_idx].plot(t, method_curves["Kalman"][start:stop, axis_idx], label="Kalman", linewidth=1.2)
        axes[axis_idx].plot(t, method_curves["Smith"][start:stop, axis_idx], label="Smith", linewidth=1.2)
        axes[axis_idx].plot(t, method_curves["Ours"][start:stop, axis_idx], label="Ours", linewidth=1.6)
        axes[axis_idx].set_ylabel(f"{axis_name}-axis")
        axes[axis_idx].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_xlabel("frame index")
    axes[0].legend(ncol=5, fontsize=9, loc="upper center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path, segments, split_map, split_meta, data_dict, stats = build_data(args)
    val_signal = split_map["val"][0]["signal"]
    test_signal = split_map["test"][0]["signal"]
    val_eval = build_eval_sequences(val_signal, stats, args.history_len, args.window_size)
    test_eval = build_eval_sequences(test_signal, stats, args.history_len, args.window_size)

    device = torch.device("cpu")
    main_model_info = maybe_load_main_model(args, data_dict, output_dir)
    main_delta = infer_delta(main_model_info["model"], test_eval["x_norm"], device)
    ours_curve = fuse_delta(test_eval["base_ref"], main_delta, args.alpha, args.delta_max)

    kalman_cfg = tune_kalman(val_eval)
    smith_cfg = tune_smith(val_eval)
    kalman_curve = align_predictions(kalman_predict_next(test_eval["signal"], kalman_cfg["q"], kalman_cfg["r"]), test_eval["future_ref"])
    smith_curve = align_predictions(
        smith_predict_next(test_eval["signal"], smith_cfg["a"], smith_cfg["b"], smith_cfg["beta"]),
        test_eval["future_ref"],
    )

    main_comparison = {
        "Ref Only": compute_metrics(test_eval["base_ref"], test_eval["future_ref"]),
        "Kalman": compute_metrics(kalman_curve, test_eval["future_ref"]),
        "Smith": compute_metrics(smith_curve, test_eval["future_ref"]),
        "Prediction Only": compute_metrics(test_eval["base_ref"] + main_delta, test_eval["future_ref"]),
        "Ours": compute_metrics(ours_curve, test_eval["future_ref"]),
    }

    variants = {
        "full_model": {
            "display": "Full model",
            "model": main_model_info["model"],
            "checkpoint": main_model_info["checkpoint"],
            "params": count_params(main_model_info["model"]),
            "delta": main_delta,
        }
    }

    trained_variants = {
        "no_attn": (
            "w/o temporal attention",
            NoAttnCausalCNNGRU(input_dim=4, hidden_dim=args.hidden_dim, output_dim=2),
        ),
        "no_conv": (
            "w/o causal conv",
            NoConvGRUAttn(input_dim=4, hidden_dim=args.hidden_dim, output_dim=2, max_len=args.history_len),
        ),
        "single_stream": (
            "w/o branch split",
            SingleStreamCausalCNNGRUAttn(
                input_dim=4, hidden_dim=args.hidden_dim, output_dim=2, max_len=args.history_len
            ),
        ),
    }

    module_ablation = {}
    for key, (display_name, model) in trained_variants.items():
        print(f"Training ablation variant: {display_name}")
        trained = train_variant(key, model, data_dict, args, output_dir)
        delta_pred = infer_delta(trained["model"], test_eval["x_norm"], device)
        fused = fuse_delta(test_eval["base_ref"], delta_pred, args.alpha, args.delta_max)
        module_ablation[display_name] = {
            "mae_future": compute_metrics(fused, test_eval["future_ref"])["mae_future"],
            "rmse_future": compute_metrics(fused, test_eval["future_ref"])["rmse_future"],
            "jitter": compute_metrics(fused, test_eval["future_ref"])["jitter"],
            "params": count_params(trained["model"]),
            "checkpoint": trained["checkpoint"],
            "test_huber": trained["test_eval"]["huber"],
            "test_mae_delta": trained["test_eval"]["mae"],
            "test_rmse_delta": trained["test_eval"]["rmse"],
        }

    module_ablation["Full model"] = {
        "mae_future": main_comparison["Ours"]["mae_future"],
        "rmse_future": main_comparison["Ours"]["rmse_future"],
        "jitter": main_comparison["Ours"]["jitter"],
        "params": count_params(main_model_info["model"]),
        "checkpoint": main_model_info["checkpoint"],
    }

    method_curves = {
        "future_ref": test_eval["future_ref"],
        "Ref Only": test_eval["base_ref"],
        "Kalman": kalman_curve,
        "Smith": smith_curve,
        "Ours": ours_curve,
    }
    fig_case = output_dir / "fig_actual_case.png"
    save_case_figure(method_curves, fig_case)

    dataset_summary = {
        "csv_path": str(csv_path),
        "raw_rows": int(sum(seg["raw_len"] for seg in segments)),
        "segment_lengths": [int(seg["raw_len"]) for seg in segments],
        "segment_count": len(segments),
        "split_segment_count": {k: len(v) for k, v in split_map.items()},
        "split_meta": split_meta,
        "sequence_shape": {
            "X_train": list(data_dict["X_train"].shape),
            "X_val": list(data_dict["X_val"].shape),
            "X_test": list(data_dict["X_test"].shape),
        },
        "history_len": args.history_len,
        "window_size": args.window_size,
        "alpha": args.alpha,
        "delta_max": args.delta_max,
    }

    results = {
        "dataset_summary": dataset_summary,
        "kalman_best": kalman_cfg,
        "smith_best": smith_cfg,
        "main_model": {
            "checkpoint": main_model_info["checkpoint"],
            "retrained": main_model_info["retrained"],
            "params": count_params(main_model_info["model"]),
        },
        "main_comparison": main_comparison,
        "module_ablation": module_ablation,
        "artifacts": {
            "case_figure": str(fig_case),
        },
    }

    results_path = output_dir / "paper_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
