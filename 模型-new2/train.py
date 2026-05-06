import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_data import normalize_data, prepare_sequences
from models.causal_cnn_gru import CausalCNNGRU


AXIS_NAMES = ["dx", "dy"]


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain temporal compensation model from tracker CSV segments.")
    parser.add_argument("--csv-path", default="../data/track-fusion-move-baseline.csv", help="Fallback CSV path.")
    parser.add_argument("--csv-paths", nargs="*", default=None, help="Explicit CSV paths to merge.")
    parser.add_argument("--csv-glob", default=None, help="Optional glob for CSV discovery.")
    parser.add_argument(
        "--signal-x-col",
        default="window8_10_no.3_first_filter_x",
        help="CSV column used as baseline signal x.",
    )
    parser.add_argument(
        "--signal-y-col",
        default="window8_10_no.3_first_filter_y",
        help="CSV column used as baseline signal y.",
    )
    parser.add_argument("--frame-col", default="frame_id", help="CSV frame id column.")
    parser.add_argument("--track-col", default="track_id", help="CSV track id column.")
    parser.add_argument("--history-len", type=int, default=8, help="Temporal history length.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Reference smoothing window for feature generation. Use 1 when CSV input already equals baseline.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed seed. Omit to train without forcing a seed.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--min-segment-len", type=int, default=12)
    parser.add_argument("--keep-frame", choices=["first", "last"], default="last")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for model artifacts.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--sign-loss-weight", type=float, default=0.25)
    parser.add_argument("--bias-loss-weight", type=float, default=0.10)
    parser.add_argument("--mag-loss-weight", type=float, default=0.05)
    parser.add_argument("--augment-axis-flip", dest="augment_axis_flip", action="store_true")
    parser.add_argument("--no-augment-axis-flip", dest="augment_axis_flip", action="store_false")
    parser.set_defaults(augment_axis_flip=True)
    parser.add_argument("--fail-on-collapse", dest="fail_on_collapse", action="store_true")
    parser.add_argument("--no-fail-on-collapse", dest="fail_on_collapse", action="store_false")
    parser.set_defaults(fail_on_collapse=True)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_csv_paths(args):
    paths = []
    if args.csv_paths:
        paths.extend(args.csv_paths)
    if args.csv_glob:
        paths.extend(str(p) for p in sorted(Path(".").glob(args.csv_glob)))
    if not paths:
        paths.append(args.csv_path)

    resolved = []
    seen = set()
    for raw in paths:
        path = Path(raw).resolve()
        if path in seen:
            continue
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        seen.add(path)
        resolved.append(path)
    return resolved


def _to_float(row, key):
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing column value: {key}")
    return float(value)


def _to_int(row, key):
    return int(round(_to_float(row, key)))


def load_signal_segments_from_csv(csv_path, signal_x_col, signal_y_col, frame_col, track_col, keep_frame, min_segment_len):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {signal_x_col, signal_y_col, frame_col, track_col} - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{csv_path.name} missing columns: {sorted(missing)}")

        dedup_rows = []
        pending = None
        pending_frame = None
        for row in reader:
            frame_value = row.get(frame_col, "")
            if frame_value == "":
                continue
            if pending is None:
                pending = row
                pending_frame = frame_value
                continue
            if frame_value == pending_frame:
                if keep_frame == "last":
                    pending = row
                continue
            dedup_rows.append(pending)
            pending = row
            pending_frame = frame_value
        if pending is not None:
            dedup_rows.append(pending)

    segments = []
    current = []
    prev_frame = None
    prev_track = None
    prev_signal = None
    local_segment_idx = 0

    def flush_current():
        nonlocal current, local_segment_idx
        if len(current) >= min_segment_len:
            segment = np.stack(current, axis=0)
            segments.append(
                {
                    "signal": segment,
                    "source": str(csv_path),
                    "source_name": csv_path.name,
                    "segment_index": local_segment_idx,
                    "raw_len": int(len(segment)),
                }
            )
            local_segment_idx += 1
        current = []

    for row in dedup_rows:
        try:
            frame_id = _to_int(row, frame_col)
            track_id = _to_int(row, track_col)
            signal = np.array([_to_float(row, signal_x_col), _to_float(row, signal_y_col)], dtype=np.float32)
        except ValueError:
            flush_current()
            prev_frame = None
            prev_track = None
            prev_signal = None
            continue

        track_changed = prev_track is not None and track_id != prev_track
        frame_reversed = prev_frame is not None and frame_id <= prev_frame
        signal_invalid = not np.all(np.isfinite(signal))
        if track_changed or frame_reversed or signal_invalid:
            flush_current()

        if prev_signal is not None and np.allclose(signal, prev_signal, atol=1e-9) and frame_id == prev_frame:
            continue

        current.append(signal)
        prev_frame = frame_id
        prev_track = track_id
        prev_signal = signal

    flush_current()
    return segments


def load_signal_segments(csv_paths, signal_x_col, signal_y_col, frame_col, track_col, keep_frame, min_segment_len):
    all_segments = []
    for csv_path in csv_paths:
        segments = load_signal_segments_from_csv(
            csv_path=csv_path,
            signal_x_col=signal_x_col,
            signal_y_col=signal_y_col,
            frame_col=frame_col,
            track_col=track_col,
            keep_frame=keep_frame,
            min_segment_len=min_segment_len,
        )
        print(f"{csv_path.name}: loaded {len(segments)} segments")
        all_segments.extend(segments)
    if not all_segments:
        raise RuntimeError("No valid signal segments found for training.")
    return all_segments


def augment_axis_flip(x_seq, y_seq):
    variants = []
    for name, sx, sy in [("orig", 1.0, 1.0), ("flip_x", -1.0, 1.0), ("flip_y", 1.0, -1.0), ("flip_xy", -1.0, -1.0)]:
        x_aug = x_seq.copy()
        y_aug = y_seq.copy()
        x_aug[:, :, 0] *= sx
        x_aug[:, :, 2] *= sx
        y_aug[:, 0] *= sx
        x_aug[:, :, 1] *= sy
        x_aug[:, :, 3] *= sy
        y_aug[:, 1] *= sy
        variants.append((name, x_aug, y_aug))
    return variants


def split_segments(segments, train_ratio, val_ratio):
    total = len(segments)
    if total < 3:
        raise RuntimeError(f"Need at least 3 segments for train/val/test split, got {total}.")
    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio))
    if train_count + val_count >= total:
        val_count = 1
        train_count = max(1, total - 2)
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        else:
            val_count -= 1
    return {
        "train": segments[:train_count],
        "val": segments[train_count : train_count + val_count],
        "test": segments[train_count + val_count :],
    }


def prepare_split_records(segments, window_size, history_len, augment_train):
    split_arrays = {}
    split_meta = {}
    for split_name, split_segments_list in segments.items():
        x_parts = []
        y_parts = []
        meta = []
        for seg in split_segments_list:
            signal = seg["signal"]
            if len(signal) < history_len + 1:
                continue
            x_seq, y_seq, _ = prepare_sequences(signal, window_size=window_size, history_len=history_len)
            if len(x_seq) == 0:
                continue
            if split_name == "train" and augment_train:
                variants = augment_axis_flip(x_seq.astype(np.float32), y_seq.astype(np.float32))
            else:
                variants = [("orig", x_seq.astype(np.float32), y_seq.astype(np.float32))]
            for variant_name, x_variant, y_variant in variants:
                x_parts.append(x_variant)
                y_parts.append(y_variant)
                meta.append(
                    {
                        "source": seg["source"],
                        "source_name": seg["source_name"],
                        "segment_index": seg["segment_index"],
                        "raw_len": seg["raw_len"],
                        "seq_count": int(len(x_variant)),
                        "variant": variant_name,
                    }
                )
        if not x_parts:
            raise RuntimeError(f"{split_name} split is empty after sequence construction.")
        split_arrays[split_name] = {
            "X": np.concatenate(x_parts, axis=0),
            "Y": np.concatenate(y_parts, axis=0),
        }
        split_meta[split_name] = meta
    return split_arrays, split_meta


def compute_sign_stats(arr):
    arr = np.asarray(arr)
    stats = {}
    for axis, axis_name in enumerate(AXIS_NAMES):
        axis_arr = arr[:, axis]
        stats[axis_name] = {
            "pos": int(np.sum(axis_arr > 0)),
            "neg": int(np.sum(axis_arr < 0)),
            "zero": int(np.sum(np.abs(axis_arr) < 1e-12)),
            "min": float(np.min(axis_arr)),
            "max": float(np.max(axis_arr)),
            "mean": float(np.mean(axis_arr)),
        }
    return stats


def find_collapse_axes(preds, targets):
    collapsed = []
    for axis, axis_name in enumerate(AXIS_NAMES):
        pred_axis = preds[:, axis]
        target_axis = targets[:, axis]
        target_has_both = np.any(target_axis > 0) and np.any(target_axis < 0)
        pred_has_both = np.any(pred_axis > 0) and np.any(pred_axis < 0)
        if target_has_both and not pred_has_both:
            collapsed.append(axis_name)
    return collapsed


def evaluate_predictions(model, loader, device):
    model.eval()
    huber = nn.HuberLoss()
    total_huber = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            total_huber += huber(pred, y_batch).item()
            preds.append(pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    preds = np.concatenate(preds, axis=0) if preds else np.zeros((0, 2), dtype=np.float32)
    targets = np.concatenate(targets, axis=0) if targets else np.zeros((0, 2), dtype=np.float32)
    mae = float(np.mean(np.abs(preds - targets))) if len(preds) else 0.0
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2))) if len(preds) else 0.0
    avg_huber = total_huber / max(1, len(loader))
    return {
        "huber": avg_huber,
        "mae": mae,
        "rmse": rmse,
        "preds": preds,
        "targets": targets,
        "pred_sign": compute_sign_stats(preds) if len(preds) else {},
        "target_sign": compute_sign_stats(targets) if len(targets) else {},
        "collapsed_axes": find_collapse_axes(preds, targets) if len(preds) else [],
    }


def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, patience, sign_loss_weight, bias_loss_weight, mag_loss_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    huber = nn.HuberLoss()

    best_val = float("inf")
    best_collapse = 1_000_000
    best_state = copy.deepcopy(model.state_dict())
    best_eval = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_pred = 0.0
        train_mag = 0.0
        train_sign = 0.0
        train_bias = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)

            loss_pred = huber(pred, y_batch)
            loss_mag = torch.abs(pred).mean()
            loss_sign = torch.relu(-(pred * y_batch)).mean()
            loss_bias = torch.abs(pred.mean(dim=0) - y_batch.mean(dim=0)).mean()
            loss = loss_pred + mag_loss_weight * loss_mag + sign_loss_weight * loss_sign + bias_loss_weight * loss_bias

            loss.backward()
            optimizer.step()

            train_total += loss.item()
            train_pred += loss_pred.item()
            train_mag += loss_mag.item()
            train_sign += loss_sign.item()
            train_bias += loss_bias.item()

        val_eval = evaluate_predictions(model, val_loader, device)
        avg_train_total = train_total / max(1, len(train_loader))
        avg_train_pred = train_pred / max(1, len(train_loader))
        avg_train_mag = train_mag / max(1, len(train_loader))
        avg_train_sign = train_sign / max(1, len(train_loader))
        avg_train_bias = train_bias / max(1, len(train_loader))
        collapse_count = len(val_eval["collapsed_axes"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_total": avg_train_total,
                "train_pred": avg_train_pred,
                "train_mag": avg_train_mag,
                "train_sign": avg_train_sign,
                "train_bias": avg_train_bias,
                "val_huber": val_eval["huber"],
                "val_mae": val_eval["mae"],
                "val_rmse": val_eval["rmse"],
                "val_collapsed_axes": val_eval["collapsed_axes"],
                "val_pred_sign": val_eval["pred_sign"],
            }
        )

        better = collapse_count < best_collapse or (collapse_count == best_collapse and val_eval["huber"] < best_val)
        if better:
            best_collapse = collapse_count
            best_val = val_eval["huber"]
            best_state = copy.deepcopy(model.state_dict())
            best_eval = val_eval
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}: train={avg_train_total:.6f} val_huber={val_eval['huber']:.6f} "
                f"val_mae={val_eval['mae']:.6f} collapse={val_eval['collapsed_axes']}"
            )

    model.load_state_dict(best_state)
    return model, history, best_eval


def export_onnx(model, onnx_path, seq_len, input_dim, opset):
    model_cpu = copy.deepcopy(model).cpu().eval()
    dummy = torch.zeros(1, seq_len, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["delta_cmd"],
        dynamic_axes={"input": {0: "B"}, "delta_cmd": {0: "B"}},
        opset_version=opset,
    )


def save_stats_json(path, stats, seq_len, input_dim, window_size):
    payload = {
        "mean": [float(v) for v in stats["mean"]],
        "std": [float(v) for v in stats["std"]],
        "seq_len": int(seq_len),
        "input_dim": int(input_dim),
        "window_size": int(window_size),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using fixed seed: {args.seed}")
    else:
        print("Using actual data split without forcing a random seed.")
    csv_paths = resolve_csv_paths(args)
    print(f"Loading {len(csv_paths)} CSV files")

    segments = load_signal_segments(
        csv_paths=csv_paths,
        signal_x_col=args.signal_x_col,
        signal_y_col=args.signal_y_col,
        frame_col=args.frame_col,
        track_col=args.track_col,
        keep_frame=args.keep_frame,
        min_segment_len=args.min_segment_len,
    )
    print(f"Total segments: {len(segments)}")

    split_segments_map = split_segments(segments, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    split_arrays, split_meta = prepare_split_records(
        segments=split_segments_map,
        window_size=args.window_size,
        history_len=args.history_len,
        augment_train=args.augment_axis_flip,
    )

    data_dict = {
        "X_train": split_arrays["train"]["X"],
        "Y_train": split_arrays["train"]["Y"],
        "X_val": split_arrays["val"]["X"],
        "Y_val": split_arrays["val"]["Y"],
        "X_test": split_arrays["test"]["X"],
        "Y_test": split_arrays["test"]["Y"],
    }
    data_dict, stats = normalize_data(data_dict)

    train_dataset = TensorDataset(torch.from_numpy(data_dict["X_train"]).float(), torch.from_numpy(data_dict["Y_train"]).float())
    val_dataset = TensorDataset(torch.from_numpy(data_dict["X_val"]).float(), torch.from_numpy(data_dict["Y_val"]).float())
    test_dataset = TensorDataset(torch.from_numpy(data_dict["X_test"]).float(), torch.from_numpy(data_dict["Y_test"]).float())

    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, max(1, len(train_dataset))), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, max(1, len(val_dataset))))
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, max(1, len(test_dataset))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CausalCNNGRU(input_dim=4, hidden_dim=args.hidden_dim, output_dim=2)
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

    if args.fail_on_collapse and best_eval and best_eval["collapsed_axes"]:
        raise RuntimeError(f"Validation sign collapse detected on axes: {best_eval['collapsed_axes']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_eval = evaluate_predictions(model, test_loader, device)

    pth_path = output_dir / "causal_cnn_gru.pth"
    onnx_path = output_dir / "causal_cnn_gru.onnx"
    stats_path = output_dir / "stats.json"
    summary_path = output_dir / "train_summary.json"
    history_path = output_dir / "train_history.json"
    data_path = output_dir / "data.npz"

    torch.save(model.cpu().state_dict(), pth_path)
    export_onnx(model, onnx_path, seq_len=args.history_len, input_dim=4, opset=args.opset)
    save_stats_json(stats_path, stats, seq_len=args.history_len, input_dim=4, window_size=args.window_size)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez(
        data_path,
        **data_dict,
        X_all=np.concatenate([split_arrays["train"]["X"], split_arrays["val"]["X"], split_arrays["test"]["X"]], axis=0),
        Y_all=np.concatenate([split_arrays["train"]["Y"], split_arrays["val"]["Y"], split_arrays["test"]["Y"]], axis=0),
    )

    summary = {
        "csv_paths": [str(p) for p in csv_paths],
        "signal_columns": [args.signal_x_col, args.signal_y_col],
        "segment_count": len(segments),
        "split_segment_count": {k: len(v) for k, v in split_segments_map.items()},
        "split_meta": split_meta,
        "sequence_shape": {
            "X_train": list(data_dict["X_train"].shape),
            "X_val": list(data_dict["X_val"].shape),
            "X_test": list(data_dict["X_test"].shape),
        },
        "stats": {
            "mean": [float(v) for v in stats["mean"]],
            "std": [float(v) for v in stats["std"]],
        },
        "best_val": {
            "huber": best_eval["huber"],
            "mae": best_eval["mae"],
            "rmse": best_eval["rmse"],
            "collapsed_axes": best_eval["collapsed_axes"],
            "pred_sign": best_eval["pred_sign"],
            "target_sign": best_eval["target_sign"],
        },
        "test_metrics": {
            "huber": test_eval["huber"],
            "mae": test_eval["mae"],
            "rmse": test_eval["rmse"],
            "collapsed_axes": test_eval["collapsed_axes"],
            "pred_sign": test_eval["pred_sign"],
            "target_sign": test_eval["target_sign"],
        },
        "artifacts": {
            "pth": str(pth_path.resolve()),
            "onnx": str(onnx_path.resolve()),
            "stats": str(stats_path.resolve()),
            "history": str(history_path.resolve()),
            "npz": str(data_path.resolve()),
        },
        "args": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete")
    print(json.dumps(summary["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
