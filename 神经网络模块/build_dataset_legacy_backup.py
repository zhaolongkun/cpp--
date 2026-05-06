import argparse
import json
import os

import numpy as np

from data_utils import FEATURE_NAMES, TARGET_NAMES, Bounds, build_pseudo_targets, build_sequences, load_tracker_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build sequence dataset for neural Kalman module")
    p.add_argument("--csv", required=True, help="path to tracker_log.csv")
    p.add_argument("--out", required=True, help="output npz path")
    p.add_argument("--seq_len", type=int, default=12)
    p.add_argument("--train_ratio", type=float, default=0.85)
    p.add_argument("--min_valid_ratio", type=float, default=0.10)
    p.add_argument("--trend_alpha", type=float, default=0.08)
    p.add_argument("--mad_window", type=int, default=21)
    p.add_argument("--alpha_q_min", type=float, default=0.35)
    p.add_argument("--alpha_q_max", type=float, default=3.00)
    p.add_argument("--alpha_r_min", type=float, default=0.35)
    p.add_argument("--alpha_r_max", type=float, default=4.50)
    p.add_argument("--bias_limit_px", type=float, default=4.0)
    p.add_argument("--outlier_prob_min", type=float, default=0.05)
    p.add_argument("--outlier_prob_max", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    bounds = Bounds(
        alpha_q_min=args.alpha_q_min,
        alpha_q_max=args.alpha_q_max,
        alpha_r_min=args.alpha_r_min,
        alpha_r_max=args.alpha_r_max,
        bias_limit_px=args.bias_limit_px,
        outlier_prob_min=args.outlier_prob_min,
        outlier_prob_max=args.outlier_prob_max,
    )

    series = load_tracker_csv(args.csv)
    targets = build_pseudo_targets(series, bounds, trend_alpha=args.trend_alpha, mad_window=args.mad_window)
    x, y = build_sequences(
        features=series["features"],
        targets=targets,
        valid_track=series["valid_track"],
        seq_len=args.seq_len,
        min_valid_ratio=args.min_valid_ratio,
    )

    n = x.shape[0]
    split = int(max(1, min(n - 1, round(n * args.train_ratio))))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    np.savez_compressed(
        args.out,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        feature_names=np.array(FEATURE_NAMES),
        target_names=np.array(TARGET_NAMES),
        seq_len=np.array([args.seq_len], dtype=np.int32),
    )

    meta = {
        "csv": os.path.abspath(args.csv),
        "dataset": os.path.abspath(args.out),
        "num_samples": int(n),
        "num_train": int(x_train.shape[0]),
        "num_val": int(x_val.shape[0]),
        "seq_len": int(args.seq_len),
        "num_features": int(x.shape[2]),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "bounds": bounds.to_dict(),
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[build_dataset] done")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
