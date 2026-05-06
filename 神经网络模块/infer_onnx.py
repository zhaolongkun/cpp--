import argparse
import csv
import json
import os
from typing import List

import numpy as np
import onnxruntime as ort

from data_utils import load_tracker_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run ONNX inference for NCE module")
    p.add_argument("--onnx", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--csv", required=True, help="tracker_log.csv")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--batch_size", type=int, default=512)
    return p.parse_args()


def build_windows(features: np.ndarray, seq_len: int) -> np.ndarray:
    n, f = features.shape
    if n < seq_len:
        raise RuntimeError(f"not enough rows: n={n}, seq_len={seq_len}")
    xs: List[np.ndarray] = []
    for i in range(seq_len - 1, n):
        xs.append(features[i - seq_len + 1 : i + 1, :])
    return np.stack(xs, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    seq_len = int(meta["input_shape"][1])
    mean = np.array(meta["norm_mean"], dtype=np.float32)
    std = np.array(meta["norm_std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    series = load_tracker_csv(args.csv)
    feat = series["features"].astype(np.float32)
    x = build_windows(feat, seq_len)
    x = (x - mean[None, None, :]) / std[None, None, :]

    try:
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    except TypeError:
        # Compatibility for old onnxruntime versions.
        try:
            sess = ort.InferenceSession(args.onnx)
        except Exception:
            # Some old runtimes may fail on non-ASCII paths; load from bytes.
            with open(args.onnx, "rb") as f:
                model_bytes = f.read()
            sess = ort.InferenceSession(model_bytes)
    inp = meta.get("input_name", "features")
    out = meta.get("output_name", "pnr_params")

    preds = []
    for i in range(0, x.shape[0], args.batch_size):
        xb = x[i : i + args.batch_size]
        yb = sess.run([out], {inp: xb})[0]
        preds.append(yb)
    pred = np.concatenate(preds, axis=0).astype(np.float32)

    # Align with original timeline: first (seq_len-1) rows do not have full history.
    pad = np.zeros((seq_len - 1, pred.shape[1]), dtype=np.float32)
    pred_full = np.concatenate([pad, pred], axis=0)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_ns", "bias_x", "bias_y", "alpha_q", "alpha_r", "outlier_prob"])
        time_ns = series["time_ns"]
        n = min(len(time_ns), pred_full.shape[0])
        for i in range(n):
            w.writerow(
                [
                    int(time_ns[i]),
                    float(pred_full[i, 0]),
                    float(pred_full[i, 1]),
                    float(pred_full[i, 2]),
                    float(pred_full[i, 3]),
                    float(pred_full[i, 4]),
                ]
            )

    print(f"[infer_onnx] rows={pred_full.shape[0]} out={args.out_csv}")


if __name__ == "__main__":
    main()
