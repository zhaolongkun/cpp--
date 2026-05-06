import argparse
import json
import os

import torch

from model import ModelBounds, NCEGRU


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export NCEGRU to ONNX")
    p.add_argument("--ckpt", required=True, help="best.pt from train_nce.py")
    p.add_argument("--out", required=True, help="output onnx file")
    p.add_argument("--opset", type=int, default=9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    bounds = ModelBounds(**ckpt["bounds"])
    model = NCEGRU(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
        bounds=bounds,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    seq_len = int(ckpt["seq_len"])
    input_dim = int(ckpt["input_dim"])
    dummy = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        export_params=True,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["pnr_params"],
        dynamic_axes={"features": {0: "batch"}, "pnr_params": {0: "batch"}},
        opset_version=args.opset,
    )

    meta = {
        "onnx_path": os.path.abspath(args.out),
        "input_name": "features",
        "output_name": "pnr_params",
        "input_shape": ["B", seq_len, input_dim],
        "output_shape": ["B", 5],
        "output_order": ["bias_x", "bias_y", "alpha_q", "alpha_r", "outlier_prob"],
        "norm_mean": ckpt["norm_mean"],
        "norm_std": ckpt["norm_std"],
        "feature_names": ckpt["feature_names"],
        "target_names": ckpt["target_names"],
        "bounds": ckpt["bounds"],
        "checkpoint_metrics": ckpt.get("metrics", {}),
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[export_onnx] onnx: {args.out}")
    print(f"[export_onnx] meta: {meta_path}")


if __name__ == "__main__":
    main()
