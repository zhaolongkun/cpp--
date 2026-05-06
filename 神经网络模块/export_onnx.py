#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2S-LAVS-Lite ONNX 导出脚本

从 train.py 生成的 best.pt / last.pt 导出 ONNX，供 C++ (ONNX Runtime) 在线推理。
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from gru_residual import build_model_from_cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export GRU residual model to ONNX")
    p.add_argument("--ckpt", type=str, required=True, help="train.py 输出的 checkpoint 路径（best.pt/last.pt）")
    p.add_argument("--onnx_out", type=str, required=True, help="导出的 onnx 文件路径")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset 版本，默认 17")
    p.add_argument("--dynamic_batch", action="store_true", help="导出动态 batch 维度")

    # 可选：同步拷贝 scaler / feature_spec 到 onnx 目录
    p.add_argument("--scaler_json", type=str, default="", help="可选，scaler.json 路径")
    p.add_argument("--feature_spec_json", type=str, default="", help="可选，feature_spec.json 路径")
    return p.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        fallback = ckpt_path.parent / "last.pt"
        if ckpt_path.name.lower() == "best.pt" and fallback.is_file():
            print(f"[Warn] checkpoint 不存在: {ckpt_path}，自动回退到: {fallback}")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    onnx_path = Path(args.onnx_out)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "model_cfg" not in ckpt:
        raise KeyError("checkpoint 缺少 model_cfg，无法重建模型。")

    model_cfg = ckpt["model_cfg"]
    model = build_model_from_cfg(model_cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    input_dim = int(model_cfg.get("input_dim", 12))
    seq_len = int(ckpt.get("train_cfg", {}).get("seq_len", 8))
    output_dim = int(model_cfg.get("output_dim", 2))

    dummy = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "delta_cmd": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["delta_cmd"],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
    )

    # 导出 meta
    meta = {
        "checkpoint": str(ckpt_path.resolve()),
        "onnx_path": str(onnx_path.resolve()),
        "input_name": "input",
        "output_name": "delta_cmd",
        "input_shape": ["B", seq_len, input_dim] if args.dynamic_batch else [1, seq_len, input_dim],
        "output_shape": ["B", output_dim] if args.dynamic_batch else [1, output_dim],
        "feature_names": ckpt.get("feature_names", []),
        "target_names": ckpt.get("target_names", []),
        "model_cfg": model_cfg,
        "train_cfg": ckpt.get("train_cfg", {}),
        "metrics": ckpt.get("metrics", {}),
    }
    meta_path = onnx_path.with_suffix(".meta.json")
    save_json(meta, meta_path)

    # 可选复制 scaler / feature_spec，便于 C++ 一次性加载
    copied_files = []
    if args.scaler_json:
        src = Path(args.scaler_json)
        if src.is_file():
            dst = onnx_path.parent / "scaler.json"
            if src.resolve() != dst.resolve():
                shutil.copy2(str(src), str(dst))
                copied_files.append(str(dst))
        else:
            print(f"[Warn] scaler_json 不存在，跳过: {src}")
    if args.feature_spec_json:
        src = Path(args.feature_spec_json)
        if src.is_file():
            dst = onnx_path.parent / "feature_spec.json"
            if src.resolve() != dst.resolve():
                shutil.copy2(str(src), str(dst))
                copied_files.append(str(dst))
        else:
            print(f"[Warn] feature_spec_json 不存在，跳过: {src}")

    print(f"[Done] ONNX: {onnx_path}")
    print(f"[Done] Meta: {meta_path}")
    if copied_files:
        for p in copied_files:
            print(f"[Done] Copied: {p}")


if __name__ == "__main__":
    main()
