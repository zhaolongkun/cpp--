from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.datasets import load_npz_bundle, make_datasets
from la_cspc_ornet.features import check_feature_index_alignment
from la_cspc_ornet.models.factory import build_model
from la_cspc_ornet.stage1_common import build_window_segment_masks, compute_metrics


def null_gate_stats() -> dict:
    return {
        "switch_gate_mean": None,
        "switch_gate_std": None,
        "switch_gate_p90": None,
        "switch_gate_active_ratio": None,
    }


def compute_switch_gate_stats(gates: np.ndarray | None, mask: np.ndarray, active_threshold: float) -> dict:
    if gates is None:
        return null_gate_stats()
    if int(mask.sum()) == 0:
        return null_gate_stats()
    seg_gates = gates[mask].astype(np.float64)
    return {
        "switch_gate_mean": float(np.mean(seg_gates)),
        "switch_gate_std": float(np.std(seg_gates)),
        "switch_gate_p90": float(np.percentile(seg_gates, 90)),
        "switch_gate_active_ratio": float(np.mean(seg_gates > active_threshold)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate stage1 clean-only model with segmented metrics")
    ap.add_argument("--config", default="")
    ap.add_argument("--dataset_npz", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()
    if not check_feature_index_alignment():
        raise RuntimeError("feature index alignment check failed; aborting stage1 eval")

    dataset_npz = args.dataset_npz
    checkpoint = args.checkpoint
    output_json = args.output_json
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        dataset_npz = dataset_npz or cfg["paths"]["dataset_npz"]
        out_dir = Path(cfg["paths"]["out_dir"])
        checkpoint = checkpoint or str(out_dir / "best_stage1_clean.pt")
        output_json = output_json or str(out_dir / "eval_summary.json")
    if not dataset_npz or not checkpoint or not output_json:
        raise ValueError("either --config or all of --dataset_npz/--checkpoint/--output_json must be provided")

    bundle = load_npz_bundle(dataset_npz)
    _, _, test_ds = make_datasets(bundle)
    ckpt = torch.load(checkpoint, map_location="cpu")
    _, _, model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    model_name = str(ckpt["config"]["model"].get("name", ""))
    gate_active_threshold = float(cfg.get("eval", {}).get("gate_active_threshold", 0.5)) if args.config else 0.5

    x = test_ds.X
    y = test_ds.y_clean
    baseline_seq = test_ds.baseline_seq
    stable_baseline_seq = test_ds.stable_baseline_seq
    with torch.no_grad():
        outputs = model(x, baseline_seq=baseline_seq, stable_baseline_seq=stable_baseline_seq)
        pred = outputs["clean"]

    pred_np = pred.numpy()
    y_np = y.numpy()
    x_np = x.numpy()
    switch_gates = outputs["switch_gate"].numpy().reshape(-1) if model_name == "dual_state" and "switch_gate" in outputs else None
    masks = build_window_segment_masks(x_np)
    metrics = {}
    for seg_name, seg_mask in masks.items():
        seg_metrics = compute_metrics(pred_np, y_np, x_np, seg_mask)
        seg_metrics.update(compute_switch_gate_stats(switch_gates, seg_mask, gate_active_threshold))
        metrics[seg_name] = seg_metrics
    Path(output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
