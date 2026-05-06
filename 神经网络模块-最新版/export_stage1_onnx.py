from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.models.factory import build_model


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        baseline_seq: torch.Tensor,
        stable_baseline_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            x,
            baseline_seq=baseline_seq,
            stable_baseline_seq=stable_baseline_seq,
        )
        return out["clean"], out["switch_gate"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export stage-1 clean model to ONNX")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_onnx", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--dataset_meta", default="")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--input_dim", type=int, default=20)
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    dataset_meta = {}
    if args.dataset_meta:
        dataset_meta = json.loads(Path(args.dataset_meta).read_text(encoding="utf-8"))

    seq_len = int(dataset_meta.get("seq_len", args.seq_len))
    input_dim = int(cfg.get("model", {}).get("input_dim", args.input_dim))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    _, _, model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = ExportWrapper(model)
    dummy_x = torch.zeros(1, seq_len, input_dim, dtype=torch.float32)
    dummy_baseline = torch.zeros(1, seq_len, 2, dtype=torch.float32)
    dummy_stable_baseline = torch.zeros(1, seq_len, 2, dtype=torch.float32)

    out_path = Path(args.output_onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_baseline, dummy_stable_baseline),
        out_path,
        input_names=["x", "baseline_seq", "stable_baseline_seq"],
        output_names=["clean", "switch_gate"],
        opset_version=17,
    )

    export_info = {
        "checkpoint": args.checkpoint,
        "output_onnx": str(out_path),
        "config": args.config,
        "dataset_meta": args.dataset_meta,
        "seq_len": seq_len,
        "input_dim": input_dim,
        "model_name": str(ckpt["config"]["model"].get("name", "")),
        "input_names": ["x", "baseline_seq", "stable_baseline_seq"],
        "output_names": ["clean", "switch_gate"],
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(export_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(export_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
