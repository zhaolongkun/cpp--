from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.runtime_inference import runtime_from_best_release


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test current best stage1 ONNX runtime on a real tracker log")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    runtime = runtime_from_best_release(ROOT)
    df = pd.read_csv(args.csv)

    outputs = []
    for _, row in df.iterrows():
        out = runtime.push(row.to_dict())
        if out.ready:
            outputs.append(
                {
                    "frame_id": int(row.get("frame_id", len(outputs))),
                    "timestamp_ms": float(row.get("timestamp_ms", 0.0)),
                    "clean_dx": out.clean_dx,
                    "clean_dy": out.clean_dy,
                    "switch_gate": out.switch_gate,
                }
            )

    if not outputs:
        raise RuntimeError("runtime never became ready; csv shorter than seq_len")

    summary = {
        "csv": args.csv,
        "output_count": len(outputs),
        "first_output": outputs[0],
        "last_output": outputs[-1],
        "mean_switch_gate": float(sum(v["switch_gate"] for v in outputs) / len(outputs)),
    }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
