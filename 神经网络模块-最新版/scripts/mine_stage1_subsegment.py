from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.stage1_common import parse_segment_tag_from_path, parse_segment_type


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine a labeled subsegment from an existing stage1 raw log")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--start_row", type=int, required=True)
    ap.add_argument("--end_row", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"missing input csv: {input_csv}")
    if args.end_row <= args.start_row:
        raise ValueError("end_row must be greater than start_row")

    df = pd.read_csv(input_csv)
    sub = df.iloc[args.start_row : args.end_row].copy().reset_index(drop=True)
    if len(sub) == 0:
        raise RuntimeError("empty subsegment after slicing")

    source_tag = parse_segment_tag_from_path(input_csv)
    sub["segment_tag"] = args.tag
    sub["segment_type"] = parse_segment_type(args.tag)
    sub["source_file"] = f"tracker_log_{args.tag}.csv"
    sub["source_origin_file"] = str(sub["source_origin_file"].iloc[0]) if "source_origin_file" in sub.columns else input_csv.name
    sub["source_origin_tag"] = str(sub["source_origin_tag"].iloc[0]) if "source_origin_tag" in sub.columns else source_tag
    sub["source_group"] = str(sub["source_group"].iloc[0]) if "source_group" in sub.columns else input_csv.name
    sub["mined_from_file"] = input_csv.name
    sub["mined_from_start_row"] = int(args.start_row)
    sub["mined_from_end_row"] = int(args.end_row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"tracker_log_{args.tag}.csv"
    sub.to_csv(out_path, index=False)
    print(str(out_path))
    print(f"rows={len(sub)}")


if __name__ == "__main__":
    main()
