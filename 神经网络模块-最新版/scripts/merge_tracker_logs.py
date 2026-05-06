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
    ap = argparse.ArgumentParser(description="Merge multiple tracker full logs into one CSV")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    parts = []
    for path in sorted(input_dir.glob('*.csv')):
        df = pd.read_csv(path)
        seg_tag = parse_segment_tag_from_path(path)
        df["segment_tag"] = seg_tag
        df["segment_type"] = parse_segment_type(seg_tag)
        df["source_file"] = path.name
        if "source_origin_file" not in df.columns:
            df["source_origin_file"] = path.name
        if "source_origin_tag" not in df.columns:
            df["source_origin_tag"] = seg_tag
        if "source_group" not in df.columns:
            df["source_group"] = df["source_origin_file"].astype(str)
        parts.append(df)
    if not parts:
        raise RuntimeError(f"no csv found under {input_dir}")
    merged = pd.concat(parts, axis=0, ignore_index=True)
    merged.to_csv(args.output_csv, index=False)
    print(args.output_csv)
    print(f"rows={len(merged)} files={len(parts)}")


if __name__ == "__main__":
    main()
