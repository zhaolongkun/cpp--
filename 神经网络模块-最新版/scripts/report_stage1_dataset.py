from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.stage1_common import build_segmented_pseudo_targets, ensure_segment_columns, split_type_distribution
from la_cspc_ornet.features import check_feature_index_alignment

REQUIRED_STAGE1_TYPES = ("stable", "jitter", "recover", "turn")


def main() -> None:
    ap = argparse.ArgumentParser(description="Report stage1 dataset quality and split coverage")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset_npz", default="")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    if not check_feature_index_alignment():
        raise RuntimeError("feature index alignment check failed; aborting dataset report")

    df = pd.read_csv(args.csv)
    df = ensure_segment_columns(df)
    df = build_segmented_pseudo_targets(df)

    zoom_transition = np.abs(df["zoom_delta"].to_numpy(dtype=np.float64)) > 1e-3
    high_jitter = df["event_score"].to_numpy(dtype=np.float64) >= 0.5
    lost_or_coasting = (df["lost_flag"].to_numpy(dtype=np.float64) > 0.5) | (df["coast_count"].to_numpy(dtype=np.float64) > 0.0)
    segment_info = df[["segment_tag", "segment_type"]].drop_duplicates().sort_values(["segment_type", "segment_tag"])

    report: Dict[str, object] = {
        "input_csv": args.csv,
        "required_stage1_types": list(REQUIRED_STAGE1_TYPES),
        "total_frames": int(len(df)),
        "valid_target_frames": int(((df["lost_flag"].to_numpy(dtype=np.float64) < 0.5) & (df["det_conf"].to_numpy(dtype=np.float64) > 0.0)).sum()),
        "valid_target_ratio": float(((df["lost_flag"].to_numpy(dtype=np.float64) < 0.5) & (df["det_conf"].to_numpy(dtype=np.float64) > 0.0)).mean()) if len(df) else 0.0,
        "lost_or_coasting_frames": int(lost_or_coasting.sum()),
        "lost_or_coasting_ratio": float(lost_or_coasting.mean()) if len(df) else 0.0,
        "zoom_transition_frames": int(zoom_transition.sum()),
        "zoom_transition_ratio": float(zoom_transition.mean()) if len(df) else 0.0,
        "high_jitter_frames": int(high_jitter.sum()),
        "high_jitter_ratio": float(high_jitter.mean()) if len(df) else 0.0,
        "switch_event_segments": int((df.groupby("segment_tag")["switch_score"].max() >= 0.5).sum()),
        "turn_event_segments": int((df.groupby("segment_tag")["turn_score"].max() >= 0.5).sum()),
        "segment_count": int(len(segment_info)),
        "segment_type_distribution": segment_info["segment_type"].value_counts().to_dict(),
    }

    if args.dataset_npz:
        data = np.load(args.dataset_npz, allow_pickle=False)
        split_tags = {
            "train": [str(x) for x in data["segment_tag_train"]],
            "val": [str(x) for x in data["segment_tag_val"]],
            "test": [str(x) for x in data["segment_tag_test"]],
        }
        unique_split_tags = {k: sorted(set(v)) for k, v in split_tags.items()}
        report["dataset_splits"] = {
            "train_windows": int(data["X_train"].shape[0]),
            "val_windows": int(data["X_val"].shape[0]),
            "test_windows": int(data["X_test"].shape[0]),
            "train_segments": len(unique_split_tags["train"]),
            "val_segments": len(unique_split_tags["val"]),
            "test_segments": len(unique_split_tags["test"]),
            "train_type_distribution": split_type_distribution(segment_info, unique_split_tags["train"]),
            "val_type_distribution": split_type_distribution(segment_info, unique_split_tags["val"]),
            "test_type_distribution": split_type_distribution(segment_info, unique_split_tags["test"]),
        }

    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
