#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


FIXED_MATRIX = [
    {
        "experiment_id": "baseline_only",
        "role": "classic_baseline",
        "supervision_variant": "none",
    },
    {
        "experiment_id": "gru_future_error_aware",
        "role": "main_method",
        "supervision_variant": "future_error_aware",
    },
    {
        "experiment_id": "gru_future_smoothed_base",
        "role": "supervision_ablation",
        "supervision_variant": "future_smoothed_base",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the fixed formal training matrix for R2S-LAVS-Lite")
    p.add_argument("--batch_summary_csv", required=True)
    p.add_argument("--registry_csv", required=True)
    p.add_argument("--output_summary_csv", required=True)
    p.add_argument("--output_eval_csv", required=True)
    p.add_argument("--output_ablation_csv", required=True)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def resolve_path(project_root: Path, path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "鍙嶆棤" and parts[1] == "cpp鏅鸿兘鎺у埗":
        repo_root = project_root.parent.parent
        return (repo_root / p).resolve()
    return (project_root / p).resolve()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print({"run": cmd})
    subprocess.run(cmd, cwd=str(cwd), check=True)


def concat_variant_csv(project_root: Path, batch_df: pd.DataFrame, variant: str, output_csv: Path) -> Dict[str, int]:
    seg = batch_df[
        (batch_df["variant"].astype(str) == variant)
        & (batch_df["status"].astype(str) == "ok")
    ].copy()
    if seg.empty:
        raise RuntimeError(f"no augmented rows found for variant={variant}")

    frames: List[pd.DataFrame] = []
    num_source_files = 0
    total_rows = 0
    total_valid = 0
    for _, row in seg.iterrows():
        aug_path = resolve_path(project_root, row["augmented_csv"])
        df = pd.read_csv(aug_path)
        frames.append(df)
        num_source_files += 1
        total_rows += int(len(df))
        if "pseudo_expert_valid" in df.columns:
            total_valid += int(pd.to_numeric(df["pseudo_expert_valid"], errors="coerce").fillna(0).sum())

    merged = pd.concat(frames, axis=0, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8")
    return {
        "source_files": num_source_files,
        "merged_rows": int(len(merged)),
        "pseudo_valid_rows": total_valid,
    }


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    module_dir = script_path.parent
    project_root = module_dir.parent

    batch_df = pd.read_csv(args.batch_summary_csv)
    registry_path = Path(args.registry_csv)
    registry_df = pd.read_csv(registry_path)

    summary_rows: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []

    build_dataset_py = module_dir / "build_dataset.py"
    train_py = module_dir / "train.py"
    export_onnx_py = module_dir / "export_onnx.py"

    for slot in FIXED_MATRIX:
        experiment_id = slot["experiment_id"]
        role = slot["role"]
        supervision_variant = slot["supervision_variant"]

        reg_seg = registry_df[registry_df["experiment_id"].astype(str) == experiment_id]
        if reg_seg.empty:
            raise RuntimeError(f"missing registry row for {experiment_id}")
        artifact_dir = resolve_path(project_root, reg_seg.iloc[0]["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if supervision_variant == "none":
            manifest = {
                "experiment_id": experiment_id,
                "role": role,
                "status": "reference_only",
                "note": "baseline controller requires no neural training",
            }
            save_json(artifact_dir / "baseline_manifest.json", manifest)
            summary_rows.append(
                {
                    "experiment_id": experiment_id,
                    "role": role,
                    "supervision_variant": supervision_variant,
                    "status": "reference_only",
                    "artifact_dir": str(artifact_dir),
                    "dataset_npz": "",
                    "onnx_path": "",
                    "train_size": 0,
                    "val_size": 0,
                    "test_size": 0,
                    "best_selection_metric": "",
                    "best_val_total": "",
                    "test_total": "",
                    "test_mae": "",
                    "test_rmse": "",
                }
            )
            continue

        merged_csv = artifact_dir / f"{supervision_variant}_merged.csv"
        merge_stats = concat_variant_csv(project_root, batch_df, supervision_variant, merged_csv)

        dataset_name = "dataset_t8.npz"
        dataset_path = artifact_dir / dataset_name
        scaler_path = artifact_dir / "scaler.json"
        feature_spec_path = artifact_dir / "feature_spec.json"

        run_cmd(
            [
                sys.executable,
                str(build_dataset_py),
                "--input_csv",
                str(merged_csv),
                "--output_dir",
                str(artifact_dir),
                "--seq_len",
                str(args.seq_len),
                "--dataset_name",
                dataset_name,
            ],
            cwd=project_root,
        )

        run_cmd(
            [
                sys.executable,
                str(train_py),
                "--dataset_npz",
                str(dataset_path),
                "--out_dir",
                str(artifact_dir),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
            ],
            cwd=project_root,
        )

        onnx_path = artifact_dir / "model.onnx"
        run_cmd(
            [
                sys.executable,
                str(export_onnx_py),
                "--ckpt",
                str(artifact_dir / "best.pt"),
                "--onnx_out",
                str(onnx_path),
                "--dynamic_batch",
                "--scaler_json",
                str(scaler_path),
                "--feature_spec_json",
                str(feature_spec_path),
            ],
            cwd=project_root,
        )

        train_summary_path = artifact_dir / "train_summary.json"
        train_history_path = artifact_dir / "train_history.json"
        train_summary = load_json(train_summary_path)
        shutil.copy2(str(train_summary_path), str(artifact_dir / "train_metrics.json"))

        best_test = train_summary.get("best_test_metrics") or {}
        summary_row = {
            "experiment_id": experiment_id,
            "role": role,
            "supervision_variant": supervision_variant,
            "status": "completed",
            "artifact_dir": str(artifact_dir),
            "dataset_npz": str(dataset_path),
            "merged_csv": str(merged_csv),
            "source_files": merge_stats["source_files"],
            "merged_rows": merge_stats["merged_rows"],
            "pseudo_valid_rows": merge_stats["pseudo_valid_rows"],
            "onnx_path": str(onnx_path),
            "train_size": int(train_summary.get("train_size", 0)),
            "val_size": int(train_summary.get("val_size", 0)),
            "test_size": int(train_summary.get("test_size", 0)),
            "best_selection_metric": train_summary.get("best_selection_metric", ""),
            "best_val_total": train_summary.get("best_val_total", ""),
            "test_total": best_test.get("total", ""),
            "test_mae": best_test.get("mae", ""),
            "test_rmse": best_test.get("rmse", ""),
            "train_summary_json": str(train_summary_path),
            "train_history_json": str(train_history_path),
        }
        summary_rows.append(summary_row)
        eval_rows.append(
            {
                "experiment_id": experiment_id,
                "role": role,
                "supervision_variant": supervision_variant,
                "split": "test",
                "metric_total": best_test.get("total", ""),
                "metric_mae": best_test.get("mae", ""),
                "metric_rmse": best_test.get("rmse", ""),
            }
        )
        ablation_rows.append(
            {
                "experiment_id": experiment_id,
                "role": role,
                "supervision_variant": supervision_variant,
                "test_mae": best_test.get("mae", ""),
                "test_rmse": best_test.get("rmse", ""),
                "test_total": best_test.get("total", ""),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    eval_df = pd.DataFrame(eval_rows)
    ablation_df = pd.DataFrame(ablation_rows)

    Path(args.output_summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_summary_csv, index=False, encoding="utf-8")
    eval_df.to_csv(args.output_eval_csv, index=False, encoding="utf-8")
    ablation_df.to_csv(args.output_ablation_csv, index=False, encoding="utf-8")

    registry_df = registry_df.copy()
    for _, row in summary_df.iterrows():
        mask = registry_df["experiment_id"].astype(str) == str(row["experiment_id"])
        if not mask.any():
            continue
        if str(row["status"]) == "completed":
            registry_df.loc[mask, "status"] = "completed"
            registry_df.loc[mask, "ready_to_train"] = 1
            registry_df.loc[mask, "gate_status"] = "completed"
        elif str(row["status"]) == "reference_only":
            registry_df.loc[mask, "status"] = "completed"
            registry_df.loc[mask, "ready_to_train"] = 1
            registry_df.loc[mask, "gate_status"] = "no_training_required"
    registry_df.to_csv(registry_path, index=False, encoding="utf-8")

    print(
        {
            "summary_csv": str(args.output_summary_csv),
            "eval_csv": str(args.output_eval_csv),
            "ablation_csv": str(args.output_ablation_csv),
            "completed_experiments": int((summary_df["status"].astype(str) == "completed").sum()),
        }
    )


if __name__ == "__main__":
    main()
