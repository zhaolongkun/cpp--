from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


LOSS_DIR = Path(__file__).resolve().parent
PRECISION_DIR = LOSS_DIR.parent / "Precision"


def find_project_dir(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and any(
            child.is_dir() and child.name.endswith("-new2") and (child / "models").exists()
            for child in candidate.iterdir()
        ):
            return candidate
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_DIR = find_project_dir(LOSS_DIR)
PAPER_DIR = PROJECT_DIR / "paper" / "仪器与测量汇刊" / "new2"
ABLATION_SCRIPT = PAPER_DIR / "picture" / "消融" / "run_ablation_experiments.py"
LOSS_CSV_PATH = LOSS_DIR / "loss.csv"
PRECISION_CSV_PATH = PRECISION_DIR / "precisiion.csv"
CHECKPOINT_PATH = LOSS_DIR / "dscgnet_best.pt"
SUMMARY_JSON_PATH = LOSS_DIR / "train_summary.json"
PLOT_SCRIPT_PATH = LOSS_DIR / "plot_loss.py"
PRECISION_PLOT_SCRIPT_PATH = PRECISION_DIR / "plot_precision.py"


def load_ablation_module():
    spec = importlib.util.spec_from_file_location("dscgnet_ablation_module", ABLATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {ABLATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ablation = load_ablation_module()


@dataclass(frozen=True)
class TrainConfig:
    raw_csv: Path
    split_ref_csv: Path
    device: str
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden: int
    rmax_x: float
    rmax_y: float
    lambda_ahead: float
    lambda_inc: float
    lambda_stab: float
    lambda_dir: float
    lambda_gate: float
    tau: float
    spike_threshold: float
    num_workers: int


class LossCsvLogger:
    FIELDNAMES = [
        "epoch",
        "train_total_loss",
        "val_total_loss",
        "val_ahead_mae",
        "val_ahead_rmse",
        "learning_rate",
        "train_l_ahead",
        "train_l_inc",
        "train_l_smooth",
        "train_l_dir",
        "train_l_gate",
        "val_l_ahead",
        "val_l_inc",
        "val_l_smooth",
        "val_l_dir",
        "val_l_gate",
    ]

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.csv_path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.fp, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()
        self.flush()

    def write(self, row: Dict[str, float]) -> None:
        payload = {key: row.get(key, "") for key in self.FIELDNAMES}
        self.writer.writerow(payload)
        self.flush()

    def flush(self) -> None:
        self.fp.flush()
        os.fsync(self.fp.fileno())

    def close(self) -> None:
        self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class PrecisionCsvLogger:
    FIELDNAMES = [
        "epoch",
        "train_precision",
        "train_precision_tol_2",
        "train_precision_tol_3",
        "val_precision",
        "val_precision_tol_2",
        "val_precision_tol_3",
        "train_ahead_mae",
        "train_ahead_rmse",
        "val_ahead_mae",
        "val_ahead_rmse",
        "val_jitter",
        "val_spike_rate",
        "val_sign_flip_rate",
        "train_total_loss",
        "val_total_loss",
        "learning_rate",
        "best_so_far_epoch_by_mae",
        "best_so_far_val_ahead_mae",
        "best_so_far_val_ahead_rmse",
        "best_so_far_precision",
        "best_precision_epoch",
    ]

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.csv_path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.fp, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()
        self.flush()

    def write(self, row: Dict[str, float]) -> None:
        payload = {key: row.get(key, "") for key in self.FIELDNAMES}
        self.writer.writerow(payload)
        self.flush()

    def flush(self) -> None:
        self.fp.flush()
        os.fsync(self.fp.fileno())

    def close(self) -> None:
        self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class WeightedPartsMeter:
    def __init__(self) -> None:
        self.total_weight = 0
        self.sums = {
            "ahead": 0.0,
            "inc": 0.0,
            "stab": 0.0,
            "dir": 0.0,
            "gate": 0.0,
            "total": 0.0,
        }

    def update(self, parts: Dict[str, float], weight: int) -> None:
        self.total_weight += weight
        for key in self.sums:
            self.sums[key] += float(parts[key]) * weight

    def mean(self) -> Dict[str, float]:
        if self.total_weight <= 0:
            return {key: 0.0 for key in self.sums}
        return {key: value / self.total_weight for key, value in self.sums.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full DSCGNet setting and record loss curves.")
    parser.add_argument(
        "--raw-csv",
        default=str(PROJECT_DIR / "data" / "track-fusion-move_2026-4-17_new.csv"),
    )
    parser.add_argument(
        "--split-ref-csv",
        default=str(PROJECT_DIR / "data" / "train" / "track-fusion-move_2026-4-17_new_dscgnet_predictions.csv"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--rmax-x", type=float, default=12.0)
    parser.add_argument("--rmax-y", type=float, default=12.0)
    parser.add_argument("--lambda-ahead", type=float, default=1.0)
    parser.add_argument("--lambda-inc", type=float, default=0.5)
    parser.add_argument("--lambda-stab", type=float, default=0.1)
    parser.add_argument("--lambda-dir", type=float, default=0.1)
    parser.add_argument("--lambda-gate", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=12.0)
    parser.add_argument("--spike-threshold", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        raw_csv=Path(args.raw_csv).resolve(),
        split_ref_csv=Path(args.split_ref_csv).resolve(),
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        rmax_x=args.rmax_x,
        rmax_y=args.rmax_y,
        lambda_ahead=args.lambda_ahead,
        lambda_inc=args.lambda_inc,
        lambda_stab=args.lambda_stab,
        lambda_dir=args.lambda_dir,
        lambda_gate=args.lambda_gate,
        tau=args.tau,
        spike_threshold=args.spike_threshold,
        num_workers=args.num_workers,
    )


def evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    use_quality: bool,
    legacy_mean: np.ndarray,
    legacy_std: np.ndarray,
) -> Dict[str, object]:
    model.eval()
    meter = WeightedPartsMeter()
    rows: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    current_u_list: List[np.ndarray] = []

    loss_args = argparse.Namespace(
        lambda_ahead=cfg.lambda_ahead,
        lambda_inc=cfg.lambda_inc,
        lambda_stab=cfg.lambda_stab,
        lambda_dir=cfg.lambda_dir,
        lambda_gate=cfg.lambda_gate,
        tau=cfg.tau,
        spike_threshold=cfg.spike_threshold,
    )

    with torch.no_grad():
        for x_base, x_quality, target_u, current_u, prev_u, row_index in loader:
            x_base = x_base.to(device)
            x_quality = x_quality.to(device)
            target_u = target_u.to(device)
            current_u = current_u.to(device)
            prev_u = prev_u.to(device)

            output = ablation.forward_model(model, x_base, x_quality, use_quality=use_quality)
            _, parts = ablation.compute_loss(output, target_u, current_u, prev_u, loss_args)
            meter.update(parts, int(x_base.shape[0]))

            rows.append(row_index.numpy())
            preds.append(output["u_pred"].cpu().numpy())
            targets.append(target_u.cpu().numpy())
            current_u_list.append(current_u.cpu().numpy())

    row_index = np.concatenate(rows, axis=0)
    pred_u = np.concatenate(preds, axis=0)
    target_u = np.concatenate(targets, axis=0)
    current_u = np.concatenate(current_u_list, axis=0)

    order = np.argsort(row_index)
    row_index = row_index[order]
    pred_u = pred_u[order]
    target_u = target_u[order]
    current_u = current_u[order]

    pred_u = pred_u * legacy_std + legacy_mean
    target_u = target_u * legacy_std + legacy_mean
    current_u = current_u * legacy_std + legacy_mean

    pred_inc = pred_u - current_u
    true_inc = target_u - current_u
    mean_parts = meter.mean()

    metrics = {
        "loss": float(mean_parts["total"]),
        "ahead_mae": float(np.mean(np.abs(pred_u - target_u))),
        "ahead_rmse": float(np.sqrt(np.mean((pred_u - target_u) ** 2))),
        "jitter": float(np.mean(np.abs(np.diff(pred_u, axis=0)))) if len(pred_u) > 1 else 0.0,
        "spike_rate": float(np.mean(np.abs(np.diff(pred_u, axis=0)) > cfg.spike_threshold))
        if len(pred_u) > 1
        else 0.0,
        "sign_flip_rate": float(np.mean(np.sign(pred_inc) != np.sign(true_inc))),
        "count": int(len(pred_u)),
        "row_start": int(row_index[0]),
        "row_end": int(row_index[-1]),
    }
    abs_err = np.abs(pred_u - target_u)
    metrics["precision"] = float(np.mean((abs_err[:, 0] <= 1.0) & (abs_err[:, 1] <= 1.0)))
    metrics["precision_tol_2"] = float(np.mean((abs_err[:, 0] <= 2.0) & (abs_err[:, 1] <= 2.0)))
    metrics["precision_tol_3"] = float(np.mean((abs_err[:, 0] <= 3.0) & (abs_err[:, 1] <= 3.0)))
    return {
        "metrics": metrics,
        "loss_parts": mean_parts,
        "pred_u": pred_u,
        "target_u": target_u,
        "current_u": current_u,
        "row_index": row_index,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model_state: Dict[str, torch.Tensor],
    epoch: int,
    best_val_mae: float,
    best_val_rmse: float,
    config: TrainConfig,
    stats: Dict[str, List[float]],
) -> None:
    payload = {
        "epoch": epoch,
        "best_val_ahead_mae": best_val_mae,
        "best_val_ahead_rmse": best_val_rmse,
        "train_config": {
            **asdict(config),
            "raw_csv": str(config.raw_csv),
            "split_ref_csv": str(config.split_ref_csv),
        },
        "normalization_stats": stats,
        "model_state_dict": model_state,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def run_training(config: TrainConfig) -> Dict[str, object]:
    ablation.set_seed(config.seed)
    device = ablation.pick_device(config.device)

    split_ranges = ablation.extract_split_ranges(config.split_ref_csv)
    df = ablation.load_raw_dataframe(config.raw_csv)
    arrays = ablation.build_feature_arrays(df)

    exp_cfg = ablation.ExperimentConfig(
        name="full",
        display_name="T=16 + quality + attention",
        history_len=16,
        use_quality=True,
        use_attention=True,
        output_head="state",
    )
    samples = ablation.build_samples(arrays=arrays, history_len=exp_cfg.history_len, split_ranges=split_ranges)
    normalized = ablation.normalize_splits(samples)

    train_loader = ablation.make_loader(normalized["train"], config.batch_size, True, config.num_workers)
    val_loader = ablation.make_loader(normalized["val"], config.batch_size, False, config.num_workers)
    legacy_mean = np.asarray(normalized["stats"]["legacy_mean"], dtype=np.float32)
    legacy_std = np.asarray(normalized["stats"]["legacy_std"], dtype=np.float32)

    model = ablation.DSCGNet(
        hidden=config.hidden,
        use_attention=exp_cfg.use_attention,
        use_quality=exp_cfg.use_quality,
        rmax_x=config.rmax_x,
        rmax_y=config.rmax_y,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    loss_args = argparse.Namespace(
        lambda_ahead=config.lambda_ahead,
        lambda_inc=config.lambda_inc,
        lambda_stab=config.lambda_stab,
        lambda_dir=config.lambda_dir,
        lambda_gate=config.lambda_gate,
        tau=config.tau,
        spike_threshold=config.spike_threshold,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_mae = math.inf
    best_val_rmse = math.inf
    best_precision = -math.inf
    best_precision_epoch = 0
    history_rows: List[Dict[str, float]] = []

    with LossCsvLogger(LOSS_CSV_PATH) as loss_logger, PrecisionCsvLogger(PRECISION_CSV_PATH) as precision_logger:
        for epoch in range(1, config.epochs + 1):
            model.train()
            train_meter = WeightedPartsMeter()

            for x_base, x_quality, target_u, current_u, prev_u, _ in train_loader:
                x_base = x_base.to(device)
                x_quality = x_quality.to(device)
                target_u = target_u.to(device)
                current_u = current_u.to(device)
                prev_u = prev_u.to(device)

                optimizer.zero_grad(set_to_none=True)
                output = ablation.forward_model(model, x_base, x_quality, use_quality=exp_cfg.use_quality)
                loss, parts = ablation.compute_loss(output, target_u, current_u, prev_u, loss_args)
                loss.backward()
                optimizer.step()
                train_meter.update(parts, int(x_base.shape[0]))

            train_parts = train_meter.mean()
            train_eval = evaluate_loader(
                model=model,
                loader=train_loader,
                device=device,
                cfg=config,
                use_quality=exp_cfg.use_quality,
                legacy_mean=legacy_mean,
                legacy_std=legacy_std,
            )
            train_metrics = train_eval["metrics"]
            val_eval = evaluate_loader(
                model=model,
                loader=val_loader,
                device=device,
                cfg=config,
                use_quality=exp_cfg.use_quality,
                legacy_mean=legacy_mean,
                legacy_std=legacy_std,
            )
            val_parts = val_eval["loss_parts"]
            val_metrics = val_eval["metrics"]
            current_lr = float(optimizer.param_groups[0]["lr"])

            loss_row = {
                "epoch": epoch,
                "train_total_loss": train_parts["total"],
                "val_total_loss": val_parts["total"],
                "val_ahead_mae": val_metrics["ahead_mae"],
                "val_ahead_rmse": val_metrics["ahead_rmse"],
                "learning_rate": current_lr,
                "train_l_ahead": train_parts["ahead"],
                "train_l_inc": train_parts["inc"],
                "train_l_smooth": train_parts["stab"],
                "train_l_dir": train_parts["dir"],
                "train_l_gate": train_parts["gate"],
                "val_l_ahead": val_parts["ahead"],
                "val_l_inc": val_parts["inc"],
                "val_l_smooth": val_parts["stab"],
                "val_l_dir": val_parts["dir"],
                "val_l_gate": val_parts["gate"],
            }
            loss_logger.write(loss_row)
            history_rows.append(loss_row)

            if val_metrics["ahead_mae"] < best_val_mae - 1e-12:
                best_val_mae = float(val_metrics["ahead_mae"])
                best_val_rmse = float(val_metrics["ahead_rmse"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                save_checkpoint(
                    checkpoint_path=CHECKPOINT_PATH,
                    model_state=best_state,
                    epoch=best_epoch,
                    best_val_mae=best_val_mae,
                    best_val_rmse=best_val_rmse,
                    config=config,
                    stats=normalized["stats"],
                )

            if val_metrics["precision"] > best_precision + 1e-12:
                best_precision = float(val_metrics["precision"])
                best_precision_epoch = epoch

            precision_row = {
                "epoch": epoch,
                "train_precision": train_metrics["precision"],
                "train_precision_tol_2": train_metrics["precision_tol_2"],
                "train_precision_tol_3": train_metrics["precision_tol_3"],
                "val_precision": val_metrics["precision"],
                "val_precision_tol_2": val_metrics["precision_tol_2"],
                "val_precision_tol_3": val_metrics["precision_tol_3"],
                "train_ahead_mae": train_metrics["ahead_mae"],
                "train_ahead_rmse": train_metrics["ahead_rmse"],
                "val_ahead_mae": val_metrics["ahead_mae"],
                "val_ahead_rmse": val_metrics["ahead_rmse"],
                "val_jitter": val_metrics["jitter"],
                "val_spike_rate": val_metrics["spike_rate"],
                "val_sign_flip_rate": val_metrics["sign_flip_rate"],
                "train_total_loss": train_parts["total"],
                "val_total_loss": val_parts["total"],
                "learning_rate": current_lr,
                "best_so_far_epoch_by_mae": best_epoch,
                "best_so_far_val_ahead_mae": best_val_mae,
                "best_so_far_val_ahead_rmse": best_val_rmse,
                "best_so_far_precision": best_precision,
                "best_precision_epoch": best_precision_epoch,
            }
            precision_logger.write(precision_row)

            print(
                f"[epoch {epoch:03d}/{config.epochs}] "
                f"train_precision={train_metrics['precision']:.6f} "
                f"val_precision={val_metrics['precision']:.6f} "
                f"train_total={train_parts['total']:.6f} "
                f"val_total={val_parts['total']:.6f} "
                f"val_mae={val_metrics['ahead_mae']:.6f} "
                f"val_rmse={val_metrics['ahead_rmse']:.6f} "
                f"best_epoch={best_epoch} "
                f"best_mae={best_val_mae:.6f} "
                f"lr={current_lr:.6g}",
                flush=True,
            )

    model.load_state_dict(best_state)
    summary = {
        "loss_csv": str(LOSS_CSV_PATH),
        "precision_csv": str(PRECISION_CSV_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "plot_script": str(PLOT_SCRIPT_PATH),
        "precision_plot_script": str(PRECISION_PLOT_SCRIPT_PATH),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_ahead_mae": best_val_mae,
        "best_val_ahead_rmse": best_val_rmse,
        "best_precision": best_precision,
        "best_precision_epoch": best_precision_epoch,
        "history_rows": len(history_rows),
        "train_config": {
            **asdict(config),
            "raw_csv": str(config.raw_csv),
            "split_ref_csv": str(config.split_ref_csv),
        },
        "experiment_config": {
            "name": exp_cfg.name,
            "display_name": exp_cfg.display_name,
            "history_len": exp_cfg.history_len,
            "use_quality": exp_cfg.use_quality,
            "use_attention": exp_cfg.use_attention,
            "output_head": exp_cfg.output_head,
            "params": ablation.count_params(model),
        },
        "split_ranges": {name: [int(v[0]), int(v[1])] for name, v in split_ranges.items()},
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_plot_scripts() -> None:
    subprocess.run([sys.executable, str(PLOT_SCRIPT_PATH)], check=True, cwd=str(LOSS_DIR))
    subprocess.run([sys.executable, str(PRECISION_PLOT_SCRIPT_PATH)], check=True, cwd=str(PRECISION_DIR))


def main() -> None:
    args = parse_args()
    config = build_train_config(args)
    summary = run_training(config)
    run_plot_scripts()
    print(f"[saved] {LOSS_CSV_PATH}")
    print(f"[saved] {PRECISION_CSV_PATH}")
    print(f"[saved] {CHECKPOINT_PATH}")
    print(f"[saved] {SUMMARY_JSON_PATH}")
    print(f"[best] epoch={summary['best_epoch']} val_ahead_mae={summary['best_val_ahead_mae']:.6f} "
          f"val_ahead_rmse={summary['best_val_ahead_rmse']:.6f}")


if __name__ == "__main__":
    main()
