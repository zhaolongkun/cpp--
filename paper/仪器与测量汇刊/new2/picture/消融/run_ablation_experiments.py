from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent


def find_project_dir(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            for child in candidate.iterdir():
                if child.is_dir() and child.name.endswith("-new2") and (child / "models").exists():
                    return candidate
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_DIR = find_project_dir(THIS_DIR)
MODEL_DIR = next(
    child for child in PROJECT_DIR.iterdir() if child.is_dir() and child.name.endswith("-new2") and (child / "models").exists()
)
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from models.dscgnet import DSCGNet, MotionStreamEncoder, QualityBranchEncoder  # noqa: E402


LEGACY_X_COL = "window8_10_no.3_first_filter_x"
LEGACY_Y_COL = "window8_10_no.3_first_filter_y"
CONF_COL = "det_conf"
WIDTH_COL = "det_w"
HEIGHT_COL = "det_h"
MATCHED_COL = "matched_in_frame"
MISS_COL = "miss_count"
TIMESTAMP_COL = "timestamp_ms"
SPLIT_COL = "prediction_split"
INDEX_COL = "current_row_index"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    display_name: str
    history_len: int
    use_quality: bool
    use_attention: bool
    output_head: str = "state"


class DirectRegressionDSCGNet(nn.Module):
    """Keep the DSCGNet backbone unchanged and regress next-step control directly."""

    def __init__(self, hidden=32, use_attention=True, use_quality=False):
        super().__init__()
        self.use_quality = use_quality
        self.enc_x = MotionStreamEncoder(hidden, use_attention)
        self.enc_y = MotionStreamEncoder(hidden, use_attention)

        feat_dim = hidden * 2
        if use_quality:
            self.enc_q = QualityBranchEncoder(16)
            feat_dim += 16

        self.pred_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 2),
        )

    def forward(self, x, quality=None):
        x_stream = x[:, :, [0, 2, 4]]
        y_stream = x[:, :, [1, 3, 5]]

        hx = self.enc_x(x_stream)
        hy = self.enc_y(y_stream)
        h = torch.cat([hx, hy], dim=-1)

        if self.use_quality and quality is not None:
            hq = self.enc_q(quality)
            h = torch.cat([h, hq], dim=-1)

        u_pred = self.pred_head(h)
        e_f_t = x[:, -1, :2]
        delta_pred = u_pred - e_f_t
        gate = torch.ones_like(u_pred)
        return {
            "u_pred": u_pred,
            "delta_pred": delta_pred,
            "gate": gate,
            "raw_delta": delta_pred,
            "e_f_t": e_f_t,
        }


class SequenceDataset(Dataset):
    def __init__(
        self,
        x_base: np.ndarray,
        x_quality: np.ndarray,
        target_u: np.ndarray,
        current_u: np.ndarray,
        prev_u: np.ndarray,
        row_index: np.ndarray,
    ) -> None:
        self.x_base = torch.from_numpy(x_base).float()
        self.x_quality = torch.from_numpy(x_quality).float()
        self.target_u = torch.from_numpy(target_u).float()
        self.current_u = torch.from_numpy(current_u).float()
        self.prev_u = torch.from_numpy(prev_u).float()
        self.row_index = torch.from_numpy(row_index).long()

    def __len__(self) -> int:
        return self.x_base.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.x_base[idx],
            self.x_quality[idx],
            self.target_u[idx],
            self.current_u[idx],
            self.prev_u[idx],
            self.row_index[idx],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DSCGNet ablations on the paper train/val/test split."
    )
    parser.add_argument(
        "--raw-csv",
        default=str(PROJECT_DIR / "data" / "track-fusion-move_2026-4-17_new.csv"),
        help="Full CSV with legacy signal and quality-related columns.",
    )
    parser.add_argument(
        "--split-ref-csv",
        default=str(
            PROJECT_DIR
            / "data"
            / "train"
            / "track-fusion-move_2026-4-17_new_dscgnet_predictions.csv"
        ),
        help="CSV whose prediction_split/current_row_index columns define the reference split.",
    )
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument(
        "--experiments",
        default="",
        help="Comma-separated subset of experiment names to run. Default runs all.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(THIS_DIR),
        help="Directory for JSON/CSV/PNG outputs.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def extract_split_ranges(split_ref_csv: Path) -> Dict[str, Tuple[int, int]]:
    split_df = pd.read_csv(split_ref_csv)
    valid = split_df.loc[split_df[SPLIT_COL].isin(["train", "val", "test"])]
    if valid.empty:
        raise RuntimeError(f"No valid split rows found in {split_ref_csv}")
    ranges: Dict[str, Tuple[int, int]] = {}
    for split_name in ["train", "val", "test"]:
        rows = valid.loc[valid[SPLIT_COL] == split_name, INDEX_COL].astype(int)
        if rows.empty:
            raise RuntimeError(f"Split '{split_name}' is empty in {split_ref_csv}")
        ranges[split_name] = (int(rows.min()), int(rows.max()))
    return ranges


def load_raw_dataframe(raw_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    required = [
        TIMESTAMP_COL,
        LEGACY_X_COL,
        LEGACY_Y_COL,
        CONF_COL,
        WIDTH_COL,
        HEIGHT_COL,
        MATCHED_COL,
        MISS_COL,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {raw_csv}: {missing}")
    return df


def build_feature_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    legacy = df[[LEGACY_X_COL, LEGACY_Y_COL]].to_numpy(dtype=np.float32)

    d = np.zeros_like(legacy)
    d[1:] = legacy[1:] - legacy[:-1]

    dd = np.zeros_like(legacy)
    dd[1:] = d[1:] - d[:-1]

    timestamp = df[TIMESTAMP_COL].to_numpy(dtype=np.float64)
    dt = np.zeros(len(df), dtype=np.float32)
    dt[1:] = np.maximum((timestamp[1:] - timestamp[:-1]) / 1000.0, 0.0)

    conf = df[CONF_COL].to_numpy(dtype=np.float32)
    area = (df[WIDTH_COL].to_numpy(dtype=np.float32) * df[HEIGHT_COL].to_numpy(dtype=np.float32)).clip(min=0.0)
    matched = df[MATCHED_COL].to_numpy(dtype=np.float32)
    miss_count = df[MISS_COL].to_numpy(dtype=np.float32)
    miss_flag = ((matched <= 0.0) | (miss_count > 0.0)).astype(np.float32)

    base = np.concatenate([legacy, d, dd], axis=1).astype(np.float32)
    quality = np.stack([conf, np.log1p(area), miss_flag, dt], axis=1).astype(np.float32)

    return {
        "legacy": legacy,
        "base": base,
        "quality": quality,
    }


def resolve_split_name(current_idx: int, split_ranges: Dict[str, Tuple[int, int]]) -> str | None:
    for split_name, (start_idx, end_idx) in split_ranges.items():
        if start_idx <= current_idx <= end_idx:
            return split_name
    return None


def build_samples(
    arrays: Dict[str, np.ndarray],
    history_len: int,
    split_ranges: Dict[str, Tuple[int, int]],
) -> Dict[str, Dict[str, np.ndarray]]:
    base = arrays["base"]
    quality = arrays["quality"]
    legacy = arrays["legacy"]
    n_rows = len(base)
    splits = {
        "train": {"x_base": [], "x_quality": [], "target_u": [], "current_u": [], "prev_u": [], "row_index": []},
        "val": {"x_base": [], "x_quality": [], "target_u": [], "current_u": [], "prev_u": [], "row_index": []},
        "test": {"x_base": [], "x_quality": [], "target_u": [], "current_u": [], "prev_u": [], "row_index": []},
    }

    first_valid_idx = max(history_len - 1, split_ranges["train"][0])
    for current_idx in range(first_valid_idx, n_rows - 1):
        split_name = resolve_split_name(current_idx, split_ranges)
        if split_name is None:
            continue
        start = current_idx - history_len + 1
        x_base = base[start : current_idx + 1]
        x_quality = quality[start : current_idx + 1]
        target_u = legacy[current_idx + 1]
        current_u = legacy[current_idx]
        prev_u = legacy[current_idx - 1]

        splits[split_name]["x_base"].append(x_base)
        splits[split_name]["x_quality"].append(x_quality)
        splits[split_name]["target_u"].append(target_u)
        splits[split_name]["current_u"].append(current_u)
        splits[split_name]["prev_u"].append(prev_u)
        splits[split_name]["row_index"].append(current_idx)

    packed: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name, items in splits.items():
        if not items["x_base"]:
            raise RuntimeError(f"Split '{split_name}' is empty for history_len={history_len}")
        packed[split_name] = {
            "x_base": np.asarray(items["x_base"], dtype=np.float32),
            "x_quality": np.asarray(items["x_quality"], dtype=np.float32),
            "target_u": np.asarray(items["target_u"], dtype=np.float32),
            "current_u": np.asarray(items["current_u"], dtype=np.float32),
            "prev_u": np.asarray(items["prev_u"], dtype=np.float32),
            "row_index": np.asarray(items["row_index"], dtype=np.int64),
        }
    return packed


def normalize_splits(splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    train_base = splits["train"]["x_base"]
    train_quality = splits["train"]["x_quality"]
    base_mean = train_base.mean(axis=(0, 1), keepdims=True)
    base_std = train_base.std(axis=(0, 1), keepdims=True) + 1e-8
    quality_mean = train_quality.mean(axis=(0, 1), keepdims=True)
    quality_std = train_quality.std(axis=(0, 1), keepdims=True) + 1e-8
    legacy_mean = base_mean[:, :, :2]
    legacy_std = base_std[:, :, :2]

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name, payload in splits.items():
        out[split_name] = {
            **payload,
            "x_base": ((payload["x_base"] - base_mean) / base_std).astype(np.float32),
            "x_quality": ((payload["x_quality"] - quality_mean) / quality_std).astype(np.float32),
            "target_u_norm": ((payload["target_u"] - legacy_mean.squeeze(0)) / legacy_std.squeeze(0)).astype(np.float32),
            "current_u_norm": ((payload["current_u"] - legacy_mean.squeeze(0)) / legacy_std.squeeze(0)).astype(np.float32),
            "prev_u_norm": ((payload["prev_u"] - legacy_mean.squeeze(0)) / legacy_std.squeeze(0)).astype(np.float32),
        }
    out["stats"] = {
        "base_mean": base_mean.squeeze(0).squeeze(0).tolist(),
        "base_std": base_std.squeeze(0).squeeze(0).tolist(),
        "quality_mean": quality_mean.squeeze(0).squeeze(0).tolist(),
        "quality_std": quality_std.squeeze(0).squeeze(0).tolist(),
        "legacy_mean": legacy_mean.squeeze(0).squeeze(0).tolist(),
        "legacy_std": legacy_std.squeeze(0).squeeze(0).tolist(),
    }
    return out


def make_loader(
    payload: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = SequenceDataset(
        x_base=payload["x_base"],
        x_quality=payload["x_quality"],
        target_u=payload["target_u_norm"],
        current_u=payload["current_u_norm"],
        prev_u=payload["prev_u_norm"],
        row_index=payload["row_index"],
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def compute_loss(
    output: Dict[str, torch.Tensor],
    target_u: torch.Tensor,
    current_u: torch.Tensor,
    prev_u: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    u_pred = output["u_pred"]
    gate = output["gate"]

    loss_ahead = F.huber_loss(u_pred, target_u)

    pred_inc = u_pred - current_u
    true_inc = target_u - current_u
    loss_inc = torch.mean(torch.abs(pred_inc - true_inc))

    motion = torch.sum(torch.abs(current_u - prev_u), dim=-1)
    weight = torch.exp(-motion / args.tau)
    loss_stab = torch.mean(weight * torch.sum(torch.abs(u_pred - current_u), dim=-1))

    direction_dot = torch.sum(pred_inc * true_inc, dim=-1)
    loss_dir = torch.mean(F.relu(-direction_dot))

    loss_gate = torch.mean(torch.sum(torch.abs(gate), dim=-1))

    total = (
        args.lambda_ahead * loss_ahead
        + args.lambda_inc * loss_inc
        + args.lambda_stab * loss_stab
        + args.lambda_dir * loss_dir
        + args.lambda_gate * loss_gate
    )

    return total, {
        "ahead": float(loss_ahead.detach().cpu()),
        "inc": float(loss_inc.detach().cpu()),
        "stab": float(loss_stab.detach().cpu()),
        "dir": float(loss_dir.detach().cpu()),
        "gate": float(loss_gate.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def forward_model(
    model: nn.Module,
    x_base: torch.Tensor,
    x_quality: torch.Tensor,
    use_quality: bool,
) -> Dict[str, torch.Tensor]:
    if use_quality:
        return model(x_base, quality=x_quality)
    return model(x_base, quality=None)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    use_quality: bool,
    legacy_mean: np.ndarray,
    legacy_std: np.ndarray,
) -> Dict[str, object]:
    model.eval()
    losses = []
    rows: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    current_u_list: List[np.ndarray] = []

    with torch.no_grad():
        for x_base, x_quality, target_u, current_u, prev_u, row_index in loader:
            x_base = x_base.to(device)
            x_quality = x_quality.to(device)
            target_u = target_u.to(device)
            current_u = current_u.to(device)
            prev_u = prev_u.to(device)

            output = forward_model(model, x_base, x_quality, use_quality=use_quality)
            _, parts = compute_loss(output, target_u, current_u, prev_u, args)
            losses.append(parts["total"])

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

    metrics = {
        "loss": float(np.mean(losses)),
        "ahead_mae": float(np.mean(np.abs(pred_u - target_u))),
        "ahead_rmse": float(np.sqrt(np.mean((pred_u - target_u) ** 2))),
        "jitter": float(np.mean(np.abs(np.diff(pred_u, axis=0)))) if len(pred_u) > 1 else 0.0,
        "spike_rate": float(np.mean(np.abs(np.diff(pred_u, axis=0)) > args.spike_threshold))
        if len(pred_u) > 1
        else 0.0,
        "sign_flip_rate": float(np.mean(np.sign(pred_inc) != np.sign(true_inc))),
        "count": int(len(pred_u)),
        "row_start": int(row_index[0]),
        "row_end": int(row_index[-1]),
    }
    return {
        "metrics": metrics,
        "pred_u": pred_u,
        "target_u": target_u,
        "current_u": current_u,
        "row_index": row_index,
    }


def train_one_experiment(
    cfg: ExperimentConfig,
    arrays: Dict[str, np.ndarray],
    split_ranges: Dict[str, Tuple[int, int]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    samples = build_samples(arrays=arrays, history_len=cfg.history_len, split_ranges=split_ranges)
    normalized = normalize_splits(samples)

    train_loader = make_loader(normalized["train"], args.batch_size, True, args.num_workers)
    val_loader = make_loader(normalized["val"], args.batch_size, False, args.num_workers)
    test_loader = make_loader(normalized["test"], args.batch_size, False, args.num_workers)
    legacy_mean = np.asarray(normalized["stats"]["legacy_mean"], dtype=np.float32)
    legacy_std = np.asarray(normalized["stats"]["legacy_std"], dtype=np.float32)

    if cfg.output_head == "state":
        model = DSCGNet(
            hidden=args.hidden,
            use_attention=cfg.use_attention,
            use_quality=cfg.use_quality,
            rmax_x=args.rmax_x,
            rmax_y=args.rmax_y,
        ).to(device)
    elif cfg.output_head == "direct":
        model = DirectRegressionDSCGNet(
            hidden=args.hidden,
            use_attention=cfg.use_attention,
            use_quality=cfg.use_quality,
        ).to(device)
    else:
        raise ValueError(f"Unsupported output_head: {cfg.output_head}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = math.inf
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []

        for x_base, x_quality, target_u, current_u, prev_u, _ in train_loader:
            x_base = x_base.to(device)
            x_quality = x_quality.to(device)
            target_u = target_u.to(device)
            current_u = current_u.to(device)
            prev_u = prev_u.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = forward_model(model, x_base, x_quality, use_quality=cfg.use_quality)
            loss, _ = compute_loss(output, target_u, current_u, prev_u, args)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        val_eval = evaluate_model(
            model,
            val_loader,
            device,
            args,
            use_quality=cfg.use_quality,
            legacy_mean=legacy_mean,
            legacy_std=legacy_std,
        )
        val_mae = float(val_eval["metrics"]["ahead_mae"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "val_ahead_mae": val_mae,
                "val_ahead_rmse": float(val_eval["metrics"]["ahead_rmse"]),
            }
        )

        if val_mae < best_val_mae - 1e-8:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    model.load_state_dict(best_state)
    val_eval = evaluate_model(
        model,
        val_loader,
        device,
        args,
        use_quality=cfg.use_quality,
        legacy_mean=legacy_mean,
        legacy_std=legacy_std,
    )
    test_eval = evaluate_model(
        model,
        test_loader,
        device,
        args,
        use_quality=cfg.use_quality,
        legacy_mean=legacy_mean,
        legacy_std=legacy_std,
    )

    return {
        "config": {
            "history_len": cfg.history_len,
            "use_quality": cfg.use_quality,
            "use_attention": cfg.use_attention,
            "output_head": cfg.output_head,
            "params": count_params(model),
        },
        "best_epoch": best_epoch,
        "train_samples": int(len(normalized["train"]["target_u"])),
        "val_samples": int(len(normalized["val"]["target_u"])),
        "test_samples": int(len(normalized["test"]["target_u"])),
        "stats": normalized["stats"],
        "val_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "history": history,
    }


def build_results_table(results: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for key, payload in results.items():
        metrics = payload["test_metrics"]
        rows.append(
            {
                "experiment": key,
                "display_name": payload["display_name"],
                "history_len": payload["config"]["history_len"],
                "use_quality": payload["config"]["use_quality"],
                "use_attention": payload["config"]["use_attention"],
                "output_head": payload["config"].get("output_head", "state"),
                "params": payload["config"]["params"],
                "ahead_mae": metrics["ahead_mae"],
                "ahead_rmse": metrics["ahead_rmse"],
                "jitter": metrics["jitter"],
                "spike_rate": metrics["spike_rate"],
                "sign_flip_rate": metrics["sign_flip_rate"],
                "best_epoch": payload["best_epoch"],
            }
        )
    return pd.DataFrame(rows)


def plot_ablation_figure(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    history_keys = ["hist_8", "hist_12", "full", "hist_20"]
    quality_keys = ["quality_off", "full"]
    attention_keys = ["attn_off", "full"]
    output_keys = ["direct_out", "full"]

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 3.8), dpi=220)

    x_hist = np.arange(len(history_keys))
    hist_labels = [str(results[k]["config"]["history_len"]) for k in history_keys]
    hist_vals = [results[k]["test_metrics"]["ahead_mae"] for k in history_keys]
    axes[0].plot(x_hist, hist_vals, marker="o", linewidth=1.8, color="#1f5aa6")
    axes[0].set_xticks(x_hist)
    axes[0].set_xticklabels(hist_labels)
    axes[0].set_xlabel("History length T")
    axes[0].set_ylabel("Test Ahead MAE")
    axes[0].set_title("History length ablation")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    best_idx = int(np.argmin(hist_vals))
    axes[0].scatter([best_idx], [hist_vals[best_idx]], color="#c0392b", zorder=5)

    x_quality = np.arange(len(quality_keys))
    quality_vals = [results[k]["test_metrics"]["ahead_mae"] for k in quality_keys]
    axes[1].bar(x_quality, quality_vals, color=["#d4b483", "#3a7d44"], width=0.58)
    axes[1].set_xticks(x_quality)
    axes[1].set_xticklabels(["w/o quality", "w/ quality"])
    axes[1].set_ylabel("Test Ahead MAE")
    axes[1].set_title("Quality branch ablation")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)

    x_attention = np.arange(len(attention_keys))
    attention_vals = [results[k]["test_metrics"]["ahead_mae"] for k in attention_keys]
    axes[2].bar(x_attention, attention_vals, color=["#8e6bbd", "#2b6cb0"], width=0.58)
    axes[2].set_xticks(x_attention)
    axes[2].set_xticklabels(["w/o attention", "w/ attention"])
    axes[2].set_ylabel("Test Ahead MAE")
    axes[2].set_title("Causal attention ablation")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)

    x_output = np.arange(len(output_keys))
    output_vals = [results[k]["test_metrics"]["ahead_mae"] for k in output_keys]
    axes[3].bar(x_output, output_vals, color=["#b35c1e", "#1f5aa6"], width=0.58)
    axes[3].set_xticks(x_output)
    axes[3].set_xticklabels(["direct regression", "state transition"])
    axes[3].set_ylabel("Test Ahead MAE")
    axes[3].set_title("Output head ablation")
    axes[3].grid(True, axis="y", linestyle="--", alpha=0.35)

    for ax in axes:
        ax.set_axisbelow(True)
        for label in ax.get_xticklabels():
            label.set_fontsize(9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = Path(args.raw_csv).resolve()
    split_ref_csv = Path(args.split_ref_csv).resolve()

    split_ranges = extract_split_ranges(split_ref_csv)
    df = load_raw_dataframe(raw_csv)
    arrays = build_feature_arrays(df)

    configs = [
        ExperimentConfig("hist_8", "T=8", history_len=8, use_quality=True, use_attention=True),
        ExperimentConfig("hist_12", "T=12", history_len=12, use_quality=True, use_attention=True),
        ExperimentConfig("full", "T=16 + quality + attention", history_len=16, use_quality=True, use_attention=True),
        ExperimentConfig("hist_20", "T=20", history_len=20, use_quality=True, use_attention=True),
        ExperimentConfig("quality_off", "w/o quality branch", history_len=16, use_quality=False, use_attention=True),
        ExperimentConfig("attn_off", "w/o causal attention", history_len=16, use_quality=True, use_attention=False),
        ExperimentConfig(
            "direct_out",
            "T=16 + quality + attention + direct output",
            history_len=16,
            use_quality=True,
            use_attention=True,
            output_head="direct",
        ),
    ]

    selected = {name.strip() for name in args.experiments.split(",") if name.strip()}
    if selected:
        configs = [cfg for cfg in configs if cfg.name in selected]
        if not configs:
            raise RuntimeError(f"No experiments matched --experiments={args.experiments!r}")

    results: Dict[str, Dict[str, object]] = {}
    for cfg in configs:
        print(
            f"[ablation] {cfg.name}: T={cfg.history_len}, "
            f"quality={cfg.use_quality}, attention={cfg.use_attention}, output={cfg.output_head}"
        )
        payload = train_one_experiment(
            cfg=cfg,
            arrays=arrays,
            split_ranges=split_ranges,
            args=args,
            device=device,
        )
        payload["display_name"] = cfg.display_name
        results[cfg.name] = payload
        print(
            f"  test Ahead MAE={payload['test_metrics']['ahead_mae']:.4f}, "
            f"Ahead RMSE={payload['test_metrics']['ahead_rmse']:.4f}"
        )

    summary = {
        "raw_csv": str(raw_csv),
        "split_ref_csv": str(split_ref_csv),
        "split_ranges": {k: list(v) for k, v in split_ranges.items()},
        "device": str(device),
        "seed": args.seed,
        "experiments": results,
    }

    json_path = output_dir / "ablation_results.json"
    csv_path = output_dir / "ablation_results.csv"
    fig_path = output_dir / "ablation_overview.png"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    build_results_table(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    if {"hist_8", "hist_12", "full", "hist_20", "quality_off", "attn_off", "direct_out"}.issubset(results):
        plot_ablation_figure(results, fig_path)

    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {fig_path}")


if __name__ == "__main__":
    main()
