#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2S-LAVS-Lite 训练脚本

读取 build_dataset.py 产出的 dataset_t8.npz，训练 GRU 残差补偿网络。
"""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gru_residual import build_model_from_cfg


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3:
            raise ValueError(f"X 维度应为 [N,T,F]，当前: {X.shape}")
        if y.ndim != 2:
            raise ValueError(f"y 维度应为 [N,2]，当前: {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y 样本数不一致: {X.shape[0]} vs {y.shape[0]}")
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GRU residual compensator for R2S-LAVS-Lite")
    p.add_argument("--dataset_npz", type=str, required=True, help="build_dataset.py 输出的 npz 路径")
    p.add_argument("--out_dir", type=str, required=True, help="训练输出目录（ckpt/日志）")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0, help="<=0 表示不启用")

    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fc_hidden", type=int, default=32)

    p.add_argument("--lambda_smooth", type=float, default=0.1)
    p.add_argument("--lambda_mag", type=float, default=0.01)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=0, help=">0 时每 N epoch 存一个中间ckpt")
    p.add_argument("--resume", type=str, default="", help="从 checkpoint 恢复训练（可选）")
    p.add_argument("--print_every", type=int, default=1)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda，但当前环境不可用。")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    required = [
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test",
        "y_test",
    ]
    for k in required:
        if k not in data:
            raise KeyError(f"npz 缺少键: {k}")

    out = {k: data[k] for k in data.files}
    return out


def _safe_loader(ds: SequenceDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    if len(ds) == 0:
        # 空集仍返回 DataLoader，后续评估函数会处理 count=0 的情况
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_smooth: float,
    lambda_mag: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total_samples = 0
    sum_reg = 0.0
    sum_smooth = 0.0
    sum_mag = 0.0
    sum_total = 0.0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = X.size(0)

        optimizer.zero_grad()

        pred = model(X)  # [B,2]
        seq_pred = model.forward_sequence(X)  # [B,T,2]

        loss_reg = criterion(pred, y)
        if seq_pred.size(1) > 1:
            loss_smooth = torch.mean(torch.abs(seq_pred[:, 1:, :] - seq_pred[:, :-1, :]))
        else:
            loss_smooth = torch.zeros((), dtype=pred.dtype, device=pred.device)
        loss_mag = torch.mean(pred * pred)

        loss = loss_reg + float(lambda_smooth) * loss_smooth + float(lambda_mag) * loss_mag
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        total_samples += bs
        sum_reg += float(loss_reg.item()) * bs
        sum_smooth += float(loss_smooth.item()) * bs
        sum_mag += float(loss_mag.item()) * bs
        sum_total += float(loss.item()) * bs

    if total_samples == 0:
        return {
            "reg": float("nan"),
            "smooth": float("nan"),
            "mag": float("nan"),
            "total": float("nan"),
        }
    return {
        "reg": sum_reg / total_samples,
        "smooth": sum_smooth / total_samples,
        "mag": sum_mag / total_samples,
        "total": sum_total / total_samples,
    }


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lambda_smooth: float,
    lambda_mag: float,
) -> Dict[str, float]:
    model.eval()
    total_samples = 0
    sum_reg = 0.0
    sum_smooth = 0.0
    sum_mag = 0.0
    sum_total = 0.0

    abs_err_sum = 0.0
    sq_err_sum = 0.0
    err_count = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = X.size(0)

        pred = model(X)
        seq_pred = model.forward_sequence(X)

        loss_reg = criterion(pred, y)
        if seq_pred.size(1) > 1:
            loss_smooth = torch.mean(torch.abs(seq_pred[:, 1:, :] - seq_pred[:, :-1, :]))
        else:
            loss_smooth = torch.zeros((), dtype=pred.dtype, device=pred.device)
        loss_mag = torch.mean(pred * pred)
        loss = loss_reg + float(lambda_smooth) * loss_smooth + float(lambda_mag) * loss_mag

        err = pred - y
        abs_err_sum += float(torch.sum(torch.abs(err)).item())
        sq_err_sum += float(torch.sum(err * err).item())
        err_count += int(err.numel())

        total_samples += bs
        sum_reg += float(loss_reg.item()) * bs
        sum_smooth += float(loss_smooth.item()) * bs
        sum_mag += float(loss_mag.item()) * bs
        sum_total += float(loss.item()) * bs

    if total_samples == 0 or err_count == 0:
        return {
            "reg": float("nan"),
            "smooth": float("nan"),
            "mag": float("nan"),
            "total": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }

    mae = abs_err_sum / err_count
    rmse = np.sqrt(sq_err_sum / err_count)
    return {
        "reg": sum_reg / total_samples,
        "smooth": sum_smooth / total_samples,
        "mag": sum_mag / total_samples,
        "total": sum_total / total_samples,
        "mae": float(mae),
        "rmse": float(rmse),
    }


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    model_cfg: dict,
    train_cfg: dict,
    feature_names: list,
    target_names: list,
    metrics: dict,
) -> None:
    ckpt = {
        "epoch": int(epoch),
        "best_val_total": float(best_val),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "feature_names": feature_names,
        "target_names": target_names,
        "metrics": metrics,
    }
    torch.save(ckpt, str(path))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = choose_device(args.device)
    print(f"[Info] device: {device}")

    data = load_dataset(args.dataset_npz)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    if X_train.shape[0] == 0:
        raise RuntimeError("训练集为空，无法训练。")
    if X_train.shape[1] <= 0 or X_train.shape[2] <= 0:
        raise RuntimeError(f"训练输入 shape 非法: {X_train.shape}")
    if y_train.shape[1] != 2:
        raise RuntimeError(f"训练标签维度应为2，当前: {y_train.shape}")

    feature_names = data["feature_names"].tolist() if "feature_names" in data else []
    target_names = data["target_names"].tolist() if "target_names" in data else []

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = _safe_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = _safe_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = _safe_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_cfg = {
        "input_dim": int(X_train.shape[2]),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "fc_hidden": int(args.fc_hidden),
        "output_dim": int(y_train.shape[1]),
    }
    model = build_model_from_cfg(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.SmoothL1Loss(reduction="mean")

    start_epoch = 1
    best_val_total = float("inf")

    # 可选恢复训练
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"resume checkpoint 不存在: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_total = float(ckpt.get("best_val_total", float("inf")))
        print(f"[Info] Resume from {args.resume}, start_epoch={start_epoch}, best_val_total={best_val_total:.6f}")

    train_cfg = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "lambda_smooth": float(args.lambda_smooth),
        "lambda_mag": float(args.lambda_mag),
        "seq_len": int(X_train.shape[1]),
        "input_dim": int(X_train.shape[2]),
        "output_dim": int(y_train.shape[1]),
        "seed": int(args.seed),
    }

    history = []
    best_ckpt_path = out_dir / "best.pt"
    last_ckpt_path = out_dir / "last.pt"
    use_val_for_selection = len(val_ds) > 0
    if not use_val_for_selection:
        print("[Warn] 验证集为空，将使用 train_total 作为 best checkpoint 选择指标。")

    t0 = time.time()
    for epoch in range(start_epoch, int(args.epochs) + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            lambda_smooth=args.lambda_smooth,
            lambda_mag=args.lambda_mag,
            grad_clip=args.grad_clip,
        )
        va = eval_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            lambda_smooth=args.lambda_smooth,
            lambda_mag=args.lambda_mag,
        )

        row = {
            "epoch": int(epoch),
            "train_total": tr["total"],
            "train_reg": tr["reg"],
            "train_smooth": tr["smooth"],
            "train_mag": tr["mag"],
            "val_total": va["total"],
            "val_reg": va["reg"],
            "val_smooth": va["smooth"],
            "val_mag": va["mag"],
            "val_mae": va["mae"],
            "val_rmse": va["rmse"],
        }
        history.append(row)

        if (epoch % int(args.print_every) == 0) or (epoch == start_epoch):
            print(
                "[Epoch {}/{}] "
                "train_total={:.6f} val_total={:.6f} val_mae={:.6f} val_rmse={:.6f}".format(
                    epoch, args.epochs, tr["total"], va["total"], va["mae"], va["rmse"]
                )
            )

        # 每轮存 last
        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val=best_val_total,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            feature_names=feature_names,
            target_names=target_names,
            metrics=row,
        )

        # 保存 best
        monitor_value = float(va["total"]) if use_val_for_selection else float(tr["total"])
        monitor_name = "val_total" if use_val_for_selection else "train_total"
        if np.isfinite(monitor_value) and (monitor_value < best_val_total):
            best_val_total = monitor_value
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val=best_val_total,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                feature_names=feature_names,
                target_names=target_names,
                metrics=row,
            )
            print(f"[Info] New best checkpoint saved: {best_ckpt_path} ({monitor_name}={best_val_total:.6f})")

        # 可选中间ckpt
        if int(args.save_every) > 0 and epoch % int(args.save_every) == 0:
            ep_ckpt = out_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                path=ep_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val=best_val_total,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                feature_names=feature_names,
                target_names=target_names,
                metrics=row,
            )

    # 兜底：若未产生 best，则复制 last 为 best，避免导出阶段失败
    if (not best_ckpt_path.exists()) and last_ckpt_path.exists():
        shutil.copy2(str(last_ckpt_path), str(best_ckpt_path))
        print(f"[Warn] 未生成 best.pt，已复制 last.pt -> {best_ckpt_path}")

    # 训练完成后，加载 best 并在 test 上评估
    best_metrics: Optional[Dict[str, float]] = None
    if best_ckpt_path.exists():
        ckpt = torch.load(str(best_ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device)
        best_metrics = eval_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            lambda_smooth=args.lambda_smooth,
            lambda_mag=args.lambda_mag,
        )
        print(
            "[Test@Best] total={:.6f} reg={:.6f} mae={:.6f} rmse={:.6f}".format(
                best_metrics["total"],
                best_metrics["reg"],
                best_metrics["mae"],
                best_metrics["rmse"],
            )
        )

    elapsed = time.time() - t0
    summary = {
        "dataset_npz": os.path.abspath(args.dataset_npz),
        "out_dir": os.path.abspath(str(out_dir)),
        "device": str(device),
        "elapsed_sec": float(elapsed),
        "best_val_total": float(best_val_total),
        "best_selection_metric": "val_total" if use_val_for_selection else "train_total",
        "last_epoch": int(args.epochs),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "test_size": int(len(test_ds)),
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "best_test_metrics": best_metrics,
    }

    history_path = out_dir / "train_history.json"
    save_json({"history": history}, history_path)
    summary_path = out_dir / "train_summary.json"
    save_json(summary, summary_path)

    print(f"[Done] best ckpt: {best_ckpt_path}")
    print(f"[Done] last ckpt: {last_ckpt_path}")
    print(f"[Done] history: {history_path}")
    print(f"[Done] summary: {summary_path}")


if __name__ == "__main__":
    main()
