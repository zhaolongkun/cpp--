import argparse
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from build_dataset import main as build_dataset_main
from model import ModelBounds, NCEGRU


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train neural module for PNR-IMM-KF")
    p.add_argument("--dataset", required=True, help="npz from build_dataset.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--w_bias", type=float, default=1.0)
    p.add_argument("--w_alpha", type=float, default=0.8)
    p.add_argument("--w_outlier", type=float, default=0.6)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--auto_build", action="store_true", help="auto build dataset if path does not exist")
    p.add_argument("--csv", default="", help="input csv path for --auto_build")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _auto_build_dataset(dataset_path: str, csv_path: str) -> None:
    if not csv_path:
        raise RuntimeError("--auto_build requires --csv")
    argv_backup = list(os.sys.argv)
    try:
        os.sys.argv = [
            "build_dataset.py",
            "--csv",
            csv_path,
            "--out",
            dataset_path,
        ]
        build_dataset_main()
    finally:
        os.sys.argv = argv_backup


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    d = np.load(path, allow_pickle=True)
    x_train = d["x_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.float32)
    x_val = d["x_val"].astype(np.float32)
    y_val = d["y_val"].astype(np.float32)

    meta = {
        "feature_names": [str(x) for x in d["feature_names"].tolist()],
        "target_names": [str(x) for x in d["target_names"].tolist()],
        "seq_len": int(d["seq_len"][0]),
    }
    return x_train, y_train, x_val, y_val, meta


def standardize(x_train: np.ndarray, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0).astype(np.float32)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x_train_n = (x_train - mean[None, None, :]) / std[None, None, :]
    x_val_n = (x_val - mean[None, None, :]) / std[None, None, :]
    return x_train_n.astype(np.float32), x_val_n.astype(np.float32), mean, std


def compute_loss(pred: torch.Tensor, tgt: torch.Tensor, w_bias: float, w_alpha: float, w_outlier: float) -> torch.Tensor:
    bias_loss = F.smooth_l1_loss(pred[:, 0:2], tgt[:, 0:2])
    alpha_loss = F.mse_loss(pred[:, 2:4], tgt[:, 2:4])
    outlier_loss = F.binary_cross_entropy(
        torch.clamp(pred[:, 4:5], 1e-4, 1.0 - 1e-4),
        torch.clamp(tgt[:, 4:5], 1e-4, 1.0 - 1e-4),
    )
    return w_bias * bias_loss + w_alpha * alpha_loss + w_outlier * outlier_loss


def eval_metrics(pred: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    e = pred - tgt
    return {
        "rmse_bias_x": float(np.sqrt(np.mean(e[:, 0] ** 2))),
        "rmse_bias_y": float(np.sqrt(np.mean(e[:, 1] ** 2))),
        "rmse_alpha_q": float(np.sqrt(np.mean(e[:, 2] ** 2))),
        "rmse_alpha_r": float(np.sqrt(np.mean(e[:, 3] ** 2))),
        "mae_outlier_prob": float(np.mean(np.abs(e[:, 4]))),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(
        json.dumps(
            {
                "train_mode": "from_scratch_random_init",
                "pretrained_weights": False,
                "seed": int(args.seed),
            },
            ensure_ascii=False,
        )
    )

    if not os.path.exists(args.dataset):
        if not args.auto_build:
            raise RuntimeError(f"dataset not found: {args.dataset}")
        _auto_build_dataset(args.dataset, args.csv)

    x_train, y_train, x_val, y_val, meta = load_data(args.dataset)
    x_train, x_val, mean, std = standardize(x_train, x_val)

    train_ds = SeqDataset(x_train, y_train)
    val_ds = SeqDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    bounds = ModelBounds()
    # Strict from-scratch policy: initialize model weights randomly each run.
    model = NCEGRU(
        input_dim=x_train.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bounds=bounds,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")
    no_improve = 0

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            pred = model(xb)
            loss = compute_loss(pred, yb, args.w_bias, args.w_alpha, args.w_outlier)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            train_loss += float(loss.detach().cpu().item()) * xb.shape[0]
            train_count += int(xb.shape[0])
        train_loss /= max(1, train_count)

        model.eval()
        val_loss = 0.0
        val_count = 0
        preds = []
        tgts = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                pred = model(xb)
                loss = compute_loss(pred, yb, args.w_bias, args.w_alpha, args.w_outlier)
                val_loss += float(loss.detach().cpu().item()) * xb.shape[0]
                val_count += int(xb.shape[0])
                preds.append(pred.cpu().numpy())
                tgts.append(yb.cpu().numpy())
        val_loss /= max(1, val_count)

        pred_np = np.concatenate(preds, axis=0) if preds else np.zeros((0, 5), dtype=np.float32)
        tgt_np = np.concatenate(tgts, axis=0) if tgts else np.zeros((0, 5), dtype=np.float32)
        metrics = eval_metrics(pred_np, tgt_np) if pred_np.shape[0] > 0 else {}

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": int(x_train.shape[-1]),
                    "seq_len": int(meta["seq_len"]),
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "dropout": float(args.dropout),
                    "bounds": bounds.to_dict(),
                    "training_mode": "from_scratch_random_init",
                    "pretrained_weights": False,
                    "norm_mean": mean.tolist(),
                    "norm_std": std.tolist(),
                    "feature_names": meta["feature_names"],
                    "target_names": meta["target_names"],
                    "metrics": metrics,
                },
                best_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[train_nce] early stop at epoch={epoch}")
                break

    hist_path = os.path.join(args.out_dir, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[train_nce] best checkpoint: {best_path}")
    print(f"[train_nce] history: {hist_path}")


if __name__ == "__main__":
    main()
