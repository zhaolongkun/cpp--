from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
import math

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.datasets import load_npz_bundle, make_datasets
from la_cspc_ornet.features import check_feature_index_alignment
from la_cspc_ornet.losses import Stage1LossConfig, compute_stage1_losses
from la_cspc_ornet.models.factory import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, device, optimizer, loss_cfg: Stage1LossConfig):
    train = optimizer is not None
    model.train(train)
    meters = {
        "clean": 0.0,
        "smooth": 0.0,
        "turn": 0.0,
        "delta": 0.0,
        "peak": 0.0,
        "gate_align": 0.0,
        "gate_sparse": 0.0,
        "total": 0.0,
    }
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            batch["x"],
            baseline_seq=batch.get("baseline_seq"),
            stable_baseline_seq=batch.get("stable_baseline_seq"),
        )
        losses = compute_stage1_losses(outputs, batch, loss_cfg)
        if train:
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        bs = int(batch["x"].shape[0])
        count += bs
        for k in meters:
            meters[k] += float(losses[k].item()) * bs
    if count == 0:
        return {k: float("nan") for k in meters}
    return {k: v / count for k, v in meters.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train stage-1 clean offset reconstruction model")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    if not check_feature_index_alignment():
        raise RuntimeError("feature index alignment check failed; aborting stage1 training")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["train"]["seed"]))
    device = choose_device(str(cfg["train"]["device"]))

    bundle = load_npz_bundle(cfg["paths"]["dataset_npz"])
    train_ds, val_ds, test_ds = make_datasets(bundle)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    model_name, model_cfg, model = build_model(cfg["model"])
    model = model.to(device)
    loss_cfg = Stage1LossConfig(**cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    history = []
    best_val = float("inf")
    best_path = out_dir / "best_stage1_clean.pt"
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, loss_cfg)
        val_metrics = run_epoch(model, val_loader, device, None, loss_cfg)
        rec = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(rec)
        monitor = val_metrics["total"]
        if math.isnan(monitor):
            monitor = train_metrics["total"]
        if epoch == 1 or monitor < best_val:
            best_val = monitor
            torch.save({"model": model.state_dict(), "config": cfg, "model_name": model_name}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = run_epoch(model, test_loader, device, None, loss_cfg)
    summary = {"best_val_total": best_val, "test": test_metrics, "best_ckpt": str(best_path), "device": str(device)}
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
