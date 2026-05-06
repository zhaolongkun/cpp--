from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class OffsetDatasetBundle:
    X_train: np.ndarray
    baseline_seq_train: np.ndarray
    baseline_train: np.ndarray
    stable_baseline_seq_train: np.ndarray
    stable_baseline_train: np.ndarray
    y_clean_seq_train: np.ndarray
    y_clean_train: np.ndarray
    w_train: np.ndarray
    X_val: np.ndarray
    baseline_seq_val: np.ndarray
    baseline_val: np.ndarray
    stable_baseline_seq_val: np.ndarray
    stable_baseline_val: np.ndarray
    y_clean_seq_val: np.ndarray
    y_clean_val: np.ndarray
    w_val: np.ndarray
    X_test: np.ndarray
    baseline_seq_test: np.ndarray
    baseline_test: np.ndarray
    stable_baseline_seq_test: np.ndarray
    stable_baseline_test: np.ndarray
    y_clean_seq_test: np.ndarray
    y_clean_test: np.ndarray
    w_test: np.ndarray


class OffsetWindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        baseline_seq: np.ndarray,
        baseline: np.ndarray,
        stable_baseline_seq: np.ndarray,
        stable_baseline: np.ndarray,
        y_clean_seq: np.ndarray,
        y_clean: np.ndarray,
        weights: np.ndarray,
    ):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.baseline_seq = torch.from_numpy(baseline_seq.astype(np.float32))
        self.baseline = torch.from_numpy(baseline.astype(np.float32))
        self.stable_baseline_seq = torch.from_numpy(stable_baseline_seq.astype(np.float32))
        self.stable_baseline = torch.from_numpy(stable_baseline.astype(np.float32))
        self.y_clean_seq = torch.from_numpy(y_clean_seq.astype(np.float32))
        self.y_clean = torch.from_numpy(y_clean.astype(np.float32))
        self.weights = torch.from_numpy(weights.astype(np.float32))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return {
            "x": self.X[idx],
            "baseline_seq": self.baseline_seq[idx],
            "baseline": self.baseline[idx],
            "stable_baseline_seq": self.stable_baseline_seq[idx],
            "stable_baseline": self.stable_baseline[idx],
            "y_clean_seq": self.y_clean_seq[idx],
            "y_clean": self.y_clean[idx],
            "weight": self.weights[idx],
        }


def load_npz_bundle(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def make_datasets(bundle: Dict[str, np.ndarray]) -> Tuple[OffsetWindowDataset, OffsetWindowDataset, OffsetWindowDataset]:
    train = OffsetWindowDataset(
        bundle["X_train"],
        bundle["baseline_seq_train"],
        bundle["baseline_train"],
        bundle["stable_baseline_seq_train"],
        bundle["stable_baseline_train"],
        bundle["y_clean_seq_train"],
        bundle["y_clean_train"],
        bundle["w_train"],
    )
    val = OffsetWindowDataset(
        bundle["X_val"],
        bundle["baseline_seq_val"],
        bundle["baseline_val"],
        bundle["stable_baseline_seq_val"],
        bundle["stable_baseline_val"],
        bundle["y_clean_seq_val"],
        bundle["y_clean_val"],
        bundle["w_val"],
    )
    test = OffsetWindowDataset(
        bundle["X_test"],
        bundle["baseline_seq_test"],
        bundle["baseline_test"],
        bundle["stable_baseline_seq_test"],
        bundle["stable_baseline_test"],
        bundle["y_clean_seq_test"],
        bundle["y_clean_test"],
        bundle["w_test"],
    )
    return train, val, test
