from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Mapping

import numpy as np
import onnxruntime as ort
import pandas as pd

from .features import ALL_FEATURE_COLUMNS
from .stage1_common import (
    Stage1NormalizationStats,
    Stage1PrefilterConfig,
    Stage1TeacherConfig,
    apply_feature_stats,
    apply_stage1_feature_transforms,
    prefilter_cfg_from_dict,
    teacher_cfg_from_dict,
)


RUNTIME_REQUIRED_COLUMNS: List[str] = [
    "run_id",
    "timestamp_ms",
    "frame_id",
    "segment_tag",
    "segment_type",
    "dt_ms",
    "img_w",
    "img_h",
    "bbox_raw_x1",
    "bbox_raw_y1",
    "bbox_raw_x2",
    "bbox_raw_y2",
    "bbox_area_px",
    "det_conf",
    "dx_raw",
    "dy_raw",
    "dx_hat",
    "dy_hat",
    "vx_hat",
    "vy_hat",
    "lost_flag",
    "is_meas_update",
    "meas_age_ms",
    "zoom_delta",
    "coast_count",
    "note",
]


@dataclass(frozen=True)
class Stage1RuntimeOutput:
    ready: bool
    clean_dx: float
    clean_dy: float
    switch_gate: float
    seq_len: int


def load_runtime_normalization(meta_json: str | Path) -> Stage1NormalizationStats:
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    feature_names = list(meta["feature_columns"])
    norm = meta["normalization"]
    median = np.asarray([norm[name]["median"] for name in feature_names], dtype=np.float32)
    iqr = np.asarray([norm[name]["iqr"] for name in feature_names], dtype=np.float32)
    return Stage1NormalizationStats(median=median, iqr=iqr)


class Stage1OnnxRuntime:
    def __init__(
        self,
        onnx_path: str | Path,
        meta_json: str | Path,
        session_providers: List[str] | None = None,
    ):
        self.onnx_path = str(onnx_path)
        self.meta_path = str(meta_json)
        self.meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
        self.seq_len = int(self.meta["seq_len"])
        self.feature_columns = list(self.meta["feature_columns"])
        self.norm = load_runtime_normalization(meta_json)
        self.teacher_cfg = teacher_cfg_from_dict(self.meta.get("teacher"))
        self.prefilter_cfg = prefilter_cfg_from_dict(self.meta.get("prefilter"))
        providers = session_providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.buffer: Deque[Dict[str, object]] = deque(maxlen=self.seq_len)

    def reset(self) -> None:
        self.buffer.clear()

    def push(self, frame: Mapping[str, object]) -> Stage1RuntimeOutput:
        record = self._coerce_frame(frame)
        self.buffer.append(record)
        if len(self.buffer) < self.seq_len:
            return Stage1RuntimeOutput(
                ready=False,
                clean_dx=0.0,
                clean_dy=0.0,
                switch_gate=0.0,
                seq_len=len(self.buffer),
            )
        return self._infer_current_window()

    def _coerce_frame(self, frame: Mapping[str, object]) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for key in RUNTIME_REQUIRED_COLUMNS:
            if key in frame:
                out[key] = frame[key]
            elif key == "note":
                out[key] = ""
            elif key in {"run_id", "segment_tag", "segment_type"}:
                out[key] = "runtime"
            else:
                out[key] = 0.0
        return out

    def _infer_current_window(self) -> Stage1RuntimeOutput:
        df = pd.DataFrame(list(self.buffer))
        df = apply_stage1_feature_transforms(
            df,
            teacher_cfg=self.teacher_cfg,
            prefilter_cfg=self.prefilter_cfg,
        )
        x = df[self.feature_columns].to_numpy(dtype=np.float32)[None, :, :]
        x = apply_feature_stats(x, self.norm)
        baseline_seq = df[["input_dx", "input_dy"]].to_numpy(dtype=np.float32)[None, :, :]
        stable_baseline_seq = df[["stable_baseline_dx", "stable_baseline_dy"]].to_numpy(dtype=np.float32)[None, :, :]
        clean, switch_gate = self.session.run(
            ["clean", "switch_gate"],
            {
                "x": x,
                "baseline_seq": baseline_seq,
                "stable_baseline_seq": stable_baseline_seq,
            },
        )
        return Stage1RuntimeOutput(
            ready=True,
            clean_dx=float(clean[0, 0]),
            clean_dy=float(clean[0, 1]),
            switch_gate=float(switch_gate[0, 0]),
            seq_len=self.seq_len,
        )


def runtime_from_best_release(repo_root: str | Path) -> Stage1OnnxRuntime:
    root = Path(repo_root)
    return Stage1OnnxRuntime(
        onnx_path=root / "outputs" / "stage1_best_release" / "dual_state_best.onnx",
        meta_json=root / "data" / "run01" / "stage1_clean_meta.json",
    )
