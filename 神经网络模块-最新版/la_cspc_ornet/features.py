from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List

STAGE1_REQUIRED_FEATURE_COLUMNS: List[str] = [
    "dx_raw",
    "dy_raw",
    "d1_dx",
    "d1_dy",
    "trend_dx",
    "trend_dy",
    "det_conf",
    "lost_flag",
    "is_meas_update",
    "meas_age_ms",
    "zoom_delta",
    "switch_score",
    "turn_score",
]

STAGE1_OPTIONAL_FEATURE_COLUMNS: List[str] = [
    "d2_dx",
    "d2_dy",
    "trend_vx",
    "trend_vy",
    "bbox_area_norm",
    "dt_ms",
    "coast_count",
]

# Single source of truth for runtime feature order.
STAGE1_FEATURE_COLUMNS: List[str] = [
    "dx_raw",
    "dy_raw",
    "d1_dx",
    "d1_dy",
    "d2_dx",
    "d2_dy",
    "trend_dx",
    "trend_dy",
    "trend_vx",
    "trend_vy",
    "det_conf",
    "bbox_area_norm",
    "lost_flag",
    "is_meas_update",
    "meas_age_ms",
    "zoom_delta",
    "dt_ms",
    "switch_score",
    "turn_score",
    "coast_count",
]
ALL_FEATURE_COLUMNS: List[str] = STAGE1_FEATURE_COLUMNS

REQUIRED_LOG_COLUMNS_STAGE1: List[str] = [
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

FEATURE_NAME_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(STAGE1_FEATURE_COLUMNS)}
FEATURE_INDEX = SimpleNamespace(**FEATURE_NAME_TO_INDEX)


def check_feature_index_alignment() -> bool:
    return all(getattr(FEATURE_INDEX, name) == idx for idx, name in enumerate(STAGE1_FEATURE_COLUMNS))


def feature_spec_dict() -> Dict[str, List[str]]:
    return {
        "stage1_required_features": STAGE1_REQUIRED_FEATURE_COLUMNS,
        "stage1_optional_features": STAGE1_OPTIONAL_FEATURE_COLUMNS,
        "stage1_features": STAGE1_FEATURE_COLUMNS,
        "all_features": ALL_FEATURE_COLUMNS,
        "required_log_columns_stage1": REQUIRED_LOG_COLUMNS_STAGE1,
    }
