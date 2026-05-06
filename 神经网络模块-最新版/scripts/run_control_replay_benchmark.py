from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[2]
ASCII_ROOT = WORKSPACE_ROOT / "cpp_control_ascii"
TRACKER_EXE = ASCII_ROOT / "build" / "nmake-msvc-onnx-release" / "tracker.exe"
BASE_CONFIG = ASCII_ROOT / "outputs" / "stage1_online_smoke" / "tracker_replay_smoke.yaml"
STAGE1_MODEL = ASCII_ROOT / "outputs" / "stage1_runtime_assets" / "dual_state_best.onnx"
STAGE1_META = ASCII_ROOT / "outputs" / "stage1_runtime_assets" / "stage1_clean_meta.json"
PY310 = Path(r"C:\Users\Administrator\miniconda3\envs\py310")
PYTHON_EXE = PY310 / "python.exe"
EVAL_SCRIPT = ROOT / "scripts" / "eval_control_metrics.py"


def _tracker_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = ";".join(
        [
            str(PY310),
            str(PY310 / "Library" / "bin"),
            str(PY310 / "DLLs"),
            str(PY310 / "Scripts"),
            str(TRACKER_EXE.parent),
            env.get("PATH", ""),
        ]
    )
    env["QT_PLUGIN_PATH"] = str(PY310 / "Library" / "lib" / "qt6" / "plugins")
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(PY310 / "Library" / "lib" / "qt6" / "plugins" / "platforms")
    env["STAGE1_MODEL_ONNX"] = str(STAGE1_MODEL)
    env["STAGE1_META_JSON"] = str(STAGE1_META)
    return env


def _load_base_config() -> Dict[str, object]:
    return yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))


def _write_config(cfg: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _run_tracker(config_path: Path, replay_csv: Path, trace_path: Path) -> None:
    env = _tracker_env()
    env["TRACKER_STARTUP_TRACE"] = str(trace_path)
    cmd = [str(TRACKER_EXE), "--mode", "replay", "--config", str(config_path), "--replay_csv", str(replay_csv)]
    proc = subprocess.run(cmd, cwd=str(ASCII_ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "tracker replay failed\n"
            f"config={config_path}\n"
            f"stdout=\n{proc.stdout}\n"
            f"stderr=\n{proc.stderr}"
        )


def _run_eval(log_csv: Path, output_json: Path, label: str) -> Dict[str, object]:
    cmd = [
        str(PYTHON_EXE),
        str(EVAL_SCRIPT),
        "--input_csv",
        str(log_csv),
        "--output_json",
        str(output_json),
        "--label",
        label,
    ]
    proc = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"eval_control_metrics failed for {log_csv}\nstdout=\n{proc.stdout}\nstderr=\n{proc.stderr}")
    return json.loads(output_json.read_text(encoding="utf-8"))


def _method_defs(out_root: Path) -> List[Dict[str, object]]:
    return [
        {
            "name": "direct_control",
            "filter.neural_enable": False,
            "residual.enable": False,
            "config_path": out_root / "configs" / "tracker_replay_direct_control.yaml",
            "log_csv": out_root / "logs" / "direct_control" / "tracker_log.csv",
            "trace_txt": out_root / "logs" / "direct_control" / "startup_trace.txt",
            "metrics_json": out_root / "metrics" / "direct_control.json",
        },
        {
            "name": "stage1_clean_control",
            "filter.neural_enable": True,
            "residual.enable": False,
            "config_path": out_root / "configs" / "tracker_replay_stage1_clean_control.yaml",
            "log_csv": out_root / "logs" / "stage1_clean_control" / "tracker_log.csv",
            "trace_txt": out_root / "logs" / "stage1_clean_control" / "startup_trace.txt",
            "metrics_json": out_root / "metrics" / "stage1_clean_control.json",
        },
    ]


def _set_nested(cfg: Dict[str, object], dotted_key: str, value: object) -> None:
    keys = dotted_key.split(".")
    ref = cfg
    for key in keys[:-1]:
        ref = ref.setdefault(key, {})
    ref[keys[-1]] = value


def main() -> None:
    ap = argparse.ArgumentParser(description="Run replay control benchmark for direct control vs stage1 clean control")
    ap.add_argument("--replay_csv", default=str(ASCII_ROOT / "data" / "detections.csv"))
    ap.add_argument("--output_dir", default=str(ASCII_ROOT / "outputs" / "control_benchmark_replay"))
    args = ap.parse_args()

    replay_csv = Path(args.replay_csv)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not TRACKER_EXE.exists():
        raise FileNotFoundError(f"tracker.exe not found: {TRACKER_EXE}")
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"base replay config not found: {BASE_CONFIG}")
    if not replay_csv.exists():
        raise FileNotFoundError(f"replay csv not found: {replay_csv}")
    if not STAGE1_MODEL.exists() or not STAGE1_META.exists():
        raise FileNotFoundError("stage1 runtime assets missing")

    summary: Dict[str, object] = {
        "replay_csv": str(replay_csv),
        "methods": [],
    }

    for method in _method_defs(out_root):
        cfg = _load_base_config()
        _set_nested(cfg, "camera.show_window", False)
        _set_nested(cfg, "camera.auto_focus_enable", False)
        _set_nested(cfg, "camera.auto_zoom_enable", False)
        _set_nested(cfg, "log.enable", True)
        _set_nested(cfg, "log.path", str(method["log_csv"]).replace("\\", "/"))
        _set_nested(cfg, "log.profile", "brief")
        _set_nested(cfg, "log.dedup_by_frame_id", True)
        _set_nested(cfg, "filter.neural_enable", method["filter.neural_enable"])
        _set_nested(cfg, "residual.enable", method["residual.enable"])

        _write_config(cfg, method["config_path"])
        method["log_csv"].parent.mkdir(parents=True, exist_ok=True)
        method["trace_txt"].parent.mkdir(parents=True, exist_ok=True)

        if method["log_csv"].exists():
            method["log_csv"].unlink()
        if method["trace_txt"].exists():
            method["trace_txt"].unlink()

        _run_tracker(config_path=method["config_path"], replay_csv=replay_csv, trace_path=method["trace_txt"])
        if not method["log_csv"].exists():
            raise FileNotFoundError(f"expected tracker log missing: {method['log_csv']}")

        metrics = _run_eval(log_csv=method["log_csv"], output_json=method["metrics_json"], label=method["name"])
        summary["methods"].append(
            {
                "name": method["name"],
                "config_path": str(method["config_path"]),
                "log_csv": str(method["log_csv"]),
                "trace_txt": str(method["trace_txt"]),
                "metrics_json": str(method["metrics_json"]),
                "tracking_retention_rate": metrics["tracking_retention_rate"],
                "post_control_residual_mean_px": metrics["post_control_residual"]["mean_px"],
                "post_control_residual_p95_px": metrics["post_control_residual"]["p95_px"],
                "settling_time_mean_ms": metrics["settling_time"]["mean_ms"],
                "overshoot_ratio_mean": metrics["overshoot"]["ratio_mean"],
                "lost_recovery_time_mean_ms": metrics["lost_recovery_time"]["mean_ms"],
                "lost_recovery_success_rate": metrics["lost_recovery_time"]["success_rate"],
            }
        )

    summary_path = out_root / "control_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
