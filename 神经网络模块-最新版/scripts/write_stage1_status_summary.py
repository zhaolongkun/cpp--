from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.features import check_feature_index_alignment


def _load_json(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a consolidated stage1 status summary")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    root = ROOT
    outputs = root / "outputs"
    data_dir = root / "data" / "run01"

    obsolete = _load_json(outputs / "obsolete_stage1_artifacts.json")
    tag_summary = _load_json(outputs / "stage1_tag_validation_summary.json")
    dataset_report = _load_json(outputs / "stage1_dataset_report.json")
    baseline_summary = _load_json(outputs / "stage1_baseline_summary.json")
    tcn_eval = _load_json(outputs / "stage1_tcn_gru" / "eval_summary.json")
    dual_eval = _load_json(outputs / "stage1_clean" / "eval_summary.json")
    visuals_summary = _load_json(outputs / "stage1_visuals" / "summary.json")

    payload = {
        "phase": "stage1_revalidation",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_alignment_ok": check_feature_index_alignment(),
        "obsolete_artifacts_manifest_exists": obsolete is not None,
        "raw_tag_validation_summary_exists": tag_summary is not None,
        "ready_for_merge": bool(tag_summary.get("ready_for_merge", False)) if tag_summary else False,
        "required_stage1_types": tag_summary.get("required_stage1_types", []) if tag_summary else [],
        "usable_counts_by_type": tag_summary.get("usable_counts_by_type", {}) if tag_summary else {},
        "decision_counts": tag_summary.get("decision_counts", {}) if tag_summary else {},
        "dataset_exists": (data_dir / "stage1_clean_dataset.npz").exists(),
        "meta_exists": (data_dir / "stage1_clean_meta.json").exists(),
        "dataset_report_exists": dataset_report is not None,
        "baseline_summary_exists": baseline_summary is not None,
        "tcn_eval_exists": tcn_eval is not None,
        "dual_eval_exists": dual_eval is not None,
        "visuals_summary_exists": visuals_summary is not None,
        "blocked_by": None,
    }

    if not payload["feature_alignment_ok"]:
        payload["blocked_by"] = "feature_alignment_failed"
    elif not payload["obsolete_artifacts_manifest_exists"]:
        payload["blocked_by"] = "obsolete_manifest_missing"
    elif not payload["raw_tag_validation_summary_exists"]:
        payload["blocked_by"] = "tag_validation_summary_missing"
    elif not payload["ready_for_merge"]:
        payload["blocked_by"] = "ready_for_merge_false"

    out = Path(args.output_json)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
