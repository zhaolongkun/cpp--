import argparse
import json
from typing import Dict


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct_improve(base: float, new: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return (base - new) / abs(base) * 100.0


def main() -> None:
    p = argparse.ArgumentParser("Compare two metric json files")
    p.add_argument("--base", required=True)
    p.add_argument("--new", required=True)
    args = p.parse_args()

    b = load_json(args.base)
    n = load_json(args.new)

    keys_smaller_better = [
        "rmse_x_px",
        "rmse_y_px",
        "mae_x_px",
        "mae_y_px",
        "sign_flip_x_hz",
        "sign_flip_y_hz",
        "cmd_jitter_x",
        "cmd_jitter_y",
        "mean_gate_d2",
        "mean_outlier_prob",
    ]
    keys_larger_better = [
        "tracked_ratio",
        "valid_track_ratio",
    ]

    report = {"base": args.base, "new": args.new, "delta": {}}

    for k in keys_smaller_better:
        if k in b and k in n:
            report["delta"][k] = {
                "base": b[k],
                "new": n[k],
                "improve_percent": pct_improve(float(b[k]), float(n[k])),
            }

    for k in keys_larger_better:
        if k in b and k in n:
            report["delta"][k] = {
                "base": b[k],
                "new": n[k],
                "improve_percent": pct_improve(float(n[k]), float(b[k])),
            }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

