from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from la_cspc_ornet.features import REQUIRED_LOG_COLUMNS_STAGE1


def main() -> None:
    ap = argparse.ArgumentParser(description="Check whether tracker_log.csv has enough fields for LA-CSPC-ORNet")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, nrows=5)
    cols = set(df.columns)
    missing_stage1 = [c for c in REQUIRED_LOG_COLUMNS_STAGE1 if c not in cols]
    print(f"csv={Path(args.csv)}")
    print(f"column_count={len(cols)}")
    print(f"missing_stage1={missing_stage1}")


if __name__ == "__main__":
    main()
