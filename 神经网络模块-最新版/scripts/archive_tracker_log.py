from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

VALID_PREFIXES = {"stable", "jitter", "zoom", "recover", "turn"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Archive current tracker_log.csv into raw_logs")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst_dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    src = Path(args.src)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not args.tag:
        raise ValueError("--tag is required for stage1 archive")
    prefix = args.tag.split("_", 1)[0].lower()
    if prefix not in VALID_PREFIXES:
        raise ValueError(f"unsupported stage1 segment tag: {args.tag}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    dst = dst_dir / f"tracker_log{tag}_{stamp}.csv"
    shutil.copy2(src, dst)
    print(dst)


if __name__ == "__main__":
    main()
