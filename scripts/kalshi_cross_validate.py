#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from glob import glob

from argus_kalshi.offline_sim import build_param_grid, cross_validate, summarize_cross_validation
from argus_kalshi.settlement_index import SettlementIndex


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-validate deterministic Kalshi backtest params.")
    ap.add_argument("--tape-dir", required=True)
    ap.add_argument("--settlement", action="append", default=[], help="Settlement JSONL glob/path (repeatable)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--scenario", default="base")
    args = ap.parse_args()

    tape_paths = sorted(glob(f"{args.tape_dir}/**/*.jsonl", recursive=True))
    settlement_paths = sorted({path for pattern in args.settlement for path in glob(pattern)})
    settlement_index = SettlementIndex.from_jsonl_files(settlement_paths)
    folds = cross_validate(
        tape_paths=tape_paths,
        settlement_index=settlement_index,
        param_grid=build_param_grid(),
        n_workers=args.workers,
        scenario=args.scenario,
    )
    print(json.dumps(summarize_cross_validation(folds), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
