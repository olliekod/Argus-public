#!/usr/bin/env python3
# Created by Oliver Meihls

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import yaml

from argus_kalshi.offline_sim import diff_strategies, params_from_kalshi_config
from argus_kalshi.settlement_index import SettlementIndex


def _load_argus_kalshi(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    block = data.get("argus_kalshi", {})
    if isinstance(block, list):
        return dict(block[0] if block else {})
    return dict(block or {})


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two deterministic Kalshi parameter sets.")
    ap.add_argument("--tape", action="append", default=[], help="Decision tape JSONL path (repeatable)")
    ap.add_argument("--settlement", action="append", default=[], help="Settlement JSONL glob/path (repeatable)")
    ap.add_argument("--old-config", required=True)
    ap.add_argument("--new-config", required=True)
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    settlement_paths = sorted({path for pattern in args.settlement for path in glob(pattern)})
    settlement_index = SettlementIndex.from_jsonl_files(settlement_paths)
    result = diff_strategies(
        tape_paths=args.tape,
        settlement_index=settlement_index,
        params_old=params_from_kalshi_config(_load_argus_kalshi(args.old_config)),
        params_new=params_from_kalshi_config(_load_argus_kalshi(args.new_config)),
        scenario=args.scenario,
    )
    payload = result.to_dict()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved={output_path}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
