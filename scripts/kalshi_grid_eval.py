#!/usr/bin/env python3
# Created by Oliver Meihls

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

from argus_kalshi.offline_sim import (
    build_param_grid,
    evaluate_grid,
    rank_results,
    save_ranked_results,
)
from argus_kalshi.settlement_index import SettlementIndex


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a deterministic Kalshi parameter grid.")
    ap.add_argument("--tape", action="append", default=[], help="Decision tape JSONL path (repeatable)")
    ap.add_argument("--settlement", action="append", default=[], help="Settlement JSONL glob/path (repeatable)")
    ap.add_argument("--output-dir", default="logs/grid_results")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--build-index-only", action="store_true")
    args = ap.parse_args()

    settlement_paths = sorted({path for pattern in args.settlement for path in glob(pattern)})
    settlement_index = SettlementIndex.from_jsonl_files(settlement_paths)
    if args.build_index_only:
        print(json.dumps(settlement_index.coverage_stats(), indent=2))
        return 0
    if not args.tape:
        raise SystemExit("Provide at least one --tape path.")
    param_grid = build_param_grid()
    results = evaluate_grid(
        tape_paths=args.tape,
        settlement_index=settlement_index,
        param_grid=param_grid,
        n_workers=args.workers,
        scenario=args.scenario,
    )
    ranked = rank_results(results)
    json_path, csv_path = save_ranked_results(ranked, args.output_dir)
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")
    for row in ranked[: max(0, args.top_n)]:
        params = row["params"]
        print(
            f"rank={row['rank']:>3} pnl={row['total_pnl_usd']:+8.3f} score={row['robustness_score']:>6.2f} "
            f"fills={row['fills_count']:>4} min_entry={params['min_entry_cents']} max_entry={params['max_entry_cents']} "
            f"edge={params['min_edge_threshold']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
