#!/usr/bin/env python3
"""Run the full Polymarket wallet pipeline for the fixed watchlist.

Pipeline:
1) Ingest all watched wallets
2) Analyze each wallet
3) Compare wallets
4) Generate Kalshi hypotheses for each wallet analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import List


WATCH_WALLETS: List[str] = [
    "0x63ce342161250d705dc0b16df89036c8e5f9ba9a",
    "0xd0d6053c3c37e727402d84c14069780d360993aa",
    "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d",
    "0x1979ae6b7e6534de9c4539d0c205e582ca637c9d",
]


def _run(cmd: List[str]) -> None:
    print(f"[watch] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _latest_wallet_analysis_json(out_root: str, wallet: str) -> str:
    wallet_key = wallet.lower()
    pattern = os.path.join(out_root, "*", f"wallet_{wallet_key}_analysis.json")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No analysis json found for wallet={wallet}")
    return max(candidates, key=os.path.getmtime)


def _latest_compare_json(out_root: str) -> str:
    pattern = os.path.join(out_root, "*", "wallet_compare.json")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError("No wallet_compare.json found")
    return max(candidates, key=os.path.getmtime)

def _latest_consensus_json(out_root: str) -> str:
    pattern = os.path.join(out_root, "*", "wallet_consensus.json")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError("No wallet_consensus.json found")
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ingest/analyze/compare/hypothesis for fixed Polymarket wallets."
    )
    parser.add_argument("--db", default="data/polymarket_wallets.db", help="SQLite DB path")
    parser.add_argument(
        "--out",
        default=os.path.join("logs", "analysis", "polymarket"),
        help="Output root used by analyze/compare/hypotheses",
    )
    parser.add_argument("--hours", type=int, default=0, help="Analyze lookback hours (0=all)")
    parser.add_argument("--max-pages", type=int, default=100, help="Ingest max pages per endpoint")
    parser.add_argument("--page-size", type=int, default=50, help="Ingest rows per page")
    parser.add_argument("--sleep-ms", type=int, default=100, help="Ingest sleep between pages (ms)")
    args = parser.parse_args()

    py = sys.executable

    ingest_cmd = [
        py,
        os.path.join("scripts", "polymarket_wallet_ingest.py"),
        "--db",
        args.db,
        "--max-pages",
        str(args.max_pages),
        "--page-size",
        str(args.page_size),
        "--sleep-ms",
        str(args.sleep_ms),
    ]
    for wallet in WATCH_WALLETS:
        ingest_cmd.extend(["--wallet", wallet])
    _run(ingest_cmd)

    analysis_json_paths: List[str] = []
    for wallet in WATCH_WALLETS:
        _run(
            [
                py,
                os.path.join("scripts", "polymarket_wallet_analyze.py"),
                "--wallet",
                wallet,
                "--db",
                args.db,
                "--hours",
                str(args.hours),
                "--out",
                args.out,
            ]
        )
        analysis_json_paths.append(_latest_wallet_analysis_json(args.out, wallet))

    compare_cmd = [
        py,
        os.path.join("scripts", "polymarket_wallet_compare.py"),
        "--db",
        args.db,
        "--out",
        args.out,
    ]
    for wallet in WATCH_WALLETS:
        compare_cmd.extend(["--wallet", wallet])
    _run(compare_cmd)

    for analysis_json in analysis_json_paths:
        _run(
            [
                py,
                os.path.join("scripts", "polymarket_to_kalshi_hypotheses.py"),
                "--analysis",
                analysis_json,
                "--out",
                args.out,
            ]
        )

    consensus_cmd = [
        py,
        os.path.join("scripts", "polymarket_wallet_consensus.py"),
        "--out",
        args.out,
    ]
    for analysis_json in analysis_json_paths:
        consensus_cmd.extend(["--analysis", analysis_json])
    _run(consensus_cmd)

    summary = {
        "wallets": WATCH_WALLETS,
        "db": args.db,
        "out_root": args.out,
        "analysis_json": analysis_json_paths,
        "compare_json": _latest_compare_json(args.out),
        "consensus_json": _latest_consensus_json(args.out),
    }
    print("[watch] completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
