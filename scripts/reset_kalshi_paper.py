#!/usr/bin/env python3
# Created by Oliver Meihls

# Reset Kalshi paper trading performance and balances.
#
# Clears the paper-trade log and Kalshi outcome tables so the bot starts
# as if new: balance from config (bankroll_usd), zero PnL, zero wins/losses.
#
# Usage:
# python scripts/reset_kalshi_paper.py
# python scripts/reset_kalshi_paper.py --no-backup --db data/kalshi.db

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_JSONL = "logs/paper_trades.jsonl"
DEFAULT_DB = "data/kalshi.db"
DEFAULT_CONTEXT_POLICY = "config/kalshi_context_policy.json"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reset Kalshi paper performance and balances so the bot restarts like new.",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup of paper_trades.jsonl before clearing",
    )
    ap.add_argument(
        "--db",
        default=DEFAULT_DB,
        metavar="PATH",
        help=f"Path to Kalshi SQLite DB (default: {DEFAULT_DB})",
    )
    ap.add_argument(
        "--context-policy",
        default=DEFAULT_CONTEXT_POLICY,
        metavar="PATH",
        help=f"Path to context policy JSON to reset (default: {DEFAULT_CONTEXT_POLICY})",
    )
    ap.add_argument(
        "--keep-policy",
        action="store_true",
        help="Do not reset the context policy (preserve learned weights across cycles)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    log_path = root / DEFAULT_JSONL
    db_path = root / args.db
    context_policy_path = root / args.context_policy

    # 1. Back up and clear paper_trades.jsonl
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8").strip()
        if content and not args.no_backup:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
            backup_path = log_path.parent / (log_path.name + ".bak." + timestamp)
            backup_path.write_text(content, encoding="utf-8")
            print(f"Backed up to {backup_path}")
        log_path.write_text("", encoding="utf-8")
        print(f"Cleared {log_path}")
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        print(f"Created empty {log_path}")

    # 2. Clear Kalshi outcome/decision/event tables in DB
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('kalshi_outcomes','kalshi_decisions','kalshi_terminal_events')")
            tables = [row[0] for row in cur.fetchall()]
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
                conn.commit()
                print(f"Cleared table {table} in {db_path}")
        finally:
            conn.close()
    else:
        print(f"No DB at {db_path} (skipped)")

    # 3. Reset context policy file to clean baseline (skip if --keep-policy).
    if args.keep_policy:
        print(f"Kept context policy {context_policy_path} (--keep-policy)")
        print("Done. Restart the bot; balance and stats will start fresh from config.")
        return 0
    baseline_policy = {
        "version": 1,
        "timestamp": 0,
        "keys": {},
        "summary": {
            "core_count": 0,
            "explore_count": 0,
            "unknown_count": 0,
            "total_settlements": 0,
        },
    }
    context_policy_path.parent.mkdir(parents=True, exist_ok=True)
    if context_policy_path.exists() and not args.no_backup:
        existing = context_policy_path.read_text(encoding="utf-8").strip()
        if existing:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
            backup_path = context_policy_path.parent / (context_policy_path.name + ".bak." + timestamp)
            backup_path.write_text(existing, encoding="utf-8")
            print(f"Backed up context policy to {backup_path}")
    context_policy_path.write_text(
        json.dumps(baseline_policy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Reset context policy {context_policy_path}")

    print("Done. Restart the bot; balance and stats will start fresh from config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
