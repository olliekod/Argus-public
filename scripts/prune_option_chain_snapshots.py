#!/usr/bin/env python3
"""Prune old rows from option_chain_snapshots by timestamp_ms.

Used for replay packs (SPY, QQQ, IBIT, BITO, etc.). Retention is also applied
hourly by the orchestrator when data_retention.option_chain_snapshots_days is set
in config.yaml.

Usage:
  python scripts/prune_option_chain_snapshots.py --days 30
  python scripts/prune_option_chain_snapshots.py --days 30 --db data/argus.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune option_chain_snapshots older than N days (by timestamp_ms).",
    )
    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Delete rows with timestamp_ms older than this many days",
    )
    parser.add_argument(
        "--db",
        default="data/argus.db",
        help="SQLite database path",
    )
    args = parser.parse_args()

    if args.days <= 0:
        print("--days must be > 0", file=sys.stderr)
        return 1

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"SKIP: database not found ({db_path})")
        return 0

    cutoff_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000
    )

    conn = sqlite3.connect(db_path)
    try:
        before = conn.execute(
            "SELECT COUNT(*) FROM option_chain_snapshots"
        ).fetchone()[0]
        conn.execute(
            "DELETE FROM option_chain_snapshots WHERE timestamp_ms < ?",
            (cutoff_ms,),
        )
        conn.commit()
        after = conn.execute(
            "SELECT COUNT(*) FROM option_chain_snapshots"
        ).fetchone()[0]
    except sqlite3.OperationalError as exc:
        print(f"SKIP: option_chain_snapshots unavailable ({exc})")
        return 0
    finally:
        conn.close()

    print(f"Pruned {before - after} option_chain_snapshots rows older than {args.days} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
