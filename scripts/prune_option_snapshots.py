#!/usr/bin/env python3
"""Prune old rows from option_quote_snapshots."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import sqlite3

from scripts.tastytrade_health_audit import _prune_snapshots_sql


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, required=True, help="Delete rows older than this many days")
    parser.add_argument("--db", default="data/argus.db", help="SQLite database path")
    args = parser.parse_args()

    if args.days <= 0:
        raise SystemExit("--days must be > 0")

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"SKIP: database not found ({db_path})")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        sql, params = _prune_snapshots_sql(args.days)
        before = conn.execute("SELECT COUNT(*) FROM option_quote_snapshots").fetchone()[0]
        conn.execute(sql, params)
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM option_quote_snapshots").fetchone()[0]
    except sqlite3.OperationalError as exc:
        print(f"SKIP: option_quote_snapshots unavailable ({exc})")
        return 0
    finally:
        conn.close()

    print(f"Pruned {before - after} rows older than {args.days} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
