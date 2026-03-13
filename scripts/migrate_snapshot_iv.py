"""One-time migration: re-derive IV from quotes_json for snapshots with missing atm_iv.

For each option_chain_snapshot where atm_iv IS NULL or atm_iv <= 0, attempts to derive
IV from quotes_json (put-level iv, bid/ask, GreeksEngine). Updates the DB when derived
IV is valid.

Usage:
  python scripts/migrate_snapshot_iv.py [--db PATH] [--symbol SYM] [--dry-run] [--verbose]

  --db       Path to argus.db (default: data/argus.db)
  --symbol   Migrate only this symbol (default: all)
  --dry-run  Don't write; print what would be updated
  --verbose  Log derivation failures (useful to diagnose why derivation fails)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.core.database import Database
from src.tools.replay_pack import _atm_iv_from_quotes_json


async def run(db_path: str, symbol: str | None, dry_run: bool, verbose: bool) -> int:
    db = Database(db_path)
    await db.connect()

    # Fetch snapshots with missing or zero atm_iv
    cursor = await db._connection.execute(
        """SELECT snapshot_id, symbol, provider, timestamp_ms, underlying_price, quotes_json, atm_iv
           FROM option_chain_snapshots
           WHERE (atm_iv IS NULL OR atm_iv <= 0)
           AND quotes_json IS NOT NULL AND quotes_json != ''
           """ + (" AND symbol = ?" if symbol else ""),
        (symbol,) if symbol else (),
    )
    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    snapshots = [dict(zip(cols, row)) for row in rows]

    total = len(snapshots)
    updated = 0
    failed_reasons: dict[str, int] = {}

    for s in snapshots:
        sid = s["snapshot_id"]
        qj = s.get("quotes_json") or ""
        underlying = float(s.get("underlying_price") or 0)

        derived = _atm_iv_from_quotes_json(qj, underlying)

        if derived is not None and derived > 0:
            if not dry_run:
                await db._connection.execute(
                    "UPDATE option_chain_snapshots SET atm_iv = ? WHERE snapshot_id = ?",
                    (derived, sid),
                )
            updated += 1
        else:
            # Collect failure reason for summary (import here to avoid circular deps)
            reason = _diagnose_failure(qj, underlying)
            failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
            if verbose:
                print(f"  skip {sid}: {reason} (underlying={underlying}, len_quotes={len(qj)})")

    if not dry_run and updated > 0:
        await db._connection.commit()

    await db.close()

    print(f"Migration: {total} snapshots with missing atm_iv")
    print(f"  Updated: {updated}")
    if failed_reasons:
        print(f"  Failed (reasons):")
        for reason, count in sorted(failed_reasons.items(), key=lambda x: -x[1]):
            print(f"    {count}: {reason}")
    if dry_run and updated > 0:
        print("  (dry-run: no changes written)")

    return 0 if total == 0 or updated > 0 else 1


def _diagnose_failure(quotes_json: str, underlying: float) -> str:
    """Return a short reason why derivation failed (for diagnostics)."""
    import json
    if not quotes_json or not quotes_json.strip():
        return "empty quotes_json"
    try:
        data = json.loads(quotes_json)
    except Exception as e:
        return f"invalid json: {e}"
    if not isinstance(data, dict):
        return "quotes_json not dict"
    underlying_resolved = float(underlying or 0)
    if underlying_resolved <= 0:
        raw = data.get("underlying_price")
        if raw is not None:
            try:
                underlying_resolved = float(raw)
            except (TypeError, ValueError):
                pass
    if underlying_resolved <= 0:
        return "underlying_price missing or 0"
    puts = data.get("puts") or []
    if not puts:
        return "no puts in chain"
    # Check if any put has usable data (safe float)
    def _safe_float(v):
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    n_with_iv = sum(1 for p in puts if isinstance(p, dict) and _safe_float(p.get("iv")) > 0)
    n_with_bid_ask = sum(1 for p in puts if isinstance(p, dict) and _safe_float(p.get("bid")) > 0 and _safe_float(p.get("ask")) > 0)
    n_with_mid = sum(1 for p in puts if isinstance(p, dict) and _safe_float(p.get("mid")) > 0)
    n_with_last = sum(1 for p in puts if isinstance(p, dict) and _safe_float(p.get("last")) > 0)
    if n_with_iv == 0 and n_with_bid_ask == 0 and n_with_mid == 0 and n_with_last == 0:
        return f"no put has iv/bid_ask/mid/last (puts={len(puts)})"
    return "GreeksEngine or ATM selection failed"


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate snapshots: derive IV from quotes_json")
    ap.add_argument("--db", default="data/argus.db", help="Path to argus.db")
    ap.add_argument("--symbol", default=None, help="Migrate only this symbol")
    ap.add_argument("--dry-run", action="store_true", help="Don't write changes")
    ap.add_argument("--verbose", "-v", action="store_true", help="Log each failure + derivation debug")
    args = ap.parse_args()
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("argus.replay_pack").setLevel(logging.DEBUG)
    return asyncio.run(run(args.db, args.symbol, args.dry_run, args.verbose))


if __name__ == "__main__":
    sys.exit(main())
