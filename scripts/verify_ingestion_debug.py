"""Verify option chain snapshot ingestion from policy providers.

Uses data_sources from config: options_snapshots_primary (Tastytrade) and
options_snapshots_secondary (Public when public_options.enabled). Does not
assume Alpaca options (Alpaca is bars-only in the modern setup).

Shows recent snapshots grouped by provider, stats per symbol, and validates:
- Fresh option_chain_snapshots for target symbols (from config)
- Primary provider present; when secondary enabled, both present for at least one symbol
- timestamp_ms minute-aligned (floored)
- recv_ts_ms now-ish (within tolerance)

Usage:
  python scripts/verify_ingestion_debug.py
  python scripts/verify_ingestion_debug.py --limit 50
  python scripts/verify_ingestion_debug.py --validate   # exit 0 if all checks pass
"""

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone

# Default symbols when config unavailable
DEFAULT_TARGET_SYMBOLS = ("SPY", "QQQ", "IBIT")
FRESH_WINDOW_MS = 10 * 60 * 1000   # 10 min
POLLING_FRESH_MS = 5 * 60 * 1000   # 5 min to infer "task started"
RECV_TOLERANCE_SEC = 120            # recv_ts_ms within this many seconds of "now"


def _get_target_symbols_and_providers():
    """Read target symbols and options providers from config/data_sources."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.core.config import load_config
        from src.core.data_sources import get_data_source_policy
        cfg = load_config()
        policy = get_data_source_policy(cfg)
        providers = policy.snapshot_providers(include_secondary=True)
        public_enabled = (cfg.get("public_options") or {}).get("enabled", False)
        if not public_enabled and "public" in providers:
            providers = [p for p in providers if p != "public"]
        # Target symbols: union of tastytrade underlyings and public_options.symbols
        tt = (cfg.get("tastytrade") or {}).get("underlyings", [])
        pub = (cfg.get("public_options") or {}).get("symbols", [])
        symbols = tuple(sorted(set(tt) | set(pub)))[:10] or DEFAULT_TARGET_SYMBOLS
        return symbols, providers
    except Exception:
        return DEFAULT_TARGET_SYMBOLS, ("tastytrade", "public")


def _run_validation(cursor, target_symbols: tuple, expected_providers: list) -> tuple[bool, list[str]]:
    """Run validation checks. Returns (all_passed, list of failure messages)."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    failures = []
    placeholders = ",".join("?" for _ in target_symbols)

    # Fresh rows for target symbols
    cursor.execute(
        f"""
        SELECT DISTINCT symbol FROM option_chain_snapshots
        WHERE symbol IN ({placeholders}) AND timestamp_ms >= ?
        """,
        (*target_symbols, now_ms - FRESH_WINDOW_MS),
    )
    seen = {r[0] for r in cursor.fetchall()}
    for sym in target_symbols:
        if sym not in seen:
            failures.append(f"No fresh rows for {sym} (last {FRESH_WINDOW_MS // 60000} min)")

    # Primary (and when enabled secondary) present for at least one symbol
    cursor.execute(
        f"""
        SELECT symbol, GROUP_CONCAT(DISTINCT provider) as providers
        FROM option_chain_snapshots
        WHERE symbol IN ({placeholders}) AND timestamp_ms >= ?
        GROUP BY symbol
        """,
        (*target_symbols, now_ms - FRESH_WINDOW_MS),
    )
    prov_set_seen = set()
    for symbol, providers in cursor.fetchall():
        if not providers:
            continue
        prov_set = set(p.strip().lower() for p in (providers or "").split(",") if p.strip())
        prov_set_seen |= prov_set
    for exp in expected_providers:
        if exp.lower() not in prov_set_seen:
            failures.append(f"Provider '{exp}' not present for any of {target_symbols} in window")

    # timestamp_ms minute-aligned for recent rows
    cursor.execute(
        f"""
        SELECT timestamp_ms FROM option_chain_snapshots
        WHERE symbol IN ({placeholders}) AND timestamp_ms >= ?
        LIMIT 500
        """,
        (*target_symbols, now_ms - FRESH_WINDOW_MS),
    )
    rows = cursor.fetchall()
    misaligned = [r[0] for r in rows if r[0] is not None and (r[0] % 60_000) != 0]
    if misaligned:
        failures.append(f"timestamp_ms not minute-aligned: {len(misaligned)} recent rows (e.g. {misaligned[0]})")

    # recv_ts_ms now-ish for most recent row
    cursor.execute(
        f"""
        SELECT recv_ts_ms FROM option_chain_snapshots
        WHERE symbol IN ({placeholders})
        ORDER BY timestamp_ms DESC LIMIT 1
        """,
        target_symbols,
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
        recv_sec_ago = (now_ms - row[0]) / 1000
        if abs(recv_sec_ago) > RECV_TOLERANCE_SEC:
            failures.append(f"recv_ts_ms not now-ish: most recent {abs(recv_sec_ago):.0f}s from now (tolerance {RECV_TOLERANCE_SEC}s)")
    else:
        failures.append("No recv_ts_ms found on recent snapshots")

    return (len(failures) == 0, failures)


async def verify_ingestion(limit: int = 30, validate_only: bool = False):
    target_symbols, expected_providers = _get_target_symbols_and_providers()
    db_path = Path("data/argus.db")
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        if validate_only:
            sys.exit(2)
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if validate_only:
        ok, failures = _run_validation(cursor, target_symbols, expected_providers)
        conn.close()
        if ok:
            print("Validation PASSED: fresh symbols, required providers, minute-aligned timestamp_ms, now-ish recv_ts_ms")
            sys.exit(0)
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(1)
    # --------

    print("=" * 90)
    print("Option Chain Snapshot Ingestion Report (policy: primary + secondary)")
    print("=" * 90)
    print(f"  Target symbols: {target_symbols}")
    print(f"  Expected providers: {expected_providers}")

    # ── Polling health (inferred from recent data) ─────────────────────
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cutoff = now_ms - POLLING_FRESH_MS
    placeholders = ",".join("?" for _ in target_symbols)
    cursor.execute(
        f"""
        SELECT provider, MAX(timestamp_ms) as last_ts
        FROM option_chain_snapshots
        WHERE symbol IN ({placeholders})
        GROUP BY provider
        """,
        target_symbols,
    )
    provider_last = {r[0]: r[1] for r in cursor.fetchall()}
    print("\n[0] Polling health (inferred from DB)")
    print("-" * 70)
    for prov in expected_providers:
        last = provider_last.get(prov)
        if last is None:
            print(f"  {prov:<15} | no rows in DB (task may not be running or no data yet)")
        elif last >= cutoff:
            print(f"  {prov:<15} | OK (last snapshot within last {POLLING_FRESH_MS // 60000} min)")
        else:
            age_sec = (now_ms - last) / 1000
            print(f"  {prov:<15} | STALE (last snapshot {age_sec:.0f}s ago)")

    # ── Provider summary ──────────────────────────────────────────────
    print("\n[1] Provider Summary")
    print("-" * 60)
    summary_query = """
    SELECT provider, COUNT(*) as cnt,
           MIN(timestamp_ms) as first_ts,
           MAX(timestamp_ms) as last_ts
    FROM option_chain_snapshots
    GROUP BY provider
    ORDER BY provider;
    """
    cursor.execute(summary_query)
    rows = cursor.fetchall()

    if not rows:
        print("  No snapshots found in database.")
        conn.close()
        return

    for provider, cnt, first_ts, last_ts in rows:
        first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"  {provider or '(empty)':<15} | {cnt:>6} snapshots | first: {first_dt} | last: {last_dt}")

    # ── Per-symbol breakdown ──────────────────────────────────────────
    print(f"\n[2] Per-Symbol Breakdown")
    print("-" * 70)
    ph = ",".join(repr(s) for s in target_symbols)
    symbol_query = f"""
    SELECT provider, symbol, COUNT(*) as cnt,
           MAX(timestamp_ms) as last_ts,
           AVG(n_strikes) as avg_strikes
    FROM option_chain_snapshots
    WHERE symbol IN ({ph})
    GROUP BY provider, symbol
    ORDER BY symbol, provider;
    """
    cursor.execute(symbol_query)
    rows = cursor.fetchall()

    if rows:
        print(f"  {'Provider':<15} | {'Symbol':<8} | {'Count':>6} | {'Avg Strikes':>11} | {'Last Snapshot':<20}")
        print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*6}-+-{'-'*11}-+-{'-'*20}")
        for provider, symbol, cnt, last_ts, avg_strikes in rows:
            last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"  {provider or '(empty)':<15} | {symbol:<8} | {cnt:>6} | {avg_strikes:>11.1f} | {last_dt}")
    else:
        print(f"  No snapshots for {target_symbols}.")

    # ── Recent snapshots ──────────────────────────────────────────────
    print(f"\n[3] Recent Snapshots (last {limit})")
    print("-" * 100)
    ph2 = ",".join(repr(s) for s in target_symbols)
    recent_query = f"""
    SELECT provider, symbol, timestamp_ms, recv_ts_ms, n_strikes, underlying_price, atm_iv
    FROM option_chain_snapshots
    WHERE symbol IN ({ph2})
    ORDER BY timestamp_ms DESC
    LIMIT {limit};
    """
    cursor.execute(recent_query)
    rows = cursor.fetchall()

    if rows:
        print(f"  {'Provider':<12} | {'Symbol':<6} | {'Timestamp (UTC)':<20} | {'Lag(ms)':>8} | {'Strikes':>7} | {'Price':>9} | {'ATM IV':>8}")
        print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*20}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}")
        for provider, symbol, ts_ms, recv_ms, n_strikes, price, atm_iv in rows:
            ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            lag = (recv_ms - ts_ms) if recv_ms and ts_ms else 0
            price_str = f"${price:.2f}" if price else "N/A"
            iv_str = f"{atm_iv:.4f}" if atm_iv else "N/A"
            print(f"  {provider or '?':<12} | {symbol:<6} | {ts_dt:<20} | {lag:>8} | {n_strikes:>7} | {price_str:>9} | {iv_str:>8}")
    else:
        print("  No recent snapshots found.")

    # ── Multi-provider overlap check ──────────────────────────────────
    print(f"\n[4] Multi-Provider Overlap (same symbol, same minute)")
    print("-" * 60)
    ph3 = ",".join(repr(s) for s in target_symbols)
    overlap_query = f"""
    SELECT symbol,
           datetime(timestamp_ms/1000, 'unixepoch') as ts_min,
           GROUP_CONCAT(DISTINCT provider) as providers,
           COUNT(*) as cnt
    FROM option_chain_snapshots
    WHERE symbol IN ({ph3})
    GROUP BY symbol, timestamp_ms / 60000
    HAVING COUNT(DISTINCT provider) > 1
    ORDER BY timestamp_ms DESC
    LIMIT 10;
    """
    cursor.execute(overlap_query)
    rows = cursor.fetchall()

    if rows:
        print(f"  {'Symbol':<8} | {'Minute':<20} | {'Providers':<30} | {'Count':>5}")
        print(f"  {'-'*8}-+-{'-'*20}-+-{'-'*30}-+-{'-'*5}")
        for symbol, ts_min, providers, cnt in rows:
            print(f"  {symbol:<8} | {ts_min:<20} | {providers:<30} | {cnt:>5}")
    else:
        print("  No multi-provider overlaps found (this is normal if only one provider is active).")

    # ── Validation summary ────────────────────────────────────────────
    print(f"\n[5] Validation (fresh symbols, required providers, minute-aligned timestamp_ms, now-ish recv_ts_ms)")
    print("-" * 70)
    ok, failures = _run_validation(cursor, target_symbols, expected_providers)
    if ok:
        print("  All checks PASSED.")
    else:
        for f in failures:
            print(f"  FAIL: {f}")
    print("  (Use --validate to run only these checks and exit 0/1)")

    conn.close()
    print(f"\n{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(description="Verify option chain snapshot ingestion")
    parser.add_argument("--limit", type=int, default=30, help="Number of recent snapshots to show")
    parser.add_argument("--validate", action="store_true", help="Run validation only; exit 0 if pass, 1 if fail")
    args = parser.parse_args()
    asyncio.run(verify_ingestion(limit=args.limit, validate_only=args.validate))


if __name__ == "__main__":
    main()
