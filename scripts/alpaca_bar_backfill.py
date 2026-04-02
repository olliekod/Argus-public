#!/usr/bin/env python3
# Created by Oliver Meihls

# Alpaca ETF Bar Backfill
#
# Fetches historical 1-minute OHLCV bars from Alpaca for equity/ETF symbols
# and persists them to the Argus database.
#
# Alpaca Data API v2: GET /v2/stocks/{symbol}/bars
# - timeframe: 1Min
# - limit: up to 10,000 bars per request
# - requires APCA-API-KEY-ID and APCA-API-SECRET-KEY headers
#
# Usage::
#
# # Dry-run: show how many bars exist for IBIT since 2021
# python scripts/alpaca_bar_backfill.py --symbols IBIT --start 2021-01-01 --dry-run
#
# # Backfill IBIT, BITO, SPY from 2021-01-01 to now
# python scripts/alpaca_bar_backfill.py --start 2021-01-01
#
# # Specify a custom date range
# python scripts/alpaca_bar_backfill.py --start 2021-01-01 --end 2021-12-31
#
# # Only specific symbols
# python scripts/alpaca_bar_backfill.py --symbols IBIT BITO SPY --start 2021-01-01

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import aiohttp

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.core.database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("argus.alpaca_bar_backfill")

ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
# Alpaca allows up to 10,000 bars per request (1Min)
BARS_PER_REQUEST = 10_000
# 1 week of 1-min bars during market hours (~390 per day * 5 * 5)
CHUNK_DAYS = 7

DEFAULT_SYMBOLS = [
    "IBIT", "BITO", "SPY", "QQQ", "IWM", "DIA",
    "GLD", "TLT", "XLE", "XLF", "XLK", "SMH", "NVDA",
]


def _load_credentials() -> tuple[str, str]:
    # Load Alpaca API credentials from secrets.yaml or environment.
    # Try environment first
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if key and secret:
        return key, secret

    # Try secrets.yaml
    try:
        import yaml
        secrets_path = _REPO / "config" / "secrets.yaml"
        with open(secrets_path) as f:
            secrets = yaml.safe_load(f)
        alpaca = secrets.get("alpaca", {})
        key = alpaca.get("api_key", "")
        secret = alpaca.get("api_secret", "")
        if key and secret:
            return key, secret
    except (FileNotFoundError, ImportError, KeyError):
        pass

    logger.error(
        "No Alpaca credentials found. Set ALPACA_API_KEY and ALPACA_API_SECRET "
        "env vars, or add alpaca.api_key / alpaca.api_secret to config/secrets.yaml"
    )
    sys.exit(1)


def _parse_rfc3339_to_ms(ts_str: str) -> int:
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


async def fetch_bars_chunk(
    session: aiohttp.ClientSession,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int = BARS_PER_REQUEST,
) -> List[Dict[str, Any]]:
    # Fetch a single paginated chunk of bars from Alpaca.
    url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
    all_bars: List[Dict[str, Any]] = []
    next_page_token = None

    while True:
        params: Dict[str, Any] = {
            "timeframe": "1Min",
            "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": limit,
            "sort": "asc",
            "feed": "sip",  # consolidated feed
        }
        if next_page_token:
            params["page_token"] = next_page_token

        async with session.get(url, params=params) as resp:
            if resp.status == 422:
                # Symbol may not have existed yet (e.g. IBIT before 2024)
                logger.warning("Alpaca 422 for %s (%s to %s): symbol may not exist yet", symbol, start_dt.date(), end_dt.date())
                return all_bars
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Alpaca API error {resp.status} for {symbol}: {text[:300]}")

            data = await resp.json()
            bars = data.get("bars", []) or []
            all_bars.extend(bars)

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

    return all_bars


def bars_to_rows(bars: List[Dict[str, Any]], symbol: str) -> List[tuple]:
    # Convert Alpaca bar list to market_bars INSERT tuples.
    rows = []
    for bar in bars:
        ts_str = bar["t"]
        bar_ts_ms = _parse_rfc3339_to_ms(ts_str)
        bar_ts_sec = bar_ts_ms / 1000.0

        # Normalize timestamp to DB format
        ts_iso = datetime.fromtimestamp(bar_ts_sec, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        rows.append((
            ts_iso,              # timestamp
            symbol,              # symbol
            "alpaca",            # source
            float(bar["o"]),     # open
            float(bar["h"]),     # high
            float(bar["l"]),     # low
            float(bar["c"]),     # close
            float(bar["v"]),     # volume
            0,                   # tick_count
            0,                   # n_ticks
            bar_ts_sec,          # first_source_ts
            bar_ts_sec,          # last_source_ts
            0,                   # late_ticks_dropped
            4,                   # close_reason = 4 (REST_BACKFILL)
            60,                  # bar_duration
        ))
    return rows


async def backfill_symbol(
    session: aiohttp.ClientSession,
    db: Database,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    dry_run: bool = False,
) -> int:
    # Backfill a single symbol by chunking into CHUNK_DAYS windows.
    total_inserted = 0
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_dt)
        logger.info("  [%s] %s → %s", symbol, chunk_start.date(), chunk_end.date())

        try:
            bars = await fetch_bars_chunk(session, symbol, chunk_start, chunk_end)
            if bars:
                rows = bars_to_rows(bars, symbol)
                if dry_run:
                    logger.info("    [DRY RUN] Would insert %d bars", len(rows))
                else:
                    inserted = await db.upsert_bars_backfill(rows)
                    total_inserted += inserted
                    logger.info("    Inserted %d / %d bars", inserted, len(rows))
            else:
                logger.info("    No bars returned")
        except Exception as e:
            logger.error("    Error fetching chunk %s: %s", chunk_start.date(), e)

        chunk_start = chunk_end
        await asyncio.sleep(0.15)  # ~6 req/sec

    return total_inserted


async def run(
    symbols: List[str],
    start_dt: datetime,
    end_dt: datetime,
    db_path: str = "data/argus.db",
    dry_run: bool = False,
):
    api_key, api_secret = _load_credentials()

    db = Database(db_path)
    await db.connect()

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    grand_total = 0
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            for symbol in symbols:
                logger.info("Backfilling %s from %s to %s", symbol, start_dt.date(), end_dt.date())
                n = await backfill_symbol(session, db, symbol, start_dt, end_dt, dry_run)
                grand_total += n
                logger.info("[%s] Done. Inserted %d bars total", symbol, n)
    finally:
        await db.close()

    logger.info("Backfill complete. Grand total inserted: %d", grand_total)


def main():
    parser = argparse.ArgumentParser(description="Alpaca historical 1-min bar backfill.")
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help="Symbols to backfill (default: all configured ETFs)"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--db", default="data/argus.db", help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    asyncio.run(run(
        symbols=args.symbols,
        start_dt=start_dt,
        end_dt=end_dt,
        db_path=args.db,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
