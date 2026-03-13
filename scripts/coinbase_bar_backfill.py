#!/usr/bin/env python3
"""
Coinbase BTC Bar Backfill
=========================

Fetches historical 1-minute BTC bars from Coinbase and persists to the Argus database.
Uses the public Coinbase Exchange candles API (US-friendly).

Usage::

    # Backfill last 24 hours of BTC
    python scripts/coinbase_bar_backfill.py --hours 24

    # Backfill from 2021-01-01 to 2021-02-01
    python scripts/coinbase_bar_backfill.py --start 2021-01-01 --end 2021-02-01

    # Dry-run
    python scripts/coinbase_bar_backfill.py --dry-run
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import aiohttp

# Ensure repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.core.database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("argus.coinbase_backfill")

COINBASE_API_URL = "https://api.exchange.coinbase.com/products/{}/candles"

async def fetch_candles_chunk(
    session: aiohttp.ClientSession,
    product_id: str,
    start_dt: datetime,
    end_dt: datetime,
    granularity: int = 60
) -> List[List]:
    """Fetch a single chunk of candles from Coinbase."""
    url = COINBASE_API_URL.format(product_id)
    params = {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "granularity": granularity
    }
    
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"Coinbase API error {resp.status}: {text}")
        return await resp.json()

async def backfill(
    product_id: str,
    start_dt: datetime,
    end_dt: datetime,
    db_path: str = "data/argus.db",
    dry_run: bool = False,
):
    """Run the backfill by chunking requests."""
    db = Database(db_path)
    await db.connect()

    async with aiohttp.ClientSession(headers={"User-Agent": "Argus/1.0"}) as session:
        current_start = start_dt
        # Coinbase returns max 300 points per request (5 hours at 1-min granularity)
        chunk_delta = timedelta(minutes=300)
        
        total_inserted = 0
        
        while current_start < end_dt:
            current_end = min(current_start + chunk_delta, end_dt)
            logger.info("Fetching chunk: %s to %s", current_start.isoformat(), current_end.isoformat())
            
            try:
                # Coinbase returns [time, low, high, open, close, volume]
                # 'time' is bucket start time in seconds since epoch
                candles = await fetch_candles_chunk(session, product_id, current_start, current_end)
                
                if not candles:
                    logger.warning("No candles in chunk %s to %s", current_start.isoformat(), current_end.isoformat())
                else:
                    rows = []
                    for c in candles:
                        ts_ms = c[0] * 1000
                        ts_iso = datetime.fromtimestamp(c[0], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        
                        # (timestamp, symbol, source, open, high, low, close, volume,
                        #  tick_count, n_ticks, first_source_ts, last_source_ts,
                        #  late_ticks_dropped, close_reason, bar_duration)
                        rows.append((
                            ts_iso,
                            product_id.replace("-", ""), # BTCUSD
                            "coinbase",
                            c[3], # open
                            c[2], # high
                            c[1], # low
                            c[4], # close
                            c[5], # volume
                            0,    # tick_count
                            0,    # n_ticks
                            c[0], # first_source_ts
                            c[0], # last_source_ts
                            0,    # late_ticks_dropped
                            4,    # close_reason = 4 (REST_BACKFILL)
                            60    # bar_duration
                        ))
                    
                    if dry_run:
                        logger.info("  [DRY RUN] Would insert %d bars", len(rows))
                    else:
                        inserted = await db.upsert_bars_backfill(rows)
                        total_inserted += inserted
                        logger.info("  Inserted %d bars", inserted)
            
            except Exception as e:
                logger.error("Failed to fetch chunk: %s", e)
                # Wait before retry or skip? Let's stop to be safe.
                break

            current_start = current_end
            # Respect rate limits (10 req/sec usually, but let's be kind)
            await asyncio.sleep(0.2)

    await db.close()
    logger.info("Backfill complete. Total inserted: %d", total_inserted)

def main():
    parser = argparse.ArgumentParser(description="Coinbase historical bar backfill.")
    parser.add_argument("--product", default="BTC-USD", help="Coinbase product ID (default: BTC-USD)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD, default: now)")
    parser.add_argument("--hours", type=int, help="Lookback hours from now (instead of --start)")
    parser.add_argument("--db", default="data/argus.db", help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")

    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    
    if args.hours:
        start_dt = now_utc - timedelta(hours=args.hours)
        end_dt = now_utc
    elif args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end:
            end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            end_dt = now_utc
    else:
        logger.error("Must specify --start or --hours")
        sys.exit(1)

    asyncio.run(backfill(
        product_id=args.product,
        start_dt=start_dt,
        end_dt=end_dt,
        db_path=args.db,
        dry_run=args.dry_run
    ))

if __name__ == "__main__":
    main()
