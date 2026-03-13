#!/usr/bin/env python3
"""
Alpha Vantage Daily Backfill
============================

Fetches daily OHLCV bars for global ETF proxies and FX pairs from
Alpha Vantage and persists them to the Argus database.

Usage::

    # Backfill all configured symbols (compact = last 100 days)
    python scripts/alphavantage_daily_backfill.py

    # Full history (20+ years)
    python scripts/alphavantage_daily_backfill.py --full

    # Specific symbols only
    python scripts/alphavantage_daily_backfill.py --symbols EWJ FXI

    # Dry-run (no DB writes)
    python scripts/alphavantage_daily_backfill.py --dry-run

Persists to ``market_bars`` with ``source="alphavantage"``,
``bar_duration=86400``.  Uses INSERT OR IGNORE so live bars are
never overwritten.

Budget: 14 symbols × 1 call each = 14 calls.  Free tier allows
25/day at 5/min.  The client enforces 12.5s between calls.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml

from src.connectors.alphavantage_client import AlphaVantageClient
from src.core.config import load_secrets, get_secret
from src.core.database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("argus.alphavantage_backfill")

# Bar duration for daily bars
_BAR_DURATION = 86400

# ISO timestamp format for display
_ISO = "%Y-%m-%d"


def _load_config() -> Dict[str, Any]:
    """Load alphavantage config from config/config.yaml."""
    cfg_path = _REPO / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("exchanges", {}).get("alphavantage", {})


def _bar_to_row(bar: Dict[str, Any]) -> tuple:
    """Convert a bar dict to a market_bars INSERT tuple.

    Columns: (timestamp, symbol, source, open, high, low, close,
              volume, tick_count, n_ticks, first_source_ts,
              last_source_ts, late_ticks_dropped, close_reason,
              bar_duration)
    """
    ts_iso = datetime.fromtimestamp(
        bar["timestamp_ms"] / 1000, tz=timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    return (
        ts_iso,                     # timestamp (ISO string)
        bar["symbol"],              # symbol
        "alphavantage",             # source
        bar["open"],                # open
        bar["high"],                # high
        bar["low"],                 # low
        bar["close"],               # close
        bar["volume"],              # volume
        0,                          # tick_count (N/A for daily)
        0,                          # n_ticks
        None,                       # first_source_ts
        None,                       # last_source_ts
        0,                          # late_ticks_dropped
        "DAILY_BAR",                # close_reason
        _BAR_DURATION,              # bar_duration
    )


async def backfill(
    symbols: List[str],
    fx_pairs: List[str],
    *,
    outputsize: str = "compact",
    db_path: str = "data/argus.db",
    dry_run: bool = False,
) -> Dict[str, int]:
    """Run the backfill for all symbols and FX pairs.

    Returns dict mapping symbol → bars_inserted.
    """
    secrets = load_secrets()
    api_key = get_secret(secrets, "alphavantage", "api_key")
    if not api_key:
        logger.error(
            "No Alpha Vantage API key found. "
            "Set alphavantage.api_key in config/secrets.yaml"
        )
        return {}

    client = AlphaVantageClient(api_key=api_key)
    db = Database(db_path)
    await db.connect()

    results: Dict[str, int] = {}
    total_calls = len(symbols) + len(fx_pairs)
    call_num = 0

    today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    try:
        # Fetch ETF daily bars
        for symbol in symbols:
            call_num += 1
            logger.info(
                "[%d/%d] Fetching daily bars for %s ...",
                call_num, total_calls, symbol,
            )
            try:
                bars = await client.fetch_daily_bars(
                    symbol, outputsize=outputsize,
                )
                # Filter out today's bar if we are running mid-day
                full_bars = [
                    b for b in bars 
                    if datetime.fromtimestamp(b["timestamp_ms"]/1000, tz=timezone.utc).strftime("%Y-%m-%d") != today_date
                ]
                if len(full_bars) < len(bars):
                    logger.info("  Filtered out today's partial bar for %s", symbol)
                
                rows = [_bar_to_row(b) for b in full_bars]

                if dry_run:
                    logger.info(
                        "  [DRY RUN] Would insert %d bars for %s",
                        len(rows), symbol,
                    )
                    results[symbol] = 0
                else:
                    inserted = await db.upsert_bars_backfill(rows)
                    results[symbol] = inserted
                    logger.info(
                        "  Inserted %d / %d bars for %s",
                        inserted, len(rows), symbol,
                    )
            except AlphaVantageClient.AlphaVantageRateLimitError as exc:
                logger.error("  Stopped: rate limit exhausted: %s", exc)
                results[symbol] = -1
                return results # Stop everything
            except Exception as exc:
                logger.warning(
                    "  Failed to fetch %s: %s", symbol, exc,
                )
                results[symbol] = -1

        # Fetch FX daily rates
        for pair in fx_pairs:
            call_num += 1
            parts = pair.replace("/", " ").split()
            if len(parts) != 2:
                logger.warning("Invalid FX pair format: %s", pair)
                continue

            from_ccy, to_ccy = parts
            fx_symbol = f"FX:{from_ccy}{to_ccy}"
            logger.info(
                "[%d/%d] Fetching daily FX for %s ...",
                call_num, total_calls, fx_symbol,
            )
            try:
                bars = await client.fetch_fx_daily(
                    from_ccy, to_ccy, outputsize=outputsize,
                )
                # Filter out today's bar
                full_bars = [
                    b for b in bars 
                    if datetime.fromtimestamp(b["timestamp_ms"]/1000, tz=timezone.utc).strftime("%Y-%m-%d") != today_date
                ]
                if len(full_bars) < len(bars):
                    logger.info("  Filtered out today's partial bar for %s", fx_symbol)

                rows = [_bar_to_row(b) for b in full_bars]

                if dry_run:
                    logger.info(
                        "  [DRY RUN] Would insert %d bars for %s",
                        len(rows), fx_symbol,
                    )
                    results[fx_symbol] = 0
                else:
                    inserted = await db.upsert_bars_backfill(rows)
                    results[fx_symbol] = inserted
                    logger.info(
                        "  Inserted %d / %d bars for %s",
                        inserted, len(rows), fx_symbol,
                    )
            except AlphaVantageClient.AlphaVantageRateLimitError as exc:
                logger.error("  Stopped: rate limit exhausted: %s", exc)
                results[fx_symbol] = -1
                return results # Stop everything
            except Exception as exc:
                logger.warning(
                    "  Failed to fetch %s: %s", fx_symbol, exc,
                )
                results[fx_symbol] = -1

    finally:
        await client.close()
        await db.close()

    # Summary
    total_inserted = sum(v for v in results.values() if v > 0)
    total_symbols = len(results)
    failed = sum(1 for v in results.values() if v < 0)
    logger.info(
        "Backfill complete: %d symbols, %d bars inserted, %d failed",
        total_symbols, total_inserted, failed,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpha Vantage daily backfill for global ETF proxies.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Fetch full history (20+ years) instead of compact (100 days)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override symbol list (e.g. --symbols EWJ FXI)",
    )
    parser.add_argument(
        "--db",
        default="data/argus.db",
        help="Database path (default: data/argus.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Fetch data but don't write to DB",
    )

    args = parser.parse_args()

    # Load config
    av_config = _load_config()

    # Resolve symbols
    if args.symbols:
        symbols = args.symbols
        fx_pairs: List[str] = []
    else:
        symbols = av_config.get("daily_symbols", [])
        fx_pairs = av_config.get("fx_pairs", [])

    if not symbols and not fx_pairs:
        logger.error("No symbols configured. Check config/config.yaml.")
        sys.exit(1)

    outputsize = "full" if args.full else "compact"

    asyncio.run(backfill(
        symbols=symbols,
        fx_pairs=fx_pairs,
        outputsize=outputsize,
        db_path=args.db,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
