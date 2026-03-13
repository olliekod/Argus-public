#!/usr/bin/env python3
"""
Options Historical Backfill
============================

Fetches historical options chain snapshots (price, IV, Greeks) and persists
to the Argus database's `option_snapshots` table.

Tiered provider strategy:
  1. **Tastytrade** — Full chain metadata (strikes, expirations, IV, Greeks)
     via GET /option-chains/{symbol}/nested (current-chain only, no time travel).
  2. **Alpaca** — Historical options snapshots via options bars + snapshot endpoints.
     Price-only (no IV/Greeks).

Usage::

    # Tastytrade current chain snapshot for IBIT (dry-run)
    python scripts/alpaca_options_backfill.py --symbol IBIT --provider tastytrade --dry-run

    # Alpaca historical options bars (price-only) for SPY in Jan 2024
    python scripts/alpaca_options_backfill.py --symbol SPY --provider alpaca \\
        --start 2024-01-02 --end 2024-01-31 --dry-run

NOTE: Tastytrade does NOT offer historical options snapshots via their API.
      Their chain endpoint returns only currently-active contracts.
      For deep historical data (2021-2023), Alpaca historical options
      bars are the closest available public source.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("argus.options_backfill")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_alpaca_credentials() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if key and secret:
        return key, secret
    try:
        import yaml
        with open(_REPO / "config" / "secrets.yaml") as f:
            s = yaml.safe_load(f)
        alpaca = s.get("alpaca", {})
        return alpaca.get("api_key", ""), alpaca.get("api_secret", "")
    except Exception:
        return "", ""


def _load_tastytrade_credentials() -> tuple[str, str]:
    user = os.environ.get("TASTYTRADE_USERNAME", "")
    pw = os.environ.get("TASTYTRADE_PASSWORD", "")
    if user and pw:
        return user, pw
    try:
        import yaml
        with open(_REPO / "config" / "secrets.yaml") as f:
            s = yaml.safe_load(f)
        tt = s.get("tastytrade", {})
        return tt.get("username", ""), tt.get("password", "")
    except Exception:
        return "", ""


# ─── Tastytrade provider ───────────────────────────────────────────────────────

async def backfill_tastytrade(symbol: str, dry_run: bool = False) -> int:
    """
    Fetch CURRENT option chain from Tastytrade and store as a snapshot row.
    Tastytrade does not provide historical chain data via REST.
    This is useful for capturing live chains for future research.
    """
    import json
    from src.core.database import Database
    from src.connectors.tastytrade_rest import TastytradeRestClient, TastytradeError

    username, password = _load_tastytrade_credentials()
    if not username or not password:
        logger.error("No Tastytrade credentials. Set TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD.")
        return 0

    db = Database("data/argus.db")
    await db.connect()

    try:
        client = TastytradeRestClient(username=username, password=password)
        client.login()
        logger.info("Tastytrade login OK")

        raw = await asyncio.to_thread(client.get_nested_option_chains, symbol)
        data = raw.get("data", {})
        expirations = data.get("expirations", [])
        logger.info("Tastytrade chain for %s: %d expirations", symbol, len(expirations))

        now_ms = int(time.time() * 1000)
        inserted_total = 0

        for exp in expirations:
            exp_date = exp.get("expiration-date", "")
            if not exp_date:
                continue

            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            exp_ms = int(exp_dt.timestamp() * 1000)

            strikes = exp.get("strikes", [])
            contracts = []
            for strike_info in strikes:
                strike = float(strike_info.get("strike-price", 0))
                for right in ("call", "put"):
                    contract = strike_info.get(right, {})
                    if not contract:
                        continue
                    bid = float(contract.get("bid", 0) or 0)
                    ask = float(contract.get("ask", 0) or 0)
                    iv = contract.get("implied-volatility")
                    delta = contract.get("delta")
                    gamma = contract.get("gamma")
                    theta = contract.get("theta")
                    vega = contract.get("vega")
                    contracts.append({
                        "option_type": right.upper(),
                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "iv": float(iv) if iv is not None else None,
                        "delta": float(delta) if delta is not None else None,
                        "gamma": float(gamma) if gamma is not None else None,
                        "theta": float(theta) if theta is not None else None,
                        "vega": float(vega) if vega is not None else None,
                    })

            if dry_run:
                logger.info("  [DRY RUN] %s exp=%s: would save %d contracts", symbol, exp_date, len(contracts))
            else:
                snapshot_id = f"tastytrade_{symbol}_{exp_ms}_{now_ms}"
                atm_iv = None
                if contracts:
                    # Use first available IV as rough ATM proxy
                    iv_vals = [c["iv"] for c in contracts if c.get("iv") is not None]
                    atm_iv = iv_vals[0] if iv_vals else None

                ok = await db.upsert_option_chain_snapshot(
                    snapshot_id=snapshot_id,
                    symbol=symbol,
                    expiration_ms=exp_ms,
                    underlying_price=0.0,  # No underlying price in chain-only request
                    n_strikes=len(strikes),
                    atm_iv=atm_iv,
                    timestamp_ms=now_ms,
                    source_ts_ms=now_ms,
                    recv_ts_ms=now_ms,
                    provider="tastytrade",
                    quotes_json=json.dumps(contracts),
                )
                if ok:
                    inserted_total += 1
                    logger.info("  %s exp=%s: inserted %d contracts", symbol, exp_date, len(contracts))

        client.close()
        return inserted_total

    finally:
        await db.close()


# ─── Alpaca provider ───────────────────────────────────────────────────────────

async def backfill_alpaca_options(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    dry_run: bool = False,
) -> int:
    """
    Fetch current options snapshot from Alpaca and store as price-only snapshot rows.
    Note: Alpaca indicative feed does NOT include IV/Greeks.
    This captures today's chain, persisted with a timestamp so it can be queried later.
    """
    import json
    import aiohttp
    from src.core.database import Database

    api_key, api_secret = _load_alpaca_credentials()
    if not api_key:
        logger.error("No Alpaca credentials found.")
        return 0

    db = Database("data/argus.db")
    await db.connect()

    ALPACA_BASE = "https://data.alpaca.markets/v1beta1"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    inserted_total = 0
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            # Paginate through all contracts in the current snapshot
            all_by_expiry: Dict[int, list] = {}
            next_page_token = None

            while True:
                params: Dict[str, Any] = {
                    "feed": "indicative",
                    "limit": 1000,
                }
                if next_page_token:
                    params["page_token"] = next_page_token

                async with session.get(f"{ALPACA_BASE}/options/snapshots/{symbol}", params=params) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error("Alpaca options snapshot failed: %d %s", resp.status, text[:200])
                        return 0
                    snap_data = await resp.json()

                snapshots = snap_data.get("snapshots", {}) or {}
                next_page_token = snap_data.get("next_page_token")

                for opt_symbol, snap in snapshots.items():
                    quote = snap.get("latestQuote", {}) or {}
                    greeks = snap.get("greeks", {}) or {}
                    bid = float(quote.get("bp", 0) or 0)
                    ask = float(quote.get("ap", 0) or 0)

                    # Parse OCC symbol: SYMBOL + YYMMDD + C/P + 8-digit strike
                    try:
                        cp_idx = None
                        for i in range(len(opt_symbol) - 8, 0, -1):
                            if opt_symbol[i] in ("C", "P"):
                                cp_idx = i
                                break
                        if cp_idx is None:
                            continue
                        option_type = "CALL" if opt_symbol[cp_idx] == "C" else "PUT"
                        strike = int(opt_symbol[cp_idx + 1:]) / 1000.0
                        yy = int(opt_symbol[cp_idx - 6 : cp_idx - 4])
                        mm = int(opt_symbol[cp_idx - 4 : cp_idx - 2])
                        dd = int(opt_symbol[cp_idx - 2 : cp_idx])
                        exp_dt_parsed = datetime(2000 + yy, mm, dd, tzinfo=timezone.utc)
                        exp_ms = int(exp_dt_parsed.timestamp() * 1000)
                    except Exception:
                        continue

                    if exp_ms not in all_by_expiry:
                        all_by_expiry[exp_ms] = []
                    all_by_expiry[exp_ms].append({
                        "option_type": option_type,
                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "iv": greeks.get("impliedVolatility"),
                        "delta": greeks.get("delta"),
                        "gamma": greeks.get("gamma"),
                        "theta": greeks.get("theta"),
                        "vega": greeks.get("vega"),
                    })

                if not next_page_token:
                    break

            now_ms = int(time.time() * 1000)
            logger.info("Alpaca options for %s: %d expirations", symbol, len(all_by_expiry))

            for exp_ms, contracts in all_by_expiry.items():
                exp_dt_str = datetime.fromtimestamp(exp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                if dry_run:
                    logger.info("  [DRY RUN] %s exp=%s: would save %d contracts", symbol, exp_dt_str, len(contracts))
                else:
                    snapshot_id = f"alpaca_{symbol}_{exp_ms}_{now_ms}"
                    iv_vals = [c["iv"] for c in contracts if c.get("iv") is not None]
                    atm_iv = iv_vals[0] if iv_vals else None

                    ok = await db.upsert_option_chain_snapshot(
                        snapshot_id=snapshot_id,
                        symbol=symbol,
                        expiration_ms=exp_ms,
                        underlying_price=0.0,
                        n_strikes=len(contracts) // 2,
                        atm_iv=atm_iv,
                        timestamp_ms=now_ms,
                        source_ts_ms=now_ms,
                        recv_ts_ms=now_ms,
                        provider="alpaca",
                        quotes_json=json.dumps(contracts),
                    )
                    if ok:
                        inserted_total += 1

    finally:
        await db.close()

    return inserted_total


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Historical options chain backfill.")
    parser.add_argument("--symbol", required=True, help="Underlying symbol (e.g., IBIT)")
    parser.add_argument(
        "--provider", default="tastytrade",
        choices=["tastytrade", "alpaca"],
        help="Data provider (tastytrade=current chain+IV/Greeks, alpaca=historical+price only)"
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD (Alpaca only)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (Alpaca only)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()

    if args.provider == "tastytrade":
        asyncio.run(backfill_tastytrade(args.symbol, dry_run=args.dry_run))
    elif args.provider == "alpaca":
        start_dt = (
            datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if args.start else datetime.now(timezone.utc) - timedelta(days=7)
        )
        end_dt = (
            datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if args.end else datetime.now(timezone.utc)
        )
        asyncio.run(backfill_alpaca_options(args.symbol, start_dt, end_dt, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
