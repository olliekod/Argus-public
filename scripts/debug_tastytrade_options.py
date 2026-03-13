#!/usr/bin/env python3
"""Debug Tastytrade options snapshot pipeline.

Fetches nested chain, normalizes, builds snapshots, and prints stats.
Use this to verify the Tastytrade snapshot connector is working correctly.

Usage:
  python scripts/debug_tastytrade_options.py --symbol SPY
  python scripts/debug_tastytrade_options.py --symbol IBIT --min-dte 7 --max-dte 21
  python scripts/debug_tastytrade_options.py --symbol QQQ --verbose
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import ConfigurationError, load_config, load_secrets
from src.core.options_normalize import normalize_tastytrade_nested_chain
from src.core.option_events import option_chain_to_dict
from src.connectors.tastytrade_rest import TastytradeError
from src.connectors.tastytrade_options import (
    TastytradeOptionsConnector,
    TastytradeOptionsConfig,
)


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug Tastytrade options snapshot pipeline")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol (default: SPY)")
    parser.add_argument("--min-dte", type=int, default=7, help="Min DTE (default: 7)")
    parser.add_argument("--max-dte", type=int, default=21, help="Max DTE (default: 21)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed contract info")
    parser.add_argument("--json", action="store_true", help="Output snapshot as JSON")
    args = parser.parse_args()

    # Load config
    try:
        config = load_config()
        secrets = load_secrets()
    except ConfigurationError as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        from scripts.tastytrade_health_audit import get_tastytrade_rest_client
        client = get_tastytrade_rest_client(config, secrets)
    except Exception as e:
        print(f"Tastytrade credentials missing or auth failed: {e}")
        return 1

    tt_config = config.get("tastytrade", {})
    retry_cfg = tt_config.get("retries", {})

    print(f"{'=' * 60}")
    print(f"Tastytrade Options Snapshot Debug")
    print(f"{'=' * 60}")
    print(f"Symbol:      {args.symbol}")
    print(f"DTE range:   {args.min_dte} - {args.max_dte}")
    print(f"Environment: {tt_config.get('environment', 'live')}")
    print()

    # Step 1: Fetch raw nested chain
    print("[1/4] Fetching nested chain...")
    start = time.time()
    try:
        raw = client.get_nested_option_chains(args.symbol)
        elapsed = time.time() - start
        print(f"  Fetch OK ({elapsed:.2f}s)")
    except TastytradeError as exc:
        print(f"  Fetch FAILED: {exc}")
        client.close()
        return 1

    client.close()

    # Step 2: Normalize
    print("\n[2/4] Normalizing chain...")
    normalized = normalize_tastytrade_nested_chain(raw)
    expirations = sorted({c["expiry"] for c in normalized if c.get("expiry")})
    strikes = {c["strike"] for c in normalized if c.get("strike") is not None}
    calls = [c for c in normalized if c.get("right") == "C"]
    puts = [c for c in normalized if c.get("right") == "P"]

    print(f"  Total contracts: {len(normalized)}")
    print(f"  Expirations:     {len(expirations)}")
    print(f"  Unique strikes:  {len(strikes)}")
    print(f"  Calls:           {len(calls)}")
    print(f"  Puts:            {len(puts)}")

    if expirations:
        print(f"  First expiry:    {expirations[0]}")
        print(f"  Last expiry:     {expirations[-1]}")

    if args.verbose and normalized:
        print("\n  Sample contracts:")
        for c in normalized[:5]:
            print(f"    {c['right']} {c.get('expiry', '?')} ${c.get('strike', '?')} — {c.get('option_symbol', '?')}")

    # Step 3: Build snapshots via connector (prefer OAuth when configured)
    print("\n[3/4] Building snapshots via TastytradeOptionsConnector...")
    tasty_secrets = secrets.get("tastytrade", {})
    oauth_cfg = secrets.get("tastytrade_oauth2", {}) or {}
    use_oauth = not _is_placeholder(oauth_cfg.get("client_id") or "") and not _is_placeholder(oauth_cfg.get("client_secret") or "") and not _is_placeholder(oauth_cfg.get("refresh_token") or "")
    opts_config = TastytradeOptionsConfig(
        username=tasty_secrets.get("username", ""),
        password=tasty_secrets.get("password", ""),
        oauth_client_id=oauth_cfg.get("client_id", ""),
        oauth_client_secret=oauth_cfg.get("client_secret", ""),
        oauth_refresh_token=oauth_cfg.get("refresh_token", ""),
        environment=tt_config.get("environment", "live"),
        timeout_seconds=tt_config.get("timeout_seconds", 20),
        max_attempts=retry_cfg.get("max_attempts", 3),
        backoff_seconds=retry_cfg.get("backoff_seconds", 1.0),
        backoff_multiplier=retry_cfg.get("backoff_multiplier", 2.0),
        min_dte=args.min_dte,
        max_dte=args.max_dte,
    )
    connector = TastytradeOptionsConnector(config=opts_config)

    try:
        snapshots = connector.build_snapshots_for_symbol(
            args.symbol,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
        )
    except Exception as exc:
        print(f"  Build FAILED: {exc}")
        connector.close()
        return 1

    print(f"  Snapshots built: {len(snapshots)}")

    for snap in snapshots:
        from datetime import datetime, timezone
        exp_dt = datetime.fromtimestamp(snap.expiration_ms / 1000, tz=timezone.utc)
        print(f"\n  Snapshot: {snap.symbol} exp={exp_dt.strftime('%Y-%m-%d')}")
        print(f"    provider:         {snap.provider}")
        print(f"    snapshot_id:      {snap.snapshot_id}")
        print(f"    timestamp_ms:     {snap.timestamp_ms}")
        print(f"    recv_ts_ms:       {snap.recv_ts_ms}")
        print(f"    underlying_price: {snap.underlying_price}")
        print(f"    n_strikes:        {snap.n_strikes}")
        print(f"    puts:             {len(snap.puts)}")
        print(f"    calls:            {len(snap.calls)}")
        print(f"    atm_iv:           {snap.atm_iv}")

        if args.json:
            snap_dict = option_chain_to_dict(snap)
            print(f"\n    JSON ({len(json.dumps(snap_dict))} bytes):")
            # Just print summary, not full JSON (it's huge)
            summary = {k: v for k, v in snap_dict.items() if k not in ('puts', 'calls')}
            summary['puts_count'] = len(snap_dict.get('puts', []))
            summary['calls_count'] = len(snap_dict.get('calls', []))
            print(f"    {json.dumps(summary, indent=2)}")

    # Step 4: Validate
    print(f"\n[4/4] Validation...")
    health = connector.get_health_status()
    print(f"  Connector health: {health['health']}")
    print(f"  Requests:         {health['request_count']}")
    print(f"  Errors:           {health['error_count']}")
    print(f"  Authenticated:    {health['authenticated']}")
    print(f"  Latency:          {health['last_latency_ms']:.0f}ms")

    all_ok = True
    for snap in snapshots:
        if snap.provider != "tastytrade":
            print(f"  FAIL: provider is '{snap.provider}', expected 'tastytrade'")
            all_ok = False
        if snap.timestamp_ms <= 0:
            print(f"  FAIL: timestamp_ms is {snap.timestamp_ms}")
            all_ok = False
        if snap.recv_ts_ms <= 0:
            print(f"  FAIL: recv_ts_ms is {snap.recv_ts_ms}")
            all_ok = False
        if not snap.snapshot_id:
            print(f"  FAIL: snapshot_id is empty")
            all_ok = False

    if all_ok and snapshots:
        print(f"\n  ALL CHECKS PASSED ({len(snapshots)} snapshots)")
    elif not snapshots:
        print(f"\n  WARNING: No snapshots in DTE range [{args.min_dte}, {args.max_dte}]")
    else:
        print(f"\n  SOME CHECKS FAILED")

    connector.close()
    client.close()

    print(f"\n{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
