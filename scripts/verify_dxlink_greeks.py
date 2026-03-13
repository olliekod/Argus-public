#!/usr/bin/env python3
"""Verify that DXLink is delivering Greeks (IV) when subscribing with option symbols.

Run: python scripts/verify_dxlink_greeks.py

If you see "Greeks events received: 0", the streamer was likely subscribing to
underlying symbols only; the orchestrator fix uses option-level symbols so IV
is populated. This script uses option symbols to confirm the pipeline works.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_config, load_secrets
from src.connectors.tastytrade_streamer import TastytradeStreamer
from src.connectors.tastytrade_dxlink_parser import GreeksEvent


async def main() -> int:
    try:
        config = load_config()
        secrets = load_secrets()
    except Exception as e:
        print(f"Config error: {e}")
        return 1

    try:
        from scripts.tastytrade_health_audit import get_tastytrade_rest_client
        client = get_tastytrade_rest_client(config, secrets)
    except Exception as e:
        print(f"Tastytrade auth failed: {e}")
        return 1

    try:
        quote_info = client.get_api_quote_token()
    except Exception as e:
        print(f"Quote token failed: {e}")
        client.close()
        return 1

    dxlink_url = quote_info["dxlink-url"]
    dxlink_token = quote_info["token"]
    client.close()

    # Get option symbols from connector (same logic as orchestrator)
    from src.connectors.tastytrade_options import TastytradeOptionsConnector, TastytradeOptionsConfig
    tt_cfg = config.get("tastytrade", {})
    tt_sec = secrets.get("tastytrade", {})
    oauth = secrets.get("tastytrade_oauth2", {})
    connector = TastytradeOptionsConnector(
        config=TastytradeOptionsConfig(
            username=tt_sec.get("username", ""),
            password=tt_sec.get("password", ""),
            oauth_client_id=oauth.get("client_id", ""),
            oauth_client_secret=oauth.get("client_secret", ""),
            oauth_refresh_token=oauth.get("refresh_token", ""),
            environment=tt_cfg.get("environment", "live"),
            timeout_seconds=tt_cfg.get("timeout_seconds", 20),
            max_attempts=tt_cfg.get("retries", {}).get("max_attempts", 3),
            backoff_seconds=tt_cfg.get("retries", {}).get("backoff_seconds", 1.0),
            backoff_multiplier=tt_cfg.get("retries", {}).get("backoff_multiplier", 2.0),
        )
    )
    underlyings = tt_cfg.get("underlyings", ["SPY"])[:3]
    option_symbols = connector.get_dxlink_option_symbols(
        underlyings,
        min_dte=7,
        max_dte=60,
        max_total=50,
    )
    connector.close()

    if not option_symbols:
        print("No option symbols from chain; subscribing to underlyings only (expect 0 Greeks).")
        symbols_for_stream = underlyings
    else:
        print(f"Subscribing to {len(option_symbols)} option symbols for Greeks.")
        symbols_for_stream = option_symbols

    greeks_count = 0
    sample_ivs: list[float] = []

    def on_event(event) -> None:
        nonlocal greeks_count, sample_ivs
        if isinstance(event, GreeksEvent) and event.volatility is not None:
            greeks_count += 1
            if len(sample_ivs) < 5:
                sample_ivs.append(event.volatility * 100)

    streamer = TastytradeStreamer(
        dxlink_url=dxlink_url,
        token=dxlink_token,
        symbols=symbols_for_stream,
        event_types=["Greeks"],
        on_event=on_event,
    )

    print("Running DXLink for 15s (Greeks only)...")
    try:
        await asyncio.wait_for(streamer.run_for(15.0), timeout=20.0)
    except asyncio.TimeoutError:
        pass

    print(f"Greeks events received: {greeks_count}")
    if sample_ivs:
        print(f"Sample IVs (%%): {[round(v, 2) for v in sample_ivs]}")
    if greeks_count == 0 and symbols_for_stream == underlyings:
        print("-> Subscribing to underlyings does not yield Greeks; use option symbols (orchestrator fix).")
    elif greeks_count == 0:
        print("-> No Greeks received; check market hours or symbol format.")
    else:
        print("-> IV pipeline OK.")

    return 0 if greeks_count > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
