#!/usr/bin/env python3
"""Benchmark providers for the liquid ETF universe with separate scorecards."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import asyncio
import json
import time
import requests
from datetime import datetime, timezone
from typing import Any

from scripts.tastytrade_health_audit import audit_oauth, run_quotes_audit
from src.connectors.alpaca_client import AlpacaDataClient
from src.connectors.yahoo_client import YahooFinanceClient
from src.connectors.tastytrade_streamer import TastytradeStreamer
from src.core.bus import EventBus
from src.core.config import ConfigurationError, load_config, load_secrets
from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE
from src.core.options_normalize import normalize_tastytrade_nested_chain
from src.connectors.tastytrade_dxlink_parser import QuoteEvent

BARS_WEIGHTS = {"success_rate": 0.5, "latency_p95": 0.25, "bar_age_p95": 0.25}
BARS_THRESHOLDS = {"latency_ms": 3000.0, "bar_age_sec": 300.0}
OPTIONS_WEIGHTS = {"missing_rate": 0.45, "latency": 0.25, "stale": 0.15, "spread": 0.15}
OPTIONS_THRESHOLDS = {"latency_ms": 5000.0, "stale_sec": 5.0, "spread_bps": 150.0}
GREEKS_WEIGHTS = {"presence": 0.7, "stale": 0.3}
GREEKS_THRESHOLDS = {"stale_sec": 5.0}


def _is_placeholder(v: str) -> bool:
    return (not v) or v.startswith("PASTE_") or v.startswith("YOUR_")


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int((p / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def bars_score(success_rate: float, latency_p95: float | None, bar_age_p95: float | None) -> float | None:
    if latency_p95 is None or bar_age_p95 is None:
        return None
    return round(
        BARS_WEIGHTS["success_rate"] * success_rate
        + BARS_WEIGHTS["latency_p95"] * _clamp(1 - latency_p95 / BARS_THRESHOLDS["latency_ms"])
        + BARS_WEIGHTS["bar_age_p95"] * _clamp(1 - bar_age_p95 / BARS_THRESHOLDS["bar_age_sec"]),
        4,
    )


def options_score(missing_rate: float, latency_p95: float | None, stale_p95: float | None, spread_p95: float | None) -> float | None:
    if latency_p95 is None or stale_p95 is None or spread_p95 is None:
        return None
    return round(
        OPTIONS_WEIGHTS["missing_rate"] * (1 - missing_rate)
        + OPTIONS_WEIGHTS["latency"] * _clamp(1 - latency_p95 / OPTIONS_THRESHOLDS["latency_ms"])
        + OPTIONS_WEIGHTS["stale"] * _clamp(1 - stale_p95 / OPTIONS_THRESHOLDS["stale_sec"])
        + OPTIONS_WEIGHTS["spread"] * _clamp(1 - spread_p95 / OPTIONS_THRESHOLDS["spread_bps"]),
        4,
    )


def greeks_score(presence_rate: float, stale_p95: float | None) -> float | None:
    if stale_p95 is None:
        return None
    return round(
        GREEKS_WEIGHTS["presence"] * presence_rate
        + GREEKS_WEIGHTS["stale"] * _clamp(1 - stale_p95 / GREEKS_THRESHOLDS["stale_sec"]),
        4,
    )


async def _bars_rows(secrets: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skips: list[str] = []
    universe = list(LIQUID_ETF_UNIVERSE)

    yahoo = YahooFinanceClient(symbols=universe)
    for sym in universe:
        start = time.perf_counter()
        success = False
        age = None
        try:
            quote = await yahoo.get_quote(sym)
            success = bool(quote and quote.get("price"))
            source_ts = (quote or {}).get("source_ts")
            if source_ts is not None:
                age = max(0.0, time.time() - float(source_ts))
        except Exception:
            success = False
        rows.append({"provider": "yahoo", "symbol": sym, "request_latency_ms": (time.perf_counter() - start) * 1000, "bar_age_sec": age, "success": success})
    await yahoo.close()

    key = secrets.get("alpaca", {}).get("api_key", "")
    sec = secrets.get("alpaca", {}).get("api_secret", "")
    if _is_placeholder(key) or _is_placeholder(sec):
        skips.append("alpaca bars skipped: credentials missing")
    else:
        alpaca = AlpacaDataClient(api_key=key, api_secret=sec, symbols=universe, event_bus=EventBus(), poll_interval=60)
        for sym in universe:
            start = time.perf_counter()
            success = False
            age = None
            try:
                bars = await alpaca.fetch_bars(sym, limit=1)
                success = bool(bars)
                if bars and isinstance(bars[0].get("t"), str):
                    ts = datetime.fromisoformat(bars[0]["t"].replace("Z", "+00:00"))
                    age = (datetime.now(timezone.utc) - ts).total_seconds()
            except Exception:
                success = False
            rows.append({"provider": "alpaca", "symbol": sym, "request_latency_ms": (time.perf_counter() - start) * 1000, "bar_age_sec": age, "success": success})
        await alpaca.close()

    return rows, skips


async def _options_rows(config: dict[str, Any], secrets: dict[str, Any], duration: int, greeks: bool) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skips: list[str] = []
    try:
        from scripts.tastytrade_health_audit import get_tastytrade_rest_client
        client = get_tastytrade_rest_client(config, secrets)
        oauth = audit_oauth(secrets)
    except Exception as exc:
        skips.append(f"tastytrade options skipped: {exc}")
        return rows, skips

    try:
        # We audit a subset to keep benchmark duration reasonable
        subset = list(LIQUID_ETF_UNIVERSE)[:5]
        for sym in subset:
            chain = client.get_nested_option_chains(sym)
            normalized = normalize_tastytrade_nested_chain(chain)
            probe = await run_quotes_audit(sym, normalized, oauth["access_token"], duration=duration, greeks=greeks)
            rows.append({"provider": "tastytrade_dxlink", **probe})
    except Exception as exc:
        import traceback
        skips.append(f"tastytrade options error: {exc}\n{traceback.format_exc()}")
    finally:
        client.close()
    return rows, skips


async def _underlying_head_to_head(secrets: dict[str, Any], duration: int = 5) -> list[dict[str, Any]]:
    """Compare Alpaca REST vs Tastytrade DXLink for underlying quotes."""
    results = []
    symbols = ["SPY", "IBIT", "QQQ"]
    
    # 1. Tastytrade DXLink
    try:
        from scripts.tastytrade_health_audit import audit_oauth
        oauth = audit_oauth(secrets)
        
        quote_resp = requests.get(
            "https://api.tastytrade.com/api-quote-tokens",
            headers={"Authorization": f"Bearer {oauth['access_token']}"},
            timeout=20
        )
        quote_resp.raise_for_status()
        quote_data = quote_resp.json().get("data", {})
        
        streamer = TastytradeStreamer(quote_data["dxlink-url"], quote_data["token"], symbols)
        events = await streamer.run_for(duration)
        
        tasty_lags = []
        event_counts = {}
        events_missing_ts = 0
        first_event_time = None
        for e in events:
            etype = type(e).__name__
            event_counts[etype] = event_counts.get(etype, 0) + 1
            if isinstance(e, QuoteEvent):
                if first_event_time is None and hasattr(e, 'receipt_time') and e.receipt_time:
                    first_event_time = e.receipt_time
                exch_ts = max(e.bid_time or 0, e.ask_time or 0)
                if exch_ts > 0 and e.receipt_time:
                    tasty_lags.append(e.receipt_time - exch_ts)
                elif exch_ts <= 0:
                    events_missing_ts += 1

        total_events = sum(event_counts.values())
        # Receipt-time based metrics: quote_update_rate, quote_age
        quote_update_rate = total_events / max(1.0, duration) if total_events > 0 else 0.0

        if tasty_lags:
            results.append({
                "provider": "tastytrade_dxlink",
                "type": "streaming",
                "lag_p50": _pct(tasty_lags, 50),
                "lag_p95": _pct(tasty_lags, 95),
                "events": len(tasty_lags),
                "total_events": total_events,
                "events_missing_ts": events_missing_ts,
                "timestamp_missing_rate": round(events_missing_ts / max(1, total_events), 4),
                "quote_update_rate": round(quote_update_rate, 2),
            })
        else:
            # All timestamps are zero — report receipt-based metrics only
            results.append({
                "provider": "tastytrade_dxlink",
                "type": "streaming",
                "lag_p50": None,
                "lag_p95": None,
                "events": total_events,
                "total_events": total_events,
                "events_missing_ts": events_missing_ts,
                "timestamp_missing_rate": round(events_missing_ts / max(1, total_events), 4),
                "quote_update_rate": round(quote_update_rate, 2),
                "note": "provider timestamps zero — latency not measurable"
            })
    except Exception as e:
        results.append({"provider": "tastytrade_dxlink", "error": str(e)})

    # 2. Alpaca REST Snapshot
    try:
        key = secrets.get("alpaca", {}).get("api_key", "")
        sec = secrets.get("alpaca", {}).get("api_secret", "")
        if not _is_placeholder(key):
            alpaca_lags = []
            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
            
            for sym in symbols:
                url = f"https://data.alpaca.markets/v2/stocks/{sym}/snapshot"
                r = requests.get(url, headers=headers, timeout=5)
                receipt = int(time.time() * 1000)
                if r.status_code == 200:
                    snap = r.json()
                    q = snap.get("latestQuote")
                    if q and q.get("t"):
                        ts_dt = datetime.fromisoformat(q["t"].replace("Z", "+00:00"))
                        ts_ms = int(ts_dt.timestamp() * 1000)
                        lag = receipt - ts_ms
                        # If lag is slightly negative, it's clock skew (Alpaca clock is ahead)
                        # We use 0 as floor for freshness comparison.
                        alpaca_lags.append(max(0, lag))
            
            results.append({
                "provider": "alpaca_rest",
                "type": "polling",
                "lag_p50": _pct(alpaca_lags, 50),
                "lag_p95": _pct(alpaca_lags, 95),
                "events": len(alpaca_lags)
            })
    except Exception as e:
        results.append({"provider": "alpaca_rest", "error": str(e)})
        
    return results


def _build_scorecards(bar_rows: list[dict[str, Any]], option_rows: list[dict[str, Any]], greeks_enabled: bool) -> dict[str, Any]:
    bars_card: list[dict[str, Any]] = []
    for provider in sorted({r["provider"] for r in bar_rows}):
        subset = [r for r in bar_rows if r["provider"] == provider]
        success_rate = sum(1 for r in subset if r["success"]) / max(1, len(subset))
        lat = [float(r["request_latency_ms"]) for r in subset]
        age = [float(r["bar_age_sec"]) for r in subset if r.get("bar_age_sec") is not None]
        row = {
            "provider": provider,
            "request_latency_ms_p50": _pct(lat, 50),
            "request_latency_ms_p95": _pct(lat, 95),
            "bar_age_sec_p50": _pct(age, 50),
            "bar_age_sec_p95": _pct(age, 95),
            "success_rate": round(success_rate, 4),
        }
        row["score"] = bars_score(row["success_rate"], row["request_latency_ms_p95"], row["bar_age_sec_p95"])
        bars_card.append(row)

    options_card: list[dict[str, Any]] = []
    if option_rows:
        missing = [float(r["missing_rate"]) for r in option_rows]
        lat = [float(r["handshake_latency"]) for r in option_rows if r.get("handshake_latency") is not None]
        # Use receipt-time-based staleness — filter out rows with zero/missing lag
        stale = [float(r["lag_p95"] / 1000.0) for r in option_rows if r.get("lag_p95") is not None and r["lag_p95"] > 0]
        spread = [float(r["spread_p95"]) for r in option_rows if r.get("spread_p95") is not None]

        # Count events with missing timestamps
        total_events = sum(int(r.get("total_events", 0)) for r in option_rows)
        events_missing_ts = sum(int(r.get("events_missing_ts", 0)) for r in option_rows)
        ts_missing_rate = events_missing_ts / max(1, total_events)

        # Quote update rate (events/sec)
        total_duration_sec = sum(float(r.get("duration_sec", 0)) for r in option_rows)
        quote_update_rate = total_events / max(1.0, total_duration_sec)

        row = {
            "provider": "tastytrade_dxlink",
            "time_to_first_quote_ms_p50": _pct(lat, 50),
            "time_to_first_quote_ms_p95": _pct(lat, 95),
            "missing_quote_rate": round(sum(missing) / len(missing), 4) if missing else None,
            "quote_update_rate": round(quote_update_rate, 2),
            "quote_age_p50": _pct(stale, 50),
            "quote_age_p95": _pct(stale, 95),
            "spread_bps_p50": _pct(spread, 50),
            "spread_bps_p95": _pct(spread, 95),
            "timestamp_missing_rate": round(ts_missing_rate, 4),
        }
        row["score"] = options_score(
            row["missing_quote_rate"] if row["missing_quote_rate"] is not None else 1.0,
            row["time_to_first_quote_ms_p95"],
            row["quote_age_p95"] if row["quote_age_p95"] is not None else row.get("stale_p95"),
            row["spread_bps_p95"]
        )
        options_card.append(row)

    greeks_card: list[dict[str, Any]] = []
    if greeks_enabled and option_rows:
        presence = [float(r.get("greeks_presence", 0.0)) for r in option_rows]
        # Greeks staleness based on effective timestamp (event or receipt)
        greeks_age = [float(r.get("greeks_age_p95", 0.0)) for r in option_rows if r.get("greeks_age_p95") is not None and r["greeks_age_p95"] > 0]
        # Fallback to lag_p95 if greeks_age_p95 not available
        if not greeks_age:
            greeks_age = [float(r["lag_p95"] / 1000.0) for r in option_rows if r.get("lag_p95") is not None and r["lag_p95"] > 0]
        row = {
            "provider": "tastytrade_dxlink",
            "greeks_presence_rate": round(sum(presence) / len(presence), 4) if presence else 0.0,
            "greeks_age_p50": _pct(greeks_age, 50),
            "greeks_age_p95": _pct(greeks_age, 95),
        }
        row["score"] = greeks_score(row["greeks_presence_rate"], row["greeks_age_p95"])
        greeks_card.append(row)

    composite_required = bool(bars_card) and bool(options_card) and (bool(greeks_card) if greeks_enabled else True)
    composite = {
        "status": "complete" if composite_required else "partial",
        "score": None,
    }
    if composite_required:
        parts = [r["score"] for r in bars_card if r.get("score") is not None]
        parts += [r["score"] for r in options_card if r.get("score") is not None]
        if greeks_enabled:
            parts += [r["score"] for r in greeks_card if r.get("score") is not None]
        if parts:
            composite["score"] = round(sum(parts) / len(parts), 4)

    return {
        "BarsScorecard": {"weights": BARS_WEIGHTS, "thresholds": BARS_THRESHOLDS, "rows": bars_card},
        "OptionsQuoteScorecard": {"weights": OPTIONS_WEIGHTS, "thresholds": OPTIONS_THRESHOLDS, "rows": options_card},
        "OptionsGreeksScorecard": {"weights": GREEKS_WEIGHTS, "thresholds": GREEKS_THRESHOLDS, "rows": greeks_card},
        "Composite": composite,
    }


def _market_hours_flag() -> str:
    # Simple deterministic flag from UTC weekday and US RTH hour window proxy.
    now = datetime.now(timezone.utc)
    return "likely_rth" if now.weekday() < 5 and 14 <= now.hour <= 21 else "likely_off_hours"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--greeks", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": list(LIQUID_ETF_UNIVERSE),
        "market_hours_flag": _market_hours_flag(),
        "skips": [],
    }

    try:
        config = load_config()
        secrets = load_secrets()
    except ConfigurationError as exc:
        payload["skips"].append(f"config/secrets unavailable: {exc}")
        payload.update(_build_scorecards([], [], args.greeks))
        out = Path(args.json_out) if args.json_out else Path("logs") / f"provider_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"SKIP: not configured ({exc})")
        print(f"Wrote {out}")
        return 0

    bar_rows, bar_skips = asyncio.run(_bars_rows(secrets))
    opt_rows, opt_skips = asyncio.run(_options_rows(config, secrets, args.duration, args.greeks))
    head_to_head = asyncio.run(_underlying_head_to_head(secrets))
    
    payload["skips"].extend(bar_skips + opt_skips)
    payload["bars_raw"] = bar_rows
    payload["options_raw"] = opt_rows
    payload["head_to_head"] = head_to_head
    payload.update(_build_scorecards(bar_rows, opt_rows, args.greeks))

    print("\nHead-to-Head Latency (Underlying):")
    for res in head_to_head:
        if "error" in res:
            print(f"  {res['provider']}: ERROR {res['error']}")
        else:
            print(f"  {res['provider']} ({res['type']}): p50={res['lag_p50'] or 0:.0f}ms p95={res['lag_p95'] or 0:.0f}ms events={res['events']}")

    print("\nBarsScorecard:")
    for row in payload["BarsScorecard"]["rows"]:
        print(f"  {row['provider']}: success={row['success_rate']:.2%} score={row['score']}")
    print("OptionsQuoteScorecard:")
    for row in payload["OptionsQuoteScorecard"]["rows"]:
        mqr = f"{row['missing_quote_rate']:.2%}" if row['missing_quote_rate'] is not None else "n/a"
        print(f"  {row['provider']}: missing={mqr} score={row['score']}")
    if args.greeks:
        print("OptionsGreeksScorecard:")
        for row in payload["OptionsGreeksScorecard"]["rows"]:
            gpr = f"{row['greeks_presence_rate']:.2%}" if row['greeks_presence_rate'] is not None else "n/a"
            print(f"  {row['provider']}: presence={gpr} score={row['score']}")
    print(f"Composite: {payload['Composite']}")

    out = Path(args.json_out) if args.json_out else Path("logs") / f"provider_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
