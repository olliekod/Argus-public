import sys
import asyncio
import time
import argparse
import sqlite3
import requests
from pathlib import Path
from datetime import date, datetime, timezone
from statistics import median
from typing import Any, Tuple, Literal

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectors.tastytrade_oauth import TastytradeOAuthClient
from src.connectors.tastytrade_rest import RetryConfig, TastytradeError, TastytradeRestClient
from src.connectors.tastytrade_streamer import TastytradeStreamer, StreamerConfig
from src.core.config import ConfigurationError, load_config, load_secrets
from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE
from src.core.options_normalize import normalize_tastytrade_nested_chain
from src.connectors.tastytrade_dxlink_parser import QuoteEvent, GreeksEvent
from src.analysis.greeks_engine import GreeksEngine


def _ensure_snapshot_table(db_path: Path) -> None:
    """Create option_quote_snapshots table and indexes if missing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS option_quote_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                provider TEXT NOT NULL,
                underlying TEXT NOT NULL,
                option_symbol TEXT NOT NULL,
                expiry TEXT,
                strike REAL,
                right TEXT,
                bid REAL,
                ask REAL,
                mid REAL,
                event_ts REAL,
                recv_ts REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_option_quote_snapshots_underlying_ts "
            "ON option_quote_snapshots(underlying, ts_utc)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_option_quote_snapshots_symbol_ts "
            "ON option_quote_snapshots(option_symbol, ts_utc)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_option_quote_snapshots_provider "
            "ON option_quote_snapshots(provider)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_option_quote_snapshots_contract "
            "ON option_quote_snapshots(underlying, expiry, strike, right)"
        )
        conn.commit()
    finally:
        conn.close()


def _prune_snapshots_sql(days: int) -> tuple[str, tuple]:
    return (
        "DELETE FROM option_quote_snapshots WHERE ts_utc < datetime('now', ?)",
        (f"-{days} days",),
    )


def select_spot(
    dxlink_quote: dict | None,
    cli_spot: float | None,
    strike_medians: list[float],
) -> dict[str, Any]:
    """Select spot using dxlink -> CLI -> median strike fallback."""
    if dxlink_quote:
        bid = dxlink_quote.get("bidPrice", dxlink_quote.get("bid"))
        ask = dxlink_quote.get("askPrice", dxlink_quote.get("ask"))
        if bid is not None and ask is not None:
            return {"spot_source": "dxlink", "spot_value": (float(bid) + float(ask)) / 2.0}
        if bid is not None:
            return {"spot_source": "dxlink", "spot_value": float(bid)}
        if ask is not None:
            return {"spot_source": "dxlink", "spot_value": float(ask)}

    if cli_spot is not None:
        return {"spot_source": "cli", "spot_value": float(cli_spot)}

    fallback = float(median(strike_medians)) if strike_medians else 0.0
    return {
        "spot_source": "median_strike",
        "spot_value": fallback,
        "warning": "WARNING: DXLink/CLI spot unavailable; using median strike fallback.",
    }


def _select_sampled_contracts(
    chain: list[dict[str, Any]],
    spot_value: float | None,
    now_utc: datetime,
    expiry_count: int = 2,
    strike_window: int = 5,
    max_contracts: int = 40,
) -> list[dict[str, Any]]:
    """Deterministically sample contracts near spot from nearest expiries."""
    if not chain:
        return []

    today = now_utc.date()
    candidates: list[dict[str, Any]] = []
    for c in chain:
        expiry_raw = c.get("expiry")
        strike = c.get("strike")
        symbol = c.get("option_symbol")
        if not expiry_raw or strike is None or not symbol:
            continue
        try:
            expiry_dt = datetime.fromisoformat(str(expiry_raw)).date()
            strike_val = float(strike)
        except (TypeError, ValueError):
            continue
        if expiry_dt < today:
            continue
        candidates.append({**c, "_expiry_dt": expiry_dt, "_strike_val": strike_val})

    if not candidates:
        return []

    expiries = sorted({c["_expiry_dt"] for c in candidates})[: max(0, int(expiry_count))]
    if not expiries:
        return []

    if spot_value is None:
        spot_used = float(median([c["_strike_val"] for c in candidates]))
    else:
        spot_used = float(spot_value)

    selected: list[dict[str, Any]] = []
    for expiry in expiries:
        exp_contracts = [c for c in candidates if c["_expiry_dt"] == expiry]
        strikes = sorted({c["_strike_val"] for c in exp_contracts})
        if not strikes:
            continue

        center_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_used))
        low = max(0, center_idx - max(0, int(strike_window)))
        high = min(len(strikes), center_idx + max(0, int(strike_window)) + 1)
        keep = set(strikes[low:high])
        selected.extend([c for c in exp_contracts if c["_strike_val"] in keep])

    right_order = {"C": 0, "P": 1}
    selected.sort(
        key=lambda c: (
            c["_expiry_dt"],
            c["_strike_val"],
            right_order.get(str(c.get("right", "")).upper(), 9),
            c.get("option_symbol", ""),
        )
    )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in selected:
        sym = str(c.get("option_symbol", ""))
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out = dict(c)
        out.pop("_expiry_dt", None)
        out.pop("_strike_val", None)
        deduped.append(out)
        if len(deduped) >= max_contracts:
            break

    return deduped


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


def _parse_expiry(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        # Handle '2026-02-14Z' or '2026-02-14'
        raw = value[:-1] if value.endswith("Z") else value
        try:
            return datetime.fromisoformat(raw.split("T")[0]).date()
        except ValueError:
            return None
    return None


def _compute_spread_bps(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid * 10000


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def audit_nested_chain(raw: dict[str, Any]) -> dict[str, Any]:
    """Analyze the raw nested chain response for health."""
    data = raw.get("data")
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        chains = data.get("items", [])
    elif isinstance(data, dict) and isinstance(data.get("expirations"), list):
        chains = [data]
    else:
        chains = []

    today = date.today()
    total_expirations = 0
    expired = 0
    strike_counts: list[int] = []
    call_count = 0
    put_count = 0
    missing_strikes = 0

    for chain in chains:
        expirations = chain.get("expirations") or []
        total_expirations += len(expirations)
        for exp in expirations:
            expiry_str = exp.get("expiration-date") or exp.get("date")
            expiry = _parse_expiry(expiry_str)
            if expiry and expiry < today:
                expired += 1
            
            strikes = exp.get("strikes") or exp.get("strike-prices") or []
            strike_counts.append(len(strikes))
            for s in strikes:
                has_call = bool(s.get("call"))
                has_put = bool(s.get("put"))
                if has_call: call_count += 1
                if has_put: put_count += 1
                if has_call != has_put:
                    missing_strikes += 1

    return {
        "expirations": total_expirations,
        "expired": expired,
        "strike_counts": strike_counts,
        "calls": call_count,
        "puts": put_count,
        "missing_strikes": missing_strikes
    }


def _load_tasty_client(config: dict[str, Any], secrets: dict[str, Any]) -> TastytradeRestClient:
    """Create TastytradeRestClient with session (username/password). Caller must call login()."""
    tasty_secrets = secrets.get("tastytrade", {})
    username = tasty_secrets.get("username", "")
    password = tasty_secrets.get("password", "")
    if _is_placeholder(username) or _is_placeholder(password):
        raise RuntimeError("SKIP: not configured (tastytrade.username/password).")

    tt_config = config.get("tastytrade", {})
    retry_cfg = tt_config.get("retries", {})
    client = TastytradeRestClient(
        username=username,
        password=password,
        environment=tt_config.get("environment", "live"),
        timeout_seconds=tt_config.get("timeout_seconds", 20),
        retries=RetryConfig(
            max_attempts=retry_cfg.get("max_attempts", 3),
            backoff_seconds=retry_cfg.get("backoff_seconds", 1.0),
            backoff_multiplier=retry_cfg.get("backoff_multiplier", 2.0),
        ),
    )
    return client


def get_tastytrade_rest_client(config: dict[str, Any], secrets: dict[str, Any]) -> TastytradeRestClient:
    """Return an authenticated TastytradeRestClient. Prefers OAuth when configured (session auth is deprecated)."""
    oauth_cfg = secrets.get("tastytrade_oauth2", {}) or {}
    cid = oauth_cfg.get("client_id", "")
    csec = oauth_cfg.get("client_secret", "")
    ref = oauth_cfg.get("refresh_token", "")
    if not _is_placeholder(cid) and not _is_placeholder(csec) and not _is_placeholder(ref):
        try:
            oauth = TastytradeOAuthClient(client_id=cid, client_secret=csec, refresh_token=ref)
            token = oauth.refresh_access_token().access_token
            tt_config = config.get("tastytrade", {})
            retry_cfg = tt_config.get("retries", {})
            client = TastytradeRestClient(
                environment=tt_config.get("environment", "live"),
                timeout_seconds=tt_config.get("timeout_seconds", 20),
                retries=RetryConfig(
                    max_attempts=retry_cfg.get("max_attempts", 3),
                    backoff_seconds=retry_cfg.get("backoff_seconds", 1.0),
                    backoff_multiplier=retry_cfg.get("backoff_multiplier", 2.0),
                ),
                oauth_access_token=token,
            )
            return client
        except Exception:
            pass
    client = _load_tasty_client(config, secrets)
    client.login()
    return client


def audit_oauth(secrets: dict[str, Any]) -> dict[str, Any]:
    oauth_cfg = secrets.get("tastytrade_oauth2", {}) or {}
    client_id = oauth_cfg.get("client_id", "")
    client_secret = oauth_cfg.get("client_secret", "")
    refresh_token = oauth_cfg.get("refresh_token", "")
    if _is_placeholder(client_id) or _is_placeholder(client_secret) or _is_placeholder(refresh_token):
        raise RuntimeError("SKIP: not configured (tastytrade_oauth2 client_id/client_secret/refresh_token).")
    client = TastytradeOAuthClient(client_id=client_id, client_secret=client_secret, refresh_token=refresh_token)
    start = time.perf_counter()
    token_result = client.refresh_access_token()
    return {
        "access_token": token_result.access_token,
        "latency_s": round(time.perf_counter() - start, 3),
    }


async def run_quotes_audit(
    symbol: str,
    normalized_contracts: list[dict[str, Any]],
    access_token: str,
    duration: int,
    greeks: bool = False
) -> dict[str, Any]:
    """Run a live quotes probe using TastytradeStreamer."""
    # 1. Get DXLink token
    quote_resp = requests.get(
        "https://api.tastytrade.com/api-quote-tokens",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20
    )
    quote_resp.raise_for_status()
    quote_data = quote_resp.json().get("data", {})
    dxlink_url = quote_data.get("dxlink-url")
    dxlink_token = quote_data.get("token")

    print(f"DXLink URL: {dxlink_url}")

    # 2. Sample contracts for subscription
    # We take the nearest 2 expiries, ±5 strikes around an estimated midpoint
    now_utc = datetime.now(timezone.utc)
    valid_expirations = sorted({c["expiry"] for c in normalized_contracts if c["expiry"] >= now_utc.date().isoformat()})[:2]
    
    # Estimate midpoint strike
    all_strikes = sorted([c["strike"] for c in normalized_contracts if c["strike"] is not None])
    est_spot = all_strikes[len(all_strikes) // 2] if all_strikes else 0
    
    print(f"Fetching spot for {symbol} via DXLink...")
    
    # 3. Initialize Streamer for single-shot probe
    # First probe: just the underlying to get a fresh spot
    streamer = TastytradeStreamer(
        dxlink_url=dxlink_url,
        token=dxlink_token,
        symbols=[symbol],
        event_types=["Quote"]
    )
    
    events = await streamer.run_for(3.0)
    underlying_quotes = [e for e in events if isinstance(e, QuoteEvent) and e.event_symbol == symbol]
    
    if underlying_quotes:
        q = underlying_quotes[-1]
        print(f"DEBUG: Underlying Quote bid={q.bid_price} ask={q.ask_price} mid={(q.bid_price+q.ask_price)/2.0 if q.bid_price and q.ask_price else None}")
        mid = (q.bid_price + q.ask_price) / 2.0 if q.bid_price and q.ask_price else q.bid_price
        if mid: est_spot = mid
        print(f"Spot (DXLink): {est_spot:.4f}")
    else:
        print("Warning: Could not fetch spot via DXLink; using median strike fallback.")

    # 4. Filter sampled contracts for the real probe
    sampled = [
        c for c in normalized_contracts 
        if c["expiry"] in valid_expirations 
        and abs(c["strike"] - est_spot) < (est_spot * 0.1) # 10% window
    ]
    # Further limit to ±5 strikes per side
    contracts_by_streamer = {
        (c["meta"].get("streamer_symbol") or c["option_symbol"]): c 
        for c in sampled
    }
    sampled_symbols = list(contracts_by_streamer.keys())[:40]
    
    print(f"Sampled {len(sampled_symbols)} option symbols for DXLink probe.")
    
    # 5. Full probe: Underlying + Options + (Optional) Greeks
    full_streamer = TastytradeStreamer(
        dxlink_url=dxlink_url,
        token=dxlink_token,
        symbols=[symbol] + sampled_symbols,
        event_types=["Quote"] + (["Greeks"] if greeks else [])
    )
    
    start_time = time.perf_counter()
    events = await full_streamer.run_for(duration)
    handshake_latency = time.perf_counter() - start_time - duration # Heuristic
    
    # 6. Analyze received events
    quotes_received = {}
    greeks_received = {}
    spreads = []
    observed_lags = []
    
    # Initialize Greeks engine for derived stats
    engine = GreeksEngine(auto_refresh_rate=False) # Use default 4.5
    derived_ivs = []
    provider_ivs = []
    
    debug_quote_printed = False
    debug_greeks_printed = False
    
    for e in events:
        if isinstance(e, QuoteEvent):
            norm_sym = " ".join(e.event_symbol.split())
            quotes_received[norm_sym] = e
            spread = _compute_spread_bps(e.bid_price, e.ask_price)
            if spread is not None: spreads.append(spread)
            
            # Observed Lag (Receipt - Exchange)
            exch_ts = max(e.bid_time or 0, e.ask_time or 0)
            
            # Requested Debug Print (One-shot)
            if not debug_quote_printed and e.event_symbol != symbol:
                debug_quote_printed = True
                rx_dt = datetime.fromtimestamp(e.receipt_time / 1000.0, tz=timezone.utc)
                ex_dt = datetime.fromtimestamp(exch_ts / 1000.0, tz=timezone.utc) if exch_ts > 0 else None
                print(f"\nDEBUG: LAG VERIFICATION (Quote)")
                print(f"  Symbol: {e.event_symbol}")
                print(f"  Event TS (raw): {exch_ts} ({ex_dt.strftime('%H:%M:%S.%f')[:-3] if ex_dt else 'N/A'} UTC)")
                print(f"  Receipt TS:     {e.receipt_time} ({rx_dt.strftime('%H:%M:%S.%f')[:-3]} UTC)")
                print(f"  Computed Lag:   {e.receipt_time - exch_ts if exch_ts > 0 else 'n/a'}ms")

            if exch_ts > 0 and e.receipt_time:
                lag = e.receipt_time - exch_ts
                if 0 < lag < 300000: # allow up to 300s for off-hours/stale
                    observed_lags.append(lag)

        elif isinstance(e, GreeksEvent):
            norm_sym = " ".join(e.event_symbol.split())
            greeks_received[norm_sym] = e
            if e.volatility: provider_ivs.append(e.volatility * 100)
            
            # Greeks Lag
            if e.timestamp and e.receipt_time:
                lag = e.receipt_time - e.timestamp
                
                # Requested Debug Print (One-shot)
                if not debug_greeks_printed:
                    debug_greeks_printed = True
                    rx_dt = datetime.fromtimestamp(e.receipt_time / 1000.0, tz=timezone.utc)
                    ex_dt = datetime.fromtimestamp(e.timestamp / 1000.0, tz=timezone.utc)
                    print(f"\nDEBUG: LAG VERIFICATION (Greeks)")
                    print(f"  Symbol: {e.event_symbol}")
                    print(f"  Event TS (raw): {e.timestamp} ({ex_dt.strftime('%H:%M:%S.%f')[:-3]} UTC)")
                    print(f"  Receipt TS:     {e.receipt_time} ({rx_dt.strftime('%H:%M:%S.%f')[:-3]} UTC)")
                    print(f"  Computed Lag:   {lag}ms")

                if 0 < lag < 300000:
                    observed_lags.append(lag)

    # Calculate derived Greeks for quotes we got
    for sym, q in quotes_received.items():
        if sym == symbol: continue # skip underlying
        if sym not in sampled_symbols: continue
        
        c = contracts_by_streamer.get(sym)
        if not c: continue
        
        # Calculate derived IV
        mid = (q.bid_price + q.ask_price) / 2.0 if q.bid_price and q.ask_price else None
        if mid and est_spot > 0:
            # Simple DTE calc
            expiry_dt = datetime.fromisoformat(c["expiry"])
            dte = (expiry_dt.date() - now_utc.date()).days
            T = max(1/365, dte / 365)
            
            opt_type: Literal["call", "put"] = "call" if c["right"] == "C" else "put"
            iv, src = engine.implied_volatility(
                mid, est_spot, c["strike"], T, opt_type
            )
            if iv: 
                derived_ivs.append(iv * 100)

    norm_samples = [" ".join(s.split()) for s in sampled_symbols]
    received_count = len([s for s in norm_samples if s in quotes_received])
    
    return {
        "received": received_count,
        "total": len(sampled_symbols),
        "missing_rate": 1.0 - (received_count / len(sampled_symbols)) if sampled_symbols else 1.0,
        "spread_p50": _percentile(spreads, 50),
        "spread_p95": _percentile(spreads, 95),
        "underlying_ok": symbol in [s for s in quotes_received if s == symbol],
        "handshake_latency": max(0.1, handshake_latency),
        "greeks_presence": len(greeks_received) / len(sampled_symbols) if sampled_symbols else 0,
        "iv_derived_p50": _percentile(derived_ivs, 50),
        "iv_provider_p50": _percentile(provider_ivs, 50),
        "lag_p50": _percentile(observed_lags, 50),
        "lag_p95": _percentile(observed_lags, 95)
    }


async def main():
    parser = argparse.ArgumentParser(description="Tastytrade health audit.")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol")
    parser.add_argument("--universe", action="store_true", help="Audit the entire Liquid ETF universe")
    parser.add_argument("--quotes", action="store_true", help="Run DXLink quote probe")
    parser.add_argument("--greeks", action="store_true", help="Attempt Greeks events")
    parser.add_argument("--duration", type=int, default=15)
    args = parser.parse_args()

    symbols = LIQUID_ETF_UNIVERSE if args.universe else [args.symbol.upper()]
    
    # Configure logging
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Quiet noisy trackers
    logging.getLogger("argus.dxlink").setLevel(logging.INFO)
    
    print(f"Starting audit for {len(symbols)} symbols...")
    
    pass_count = 0
    total_count = len(symbols)

    try:
        config = load_config()
        secrets = load_secrets()
        
        # 1. REST Client Audit (perform login once)
        client = _load_tasty_client(config, secrets)
        client.login()
        
        # 2. OAuth Audit (once)
        print("\n--- OAuth Refresh ---")
        oauth = audit_oauth(secrets)
        print(f"Access token refresh OK (TTL 900s, {oauth['latency_s']}s)")
        
        for symbol in symbols:
            print(f"\n{'='*60}")
            print(f" AUDITING: {symbol}")
            print(f"{'='*60}")
            
            try:
                # Chain Audit
                chain = client.get_nested_option_chains(symbol)
                normalized = normalize_tastytrade_nested_chain(chain)
                stats = audit_nested_chain(chain)
                
                print("\n--- Chain Summary ---")
                print(f"Expirations: {stats['expirations']}")
                print(f"Contracts: {stats['calls']} calls, {stats['puts']} puts")
                
                # Live Probes
                if args.quotes or args.greeks:
                    probe = await run_quotes_audit(symbol, normalized, oauth["access_token"], args.duration, args.greeks)
                    
                    print(f"\nQuotes received: {probe['received']}/{probe['total']}")
                    print(f"Missing quote rate: {probe['missing_rate']:.2%}")
                    print(f"Underlying quote OK: {probe['underlying_ok']}")
                    print(f"Greeks presence: {probe['greeks_presence']:.2%}")
                    
                    print(f"IV Provider (p50): {probe['iv_provider_p50']:.1f}%" if probe['iv_provider_p50'] else "IV Provider (p50): n/a")
                    print(f"IV Derived (p50): {probe['iv_derived_p50']:.1f}%" if probe['iv_derived_p50'] else "IV Derived (p50): n/a")
                    
                    if probe['lag_p50']:
                        print(f"Observed Lag (p50): {probe['lag_p50']:.0f}ms")
                        print(f"Observed Lag (p95): {probe['lag_p95']:.0f}ms")
                    else:
                        print("Observed Lag (p50): n/a (no exchange timestamps provided)")
                        print("Observed Lag (p95): n/a (no exchange timestamps provided)")
                    
                    pass_audit = probe['underlying_ok'] and (probe['received'] > 0 or probe['total'] == 0)
                    print(f"\nConclusion: {'PASS' if pass_audit else 'FAIL'}")
                    if pass_audit: pass_count += 1
                else:
                    pass_count += 1 # If only checking chain retrieval

            except Exception as e:
                print(f"Error auditing {symbol}: {e}")
                
        print(f"\n{'='*60}")
        print(f"FINAL RESULT: {pass_count}/{total_count} PASSED")
        print(f"{'='*60}")
        return 0 if pass_count == total_count else 1
            
        return 0

    except Exception as e:
        print(f"\nConclusion: FAIL (Error: {e})")
        return 1
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
