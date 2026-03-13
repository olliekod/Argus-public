"""
Argus System Verification
=========================

Comprehensive test to verify all components work correctly.
Uses data-source policy: bars from bars_primary (e.g. Alpaca); options from
options_snapshots_primary (Tastytrade) and optionally secondary (Public).
Does not assume Alpaca provides options data.

Run: python scripts/verify_system.py
      python scripts/verify_system.py --deep
"""

import sys
import asyncio
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def ok(msg):
    print(f"  {GREEN}[OK]{RESET} {msg}")

def fail(msg, error=None):
    print(f"  {RED}[FAIL]{RESET} {msg}")
    if error:
        print(f"        {error}")

def warn(msg):
    print(f"  {YELLOW}[WARN]{RESET} {msg}")

def info(msg):
    print(f"  {CYAN}[INFO]{RESET} {msg}")


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


def _contains_universe(config_symbols):
    symbols = {str(sym).upper() for sym in (config_symbols or [])}
    missing = [sym for sym in LIQUID_ETF_UNIVERSE if sym not in symbols]
    return missing

async def verify_system(deep: bool = False):
    """Run all verification checks."""
    print("=" * 60)
    print("ARGUS SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {"passed": 0, "failed": 0, "warnings": 0}
    
    # === 1. CORE IMPORTS ===
    print("1. Core Imports")
    print("-" * 40)
    
    try:
        from src.core.database import Database
        ok("Database")
        results["passed"] += 1
    except Exception as e:
        fail("Database", str(e))
        results["failed"] += 1
    
    try:
        from src.core.config import load_all_config
        ok("Config")
        results["passed"] += 1
    except Exception as e:
        fail("Config", str(e))
        results["failed"] += 1
    
    try:
        from src.core.economic_calendar import EconomicCalendar
        ok("Economic Calendar")
        results["passed"] += 1
    except Exception as e:
        fail("Economic Calendar", str(e))
        results["failed"] += 1
    
    try:
        from src.core.sentiment_collector import SentimentCollector
        ok("Sentiment Collector")
        results["passed"] += 1
    except Exception as e:
        fail("Sentiment Collector", str(e))
        results["failed"] += 1
    
    print()
    
    # === 2. CONNECTORS ===
    print("2. Connectors")
    print("-" * 40)
    
    try:
        from src.connectors.bybit_ws import BybitWebSocket
        ok("Bybit WebSocket")
        results["passed"] += 1
    except Exception as e:
        fail("Bybit WebSocket", str(e))
        results["failed"] += 1
    
    try:
        from src.connectors.deribit_client import DeribitClient
        ok("Deribit Client")
        results["passed"] += 1
    except Exception as e:
        fail("Deribit Client", str(e))
        results["failed"] += 1
    
    try:
        from src.connectors.yahoo_client import YahooFinanceClient
        ok("Yahoo Finance Client")
        results["passed"] += 1
    except Exception as e:
        fail("Yahoo Finance Client", str(e))
        results["failed"] += 1
    
    try:
        from src.connectors.ibit_options_client import IBITOptionsClient
        ok("Options Client (IBIT/BITO)")
        results["passed"] += 1
    except Exception as e:
        fail("Options Client", str(e))
        results["failed"] += 1
    
    print()
    
    # === 3. TASTYTRADE ===
    print("3. Tastytrade Verification")
    print("-" * 40)

    try:
        from src.core.config import load_config, load_secrets
        from src.connectors.tastytrade_rest import (
            RetryConfig,
            TastytradeError,
            TastytradeRestClient,
        )
        from src.connectors.tastytrade_oauth import TastytradeOAuthClient
        from src.core.options_normalize import normalize_tastytrade_nested_chain
    except Exception as e:
        fail("Tastytrade imports", str(e))
        results["failed"] += 1
    else:
        try:
            config = load_config()
            secrets = load_secrets()
        except Exception as e:
            fail("Load config/secrets for Tastytrade", str(e))
            results["failed"] += 1
        else:
            try:
                from scripts.tastytrade_health_audit import get_tastytrade_rest_client
                client = get_tastytrade_rest_client(config, secrets)
            except Exception as e:
                warn(f"Tastytrade credentials missing or auth failed; skipping auth/chain test: {e}")
                results["warnings"] += 1
            else:
                try:
                    start = time.perf_counter()
                    tasty_symbols = list(LIQUID_ETF_UNIVERSE) if deep else ["SPY"]
                    chain_fail = False
                    for tsym in tasty_symbols:
                        chain = client.get_nested_option_chains(tsym)
                        normalized = normalize_tastytrade_nested_chain(chain)
                        if normalized:
                            ok(
                                f"Tastytrade nested chain ({tsym}) ({len(normalized)} contracts, "
                                f"{(time.perf_counter() - start):.2f}s)"
                            )
                        else:
                            fail(f"Tastytrade nested chain normalization empty ({tsym})")
                            chain_fail = True
                    if chain_fail:
                        results["failed"] += 1
                    else:
                        results["passed"] += 1
                except TastytradeError as e:
                    fail("Tastytrade auth/chain", str(e))
                    results["failed"] += 1
                finally:
                    client.close()

            oauth_cfg = secrets.get("tastytrade_oauth2", {}) or {}
            client_id = oauth_cfg.get("client_id", "")
            client_secret = oauth_cfg.get("client_secret", "")
            refresh_token = oauth_cfg.get("refresh_token", "")
            if (
                _is_placeholder(client_id)
                or _is_placeholder(client_secret)
                or _is_placeholder(refresh_token)
            ):
                warn("Tastytrade OAuth refresh token missing; skipping OAuth test")
                results["warnings"] += 1
            else:
                try:
                    oauth_client = TastytradeOAuthClient(
                        client_id=client_id,
                        client_secret=client_secret,
                        refresh_token=refresh_token,
                    )
                    token_result = oauth_client.refresh_access_token()
                    ttl = (
                        f"{token_result.expires_in}s"
                        if token_result.expires_in is not None
                        else "unknown TTL"
                    )
                    ok(f"Tastytrade OAuth refresh ({ttl})")
                    results["passed"] += 1
                except Exception as e:
                    fail("Tastytrade OAuth refresh", str(e))
                    results["failed"] += 1

    print()

    # === 4. DETECTORS ===
    print("4. Detectors")
    print("-" * 40)
    
    try:
        from src.detectors.etf_options_detector import ETFOptionsDetector
        ok("ETF Options Detector (IBIT/BITO)")
        results["passed"] += 1
    except Exception as e:
        fail("ETF Options Detector", str(e))
        results["failed"] += 1
    
    print()
    
    # === 5. ANALYSIS ===
    print("5. Analysis Components")
    print("-" * 40)
    
    try:
        from src.analysis.paper_trader import PaperTrader
        ok("Paper Trader")
        results["passed"] += 1
    except Exception as e:
        fail("Paper Trader", str(e))
        results["failed"] += 1
    
    try:
        from src.analysis.trade_calculator import TradeCalculator
        ok("Trade Calculator")
        results["passed"] += 1
    except Exception as e:
        fail("Trade Calculator", str(e))
        results["failed"] += 1
    
    try:
        from src.analysis.backtester import StrategyBacktester
        ok("Backtester")
        results["passed"] += 1
    except Exception as e:
        fail("Backtester", str(e))
        results["failed"] += 1
    
    try:
        from src.analysis.greeks_engine import GreeksEngine
        ok("Greeks Engine")
        results["passed"] += 1
    except Exception as e:
        fail("Greeks Engine", str(e))
        results["failed"] += 1
    
    print()
    
    # === 6. ALERTS ===
    print("6. Alerts")
    print("-" * 40)
    
    try:
        from src.alerts.telegram_bot import TelegramBot
        ok("Telegram Bot")
        results["passed"] += 1
    except Exception as e:
        fail("Telegram Bot", str(e))
        results["failed"] += 1
    
    print()
    
    # === 7. ORCHESTRATOR ===
    print("7. Main Orchestrator")
    print("-" * 40)
    
    try:
        from src.orchestrator import ArgusOrchestrator
        ok("ArgusOrchestrator")
        results["passed"] += 1
    except Exception as e:
        fail("ArgusOrchestrator", str(e))
        results["failed"] += 1
    
    print()
    
    # === 8. CONFIG FILES ===
    print("8. Configuration Files")
    print("-" * 40)
    
    config_dir = Path(__file__).parent.parent / "config"
    
    if (config_dir / "secrets.yaml").exists():
        ok("secrets.yaml exists")
        results["passed"] += 1
    else:
        fail("secrets.yaml missing")
        results["failed"] += 1
    
    if (config_dir / "thresholds.yaml").exists():
        # Check for BITO config
        content = (config_dir / "thresholds.yaml").read_text(encoding='utf-8')
        if 'bito:' in content:
            ok("thresholds.yaml (IBIT + BITO configured)")
        else:
            ok("thresholds.yaml (IBIT only)")
        results["passed"] += 1
    else:
        warn("thresholds.yaml missing (using defaults)")
        results["warnings"] += 1

    try:
        from src.core.config import load_config
        cfg = load_config()
        alpaca_missing = _contains_universe(cfg.get("exchanges", {}).get("alpaca", {}).get("symbols", []))
        yahoo_missing = _contains_universe(cfg.get("exchanges", {}).get("yahoo", {}).get("symbols", []))
        if not alpaca_missing:
            ok("Alpaca symbol config includes full liquid ETF universe")
            results["passed"] += 1
        else:
            fail("Alpaca symbol config missing universe tickers", str(alpaca_missing))
            results["failed"] += 1
        if not yahoo_missing:
            ok("Yahoo symbol config includes full liquid ETF universe")
            results["passed"] += 1
        else:
            fail("Yahoo symbol config missing universe tickers", str(yahoo_missing))
            results["failed"] += 1
    except Exception as e:
        fail("Universe configuration check", str(e))
        results["failed"] += 1
    
    print()
    
    # === 9. DATABASE ===
    print("9. Database")
    print("-" * 40)
    
    db_path = Path(__file__).parent.parent / "data" / "argus.db"
    if db_path.exists():
        ok(f"Database exists ({db_path.stat().st_size / 1024:.1f} KB)")
        results["passed"] += 1
        
        # Check paper trades count
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM paper_trades")
            count = cursor.fetchone()[0]
            conn.close()
            if count > 0:
                info(f"Paper trades logged: {count}")
            else:
                info("No paper trades yet")
        except:
            pass
    else:
        warn("Database doesn't exist yet (will be created on first run)")
        results["warnings"] += 1
    
    print()
    
    # === 10. LIVE DATA TESTS ===
    print("10. Live Data Tests")
    print("-" * 40)
    
    # Test IBIT Options Client
    try:
        from src.connectors.ibit_options_client import IBITOptionsClient
        client = IBITOptionsClient(symbol="IBIT")
        price = client.get_current_price()
        if price > 0:
            ok(f"IBIT Price: ${price:.2f}")
            results["passed"] += 1
        else:
            warn("IBIT price unavailable (market closed?)")
            results["warnings"] += 1
    except Exception as e:
        fail("IBIT Price", str(e))
        results["failed"] += 1
    
    # Test BITO Options Client
    try:
        from src.connectors.ibit_options_client import IBITOptionsClient
        client = IBITOptionsClient(symbol="BITO")
        price = client.get_current_price()
        if price > 0:
            ok(f"BITO Price: ${price:.2f}")
            results["passed"] += 1
        else:
            warn("BITO price unavailable (market closed?)")
            results["warnings"] += 1
    except Exception as e:
        fail("BITO Price", str(e))
        results["failed"] += 1

    # Live DXLink Options Probe
    try:
        from src.connectors.tastytrade_oauth import TastytradeOAuthClient
        from src.connectors.tastytrade_rest import TastytradeRestClient
        from src.connectors.tastytrade_streamer import TastytradeStreamer
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        conf = load_config()
        sec = load_secrets()
        tt_sec = sec.get("tastytrade_oauth2", {})
        if not tt_sec or _is_placeholder(tt_sec.get("client_id") or "") or _is_placeholder(tt_sec.get("refresh_token") or ""):
            raise RuntimeError("tastytrade_oauth2 (client_id, client_secret, refresh_token) required for DXLink probe")
        oauth = TastytradeOAuthClient(tt_sec["client_id"], tt_sec["client_secret"], tt_sec["refresh_token"])
        token = oauth.refresh_access_token().access_token
        # Get DXLink quote token (OAuth only)
        quote_resp = requests.get("https://api.tastytrade.com/api-quote-tokens", headers={"Authorization": f"Bearer {token}"}, timeout=10)
        quote_resp.raise_for_status()
        quote_data = quote_resp.json()["data"]
        tt_config = conf.get("tastytrade", {})
        retry_cfg = tt_config.get("retries", {})
        from src.connectors.tastytrade_rest import RetryConfig
        rest = TastytradeRestClient(
            environment=tt_config.get("environment", "live"),
            timeout_seconds=tt_config.get("timeout_seconds", 20),
            retries=RetryConfig(
                max_attempts=retry_cfg.get("max_attempts", 3),
                backoff_seconds=retry_cfg.get("backoff_seconds", 1.0),
                backoff_multiplier=retry_cfg.get("backoff_multiplier", 2.0),
            ),
            oauth_access_token=token,
        )
        chain = rest.get_nested_option_chains("IBIT")
        norm = normalize_tastytrade_nested_chain(chain)
        rest.close()
        
        sample_syms = [c["meta"].get("streamer_symbol") or c["option_symbol"] for c in norm if c["expiry"] >= datetime.now().date().isoformat()][:5]
        
        streamer = TastytradeStreamer(quote_data["dxlink-url"], quote_data["token"], ["IBIT"] + sample_syms)
        events = await streamer.run_for(5.0)
        
        received = len({e.event_symbol for e in events})
        if received > 0:
            ok(f"DXLink Live Probe: received data for {received} symbols (IBIT + options)")
            results["passed"] += 1
        else:
            fail("DXLink Live Probe: zero events received in 5s")
            results["failed"] += 1
    except Exception as e:
        fail("DXLink Live Probe", str(e))
        results["failed"] += 1
    
        # Test Alpaca bar pull (SPY default, universe with --deep)
    try:
        from src.core.config import load_config, load_secrets
        from src.connectors.alpaca_client import AlpacaDataClient
        from src.core.bus import EventBus
        cfg = load_config()
        secrets = load_secrets()
        key = secrets.get("alpaca", {}).get("api_key")
        sec = secrets.get("alpaca", {}).get("api_secret")
        if _is_placeholder(key or "") or _is_placeholder(sec or ""):
            warn("Alpaca credentials missing; skipping SPY bars pull")
            results["warnings"] += 1
        else:
            test_symbols = list(LIQUID_ETF_UNIVERSE) if deep else ["SPY"]
            c = AlpacaDataClient(api_key=key, api_secret=sec, symbols=test_symbols, event_bus=EventBus(), poll_interval=60)
            ok_all = True
            for tsym in test_symbols:
                bars = await c.fetch_bars(tsym, limit=1)
                if bars:
                    ok(f"Alpaca {tsym} bars pull succeeded")
                else:
                    fail(f"Alpaca {tsym} bars pull returned empty")
                    ok_all = False
            await c.close()
            if ok_all:
                results["passed"] += 1
            else:
                results["failed"] += 1
    except Exception as e:
        fail("Alpaca SPY bars pull", str(e))
        results["failed"] += 1

        # Test Yahoo quote pull (SPY default, universe with --deep)
    try:
        from src.connectors.yahoo_client import YahooFinanceClient
        test_symbols = list(LIQUID_ETF_UNIVERSE) if deep else ["SPY"]
        y = YahooFinanceClient(symbols=test_symbols)
        ok_all = True
        for tsym in test_symbols:
            quote = await y.get_quote(tsym)
            if quote and quote.get("price"):
                ok(f"Yahoo {tsym} quote pull succeeded")
            else:
                fail(f"Yahoo {tsym} quote pull returned empty")
                ok_all = False
        await y.close()
        if ok_all:
            results["passed"] += 1
        else:
            results["failed"] += 1
    except Exception as e:
        fail("Yahoo SPY quote pull", str(e))
        results["failed"] += 1

    # Test sentiment
    try:
        from src.core.sentiment_collector import SentimentCollector
        collector = SentimentCollector()
        data = await collector.get_sentiment()
        if data and data.fear_greed_value > 0:
            ok(f"Fear & Greed: {data.fear_greed_value} ({data.fear_greed_label})")
            results["passed"] += 1
        else:
            warn("Fear & Greed returned empty")
            results["warnings"] += 1
    except Exception as e:
        fail("Fear & Greed API", str(e))
        results["failed"] += 1
    
    # Test economic calendar
    try:
        from src.core.economic_calendar import EconomicCalendar
        cal = EconomicCalendar()
        is_blackout, reason = cal.is_blackout_period()
        ok(f"Economic Calendar: {'BLACKOUT - ' + reason if is_blackout else 'Clear'}")
        results["passed"] += 1
    except Exception as e:
        fail("Economic Calendar", str(e))
        results["failed"] += 1
    
    print()
    
    # === SUMMARY ===
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = results["passed"] + results["failed"]
    print(f"  Passed:   {GREEN}{results['passed']}/{total}{RESET}")
    print(f"  Failed:   {RED}{results['failed']}{RESET}")
    print(f"  Warnings: {YELLOW}{results['warnings']}{RESET}")
    print()
    
    if results["failed"] == 0:
        print(f"{GREEN}System is ready for paper trading!{RESET}")
        print()
        print("Configured Tickers:")
        print("  • IBIT, BITO")
        print("  • Liquid ETF universe: " + ", ".join(LIQUID_ETF_UNIVERSE))
        print()
        print("Next steps:")
        print("  1. Double-click 'Start Argus.vbs' to start monitoring")
        print("  2. Check Telegram for alerts")
        print("  3. Run 'python scripts\\paper_performance.py' to see trades")
    else:
        print(f"{RED}Some components failed. Please review errors above.{RESET}")
    
    print()
    print("=" * 60)
    
    return results["failed"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argus system verification")
    parser.add_argument("--deep", action="store_true", help="Run full-universe provider checks")
    args = parser.parse_args()
    success = asyncio.run(verify_system(deep=args.deep))
    sys.exit(0 if success else 1)
