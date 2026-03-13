"""
Smoke tests for the Argus Kalshi runner.

Tests exercise the startup / discovery / wiring / shutdown path with
mocked REST responses — no real network calls are made.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from argus_kalshi.btc_window_engine import BtcWindowEngine
from argus_kalshi.bus import Bus
from argus_kalshi.config import KalshiConfig
from argus_kalshi.kalshi_execution import ExecutionEngine
from argus_kalshi.kalshi_markets import MarketDiscovery
from argus_kalshi.kalshi_probability import ProbabilityEngine
from argus_kalshi.kalshi_strategy import StrategyEngine
from argus_kalshi.market_selectors import (
    filter_btc_15min_markets,
    is_15min_window,
    is_btc_related,
)
from argus_kalshi.models import BtcMidPrice, SelectedMarkets
from argus_kalshi.runner import MockBtcFeed, _build_configs, _rank_active_near_money
from argus_kalshi.models import MarketMetadata

# ---------------------------------------------------------------------------
#  _build_configs tests
# ---------------------------------------------------------------------------

def test_build_config_merges_secrets():
    settings = {"argus_kalshi": {"bankroll_usd": 1234.0}}
    secrets = {"kalshi": {"key_id": "secret_key", "private_key_path": "/path"}}

    configs = _build_configs(settings, secrets)
    cfg = configs[0]
    assert cfg.bankroll_usd == 1234.0
    assert cfg.kalshi_key_id == "secret_key"
    assert cfg.kalshi_private_key_path == "/path"


def test_build_config_enforces_safe_defaults():
    configs = _build_configs({}, {})
    cfg = configs[0]
    assert cfg.dry_run is True
    assert cfg.ws_trading_enabled is False


def test_build_config_does_not_override_explicit_dry_run_false():
    """If the user explicitly sets dry_run: false, _build_configs uses it."""
    settings = {"argus_kalshi": {"dry_run": False}}
    configs = _build_configs(settings, {})
    cfg = configs[0]
    assert cfg.dry_run is False


# ---------------------------------------------------------------------------
#  Test market data — deterministic fixtures
# ---------------------------------------------------------------------------

# 15m BTC uses KXBTCM (M = minute); KXBTCD is hourly (D = daily).
MOCK_MARKETS = [
    {
        "ticker": "KXBTCM-26FEB25-B65000",
        "title": "Bitcoin 15-Minute Price: $65,000 or above?",
        "subtitle": "CF Benchmarks BRTI 15-min window",
        "status": "open",
        "series_ticker": "KXBTCM",
        "event_ticker": "KXBTCM-26FEB25",
        "strike_price": 65000.0,
        "expiration_time": "2099-12-31T16:15:00Z",
        "last_trade_time": "2099-12-31T16:15:00Z",
        "contract_type": "binary",
    },
    {
        "ticker": "KXBTCM-26FEB25-B66000",
        "title": "Bitcoin 15-Minute Price: $66,000 or above?",
        "subtitle": "CF Benchmarks BRTI 15-min window",
        "status": "open",
        "series_ticker": "KXBTCM",
        "event_ticker": "KXBTCM-26FEB25",
        "strike_price": 66000.0,
        "expiration_time": "2099-12-31T16:30:00Z",
        "last_trade_time": "2099-12-31T16:30:00Z",
        "contract_type": "binary",
    },
    {
        "ticker": "KXOTHER-26FEB25",
        "title": "Some other market entirely",
        "subtitle": "",
        "status": "open",
        "series_ticker": "KXOTHER",
        "event_ticker": "KXOTHER-26FEB25",
        "strike_price": 100.0,
        "expiration_time": "2099-12-31T17:00:00Z",
        "last_trade_time": "2099-12-31T17:00:00Z",
        "contract_type": "binary",
    },
]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides: object) -> KalshiConfig:
    """Build a KalshiConfig suitable for testing (no real credentials)."""
    defaults = dict(
        kalshi_key_id="test-key",
        kalshi_private_key_path="/dev/null",
        dry_run=True,
        ws_trading_enabled=False,
        series_filter="KXBTCM",
        enable_clock_offset_calibration=False,
        paper_slippage_cents=0,
    )
    defaults.update(overrides)
    return KalshiConfig(**defaults)


def _mock_rest() -> AsyncMock:
    """Return a mocked KalshiRestClient."""
    rest = AsyncMock()
    rest.start = AsyncMock()
    rest.close = AsyncMock()

    async def _paginate(*args, **kwargs):
        for m in MOCK_MARKETS:
            yield m

    rest.paginate = _paginate
    rest.get_market = AsyncMock(
        side_effect=lambda t: {
            "market": next(
                (m for m in MOCK_MARKETS if m["ticker"] == t), MOCK_MARKETS[0]
            )
        }
    )
    return rest


# ---------------------------------------------------------------------------
#  market_selectors unit tests
# ---------------------------------------------------------------------------

def test_is_btc_related_positive():
    assert is_btc_related(MOCK_MARKETS[0]) is True


def test_is_btc_related_negative():
    assert is_btc_related(MOCK_MARKETS[2]) is False


def test_is_15min_window_positive():
    assert is_15min_window(MOCK_MARKETS[0]) is True


def test_is_15min_window_negative():
    assert is_15min_window(MOCK_MARKETS[2]) is False


def test_filter_btc_15min_markets_selects_correctly():
    result = filter_btc_15min_markets(MOCK_MARKETS)
    tickers = [m["ticker"] for m in result]
    assert "KXBTCM-26FEB25-B65000" in tickers
    assert "KXBTCM-26FEB25-B66000" in tickers
    assert "KXOTHER-26FEB25" not in tickers


def test_filter_btc_15min_markets_with_title_regex():
    result = filter_btc_15min_markets(
        MOCK_MARKETS, title_regex=r"\$65,000"
    )
    assert len(result) == 1
    assert result[0]["ticker"] == "KXBTCM-26FEB25-B65000"


def test_filter_skips_closed_markets():
    closed = [dict(MOCK_MARKETS[0], status="closed")]
    result = filter_btc_15min_markets(closed)
    assert result == []


def test_rank_active_near_money_excludes_far_expiry_markets():
    now_ts = time.time()
    discovery = MagicMock()
    discovery.metadata = {
        "near-15m": MarketMetadata(
            market_ticker="near-15m",
            strike_price=65000.0,
            settlement_time_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts + 900)),
            last_trade_time_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts + 900)),
            asset="BTC",
            window_minutes=15,
            is_range=False,
        ),
        "far-60m": MarketMetadata(
            market_ticker="far-60m",
            strike_price=65000.0,
            settlement_time_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts + (73 * 3600))),
            last_trade_time_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts + (73 * 3600))),
            asset="BTC",
            window_minutes=60,
            is_range=False,
        ),
    }
    probability = MagicMock()
    probability._last_price_by_asset = {"BTC": 65020.0}

    ranked = _rank_active_near_money(set(discovery.metadata), discovery, probability, 10)

    assert ranked == ["near-15m"]




# ---------------------------------------------------------------------------
#  MarketDiscovery integration (mocked REST)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discovery_selects_btc_15min_tickers():
    """MarketDiscovery filters to only BTC 15-min markets via series filter."""
    cfg = _make_config()
    bus = Bus()
    rest = _mock_rest()

    selected_q = await bus.subscribe("kalshi.selected_markets")
    metadata_q = await bus.subscribe("kalshi.market_metadata")

    discovery = MarketDiscovery(cfg, rest, bus)
    await discovery.start()

    tickers = discovery.get_tickers()
    assert len(tickers) == 2
    assert "KXBTCM-26FEB25-B65000" in tickers
    assert "KXBTCM-26FEB25-B66000" in tickers
    assert "KXOTHER-26FEB25" not in tickers

    # kalshi.selected_markets was published.
    assert not selected_q.empty()
    msg = selected_q.get_nowait()
    assert isinstance(msg, SelectedMarkets)
    assert set(msg.tickers) == set(tickers)

    # Per-market metadata was published.
    assert not metadata_q.empty()

    await discovery.stop()


@pytest.mark.asyncio
async def test_discovery_with_explicit_tickers():
    """When target_market_tickers is set, discovery fetches them directly."""
    cfg = _make_config(
        target_market_tickers=["KXBTCM-26FEB25-B65000"],
        series_filter=None,
    )
    bus = Bus()
    rest = _mock_rest()

    discovery = MarketDiscovery(cfg, rest, bus)
    await discovery.start()

    tickers = discovery.get_tickers()
    assert tickers == ["KXBTCM-26FEB25-B65000"]

    await discovery.stop()


# ---------------------------------------------------------------------------
#  MockBtcFeed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mock_btc_feed_publishes():
    bus = Bus()
    q = await bus.subscribe("btc.mid_price")

    feed = MockBtcFeed(bus, price=42000.0, interval=0.05)
    await feed.start()
    await asyncio.sleep(0.15)
    await feed.stop()

    assert not q.empty()
    msg = q.get_nowait()
    assert isinstance(msg, BtcMidPrice)
    assert msg.price == 42000.0


# ---------------------------------------------------------------------------
#  Full wiring smoke test (dry-run, short-lived)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_wiring_smoke():
    """Start all engines in dry_run mode for a brief period, then shut down.

    Confirms:
    - Tickers are selected deterministically from mocked response.
    - kalshi.selected_markets is published.
    - Window engine initialises from mock feed ticks.
    - Everything cancels cleanly without exceptions.
    """
    cfg = _make_config()
    bus = Bus()
    rest = _mock_rest()

    # ── Discovery ───────────────────────────────────────────────────
    selected_q = await bus.subscribe("kalshi.selected_markets")

    discovery = MarketDiscovery(cfg, rest, bus)
    await discovery.start()
    tickers = discovery.get_tickers()
    assert len(tickers) == 2

    # ── Window engine ───────────────────────────────────────────────
    window_engine = BtcWindowEngine(bus, truth_topic=cfg.truth_feed_topic)
    await window_engine.start()

    # ── Probability engine ──────────────────────────────────────────
    prob_engine = ProbabilityEngine(bus, discovery.metadata)
    await prob_engine.start()

    # ── Strategy engine ─────────────────────────────────────────────
    strategy = StrategyEngine(cfg, bus)
    await strategy.start(tickers)

    # ── Execution engine ────────────────────────────────────────────
    execution = ExecutionEngine(cfg, bus, rest)
    await execution.start()

    # ── Mock feed ───────────────────────────────────────────────────
    mock_feed = MockBtcFeed(bus, topic=cfg.truth_feed_topic, interval=0.05)
    await mock_feed.start()

    # Let everything run briefly.
    await asyncio.sleep(0.3)

    # ── Assertions ──────────────────────────────────────────────────
    assert window_engine.initialised
    assert not selected_q.empty()

    # No orders should be placed (dry_run).
    assert execution.pending_count == 0

    # ── Clean shutdown ──────────────────────────────────────────────
    await mock_feed.stop()
    await execution.stop()
    await strategy.stop()
    await window_engine.stop()
    await discovery.stop()

    # If we got here without an unhandled exception, the test passes.


# ---------------------------------------------------------------------------
#  Per-asset sigma isolation regression test
# ---------------------------------------------------------------------------

def test_per_asset_sigma_not_contaminated():
    """Mixed asset prices must NOT contaminate each other's sigma.

    When BTC prices ($67k) and ETH prices ($2k) are received,
    the BTC sigma must be consistent with BTC-scale prices only.
    """
    from argus_kalshi.bus import Bus
    from argus_kalshi.kalshi_probability import ProbabilityEngine

    bus = Bus()
    engine = ProbabilityEngine(bus, {})

    # Simulate 10 BTC ticks (realistic: ~$10 moves around $67k = ~0.015% per tick)
    btc_prices = [67000 + i * 10 for i in range(10)]
    for p in btc_prices:
        engine._ingest_price("BTC", p)

    # Simulate 10 ETH ticks (realistic: ~$0.5 moves around $2k = ~0.025% per tick)
    eth_prices = [2000 + i * 0.5 for i in range(10)]
    for p in eth_prices:
        engine._ingest_price("ETH", p)

    btc_sigma = engine._sigma_by_asset.get("BTC", 0.0)
    eth_sigma = engine._sigma_by_asset.get("ETH", 0.0)

    # Both sigmas should be small (sub-0.1% per second for these price moves)
    # The SIGMA_CEILING is 0.05. Old bug: sigma would be >> 0.05 due to mixed prices.
    assert btc_sigma < 0.05, f"BTC sigma {btc_sigma:.6f} blown up (mixed with ETH prices?)"
    assert eth_sigma < 0.05, f"ETH sigma {eth_sigma:.6f} blown up (mixed with BTC prices?)"

    # Both should be non-zero (we have enough price variation)
    assert btc_sigma > 1e-9, f"BTC sigma {btc_sigma} is zero"
    assert eth_sigma > 1e-9, f"ETH sigma {eth_sigma} is zero"

    # Sigmas must be independent — BTC sigma should reflect ~$10/$67k moves
    # which is ~1.5e-4 log-return. ETH sigma reflects ~$0.5/$2k = 2.5e-4.
    # Both should be << 0.001 for these tiny moves.
    assert btc_sigma < 0.001, (
        f"BTC sigma {btc_sigma:.6f} too large for $10 moves on $67k base "
        f"(old bug would give ~3.0 from mixed asset log-returns)"
    )
    assert eth_sigma < 0.001, (
        f"ETH sigma {eth_sigma:.6f} too large for $0.50 moves on $2k base "
        f"(old bug would give ~3.0 from mixed asset log-returns)"
    )


# ---------------------------------------------------------------------------
#  Signal cooldown test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_signal_cooldown_suppresses_duplicates():
    """After a signal fires for (ticker, side), the same side should not
    fire again within signal_cooldown_s."""
    import time as _time
    from argus_kalshi.models import FairProbability, OrderbookState

    cfg = _make_config(
        signal_cooldown_s=5.0,       # 5-second cooldown for test speed
        min_edge_threshold=0.01,
        persistence_window_ms=0,     # no persistence delay
        latency_circuit_breaker_ms=0,
        dry_run=True,
        ws_trading_enabled=True,
    )
    bus = Bus()
    ticker = "KXBTCM-TEST-65000"
    strategy = StrategyEngine(cfg, bus)
    await strategy.start([ticker])

    # Prime truth feed so staleness check passes.
    from argus_kalshi.models import BtcMidPrice
    await bus.publish("btc.mid_price", BtcMidPrice(price=65100.0, timestamp=_time.time(), source="test", asset="BTC"))
    await asyncio.sleep(0.05)

    settle_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_time.time() + 150))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=ticker,
        strike_price=65000.0,
        settlement_time_iso=settle_iso,
        last_trade_time_iso=settle_iso,
        series_ticker="KXBTCM",
        event_ticker="KXBTCM-TEST",
        status="open",
        asset="BTC",
        window_minutes=15,
        is_range=False,
    ))
    await asyncio.sleep(0.05)

    signal_q = await bus.subscribe("kalshi.trade_signal")

    # Inject OB and fair prob that produces a clear YES signal.
    ob = OrderbookState(
        market_ticker=ticker,
        implied_yes_ask_cents=20,
        implied_no_ask_cents=82,
        valid=True,
        seq=1,
        best_yes_bid_cents=18,
        best_no_bid_cents=80,
        best_yes_depth=500,
        best_no_depth=500,
    )
    fp = FairProbability(market_ticker=ticker, p_yes=0.70, model_inputs={})

    await bus.publish(f"kalshi.orderbook.{ticker}", ob)
    await bus.publish("kalshi.fair_prob", fp)
    await asyncio.sleep(0.1)

    # First signal should have fired.
    assert not signal_q.empty(), "First signal should have fired"
    signal_q.get_nowait()  # drain it

    # Emit another fair prob immediately — should be suppressed by cooldown.
    await bus.publish("kalshi.fair_prob", fp)
    await asyncio.sleep(0.05)

    assert signal_q.empty(), "Second signal within cooldown should be suppressed"

    await strategy.stop()


@pytest.mark.asyncio
async def test_signal_cooldown_not_reset_by_metadata_refresh():
    """Metadata republication for an already-known live contract must NOT
    reset the signal cooldown (discovery refreshes every 15s, which is
    shorter than the 30s default cooldown)."""
    import time as _time
    from argus_kalshi.models import FairProbability, OrderbookState, MarketMetadata

    cfg = _make_config(
        signal_cooldown_s=5.0,
        min_edge_threshold=0.01,
        persistence_window_ms=0,
        latency_circuit_breaker_ms=0,
        dry_run=True,
        ws_trading_enabled=True,
    )
    bus = Bus()
    ticker = "KXBTCM-TEST-65000"
    strategy = StrategyEngine(cfg, bus)
    await strategy.start([ticker])

    from argus_kalshi.models import BtcMidPrice
    await bus.publish("btc.mid_price", BtcMidPrice(price=65100.0, timestamp=_time.time(), source="test", asset="BTC"))
    await asyncio.sleep(0.05)

    settle_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_time.time() + 150))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=ticker,
        strike_price=65000.0,
        settlement_time_iso=settle_iso,
        last_trade_time_iso=settle_iso,
        series_ticker="KXBTCM",
        event_ticker="KXBTCM-TEST",
        status="open",
        asset="BTC",
        window_minutes=15,
        is_range=False,
    ))
    await asyncio.sleep(0.05)

    signal_q = await bus.subscribe("kalshi.trade_signal")

    ob = OrderbookState(
        market_ticker=ticker,
        implied_yes_ask_cents=20, implied_no_ask_cents=82,
        valid=True, seq=1, best_yes_bid_cents=18, best_no_bid_cents=80,
        best_yes_depth=500, best_no_depth=500,
    )
    fp = FairProbability(market_ticker=ticker, p_yes=0.70, model_inputs={})

    await bus.publish(f"kalshi.orderbook.{ticker}", ob)
    await bus.publish("kalshi.fair_prob", fp)
    await asyncio.sleep(0.1)

    assert not signal_q.empty(), "First signal should fire"
    signal_q.get_nowait()

    # Simulate a discovery metadata refresh for the same live ticker.
    meta_refresh = MarketMetadata(
        market_ticker=ticker,
        strike_price=65000.0,
        settlement_time_iso=settle_iso,
        last_trade_time_iso=settle_iso,
        index_name="",
        contract_type="binary",
        series_ticker="KXBTCM",
        event_ticker="KXBTCM-TEST",
        status="open",
        asset="BTC",
        window_minutes=15,
        is_range=False,
        strike_floor=None,
        strike_cap=None,
    )
    await bus.publish("kalshi.market_metadata", meta_refresh)
    await asyncio.sleep(0.05)

    # Now emit another fair prob — cooldown should still be active.
    await bus.publish("kalshi.fair_prob", fp)
    await asyncio.sleep(0.05)

    assert signal_q.empty(), (
        "Signal cooldown must not be reset by a metadata refresh of an already-known live contract"
    )

    await strategy.stop()


# ---------------------------------------------------------------------------
#  Strike misparse regression tests (ETH, BTC)
# ---------------------------------------------------------------------------

def test_btc_15m_wrong_strike_corrected_from_ticker():
    """BTC 15m market with API strike_price=30 (e.g. time segment) is corrected
    using ticker-derived strike when in range."""
    from argus_kalshi.kalshi_markets import MarketDiscovery
    raw = {
        "ticker": "KXBTC15M-02MAR0815-69550",
        "series_ticker": "KXBTC15M",
        "event_ticker": "KXBTC15M-02MAR08",
        "strike_price": 30.0,
        "status": "open",
        "close_time": "2026-03-02T08:15:00Z",
        "expiration_time": "2026-03-02T08:15:00Z",
        "last_trade_time": "2026-03-02T08:00:00Z",
    }
    meta = MarketDiscovery._parse_market(raw)
    assert meta is not None
    assert meta.strike_price == 69550.0, (
        f"Got {meta.strike_price}; expected 69550 (ticker-derived for BTC)"
    )
    assert meta.asset == "BTC"


def test_btc_15m_wrong_strike_corrected_from_floor_strike():
    """When strike_price is wrong and ticker has no valid number, use floor_strike."""
    from argus_kalshi.kalshi_markets import MarketDiscovery
    raw = {
        "ticker": "KXBTC15M-02MAR0830",
        "series_ticker": "KXBTC15M",
        "event_ticker": "KXBTC15M-02MAR08",
        "strike_price": 30.0,
        "floor_strike": 69550.0,
        "status": "open",
        "close_time": "2026-03-02T08:30:00Z",
        "expiration_time": "2026-03-02T08:30:00Z",
        "last_trade_time": "2026-03-02T08:00:00Z",
    }
    meta = MarketDiscovery._parse_market(raw)
    assert meta is not None
    assert meta.strike_price == 69550.0, (
        f"Got {meta.strike_price}; expected 69550 from floor_strike"
    )


def test_eth_market_strike_uses_api_field_not_ticker_date():
    """ETH market with API-provided strike_price must use that value, not
    a date component extracted from the ticker string."""
    from argus_kalshi.kalshi_markets import MarketDiscovery
    raw = {
        "ticker": "KXETH15M-26MAR25-2030",
        "series_ticker": "KXETH15M",
        "event_ticker": "KXETH15M-26MAR25",
        "strike_price": 2030.0,
        "status": "open",
        "close_time": "2025-03-26T15:15:00Z",
        "expiration_time": "2025-03-26T15:15:00Z",
        "last_trade_time": "2025-03-26T15:15:00Z",
    }
    meta = MarketDiscovery._parse_market(raw)
    assert meta is not None
    assert meta.strike_price == 2030.0, f"Got {meta.strike_price}, expected 2030.0"
    assert meta.asset == "ETH", f"Got asset={meta.asset}, expected ETH"
    assert meta.window_minutes == 15


def test_eth_15m_rollover_ticker_does_not_parse_time_as_strike():
    """ETH 15m rollover-style tickers with invalid strike are skipped."""
    from argus_kalshi.kalshi_markets import MarketDiscovery

    raw = {
        "ticker": "KXETH15M-26MAR032115-15",
        "series_ticker": "KXETH15M",
        "event_ticker": "KXETH15M-26MAR03",
        "strike_price": 15.0,
        "status": "open",
        "close_time": "2026-03-03T21:15:00Z",
        "expiration_time": "2026-03-03T21:15:00Z",
        "last_trade_time": "2026-03-03T21:15:00Z",
    }

    meta = MarketDiscovery._parse_market(raw)
    assert meta is None


def test_eth_15m_rollover_recovers_strike_from_title_text():
    """If API strike is a rollover fragment, recover true strike from title/subtitle."""
    from argus_kalshi.kalshi_markets import MarketDiscovery

    raw = {
        "ticker": "KXETH15M-26MAR041645-45",
        "series_ticker": "KXETH15M",
        "event_ticker": "KXETH15M-26MAR04",
        "strike_price": 45.0,
        "title": "Will ETH be above $2,160 at 4:45 PM ET?",
        "subtitle": "15 minute ETH market",
        "status": "open",
        "close_time": "2026-03-04T16:45:00Z",
        "expiration_time": "2026-03-04T16:45:00Z",
        "last_trade_time": "2026-03-04T16:45:00Z",
    }

    meta = MarketDiscovery._parse_market(raw)
    assert meta is not None
    assert meta.asset == "ETH"
    assert meta.strike_price == 2160.0


def test_eth_15m_rollover_does_not_use_year_token_as_strike():
    """If title text has no valid strike, invalid market is skipped."""
    from argus_kalshi.kalshi_markets import MarketDiscovery

    raw = {
        "ticker": "KXETH15M-26MAR071345-45",
        "series_ticker": "KXETH15M",
        "event_ticker": "KXETH15M-26MAR07",
        "strike_price": 45.0,
        "title": "Will ETH resolve in 2026?",
        "subtitle": "15 minute ETH market",
        "status": "open",
        "close_time": "2026-03-07T13:45:00Z",
        "expiration_time": "2026-03-07T13:45:00Z",
        "last_trade_time": "2026-03-07T13:45:00Z",
    }

    meta = MarketDiscovery._parse_market(raw)
    assert meta is None


def test_sol_15m_rollover_ticker_does_not_parse_time_as_strike():
    """SOL 15m rollover-style tickers with invalid strike are skipped."""
    from argus_kalshi.kalshi_markets import MarketDiscovery

    raw = {
        "ticker": "KXSOL15M-26MAR032100-00",
        "series_ticker": "KXSOL15M",
        "event_ticker": "KXSOL15M-26MAR03",
        "strike_price": 0.0,
        "status": "open",
        "close_time": "2026-03-03T21:00:00Z",
        "expiration_time": "2026-03-03T21:00:00Z",
        "last_trade_time": "2026-03-03T21:00:00Z",
    }

    meta = MarketDiscovery._parse_market(raw)
    assert meta is None


def test_eth_market_fallback_asset_when_series_unknown():
    """When series_ticker is not in _SERIES_MAP, asset must be inferred from
    the ticker string rather than defaulting to BTC."""
    from argus_kalshi.kalshi_markets import MarketDiscovery
    raw = {
        "ticker": "KXETH-CUSTOM-2030",
        "series_ticker": "KXETH-CUSTOM",   # not in _SERIES_MAP
        "event_ticker": "KXETH-CUSTOM-26MAR25",
        "strike_price": 2030.0,
        "status": "open",
        "close_time": "2025-03-26T15:15:00Z",
        "expiration_time": "2025-03-26T15:15:00Z",
        "last_trade_time": "2025-03-26T15:15:00Z",
    }
    meta = MarketDiscovery._parse_market(raw)
    assert meta is not None
    assert meta.asset == "ETH", (
        f"Got asset={meta.asset}; unknown ETH series should still infer ETH from ticker"
    )


# ---------------------------------------------------------------------------
#  Execution engine: dry_run fills bypass halt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dry_run_paper_fills_bypass_execution_halt():
    """In dry_run mode, paper fills must be published even when the
    execution engine is halted (e.g. due to WS disconnect)."""
    import time as _time
    from argus_kalshi.models import TradeSignal, FillEvent
    from argus_kalshi.kalshi_execution import ExecutionEngine
    bus = Bus()
    rest = _mock_rest()
    execution = ExecutionEngine(_make_config(dry_run=True, scenario_profile="best"), bus, rest)
    await execution.start()

    # Force halt the engine (simulating a WS disconnect risk event).
    execution._halted = True

    fills_q = await bus.subscribe("kalshi.fills")

    signal = TradeSignal(
        market_ticker="KXBTCM-TEST-65000",
        side="yes",
        action="buy",
        limit_price_cents=42,
        quantity_contracts=1,
        edge=0.15,
        p_yes=0.65,
        timestamp=_time.time(),
        order_style="aggressive",
    )
    await bus.publish("kalshi.trade_signal", signal)
    await asyncio.sleep(0.1)

    assert not fills_q.empty(), (
        "Paper fill must be published even when execution engine is halted"
    )
    fill: FillEvent = fills_q.get_nowait()
    assert fill.market_ticker == "KXBTCM-TEST-65000"
    assert fill.side == "yes"
    assert fill.price_cents == 42
    assert fill.count == 100  # 1 contract * 100 centicx

    await execution.stop()


# ---------------------------------------------------------------------------
#  Drawdown halt persistence: must NOT be cleared by WS reconnect
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drawdown_halt_not_cleared_by_ws_reconnect():
    """A drawdown_breach halt must NOT be cleared when the WebSocket reconnects.
    Only disconnect_halt type halts auto-resume on reconnect."""
    from argus_kalshi.kalshi_execution import ExecutionEngine
    from argus_kalshi.models import TradeSignal, FillEvent
    import time as _time

    cfg = _make_config(dry_run=False)  # live mode so halt actually blocks signals
    bus = Bus()
    rest = _mock_rest()

    execution = ExecutionEngine(cfg, bus, rest)
    await execution.start()

    # Force a drawdown_breach halt (not disconnect_halt).
    execution._halted = True
    execution._halt_reason = "drawdown_breach"

    # Simulate a WS reconnect event.
    from argus_kalshi.models import WsConnectionEvent
    await bus.publish("kalshi.ws.status", WsConnectionEvent(status="connected", timestamp=_time.time()))
    await asyncio.sleep(0.05)

    # Engine must still be halted — drawdown is a permanent intraday halt.
    assert execution._halted, (
        "drawdown_breach halt must persist after WS reconnect; "
        "only disconnect_halt auto-resumes"
    )
    assert execution._halt_reason == "drawdown_breach"

    await execution.stop()
