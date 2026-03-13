"""
End-to-end test: valid signal → order placement.

Wires StrategyEngine + ExecutionEngine with a mock REST client and
verifies that a fair-probability + orderbook combination with sufficient
edge actually calls create_order on the REST layer.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pytest

from argus_kalshi.bus import Bus
from argus_kalshi.config import KalshiConfig
from argus_kalshi.kalshi_execution import ExecutionEngine
from argus_kalshi.kalshi_strategy import StrategyEngine
from argus_kalshi.models import (
    BtcMidPrice,
    FairProbability,
    MarketMetadata,
    OrderbookState,
)

TICKER = "KXBTC-TEST-99000"


def _make_config(**overrides: Any) -> KalshiConfig:
    defaults = dict(
        bankroll_usd=10_000.0,
        dry_run=False,
        ws_trading_enabled=True,
        min_edge_threshold=0.02,
        effective_edge_fee_pct=0.0,       # no fee deduction for clarity
        persistence_window_ms=0,          # skip persistence delay
        latency_circuit_breaker_ms=0,     # skip latency check
        truth_feed_stale_timeout_s=999,   # don't go stale during test
        max_fraction_per_market=1.0,      # no position cap
        max_contracts_per_ticker=1,
        order_timeout_ms=5000,
        # E2E fixture ticker has no real settlement metadata. Disable the
        # near-expiry entry horizon gate here so this test continues to
        # validate signal -> execution plumbing.
        max_entry_minutes_to_expiry=0,
        range_max_entry_minutes_to_expiry=0,
    )
    defaults.update(overrides)
    return KalshiConfig(**defaults)


def _make_orderbook(
    ticker: str = TICKER,
    yes_bid: int = 40,
    no_bid: int = 40,
    seq: int = 1,
) -> OrderbookState:
    """Orderbook where implied YES ask = 100 - no_bid, implied NO ask = 100 - yes_bid."""
    return OrderbookState(
        market_ticker=ticker,
        best_yes_bid_cents=yes_bid,
        best_no_bid_cents=no_bid,
        implied_yes_ask_cents=100 - no_bid,
        implied_no_ask_cents=100 - yes_bid,
        seq=seq,
        valid=True,
        best_yes_depth=10_000,
        best_no_depth=10_000,
    )


def _make_metadata(
    ticker: str = TICKER,
    *,
    settlement_time_iso: Optional[str] = None,
) -> MarketMetadata:
    now = time.time()
    settle_ts = now + 60.0
    settle_iso = settlement_time_iso or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(settle_ts))
    last_trade_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    return MarketMetadata(
        market_ticker=ticker,
        strike_price=99_000.0,
        settlement_time_iso=settle_iso,
        last_trade_time_iso=last_trade_iso,
        series_ticker="KXBTC15M",
        event_ticker="KXBTC15M-TEST",
        asset="BTC",
        window_minutes=15,
        is_range=False,
        status="open",
    )


class MockRestClient:
    """Captures create_order calls for assertion."""

    def __init__(self) -> None:
        self.orders: list[Dict[str, Any]] = []
        self.cancel_order = AsyncMock()

    async def create_order(self, **kwargs: Any) -> Dict[str, Any]:
        self.orders.append(kwargs)
        return {"order": {"order_id": f"mock-oid-{len(self.orders)}"}}


@pytest.mark.asyncio
async def test_valid_edge_triggers_order() -> None:
    """
    A fair p_yes=0.80 vs implied YES ask=60¢ gives EV_yes=0.20 which
    exceeds min_edge=0.02.  This should produce a TradeSignal that the
    ExecutionEngine turns into a create_order call.
    """
    bus = Bus()
    cfg = _make_config()
    rest = MockRestClient()

    strategy = StrategyEngine(cfg, bus)
    execution = ExecutionEngine(cfg, bus, rest)  # type: ignore[arg-type]

    await strategy.start(market_tickers=[TICKER])
    await execution.start()

    # Give tasks a moment to subscribe.
    await asyncio.sleep(0.05)

    # 1. Send a truth tick so the strategy doesn't consider the feed stale.
    await bus.publish("btc.mid_price", BtcMidPrice(price=99000.0, timestamp=time.time()))
    await asyncio.sleep(0.05)

    # 2. Publish orderbook: YES ask = 60¢ (implied from NO bid 40¢).
    ob = _make_orderbook(yes_bid=40, no_bid=40)
    await bus.publish(f"kalshi.orderbook.{TICKER}", ob)
    await asyncio.sleep(0.05)
    await bus.publish("kalshi.market_metadata", _make_metadata())
    await asyncio.sleep(0.05)

    # 3. Publish fair probability: p_yes = 0.80 → EV_yes = 0.80 - 0.60 = 0.20.
    fp = FairProbability(market_ticker=TICKER, p_yes=0.80)
    await bus.publish("kalshi.fair_prob", fp)

    # Allow the signal to propagate through strategy → execution.
    await asyncio.sleep(0.2)

    await strategy.stop()
    await execution.stop()

    # Verify that create_order was called exactly once.
    assert len(rest.orders) == 1, f"Expected 1 order, got {len(rest.orders)}: {rest.orders}"
    order = rest.orders[0]
    assert order["ticker"] == TICKER
    assert order["side"] == "yes"
    assert order["action"] == "buy"
    assert order["count"] == 1
    # YES buy should have yes_price set to the implied ask.
    assert order["yes_price"] == 60


@pytest.mark.asyncio
async def test_no_edge_does_not_trigger_order() -> None:
    """
    A fair p_yes=0.55 vs implied YES ask=60¢ gives EV_yes = -0.05,
    which is below the threshold.  No order should be placed.
    """
    bus = Bus()
    cfg = _make_config()
    rest = MockRestClient()

    strategy = StrategyEngine(cfg, bus)
    execution = ExecutionEngine(cfg, bus, rest)  # type: ignore[arg-type]

    await strategy.start(market_tickers=[TICKER])
    await execution.start()
    await asyncio.sleep(0.05)

    await bus.publish("btc.mid_price", BtcMidPrice(price=99000.0, timestamp=time.time()))
    await asyncio.sleep(0.05)

    ob = _make_orderbook(yes_bid=40, no_bid=40)
    await bus.publish(f"kalshi.orderbook.{TICKER}", ob)
    await asyncio.sleep(0.05)
    await bus.publish("kalshi.market_metadata", _make_metadata())
    await asyncio.sleep(0.05)

    # p_yes=0.60 matches market price exactly → divergence=0, no edge.
    fp = FairProbability(market_ticker=TICKER, p_yes=0.60)
    await bus.publish("kalshi.fair_prob", fp)
    await asyncio.sleep(0.2)

    await strategy.stop()
    await execution.stop()

    assert len(rest.orders) == 0, f"Expected 0 orders, got {len(rest.orders)}"


@pytest.mark.asyncio
async def test_dry_run_blocks_order() -> None:
    """
    With dry_run=True and sufficient edge, the execution engine must NOT call
    create_order on the REST client.  Instead, it publishes a synthetic paper
    FillEvent on kalshi.fills so the settlement tracker and UI can track paper
    performance.  dry_run does not silently swallow the signal — it produces a
    paper fill that is observable on the bus.
    """
    bus = Bus()
    cfg = _make_config(dry_run=True)
    rest = MockRestClient()

    fills_q = await bus.subscribe("kalshi.fills")

    strategy = StrategyEngine(cfg, bus)
    execution = ExecutionEngine(cfg, bus, rest)  # type: ignore[arg-type]

    await strategy.start(market_tickers=[TICKER])
    await execution.start()
    await asyncio.sleep(0.05)

    await bus.publish("btc.mid_price", BtcMidPrice(price=99000.0, timestamp=time.time()))
    await asyncio.sleep(0.05)

    ob = _make_orderbook(yes_bid=40, no_bid=40)
    await bus.publish(f"kalshi.orderbook.{TICKER}", ob)
    await asyncio.sleep(0.05)
    await bus.publish("kalshi.market_metadata", _make_metadata())
    await asyncio.sleep(0.05)

    fp = FairProbability(market_ticker=TICKER, p_yes=0.80)
    await bus.publish("kalshi.fair_prob", fp)
    await asyncio.sleep(0.2)

    await strategy.stop()
    await execution.stop()

    # No real REST order must be placed.
    assert len(rest.orders) == 0, "dry_run must not call create_order on the REST client"

    # A paper fill must have been published on kalshi.fills instead.
    assert not fills_q.empty(), (
        "dry_run must publish a synthetic paper FillEvent on kalshi.fills"
    )


@pytest.mark.asyncio
async def test_no_side_buy_triggered() -> None:
    """
    When p_yes is low (0.20), EV_no = (1 - 0.20) - 0.60 = 0.20 which
    exceeds the threshold.  Should trigger a NO buy.
    """
    bus = Bus()
    cfg = _make_config()
    rest = MockRestClient()

    strategy = StrategyEngine(cfg, bus)
    execution = ExecutionEngine(cfg, bus, rest)  # type: ignore[arg-type]

    await strategy.start(market_tickers=[TICKER])
    await execution.start()
    await asyncio.sleep(0.05)

    await bus.publish("btc.mid_price", BtcMidPrice(price=99000.0, timestamp=time.time()))
    await asyncio.sleep(0.05)

    ob = _make_orderbook(yes_bid=40, no_bid=40)
    await bus.publish(f"kalshi.orderbook.{TICKER}", ob)
    await asyncio.sleep(0.05)
    await bus.publish("kalshi.market_metadata", _make_metadata())
    await asyncio.sleep(0.05)

    fp = FairProbability(market_ticker=TICKER, p_yes=0.20)
    await bus.publish("kalshi.fair_prob", fp)
    await asyncio.sleep(0.2)

    await strategy.stop()
    await execution.stop()

    assert len(rest.orders) == 1
    order = rest.orders[0]
    assert order["side"] == "no"
    assert order["action"] == "buy"
    assert order["count"] == 1
    assert order["no_price"] == 60  # implied NO ask = 100 - 40


@pytest.mark.asyncio
async def test_ws_trading_disabled_blocks_signal() -> None:
    """ws_trading_enabled=False should prevent signal publication."""
    bus = Bus()
    cfg = _make_config(ws_trading_enabled=False)
    rest = MockRestClient()

    strategy = StrategyEngine(cfg, bus)
    execution = ExecutionEngine(cfg, bus, rest)  # type: ignore[arg-type]

    await strategy.start(market_tickers=[TICKER])
    await execution.start()
    await asyncio.sleep(0.05)

    await bus.publish("btc.mid_price", BtcMidPrice(price=99000.0, timestamp=time.time()))
    await asyncio.sleep(0.05)

    ob = _make_orderbook(yes_bid=40, no_bid=40)
    await bus.publish(f"kalshi.orderbook.{TICKER}", ob)
    await asyncio.sleep(0.05)
    await bus.publish("kalshi.market_metadata", _make_metadata())
    await asyncio.sleep(0.05)

    fp = FairProbability(market_ticker=TICKER, p_yes=0.80)
    await bus.publish("kalshi.fair_prob", fp)
    await asyncio.sleep(0.2)

    await strategy.stop()
    await execution.stop()

    assert len(rest.orders) == 0, "ws_trading_enabled=False should block signals"
