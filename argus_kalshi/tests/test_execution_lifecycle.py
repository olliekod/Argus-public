# Created by Oliver Meihls

# Comprehensive execution lifecycle tests.
#
# Tests the full order lifecycle: signal → placement → fill → settlement → PnL,
# including dry-run paper trading, risk controls, timeout/cancellation, duplicate
# prevention, and multi-asset scenarios.

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from argus_kalshi.bus import Bus
from argus_kalshi.config import KalshiConfig
from argus_kalshi.kalshi_execution import ExecutionEngine
from argus_kalshi.settlement_tracker import SettlementTracker
from argus_kalshi.kalshi_strategy import StrategyEngine
from argus_kalshi.models import (
    BtcMidPrice,
    FairProbability,
    FillEvent,
    MarketMetadata,
    OrderbookState,
    OrderUpdate,
    RiskEvent,
    SettlementOutcome,
    TradeSignal,
)

TICKER_BTC = "KXBTC15M-TEST-67000"
TICKER_ETH = "KXETH15M-TEST-1960"


# ── Helpers ───────────────────────────────────────────────────────────

def _cfg(**overrides) -> KalshiConfig:
    defaults = dict(
        bankroll_usd=10_000.0,
        dry_run=False,
        ws_trading_enabled=True,
        min_edge_threshold=0.02,
        effective_edge_fee_pct=0.0,
        persistence_window_ms=0,
        hold_require_momentum_agreement=False,
        hold_require_flow_agreement=False,
        latency_circuit_breaker_ms=0,
        truth_feed_stale_timeout_s=999,
        max_fraction_per_market=1.0,
        max_contracts_per_ticker=1,
        order_timeout_ms=500,       # fast timeout for tests
        passive_order_timeout_ms=1000,
        paper_slippage_cents=0,
    )
    defaults.update(overrides)
    return KalshiConfig(**defaults)


def _ob(ticker: str = TICKER_BTC, yes_bid: int = 40, no_bid: int = 40,
        seq: int = 1, valid: bool = True) -> OrderbookState:
    return OrderbookState(
        market_ticker=ticker,
        best_yes_bid_cents=yes_bid,
        best_no_bid_cents=no_bid,
        implied_yes_ask_cents=100 - no_bid,
        implied_no_ask_cents=100 - yes_bid,
        seq=seq,
        valid=valid,
        best_yes_depth=10_000,
        best_no_depth=10_000,
    )


class MockRest:
    # Mock REST client that records order calls.
    def __init__(self, fail_always: bool = False):
        self.orders: List[Dict[str, Any]] = []
        self.cancels: List[str] = []
        self._fail_always = fail_always
        self.cancel_order = AsyncMock(side_effect=self._cancel)

    async def _cancel(self, order_id: str) -> Dict:
        self.cancels.append(order_id)
        return {"order_id": order_id}

    async def create_order(self, **kwargs) -> Dict[str, Any]:
        if self._fail_always:
            raise RuntimeError("Simulated API error")
        self.orders.append(kwargs)
        oid = f"oid-{len(self.orders)}"
        return {"order": {"order_id": oid}}


async def _seed_strategy(bus: Bus, ticker: str = TICKER_BTC, asset: str = "BTC"):
    # Publish truth tick + metadata so strategy doesn't reject signals.
    topic = f"{asset.lower()}.mid_price"
    await bus.publish(topic, BtcMidPrice(price=67000.0, timestamp=time.time(), asset=asset))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=ticker,
        strike_price=67000.0,
        settlement_time_iso="2026-03-01T00:00:00Z",
        last_trade_time_iso="2026-02-28T23:00:00Z",
        asset=asset,
    ))
    await asyncio.sleep(0.05)


#  1. DRY-RUN PAPER TRADING

@pytest.mark.asyncio
async def test_dry_run_emits_paper_fill():
    # In dry_run mode, execution should emit a synthetic FillEvent (no REST call).
    bus = Bus()
    rest = MockRest()
    exe = ExecutionEngine(_cfg(dry_run=True, scenario_profile="best"), bus, rest)
    await exe.start()

    fills_q = await bus.subscribe("kalshi.fills")

    # Directly publish a trade signal (bypassing strategy)
    sig = TradeSignal(
        market_ticker=TICKER_BTC, side="yes", action="buy",
        limit_price_cents=45, quantity_contracts=1, edge=0.10,
        p_yes=0.80, timestamp=time.time(),
    )
    await bus.publish("kalshi.trade_signal", sig)
    await asyncio.sleep(0.15)

    await exe.stop()

    # No REST call should have been made
    assert len(rest.orders) == 0

    # But a paper fill should be on the bus
    assert not fills_q.empty()
    fill: FillEvent = fills_q.get_nowait()
    assert fill.market_ticker == TICKER_BTC
    assert fill.side == "yes"
    assert fill.price_cents == 45
    assert fill.order_id.startswith("paper-")


@pytest.mark.asyncio
async def test_dry_run_paper_fill_respects_latency_window():
    bus = Bus()
    rest = MockRest()
    exe = ExecutionEngine(
        _cfg(
            dry_run=True,
            paper_order_latency_min_ms=60,
            paper_order_latency_max_ms=60,
            paper_slippage_cents=0,
        ),
        bus,
        rest,
    )
    await exe.start()

    fills_q = await bus.subscribe("kalshi.fills")
    sig = TradeSignal(
        market_ticker=TICKER_BTC, side="yes", action="buy",
        limit_price_cents=45, quantity_contracts=1, edge=0.10,
        p_yes=0.80, timestamp=time.time(),
    )
    await bus.publish("kalshi.trade_signal", sig)
    await asyncio.sleep(0.02)
    assert fills_q.empty()
    await asyncio.sleep(0.08)
    assert not fills_q.empty()
    await exe.stop()


@pytest.mark.asyncio
async def test_early_exit_pnl_deducts_fees():
    bus = Bus()
    tracker = SettlementTracker(bus)
    await tracker.start()
    await asyncio.sleep(0.05)

    out_q = await bus.subscribe("kalshi.settlement_outcome")
    ticker = TICKER_ETH
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=ticker,
        strike_price=1960.0,
        settlement_time_iso="2026-03-01T00:00:00Z",
        last_trade_time_iso="2026-02-28T23:50:00Z",
        asset="ETH",
        window_minutes=15,
    ))
    await asyncio.sleep(0.1)

    await bus.publish("kalshi.fills", FillEvent(
        market_ticker=ticker,
        order_id="paper-buy",
        side="yes",
        price_cents=50,
        count=100,
        is_taker=True,
        timestamp=time.time(),
        fee_usd=0.02,
        action="buy",
    ))
    await asyncio.sleep(0.1)
    await bus.publish("kalshi.fills", FillEvent(
        market_ticker=ticker,
        order_id="paper-sell",
        side="yes",
        price_cents=52,
        count=100,
        is_taker=True,
        timestamp=time.time(),
        fee_usd=0.02,
        action="sell",
    ))
    await asyncio.sleep(0.05)

    assert not out_q.empty()
    out: SettlementOutcome = out_q.get_nowait()
    assert round(out.gross_pnl, 4) == 0.02
    assert round(out.fees_usd, 4) == 0.04
    assert round(out.pnl, 4) == -0.02
    assert out.won is False

    tracker.stop()


#  2. LIVE ORDER PLACEMENT + FILL

@pytest.mark.asyncio
async def test_live_order_placement_and_fill():
    # Live mode: signal → create_order → fill arrives → position tracked.
    bus = Bus()
    rest = MockRest()
    exe = ExecutionEngine(_cfg(), bus, rest)
    await exe.start()

    sig = TradeSignal(
        market_ticker=TICKER_BTC, side="yes", action="buy",
        limit_price_cents=60, quantity_contracts=2, edge=0.10,
        p_yes=0.80, timestamp=time.time(),
    )
    await bus.publish("kalshi.trade_signal", sig)
    await asyncio.sleep(0.15)

    # Order should be placed
    assert len(rest.orders) == 1
    assert rest.orders[0]["ticker"] == TICKER_BTC
    assert rest.orders[0]["count"] == 2

    # Simulate fill arriving from WS
    fill = FillEvent(
        market_ticker=TICKER_BTC, order_id="oid-1", side="yes",
        price_cents=60, count=200, is_taker=True, timestamp=time.time(),
    )
    await bus.publish("kalshi.fills", fill)
    await asyncio.sleep(0.1)

    # Position should be tracked
    assert exe._positions.get(TICKER_BTC) == 200

    await exe.stop()


#  3. ORDER TIMEOUT → CANCEL

@pytest.mark.asyncio
async def test_order_timeout_cancels():
    # Unfilled order should be cancelled after timeout.
    bus = Bus()
    rest = MockRest()
    exe = ExecutionEngine(_cfg(order_timeout_ms=200), bus, rest)
    await exe.start()

    sig = TradeSignal(
        market_ticker=TICKER_BTC, side="yes", action="buy",
        limit_price_cents=60, quantity_contracts=1, edge=0.05,
        p_yes=0.70, timestamp=time.time(),
    )
    await bus.publish("kalshi.trade_signal", sig)
    await asyncio.sleep(0.1)

    # Order placed, should be pending
    assert exe.pending_count == 1
    assert exe.has_pending_for(TICKER_BTC)

    # Wait for timeout (200ms + buffer)
    await asyncio.sleep(0.4)

    # Should have been cancelled
    assert exe.pending_count == 0
    assert not exe.has_pending_for(TICKER_BTC)
    assert len(rest.cancels) == 1

    await exe.stop()


#  4. DUPLICATE PREVENTION

@pytest.mark.asyncio
async def test_duplicate_signal_for_same_market_blocked():
    # Only one pending order per market — duplicate signals are dropped.
    bus = Bus()
    rest = MockRest()
    exe = ExecutionEngine(_cfg(order_timeout_ms=5000), bus, rest)
    await exe.start()

    for _ in range(3):
        sig = TradeSignal(
            market_ticker=TICKER_BTC, side="yes", action="buy",
            limit_price_cents=60, quantity_contracts=1, edge=0.10,
            p_yes=0.80, timestamp=time.time(),
        )
        await bus.publish("kalshi.trade_signal", sig)
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.15)

    # Only 1 order should be placed despite 3 signals
    assert len(rest.orders) == 1

    await exe.stop()


#  5. RISK HALT

@pytest.mark.asyncio
async def test_risk_event_halts_execution():
    # Risk events should halt execution and cancel pending orders.
    bus = Bus()
    rest = MockRest()
    exe = ExecutionEngine(_cfg(order_timeout_ms=5000), bus, rest)
    await exe.start()

    # Place an order
    sig = TradeSignal(
        market_ticker=TICKER_BTC, side="yes", action="buy",
        limit_price_cents=60, quantity_contracts=1, edge=0.10,
        p_yes=0.80, timestamp=time.time(),
    )
    await bus.publish("kalshi.trade_signal", sig)
    await asyncio.sleep(0.15)
    assert exe.pending_count == 1

    # Trigger risk halt
    await bus.publish("kalshi.risk", RiskEvent(
        event_type="disconnect_halt",
        detail="test",
        timestamp=time.time(),
    ))
    await asyncio.sleep(0.15)

    assert exe.halted
    assert exe.pending_count == 0

    # New signals should be dropped
    sig2 = TradeSignal(
        market_ticker=TICKER_ETH, side="no", action="buy",
        limit_price_cents=50, quantity_contracts=1, edge=0.15,
        p_yes=0.30, timestamp=time.time(),
    )
    await bus.publish("kalshi.trade_signal", sig2)
    await asyncio.sleep(0.15)

    assert len(rest.orders) == 1  # still just the first one

    await exe.stop()


#  6. API ERROR HANDLING

@pytest.mark.asyncio
async def test_api_error_publishes_error_update():
    # REST error should publish an error OrderUpdate, not crash.
    bus = Bus()
    rest = MockRest(fail_always=True)  # all calls fail
    exe = ExecutionEngine(_cfg(), bus, rest)
    await exe.start()

    updates_q = await bus.subscribe("kalshi.order_update")

    sig = TradeSignal(
        market_ticker=TICKER_BTC, side="yes", action="buy",
        limit_price_cents=60, quantity_contracts=1, edge=0.10,
        p_yes=0.80, timestamp=time.time(),
    )
    await bus.publish("kalshi.trade_signal", sig)
    await asyncio.sleep(0.2)

    # Should have published an error update
    assert not updates_q.empty()
    update: OrderUpdate = updates_q.get_nowait()
    assert update.status == "error"
    assert "Simulated" in update.error_detail

    await exe.stop()


#  7. DEPTH CHECK

@pytest.mark.asyncio
async def test_insufficient_depth_blocks_order():
    # Order should be rejected if orderbook depth is insufficient.
    bus = Bus()
    rest = MockRest()
    exe = ExecutionEngine(_cfg(), bus, rest)
    await exe.start()
    await asyncio.sleep(0.05)

    # Publish thin orderbook (only 50 centicx available, need 100)
    thin_ob = OrderbookState(
        market_ticker=TICKER_BTC,
        best_yes_bid_cents=40, best_no_bid_cents=40,
        implied_yes_ask_cents=60, implied_no_ask_cents=60,
        seq=1, valid=True,
        best_yes_depth=50, best_no_depth=50,  # thin
    )
    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", thin_ob)
    await asyncio.sleep(0.1)

    # Need to subscribe first so exe has the OB
    await exe.subscribe_orderbook(TICKER_BTC)
    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", thin_ob)
    await asyncio.sleep(0.1)

    sig = TradeSignal(
        market_ticker=TICKER_BTC, side="yes", action="buy",
        limit_price_cents=60, quantity_contracts=1, edge=0.10,
        p_yes=0.80, timestamp=time.time(),
    )
    await bus.publish("kalshi.trade_signal", sig)
    await asyncio.sleep(0.2)

    # Should be blocked by depth check
    assert len(rest.orders) == 0

    await exe.stop()


#  8. FULL STRATEGY → EXECUTION PIPELINE

@pytest.mark.asyncio
async def test_full_pipeline_strategy_to_execution():
    # Full pipeline: price tick + orderbook + probability →
    # strategy evaluates → emits signal → execution places order.
    bus = Bus()
    rest = MockRest()
    config = _cfg()

    strategy = StrategyEngine(config, bus)
    execution = ExecutionEngine(config, bus, rest)

    await strategy.start(market_tickers=[TICKER_BTC])
    await execution.start()
    await asyncio.sleep(0.05)

    # Seed truth feed
    await bus.publish("btc.mid_price", BtcMidPrice(
        price=67000.0, timestamp=time.time(), asset="BTC"
    ))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=TICKER_BTC, strike_price=67000.0,
        settlement_time_iso="2026-03-01T00:00:00Z",
        last_trade_time_iso="2026-02-28T23:00:00Z",
        asset="BTC",
    ))
    await asyncio.sleep(0.05)

    # Publish orderbook
    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", _ob(yes_bid=40, no_bid=40))
    await asyncio.sleep(0.05)

    # Publish strong edge: p_yes=0.85 vs ask=60¢ → edge=25¢
    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC, p_yes=0.85,
    ))
    await asyncio.sleep(0.3)

    await strategy.stop()
    await execution.stop()

    assert len(rest.orders) == 1
    assert rest.orders[0]["side"] == "yes"
    assert rest.orders[0]["ticker"] == TICKER_BTC
    assert rest.orders[0]["count"] == 1


#  9. PASSIVE ORDER STYLE

@pytest.mark.asyncio
async def test_passive_order_posts_at_bid():
    # passive order_style should post at best bid, not cross spread.
    bus = Bus()
    rest = MockRest()
    config = _cfg(default_order_style="passive", passive_order_timeout_ms=500, max_contracts_per_ticker=1)

    strategy = StrategyEngine(config, bus)
    execution = ExecutionEngine(config, bus, rest)

    await strategy.start(market_tickers=[TICKER_BTC])
    await execution.start()
    await asyncio.sleep(0.05)

    await bus.publish("btc.mid_price", BtcMidPrice(
        price=67000.0, timestamp=time.time(), asset="BTC"
    ))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=TICKER_BTC, strike_price=67000.0,
        settlement_time_iso="2026-03-01T00:00:00Z",
        last_trade_time_iso="2026-02-28T23:00:00Z",
        asset="BTC",
    ))
    await asyncio.sleep(0.05)

    # OB: yes_bid=45, no_bid=35 → yes_ask=65, no_ask=55
    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", _ob(yes_bid=45, no_bid=35))
    await asyncio.sleep(0.05)

    # p_yes=0.85 → edge on YES side: 85-65=20¢
    # In passive mode, should post at yes_bid=45 instead of crossing at yes_ask=65
    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC, p_yes=0.85,
    ))
    await asyncio.sleep(0.3)

    await strategy.stop()
    await execution.stop()

    assert len(rest.orders) == 1
    # Passive posts at bid price, not ask
    assert rest.orders[0]["yes_price"] == 45
    assert rest.orders[0]["count"] == 1


#  10. MULTI-ASSET SIGNALS

@pytest.mark.asyncio
async def test_multi_asset_independent_signals():
    # BTC and ETH should generate independent signals.
    bus = Bus()
    rest = MockRest()
    config = _cfg()

    strategy = StrategyEngine(config, bus)
    execution = ExecutionEngine(config, bus, rest)

    await strategy.start(market_tickers=[TICKER_BTC, TICKER_ETH])
    await execution.start()
    await asyncio.sleep(0.05)

    # Seed both assets
    await bus.publish("btc.mid_price", BtcMidPrice(
        price=67000.0, timestamp=time.time(), asset="BTC"
    ))
    await bus.publish("eth.mid_price", BtcMidPrice(
        price=1960.0, timestamp=time.time(), asset="ETH"
    ))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=TICKER_BTC, strike_price=67000.0,
        settlement_time_iso="2026-03-01T00:00:00Z",
        last_trade_time_iso="2026-02-28T23:00:00Z", asset="BTC",
    ))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=TICKER_ETH, strike_price=1960.0,
        settlement_time_iso="2026-03-01T00:00:00Z",
        last_trade_time_iso="2026-02-28T23:00:00Z", asset="ETH",
    ))
    await asyncio.sleep(0.05)

    # Publish orderbooks and probabilities for both
    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", _ob(TICKER_BTC, 40, 40))
    await bus.publish(f"kalshi.orderbook.{TICKER_ETH}", _ob(TICKER_ETH, 35, 35))
    await asyncio.sleep(0.05)

    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC, p_yes=0.85,
    ))
    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_ETH, p_yes=0.90,
    ))
    await asyncio.sleep(0.3)

    await strategy.stop()
    await execution.stop()

    # Both should have orders
    tickers_ordered = {o["ticker"] for o in rest.orders}
    assert TICKER_BTC in tickers_ordered
    assert TICKER_ETH in tickers_ordered


#  11. EXECUTION RESUME AFTER HALT

@pytest.mark.asyncio
async def test_resume_after_halt():
    # After risk halt + resume, new signals should be processed.
    bus = Bus()
    rest = MockRest()
    exe = ExecutionEngine(_cfg(order_timeout_ms=5000), bus, rest)
    await exe.start()
    await asyncio.sleep(0.05)

    # Halt
    await bus.publish("kalshi.risk", RiskEvent(
        event_type="disconnect_halt", detail="test", timestamp=time.time()
    ))
    await asyncio.sleep(0.1)
    assert exe.halted

    # Resume
    exe.resume()
    assert not exe.halted

    # New signal should work
    sig = TradeSignal(
        market_ticker=TICKER_BTC, side="yes", action="buy",
        limit_price_cents=60, quantity_contracts=1, edge=0.10,
        p_yes=0.80, timestamp=time.time(),
    )
    await bus.publish("kalshi.trade_signal", sig)
    await asyncio.sleep(0.2)

    assert len(rest.orders) == 1

    await exe.stop()


#  12. INVALID ORDERBOOK BLOCKS TRADE

@pytest.mark.asyncio
async def test_invalid_orderbook_blocks_strategy():
    # Strategy should not emit signals when orderbook is invalid.
    bus = Bus()
    rest = MockRest()
    config = _cfg()

    strategy = StrategyEngine(config, bus)
    execution = ExecutionEngine(config, bus, rest)

    await strategy.start(market_tickers=[TICKER_BTC])
    await execution.start()
    await asyncio.sleep(0.05)

    await bus.publish("btc.mid_price", BtcMidPrice(
        price=67000.0, timestamp=time.time(), asset="BTC"
    ))
    await asyncio.sleep(0.05)

    # Invalid orderbook
    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", _ob(valid=False))
    await asyncio.sleep(0.05)

    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC, p_yes=0.85,
    ))
    await asyncio.sleep(0.2)

    await strategy.stop()
    await execution.stop()

    assert len(rest.orders) == 0


#  13. PERSISTENCE FILTER

@pytest.mark.asyncio
async def test_persistence_filter_delays_signal():
    # Signal should only emit after edge persists for persistence_window_ms.
    bus = Bus()
    rest = MockRest()
    config = _cfg(persistence_window_ms=200, max_contracts_per_ticker=1)

    strategy = StrategyEngine(config, bus)
    execution = ExecutionEngine(config, bus, rest)

    await strategy.start(market_tickers=[TICKER_BTC])
    await execution.start()
    await asyncio.sleep(0.05)

    await bus.publish("btc.mid_price", BtcMidPrice(
        price=67000.0, timestamp=time.time(), asset="BTC"
    ))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=TICKER_BTC, strike_price=67000.0,
        settlement_time_iso="2026-03-01T00:00:00Z",
        last_trade_time_iso="2026-02-28T23:00:00Z", asset="BTC",
    ))
    await asyncio.sleep(0.05)

    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", _ob(yes_bid=40, no_bid=40))
    await asyncio.sleep(0.05)

    # First prob → starts persistence timer, no order yet
    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC, p_yes=0.85,
    ))
    await asyncio.sleep(0.1)
    assert len(rest.orders) == 0  # still waiting

    # Second prob after 200ms → should now pass persistence
    await asyncio.sleep(0.15)
    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC, p_yes=0.85,
    ))
    await asyncio.sleep(0.2)

    await strategy.stop()
    await execution.stop()

    assert len(rest.orders) == 1


#  14. RANGE MARKET MAX EXPIRY CAP

@pytest.mark.asyncio
async def test_range_market_too_far_to_expiry_blocks_signal():
    # Range markets with > range_max_entry_minutes_to_expiry should not emit signals.
    from datetime import datetime, timezone, timedelta

    bus = Bus()
    rest = MockRest()
    # Settlement 2 hours from now — beyond 60 min cap
    settle_ts = datetime.now(timezone.utc) + timedelta(hours=2)
    settle_iso = settle_ts.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

    config = _cfg(
        range_max_entry_minutes_to_expiry=60,
        max_contracts_per_ticker=1,
    )
    strategy = StrategyEngine(config, bus)
    execution = ExecutionEngine(config, bus, rest)

    TICKER_RANGE = "KXBTC-26MAR0219-RANGE"
    await strategy.start(market_tickers=[TICKER_RANGE])
    await execution.start()
    await asyncio.sleep(0.05)

    await bus.publish("btc.mid_price", BtcMidPrice(
        price=68750.0, timestamp=time.time(), asset="BTC"
    ))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=TICKER_RANGE,
        strike_price=68750.0,
        strike_floor=68000.0,
        strike_cap=69500.0,
        settlement_time_iso=settle_iso,
        last_trade_time_iso=settle_iso,
        asset="BTC",
        is_range=True,
    ))
    await bus.publish(f"kalshi.orderbook.{TICKER_RANGE}", _ob(
        ticker=TICKER_RANGE, yes_bid=40, no_bid=40
    ))
    await asyncio.sleep(0.05)

    # After P1b we no longer publish "pass" StrategyDecisions; assert behavior instead.
    signals: list = []
    q_sig = await bus.subscribe("kalshi.trade_signal")
    async def collect_signals():
        for _ in range(5):
            try:
                s = await asyncio.wait_for(q_sig.get(), timeout=0.5)
                signals.append(s)
            except asyncio.TimeoutError:
                break
    collect_task = asyncio.create_task(collect_signals())

    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_RANGE, p_yes=0.70,
    ))
    await asyncio.sleep(0.2)
    collect_task.cancel()
    try:
        await collect_task
    except asyncio.CancelledError:
        pass

    await strategy.stop()
    await execution.stop()

    assert len(rest.orders) == 0
    assert len(signals) == 0, "range_too_far_to_expiry should block any trade signal"


#  15. HOLD REVERSAL EXIT — COOLDOWN MUST NOT BLOCK

@pytest.mark.asyncio
async def test_hold_reversal_exit_not_blocked_by_cooldown():
    # A protective reversal sell must fire immediately even when signal_cooldown_s
    # would ordinarily suppress a same-side signal.
    #
    # Setup:
    # - Inject a hold position for the ticker.
    # - Set reversal conditions (drift and flow both reversed against position).
    # - Set a long signal_cooldown_s (30s) and seed _last_signal_time so the
    # cooldown window is still active.
    # - Expect a sell signal despite the cooldown.
    from argus_kalshi.models import FairProbability, OrderbookState
    import time as _time

    bus = Bus()
    cfg = _cfg(
        signal_cooldown_s=30.0,
        scalp_momentum_min_drift=0.0001,
        hold_flow_reversal_threshold=0.1,
        hold_require_momentum_agreement=False,
        hold_require_flow_agreement=False,
    )
    strategy = StrategyEngine(cfg, bus)
    await strategy.start(market_tickers=[TICKER_BTC])
    await asyncio.sleep(0.05)

    # Inject a synthetic hold position
    strategy._hold_position_qty[TICKER_BTC] = 100   # 1 contract in centicx
    strategy._hold_position_side[TICKER_BTC] = "yes"
    # Pretend a signal was just sent (cooldown active)
    strategy._last_signal_time[TICKER_BTC] = ("yes", _time.monotonic())

    signals = []
    q = await bus.subscribe("kalshi.trade_signal")

    async def _collect():
        for _ in range(5):
            try:
                signals.append(await asyncio.wait_for(q.get(), timeout=0.3))
            except asyncio.TimeoutError:
                break

    collect_task = asyncio.create_task(_collect())

    # Publish truth + metadata so strategy won't gate on stale data
    await bus.publish("btc.mid_price", BtcMidPrice(price=67000.0, timestamp=_time.time(), asset="BTC"))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=TICKER_BTC, strike_price=67000.0,
        settlement_time_iso="2026-03-01T00:00:00Z",
        last_trade_time_iso="2026-02-28T23:00:00Z", asset="BTC",
    ))
    # Orderbook with valid bids
    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", _ob(yes_bid=45, no_bid=45))
    await asyncio.sleep(0.05)

    # Publish probability that creates reversal: hold_side=yes, drift very negative, flow negative
    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC,
        p_yes=0.45,
        drift=-0.01,   # strong negative drift → reversal vs YES hold
    ))
    # Inject negative trade flow via the internal dict directly
    strategy._trade_flow[TICKER_BTC] = -0.5  # reversed flow

    # Re-evaluate with updated flow
    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC,
        p_yes=0.45,
        drift=-0.01,
    ))
    await asyncio.sleep(0.4)
    await collect_task

    await strategy.stop()

    sell_signals = [s for s in signals if getattr(s, "action", None) == "sell"
                    and getattr(s, "source", None) == "mispricing_hold"]
    assert len(sell_signals) >= 1, (
        "Reversal exit must fire despite active cooldown; got signals: "
        + str([getattr(s, "action") for s in signals])
    )


@pytest.mark.asyncio
async def test_hold_agreement_min_magnitude_blocks_noise():
    # When hold_momentum_agreement_min_drift > 0, a near-zero positive drift
    # should be treated as noise and block the hold entry even though its sign
    # technically 'agrees' with the forced side.
    bus = Bus()
    cfg = _cfg(
        hold_require_momentum_agreement=True,
        hold_require_flow_agreement=False,
        hold_momentum_agreement_min_drift=0.005,   # require at least 0.005 drift
        hold_min_divergence_threshold=0.02,
        signal_cooldown_s=0,
    )
    strategy = StrategyEngine(cfg, bus)
    await strategy.start(market_tickers=[TICKER_BTC])
    await asyncio.sleep(0.05)

    signals = []
    q = await bus.subscribe("kalshi.trade_signal")

    async def _collect():
        for _ in range(5):
            try:
                signals.append(await asyncio.wait_for(q.get(), timeout=0.3))
            except asyncio.TimeoutError:
                break

    collect_task = asyncio.create_task(_collect())

    await bus.publish("btc.mid_price", BtcMidPrice(price=67000.0, timestamp=time.time(), asset="BTC"))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=TICKER_BTC, strike_price=67000.0,
        settlement_time_iso="2026-03-01T00:00:00Z",
        last_trade_time_iso="2026-02-28T23:00:00Z", asset="BTC",
    ))
    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", _ob(yes_bid=40, no_bid=40))
    await asyncio.sleep(0.05)

    # Tiny drift (0.0001) — positive so it "agrees" with YES but below min threshold
    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC,
        p_yes=0.70,   # strong divergence vs ask=60¢
        drift=0.0001, # well below hold_momentum_agreement_min_drift=0.005
    ))
    await asyncio.sleep(0.4)
    await collect_task
    await strategy.stop()

    hold_entries = [s for s in signals
                    if getattr(s, "action", None) == "buy"
                    and getattr(s, "source", None) == "mispricing_hold"]
    assert len(hold_entries) == 0, (
        "Near-zero drift below min_drift threshold must block hold entry; "
        f"got {len(hold_entries)} hold buy(s)"
    )


@pytest.mark.asyncio
async def test_hold_agreement_passes_above_min_magnitude():
    # When drift is above hold_momentum_agreement_min_drift and agrees with the
    # forced side, hold entry should be allowed (no spurious block).
    from datetime import datetime, timezone, timedelta

    bus = Bus()
    # For 15m contracts, hold_entry_horizon_seconds returns min(3, max_entry) * 60 = 180s.
    # Settlement must be within that window; 150s (~2.5 min) is safely inside it.
    settle_iso = (datetime.now(timezone.utc) + timedelta(seconds=150)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg = _cfg(
        hold_require_momentum_agreement=True,
        hold_require_flow_agreement=False,
        hold_momentum_agreement_min_drift=0.001,
        hold_min_divergence_threshold=0.02,
        hold_min_entry_minutes_to_expiry=0,  # disable min-entry guard
        max_entry_minutes_to_expiry=10,      # 10-minute entry window
        signal_cooldown_s=0,
        persistence_window_ms=0,
    )
    strategy = StrategyEngine(cfg, bus)
    await strategy.start(market_tickers=[TICKER_BTC])
    await asyncio.sleep(0.05)

    signals = []
    q = await bus.subscribe("kalshi.trade_signal")

    async def _collect():
        for _ in range(5):
            try:
                signals.append(await asyncio.wait_for(q.get(), timeout=0.3))
            except asyncio.TimeoutError:
                break

    collect_task = asyncio.create_task(_collect())

    await bus.publish("btc.mid_price", BtcMidPrice(price=67000.0, timestamp=time.time(), asset="BTC"))
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=TICKER_BTC, strike_price=67000.0,
        settlement_time_iso=settle_iso,
        last_trade_time_iso=settle_iso, asset="BTC",
    ))
    await bus.publish(f"kalshi.orderbook.{TICKER_BTC}", _ob(yes_bid=40, no_bid=40))
    await asyncio.sleep(0.05)

    # Strong drift (0.01) above threshold, agrees with YES divergence
    await bus.publish("kalshi.fair_prob", FairProbability(
        market_ticker=TICKER_BTC,
        p_yes=0.70,
        drift=0.01,  # above min_drift=0.001, agrees with YES forced_side
    ))
    await asyncio.sleep(0.4)
    await collect_task
    await strategy.stop()

    hold_entries = [s for s in signals
                    if getattr(s, "action", None) == "buy"
                    and getattr(s, "source", None) == "mispricing_hold"]
    assert len(hold_entries) >= 1, (
        "Drift above min_drift threshold agreeing with forced_side must allow hold entry"
    )
