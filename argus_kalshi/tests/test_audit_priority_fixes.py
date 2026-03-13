from __future__ import annotations

import asyncio

import pytest

from argus_kalshi.bus import Bus
from argus_kalshi.kalshi_probability import compute_probability
from argus_kalshi.models import BtcMidPrice
from argus_kalshi.settlement_tracker import Position, SettlementTracker


@pytest.mark.asyncio
async def test_bus_bounded_queue_drops_oldest_on_overflow() -> None:
    bus = Bus(subscriber_queue_maxsize=1)
    q = await bus.subscribe("topic")

    await bus.publish("topic", 1)
    await bus.publish("topic", 2)

    assert q.qsize() == 1
    assert await q.get() == 2


def test_btc_mid_price_has_source_field_default() -> None:
    tick = BtcMidPrice(price=10.0, timestamp=1.0)
    assert tick.source == "unknown"


def test_probability_tail_scale_increases_uncertainty() -> None:
    # For strike above spot, larger tails should increase chance of YES.
    base = compute_probability(
        strike=110.0,
        current_price=100.0,
        sigma=0.01,
        time_to_settle_s=120.0,
        tail_scale=1.0,
    )
    fat_tail = compute_probability(
        strike=110.0,
        current_price=100.0,
        sigma=0.01,
        time_to_settle_s=120.0,
        tail_scale=2.5,
    )
    assert fat_tail > base


@pytest.mark.asyncio
async def test_settlement_at_strike_resolves_yes() -> None:
    class _DB:
        def __init__(self) -> None:
            self.last = None

        async def insert_kalshi_outcome(self, **kwargs):
            self.last = kwargs
            return True

    bus = Bus()
    db = _DB()
    tracker = SettlementTracker(bus, db=db)
    pos = Position(
        market_ticker="KXBTC-TEST",
        side="yes",
        avg_price_cents=50,
        total_qty=100,
        settlement_time=0.0,
        strike=65000.0,
    )

    q_out = await bus.subscribe("kalshi.settlement_outcome")

    await tracker._resolve_settlement(pos, final_avg=65000.0, bot_id="farm_001")
    out = await q_out.get()

    assert out.bot_id == "farm_001"
    assert db.last is not None
    assert db.last["outcome"] == "WON"
    assert db.last["bot_id"] == "farm_001"


def test_runner_shutdown_cleanup_not_nested_under_luzia() -> None:
    source = open("argus_kalshi/runner.py", encoding="utf-8").read()
    assert "if settlement:" in source
    luzia_idx = source.index("if luzia_feed is not None:")
    settlement_idx = source.index("if settlement:")
    assert settlement_idx > luzia_idx
    # Ensure settlement cleanup is top-level in finally, not nested in luzia block.
    snippet = source[luzia_idx:settlement_idx]
    assert "if settlement:" not in snippet
