# Created by Oliver Meihls

from __future__ import annotations

import time

import pytest

from argus_kalshi.bus import Bus
from argus_kalshi.ipc import StateAggregator
from argus_kalshi.models import FillEvent, MarketMetadata, OrderUpdate, SettlementOutcome


@pytest.mark.asyncio
async def test_drain_for_snapshot_updates_bot_stats_in_separate_ui_mode():
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id="bot_a")
    await agg.start(run_aggregate_loop=False)

    await bus.publish(
        "kalshi.fills",
        FillEvent(
            market_ticker="KXBTC-TEST",
            order_id="paper-1",
            side="yes",
            price_cents=55,
            count=400,
            is_taker=True,
            timestamp=time.time(),
            source="mispricing_scalp",
            action="buy",
            bot_id="bot_a",
        ),
    )
    await bus.publish(
        "kalshi.user_orders",
        OrderUpdate(
            market_ticker="KXBTC-TEST",
            order_id="paper-1",
            status="filled",
            side="yes",
            price_cents=55,
            quantity_contracts=4,
            filled_contracts=4,
            remaining_contracts=0,
            timestamp=time.time(),
            bot_id="bot_a",
        ),
    )
    await bus.publish(
        "kalshi.settlement_outcome",
        SettlementOutcome(
            market_ticker="KXBTC-TEST",
            side="yes",
            won=True,
            pnl=3.5,
            quantity_centicx=400,
            entry_price_cents=55,
            final_avg=68100.0,
            strike=68000.0,
            timestamp=time.time(),
            source="mispricing_scalp",
            bot_id="bot_a",
        ),
    )

    agg.drain_for_snapshot()
    snap = agg.get_snapshot(max_bot_stats=100)

    bot = snap["bot_stats"]["bot_a"]
    assert bot["fills"] == 1
    assert bot["orders"] == 1
    assert bot["trade_count"] == 1
    assert bot["wins"] == 1
    assert bot["pnl"] == 3.5
    assert bot["pnl_s"] == 3.5
    assert snap["history"]
    assert snap["recent_fills"]


@pytest.mark.asyncio
async def test_partial_fill_decrements_open_order_and_bucket_count():
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id="bot_a")
    await agg.start(run_aggregate_loop=False)

    await bus.publish(
        "kalshi.market_metadata",
        MarketMetadata(
            market_ticker="KXBTC15M-TEST-000",
            strike_price=70000.0,
            settlement_time_iso="2026-03-05T00:00:00Z",
            last_trade_time_iso="2026-03-05T00:00:00Z",
            asset="BTC",
            window_minutes=15,
            is_range=False,
            status="open",
        ),
    )
    await bus.publish(
        "kalshi.user_orders",
        OrderUpdate(
            market_ticker="KXBTC15M-TEST-000",
            order_id="paper-abc",
            status="placed",
            side="yes",
            price_cents=55,
            quantity_contracts=4,
            filled_contracts=0,
            remaining_contracts=4,
            timestamp=time.time(),
            bot_id="bot_a",
        ),
    )
    await bus.publish(
        "kalshi.user_orders",
        OrderUpdate(
            market_ticker="KXBTC15M-TEST-000",
            order_id="paper-abc",
            status="partial_fill",
            side="yes",
            price_cents=55,
            quantity_contracts=2,
            filled_contracts=2,
            remaining_contracts=2,
            timestamp=time.time(),
            bot_id="bot_a",
        ),
    )

    agg.drain_for_snapshot()
    snap = agg.get_snapshot(max_bot_stats=20)
    assert snap["open_orders"] == 0
    assert (snap.get("market_fill_counts") or {}).get("BTC|15min", 0) == 0


def test_overlay_does_not_reduce_cumulative_trade_counters():
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id="bot_a")

    # Simulate persisted/base all-time stats.
    agg.seed_from_jsonl(
        {
            "bot_a": {
                "pnl": 100.0,
                "pnl_e": 100.0,
                "pnl_s": 0.0,
                "fills": 2,
                "orders": 2,
                "wins": 5,
                "losses": 0,
                "trade_count": 5,
                "last_active": 0.0,
            }
        },
        primary_pnl=100.0,
        primary_wins=5,
        primary_losses=0,
    )

    # Simulate session overlay from farm ledger with lower trade_count.
    agg.set_bot_stats_overlay_provider(lambda: {"bot_a": {"trade_count": 1, "generation": 0}})
    snap = agg.get_snapshot(max_bot_stats=20)
    bot = snap["bot_stats"]["bot_a"]
    assert bot["wins"] == 5
    assert bot["trade_count"] == 5
