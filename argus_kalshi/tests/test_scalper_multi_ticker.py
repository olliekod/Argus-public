import asyncio
import time
import pytest

from argus_kalshi.bus import Bus
from argus_kalshi.config import KalshiConfig
from argus_kalshi.mispricing_scalper import MispricingScalper
from argus_kalshi.models import (
    AccountBalance,
    FairProbability,
    KalshiOrderDeltaEvent,
    KalshiTradeEvent,
    MarketMetadata,
    OrderbookState,
    SelectedMarkets,
)

def _make_config(**overrides):
    defaults = dict(
        bankroll_usd=1000.0,
        dry_run=True,
        scalper_enabled=True,
        scalp_min_edge_cents=5,
        scalp_min_profit_cents=6,
        scalp_max_spread_cents=2,
        scalp_cooldown_s=0.1,
        base_contract_qty=1,
        scalp_max_quantity=10,
        scalp_min_reprice_move_cents=4,
        scalp_reprice_window_s=3.0,
        scalp_entry_cost_buffer_cents=0,
    )
    defaults.update(overrides)
    return KalshiConfig(**defaults)

def _make_orderbook(ticker, yes_ask, no_ask, yes_bid=None, no_bid=None, obi=0.0, depth_pressure=0.0):
    # implied_yes_ask = 100 - best_no_bid
    # implied_no_ask = 100 - best_yes_bid
    if yes_bid is None: yes_bid = yes_ask - 1
    if no_bid is None: no_bid = no_ask - 1
    return OrderbookState(
        market_ticker=ticker,
        best_yes_bid_cents=yes_bid,
        best_no_bid_cents=no_bid,
        implied_yes_ask_cents=yes_ask,
        implied_no_ask_cents=no_ask,
        seq=1,
        valid=True,
        obi=obi,
        depth_pressure=depth_pressure,
    )


async def _publish_directional_flow(bus: Bus, ticker: str, *, yes_trade: int, no_trade: int, yes_delta: int, no_delta: int) -> None:
    now = time.time()
    if yes_trade > 0:
        await bus.publish(
            "kalshi.trade",
            KalshiTradeEvent(market_ticker=ticker, taker_side="yes", count=yes_trade, ts=now),
        )
    if no_trade > 0:
        await bus.publish(
            "kalshi.trade",
            KalshiTradeEvent(market_ticker=ticker, taker_side="no", count=no_trade, ts=now),
        )
    if yes_delta > 0:
        await bus.publish(
            "kalshi.orderbook_delta_flow",
            KalshiOrderDeltaEvent(market_ticker=ticker, side="yes", is_add=True, qty=yes_delta, ts=now),
        )
    if no_delta > 0:
        await bus.publish(
            "kalshi.orderbook_delta_flow",
            KalshiOrderDeltaEvent(market_ticker=ticker, side="no", is_add=False, qty=no_delta, ts=now),
        )

@pytest.mark.asyncio
async def test_scalper_multi_ticker_execution():
    bus = Bus()
    cfg = _make_config()
    scalper = MispricingScalper(cfg, bus)
    
    signals_q = await bus.subscribe("kalshi.trade_signal")
    
    await scalper.start()
    await asyncio.sleep(0.05)
    
    T1 = "KXBTC-1"
    T2 = "KXBTC-2"
    
    # 1. Provide metadata for eligibility
    for t in [T1, T2]:
        await bus.publish("kalshi.market_metadata", MarketMetadata(
            market_ticker=t, strike_price=60000, settlement_time_iso="2026-01-01T00:00:00Z",
            last_trade_time_iso="2026-01-01T00:00:00Z", window_minutes=15, is_range=False
        ))
    
    # 2. Publish ranked markets
    await bus.publish("kalshi.ranked_markets", SelectedMarkets(tickers=[T1, T2], timestamp=time.time()))
    await asyncio.sleep(0.1)
    
    # 3. Setup mispricing for T1
    await _publish_directional_flow(bus, T1, yes_trade=40, no_trade=5, yes_delta=80, no_delta=20)
    await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=T1, p_yes=0.55, drift=0.00025))
    await bus.publish(
        f"kalshi.orderbook.{T1}",
        _make_orderbook(T1, yes_ask=45, no_ask=54, obi=0.8, depth_pressure=0.9),
    )

    # 4. Setup mispricing for T2
    await _publish_directional_flow(bus, T2, yes_trade=5, no_trade=40, yes_delta=20, no_delta=80)
    await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=T2, p_yes=0.45, drift=-0.00025))
    await bus.publish(
        f"kalshi.orderbook.{T2}",
        _make_orderbook(T2, yes_ask=55, no_ask=44, obi=-0.8, depth_pressure=-0.9),
    )
    
    await asyncio.sleep(0.2)
    
    signals = []
    while not signals_q.empty():
        signals.append(await signals_q.get())
    
    await scalper.stop()
    
    assert len(signals) >= 2
    tickers = {s.market_ticker for s in signals}
    assert T1 in tickers
    assert T2 in tickers

@pytest.mark.asyncio
async def test_scalper_respects_config_thresholds():
    bus = Bus()
    cfg = _make_config(scalp_directional_score_threshold=0.35, scalp_max_spread_cents=2)
    scalper = MispricingScalper(cfg, bus)
    
    signals_q = await bus.subscribe("kalshi.trade_signal")
    
    await scalper.start()
    T_A = "KX-A"
    T_B = "KX-B"
    T_C = "KX-C"
    for t in [T_A, T_B, T_C]:
        await bus.publish("kalshi.market_metadata", MarketMetadata(
            market_ticker=t, strike_price=60000, settlement_time_iso="2026-01-01T00:00:00Z",
            last_trade_time_iso="2026-01-01T00:00:00Z", window_minutes=15, is_range=False
        ))
    await bus.publish("kalshi.ranked_markets", SelectedMarkets(tickers=[T_A, T_B, T_C], timestamp=time.time()))
    await asyncio.sleep(0.1)
    
    # Case A: Directional score too small (< threshold)
    await _publish_directional_flow(bus, T_A, yes_trade=10, no_trade=9, yes_delta=10, no_delta=9)
    await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=T_A, p_yes=0.50, drift=0.00001))
    await bus.publish(
        f"kalshi.orderbook.{T_A}",
        _make_orderbook(T_A, yes_ask=45, no_ask=54, obi=0.05, depth_pressure=0.05),
    )
    await asyncio.sleep(0.2)
    assert signals_q.empty()
    
    # Case B: Spread too wide (6c > 2c)
    await _publish_directional_flow(bus, T_B, yes_trade=30, no_trade=3, yes_delta=40, no_delta=5)
    await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=T_B, p_yes=0.65, drift=0.00025))
    await bus.publish(
        f"kalshi.orderbook.{T_B}",
        _make_orderbook(T_B, yes_ask=45, no_ask=55, yes_bid=42, no_bid=52, obi=0.9, depth_pressure=0.9),
    )
    await asyncio.sleep(0.2)
    assert signals_q.empty()
    
    # Case C: Both fine
    await _publish_directional_flow(bus, T_C, yes_trade=30, no_trade=3, yes_delta=40, no_delta=5)
    await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=T_C, p_yes=0.70, drift=0.00025))
    await bus.publish(
        f"kalshi.orderbook.{T_C}",
        _make_orderbook(T_C, yes_ask=45, no_ask=54, yes_bid=44, no_bid=53, obi=0.9, depth_pressure=0.9),
    )
    await asyncio.sleep(0.2)
    assert not signals_q.empty()
    
    await scalper.stop()


@pytest.mark.asyncio
async def test_scalper_dynamic_sizing_from_balance_updates():
    bus = Bus()
    cfg = _make_config(risk_fraction_per_trade=0.01, scalp_max_quantity=1000)
    scalper = MispricingScalper(cfg, bus)
    q = await bus.subscribe("kalshi.trade_signal")

    ticker = "KXBTC-DYN"
    await scalper.start()
    await bus.publish("kalshi.market_metadata", MarketMetadata(
        market_ticker=ticker, strike_price=60000, settlement_time_iso="2026-01-01T00:00:00Z",
        last_trade_time_iso="2026-01-01T00:00:00Z", window_minutes=15, is_range=False
    ))
    await bus.publish("kalshi.ranked_markets", SelectedMarkets(tickers=[ticker], timestamp=time.time()))
    await asyncio.sleep(0.1)

    await bus.publish("kalshi.account_balance", AccountBalance(balance_cents=100_000, timestamp=time.time()))
    await _publish_directional_flow(bus, ticker, yes_trade=35, no_trade=2, yes_delta=40, no_delta=5)
    await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=ticker, p_yes=0.70, drift=0.00025))
    await bus.publish(
        f"kalshi.orderbook.{ticker}",
        _make_orderbook(ticker, yes_ask=50, no_ask=49, obi=0.8, depth_pressure=0.9),
    )
    await asyncio.sleep(0.15)
    sig1 = await q.get()

    # Increase balance 2x and reset cooldown window.
    await asyncio.sleep(cfg.scalp_cooldown_s + 0.05)
    await bus.publish("kalshi.account_balance", AccountBalance(balance_cents=200_000, timestamp=time.time()))
    await _publish_directional_flow(bus, ticker, yes_trade=35, no_trade=2, yes_delta=40, no_delta=5)
    await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=ticker, p_yes=0.70, drift=0.00025))
    await bus.publish(
        f"kalshi.orderbook.{ticker}",
        _make_orderbook(ticker, yes_ask=50, no_ask=49, obi=0.8, depth_pressure=0.9),
    )
    await asyncio.sleep(0.15)
    sig2 = await q.get()

    await scalper.stop()

    assert sig1.quantity_contracts > 0
    assert sig2.quantity_contracts > sig1.quantity_contracts
