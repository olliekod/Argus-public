import asyncio
import time
import uuid
from argus_kalshi.bus import Bus
from argus_kalshi.config import KalshiConfig
from argus_kalshi.kalshi_execution import ExecutionEngine
from argus_kalshi.models import TradeSignal, OrderbookState, FillEvent
from argus_kalshi.logging_utils import ComponentLogger

async def test_fidelity():
    bus = Bus()
    cfg = KalshiConfig(dry_run=True, paper_slippage_cents=1)
    engine = ExecutionEngine(cfg, bus, None)
    
    ticker = "TEST-MKT"
    
    print("\n--- Testing Insufficient Depth ---")
    # Manually populate engine's orderbook cache
    ob_low = OrderbookState(
        market_ticker=ticker,
        best_yes_bid_cents=50,
        best_no_bid_cents=48,
        implied_yes_ask_cents=52,
        implied_no_ask_cents=50,
        seq=1,
        best_yes_depth=10, 
        best_no_depth=10,
        valid=True
    )
    engine._orderbooks[ticker] = ob_low
    
    signal = TradeSignal(
        market_ticker=ticker,
        action="buy",
        side="yes",
        limit_price_cents=55,
        quantity_contracts=5,
        edge=0.05,
        p_yes=0.6,
        timestamp=time.time(),
        source="test"
    )
    
    print(f"Placing order for 5 contracts (Depth available: 10 centicx)...")
    
    fill_sub = await bus.subscribe("kalshi.fills")
    await engine._execute_signal(signal)
    
    await asyncio.sleep(0.1)
    if not fill_sub.empty():
        print("FAILED: Order was filled despite insufficient depth!")
        while not fill_sub.empty(): fill_sub.get_nowait()
    else:
        print("SUCCESS: Order was correctly skipped due to insufficient depth.")

    print("\n--- Testing Slippage ---")
    # Increase depth
    ob_high = OrderbookState(
        market_ticker=ticker,
        best_yes_bid_cents=50,
        best_no_bid_cents=48,
        implied_yes_ask_cents=52,
        implied_no_ask_cents=50,
        seq=2,
        best_yes_depth=5000,
        best_no_depth=5000,
        valid=True
    )
    engine._orderbooks[ticker] = ob_high
    
    print(f"Placing order for 1 contract (Limit: 55c, Expected Fill: 56c due to 1c slip)...")
    engine._pending_by_market.clear()
    engine._last_buy_signal_ts.clear()
    
    signal.quantity_contracts = 1
    await engine._execute_signal(signal)
    
    await asyncio.sleep(0.1)
    if not fill_sub.empty():
        fill: FillEvent = fill_sub.get_nowait()
        print(f"Fill received at {fill.price_cents}c")
        if fill.price_cents == 56:
            print("SUCCESS: 1c slippage applied correctly.")
        else:
            print(f"FAILED: Expected 56c, got {fill.price_cents}c")
    else:
        print("FAILED: No fill received.")

if __name__ == "__main__":
    asyncio.run(test_fidelity())
