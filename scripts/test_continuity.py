import asyncio
import time
import os
import json
from argus_kalshi.bus import Bus
from argus_kalshi.settlement_tracker import SettlementTracker, Position
from argus_kalshi.models import FillEvent, MarketMetadata

async def test_continuity():
    log_path = "logs/paper_trades.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)
    
    bus = Bus()
    tracker = SettlementTracker(bus)
    
    # 1. Simulate a fill being recorded
    ticker = "KXBTC-CONT-TEST"
    bot_id = "continuity_bot"
    
    # We need to mock the fill event as it would appear in the log
    # The actual execution engine writes this.
    record = {
        "type": "paper_fill",
        "order_id": "test-oid",
        "market_ticker": ticker,
        "side": "yes",
        "fill_price_cents": 50,
        "quantity_contracts": 1.0,
        "settlement_time": time.time() + 100,
        "strike": 65000,
        "asset": "BTC",
        "bot_id": bot_id,
        "timestamp": time.time()
    }
    
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        f.write(json.dumps(record) + "\n")
        
    print(f"Recorded simulated fill for {ticker} in {log_path}")
    
    # 2. Start tracker (this should trigger recovery)
    await tracker.start()
    
    # Give it a moment to load
    await asyncio.sleep(0.1)
    
    # 3. Verify position exists in tracker
    pos_map = tracker._positions_for_bot(bot_id)
    if ticker in pos_map:
        pos = pos_map[ticker]
        print(f"✅ SUCCESS: Recovered position for {ticker}")
        print(f"   Qty: {pos.total_qty} | Side: {pos.side} | Entry: {pos.avg_price_cents}¢")
        assert pos.total_qty == 100
        assert pos.side == "yes"
    else:
        print(f"❌ FAILURE: Position for {ticker} not found in tracker")
        exit(1)
        
    # 4. Simulate a settlement
    settlement_record = {
        "type": "settlement",
        "market_ticker": ticker,
        "bot_id": bot_id,
        "timestamp": time.time()
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(settlement_record) + "\n")
        
    print(f"Recorded simulated settlement for {ticker}")
    
    # 5. Restart tracker and verify ticker is GONE
    tracker.stop()
    new_tracker = SettlementTracker(bus)
    await new_tracker.start()
    await asyncio.sleep(0.1)
    
    new_pos_map = new_tracker._positions_for_bot(bot_id)
    if ticker not in new_pos_map:
        print(f"✅ SUCCESS: Settled position for {ticker} correctly cleared on restart")
    else:
        print(f"❌ FAILURE: Settled position for {ticker} still exists in tracker")
        exit(1)

    print("\n🎉 Trade Continuity Verification COMPLETE!")
    new_tracker.stop()

if __name__ == "__main__":
    asyncio.run(test_continuity())
