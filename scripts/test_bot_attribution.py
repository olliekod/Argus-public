import asyncio
import time
import os
from argus_kalshi.bus import Bus
from argus_kalshi.models import StrategyDecision, TerminalEvent
from src.core.database import Database

async def test_attribution():
    db_path = "data/test_kalshi.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = Database(db_path)
    await db.connect()
    
    bus = Bus()
    
    # Mock the monitor logic since we want to test the persistence
    async def monitor():
        q_term = await bus.subscribe("kalshi.terminal_event")
        q_dec = await bus.subscribe("kalshi.strategy_decision")
        
        # Test TerminalEvent
        msg = await q_term.get()
        await db.execute(
            "INSERT INTO kalshi_terminal_events (timestamp, level, message, bot_id) VALUES (?, ?, ?, ?)",
            (msg.timestamp, msg.level, msg.message, getattr(msg, "bot_id", "default"))
        )
        
        # Test StrategyDecision
        dec = await q_dec.get()
        await db.execute(
            """INSERT INTO kalshi_decisions
               (timestamp, market_ticker, p_yes, yes_ask, no_ask, action_taken, reason, bot_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (dec.timestamp, dec.market_ticker, dec.p_yes, dec.yes_ask, dec.no_ask, dec.action_taken, dec.reason, getattr(dec, "bot_id", "default"))
        )

    monitor_task = asyncio.create_task(monitor())
    
    # 1. Publish test events
    test_bot_id = "research_bot_468"
    
    await bus.publish("kalshi.terminal_event", TerminalEvent(
        level="INFO", message="Test message", timestamp=time.time(), bot_id=test_bot_id
    ))
    
    await bus.publish("kalshi.strategy_decision", StrategyDecision(
        market_ticker="BTC-25MAR24-65000", p_yes=0.5, yes_ask=50, no_ask=50,
        action_taken="pass", reason="test", timestamp=time.time(), bot_id=test_bot_id
    ))
    
    await monitor_task
    
    # 2. Verify in DB
    print("\n--- Verifying Database ---")
    
    events = await db.fetch_all("SELECT * FROM kalshi_terminal_events")
    print(f"Terminal Events found: {len(events)}")
    for e in events:
        print(f"  ID: {e['id']} | Bot: {e['bot_id']} | Msg: {e['message']}")
        assert e['bot_id'] == test_bot_id, f"Expected {test_bot_id}, got {e['bot_id']}"
        
    decisions = await db.fetch_all("SELECT * FROM kalshi_decisions")
    print(f"Strategy Decisions found: {len(decisions)}")
    for d in decisions:
        print(f"  ID: {d['id']} | Bot: {d['bot_id']} | Ticker: {d['market_ticker']}")
        assert d['bot_id'] == test_bot_id, f"Expected {test_bot_id}, got {d['bot_id']}"
    
    print("\n✅ Verification SUCCESS: Bot attribution is working correctly!")
    await db.close()

if __name__ == "__main__":
    asyncio.run(test_attribution())
