# Created by Oliver Meihls

import asyncio

from src.core.database import Database

async def test():
    db = Database('data/argus.db')
    await db.connect()
    
    await db.execute(
        """INSERT INTO kalshi_decisions
           (timestamp, market_ticker, p_yes, yes_ask, no_ask, action_taken, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (1234.0, "KXBTC", 0.5, 50, 50, "pass", "test")
    )
    
    rows = await db.fetch_all("SELECT * FROM kalshi_decisions")
    print(rows)
    
    await db.close()

asyncio.run(test())
