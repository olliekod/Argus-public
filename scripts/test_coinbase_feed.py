
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectors.coinbase_client import CoinbaseClient

async def test_feed():
    print("--- Starting Coinbase REST Smoke Test ---")
    
    received_ticks = []
    
    async def on_ticker(data):
        price = data.get('last_price')
        symbol = data.get('symbol')
        print(f"[{time.strftime('%H:%M:%S')}] Received tick: {symbol} @ {price}")
        received_ticks.append(data)

    # Initialize for BTC-USD
    client = CoinbaseClient(symbols=['BTC/USDT'], on_ticker=on_ticker)
    
    # Start polling in background (short interval for test)
    print("Starting polling loop (2s interval)...")
    task = asyncio.create_task(client.poll(interval_seconds=2))
    
    print("Waiting up to 10 seconds for data...")
    
    start_time = time.time()
    while time.time() - start_time < 10:
        if len(received_ticks) >= 2:
            print(f"\nSUCCESS: Received {len(received_ticks)} prices from Coinbase!")
            break
        await asyncio.sleep(1)
    
    if len(received_ticks) < 2:
        print(f"\nFAILED: Only received {len(received_ticks)} prices in 10 seconds.")
        if client.last_error:
            print(f"Last error: {client.last_error}")
    
    # Clean up
    print("Shutting down...")
    await client.close()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(test_feed())
