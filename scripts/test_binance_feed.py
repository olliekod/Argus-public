
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectors.binance_ws import BinanceWebSocket

async def test_feed():
    print("--- Starting Binance WebSocket Smoke Test ---")
    
    received_ticks = []
    
    async def on_ticker(data):
        price = data.get('last_price')
        symbol = data.get('symbol')
        print(f"[{time.strftime('%H:%M:%S')}] Received tick: {symbol} @ {price}")
        received_ticks.append(data)

    # Initialize for BTC/USDT
    ws = BinanceWebSocket(symbols=['BTC/USDT'], on_ticker=on_ticker)
    
    # Start connection in background
    task = asyncio.create_task(ws.connect())
    
    print("Connecting... waiting up to 10 seconds for data...")
    
    start_time = time.time()
    while time.time() - start_time < 10:
        if len(received_ticks) >= 3:
            print(f"\nSUCCESS: Received {len(received_ticks)} ticks from Binance!")
            break
        await asyncio.sleep(1)
    
    if len(received_ticks) < 3:
        print(f"\nFAILED: Only received {len(received_ticks)} ticks in 10 seconds.")
        if ws.last_error:
            print(f"Last error: {ws.last_error}")
    
    # Clean up
    print("Shutting down...")
    await ws.disconnect()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(test_feed())
