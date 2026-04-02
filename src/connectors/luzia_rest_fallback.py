# Created by Oliver Meihls

# Luzia.dev Unified REST Polling Connector
#
# Provides aggregate crypto ticks from multiple exchanges via REST polling.
# Used as a resilient, last-line fallback when WebSockets are unavailable.
#
# Designed for Free Tier:
# - 100 requests/minute
# - 5,000 requests/day
# (Polls every ~15-30s to stay safely under daily limits, or faster on-demand).

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import aiohttp

from ..core.logger import get_connector_logger

logger = get_connector_logger('luzia_rest')


class LuziaPollingFeed:
    # Luzia.dev REST Polling client.
    #
    # Used as an insurance policy when primary WS feeds are down.
    
    BASE_URL = "https://api.luzia.dev/v1/ticker"
    
    def __init__(
        self,
        api_key: str,
        # Format: ["binance/BTC-USDT", "coinbase/BTC-USD"]
        endpoints: List[str],
        on_ticker: Optional[Callable] = None,
        interval_seconds: float = 15.0 # Slow poll to save daily quota
    ):
        self._api_key = api_key
        self.endpoints = endpoints
        self.on_ticker = on_ticker
        self.interval = interval_seconds
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Deduplication cache
        self._last_mid_prices: Dict[str, float] = {}
        # Consecutive errors → exponential backoff (cap 5 min)
        self._consecutive_errors = 0
        self._base_backoff = 5.0
        self._max_backoff = 300.0
        
        logger.info(f"Luzia Polling Feed initialized for {self.endpoints} every {self.interval}s")

    async def start(self) -> None:
        # Start polling loop.
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        # Stop polling loop.
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json"
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            while self._running:
                try:
                    for endpoint in self.endpoints:
                        # Ensure no double slashes and correct path
                        clean_endpoint = endpoint.lstrip('/')
                        url = f"{self.BASE_URL.rstrip('/')}/{clean_endpoint}?maxAge=15000"
                        
                        logger.debug(f"Polling Luzia: {url}")
                        async with session.get(url, timeout=5) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                await self._handle_ticker(data)
                            elif resp.status == 429:
                                logger.warning("Luzia Rate Limit hit (429). Slowing down.")
                                await asyncio.sleep(60)
                            elif resp.status == 503:
                                # Often 503 "Stale Data" if Luzia hasn't seen a update recently.
                                # Since this is a fallback, we log as info to keep the console clean.
                                logger.info(f"Luzia feed stale (503) for {url} - skipping.")
                            else:
                                text = await resp.text()
                                logger.error(f"Luzia REST error: {resp.status} for {url} - {text[:200]}")
                        
                        if not self._running:
                            break
                        # Stagger calls between multiple endpoints
                        await asyncio.sleep(1.0)
                        
                except Exception as e:
                    self._consecutive_errors += 1
                    logger.error(
                        "Luzia Polling error: %s — %s (endpoints=%s)",
                        type(e).__name__, e, self.endpoints,
                    )
                else:
                    self._consecutive_errors = 0
                
                if self._running:
                    if self._consecutive_errors > 0:
                        sleep_time = min(
                            self._base_backoff * (2 ** min(self._consecutive_errors, 6)),
                            self._max_backoff,
                        )
                    else:
                        sleep_time = self.interval
                    await asyncio.sleep(sleep_time)

    async def _handle_ticker(self, payload: Dict[str, Any]) -> None:
        # Parse ticker payload and invoke callback.
        # Luzia REST Format: { "symbol": "BTC/USDT", "exchange": "binance", "last": ... }
        exchange = payload.get('exchange', 'luzia')
        symbol = payload.get('symbol', 'unknown')
        # Luzia may return 'last', 'last_price', or 'price' depending on endpoint.
        raw = payload.get('last') or payload.get('last_price') or payload.get('price') or 0
        price = float(raw) if raw is not None else 0.0
        
        cache_key = f"{exchange}:{symbol}"
        if self._last_mid_prices.get(cache_key) == price:
            return
        self._last_mid_prices[cache_key] = price

        parsed = {
            'symbol': symbol.replace('/', ''),
            'exchange': f"luzia:{exchange}",
            'timestamp': payload.get('timestamp', datetime.now(timezone.utc).isoformat()),
            'last_price': price,
            'bid_price': float(payload.get('bid', 0)),
            'ask_price': float(payload.get('ask', 0)),
            'volume_24h': float(payload.get('volume', 0)),
        }
        
        if self.on_ticker:
            try:
                if asyncio.iscoroutinefunction(self.on_ticker):
                    await self.on_ticker(parsed)
                else:
                    self.on_ticker(parsed)
            except Exception as e:
                logger.error(f"Luzia Ticker callback error: {e}")
