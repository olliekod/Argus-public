"""
Coinbase WebSocket Connector
============================

High-frequency WebSocket client for Coinbase spot market data.
Uses the public feed - no authentication required for ticker data.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import websockets
from websockets.exceptions import ConnectionClosed

from ..core.logger import get_connector_logger
from ..core.events import QuoteEvent, TOPIC_MARKET_QUOTES

logger = get_connector_logger('coinbase_ws')


class CoinbaseWebSocket:
    """
    Coinbase WebSocket client for spot market data.
    
    Provides sub-second price ticks via the 'ticker' channel.
    US-friendly and extremely reliable.
    """
    
    BASE_URL = "wss://ws-feed.exchange.coinbase.com"
    
    def __init__(
        self,
        symbols: List[str],
        on_ticker: Optional[Callable] = None,
        event_bus: Optional[Any] = None,
    ):
        """
        Initialize Coinbase WebSocket client.
        
        Args:
            symbols: List of symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            on_ticker: Callback for ticker updates
            event_bus: Optional Argus event bus for publishing QuoteEvents
        """
        self.symbols = [self._normalize_symbol(s) for s in symbols]
        self.on_ticker = on_ticker
        self._event_bus = event_bus
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_delay = 1
        self._connected_since: Optional[float] = None
        
        # Metrics
        self.last_message_ts: Optional[float] = None
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures: int = 0
        self.reconnect_attempts: int = 0
        self.request_count: int = 0
        self.error_count: int = 0
        
        # Latest data cache
        self.tickers: Dict[str, Dict] = {}
        self._last_mid_prices: Dict[str, float] = {}
        
        logger.info(f"Coinbase WebSocket initialized for {self.symbols}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert unified symbol to Coinbase format (e.g. BTC-USD)."""
        if ':' in symbol:
            symbol = symbol.split(':')[0]
        base = symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '')
        return f"{base}-USD"

    async def connect(self) -> None:
        """Start WebSocket connection and message loop."""
        self._running = True
        retry_count = 0
        
        while self._running:
            try:
                logger.info(f"Connecting to Coinbase WebSocket: {self.BASE_URL}")
                
                async with websockets.connect(
                    self.BASE_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    open_timeout=20,
                ) as ws:
                    self._ws = ws
                    self._connected_since = time.time()
                    retry_count = 0
                    
                    # Subscribe to ticker channel
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": self.symbols,
                        "channels": ["ticker"]
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to ticker for {self.symbols}")
                    
                    # Message loop
                    await self._message_loop()
                    
            except ConnectionClosed as e:
                self.last_error = f"closed:{e.code}"
                self.error_count += 1
                logger.warning(f"Coinbase WebSocket closed: {e.code}")
            except Exception as e:
                self.last_error = str(e)
                self.error_count += 1
                logger.error(f"Coinbase WebSocket error: {e}")
            
            if self._running:
                retry_count += 1
                self.reconnect_attempts += 1
                delay = min(self._reconnect_delay * (2 ** min(retry_count, 5)), 30)
                logger.info(f"Reconnecting in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        self._ws = None

    async def disconnect(self) -> None:
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("Coinbase WebSocket disconnected")

    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        if not self._ws:
            return
        
        async for message in self._ws:
            try:
                self.last_message_ts = time.time()
                self.last_success_ts = self.last_message_ts
                self.request_count += 1
                self.consecutive_failures = 0
                
                data = json.loads(message)
                if data.get('type') == 'ticker':
                    await self._handle_ticker(data)
                elif data.get('type') == 'error':
                    logger.error(f"Coinbase WS error message: {data.get('message')}")
                    
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error handling Coinbase message: {e}")

    async def _handle_ticker(self, data: Dict[str, Any]) -> None:
        """Parse ticker message and invoke callback."""
        try:
            symbol = data.get('product_id', '').replace('-', '')
            
            # Coinbase sends everything as strings
            price = float(data.get('price', 0))
            best_bid = float(data.get('best_bid', 0))
            best_ask = float(data.get('best_ask', 0))
            
            # Deduplicate - only forward if the mid price has changed
            # This follows "Best Practice" to reduce signal-to-noise.
            if self._last_mid_prices.get(symbol) == price:
                return
            self._last_mid_prices[symbol] = price

            parsed = {
                'symbol': symbol,
                'exchange': 'coinbase',
                'timestamp': data.get('time', datetime.now(timezone.utc).isoformat()),
                'last_price': price,
                'bid_price': best_bid,
                'ask_price': best_ask,
                'volume_24h': float(data.get('volume_24h', 0)),
            }
            
            # Cache
            self.tickers[symbol] = parsed
            
            # Callback
            if self.on_ticker:
                try:
                    if asyncio.iscoroutinefunction(self.on_ticker):
                        await self.on_ticker(parsed)
                    else:
                        self.on_ticker(parsed)
                except Exception as e:
                    logger.error(f"Ticker callback error: {e}")

            # Publish QuoteEvent to core bus if available
            if self._event_bus:
                try:
                    now = time.time()
                    # Use provided time if possible, otherwise wall clock
                    try:
                        source_ts = datetime.fromisoformat(data.get('time').replace('Z', '+00:00')).timestamp()
                    except (ValueError, TypeError, AttributeError):
                        source_ts = now

                    quote = QuoteEvent(
                        symbol=symbol,
                        bid=best_bid if best_bid > 0 else price,
                        ask=best_ask if best_ask > 0 else price,
                        mid=price,
                        last=price,
                        timestamp=now,
                        source='coinbase',
                        volume_24h=float(data.get('volume_24h', 0)),
                        source_ts=source_ts,
                    )
                    self._event_bus.publish(TOPIC_MARKET_QUOTES, quote)
                except Exception as e:
                    logger.error(f"Error publishing Coinbase QuoteEvent: {e}")
                    
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing Coinbase ticker data: {e}")

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        if self._ws is None:
            return False
        return not self._ws.closed

    def get_health_status(self) -> Dict[str, Any]:
        """Return health for dashboard."""
        now = time.time()
        connected = self.is_connected()
        age = (now - self.last_message_ts) if self.last_message_ts else None
        
        status = "ok" if connected and (age is None or age < 60) else "down"
        
        from ..core.status import build_status
        return build_status(
            name="coinbase_ws",
            type="ws",
            status=status,
            last_success_ts=self.last_success_ts,
            last_error=self.last_error,
            consecutive_failures=self.consecutive_failures,
            reconnect_attempts=self.reconnect_attempts,
            request_count=self.request_count,
            error_count=self.error_count,
            last_message_ts=self.last_message_ts,
            age_seconds=round(age, 1) if age is not None else None,
            extras={
                "connected": connected,
                "symbols": len(self.symbols),
            },
        )
