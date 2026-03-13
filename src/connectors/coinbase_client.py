"""
Coinbase REST API Client
========================

REST client for Coinbase spot prices (US-friendly).
Replaces Binance WebSocket which blocks US IPs.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import aiohttp

from ..core.logger import get_connector_logger

logger = get_connector_logger('coinbase')


class CoinbaseClient:
    """
    Coinbase public API client for spot prices.
    
    Uses public endpoints - no API key required for price data.
    US-friendly - no IP blocking.
    """
    
    BASE_URL = "https://api.coinbase.com/v2"
    EXCHANGE_URL = "https://api.exchange.coinbase.com"
    
    def __init__(
        self,
        symbols: List[str],
        on_ticker: Optional[Callable] = None,
    ):
        """
        Initialize Coinbase client.
        
        Args:
            symbols: List of symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            on_ticker: Callback for ticker updates
        """
        self.symbols = [self._normalize_symbol(s) for s in symbols]
        self.on_ticker = on_ticker
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self.last_message_ts: Optional[float] = None
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures: int = 0
        self.reconnect_attempts: int = 0
        self.request_count: int = 0
        self.error_count: int = 0
        self.last_latency_ms: Optional[float] = None
        self.avg_latency_ms: Optional[float] = None
        self.last_poll_ts: Optional[float] = None
        self.last_http_status: Optional[int] = None
        
        # Latest data cache
        self.tickers: Dict[str, Dict] = {}
        
        logger.info(f"Coinbase client initialized for {len(self.symbols)} symbols")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Convert unified symbol to Coinbase format."""
        # 'BTC/USDT:USDT' -> 'BTC-USD'
        if ':' in symbol:
            symbol = symbol.split(':')[0]
        
        base = symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '')
        
        # Coinbase uses USD not USDT
        return f"{base}-USD"
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=10))
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get spot price for a symbol.
        
        Args:
            symbol: Coinbase symbol (e.g., 'BTC-USD')
            
        Returns:
            Ticker data or None
        """
        session = await self._get_session()
        url = f"{self.EXCHANGE_URL}/products/{symbol}/ticker"
        
        start = asyncio.get_running_loop().time()
        self.last_poll_ts = datetime.now(timezone.utc).timestamp()
        self.request_count += 1
        try:
            async with session.get(url) as resp:
                self.last_http_status = resp.status
                if resp.status == 200:
                    data = await resp.json()
                    
                    parsed = {
                        'symbol': symbol.replace('-', ''),
                        'exchange': 'coinbase',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'last_price': float(data.get('price', 0)),
                        'bid_price': float(data.get('bid', 0)),
                        'ask_price': float(data.get('ask', 0)),
                        'volume_24h': float(data.get('volume', 0)),
                    }
                    self.last_message_ts = time.time()
                    self.last_success_ts = self.last_message_ts
                    self.consecutive_failures = 0
                    self.last_error = None
                    latency_ms = (asyncio.get_running_loop().time() - start) * 1000
                    self.last_latency_ms = latency_ms
                    self.avg_latency_ms = (
                        latency_ms if self.avg_latency_ms is None
                        else (latency_ms * 0.2) + (self.avg_latency_ms * 0.8)
                    )
                    
                    return parsed
                else:
                    self.error_count += 1
                    self.consecutive_failures += 1
                    self.last_error = f"http_{resp.status}"
                    logger.warning(f"Coinbase API error for {symbol}: {resp.status}")
                    return None
        except Exception as e:
            self.error_count += 1
            self.consecutive_failures += 1
            self.last_error = str(e)
            logger.error(f"Coinbase request failed: {e}")
            return None
    
    async def get_all_tickers(self) -> Dict[str, Dict]:
        """Get tickers for all configured symbols."""
        results = {}
        
        for symbol in self.symbols:
            ticker = await self.get_ticker(symbol)
            if ticker:
                results[symbol] = ticker
                self.tickers[symbol] = ticker
        
        return results
    
    async def poll(self, interval_seconds: int = 5) -> None:
        """
        Continuously poll for price updates.
        
        Args:
            interval_seconds: Polling interval
        """
        self._running = True
        logger.info(f"Starting Coinbase price polling ({interval_seconds}s interval)")
        
        while self._running:
            try:
                for symbol in self.symbols:
                    ticker = await self.get_ticker(symbol)
                    if ticker:
                        self.tickers[symbol] = ticker
                        
                        if self.on_ticker:
                            try:
                                if asyncio.iscoroutinefunction(self.on_ticker):
                                    await self.on_ticker(ticker)
                                else:
                                    self.on_ticker(ticker)
                            except Exception as e:
                                logger.error(f"Ticker callback error: {e}")
                    
                    # Small delay between symbols to avoid rate limiting
                    await asyncio.sleep(0.2)
                    
            except Exception as e:
                logger.error(f"Coinbase polling error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest spot price for a symbol."""
        normalized = self._normalize_symbol(symbol)
        ticker = self.tickers.get(normalized)
        return ticker['last_price'] if ticker else None
    
    @property
    def is_connected(self) -> bool:
        """Check if client is running."""
        return self._running

    def get_health_status(self) -> Dict[str, Any]:
        """Return health for dashboard."""
        now = time.time()
        age = (now - self.last_message_ts) if self.last_message_ts else None
        if self.consecutive_failures > 0:
            status = "degraded"
        elif self.last_success_ts:
            status = "ok"
        else:
            status = "unknown"

        from ..core.status import build_status

        return build_status(
            name="coinbase",
            type="rest",
            status=status,
            last_success_ts=self.last_success_ts,
            last_error=self.last_error,
            consecutive_failures=self.consecutive_failures,
            reconnect_attempts=self.reconnect_attempts,
            request_count=self.request_count,
            error_count=self.error_count,
            avg_latency_ms=round(self.avg_latency_ms, 2) if self.avg_latency_ms is not None else None,
            last_latency_ms=round(self.last_latency_ms, 2) if self.last_latency_ms is not None else None,
            last_poll_ts=self.last_poll_ts,
            age_seconds=round(age, 1) if age is not None else None,
            extras={
                "connected": self._session is not None and not self._session.closed,
                "symbols": len(self.symbols),
                "last_http_status": self.last_http_status,
            },
        )
