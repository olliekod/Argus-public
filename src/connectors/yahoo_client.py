"""
Yahoo Finance Client
====================

Fetches IBIT ETF price data for options monitoring.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple
import aiohttp
import math
from ..core.logger import get_connector_logger
from ..core.events import QuoteEvent, TOPIC_MARKET_QUOTES
from ..core.bar_builder import _ts_sane

logger = get_connector_logger('yahoo')


def _parse_yahoo_source_ts(meta: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    """Extract and normalize Yahoo's source timestamp to epoch seconds."""
    raw = meta.get("regularMarketTime")
    if raw is None:
        return None, "missing", None
    try:
        raw_val = float(raw)
    except (TypeError, ValueError):
        return None, "invalid", raw

    if _ts_sane(raw_val):
        return raw_val, None, raw_val

    if raw_val > 10_000_000_000:
        candidate = raw_val / 1000.0
        if _ts_sane(candidate):
            return candidate, "converted_ms", raw_val

    return None, "out_of_range", raw_val


class YahooFinanceClient:
    """
    Yahoo Finance client for ETF price data.
    
    Used to monitor equities/ETFs (IBIT, BITO, SPY, QQQ, NVDA).
    """
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    def __init__(
        self,
        symbols: list = None,
        on_update: Optional[Callable] = None,
        event_bus=None,
    ):
        """
        Initialize Yahoo Finance client.

        Args:
            symbols: List of stock/ETF symbols
            on_update: Callback for price updates
            event_bus: Optional EventBus instance for publishing QuoteEvents
        """
        self.symbols = symbols or ['IBIT', 'BITO', 'SPY', 'QQQ', 'NVDA']
        self.on_update = on_update
        self._event_bus = event_bus

        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self.last_message_ts: Optional[float] = None
        self.reconnect_attempts = 0
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures: int = 0
        self.request_count: int = 0
        self.error_count: int = 0
        self.last_latency_ms: Optional[float] = None
        self.avg_latency_ms: Optional[float] = None
        self.last_poll_ts: Optional[float] = None
        self.last_http_status: Optional[int] = None

        # Latest data cache
        self.prices: Dict[str, Dict] = {}

        logger.info(f"Yahoo Finance client initialized for {self.symbols}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Stock/ETF symbol (e.g., 'IBIT')
            
        Returns:
            Quote data or None
        """
        session = await self._get_session()
        url = f"{self.BASE_URL}/{symbol}"
        
        params = {
            'interval': '1d',
            'range': '5d',
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        start = time.perf_counter()
        self.last_poll_ts = time.time()
        self.request_count += 1
        try:
            async with session.get(url, params=params, headers=headers) as resp:
                self.last_http_status = resp.status
                if resp.status == 200:
                    self.last_message_ts = time.time()
                    self.last_success_ts = self.last_message_ts
                    self.consecutive_failures = 0
                    self.last_error = None
                    self.last_latency_ms = (time.perf_counter() - start) * 1000
                    self.avg_latency_ms = (
                        self.last_latency_ms if self.avg_latency_ms is None
                        else (self.last_latency_ms * 0.2) + (self.avg_latency_ms * 0.8)
                    )
                    data = await resp.json()
                    
                    result = data.get('chart', {}).get('result', [])
                    if not result:
                        return None
                    
                    meta = result[0].get('meta', {})
                    indicators = result[0].get('indicators', {})
                    quote = indicators.get('quote', [{}])[0]
                    
                    # Get latest values
                    closes = quote.get('close', [])
                    volumes = quote.get('volume', [])
                    highs = quote.get('high', [])
                    lows = quote.get('low', [])
                    
                    current_price = meta.get('regularMarketPrice') or 0
                    prev_close = (
                        meta.get('regularMarketPreviousClose')
                        or meta.get('previousClose')
                        or meta.get('chartPreviousClose')
                        or 0
                    )
                    
                    # Fallback to latest close if live price missing
                    if not current_price:
                        valid_closes = [c for c in closes if c is not None]
                        if valid_closes:
                            current_price = valid_closes[-1]

                    # Calculate price change
                    price_change = 0
                    price_change_pct = 0
                    if prev_close and prev_close > 0 and current_price:
                        price_change = current_price - prev_close
                        price_change_pct = (price_change / prev_close) * 100
                    
                    # Calculate recent volatility (5-day)
                    if len(closes) >= 2:
                        valid_closes = [c for c in closes if c is not None]
                        if len(valid_closes) >= 2:
                            returns = []
                            for i in range(1, len(valid_closes)):
                                ret = (valid_closes[i] - valid_closes[i-1]) / valid_closes[i-1]
                                returns.append(ret)
                            
                            if returns:
                                mean_ret = sum(returns) / len(returns)
                                variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
                                daily_vol = math.sqrt(variance)
                                annualized_vol = daily_vol * math.sqrt(252) * 100
                            else:
                                annualized_vol = 0
                        else:
                            annualized_vol = 0
                    else:
                        annualized_vol = 0
                    
                    source_ts, ts_reason, ts_raw = _parse_yahoo_source_ts(meta)
                    parsed = {
                        'symbol': symbol,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'price': current_price,
                        'previous_close': prev_close,
                        'price_change': price_change,
                        'price_change_pct': price_change_pct,
                        'day_high': meta.get('regularMarketDayHigh', 0),
                        'day_low': meta.get('regularMarketDayLow', 0),
                        'volume': meta.get('regularMarketVolume', 0),
                        'realized_vol_5d': annualized_vol,
                        'market_state': meta.get('marketState', 'CLOSED'),
                        'source_ts': source_ts,
                        'source_ts_raw': ts_raw,
                        'source_ts_reason': ts_reason,
                    }
                    
                    return parsed
                else:
                    self.error_count += 1
                    self.consecutive_failures += 1
                    self.last_error = f"http_{resp.status}"
                    logger.warning(f"Yahoo API error for {symbol}: {resp.status}")
                    return None
                    
        except Exception as e:
            self.error_count += 1
            self.consecutive_failures += 1
            self.last_error = str(e)
            logger.error(f"Yahoo request failed: {e}")
            return None
    
    async def poll_once(self) -> None:
        """Run a single poll cycle (all symbols once)."""
        for symbol in self.symbols:
            try:
                data = await self.get_quote(symbol)
                if data:
                    self.prices[symbol] = data
                    if self.on_update:
                        try:
                            if asyncio.iscoroutinefunction(self.on_update):
                                await self.on_update(data)
                            else:
                                self.on_update(data)
                        except Exception as e:
                            logger.error(f"Update callback error: {e}")

                    # Publish QuoteEvent to event bus
                    if self._event_bus is not None:
                        try:
                            price = data.get('price', 0)
                            source_ts = data.get('source_ts')
                            source_ts_raw = data.get('source_ts_raw')
                            source_ts_reason = data.get('source_ts_reason')
                            if source_ts is None:
                                logger.warning(
                                    "Rejected Yahoo quote for %s: %s (raw_ts=%r)",
                                    symbol,
                                    source_ts_reason or "missing source_ts",
                                    source_ts_raw,
                                )
                                source_ts = 0.0
                            timestamp = source_ts if source_ts else 0.0
                            quote = QuoteEvent(
                                symbol=symbol,
                                bid=price,
                                ask=price,
                                mid=price,
                                last=price,
                                timestamp=timestamp,
                                source='yahoo',
                                volume_24h=float(data.get('volume', 0) or 0),
                                source_ts=source_ts,
                            )
                            self._event_bus.publish(TOPIC_MARKET_QUOTES, quote)
                        except Exception as e:
                            logger.error("QuoteEvent publish error (%s): %s", type(e).__name__, e)
                            logger.debug("Yahoo QuoteEvent publish error detail", exc_info=True)
            except Exception as e:
                logger.error("Yahoo poll_once error for %s (%s): %s", symbol, type(e).__name__, e)
                logger.debug("Yahoo poll_once error detail for %s", symbol, exc_info=True)
            await asyncio.sleep(1)

    async def poll(self, interval_seconds: int = 60) -> None:
        """
        Continuously poll for price updates.

        Args:
            interval_seconds: Polling interval (default 60s for stocks)
        """
        self._running = True
        logger.info(f"Starting Yahoo Finance polling ({interval_seconds}s interval)")

        while self._running:
            try:
                await self.poll_once()
            except Exception as e:
                logger.error("Yahoo polling error (%s): %s", type(e).__name__, e)
                logger.debug("Yahoo polling error detail", exc_info=True)

            await asyncio.sleep(interval_seconds)
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        data = self.prices.get(symbol)
        return data['price'] if data else None
    
    def get_ibit_data(self) -> Optional[Dict]:
        """Get IBIT data specifically."""
        return self.prices.get('IBIT')

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
            name="yahoo",
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
