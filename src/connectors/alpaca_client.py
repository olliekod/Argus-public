"""
Alpaca Data Client
==================

REST-based polling client for IBIT/BITO 1-minute bars.
Runs in ALL modes (collector, paper, live) — only data ingestion, no execution.

DETERMINISM REQUIREMENTS
------------------------
- Fixed interval polling (no market-hours-aware gating)
- Overlap window for bar requests (covers restart scenarios)
- Bars emitted in strict bar_ts order
- Startup init from DB to prevent duplicate bars on restart

TIMESTAMP CONVENTION
--------------------
- Internal: int milliseconds (UTC epoch ms)
- BarEvent.timestamp: float seconds (for compatibility with BarBuilder)
- bar_ts used for deduplication is int milliseconds
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..core.events import (
    BarEvent,
    CloseReason,
    TOPIC_MARKET_BARS,
)
from ..core.logger import get_connector_logger

logger = get_connector_logger("alpaca")

# Alpaca Data API v2 base URL
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"

# Provider priority for tape ordering
PROVIDER_PRIORITY = 1  # Alpaca is highest priority

# Default overlap window in seconds (2 bars worth)
DEFAULT_OVERLAP_SECONDS = 120


def _parse_rfc3339_to_ms(ts_str: str) -> int:
    """
    Parse RFC3339 timestamp to UTC epoch milliseconds (int).

    Alpaca returns timestamps like: 2024-01-15T14:30:00Z

    If the parsed datetime is naive (no timezone info), it is assumed
    to be UTC.  This prevents .timestamp() from silently using the
    local timezone.
    """
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts_str)
    # Assume UTC for naive datetimes (missing timezone offset)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class AlpacaDataClient:
    """
    Alpaca market data client for equity bars.
    
    Fetches 1-minute bars via Alpaca Data API v2.
    Designed for IBIT/BITO ETF data ingestion.
    
    DETERMINISM GUARANTEES
    ----------------------
    - Poll at fixed interval regardless of market hours
    - Each poll requests bars with overlap window for robustness
    - Bars are emitted in strict increasing bar_ts order
    - Dedupe by bar_ts (int ms) prevents duplicate bars
    - On startup, initializes last_bar_ts from DB persistence
    
    Parameters
    ----------
    api_key : str
        Alpaca API key.
    api_secret : str
        Alpaca API secret.
    symbols : list[str]
        Symbols to fetch (e.g., ["IBIT", "BITO"]).
    event_bus : EventBus
        Event bus for publishing BarEvents.
    db : Database, optional
        Database for startup init (get_latest_bar_ts).
    poll_interval : int
        Seconds between polls (default 60). Fixed interval, no market-hours gating.
    overlap_seconds : int
        Overlap window for bar requests (default 120 = 2 bars).
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        event_bus,
        db=None,
        poll_interval: int = 60,
        overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._symbols = symbols
        self._event_bus = event_bus
        self._db = db
        self._poll_interval = poll_interval
        self._overlap_seconds = overlap_seconds
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._initialized = False
        
        # Deduplication: track last seen bar timestamp per symbol (int ms)
        self._last_bar_ts: Dict[str, int] = {}
        
        # Health metrics
        self._last_success_ts: Optional[float] = None
        self._last_error: Optional[str] = None
        self._consecutive_failures: int = 0
        self._request_count: int = 0
        self._error_count: int = 0
        self._bars_emitted: int = 0
        self._last_poll_ts: Optional[float] = None
        self._last_latency_ms: Optional[float] = None
        self._avg_latency_ms: Optional[float] = None
        
        logger.info(
            "AlpacaDataClient initialized for symbols=%s, poll_interval=%ds, overlap=%ds",
            symbols, poll_interval, overlap_seconds
        )
    
    async def init_from_db(self) -> None:
        """
        Initialize last_bar_ts from database persistence.
        
        Called on startup to prevent duplicate bars after restart.
        Must be called before polling starts.
        """
        if self._db is None:
            logger.warning("AlpacaDataClient: no database provided, skipping init_from_db")
            self._initialized = True
            return
        
        for symbol in self._symbols:
            try:
                latest_ts_ms = await self._db.get_latest_bar_ts(
                    source="alpaca",
                    symbol=symbol,
                    bar_duration=60,
                )
                if latest_ts_ms is not None:
                    self._last_bar_ts[symbol] = latest_ts_ms
                    logger.info(
                        "AlpacaDataClient: initialized %s last_bar_ts=%d from DB",
                        symbol, latest_ts_ms
                    )
                else:
                    logger.info(
                        "AlpacaDataClient: no existing bars for %s in DB",
                        symbol
                    )
            except Exception as e:
                logger.error(
                    "AlpacaDataClient: failed to init last_bar_ts for %s: %s",
                    symbol, e
                )
        
        self._initialized = True
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with auth headers."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "APCA-API-KEY-ID": self._api_key,
                    "APCA-API-SECRET-KEY": self._api_secret,
                }
            )
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session and stop polling."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("AlpacaDataClient closed")
    
    async def fetch_bars(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch recent bars for a symbol.
        
        Parameters
        ----------
        symbol : str
            Stock symbol (e.g., "IBIT").
        limit : int
            Number of bars to fetch (default 5, enough for dedup + overlap).
        
        Returns
        -------
        list of bar dicts with keys: t, o, h, l, c, v
        """
        session = await self._get_session()
        url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
        
        params = {
            "timeframe": "1Min",
            "limit": limit,
            "sort": "desc",  # Most recent first
        }
        
        start = time.perf_counter()
        self._last_poll_ts = time.time()
        self._request_count += 1
        
        try:
            async with session.get(url, params=params) as resp:
                latency_ms = (time.perf_counter() - start) * 1000
                self._last_latency_ms = latency_ms
                self._avg_latency_ms = (
                    latency_ms if self._avg_latency_ms is None
                    else latency_ms * 0.2 + self._avg_latency_ms * 0.8
                )
                
                if resp.status == 200:
                    data = await resp.json()
                    self._last_success_ts = time.time()
                    self._consecutive_failures = 0
                    self._last_error = None
                    return data.get("bars", []) or []
                else:
                    self._error_count += 1
                    self._consecutive_failures += 1
                    error_text = await resp.text()
                    self._last_error = f"HTTP {resp.status}: {error_text[:100]}"
                    logger.warning(
                        "Alpaca API error for %s: %s",
                        symbol, self._last_error
                    )
                    return []
                    
        except Exception as e:
            self._error_count += 1
            self._consecutive_failures += 1
            self._last_error = str(e)
            logger.error("Alpaca request failed for %s: %s", symbol, e)
            return []
    
    def _bar_to_event(self, symbol: str, bar: Dict[str, Any]) -> BarEvent:
        """
        Convert Alpaca bar dict to BarEvent.
        
        BarEvent.timestamp = bar OPEN time (UTC epoch seconds).
        Internal tracking uses int milliseconds.
        """
        bar_ts_ms = _parse_rfc3339_to_ms(bar["t"])
        bar_ts_sec = bar_ts_ms / 1000.0  # Convert to seconds for BarEvent
        
        return BarEvent(
            symbol=symbol,
            open=float(bar["o"]),
            high=float(bar["h"]),
            low=float(bar["l"]),
            close=float(bar["c"]),
            volume=float(bar["v"]),
            timestamp=bar_ts_sec,  # bar OPEN time in seconds
            source="alpaca",
            bar_duration=60,
            n_ticks=1,  # Pre-aggregated bar
            first_source_ts=bar_ts_sec,
            last_source_ts=bar_ts_sec,
            close_reason=CloseReason.MINUTE_BOUNDARY,
            source_ts=bar_ts_sec,
        )
    
    async def poll_once(self) -> int:
        """
        Run a single poll cycle for all symbols.
        
        Returns number of new bars emitted.
        
        ORDERING GUARANTEE
        ------------------
        Bars are collected, sorted by bar_ts, then emitted in strict order.
        """
        if not self._initialized:
            await self.init_from_db()
        
        total_emitted = 0
        
        for symbol in self._symbols:
            try:
                # Fetch extra bars to cover overlap window
                # overlap_seconds / 60 + 2 gives us buffer
                limit = max(5, (self._overlap_seconds // 60) + 2)
                bars = await self.fetch_bars(symbol, limit=limit)
                
                # Filter and collect new bars
                new_bars: List[tuple] = []  # (bar_ts_ms, bar_dict)
                last_seen_ms = self._last_bar_ts.get(symbol, 0)
                
                for bar in bars:
                    bar_ts_ms = _parse_rfc3339_to_ms(bar["t"])
                    
                    # Dedupe: only include bars newer than last seen
                    if bar_ts_ms > last_seen_ms:
                        new_bars.append((bar_ts_ms, bar))
                
                # Sort by bar_ts (ascending) for strict ordering
                new_bars.sort(key=lambda x: x[0])
                
                # Emit in order
                for bar_ts_ms, bar in new_bars:
                    self._last_bar_ts[symbol] = bar_ts_ms
                    
                    event = self._bar_to_event(symbol, bar)
                    self._event_bus.publish(TOPIC_MARKET_BARS, event)
                    self._bars_emitted += 1
                    total_emitted += 1
                    
                    logger.debug(
                        "Emitted bar: %s @ %s (o=%.2f h=%.2f l=%.2f c=%.2f v=%.0f)",
                        symbol,
                        datetime.fromtimestamp(bar_ts_ms / 1000, tz=timezone.utc).isoformat(),
                        event.open, event.high, event.low, event.close, event.volume,
                    )
                
            except Exception as e:
                logger.error("Error polling %s (%s): %s", symbol, type(e).__name__, e)
                logger.debug("Alpaca poll_once error detail for %s", symbol, exc_info=True)
            
            # Small delay between symbols to avoid rate limits
            await asyncio.sleep(0.5)
        
        return total_emitted
    
    async def poll(self) -> None:
        """
        Continuously poll for bar updates.
        
        FIXED INTERVAL: No market-hours-aware gating. Polls at constant interval.
        
        Runs until close() is called.
        """
        self._running = True
        
        # Initialize from DB before starting
        if not self._initialized:
            await self.init_from_db()
        
        logger.info(
            "Starting Alpaca polling for %s (fixed interval=%ds)",
            self._symbols, self._poll_interval
        )
        
        while self._running:
            try:
                emitted = await self.poll_once()
                if emitted > 0:
                    logger.info("Alpaca poll: emitted %d new bars", emitted)
            except Exception as e:
                logger.error("Alpaca polling error (%s): %s", type(e).__name__, e)
                logger.debug("Alpaca polling error detail", exc_info=True)
            
            await asyncio.sleep(self._poll_interval)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status for dashboard."""
        now = time.time()
        age = (now - self._last_success_ts) if self._last_success_ts else None
        
        if self._consecutive_failures > 0:
            status = "degraded"
        elif self._last_success_ts:
            status = "ok"
        else:
            status = "unknown"
        
        from ..core.status import build_status
        
        return build_status(
            name="alpaca",
            type="rest",
            status=status,
            last_success_ts=self._last_success_ts,
            last_error=self._last_error,
            consecutive_failures=self._consecutive_failures,
            reconnect_attempts=0,
            request_count=self._request_count,
            error_count=self._error_count,
            avg_latency_ms=round(self._avg_latency_ms, 2) if self._avg_latency_ms else None,
            last_latency_ms=round(self._last_latency_ms, 2) if self._last_latency_ms else None,
            last_poll_ts=self._last_poll_ts,
            age_seconds=round(age, 1) if age else None,
            extras={
                "symbols": self._symbols,
                "bars_emitted": self._bars_emitted,
                "poll_interval_s": self._poll_interval,
                "overlap_seconds": self._overlap_seconds,
                "last_bar_ts_ms": dict(self._last_bar_ts),
                "initialized": self._initialized,
            },
        )
