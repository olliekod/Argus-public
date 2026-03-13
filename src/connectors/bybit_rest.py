"""
Bybit REST API Client
=====================

Async REST client for Bybit v5 public endpoints.
Used for:
- Instrument discovery (GET /v5/market/instruments-info)
- Historical kline backfill (GET /v5/market/kline)

No authentication required - public endpoints only.
Rate-limited to respect provider constraints.

Reference: https://bybit-exchange.github.io/docs/v5/market/instrument
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_TESTNET_URL = "https://api-testnet.bybit.com"

# Kline API returns max 200 candles per request
KLINE_LIMIT = 200

# Default rate limit: 10 requests/sec with safety margin
DEFAULT_RATE_LIMIT_RPS = 10
RATE_LIMIT_BUFFER = 0.15  # seconds between requests (safety)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BybitInstrument:
    """Parsed instrument from instruments-info endpoint."""
    symbol: str
    base_coin: str
    quote_coin: str
    status: str          # "Trading", "Settling", "PreLaunch", etc.
    category: str        # "linear", "inverse", "spot"
    launch_time_ms: int
    settle_coin: str = ""
    contract_type: str = ""  # "LinearPerpetual", etc.
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class BybitKline:
    """Single kline (candlestick) bar from Bybit REST API."""
    timestamp_ms: int    # open time in UTC ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


# ═══════════════════════════════════════════════════════════════════════════════
#  Client
# ═══════════════════════════════════════════════════════════════════════════════


class BybitRestClient:
    """Async Bybit v5 REST client for public market data.

    Parameters
    ----------
    base_url : str
        API base URL. Defaults to mainnet.
    rate_limit_rps : int
        Max requests per second.
    session : aiohttp.ClientSession or None
        Optional shared session. If None, creates one internally.
    """

    def __init__(
        self,
        base_url: str = BYBIT_BASE_URL,
        rate_limit_rps: int = DEFAULT_RATE_LIMIT_RPS,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._rate_limit_rps = rate_limit_rps
        self._min_interval = max(1.0 / rate_limit_rps, RATE_LIMIT_BUFFER)
        self._last_request_ts: float = 0.0
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def _throttle(self) -> None:
        """Enforce rate limit between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    async def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a rate-limited GET request and return parsed JSON.

        Raises ValueError on non-zero retCode.
        """
        await self._throttle()
        session = await self._get_session()
        url = f"{self._base_url}{path}"

        for attempt in range(3):
            try:
                async with session.get(url, params=params) as resp:
                    # Handle non-JSON error responses (e.g. 403 geo-block)
                    if resp.status != 200:
                        body = await resp.text()
                        if resp.status == 403:
                            raise ValueError(
                                f"Bybit returned 403 Forbidden. "
                                f"This usually means geo-restriction. "
                                f"Consider using a VPN or proxy. "
                                f"(path={path})"
                            )
                        if attempt < 2:
                            wait = 2 ** attempt
                            logger.warning(
                                "Bybit HTTP %d, retry in %ds (path=%s)",
                                resp.status, wait, path)
                            await asyncio.sleep(wait)
                            continue
                        raise ValueError(
                            f"Bybit HTTP {resp.status}: {body[:200]} "
                            f"(path={path})"
                        )

                    data = await resp.json()
                    if data.get("retCode") != 0:
                        msg = data.get("retMsg", "unknown error")
                        code = data.get("retCode")
                        if code == 10016:  # rate limited
                            if attempt >= 4:
                                raise ValueError(
                                    f"Bybit rate limited after {attempt + 1} attempts "
                                    f"(path={path})"
                                )
                            wait = 2 ** attempt
                            logger.warning("Bybit rate limited, waiting %ds (attempt %d)", wait, attempt + 1)
                            await asyncio.sleep(wait)
                            continue
                        raise ValueError(
                            f"Bybit API error {code}: {msg} "
                            f"(path={path}, params={params})"
                        )
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning("Bybit request failed (%s), retry in %ds",
                                   exc, wait)
                    await asyncio.sleep(wait)
                else:
                    raise

        raise RuntimeError("Bybit request failed after 3 attempts")

    # ─── Instrument Discovery ─────────────────────────────────────────────

    async def get_instruments(
        self,
        category: str = "linear",
        status: Optional[str] = "Trading",
    ) -> List[BybitInstrument]:
        """Fetch all instruments for a category.

        Handles pagination via cursor automatically.

        Parameters
        ----------
        category : str
            "linear", "inverse", or "spot".
        status : str or None
            Filter by status. None = return all.
        """
        instruments: List[BybitInstrument] = []
        cursor = ""

        while True:
            params: Dict[str, Any] = {
                "category": category,
                "limit": 1000,
            }
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor

            data = await self._get("/v5/market/instruments-info", params)
            result = data.get("result", {})

            for item in result.get("list", []):
                instruments.append(BybitInstrument(
                    symbol=item.get("symbol", ""),
                    base_coin=item.get("baseCoin", ""),
                    quote_coin=item.get("quoteCoin", ""),
                    status=item.get("status", ""),
                    category=category,
                    launch_time_ms=int(item.get("launchTime", "0")),
                    settle_coin=item.get("settleCoin", ""),
                    contract_type=item.get("contractType", ""),
                    raw=item,
                ))

            cursor = result.get("nextPageCursor", "")
            if not cursor:
                break

        logger.info("Discovered %d %s instruments (status=%s)",
                     len(instruments), category, status)
        return instruments

    async def discover_perpetuals(
        self,
        quote_coin: str = "USDT",
        base_coins: Optional[List[str]] = None,
    ) -> List[BybitInstrument]:
        """Discover USDT-settled linear perpetuals.

        Parameters
        ----------
        quote_coin : str
            Filter by quote coin (default "USDT").
        base_coins : list of str or None
            If given, only return instruments with these base coins.
            If None, return all trading instruments.
        """
        all_instruments = await self.get_instruments(
            category="linear", status="Trading")

        filtered = [
            i for i in all_instruments
            if i.quote_coin == quote_coin
            and i.contract_type == "LinearPerpetual"
        ]

        if base_coins:
            base_set = {b.upper() for b in base_coins}
            filtered = [i for i in filtered if i.base_coin in base_set]

        logger.info("Filtered to %d perpetuals (quote=%s, bases=%s)",
                     len(filtered), quote_coin, base_coins)
        return filtered

    # ─── Kline (Candlestick) Data ─────────────────────────────────────────

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1",
        start_ms: int = 0,
        end_ms: int = 0,
        limit: int = KLINE_LIMIT,
        category: str = "linear",
    ) -> List[BybitKline]:
        """Fetch klines for a symbol.

        Parameters
        ----------
        symbol : str
            e.g. "BTCUSDT"
        interval : str
            "1" for 1 minute, "5", "15", "60", "240", "D", "W"
        start_ms : int
            Start time in UTC milliseconds. 0 = omit.
        end_ms : int
            End time in UTC milliseconds. 0 = omit.
        limit : int
            Max candles to return (1-200). Default 200.
        category : str
            "linear", "inverse", or "spot".

        Returns
        -------
        List[BybitKline]
            Sorted by timestamp ascending (oldest first).
        """
        params: Dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, KLINE_LIMIT),
        }
        if start_ms > 0:
            params["start"] = start_ms
        if end_ms > 0:
            params["end"] = end_ms

        data = await self._get("/v5/market/kline", params)
        raw_list = data.get("result", {}).get("list", [])

        klines: List[BybitKline] = []
        for item in raw_list:
            # Bybit returns: [startTime, open, high, low, close, volume, turnover]
            # All as strings
            try:
                klines.append(BybitKline(
                    timestamp_ms=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    turnover=float(item[6]),
                ))
            except (IndexError, ValueError) as exc:
                logger.warning("Skipping malformed kline: %s (%s)", item, exc)

        # Bybit returns newest first; reverse to oldest first
        klines.sort(key=lambda k: k.timestamp_ms)
        return klines

    async def backfill_klines(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        interval: str = "1",
        category: str = "linear",
        progress_callback=None,
    ) -> List[BybitKline]:
        """Fetch klines for a date range, chunking automatically.

        Handles the 200-candle limit by walking forward in time.
        Never invents data: only returns what the exchange provides.

        Parameters
        ----------
        symbol : str
            e.g. "BTCUSDT"
        start_ms, end_ms : int
            UTC ms range.
        interval : str
            Kline interval. "1" = 1 minute.
        category : str
            Market category.
        progress_callback : callable or None
            Called with (fetched_count, chunk_count) after each chunk.

        Returns
        -------
        List[BybitKline]
            All klines in range, sorted ascending by timestamp. No duplicates.
        """
        interval_ms = _interval_to_ms(interval)
        all_klines: List[BybitKline] = []
        seen_ts: set = set()

        cursor_ms = start_ms
        chunk_count = 0

        while cursor_ms < end_ms:
            chunk = await self.get_klines(
                symbol=symbol,
                interval=interval,
                start_ms=cursor_ms,
                end_ms=end_ms,
                limit=KLINE_LIMIT,
                category=category,
            )
            chunk_count += 1

            if not chunk:
                break

            for k in chunk:
                if k.timestamp_ms not in seen_ts and k.timestamp_ms < end_ms:
                    all_klines.append(k)
                    seen_ts.add(k.timestamp_ms)

            # Advance cursor past the last received candle
            last_ts = chunk[-1].timestamp_ms
            if last_ts + interval_ms <= cursor_ms:
                # No forward progress; break to avoid infinite loop
                break
            cursor_ms = last_ts + interval_ms

            if progress_callback:
                progress_callback(len(all_klines), chunk_count)

        all_klines.sort(key=lambda k: k.timestamp_ms)
        logger.info("Backfilled %d klines for %s (%d chunks)",
                     len(all_klines), symbol, chunk_count)
        return all_klines


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _interval_to_ms(interval: str) -> int:
    """Convert Bybit interval string to milliseconds."""
    mapping = {
        "1": 60_000,
        "3": 180_000,
        "5": 300_000,
        "15": 900_000,
        "30": 1_800_000,
        "60": 3_600_000,
        "120": 7_200_000,
        "240": 14_400_000,
        "360": 21_600_000,
        "720": 43_200_000,
        "D": 86_400_000,
        "W": 604_800_000,
    }
    return mapping.get(interval, 60_000)


def klines_to_bar_rows(
    klines: List[BybitKline],
    source: str = "bybit",
    symbol: str = "",
    bar_duration: int = 60,
) -> List[tuple]:
    """Convert BybitKline list to market_bars INSERT tuples.

    The timestamp is formatted as ISO UTC string to match the existing
    ``market_bars`` schema where ``timestamp`` is TEXT.

    Returns list of tuples matching the market_bars INSERT columns:
    (timestamp, symbol, source, open, high, low, close, volume,
     tick_count, n_ticks, first_source_ts, last_source_ts,
     late_ticks_dropped, close_reason, bar_duration)
    """
    rows = []
    for k in klines:
        ts_str = datetime.fromtimestamp(
            k.timestamp_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

        source_ts = k.timestamp_ms / 1000.0  # seconds as float

        rows.append((
            ts_str,           # timestamp (TEXT)
            symbol,           # symbol
            source,           # source
            k.open,           # open
            k.high,           # high
            k.low,            # low
            k.close,          # close
            k.volume,         # volume
            0,                # tick_count (backfill has no tick info)
            0,                # n_ticks
            source_ts,        # first_source_ts
            source_ts,        # last_source_ts
            0,                # late_ticks_dropped
            4,                # close_reason = 4 (REST_BACKFILL)
            bar_duration,     # bar_duration
        ))
    return rows
