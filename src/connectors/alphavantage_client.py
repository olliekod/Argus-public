"""
Alpha Vantage REST Client
=========================

Fetches daily bars (ETF / equity) and daily FX rates for the
GlobalRiskFlow feature.  Free tier: 5 requests/min, 25/day.

Usage::

    client = AlphaVantageClient(api_key="YOUR_KEY")
    bars = await client.fetch_daily_bars("EWJ")
    fx   = await client.fetch_fx_daily("EUR", "USD")
    await client.close()
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("argus.alphavantage")

_BASE_URL = "https://www.alphavantage.co/query"

# Default rate limit: 5 calls per minute (free tier)
_DEFAULT_CALL_INTERVAL = 12.5  # seconds → ~4.8 calls/min (safe margin)

# Retry settings for rate-limit "Note" responses
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_SECONDS = 60.0  # AV rate windows are per minute


class AlphaVantageRateLimitError(Exception):
    """Raised when Alpha Vantage returns a rate-limit "Note" response."""
    pass


class AlphaVantageClient:
    """Async REST client for Alpha Vantage daily bars and FX.

    Args:
        api_key: Alpha Vantage API key.
        base_url: Override API endpoint (for testing).
        call_interval_seconds: Minimum seconds between API calls.
            Defaults to 12.5 (free tier safe margin).
        max_retries: Max retries on rate-limit "Note" responses.
        retry_base_seconds: Base backoff delay on rate-limit retry.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _BASE_URL,
        call_interval_seconds: float = _DEFAULT_CALL_INTERVAL,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_base_seconds: float = _DEFAULT_RETRY_BASE_SECONDS,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._call_interval = call_interval_seconds
        self._max_retries = max_retries
        self._retry_base = retry_base_seconds
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_call_ts: float = 0.0
        self._calls_made: int = 0

    @property
    def calls_made(self) -> int:
        """Total API calls made (including retries)."""
        return self._calls_made

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Rate limiter ────────────────────────────────────────────────────

    async def _throttle(self) -> None:
        """Ensure minimum interval between API calls."""
        now = time.monotonic()
        elapsed = now - self._last_call_ts
        if elapsed < self._call_interval:
            await asyncio.sleep(self._call_interval - elapsed)
        self._last_call_ts = time.monotonic()

    # ── HTTP helper ─────────────────────────────────────────────────────

    async def _get_json(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make a throttled GET request and return parsed JSON.

        Raises:
            RuntimeError: On non-200 HTTP status.
            ValueError: On explicit API error messages.
            AlphaVantageRateLimitError: On rate-limit "Note" responses
                (after exhausting retries).
        """
        last_note = ""
        for attempt in range(self._max_retries + 1):
            await self._throttle()
            session = await self._get_session()
            request_params = {**params, "apikey": self._api_key}

            async with session.get(
                self._base_url, params=request_params,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Alpha Vantage HTTP {resp.status}: {text[:200]}"
                    )
                data = await resp.json(content_type=None)

            self._calls_made += 1

            # Hard error from AV (bad symbol, bad function, etc.)
            if "Error Message" in data:
                raise ValueError(
                    f"Alpha Vantage error: {data['Error Message']}"
                )

            # Rate-limit "Note" or "Information" — AV returns this instead of data 
            # when you've hit the per-minute or per-day cap.
            rate_limit_msg = data.get("Note") or data.get("Information")
            if rate_limit_msg:
                last_note = rate_limit_msg
                
                # If it's a daily limit message, raise immediately (retrying won't help)
                if "25 requests per day" in str(rate_limit_msg).lower():
                    raise AlphaVantageRateLimitError(
                        f"Alpha Vantage daily limit reached: {rate_limit_msg}"
                    )

                if attempt < self._max_retries:
                    backoff = self._retry_base * (2 ** attempt)
                    logger.warning(
                        "Alpha Vantage rate limit (attempt %d/%d): %s "
                        "— backing off %.0fs",
                        attempt + 1,
                        self._max_retries + 1,
                        last_note,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                # Exhausted retries
                raise AlphaVantageRateLimitError(
                    f"Alpha Vantage rate limit after {self._max_retries + 1} "
                    f"attempts: {last_note}"
                )

            # Valid response
            return data

        # Should not reach here, but satisfy type checker
        raise AlphaVantageRateLimitError(last_note)  # pragma: no cover

    # ── Daily Equity / ETF Bars ─────────────────────────────────────────

    async def fetch_daily_bars(
        self,
        symbol: str,
        *,
        outputsize: str = "compact",
    ) -> List[Dict[str, Any]]:
        """Fetch daily OHLCV bars for a stock / ETF.

        Args:
            symbol: Ticker (e.g. ``"EWJ"``).
            outputsize: ``"compact"`` (last 100 days) or ``"full"``
                (20+ years).

        Returns:
            List of dicts with keys: ``timestamp_ms``, ``open``, ``high``,
            ``low``, ``close``, ``volume``, ``symbol``.  Bars are sorted
            oldest-first.  Timestamps are set to **00:00 UTC** of the
            trading date (per plan: strict less-than semantics in replay).
        """
        data = await self._get_json({
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
        })

        ts_data = data.get("Time Series (Daily)", {})
        bars = []
        for date_str, values in ts_data.items():
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc,
            )
            bars.append({
                "timestamp_ms": int(dt.timestamp() * 1000),
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": float(values.get("5. volume", 0)),
                "symbol": symbol,
            })

        bars.sort(key=lambda b: b["timestamp_ms"])
        logger.info(
            "Fetched %d daily bars for %s (latest: %s)",
            len(bars), symbol,
            bars[-1]["timestamp_ms"] if bars else "N/A",
        )
        return bars

    # ── Daily FX Rates ──────────────────────────────────────────────────

    async def fetch_fx_daily(
        self,
        from_currency: str,
        to_currency: str,
        *,
        outputsize: str = "compact",
    ) -> List[Dict[str, Any]]:
        """Fetch daily FX OHLC for a currency pair.

        Args:
            from_currency: Base (e.g. ``"EUR"``).
            to_currency: Quote (e.g. ``"USD"``).
            outputsize: ``"compact"`` or ``"full"``.

        Returns:
            List of dicts with keys: ``timestamp_ms``, ``open``, ``high``,
            ``low``, ``close``, ``symbol``.  Sorted oldest-first.
            ``symbol`` is ``"FX:EURUSD"`` style.
        """
        data = await self._get_json({
            "function": "FX_DAILY",
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "outputsize": outputsize,
        })

        ts_data = data.get("Time Series FX (Daily)", {})
        symbol = f"FX:{from_currency}{to_currency}"
        bars = []
        for date_str, values in ts_data.items():
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc,
            )
            bars.append({
                "timestamp_ms": int(dt.timestamp() * 1000),
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": 0.0,  # FX has no volume from Alpha Vantage
                "symbol": symbol,
            })

        bars.sort(key=lambda b: b["timestamp_ms"])
        logger.info(
            "Fetched %d daily FX bars for %s (latest: %s)",
            len(bars), symbol,
            bars[-1]["timestamp_ms"] if bars else "N/A",
        )
        return bars
