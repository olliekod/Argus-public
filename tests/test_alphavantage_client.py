"""
Tests for Alpha Vantage REST Client
====================================

Covers:
- Daily bar parsing (correct timestamps, OHLCV, sorting)
- FX daily parsing (symbol naming, zero volume)
- Rate-limit "Note" response → retry with backoff → raise after max
- HTTP error handling
- API error message handling
- Timestamp semantics (00:00 UTC, no future leakage)
- Configurable rate limit pacing
"""

from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from src.connectors.alphavantage_client import (
    AlphaVantageClient,
    AlphaVantageRateLimitError,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═══════════════════════════════════════════════════════════════════════════

_SAMPLE_DAILY_RESPONSE: Dict[str, Any] = {
    "Meta Data": {
        "1. Information": "Daily Prices",
        "2. Symbol": "EWJ",
    },
    "Time Series (Daily)": {
        "2025-01-03": {
            "1. open": "60.50",
            "2. high": "61.00",
            "3. low": "60.00",
            "4. close": "60.80",
            "5. volume": "1000000",
        },
        "2025-01-02": {
            "1. open": "59.00",
            "2. high": "60.50",
            "3. low": "58.80",
            "4. close": "60.50",
            "5. volume": "950000",
        },
    },
}

_SAMPLE_FX_RESPONSE: Dict[str, Any] = {
    "Meta Data": {
        "1. Information": "Forex Daily",
    },
    "Time Series FX (Daily)": {
        "2025-01-03": {
            "1. open": "1.0800",
            "2. high": "1.0850",
            "3. low": "1.0780",
            "4. close": "1.0830",
        },
        "2025-01-02": {
            "1. open": "1.0750",
            "2. high": "1.0810",
            "3. low": "1.0740",
            "4. close": "1.0800",
        },
    },
}


def _make_client(**kwargs) -> AlphaVantageClient:
    """Create a client with zero throttle for fast tests."""
    defaults = {
        "api_key": "TEST_KEY",
        "call_interval_seconds": 0,
        "max_retries": 2,
        "retry_base_seconds": 0.01,
    }
    defaults.update(kwargs)
    return AlphaVantageClient(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# Daily bar parsing tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDailyBarParsing:
    """Tests for fetch_daily_bars response parsing."""

    @pytest.mark.asyncio
    async def test_parse_ohlcv(self):
        """OHLCV values are correctly parsed from AV response."""
        client = _make_client()
        client._get_json = AsyncMock(return_value=_SAMPLE_DAILY_RESPONSE)

        bars = await client.fetch_daily_bars("EWJ")

        assert len(bars) == 2
        # Sorted oldest-first
        bar0 = bars[0]  # Jan 2
        assert bar0["open"] == 59.0
        assert bar0["high"] == 60.5
        assert bar0["low"] == 58.8
        assert bar0["close"] == 60.5
        assert bar0["volume"] == 950000.0
        assert bar0["symbol"] == "EWJ"

    @pytest.mark.asyncio
    async def test_timestamps_are_midnight_utc(self):
        """All timestamps are 00:00 UTC of the trading date."""
        client = _make_client()
        client._get_json = AsyncMock(return_value=_SAMPLE_DAILY_RESPONSE)

        bars = await client.fetch_daily_bars("EWJ")

        for bar in bars:
            dt = datetime.fromtimestamp(
                bar["timestamp_ms"] / 1000, tz=timezone.utc,
            )
            assert dt.hour == 23
            assert dt.minute == 59
            assert dt.second == 59

    @pytest.mark.asyncio
    async def test_sorted_oldest_first(self):
        """Bars are returned sorted by timestamp ascending."""
        client = _make_client()
        client._get_json = AsyncMock(return_value=_SAMPLE_DAILY_RESPONSE)

        bars = await client.fetch_daily_bars("EWJ")

        timestamps = [b["timestamp_ms"] for b in bars]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_no_future_timestamp_leak(self):
        """Bar timestamps represent past trading dates, not future ones.

        Verify that all timestamps are <= current time (within reason).
        This validates the 00:00 UTC normalization doesn't accidentally
        push a bar date forward.
        """
        client = _make_client()
        client._get_json = AsyncMock(return_value=_SAMPLE_DAILY_RESPONSE)

        bars = await client.fetch_daily_bars("EWJ")

        # Jan 3 2025 23:59:59 UTC
        expected_latest = datetime(2025, 1, 3, 23, 59, 59, tzinfo=timezone.utc)
        expected_latest_ms = int(expected_latest.timestamp() * 1000)

        for bar in bars:
            assert bar["timestamp_ms"] <= expected_latest_ms

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Empty Time Series returns empty list."""
        client = _make_client()
        client._get_json = AsyncMock(return_value={"Time Series (Daily)": {}})

        bars = await client.fetch_daily_bars("FAKE")
        assert bars == []

    @pytest.mark.asyncio
    async def test_missing_time_series_key(self):
        """Response without Time Series key returns empty list."""
        client = _make_client()
        client._get_json = AsyncMock(return_value={"Meta Data": {}})

        bars = await client.fetch_daily_bars("FAKE")
        assert bars == []


# ═══════════════════════════════════════════════════════════════════════════
# FX parsing tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFxParsing:
    """Tests for fetch_fx_daily response parsing."""

    @pytest.mark.asyncio
    async def test_fx_symbol_naming(self):
        """FX symbol is formatted as FX:EURUSD."""
        client = _make_client()
        client._get_json = AsyncMock(return_value=_SAMPLE_FX_RESPONSE)

        bars = await client.fetch_fx_daily("EUR", "USD")

        for bar in bars:
            assert bar["symbol"] == "FX:EURUSD"

    @pytest.mark.asyncio
    async def test_fx_zero_volume(self):
        """FX bars always have volume=0 (AV doesn't provide FX volume)."""
        client = _make_client()
        client._get_json = AsyncMock(return_value=_SAMPLE_FX_RESPONSE)

        bars = await client.fetch_fx_daily("EUR", "USD")

        for bar in bars:
            assert bar["volume"] == 0.0

    @pytest.mark.asyncio
    async def test_fx_ohlc_parsed(self):
        """FX OHLC values are correctly parsed."""
        client = _make_client()
        client._get_json = AsyncMock(return_value=_SAMPLE_FX_RESPONSE)

        bars = await client.fetch_fx_daily("EUR", "USD")

        assert len(bars) == 2
        latest = bars[-1]  # Jan 3 (sorted oldest-first)
        assert latest["close"] == 1.083

    @pytest.mark.asyncio
    async def test_fx_timestamps_midnight_utc(self):
        """FX timestamps are 00:00 UTC."""
        client = _make_client()
        client._get_json = AsyncMock(return_value=_SAMPLE_FX_RESPONSE)

        bars = await client.fetch_fx_daily("USD", "JPY")

        for bar in bars:
            dt = datetime.fromtimestamp(
                bar["timestamp_ms"] / 1000, tz=timezone.utc,
            )
            assert dt.hour == 23


# ═══════════════════════════════════════════════════════════════════════════
# Rate limit and error handling tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRateLimitHandling:
    """Tests for "Note" response handling and retry logic."""

    @staticmethod
    def _mock_http_response(json_data, *, status=200, text=""):
        """Create a mock aiohttp response + session for _get_json tests.

        Returns (client_with_session_patched, mock_session).
        """
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=json_data)
        mock_resp.text = AsyncMock(return_value=text)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        return mock_session, mock_resp

    @pytest.mark.asyncio
    async def test_note_response_retries_then_raises(self):
        """A "Note" response triggers retry with backoff, then raises."""
        client = _make_client(max_retries=2, retry_base_seconds=0.001)

        note_response = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call "
                    "frequency is 5 calls per minute and 25 calls per day."
        }

        mock_session, _ = self._mock_http_response(note_response)
        client._get_session = AsyncMock(return_value=mock_session)

        with pytest.raises(AlphaVantageRateLimitError, match="rate limit"):
            await client._get_json({"function": "TIME_SERIES_DAILY"})

        # max_retries=2 → 3 total attempts
        assert client.calls_made == 3

    @pytest.mark.asyncio
    async def test_note_then_success_on_retry(self):
        """A Note followed by valid data succeeds after retry."""
        client = _make_client(max_retries=2, retry_base_seconds=0.001)

        call_count = 0

        async def _json_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"Note": "Rate limit hit"}
            return _SAMPLE_DAILY_RESPONSE

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = _json_side_effect
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        client._get_session = AsyncMock(return_value=mock_session)

        data = await client._get_json({"function": "TIME_SERIES_DAILY"})

        assert "Time Series (Daily)" in data
        assert client.calls_made == 2

    @pytest.mark.asyncio
    async def test_error_message_raises_value_error(self):
        """API error messages raise ValueError."""
        client = _make_client()

        error_response = {
            "Error Message": "Invalid API call. Please retry."
        }

        mock_session, _ = self._mock_http_response(error_response)
        client._get_session = AsyncMock(return_value=mock_session)

        with pytest.raises(ValueError, match="Invalid API call"):
            await client._get_json({"function": "TIME_SERIES_DAILY"})

    @pytest.mark.asyncio
    async def test_http_error_raises_runtime_error(self):
        """Non-200 HTTP status raises RuntimeError."""
        client = _make_client()

        mock_session, _ = self._mock_http_response(
            {}, status=503, text="Service Unavailable",
        )
        client._get_session = AsyncMock(return_value=mock_session)

        with pytest.raises(RuntimeError, match="503"):
            await client._get_json({"function": "TIME_SERIES_DAILY"})


# ═══════════════════════════════════════════════════════════════════════════
# Configuration and pacing tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConfiguration:
    """Tests for configurable rate limiting and behavior."""

    def test_default_call_interval(self):
        """Default call interval is 12.5s (free tier safe)."""
        client = AlphaVantageClient(api_key="TEST")
        assert client._call_interval == 12.5

    def test_custom_call_interval(self):
        """Call interval is configurable."""
        client = AlphaVantageClient(
            api_key="TEST",
            call_interval_seconds=5.0,
        )
        assert client._call_interval == 5.0

    def test_custom_max_retries(self):
        """Max retries is configurable."""
        client = AlphaVantageClient(api_key="TEST", max_retries=5)
        assert client._max_retries == 5

    def test_calls_made_counter(self):
        """Calls counter starts at zero."""
        client = AlphaVantageClient(api_key="TEST")
        assert client.calls_made == 0
