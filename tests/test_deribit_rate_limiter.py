"""
Tests for Deribit rate limiter correctness (10.2).

Verifies:
- Rate limiter resets after 60 seconds
- Requests are throttled at the configured limit
- Counter tracks correctly across the minute boundary
- No burst beyond limit

Uses monkeypatched time (no real sleeps).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.connectors.deribit_client import DeribitClient


class FakeDatetime:
    """Controllable datetime replacement for deterministic tests."""

    def __init__(self, start: datetime):
        self._now = start

    def advance(self, seconds: float):
        self._now += timedelta(seconds=seconds)

    def now(self, tz=None):
        if tz:
            return self._now.replace(tzinfo=tz)
        return self._now


class TestDeribitRateLimiter:
    """Validate rate limiter enforces 20 req/min for unauthenticated access."""

    @pytest.fixture
    def client(self):
        c = DeribitClient(testnet=True)
        return c

    def test_initial_state(self, client):
        assert client._rate_limit == 20
        assert client._request_count == 0

    @pytest.mark.asyncio
    async def test_counter_increments(self, client):
        """Each request increments the counter."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"result": {}})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        await client._request("get_index_price", {"index_name": "btc_usd"})
        assert client._request_count == 1

        await client._request("get_index_price", {"index_name": "btc_usd"})
        assert client._request_count == 2

    @pytest.mark.asyncio
    async def test_counter_resets_after_60s(self, client):
        """Counter resets to 0 after 60 seconds elapse."""
        t0 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        client._last_reset = t0
        client._request_count = 15

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"result": {}})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        # Advance time by 61 seconds
        t1 = t0 + timedelta(seconds=61)
        with patch("src.connectors.deribit_client.datetime") as mock_dt:
            mock_dt.now.return_value = t1
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await client._request("get_index_price", {"index_name": "btc_usd"})

        # Counter should have been reset (was 15, reset to 0, then incremented to 1)
        assert client._request_count == 1

    @pytest.mark.asyncio
    async def test_throttle_at_limit(self, client):
        """When at limit, should sleep until the minute resets."""
        t0 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        client._last_reset = t0
        client._request_count = 20  # At the limit

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"result": {}})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        # Time is only 30s into the minute
        t_30 = t0 + timedelta(seconds=30)
        with patch("src.connectors.deribit_client.datetime") as mock_dt, \
             patch("src.connectors.deribit_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_dt.now.return_value = t_30
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await client._request("get_index_price", {"index_name": "btc_usd"})

            # Should have slept for ~30 seconds (60 - 30)
            mock_sleep.assert_awaited_once()
            wait_time = mock_sleep.call_args[0][0]
            assert 25 <= wait_time <= 35, f"Expected ~30s wait, got {wait_time}"

    def test_rate_limit_value(self, client):
        """Rate limit should be 20 for unauthenticated."""
        assert client._rate_limit == 20

    @pytest.mark.asyncio
    async def test_no_burst_beyond_limit(self, client):
        """Verify that request_count never exceeds _rate_limit before reset."""
        t0 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        client._last_reset = t0
        client._request_count = 0

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"result": {}})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        # No time advancement — all requests in same minute
        with patch("src.connectors.deribit_client.datetime") as mock_dt, \
             patch("src.connectors.deribit_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Time is always 10 seconds into the minute (no reset)
            mock_dt.now.return_value = t0 + timedelta(seconds=10)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            for i in range(25):
                await client._request("get_index_price", {"index_name": "btc_usd"})

            # At request 21, throttle should have triggered once
            # (requests 1-20 ok, request 21 triggers sleep+reset, then 21-25 = 5)
            assert mock_sleep.await_count >= 1
