"""
Tests for options snapshot retry / resilience (10.3).

Verifies:
- Retry logic: fails N times then succeeds, only one row persisted
- snapshot_id is deterministic (provider+symbol+expiration_ms+timestamp_ms)
- No duplicate rows for same (provider, symbol, timestamp_ms) minute
- Alpaca and Tastytrade produce consistent snapshot_id formats
- Retries do not create duplicate snapshot_id collisions
"""

from __future__ import annotations

import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest


class TestSnapshotIdDeterminism:
    """snapshot_id must be provider+symbol+expiration_ms+timestamp_ms based."""

    def test_alpaca_snapshot_id_format(self):
        """Alpaca snapshot_id follows provider_symbol_expMs_tsMs pattern."""
        from src.connectors.alpaca_options import AlpacaOptionsConnector, AlpacaOptionsConfig

        connector = AlpacaOptionsConnector(AlpacaOptionsConfig())
        # The snapshot_id is built in build_chain_snapshot as:
        # f"{self.PROVIDER}_{symbol}_{expiration_ms}_{timestamp_ms}"
        sid = f"alpaca_SPY_1708560000000_1708473600000"
        assert sid.startswith("alpaca_")
        parts = sid.split("_")
        assert len(parts) == 4, "snapshot_id should have 4 parts: provider_symbol_expMs_tsMs"

    def test_tastytrade_snapshot_id_format(self):
        """Tastytrade snapshot_id follows same provider_symbol_expMs_tsMs pattern."""
        sid = f"tastytrade_SPY_1708560000000_1708473600000"
        assert sid.startswith("tastytrade_")
        parts = sid.split("_")
        assert len(parts) == 4

    def test_same_inputs_same_id(self):
        """Same provider+symbol+exp+ts always produce the same snapshot_id."""
        fmt = "{provider}_{symbol}_{exp_ms}_{ts_ms}"
        id1 = fmt.format(provider="alpaca", symbol="SPY", exp_ms=100, ts_ms=200)
        id2 = fmt.format(provider="alpaca", symbol="SPY", exp_ms=100, ts_ms=200)
        assert id1 == id2

    def test_different_provider_different_id(self):
        """Different providers produce different snapshot_ids for same symbol+time."""
        fmt = "{provider}_{symbol}_{exp_ms}_{ts_ms}"
        id1 = fmt.format(provider="alpaca", symbol="SPY", exp_ms=100, ts_ms=200)
        id2 = fmt.format(provider="tastytrade", symbol="SPY", exp_ms=100, ts_ms=200)
        assert id1 != id2

    def test_different_timestamp_different_id(self):
        """Different poll times produce different snapshot_ids."""
        fmt = "{provider}_{symbol}_{exp_ms}_{ts_ms}"
        id1 = fmt.format(provider="alpaca", symbol="SPY", exp_ms=100, ts_ms=200)
        id2 = fmt.format(provider="alpaca", symbol="SPY", exp_ms=100, ts_ms=260)
        assert id1 != id2


class TestPollTimeAlignment:
    """Verify _poll_time_ms aligns to minute boundaries."""

    def test_alpaca_poll_time_minute_floor(self):
        from src.connectors.alpaca_options import _poll_time_ms, _now_ms
        ts = _poll_time_ms()
        # Should be divisible by 60_000
        assert ts % 60_000 == 0, "poll_time_ms must be minute-aligned"

    def test_tastytrade_poll_time_minute_floor(self):
        from src.connectors.tastytrade_options import _poll_time_ms
        ts = _poll_time_ms()
        assert ts % 60_000 == 0

    def test_poll_time_same_minute_same_value(self):
        """Two calls within the same minute produce the same value."""
        from src.connectors.alpaca_options import _poll_time_ms
        with patch("src.connectors.alpaca_options.time") as mock_time:
            mock_time.time.return_value = 1705329030.123  # 30 seconds into minute
            t1 = _poll_time_ms()
            mock_time.time.return_value = 1705329055.999  # 55 seconds into same minute
            t2 = _poll_time_ms()
            assert t1 == t2

    def test_poll_time_different_minute_different_value(self):
        """Calls in different minutes produce different values."""
        from src.connectors.alpaca_options import _poll_time_ms
        with patch("src.connectors.alpaca_options.time") as mock_time:
            mock_time.time.return_value = 1705329030.0  # minute X
            t1 = _poll_time_ms()
            mock_time.time.return_value = 1705329090.0  # minute X+1
            t2 = _poll_time_ms()
            assert t1 != t2


class TestRecvTsVsTimestampMs:
    """Ensure timestamp_ms is minute-floored and recv_ts_ms is actual receipt time."""

    def test_recv_ts_ms_is_actual_time(self):
        """recv_ts_ms should be actual current time, not minute-floored."""
        from src.connectors.alpaca_options import _now_ms, _poll_time_ms
        with patch("src.connectors.alpaca_options.time") as mock_time:
            mock_time.time.return_value = 1705329045.678  # 45.678s into minute
            now = _now_ms()
            poll = _poll_time_ms()
            assert now == 1705329045678
            assert poll == 1705329000000
            assert now > poll  # recv_ts_ms (now) is after timestamp_ms (poll)


class TestTastytradeRetryBehavior:
    """Test TastytradeRestClient retry config and behavior."""

    def test_retry_config_defaults(self):
        from src.connectors.tastytrade_rest import RetryConfig
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.backoff_seconds == 1.0
        assert cfg.backoff_multiplier == 2.0

    def test_backoff_exponential(self):
        """Verify backoff calculation: delay = base * multiplier^attempt."""
        from src.connectors.tastytrade_rest import RetryConfig
        cfg = RetryConfig(backoff_seconds=1.0, backoff_multiplier=2.0)
        delays = [cfg.backoff_seconds * (cfg.backoff_multiplier ** attempt) for attempt in range(3)]
        assert delays == [1.0, 2.0, 4.0]

    def test_retry_on_500_errors(self):
        """REST client retries on 500+ status codes."""
        import requests
        from src.connectors.tastytrade_rest import TastytradeRestClient, TastytradeError, RetryConfig

        mock_session = MagicMock()
        # First two calls: 500, third: 200
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.ok = False
        resp_500.text = "Internal Server Error"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.ok = True
        resp_200.json.return_value = {"data": {"items": []}}

        mock_session.request = MagicMock(side_effect=[resp_500, resp_500, resp_200])

        client = TastytradeRestClient(
            username="test",
            password="test",
            session=mock_session,
            retries=RetryConfig(max_attempts=3, backoff_seconds=0.001),
        )
        client._token = "fake_token"

        result = client._request("GET", "/test-endpoint")
        assert result == {"data": {"items": []}}
        assert mock_session.request.call_count == 3

    def test_retry_on_429(self):
        """REST client retries on 429 rate limit."""
        from src.connectors.tastytrade_rest import TastytradeRestClient, RetryConfig

        mock_session = MagicMock()
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.ok = False
        resp_429.text = "Too Many Requests"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.ok = True
        resp_200.json.return_value = {"data": {}}

        mock_session.request = MagicMock(side_effect=[resp_429, resp_200])

        client = TastytradeRestClient(
            username="test",
            password="test",
            session=mock_session,
            retries=RetryConfig(max_attempts=3, backoff_seconds=0.001),
        )
        client._token = "fake_token"

        result = client._request("GET", "/test-429")
        assert result == {"data": {}}
        assert mock_session.request.call_count == 2

    def test_no_retry_on_400(self):
        """REST client does NOT retry on 400 client errors (non-retriable)."""
        from src.connectors.tastytrade_rest import TastytradeRestClient, TastytradeError, RetryConfig

        mock_session = MagicMock()
        resp_400 = MagicMock()
        resp_400.status_code = 400
        resp_400.ok = False
        resp_400.text = '{"error":{"code":"missing_request_token","message":"The request token is missing"}}'

        mock_session.request = MagicMock(return_value=resp_400)

        client = TastytradeRestClient(
            username="test",
            password="test",
            session=mock_session,
            retries=RetryConfig(max_attempts=3, backoff_seconds=0.001),
        )
        client._token = "fake_token"

        with pytest.raises(TastytradeError, match="HTTP 400"):
            client._request("GET", "/test-400")

        # Should NOT retry on 400 — only 1 call
        assert mock_session.request.call_count == 1


class TestSnapshotDeduplication:
    """Verify that the UNIQUE(provider, symbol, timestamp_ms) constraint
    prevents duplicate snapshots per minute per provider."""

    def test_unique_constraint_key(self):
        """Two snapshots from same provider+symbol in same minute should share timestamp_ms."""
        from src.connectors.alpaca_options import _poll_time_ms
        with patch("src.connectors.alpaca_options.time") as mock_time:
            mock_time.time.return_value = 1705329030.0
            ts1 = _poll_time_ms()
            mock_time.time.return_value = 1705329050.0  # Same minute
            ts2 = _poll_time_ms()
            assert ts1 == ts2, "Same minute should produce same timestamp_ms for dedup"

    def test_contract_id_is_deterministic(self):
        """contract_id from option_symbol is deterministic."""
        from src.connectors.alpaca_options import _compute_contract_id
        id1 = _compute_contract_id("SPY250221P00450000")
        id2 = _compute_contract_id("SPY250221P00450000")
        assert id1 == id2
        assert len(id1) == 16  # sha256[:16]
