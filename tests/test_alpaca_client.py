"""
Tests for Alpaca Data Client.

Verifies:
1. Fixed interval polling (no market-hours gating)
2. Restart dedupe from database
3. Bar ordering by bar_ts
4. Overlap window functionality

Run with: pytest tests/test_alpaca_client.py -v
"""

import asyncio
import pytest
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════
# Mock Database for Testing
# ═══════════════════════════════════════════════════════════════════════════

class MockDatabase:
    """Mock database that simulates get_latest_bar_ts."""
    
    def __init__(self):
        self._latest_bar_ts: Dict[tuple, int] = {}
    
    async def get_latest_bar_ts(
        self, source: str, symbol: str, bar_duration: int = 60
    ) -> Optional[int]:
        """Return stored latest bar timestamp or None."""
        key = (source, symbol, bar_duration)
        return self._latest_bar_ts.get(key)
    
    def set_latest_bar_ts(
        self, source: str, symbol: str, bar_duration: int, ts_ms: int
    ) -> None:
        """Set the latest bar timestamp for testing."""
        key = (source, symbol, bar_duration)
        self._latest_bar_ts[key] = ts_ms


# ═══════════════════════════════════════════════════════════════════════════
# Mock Event Bus
# ═══════════════════════════════════════════════════════════════════════════

class MockEventBus:
    """Mock event bus that collects published events."""
    
    def __init__(self):
        self.published: List[tuple] = []  # (topic, event)
    
    def publish(self, topic: str, event) -> None:
        self.published.append((topic, event))
    
    def clear(self):
        self.published.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Fixture Bars
# ═══════════════════════════════════════════════════════════════════════════

def make_alpaca_bar(ts_iso: str, o: float, h: float, l: float, c: float, v: float) -> dict:
    """Create a mock Alpaca API bar response."""
    return {"t": ts_iso, "o": o, "h": h, "l": l, "c": c, "v": v}


FIXTURE_BARS = [
    make_alpaca_bar("2024-01-15T14:30:00Z", 100.0, 101.0, 99.5, 100.5, 1000),
    make_alpaca_bar("2024-01-15T14:31:00Z", 100.5, 102.0, 100.0, 101.5, 1200),
    make_alpaca_bar("2024-01-15T14:32:00Z", 101.5, 103.0, 101.0, 102.5, 900),
    make_alpaca_bar("2024-01-15T14:33:00Z", 102.5, 104.0, 102.0, 103.5, 1100),
    make_alpaca_bar("2024-01-15T14:34:00Z", 103.5, 105.0, 103.0, 104.5, 800),
]


# ═══════════════════════════════════════════════════════════════════════════
# Dedupe Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRestartDedupe:
    @pytest.fixture
    def mock_db(self):
        return MockDatabase()
    
    @pytest.fixture
    def mock_bus(self):
        return MockEventBus()
    
    @pytest.mark.asyncio
    async def test_init_from_db_sets_last_bar_ts(self, mock_db, mock_bus):
        """On startup, init_from_db should populate last_bar_ts from database."""
        from src.connectors.alpaca_client import AlpacaDataClient
        
        # Pre-populate database with existing bar
        mock_db.set_latest_bar_ts("alpaca", "IBIT", 60, 1705329060000)  # 2024-01-15T14:31:00Z
        
        client = AlpacaDataClient(
            api_key="test_key",
            api_secret="test_secret",
            symbols=["IBIT"],
            event_bus=mock_bus,
            db=mock_db,
            poll_interval=60,
        )
        
        await client.init_from_db()
        
        assert client._last_bar_ts["IBIT"] == 1705329060000
        assert client._initialized is True
    
    @pytest.mark.asyncio
    async def test_init_from_db_no_existing_bars(self, mock_db, mock_bus):
        """If no bars exist in DB, last_bar_ts should remain empty."""
        from src.connectors.alpaca_client import AlpacaDataClient
        
        client = AlpacaDataClient(
            api_key="test_key",
            api_secret="test_secret",
            symbols=["IBIT", "BITO"],
            event_bus=mock_bus,
            db=mock_db,
            poll_interval=60,
        )
        
        await client.init_from_db()
        
        assert "IBIT" not in client._last_bar_ts
        assert "BITO" not in client._last_bar_ts
        assert client._initialized is True

    @pytest.mark.asyncio
    async def test_init_from_db_calls_get_latest_bar_ts_with_bar_duration(self, mock_bus):
        """init_from_db should call get_latest_bar_ts with bar_duration."""
        from src.connectors.alpaca_client import AlpacaDataClient

        mock_db = MagicMock()
        mock_db.get_latest_bar_ts = AsyncMock(return_value=None)

        client = AlpacaDataClient(
            api_key="test_key",
            api_secret="test_secret",
            symbols=["IBIT", "BITO"],
            event_bus=mock_bus,
            db=mock_db,
            poll_interval=60,
        )

        await client.init_from_db()

        mock_db.get_latest_bar_ts.assert_any_await(
            source="alpaca",
            symbol="IBIT",
            bar_duration=60,
        )
        mock_db.get_latest_bar_ts.assert_any_await(
            source="alpaca",
            symbol="BITO",
            bar_duration=60,
        )
        assert mock_db.get_latest_bar_ts.await_count == 2
    
    @pytest.mark.asyncio
    async def test_dedupe_only_emits_new_bars(self, mock_db, mock_bus):
        """After restart, should only emit bars newer than last_bar_ts."""
        from src.connectors.alpaca_client import AlpacaDataClient, _parse_rfc3339_to_ms
        
        # Set last bar to 14:31 (second bar)
        bar_14_31_ms = _parse_rfc3339_to_ms("2024-01-15T14:31:00Z")
        mock_db.set_latest_bar_ts("alpaca", "IBIT", 60, bar_14_31_ms)
        
        client = AlpacaDataClient(
            api_key="test_key",
            api_secret="test_secret",
            symbols=["IBIT"],
            event_bus=mock_bus,
            db=mock_db,
            poll_interval=60,
        )
        
        await client.init_from_db()
        
        # Mock fetch_bars to return all 5 bars
        async def mock_fetch_bars(symbol, limit=5):
            return FIXTURE_BARS[:limit]
        
        client.fetch_bars = mock_fetch_bars
        
        # Poll once
        emitted = await client.poll_once()
        
        # Should only emit bars after 14:31 (bars 3, 4, 5 = indices 2, 3, 4)
        assert emitted == 3
        
        # Verify emitted bars are 14:32, 14:33, 14:34
        emitted_events = [e for topic, e in mock_bus.published]
        assert len(emitted_events) == 3
        
        # Should be in order
        expected_times = [
            _parse_rfc3339_to_ms("2024-01-15T14:32:00Z"),
            _parse_rfc3339_to_ms("2024-01-15T14:33:00Z"),
            _parse_rfc3339_to_ms("2024-01-15T14:34:00Z"),
        ]
        for i, event in enumerate(emitted_events):
            event_ts_ms = int(event.timestamp * 1000)
            assert event_ts_ms == expected_times[i], f"Bar {i} has wrong timestamp"


class TestBarOrdering:
    @pytest.fixture
    def mock_bus(self):
        return MockEventBus()
    
    @pytest.mark.asyncio
    async def test_bars_emitted_in_ts_order(self, mock_bus):
        """Bars should be emitted in strict increasing bar_ts order."""
        from src.connectors.alpaca_client import AlpacaDataClient, _parse_rfc3339_to_ms
        
        client = AlpacaDataClient(
            api_key="test_key",
            api_secret="test_secret",
            symbols=["IBIT"],
            event_bus=mock_bus,
            db=None,
            poll_interval=60,
        )
        
        # API returns bars in descending order (most recent first)
        reversed_bars = list(reversed(FIXTURE_BARS))
        
        async def mock_fetch_bars(symbol, limit=5):
            return reversed_bars
        
        client.fetch_bars = mock_fetch_bars
        client._initialized = True
        
        await client.poll_once()
        
        # Verify ascending order
        emitted_events = [e for topic, e in mock_bus.published]
        
        for i in range(1, len(emitted_events)):
            prev_ts = emitted_events[i - 1].timestamp
            curr_ts = emitted_events[i].timestamp
            assert curr_ts > prev_ts, f"Bar {i} not in ascending order"


class TestFixedIntervalPolling:
    @pytest.fixture
    def mock_bus(self):
        return MockEventBus()
    
    def test_no_market_hours_logic(self, mock_bus):
        """Client should not have any market-hours-aware polling logic."""
        from src.connectors.alpaca_client import AlpacaDataClient
        
        client = AlpacaDataClient(
            api_key="test_key",
            api_secret="test_secret",
            symbols=["IBIT"],
            event_bus=mock_bus,
            db=None,
            poll_interval=60,
        )
        
        # Verify there's no market hours method
        assert not hasattr(client, '_is_market_open')
        assert not hasattr(client, '_poll_market_hours_aware')
        
        # Verify poll_interval is fixed
        assert client._poll_interval == 60
    
    def test_overlap_seconds_configurable(self, mock_bus):
        """Overlap seconds should be configurable."""
        from src.connectors.alpaca_client import AlpacaDataClient
        
        client = AlpacaDataClient(
            api_key="test_key",
            api_secret="test_secret",
            symbols=["IBIT"],
            event_bus=mock_bus,
            db=None,
            poll_interval=60,
            overlap_seconds=180,  # 3 bars
        )
        
        assert client._overlap_seconds == 180


class TestHealthStatus:
    @pytest.fixture
    def mock_bus(self):
        return MockEventBus()
    
    def test_health_status_includes_init_state(self, mock_bus):
        """Health status should report initialization state."""
        from src.connectors.alpaca_client import AlpacaDataClient
        
        client = AlpacaDataClient(
            api_key="test_key",
            api_secret="test_secret",
            symbols=["IBIT", "BITO"],
            event_bus=mock_bus,
            db=None,
            poll_interval=60,
        )
        
        status = client.get_health_status()
        
        assert status["name"] == "alpaca"
        assert "extras" in status
        assert status["extras"]["initialized"] is False
        assert status["extras"]["symbols"] == ["IBIT", "BITO"]
        assert status["extras"]["poll_interval_s"] == 60
        assert "overlap_seconds" in status["extras"]


class TestTimestampHandling:
    def test_parse_rfc3339_to_ms(self):
        """RFC3339 timestamps should be parsed to int milliseconds."""
        from src.connectors.alpaca_client import _parse_rfc3339_to_ms
        
        ts_ms = _parse_rfc3339_to_ms("2024-01-15T14:30:00Z")
        
        # Should be int
        assert isinstance(ts_ms, int)
        
        # Should be in milliseconds range
        assert ts_ms > 1e12
        
        # Verify correct value (2024-01-15 14:30:00 UTC = 1705329000)
        assert ts_ms == 1705329000000
    
    def test_last_bar_ts_is_int_ms(self):
        """Internal last_bar_ts tracking should use int milliseconds."""
        from src.connectors.alpaca_client import AlpacaDataClient
        
        bus = MockEventBus()
        client = AlpacaDataClient(
            api_key="test_key",
            api_secret="test_secret",
            symbols=["IBIT"],
            event_bus=bus,
            db=None,
            poll_interval=60,
        )
        
        # Manually set last_bar_ts
        client._last_bar_ts["IBIT"] = 1705329000000
        
        assert isinstance(client._last_bar_ts["IBIT"], int)
