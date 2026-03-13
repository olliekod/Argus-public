"""
Tests for Bybit REST backfill module.

Tests:
- klines_to_bar_rows conversion
- backfill_klines chunking with mocked HTTP
- DB idempotency: INSERT OR IGNORE preserves existing bars
- backfill fills only the gap
- No invented data when API returns partial results
- Instrument discovery parsing
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.connectors.bybit_rest import (
    BybitKline,
    BybitRestClient,
    BybitInstrument,
    klines_to_bar_rows,
    _interval_to_ms,
)
from src.core.database import Database


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures / helpers
# ═══════════════════════════════════════════════════════════════════════════════

BASE_MS = 1_700_000_000_000  # ~2023-11-14 UTC


def _ms(offset_s: int) -> int:
    return BASE_MS + offset_s * 1000


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _make_db() -> Database:
    db = Database(":memory:")
    await db.connect()
    return db


def _make_kline(offset_min: int, close: float = 100.0) -> BybitKline:
    """Create a synthetic kline at BASE_MS + offset_min minutes."""
    return BybitKline(
        timestamp_ms=_ms(offset_min * 60),
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1000.0,
        turnover=100000.0,
    )


def _make_api_response(klines: List[BybitKline]) -> dict:
    """Build a Bybit kline API response dict from BybitKline objects."""
    # Bybit returns newest first
    sorted_desc = sorted(klines, key=lambda k: k.timestamp_ms, reverse=True)
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                [
                    str(k.timestamp_ms),
                    str(k.open),
                    str(k.high),
                    str(k.low),
                    str(k.close),
                    str(k.volume),
                    str(k.turnover),
                ]
                for k in sorted_desc
            ]
        },
    }


def _make_instruments_response(symbols: List[str]) -> dict:
    """Build a Bybit instruments-info API response."""
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {
                    "symbol": s,
                    "baseCoin": s.replace("USDT", ""),
                    "quoteCoin": "USDT",
                    "status": "Trading",
                    "contractType": "LinearPerpetual",
                    "launchTime": str(BASE_MS),
                    "settleCoin": "USDT",
                }
                for s in symbols
            ],
            "nextPageCursor": "",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  A) klines_to_bar_rows conversion
# ═══════════════════════════════════════════════════════════════════════════════


class TestKlinesToBarRows:

    def test_basic_conversion(self):
        klines = [_make_kline(0, 100.0), _make_kline(1, 101.0)]
        rows = klines_to_bar_rows(klines, source="bybit",
                                  symbol="BTCUSDT", bar_duration=60)
        assert len(rows) == 2

        # Check first row structure
        ts_str, sym, src, o, h, l, c, v, *rest = rows[0]
        assert sym == "BTCUSDT"
        assert src == "bybit"
        assert c == 100.0
        assert rest[-1] == 60  # bar_duration

    def test_timestamp_is_utc_iso(self):
        klines = [_make_kline(0)]
        rows = klines_to_bar_rows(klines, source="bybit", symbol="BTCUSDT")
        ts_str = rows[0][0]
        # Should be parseable as UTC
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        assert dt.year > 2000

    def test_close_reason_is_backfill(self):
        klines = [_make_kline(0)]
        rows = klines_to_bar_rows(klines, source="bybit", symbol="BTCUSDT")
        close_reason = rows[0][13]
        assert close_reason == 4  # REST_BACKFILL

    def test_empty_klines(self):
        rows = klines_to_bar_rows([], source="bybit", symbol="BTCUSDT")
        assert rows == []


# ═══════════════════════════════════════════════════════════════════════════════
#  B) DB upsert idempotency
# ═══════════════════════════════════════════════════════════════════════════════


class TestDBBackfillIdempotency:

    @pytest.mark.asyncio
    async def test_insert_or_ignore_no_duplicates(self):
        """Running backfill twice yields same bar count."""
        db = await _make_db()
        try:
            klines = [_make_kline(i) for i in range(10)]
            rows = klines_to_bar_rows(klines, source="bybit",
                                      symbol="BTCUSDT", bar_duration=60)

            inserted1 = await db.upsert_bars_backfill(rows)
            assert inserted1 == 10

            # Run again — should insert 0
            inserted2 = await db.upsert_bars_backfill(rows)
            assert inserted2 == 0

            # Total should still be 10
            cursor = await db._connection.execute(
                "SELECT COUNT(*) FROM market_bars")
            total = (await cursor.fetchone())[0]
            assert total == 10
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_existing_live_bars_not_overwritten(self):
        """Backfill bars don't overwrite live-collected bars."""
        db = await _make_db()
        try:
            # Insert a "live" bar with specific close price
            ts_str = datetime.fromtimestamp(
                _ms(0) / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")
            await db._connection.execute(
                "INSERT INTO market_bars "
                "(timestamp, symbol, source, open, high, low, close, volume, "
                "tick_count, n_ticks, first_source_ts, last_source_ts, "
                "late_ticks_dropped, close_reason, bar_duration) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts_str, "BTCUSDT", "bybit", 99.0, 102.0, 98.0, 101.0, 500.0,
                 15, 15, _ms(0) / 1000.0, _ms(0) / 1000.0, 0, 1, 60),
            )
            await db._connection.commit()

            # Backfill a kline at the same timestamp with different close
            klines = [BybitKline(
                timestamp_ms=_ms(0), open=99.5, high=101.0,
                low=99.0, close=100.0, volume=1000.0, turnover=100000.0,
            )]
            rows = klines_to_bar_rows(klines, source="bybit",
                                      symbol="BTCUSDT", bar_duration=60)

            inserted = await db.upsert_bars_backfill(rows)
            assert inserted == 0  # should NOT insert (conflict)

            # Verify original live bar preserved
            cursor = await db._connection.execute(
                "SELECT close, tick_count FROM market_bars "
                "WHERE source='bybit' AND symbol='BTCUSDT'")
            row = await cursor.fetchone()
            assert row[0] == 101.0  # original live close, not 100.0
            assert row[1] == 15     # original tick_count, not 0
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_fills_only_the_gap(self):
        """Backfill inserts only missing bars, not existing ones."""
        db = await _make_db()
        try:
            # Insert bars at minutes 0, 1, 2, 5, 6, 7 (gap at 3, 4)
            existing_mins = [0, 1, 2, 5, 6, 7]
            for m in existing_mins:
                ts_str = datetime.fromtimestamp(
                    _ms(m * 60) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                await db._connection.execute(
                    "INSERT INTO market_bars "
                    "(timestamp, symbol, source, open, high, low, close, "
                    "volume, tick_count, n_ticks, first_source_ts, "
                    "last_source_ts, late_ticks_dropped, close_reason, "
                    "bar_duration) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts_str, "BTCUSDT", "bybit", 100, 101, 99, 100.5, 500,
                     10, 10, 0, 0, 0, 1, 60),
                )
            await db._connection.commit()

            # Backfill all 8 minutes (0-7)
            klines = [_make_kline(m) for m in range(8)]
            rows = klines_to_bar_rows(klines, source="bybit",
                                      symbol="BTCUSDT", bar_duration=60)

            inserted = await db.upsert_bars_backfill(rows)
            assert inserted == 2  # only minutes 3 and 4

            cursor = await db._connection.execute(
                "SELECT COUNT(*) FROM market_bars")
            total = (await cursor.fetchone())[0]
            assert total == 8  # 6 existing + 2 new
        finally:
            await db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  C) get_last_bar_timestamp_ms / get_first_bar_timestamp_ms
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarTimestampQueries:

    @pytest.mark.asyncio
    async def test_last_bar_timestamp_ms(self):
        db = await _make_db()
        try:
            klines = [_make_kline(i) for i in range(5)]
            rows = klines_to_bar_rows(klines, source="bybit",
                                      symbol="BTCUSDT", bar_duration=60)
            await db.upsert_bars_backfill(rows)

            last = await db.get_last_bar_timestamp_ms("bybit", "BTCUSDT", 60)
            assert last == _ms(4 * 60)
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_first_bar_timestamp_ms(self):
        db = await _make_db()
        try:
            klines = [_make_kline(i) for i in range(5)]
            rows = klines_to_bar_rows(klines, source="bybit",
                                      symbol="BTCUSDT", bar_duration=60)
            await db.upsert_bars_backfill(rows)

            first = await db.get_first_bar_timestamp_ms("bybit", "BTCUSDT", 60)
            assert first == _ms(0)
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_no_bars_returns_none(self):
        db = await _make_db()
        try:
            last = await db.get_last_bar_timestamp_ms("bybit", "BTCUSDT", 60)
            assert last is None
            first = await db.get_first_bar_timestamp_ms("bybit", "BTCUSDT", 60)
            assert first is None
        finally:
            await db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  D) REST client with mocked HTTP
# ═══════════════════════════════════════════════════════════════════════════════


class TestBybitRestClientMocked:

    @pytest.mark.asyncio
    async def test_get_klines_parses_correctly(self):
        """Mock a single kline API call and verify parsing."""
        klines = [_make_kline(i) for i in range(5)]
        response = _make_api_response(klines)

        client = BybitRestClient()
        client._get = AsyncMock(return_value=response)

        result = await client.get_klines("BTCUSDT", "1", _ms(0), _ms(300))
        assert len(result) == 5
        # Should be sorted ascending
        assert result[0].timestamp_ms < result[-1].timestamp_ms
        assert isinstance(result[0], BybitKline)

    @pytest.mark.asyncio
    async def test_backfill_klines_chunks_correctly(self):
        """Verify backfill_klines makes multiple API calls for large ranges."""
        # Simulate 400 minutes of data (needs 2 chunks of 200)
        chunk1 = [_make_kline(i) for i in range(200)]
        chunk2 = [_make_kline(200 + i) for i in range(200)]

        call_count = 0

        async def mock_get_klines(symbol, interval, start_ms, end_ms,
                                  limit, category):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return chunk1
            elif call_count == 2:
                return chunk2
            else:
                return []

        client = BybitRestClient()
        client.get_klines = AsyncMock(side_effect=mock_get_klines)

        result = await client.backfill_klines(
            "BTCUSDT", _ms(0), _ms(400 * 60), "1", "linear")

        assert len(result) == 400
        assert call_count >= 2  # at least 2 chunks

    @pytest.mark.asyncio
    async def test_backfill_no_data_returns_empty(self):
        """When API returns no klines, backfill returns empty."""
        client = BybitRestClient()
        client.get_klines = AsyncMock(return_value=[])

        result = await client.backfill_klines(
            "BTCUSDT", _ms(0), _ms(600), "1", "linear")
        assert result == []

    @pytest.mark.asyncio
    async def test_backfill_deduplicates(self):
        """Overlapping chunks don't produce duplicates."""
        overlap_klines = [_make_kline(i) for i in range(5)]

        call_count = 0

        async def mock_get_klines(symbol, interval, start_ms, end_ms,
                                  limit, category):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return overlap_klines  # same data twice
            return []

        client = BybitRestClient()
        client.get_klines = AsyncMock(side_effect=mock_get_klines)

        result = await client.backfill_klines(
            "BTCUSDT", _ms(0), _ms(300), "1", "linear")

        # Should deduplicate
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_discover_perpetuals(self):
        """Test instrument discovery parsing with mocked response."""
        response = _make_instruments_response(
            ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"])

        client = BybitRestClient()
        client._get = AsyncMock(return_value=response)

        instruments = await client.discover_perpetuals(
            quote_coin="USDT", base_coins=["BTC", "ETH"])

        assert len(instruments) == 2
        symbols = {i.symbol for i in instruments}
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols

    @pytest.mark.asyncio
    async def test_discover_all_perpetuals(self):
        """Without base_coins filter, returns all."""
        response = _make_instruments_response(
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

        client = BybitRestClient()
        client._get = AsyncMock(return_value=response)

        instruments = await client.discover_perpetuals(quote_coin="USDT")
        assert len(instruments) == 3


# ═══════════════════════════════════════════════════════════════════════════════
#  E) bar_health DB method
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarHealth:

    @pytest.mark.asyncio
    async def test_bar_health_empty(self):
        db = await _make_db()
        try:
            health = await db.get_bar_health()
            assert health == []
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_bar_health_with_data(self):
        db = await _make_db()
        try:
            klines = [_make_kline(i) for i in range(5)]
            rows = klines_to_bar_rows(klines, source="bybit",
                                      symbol="BTCUSDT", bar_duration=60)
            await db.upsert_bars_backfill(rows)

            health = await db.get_bar_health()
            assert len(health) == 1
            assert health[0]["source"] == "bybit"
            assert health[0]["symbol"] == "BTCUSDT"
            assert health[0]["bar_count"] == 5
            assert health[0]["last_ts_age_s"] > 0  # historical data = old
        finally:
            await db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  F) interval_to_ms helper
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntervalToMs:

    def test_1_minute(self):
        assert _interval_to_ms("1") == 60_000

    def test_5_minute(self):
        assert _interval_to_ms("5") == 300_000

    def test_1_hour(self):
        assert _interval_to_ms("60") == 3_600_000

    def test_daily(self):
        assert _interval_to_ms("D") == 86_400_000

    def test_unknown_defaults_to_1m(self):
        assert _interval_to_ms("xyz") == 60_000


# ═══════════════════════════════════════════════════════════════════════════════
#  G) Determinism: same klines -> same rows
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:

    def test_klines_to_rows_deterministic(self):
        klines = [_make_kline(i, close=100.0 + i) for i in range(10)]
        r1 = klines_to_bar_rows(klines, "bybit", "BTCUSDT", 60)
        r2 = klines_to_bar_rows(klines, "bybit", "BTCUSDT", 60)
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_db_backfill_deterministic(self):
        """Two independent DBs with same backfill -> identical contents."""
        db1 = await _make_db()
        db2 = await _make_db()
        try:
            klines = [_make_kline(i, close=100.0 + i * 0.5) for i in range(20)]
            rows = klines_to_bar_rows(klines, "bybit", "BTCUSDT", 60)

            await db1.upsert_bars_backfill(rows)
            await db2.upsert_bars_backfill(rows)

            cursor1 = await db1._connection.execute(
                "SELECT * FROM market_bars ORDER BY timestamp")
            cursor2 = await db2._connection.execute(
                "SELECT * FROM market_bars ORDER BY timestamp")

            rows1 = await cursor1.fetchall()
            rows2 = await cursor2.fetchall()

            assert len(rows1) == len(rows2) == 20
            for a, b in zip(rows1, rows2):
                # Compare all columns except id (autoincrement)
                assert tuple(a)[1:] == tuple(b)[1:]
        finally:
            await db1.close()
            await db2.close()
