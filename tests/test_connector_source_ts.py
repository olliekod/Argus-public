"""
Tests for connector source_ts population and BarBuilder acceptance.

Verifies that all connectors (Bybit, Deribit, Yahoo) produce QuoteEvents
with valid source_ts fields that pass BarBuilder validation.

Run with:  python -m pytest tests/test_connector_source_ts.py -v
"""

import asyncio
import time

from src.core.bar_builder import BarBuilder, _minute_floor, _ts_sane
from src.core.bus import EventBus
from src.core.events import QuoteEvent, TOPIC_MARKET_BARS, TOPIC_MARKET_QUOTES
from src.connectors.bybit_ws import BybitWebSocket
from src.connectors.yahoo_client import _parse_yahoo_source_ts
from src.connectors.deribit_client import DeribitClient


def _drain(bus, timeout=0.5):
    """Wait until all bus queues are empty or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        depths = bus.get_queue_depths()
        if all(d == 0 for d in depths.values()):
            return
        time.sleep(0.01)


# ═══════════════════════════════════════════════════════════
#  Per-connector timestamp conversion tests
# ═══════════════════════════════════════════════════════════


class TestBybitTimestampConversion:
    """Verify Bybit WebSocket 'ts' (milliseconds) is correctly converted."""

    def test_bybit_ms_to_seconds(self):
        """Bybit sends ts in milliseconds; source_ts must be epoch seconds."""
        raw_ms = 1_700_000_123_456  # Bybit WS 'ts' field
        source_ts = float(raw_ms) / 1000.0
        assert _ts_sane(source_ts), (
            f"Converted Bybit ts {source_ts} should be in sane epoch-seconds range"
        )

    def test_bybit_ms_unconverted_fails(self):
        """Raw Bybit ms timestamp must NOT pass _ts_sane."""
        raw_ms = 1_700_000_123_456.0
        assert not _ts_sane(raw_ms), (
            "Raw Bybit ms value should be rejected as too large"
        )

    def test_bybit_zero_ts_fallback(self):
        """When Bybit 'ts' is missing, fallback to time.time()."""
        now = time.time()
        assert _ts_sane(now), "Current time.time() should be sane"

    def test_bybit_quote_accepted_by_bar_builder(self):
        """End-to-end: a Bybit-style quote with correct source_ts builds a bar."""
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            # Simulate Bybit: ts=1700000060456 (ms), converted to seconds
            bybit_ts_ms = 1_700_000_060_456
            source_ts = bybit_ts_ms / 1000.0  # 1700000060.456
            now = time.time()

            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60

            q1 = QuoteEvent(
                symbol="BTCUSDT",
                bid=42000.0,
                ask=42001.0,
                mid=42000.5,
                last=42000.5,
                timestamp=minute0 + 5,
                source="bybit",
                volume_24h=1000.0,
                source_ts=minute0 + 5,
            )
            bb._on_quote(q1)

            # Trigger bar close with next-minute tick
            q2 = QuoteEvent(
                symbol="BTCUSDT",
                bid=42010.0,
                ask=42011.0,
                mid=42010.5,
                last=42010.5,
                timestamp=minute1 + 5,
                source="bybit",
                volume_24h=1020.0,
                source_ts=minute1 + 5,
            )
            bb._on_quote(q2)
            _drain(bus)

            assert len(emitted) == 1, "Bar should have been emitted"
            bar = emitted[0]
            assert bar.source == "bybit"
            assert bar.n_ticks == 1
            assert bar.first_source_ts > 0

            # No rejections
            status = bb.get_status()
            assert status["extras"]["quotes_rejected_total"] == 0
        finally:
            bus.stop()

    def test_bybit_zero_quote_rejected(self):
        """Zero bid/ask/last payloads are rejected before publishing quotes."""
        bus = EventBus()
        published = []
        bus.subscribe(TOPIC_MARKET_QUOTES, lambda quote: published.append(quote))
        bus.start()
        bybit = BybitWebSocket(symbols=["BTCUSDT"], event_bus=bus)

        try:
            payload = {
                "topic": "tickers.BTCUSDT",
                "ts": 1_700_000_123_456,
                "data": {
                    "symbol": "BTCUSDT",
                    "bid1Price": "0",
                    "ask1Price": "0",
                    "lastPrice": "0",
                    "markPrice": "0",
                    "indexPrice": "0",
                    "volume24h": "0",
                    "turnover24h": "0",
                    "fundingRate": "0",
                    "nextFundingTime": None,
                    "openInterest": "0",
                    "openInterestValue": "0",
                    "price24hPcnt": "0",
                },
            }
            asyncio.run(bybit._handle_ticker(payload))
            _drain(bus)

            assert not published
            health = bybit.get_health_status()
            extras = health["extras"]
            assert extras["quotes_invalid_total"] == 1
            assert extras["quotes_invalid_by_reason"]["zero_bidask"] == 1
            assert extras["quotes_invalid_by_symbol"]["BTCUSDT"] == 1
        finally:
            bus.stop()


class TestDeribitTimestampConversion:
    """Verify Deribit REST timestamp handling."""

    def test_deribit_us_out_microseconds_converted(self):
        """Deribit usOut microseconds should convert to sane epoch seconds."""
        data = {"usOut": 1_700_000_000_000_000}
        source_ts, reason, label, raw = DeribitClient._extract_source_ts(data)
        assert label == "usOut"
        assert raw == 1_700_000_000_000_000
        assert reason in (None, "converted_us")
        assert _ts_sane(source_ts)

    def test_deribit_ms_api_timestamp_converted(self):
        """Deribit API returns timestamps in ms; conversion to seconds is sane."""
        api_ts_ms = 1_700_000_000_000  # Deribit API timestamp
        source_ts = api_ts_ms / 1000.0
        assert _ts_sane(source_ts)

    def test_deribit_quote_accepted_by_bar_builder(self):
        """End-to-end: Deribit-style quote with source_ts builds a bar."""
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60

            q1 = QuoteEvent(
                symbol="BTC-INDEX",
                bid=42000.0,
                ask=42000.0,
                mid=42000.0,
                last=42000.0,
                timestamp=minute0 + 10,
                source="deribit",
                source_ts=minute0 + 10,
            )
            bb._on_quote(q1)

            q2 = QuoteEvent(
                symbol="BTC-INDEX",
                bid=42100.0,
                ask=42100.0,
                mid=42100.0,
                last=42100.0,
                timestamp=minute1 + 10,
                source="deribit",
                source_ts=minute1 + 10,
            )
            bb._on_quote(q2)
            _drain(bus)

            assert len(emitted) == 1
            bar = emitted[0]
            assert bar.source == "deribit"
            status = bb.get_status()
            assert status["extras"]["quotes_rejected_total"] == 0
        finally:
            bus.stop()


class TestYahooTimestampConversion:
    """Verify Yahoo Finance timestamp handling."""

    def test_yahoo_regular_market_time_seconds(self):
        """Yahoo regularMarketTime should parse as epoch seconds."""
        source_ts, reason, raw = _parse_yahoo_source_ts({"regularMarketTime": 1_700_000_123})
        assert raw == 1_700_000_123
        assert reason is None
        assert _ts_sane(source_ts)

    def test_yahoo_regular_market_time_ms_converted(self):
        """Yahoo regularMarketTime in ms should be converted to seconds."""
        source_ts, reason, raw = _parse_yahoo_source_ts({"regularMarketTime": 1_700_000_123_000})
        assert raw == 1_700_000_123_000
        assert reason == "converted_ms"
        assert _ts_sane(source_ts)

    def test_yahoo_quote_accepted_by_bar_builder(self):
        """End-to-end: Yahoo-style quote with source_ts builds a bar."""
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60

            q1 = QuoteEvent(
                symbol="IBIT",
                bid=50.0,
                ask=50.0,
                mid=50.0,
                last=50.0,
                timestamp=minute0 + 20,
                source="yahoo",
                volume_24h=5_000_000.0,
                source_ts=minute0 + 20,
            )
            bb._on_quote(q1)

            q2 = QuoteEvent(
                symbol="IBIT",
                bid=50.5,
                ask=50.5,
                mid=50.5,
                last=50.5,
                timestamp=minute1 + 20,
                source="yahoo",
                volume_24h=5_100_000.0,
                source_ts=minute1 + 20,
            )
            bb._on_quote(q2)
            _drain(bus)

            assert len(emitted) == 1
            bar = emitted[0]
            assert bar.source == "yahoo"
            status = bb.get_status()
            assert status["extras"]["quotes_rejected_total"] == 0
        finally:
            bus.stop()


# ═══════════════════════════════════════════════════════════
#  BarBuilder acceptance: valid quotes accumulate into bars
# ═══════════════════════════════════════════════════════════


class TestBarBuilderAcceptance:
    """Confirm valid quotes accumulate into bars with correct provenance."""

    def test_multiple_ticks_build_bar(self):
        """Multiple valid ticks in the same minute produce one bar on close."""
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60

            for i in range(5):
                q = QuoteEvent(
                    symbol="BTC",
                    bid=100.0 + i,
                    ask=100.5 + i,
                    mid=100.25 + i,
                    last=100.25 + i,
                    timestamp=minute0 + 10 * i + 1,
                    source="test",
                    volume_24h=1000.0 + i * 10,
                    source_ts=minute0 + 10 * i + 1,
                )
                bb._on_quote(q)

            # Close bar
            q_close = QuoteEvent(
                symbol="BTC",
                bid=106.0,
                ask=106.5,
                mid=106.25,
                last=106.25,
                timestamp=minute1 + 1,
                source="test",
                volume_24h=1060.0,
                source_ts=minute1 + 1,
            )
            bb._on_quote(q_close)
            _drain(bus)

            assert len(emitted) == 1
            bar = emitted[0]
            assert bar.n_ticks == 5
            assert bar.first_source_ts > 0
            assert bar.last_source_ts >= bar.first_source_ts
            assert bar.source_ts == bar.first_source_ts

            status = bb.get_status()
            assert status["extras"]["quotes_rejected_total"] == 0
        finally:
            bus.stop()

    def test_source_ts_provenance_tracked(self):
        """Bar carries first/last source_ts from the ingested quotes."""
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60

            ts1 = minute0 + 5
            ts2 = minute0 + 35

            bb._on_quote(QuoteEvent(
                symbol="ETH", bid=2000, ask=2001, mid=2000.5, last=2000.5,
                timestamp=ts1, source="test", volume_24h=500, source_ts=ts1,
            ))
            bb._on_quote(QuoteEvent(
                symbol="ETH", bid=2010, ask=2011, mid=2010.5, last=2010.5,
                timestamp=ts2, source="test", volume_24h=510, source_ts=ts2,
            ))
            # Close
            bb._on_quote(QuoteEvent(
                symbol="ETH", bid=2020, ask=2021, mid=2020.5, last=2020.5,
                timestamp=minute1 + 1, source="test", volume_24h=520,
                source_ts=minute1 + 1,
            ))
            _drain(bus)

            assert len(emitted) == 1
            bar = emitted[0]
            assert bar.first_source_ts == ts1
            assert bar.last_source_ts == ts2
        finally:
            bus.stop()


# ═══════════════════════════════════════════════════════════
#  BarBuilder rejection: invalid timestamps are rejected
# ═══════════════════════════════════════════════════════════


class TestBarBuilderRejection:
    """Confirm invalid timestamps are properly rejected."""

    def test_zero_source_ts_rejected(self):
        """source_ts=0.0 (the default) must be rejected."""
        bus = EventBus()
        bb = BarBuilder(bus)

        base = 1_700_000_000.0
        q = QuoteEvent(
            symbol="BTC", bid=100, ask=101, mid=100.5, last=100.5,
            timestamp=base + 1, source="test", volume_24h=100,
            source_ts=0.0,
        )
        bb._on_quote(q)

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 1

    def test_millisecond_source_ts_rejected(self):
        """source_ts in milliseconds must be rejected."""
        bus = EventBus()
        bb = BarBuilder(bus)

        base = 1_700_000_000.0
        q = QuoteEvent(
            symbol="BTC", bid=100, ask=101, mid=100.5, last=100.5,
            timestamp=base + 1, source="bybit", volume_24h=100,
            source_ts=1_700_000_001_000.0,  # ms, not seconds
        )
        bb._on_quote(q)

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 1

    def test_negative_source_ts_rejected(self):
        """Negative source_ts must be rejected."""
        bus = EventBus()
        bb = BarBuilder(bus)

        base = 1_700_000_000.0
        q = QuoteEvent(
            symbol="BTC", bid=100, ask=101, mid=100.5, last=100.5,
            timestamp=base + 1, source="test", volume_24h=100,
            source_ts=-1.0,
        )
        bb._on_quote(q)

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 1

    def test_missing_timestamp_rejected(self):
        """timestamp=0 must be rejected."""
        bus = EventBus()
        bb = BarBuilder(bus)

        q = QuoteEvent(
            symbol="BTC", bid=100, ask=101, mid=100.5, last=100.5,
            timestamp=0.0, source="test", volume_24h=100,
            source_ts=1_700_000_001.0,
        )
        bb._on_quote(q)

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 1
