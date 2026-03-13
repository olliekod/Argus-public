"""
Tests for source_ts sanity validation in BarBuilder.

Run with:  python -m pytest tests/test_timestamp_validation.py -v
"""

from src.core.bar_builder import BarBuilder, _minute_floor, _ts_sane
from src.core.bus import EventBus
from src.core.events import QuoteEvent, TOPIC_MARKET_BARS


def _quote(symbol, price, volume, ts, source_ts=None):
    return QuoteEvent(
        symbol=symbol,
        bid=price - 0.01,
        ask=price + 0.01,
        mid=price,
        last=price,
        timestamp=ts,
        source="test",
        volume_24h=volume,
        source_ts=source_ts if source_ts is not None else ts,
        event_ts=ts,
        receive_time=ts,
    )


class TestTsSane:
    """Unit tests for the _ts_sane helper."""

    def test_normal_epoch_seconds(self):
        assert _ts_sane(1_700_000_000.0)  # 2023
        assert _ts_sane(1_800_000_000.0)  # 2027
        assert _ts_sane(1_600_000_000.0)  # 2020

    def test_milliseconds_rejected(self):
        # Epoch in milliseconds (1e12+)
        assert not _ts_sane(1_700_000_000_000.0)

    def test_too_old_rejected(self):
        assert not _ts_sane(1_000_000_000.0)  # 2001

    def test_too_far_future_rejected(self):
        assert not _ts_sane(2_200_000_000.0)  # ~2039

    def test_zero_rejected(self):
        assert not _ts_sane(0.0)

    def test_negative_rejected(self):
        assert not _ts_sane(-1.0)


class TestTimestampValidation:
    """End-to-end: BarBuilder rejects bad timestamps."""

    def test_millisecond_timestamp_rejected(self):
        bus = EventBus()
        bb = BarBuilder(bus)

        # Timestamp in milliseconds (a common bug)
        ms_ts = 1_700_000_000_000.0  # 1.7 trillion
        q = _quote("BTC", 100.0, 1000.0, ms_ts, source_ts=ms_ts)
        bb._on_quote(q)

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 1
        assert status["extras"]["quotes_rejected_by_symbol"]["BTC"] == 1

    def test_millisecond_source_ts_rejected(self):
        bus = EventBus()
        bb = BarBuilder(bus)

        base = 1_700_000_000.0
        # Valid timestamp but millisecond source_ts
        q = _quote("BTC", 100.0, 1000.0, base + 1, source_ts=1_700_000_001_000.0)
        bb._on_quote(q)

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 1

    def test_very_old_timestamp_rejected(self):
        bus = EventBus()
        bb = BarBuilder(bus)

        q = _quote("BTC", 100.0, 1000.0, 1_000_000_000.0, source_ts=1_000_000_000.0)
        bb._on_quote(q)

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 1

    def test_valid_timestamp_accepted(self):
        bus = EventBus()
        bb = BarBuilder(bus)

        base = 1_700_000_000.0
        minute = _minute_floor(base)
        q = _quote("BTC", 100.0, 1000.0, minute + 1, source_ts=minute + 0.5)
        bb._on_quote(q)

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 0
        assert status["extras"]["quotes_received_by_symbol"]["BTC"] == 1

    def test_zero_source_ts_still_rejected(self):
        """source_ts=0 is already caught by the existing check."""
        bus = EventBus()
        bb = BarBuilder(bus)

        base = 1_700_000_000.0
        q = _quote("BTC", 100.0, 1000.0, base + 1, source_ts=0.0)
        bb._on_quote(q)

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 1

    def test_multiple_rejections_counted_per_symbol(self):
        bus = EventBus()
        bb = BarBuilder(bus)

        ms_ts = 1_700_000_000_000.0
        for i in range(5):
            bb._on_quote(_quote("BTC", 100.0 + i, 1000.0, ms_ts + i, source_ts=ms_ts + i))
        for i in range(3):
            bb._on_quote(_quote("ETH", 200.0 + i, 500.0, ms_ts + i, source_ts=ms_ts + i))

        status = bb.get_status()
        assert status["extras"]["quotes_rejected_total"] == 8
        assert status["extras"]["quotes_rejected_by_symbol"]["BTC"] == 5
        assert status["extras"]["quotes_rejected_by_symbol"]["ETH"] == 3
