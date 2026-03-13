"""
Tests for BarBuilder volume-delta and late-tick logic.

Run with:  python -m pytest tests/test_bar_builder.py -v

These tests exercise the core bus/events/bar_builder modules only.
"""

import time

from src.core.bar_builder import BarBuilder, _minute_floor
from src.core.bus import EventBus
from src.core.events import (
    BarEvent,
    MinuteTickEvent,
    QuoteEvent,
    TOPIC_MARKET_BARS,
)


def _quote(symbol: str, price: float, volume_24h: float, ts: float) -> QuoteEvent:
    """Helper to build a QuoteEvent with explicit timestamp."""
    return QuoteEvent(
        symbol=symbol,
        bid=price - 0.01,
        ask=price + 0.01,
        mid=price,
        last=price,
        timestamp=ts,
        source="test",
        volume_24h=volume_24h,
        source_ts=ts,
        event_ts=ts,
        receive_time=ts,
    )


def _drain(bus, timeout=0.5):
    """Wait until all bus queues are empty or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        depths = bus.get_queue_depths()
        if all(d == 0 for d in depths.values()):
            return
        time.sleep(0.01)


# ═══════════════════════════════════════════════════════════
#  Bug #1 — Volume delta
# ═══════════════════════════════════════════════════════════


class TestVolumeDelta:
    """Verify that cumulative volume_24h is correctly converted to delta."""

    def test_first_tick_has_zero_volume(self):
        bus = EventBus()
        bb = BarBuilder(bus)
        delta = bb._volume_delta("SYM", 1_000_000.0)
        assert delta == 0.0, "First tick should yield zero (no prior reference)"

    def test_normal_delta(self):
        bus = EventBus()
        bb = BarBuilder(bus)
        bb._volume_delta("SYM", 1_000.0)
        delta = bb._volume_delta("SYM", 1_050.0)
        assert delta == 50.0

    def test_exchange_reset_yields_zero(self):
        bus = EventBus()
        bb = BarBuilder(bus)
        bb._volume_delta("SYM", 5_000.0)
        delta = bb._volume_delta("SYM", 100.0)  # rollover
        assert delta == 0.0, "Negative delta (reset) must be treated as zero"

    def test_bar_volume_accumulates_deltas_not_cumulative(self):
        """End-to-end: 4 ticks in one minute.

        cum_vol sequence: 1000, 1020, 1045, 1100
        deltas:              0,   20,   25,   55  → bar.volume = 100
        """
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60

            bb._on_quote(_quote("BTC", 100.0, 1000.0, minute0 + 1))
            bb._on_quote(_quote("BTC", 101.0, 1020.0, minute0 + 15))
            bb._on_quote(_quote("BTC", 99.0,  1045.0, minute0 + 30))
            bb._on_quote(_quote("BTC", 102.0, 1100.0, minute0 + 50))

            # Trigger bar close
            bb._on_quote(_quote("BTC", 103.0, 1120.0, minute1 + 1))
            _drain(bus)

            assert len(emitted) == 1
            bar = emitted[0]
            assert bar.open == 100.0
            assert bar.high == 102.0
            assert bar.low == 99.0
            assert bar.close == 102.0
            assert bar.tick_count == 4
            assert bar.volume == 100.0, (
                f"Bar volume should be sum of deltas (100), got {bar.volume}"
            )
        finally:
            bus.stop()


# ═══════════════════════════════════════════════════════════
#  Bug #2 — Late-tick bar corruption
# ═══════════════════════════════════════════════════════════


class TestLateTick:
    """Verify that ticks older than the active bar are discarded."""

    def test_late_tick_is_discarded(self):
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60

            bb._on_quote(_quote("BTC", 100.0, 500.0, minute0 + 5))
            bb._on_quote(_quote("BTC", 105.0, 520.0, minute1 + 5))
            _drain(bus)

            assert len(emitted) == 1
            closed_bar = emitted[0]

            # Late tick for minute0 — must be silently dropped
            bb._on_quote(_quote("BTC", 50.0, 530.0, minute0 + 30))
            _drain(bus)

            # Closed bar is a frozen dataclass — structurally immutable
            assert closed_bar.close == 100.0
            assert closed_bar.low == 100.0

            # Active bar for minute1 must not be contaminated
            acc = bb._bars.get("BTC")
            assert acc is not None
            assert acc.ts_open == minute1
            assert acc.low == 105.0, "Late tick must not alter the active bar"
        finally:
            bus.stop()

    def test_same_minute_tick_accepted(self):
        bus = EventBus()
        bb = BarBuilder(bus)

        base = 1_700_000_000.0
        minute0 = _minute_floor(base)

        bb._on_quote(_quote("BTC", 100.0, 500.0, minute0 + 1))
        bb._on_quote(_quote("BTC", 102.0, 510.0, minute0 + 30))

        acc = bb._bars["BTC"]
        assert acc.high == 102.0
        assert acc.tick_count == 2


class TestStatusCounters:
    def test_status_updates_on_emit_and_late_tick(self):
        bus = EventBus()
        bb = BarBuilder(bus)

        base = 1_700_000_000.0
        minute0 = _minute_floor(base)
        minute1 = minute0 + 60

        bb._on_quote(_quote("BTC", 100.0, 500.0, minute0 + 1))
        bb._on_quote(_quote("BTC", 101.0, 510.0, minute1 + 1))

        # Late tick for the prior minute
        bb._on_quote(_quote("BTC", 99.0, 520.0, minute0 + 30))

        status = bb.get_status()
        extras = status["extras"]

        assert extras["bars_emitted_total"] == 1
        assert extras["bars_emitted_by_symbol"]["BTC"] == 1
        assert extras["late_ticks_dropped_total"] == 1
        assert extras["quotes_received_by_symbol"]["BTC"] == 3
        assert "BTC" in extras["last_bar_ts_by_symbol"]


class TestSourceTimestampPolicy:
    def test_rejects_missing_source_ts(self):
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)

            bad_quote = QuoteEvent(
                symbol="BTC",
                bid=99.0,
                ask=101.0,
                mid=100.0,
                last=100.0,
                timestamp=minute0 + 1,
                source="test",
                volume_24h=1000.0,
                source_ts=0.0,
                event_ts=minute0 + 1,
                receive_time=minute0 + 1,
            )
            bb._on_quote(bad_quote)
            _drain(bus)

            assert not emitted
            status = bb.get_status()
            extras = status["extras"]
            assert extras["quotes_rejected_total"] == 1
            assert extras["quotes_rejected_by_symbol"]["BTC"] == 1
        finally:
            bus.stop()


class TestInvalidPriceQuotes:
    def test_invalid_bid_ask_is_rejected(self):
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)

            bad_quote = QuoteEvent(
                symbol="BTC",
                bid=0.0,
                ask=0.0,
                mid=0.0,
                last=0.0,
                timestamp=minute0 + 1,
                source="test",
                volume_24h=1000.0,
                source_ts=minute0 + 1,
                event_ts=minute0 + 1,
                receive_time=minute0 + 1,
            )
            bb._on_quote(bad_quote)
            _drain(bus)

            assert not emitted
            assert "BTC" not in bb._bars
            status = bb.get_status()
            extras = status["extras"]
            assert extras["quotes_rejected_invalid_price_total"] == 1
            assert extras["quotes_rejected_invalid_price_by_symbol"]["BTC"] == 1
        finally:
            bus.stop()


class TestLateTickReset:
    def test_late_ticks_reset_between_bars(self):
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60
            minute2 = minute0 + 120
            minute3 = minute0 + 180

            bb._on_quote(_quote("BTC", 100.0, 1000.0, minute0 + 1))
            bb._on_quote(_quote("BTC", 101.0, 1010.0, minute1 + 1))
            _drain(bus)

            bb._on_quote(_quote("BTC", 50.0, 1020.0, minute0 + 30))
            bb._on_quote(_quote("BTC", 102.0, 1030.0, minute2 + 1))
            _drain(bus)

            bb._on_quote(_quote("BTC", 40.0, 1040.0, minute1 + 30))
            bb._on_quote(_quote("BTC", 103.0, 1050.0, minute3 + 1))
            _drain(bus)

            bar1 = next(b for b in emitted if b.timestamp == minute1)
            bar2 = next(b for b in emitted if b.timestamp == minute2)

            assert bar1.late_ticks_dropped == 1
            assert bar2.late_ticks_dropped == 1
        finally:
            bus.stop()


class TestContinuityWithoutSyntheticBars:
    def test_missing_minutes_do_not_emit_synthetic_bars(self):
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60
            minute2 = minute0 + 120
            minute3 = minute0 + 180

            bb._on_quote(_quote("BTC", 100.0, 100.0, minute0 + 5))
            bb._on_minute_tick(MinuteTickEvent(timestamp=minute1))

            bb._on_quote(_quote("BTC", 110.0, 150.0, minute2 + 5))
            bb._on_minute_tick(MinuteTickEvent(timestamp=minute3))
            _drain(bus)

            assert len(emitted) == 2
            timestamps = [bar.timestamp for bar in emitted]
            assert minute0 in timestamps
            assert minute2 in timestamps
            assert minute1 not in timestamps
        finally:
            bus.stop()

    def test_last_bar_ts_updates_on_minute_tick(self):
        bus = EventBus()
        bb = BarBuilder(bus)

        base = 1_700_000_000.0
        minute0 = _minute_floor(base)
        minute1 = minute0 + 60

        bb._on_quote(_quote("ETH", 200.0, 100.0, minute0 + 5))
        bb._on_minute_tick(MinuteTickEvent(timestamp=minute1))

        status = bb.get_status()
        extras = status["extras"]
        assert extras["last_bar_ts_by_symbol_epoch"]["ETH"] == minute0


# ═══════════════════════════════════════════════════════════
#  Minute-floor utility
# ═══════════════════════════════════════════════════════════


class TestMinuteFloor:
    def test_exact_boundary(self):
        assert _minute_floor(1_700_000_000.0) == 1_700_000_000.0 - (1_700_000_000 % 60)

    def test_mid_minute(self):
        ts = 1_700_000_030.5
        floored = _minute_floor(ts)
        assert floored % 60 == 0
        assert floored <= ts
        assert floored + 60 > ts
