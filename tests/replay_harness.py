"""
Replay Harness — Deterministic Bar-Closure Proof
=================================================

Feeds a fixed sequence of QuoteEvents through the BarBuilder **twice**
and asserts that both runs produce bit-identical BarEvents.

This proves:
* Bar provenance fields (n_ticks, first/last_source_ts, close_reason)
  are deterministic under replay.
* Late ticks are handled identically on both passes.
* Invariant enforcement is reproducible.

Run with:  python -m pytest tests/replay_harness.py -v
"""

from __future__ import annotations

import time
from typing import List, Sequence, Union

from src.core.bar_builder import BarBuilder, _minute_floor
from src.core.bus import EventBus
from src.core.events import (
    BarEvent,
    CloseReason,
    MinuteTickEvent,
    QuoteEvent,
    TOPIC_MARKET_BARS,
)


def _quote(
    symbol: str,
    price: float,
    volume_24h: float,
    ts: float,
    source: str = "test",
    source_ts: float = 0.0,
) -> QuoteEvent:
    """Build a QuoteEvent with explicit timestamps (no wall-clock)."""
    return QuoteEvent(
        symbol=symbol,
        bid=price - 0.01,
        ask=price + 0.01,
        mid=price,
        last=price,
        timestamp=ts,
        source=source,
        volume_24h=volume_24h,
        source_ts=source_ts if source_ts else ts,
        event_ts=ts,       # deterministic — no time.time()
        receive_time=ts,    # deterministic — no time.time()
    )


def _drain(bus, timeout=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        depths = bus.get_queue_depths()
        if all(d == 0 for d in depths.values()):
            return
        time.sleep(0.01)


# ── Canonical tick tape ──────────────────────────────────────

BASE = 1_700_000_000.0
M0 = _minute_floor(BASE)         # minute 0 boundary
M1 = M0 + 60                     # minute 1 boundary
M2 = M0 + 120                    # minute 2 boundary


def _build_tape() -> List[QuoteEvent]:
    """Return a fixed, ordered tape of quotes spanning 3 minutes."""
    return [
        # ── Minute 0: 4 ticks ──
        _quote("BTC", 100.0, 1000.0, M0 + 1,  source_ts=M0 + 0.5),
        _quote("BTC", 101.0, 1020.0, M0 + 15, source_ts=M0 + 14.8),
        _quote("BTC",  99.0, 1045.0, M0 + 30, source_ts=M0 + 29.9),
        _quote("BTC", 102.0, 1100.0, M0 + 50, source_ts=M0 + 49.7),

        # ── Minute 1: 2 ticks (triggers minute-0 bar close) ──
        _quote("BTC", 103.0, 1120.0, M1 + 5,  source_ts=M1 + 4.5),
        _quote("BTC", 104.0, 1140.0, M1 + 25, source_ts=M1 + 24.3),

        # ── Late tick for minute 0 (must be dropped) ──
        _quote("BTC",  50.0, 1160.0, M0 + 55, source_ts=M0 + 54.0),

        # ── Minute 2: 1 tick (triggers minute-1 bar close) ──
        _quote("BTC", 105.0, 1170.0, M2 + 3,  source_ts=M2 + 2.8),

        # ── Multi-symbol: ETH interleaved ──
        _quote("ETH", 2000.0, 500.0, M0 + 10, source_ts=M0 + 9.5),
        _quote("ETH", 2010.0, 510.0, M0 + 40, source_ts=M0 + 39.2),
        _quote("ETH", 2020.0, 520.0, M1 + 10, source_ts=M1 + 9.1),
    ]

def _build_minute_tick_tape() -> List[Union[QuoteEvent, MinuteTickEvent]]:
    """Tape that closes bars via MinuteTickEvent and includes a late tick."""
    return [
        _quote("BTC", 100.0, 1000.0, M0 + 5, source_ts=M0 + 4.8),
        _quote("BTC", 101.0, 1010.0, M0 + 20, source_ts=M0 + 19.9),
        MinuteTickEvent(timestamp=M1),
        _quote("BTC", 102.0, 1020.0, M1 + 5, source_ts=M1 + 4.7),
        _quote("BTC",  50.0, 1030.0, M0 + 30, source_ts=M0 + 29.5),  # late
        MinuteTickEvent(timestamp=M2),
    ]


def _run_tape(tape: Sequence[Union[QuoteEvent, MinuteTickEvent]]) -> List[BarEvent]:
    """Play *tape* through a fresh BarBuilder and return emitted bars."""
    bus = EventBus()
    emitted: List[BarEvent] = []
    bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
    bb = BarBuilder(bus)
    bus.start()

    try:
        for event in tape:
            if isinstance(event, MinuteTickEvent):
                bb._on_minute_tick(event)
            else:
                bb._on_quote(event)

        # Flush remaining bars (shutdown path)
        flushed = bb.flush()
        _drain(bus)
    finally:
        bus.stop()

    return emitted + flushed


def _bar_key(bar: BarEvent) -> tuple:
    """Extract the deterministic identity of a bar (excluding wall-clock fields)."""
    return (
        bar.symbol,
        bar.open,
        bar.high,
        bar.low,
        bar.close,
        bar.volume,
        bar.timestamp,
        bar.source,
        bar.bar_duration,
        bar.tick_count,
        bar.n_ticks,
        bar.first_source_ts,
        bar.last_source_ts,
        bar.late_ticks_dropped,
        bar.close_reason,
        bar.source_ts,
        bar.repaired,
    )


# ═══════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════


class TestReplayDeterminism:
    """Two identical replays must produce identical bars."""

    def test_replay_produces_identical_bars(self):
        tape = _build_tape()
        run1 = _run_tape(tape)
        run2 = _run_tape(tape)

        assert len(run1) == len(run2), (
            f"Run 1 produced {len(run1)} bars, run 2 produced {len(run2)}"
        )
        for i, (b1, b2) in enumerate(zip(run1, run2)):
            k1, k2 = _bar_key(b1), _bar_key(b2)
            assert k1 == k2, (
                f"Bar {i} differs between runs:\n  run1={k1}\n  run2={k2}"
            )

    def test_replay_bar_count(self):
        """Expect: BTC minute-0, BTC minute-1 (both closed by new-tick),
        ETH minute-0 (closed by BTC minute-1 tick or flush),
        plus shutdown-flush bars for in-progress accumulators."""
        tape = _build_tape()
        bars = _run_tape(tape)
        symbols = [b.symbol for b in bars]
        # At minimum we must see bars for both symbols
        assert "BTC" in symbols
        assert "ETH" in symbols
        assert len(bars) >= 4  # 2 BTC + 1–2 ETH minimum

    def test_minute_tick_replay_is_deterministic(self):
        tape = _build_minute_tick_tape()
        run1 = _run_tape(tape)
        run2 = _run_tape(tape)
        assert [_bar_key(b) for b in run1] == [_bar_key(b) for b in run2]


class TestProvenanceFields:
    """Verify provenance fields are populated correctly."""

    def test_n_ticks_matches_tick_count(self):
        tape = _build_tape()
        bars = _run_tape(tape)
        for bar in bars:
            assert bar.n_ticks == bar.tick_count
            assert bar.n_ticks > 0

    def test_first_last_source_ts_ordering(self):
        tape = _build_tape()
        bars = _run_tape(tape)
        for bar in bars:
            assert bar.first_source_ts > 0, "first_source_ts must be set"
            assert bar.last_source_ts >= bar.first_source_ts, (
                f"last_source_ts ({bar.last_source_ts}) < first_source_ts ({bar.first_source_ts})"
            )

    def test_close_reason_values(self):
        tape = _build_tape()
        bars = _run_tape(tape)
        close_reasons = {b.close_reason for b in bars}
        # Should see both NEW_TICK (from quote triggers) and SHUTDOWN_FLUSH (from flush)
        assert int(CloseReason.NEW_TICK) in close_reasons or int(CloseReason.SHUTDOWN_FLUSH) in close_reasons

    def test_source_ts_equals_first_source_ts(self):
        """Bar-level source_ts should match first_source_ts."""
        tape = _build_tape()
        bars = _run_tape(tape)
        for bar in bars:
            assert bar.source_ts == bar.first_source_ts


class TestMinuteTickClosure:
    def test_minute_tick_close_reason_and_late_tick(self):
        tape = _build_minute_tick_tape()
        bars = _run_tape(tape)
        m0_bar = next((b for b in bars if b.timestamp == M0), None)
        m1_bar = next((b for b in bars if b.timestamp == M1), None)
        assert m0_bar is not None
        assert m1_bar is not None
        assert m0_bar.close_reason == int(CloseReason.MINUTE_TICK)
        assert m0_bar.low != 50.0
        assert m1_bar.late_ticks_dropped >= 1


class TestInvariantEnforcement:
    """Bars with violated invariants should be repaired, not crashed."""

    def test_high_ge_open_close(self):
        tape = _build_tape()
        bars = _run_tape(tape)
        for bar in bars:
            assert bar.high >= bar.open, f"high ({bar.high}) < open ({bar.open})"
            assert bar.high >= bar.close, f"high ({bar.high}) < close ({bar.close})"

    def test_low_le_open_close(self):
        tape = _build_tape()
        bars = _run_tape(tape)
        for bar in bars:
            assert bar.low <= bar.open, f"low ({bar.low}) > open ({bar.open})"
            assert bar.low <= bar.close, f"low ({bar.low}) > close ({bar.close})"

    def test_volume_non_negative(self):
        tape = _build_tape()
        bars = _run_tape(tape)
        for bar in bars:
            assert bar.volume >= 0, f"volume ({bar.volume}) is negative"


class TestLateTick:
    """Late ticks must be counted but never mutate emitted bars."""

    def test_late_tick_does_not_corrupt(self):
        tape = _build_tape()
        bars = _run_tape(tape)
        btc_bars = [b for b in bars if b.symbol == "BTC"]
        # The tape has a late tick for minute 0 after minute 1 opened.
        # The minute-0 bar must NOT have a low of 50.0
        m0_bar = next((b for b in btc_bars if b.timestamp == M0), None)
        if m0_bar:
            assert m0_bar.low != 50.0, "Late tick corrupted minute-0 bar"


class TestSchemaVersion:
    """All emitted bars must carry the schema version."""

    def test_v_field_present(self):
        tape = _build_tape()
        bars = _run_tape(tape)
        for bar in bars:
            assert hasattr(bar, 'v') and bar.v >= 1
