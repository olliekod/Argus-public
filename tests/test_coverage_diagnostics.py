"""
Tests for Argus coverage diagnostics module (Phase 4A.1+).

Tests the pure-function coverage helpers:
- Uptime computation from heartbeat series
- Bar continuity / gap analysis
- Uptime-aware diagnose
- Equity session heuristic
- DB round-trip for heartbeats + bar timestamps
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.coverage import (
    UptimeResult,
    compute_uptime,
    analyze_bar_continuity,
    diagnose_coverage,
    is_likely_equity,
    GapInfo,
)
from src.core.database import Database


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

BASE_MS = 1_700_000_000_000  # ~2023-11-14 UTC


def _ms(offset_s: int) -> int:
    """Return BASE_MS + offset in seconds converted to ms."""
    return BASE_MS + offset_s * 1000


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _make_db() -> Database:
    """Create a fresh in-memory database."""
    db = Database(":memory:")
    await db.connect()
    return db


# ═══════════════════════════════════════════════════════════════════════════════
#  A) Uptime computation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestUptimeComputation:
    """Test the compute_uptime pure function."""

    def test_continuous_heartbeats_full_uptime(self):
        """60s heartbeats over 600s => ~100% uptime."""
        hbs = [_ms(i * 60) for i in range(11)]  # 0, 60, 120, ..., 600
        result = compute_uptime(hbs, _ms(0), _ms(600), gap_threshold_ms=120_000)
        assert result.heartbeat_count == 11
        assert result.uptime_s == 600
        assert result.downtime_s == 0
        assert result.off_intervals == []

    def test_single_large_gap(self):
        """Heartbeats with a 5-minute gap in the middle."""
        # 0, 60, 120, then gap, then 420, 480, 540, 600
        hbs = [_ms(0), _ms(60), _ms(120),
               _ms(420), _ms(480), _ms(540), _ms(600)]
        result = compute_uptime(hbs, _ms(0), _ms(600), gap_threshold_ms=120_000)

        assert result.heartbeat_count == 7
        assert len(result.off_intervals) == 1

        # OFF interval: 120+120=240 to 420
        off_start, off_end = result.off_intervals[0]
        assert off_start == _ms(240)  # 120s + 120s threshold
        assert off_end == _ms(420)

        # Uptime: 120 (first segment) + 120 (threshold) + 180 (second segment) = 420
        assert result.uptime_s == 420
        assert result.downtime_s == 180  # 600 - 420

    def test_empty_heartbeats(self):
        """No heartbeats => 100% downtime."""
        result = compute_uptime([], _ms(0), _ms(3600), gap_threshold_ms=120_000)
        assert result.uptime_s == 0
        assert result.downtime_s == 3600
        assert result.heartbeat_count == 0

    def test_single_heartbeat(self):
        """One heartbeat => can't determine uptime."""
        result = compute_uptime([_ms(100)], _ms(0), _ms(3600),
                                gap_threshold_ms=120_000)
        assert result.uptime_s == 0
        assert result.downtime_s == 3600

    def test_two_heartbeats_within_threshold(self):
        """Two heartbeats close together => uptime = delta."""
        result = compute_uptime([_ms(0), _ms(60)], _ms(0), _ms(60),
                                gap_threshold_ms=120_000)
        assert result.uptime_s == 60
        assert result.downtime_s == 0

    def test_two_heartbeats_beyond_threshold(self):
        """Two heartbeats far apart => partial uptime + OFF interval."""
        result = compute_uptime([_ms(0), _ms(500)], _ms(0), _ms(500),
                                gap_threshold_ms=120_000)
        assert result.uptime_s == 120  # threshold
        assert result.downtime_s == 380  # 500 - 120
        assert len(result.off_intervals) == 1

    def test_multiple_off_intervals(self):
        """Scattered heartbeats => multiple OFF intervals."""
        # ON: 0-60, OFF: 60+120..360, ON: 360-420, OFF: 420+120..720, ON: 720-780
        hbs = [_ms(0), _ms(60), _ms(360), _ms(420), _ms(720), _ms(780)]
        result = compute_uptime(hbs, _ms(0), _ms(780),
                                gap_threshold_ms=120_000)
        assert len(result.off_intervals) == 2

    def test_threshold_boundary(self):
        """Gap exactly at threshold => no OFF interval."""
        hbs = [_ms(0), _ms(120)]  # exactly 120s gap = threshold
        result = compute_uptime(hbs, _ms(0), _ms(120),
                                gap_threshold_ms=120_000)
        assert result.uptime_s == 120
        assert len(result.off_intervals) == 0

    def test_threshold_boundary_plus_one(self):
        """Gap 1ms over threshold => OFF interval."""
        # 121 seconds gap
        hbs = [BASE_MS, BASE_MS + 121_000]
        result = compute_uptime(hbs, BASE_MS, BASE_MS + 121_000,
                                gap_threshold_ms=120_000)
        assert len(result.off_intervals) == 1
        assert result.uptime_s == 120  # threshold portion


# ═══════════════════════════════════════════════════════════════════════════════
#  B) Bar continuity / gap analysis
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarContinuity:
    """Test analyze_bar_continuity."""

    def test_perfect_1m_series(self):
        """No gaps in a perfect 1-minute bar series."""
        bars = [_ms(i * 60) for i in range(100)]
        result = analyze_bar_continuity(bars, 60, gap_threshold_seconds=300)
        assert result.bar_count == 100
        assert result.expected_bars == 100  # (99*60 / 60) + 1
        assert result.coverage_pct == 100.0
        assert result.max_gap_seconds == 0
        assert result.gap_count_above_threshold == 0
        assert result.top_gaps == []

    def test_missing_bars_coverage(self):
        """50 bars over a span that should have 100 => ~50% coverage."""
        # Create bars at 0, 120, 240, ... (every 2 minutes)
        bars = [_ms(i * 120) for i in range(50)]
        result = analyze_bar_continuity(bars, 60, gap_threshold_seconds=300)
        assert result.bar_count == 50
        span_s = (49 * 120)
        assert result.expected_bars == span_s // 60 + 1  # 99
        # Coverage ~ 50/99 ≈ 50.5%
        assert 50 < result.coverage_pct < 51

    def test_single_large_gap(self):
        """One 30-minute gap in otherwise perfect data."""
        bars = [_ms(i * 60) for i in range(30)]
        # Gap from minute 29 to minute 59
        bars += [_ms((59 + i) * 60) for i in range(30)]
        result = analyze_bar_continuity(bars, 60, gap_threshold_seconds=300)
        assert result.max_gap_seconds == 30 * 60
        assert result.gap_count_above_threshold == 1
        assert len(result.top_gaps) == 1
        assert result.top_gaps[0].gap_seconds == 1800

    def test_empty_series(self):
        result = analyze_bar_continuity([], 60)
        assert result.bar_count == 0
        assert result.coverage_pct == 0.0

    def test_single_bar(self):
        result = analyze_bar_continuity([_ms(0)], 60)
        assert result.bar_count == 1
        assert result.coverage_pct == 100.0

    def test_top_n_limiting(self):
        """Verify top_n limits the number of gaps returned."""
        # Create 20 gaps of various sizes
        bars = [_ms(0)]
        for i in range(1, 21):
            bars.append(_ms(i * 600))  # 10-minute intervals = 9-min gaps
        bars.sort()
        result = analyze_bar_continuity(bars, 60, top_n=5)
        assert len(result.top_gaps) == 5

    def test_gap_threshold_filtering(self):
        """Gaps below threshold aren't counted in gap_count_above_threshold."""
        # 2-minute gaps (120s) with threshold 300s
        bars = [_ms(i * 120) for i in range(10)]
        result = analyze_bar_continuity(bars, 60, gap_threshold_seconds=300)
        # Each 120s gap is > 60s bar_dur, so they appear as gaps
        # But none are >= 300s
        assert result.gap_count_above_threshold == 0
        assert result.max_gap_seconds == 120


# ═══════════════════════════════════════════════════════════════════════════════
#  C) Diagnose (uptime-aware coverage)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiagnose:
    """Test diagnose_coverage combining uptime + bar data."""

    def test_full_uptime_full_coverage(self):
        """100% uptime + all bars present."""
        start, end = _ms(0), _ms(600)
        hbs = [_ms(i * 60) for i in range(11)]
        bars = [_ms(i * 60) for i in range(11)]
        uptime = compute_uptime(hbs, start, end, gap_threshold_ms=120_000)
        result = diagnose_coverage(bars, 60, uptime, start, end)

        assert result.uptime_s == 600
        assert result.bars_during_uptime == 11
        assert result.expected_bars_during_uptime == 10  # 600/60
        assert result.uptime_coverage_pct >= 100.0
        assert result.missing_during_uptime == 0

    def test_downtime_subtracts_missing(self):
        """Bars missing during downtime should not count as missing-during-uptime."""
        start, end = _ms(0), _ms(600)
        # Heartbeats: ON for 0-120, then OFF 120..420, then ON 420-600
        hbs = [_ms(0), _ms(60), _ms(120), _ms(420), _ms(480), _ms(540), _ms(600)]
        # Bars present only during uptime periods (0-120 and 420-600)
        bars = ([_ms(i * 60) for i in range(3)] +    # 0,60,120
                [_ms(420 + i * 60) for i in range(4)])  # 420,480,540,600

        uptime = compute_uptime(hbs, start, end, gap_threshold_ms=120_000)
        result = diagnose_coverage(bars, 60, uptime, start, end)

        # All 7 bars are during uptime
        assert result.bars_during_uptime == 7
        # No bars during OFF time
        assert result.bars_total == 7

    def test_equity_flag(self):
        """Equity diagnosis includes session note."""
        start, end = _ms(0), _ms(600)
        hbs = [_ms(i * 60) for i in range(11)]
        bars = [_ms(i * 60) for i in range(11)]
        uptime = compute_uptime(hbs, start, end, gap_threshold_ms=120_000)
        result = diagnose_coverage(bars, 60, uptime, start, end, is_equity=True)

        assert result.is_equity is True
        assert "session" in result.equity_session_note.lower()

    def test_no_heartbeats_all_downtime(self):
        """No heartbeats => all bars counted as 'during OFF'."""
        start, end = _ms(0), _ms(600)
        bars = [_ms(i * 60) for i in range(11)]
        uptime = compute_uptime([], start, end, gap_threshold_ms=120_000)
        result = diagnose_coverage(bars, 60, uptime, start, end)

        assert result.uptime_s == 0
        assert result.downtime_s == 600
        assert result.expected_bars_during_uptime == 0

    def test_gaps_during_uptime_detected(self):
        """Gaps that occur during uptime are flagged."""
        start, end = _ms(0), _ms(1200)
        hbs = [_ms(i * 60) for i in range(21)]  # continuous heartbeats
        # Bars with a 600s gap in the middle (minute 5 to minute 15)
        bars = ([_ms(i * 60) for i in range(6)] +   # 0..300
                [_ms(900 + i * 60) for i in range(6)])  # 900..1200
        uptime = compute_uptime(hbs, start, end, gap_threshold_ms=120_000)
        result = diagnose_coverage(bars, 60, uptime, start, end)

        assert len(result.top_uptime_gaps) >= 1
        assert result.top_uptime_gaps[0].gap_seconds == 600


# ═══════════════════════════════════════════════════════════════════════════════
#  D) Equity heuristic
# ═══════════════════════════════════════════════════════════════════════════════


class TestEquityHeuristic:
    def test_yahoo_is_equity(self):
        assert is_likely_equity("yahoo", "IBIT") is True

    def test_alpaca_is_equity(self):
        assert is_likely_equity("alpaca", "AAPL") is True

    def test_bybit_is_not_equity(self):
        assert is_likely_equity("bybit", "BTCUSDT") is False

    def test_known_symbol_any_provider(self):
        assert is_likely_equity("unknown_provider", "SPY") is True

    def test_unknown_everything(self):
        assert is_likely_equity("custom", "CUSTOMTOKEN") is False


# ═══════════════════════════════════════════════════════════════════════════════
#  E) DB Integration: heartbeat + bar_timestamps round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestDBHeartbeat:
    """Integration tests: write/read heartbeats via Database."""

    @pytest.mark.asyncio
    async def test_write_and_read_heartbeats(self):
        db = await _make_db()
        try:
            for i in range(10):
                await db.write_heartbeat("orchestrator", _ms(i * 60))

            hbs = await db.get_heartbeats("orchestrator", _ms(0), _ms(600))
            assert len(hbs) == 10
            assert hbs[0] == _ms(0)
            assert hbs[-1] == _ms(9 * 60)
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_heartbeat_idempotent(self):
        db = await _make_db()
        try:
            await db.write_heartbeat("orchestrator", _ms(0))
            await db.write_heartbeat("orchestrator", _ms(0))  # duplicate
            hbs = await db.get_heartbeats("orchestrator", _ms(0), _ms(1))
            assert len(hbs) == 1
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_heartbeat_range_filter(self):
        db = await _make_db()
        try:
            for i in range(20):
                await db.write_heartbeat("orchestrator", _ms(i * 60))

            hbs = await db.get_heartbeats("orchestrator", _ms(300), _ms(600))
            assert all(_ms(300) <= h <= _ms(600) for h in hbs)
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_heartbeat_component_filter(self):
        db = await _make_db()
        try:
            await db.write_heartbeat("orchestrator", _ms(0))
            await db.write_heartbeat("collector", _ms(0))
            await db.write_heartbeat("orchestrator", _ms(60))

            orch_hbs = await db.get_heartbeats("orchestrator", _ms(0), _ms(100))
            coll_hbs = await db.get_heartbeats("collector", _ms(0), _ms(100))
            assert len(orch_hbs) == 2
            assert len(coll_hbs) == 1
        finally:
            await db.close()


class TestDBBarTimestamps:
    """Integration tests: get_bar_timestamps via Database."""

    @pytest.mark.asyncio
    async def test_get_bar_timestamps(self):
        db = await _make_db()
        try:
            for i in range(10):
                ts = datetime.fromtimestamp(
                    (_ms(i * 60)) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                await db._connection.execute(
                    "INSERT INTO market_bars "
                    "(timestamp, source, symbol, bar_duration, "
                    "open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, "test", "TESTSYM", 60,
                     100.0, 101.0, 99.0, 100.5, 1000.0),
                )
            await db._connection.commit()

            timestamps = await db.get_bar_timestamps(
                "test", "TESTSYM", 60, _ms(0), _ms(600))
            assert len(timestamps) == 10
            assert timestamps[0] == _ms(0)
            assert timestamps[-1] == _ms(9 * 60)
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_bar_timestamps_range_filter(self):
        db = await _make_db()
        try:
            for i in range(20):
                ts = datetime.fromtimestamp(
                    (_ms(i * 60)) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                await db._connection.execute(
                    "INSERT INTO market_bars "
                    "(timestamp, source, symbol, bar_duration, "
                    "open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, "test", "TESTSYM", 60,
                     100.0, 101.0, 99.0, 100.5, 1000.0),
                )
            await db._connection.commit()

            timestamps = await db.get_bar_timestamps(
                "test", "TESTSYM", 60, _ms(300), _ms(600))
            for t in timestamps:
                assert _ms(300) <= t <= _ms(600)
        finally:
            await db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  F) End-to-end: heartbeat -> uptime -> diagnose pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEndPipeline:
    """Test the full pipeline from DB data through pure functions."""

    @pytest.mark.asyncio
    async def test_e2e_diagnose_with_gap(self):
        """Simulate Argus running for 20 minutes, OFF for 10, ON for 20 more.
        Verify diagnose correctly attributes missing bars."""
        db = await _make_db()
        try:
            # Write heartbeats: ON 0..1200, OFF 1200..1800, ON 1800..3000
            for i in range(21):  # 0,60,...,1200
                await db.write_heartbeat("orchestrator", _ms(i * 60))
            for i in range(20):  # 1800,1860,...,2940
                await db.write_heartbeat("orchestrator", _ms(1800 + i * 60))

            # Write bars: present during uptime, missing during downtime
            for i in range(21):  # 0..1200
                ts = datetime.fromtimestamp(
                    _ms(i * 60) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                await db._connection.execute(
                    "INSERT INTO market_bars "
                    "(timestamp, source, symbol, bar_duration, "
                    "open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, "test", "SYM", 60, 100, 101, 99, 100.5, 500),
                )
            for i in range(20):  # 1800..2940
                ts = datetime.fromtimestamp(
                    _ms(1800 + i * 60) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                await db._connection.execute(
                    "INSERT INTO market_bars "
                    "(timestamp, source, symbol, bar_duration, "
                    "open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, "test", "SYM", 60, 100, 101, 99, 100.5, 500),
                )
            await db._connection.commit()

            start_ms = _ms(0)
            end_ms = _ms(3000)

            heartbeats = await db.get_heartbeats("orchestrator", start_ms, end_ms)
            bar_ts = await db.get_bar_timestamps("test", "SYM", 60, start_ms, end_ms)

            uptime = compute_uptime(heartbeats, start_ms, end_ms,
                                    gap_threshold_ms=120_000)
            result = diagnose_coverage(bar_ts, 60, uptime, start_ms, end_ms)

            # Bars during OFF time should NOT be counted as missing-during-uptime
            assert result.bars_total == 41
            assert result.uptime_s > 0
            assert result.downtime_s > 0
            # All bars are during uptime (none during the OFF gap)
            assert result.bars_during_uptime == 41
            # Coverage during uptime should be high
            assert result.uptime_coverage_pct > 80.0
        finally:
            await db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  G) Determinism: same inputs -> same outputs
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Pure functions must produce identical results on identical inputs."""

    def test_uptime_determinism(self):
        hbs = [_ms(i * 60) for i in range(100)]
        r1 = compute_uptime(hbs, _ms(0), _ms(6000), 120_000)
        r2 = compute_uptime(hbs, _ms(0), _ms(6000), 120_000)
        assert r1.uptime_s == r2.uptime_s
        assert r1.downtime_s == r2.downtime_s
        assert r1.off_intervals == r2.off_intervals

    def test_gap_analysis_determinism(self):
        bars = [_ms(i * 60) for i in range(50)]
        bars += [_ms(3600 + i * 60) for i in range(50)]
        r1 = analyze_bar_continuity(bars, 60, 300, 10)
        r2 = analyze_bar_continuity(bars, 60, 300, 10)
        assert r1.coverage_pct == r2.coverage_pct
        assert r1.max_gap_seconds == r2.max_gap_seconds
        assert len(r1.top_gaps) == len(r2.top_gaps)
        for g1, g2 in zip(r1.top_gaps, r2.top_gaps):
            assert g1.start_ms == g2.start_ms
            assert g1.gap_seconds == g2.gap_seconds

    def test_diagnose_determinism(self):
        start, end = _ms(0), _ms(1200)
        hbs = [_ms(i * 60) for i in range(21)]
        bars = [_ms(i * 60) for i in range(21)]
        up = compute_uptime(hbs, start, end, 120_000)
        r1 = diagnose_coverage(bars, 60, up, start, end)
        r2 = diagnose_coverage(bars, 60, up, start, end)
        assert r1.uptime_coverage_pct == r2.uptime_coverage_pct
        assert r1.missing_during_uptime == r2.missing_during_uptime


# ═══════════════════════════════════════════════════════════════════════════════
#  H) Median cadence in UptimeResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestMedianCadence:
    """Test that UptimeResult includes correct median_cadence_ms."""

    def test_uniform_60s_cadence(self):
        hbs = [_ms(i * 60) for i in range(11)]
        result = compute_uptime(hbs, _ms(0), _ms(600), 120_000)
        assert result.median_cadence_ms == 60_000

    def test_cadence_ignores_off_gaps(self):
        """Median cadence uses only normal (within-threshold) deltas."""
        # 60s cadence with one 500s gap in the middle
        hbs = [_ms(0), _ms(60), _ms(120), _ms(620), _ms(680), _ms(740)]
        result = compute_uptime(hbs, _ms(0), _ms(740), 120_000)
        # Normal deltas: 60, 60, 60, 60 (the 500s gap is excluded)
        assert result.median_cadence_ms == 60_000

    def test_no_heartbeats_cadence_zero(self):
        result = compute_uptime([], _ms(0), _ms(600), 120_000)
        assert result.median_cadence_ms == 0

    def test_one_heartbeat_cadence_zero(self):
        result = compute_uptime([_ms(0)], _ms(0), _ms(600), 120_000)
        assert result.median_cadence_ms == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  I) Missing bars breakdown in DiagnoseResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestMissingBarsBreakdown:
    """Test bars_during_downtime and missing_during_downtime fields."""

    def test_bars_during_downtime_counted(self):
        """Bars that fall in OFF intervals are counted as during-downtime."""
        start, end = _ms(0), _ms(600)
        # Heartbeats: ON 0-120, OFF 240-420, ON 420-600
        hbs = [_ms(0), _ms(60), _ms(120), _ms(420), _ms(480), _ms(540), _ms(600)]
        # Bars every minute including during OFF time (240-420)
        bars = [_ms(i * 60) for i in range(11)]  # 0..600

        uptime = compute_uptime(hbs, start, end, gap_threshold_ms=120_000)
        result = diagnose_coverage(bars, 60, uptime, start, end)

        # Some bars should be in downtime
        assert result.bars_during_downtime > 0
        assert result.bars_during_uptime + result.bars_during_downtime == result.bars_total

    def test_all_bars_during_uptime_no_downtime_missing(self):
        """When all bars fall within uptime, bars_during_downtime == 0."""
        start, end = _ms(0), _ms(600)
        hbs = [_ms(i * 60) for i in range(11)]  # continuous uptime
        bars = [_ms(i * 60) for i in range(11)]

        uptime = compute_uptime(hbs, start, end, gap_threshold_ms=120_000)
        result = diagnose_coverage(bars, 60, uptime, start, end)

        assert result.bars_during_downtime == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  J) Regression: bars missing only during downtime => 100% uptime coverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegressionDowntimeOnly:
    """Key regression test: if bars are only missing during downtime,
    uptime coverage should be 100% and missing_during_uptime should be 0."""

    def test_perfect_uptime_coverage_with_downtime_gap(self):
        """Simulate:
        - ON for 30 minutes (0..1800), OFF for 30 minutes (1800..3600),
          ON for 30 minutes (3600..5400).
        - Bars present every 60s during uptime, zero bars during downtime.
        - Expected: coverage_uptime_pct ~= 100%, missing_during_uptime == 0.
        """
        start = _ms(0)
        end = _ms(5400)

        # Heartbeats: every 60s during ON periods
        hbs = (
            [_ms(i * 60) for i in range(31)] +        # 0..1800
            [_ms(3600 + i * 60) for i in range(31)]    # 3600..5400
        )

        # Bars: every 60s during ON periods only
        bars = (
            [_ms(i * 60) for i in range(31)] +        # 0..1800
            [_ms(3600 + i * 60) for i in range(31)]    # 3600..5400
        )

        uptime = compute_uptime(hbs, start, end, gap_threshold_ms=120_000)
        result = diagnose_coverage(bars, 60, uptime, start, end)

        # Core assertions: this is the key regression guard
        assert result.missing_during_uptime == 0, (
            f"Expected 0 missing during uptime, got {result.missing_during_uptime}"
        )
        assert result.uptime_coverage_pct >= 100.0, (
            f"Expected >= 100% uptime coverage, got {result.uptime_coverage_pct}%"
        )

        # Supplementary assertions
        assert result.bars_during_downtime == 0
        assert result.missing_during_downtime > 0  # expected: bars missing during OFF
        assert result.uptime_s > 0
        assert result.downtime_s > 0
        assert result.bars_during_uptime == 62  # 31 + 31

    @pytest.mark.asyncio
    async def test_perfect_uptime_coverage_e2e_db(self):
        """Same scenario as above but through the full DB pipeline."""
        db = await _make_db()
        try:
            # Heartbeats: ON 0..1800, OFF, ON 3600..5400
            for i in range(31):
                await db.write_heartbeat("orchestrator", _ms(i * 60))
            for i in range(31):
                await db.write_heartbeat("orchestrator", _ms(3600 + i * 60))

            # Bars: present only during uptime
            for offset_s in ([i * 60 for i in range(31)] +
                             [3600 + i * 60 for i in range(31)]):
                ts = datetime.fromtimestamp(
                    _ms(offset_s) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                await db._connection.execute(
                    "INSERT INTO market_bars "
                    "(timestamp, source, symbol, bar_duration, "
                    "open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, "test", "SYM", 60, 100, 101, 99, 100.5, 500),
                )
            await db._connection.commit()

            start_ms = _ms(0)
            end_ms = _ms(5400)

            heartbeats = await db.get_heartbeats("orchestrator", start_ms, end_ms)
            bar_ts = await db.get_bar_timestamps("test", "SYM", 60, start_ms, end_ms)

            uptime = compute_uptime(heartbeats, start_ms, end_ms, 120_000)
            result = diagnose_coverage(bar_ts, 60, uptime, start_ms, end_ms)

            assert result.missing_during_uptime == 0
            assert result.uptime_coverage_pct >= 100.0
            assert result.bars_during_downtime == 0
        finally:
            await db.close()
