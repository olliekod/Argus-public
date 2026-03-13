"""
Tests for OutcomeEngine (Phase 4A.1)
=====================================

Run with::

    python -m pytest tests/test_outcome_engine.py -v

Test suites
-----------
1. Determinism       — same bars → identical outcomes across runs
2. Idempotency       — run twice via DB → same row count + values
3. Gap handling       — missing bar → status=GAP, path metrics NULL
4. Window semantics  — reference is bar close; horizon boundary correct
5. Status upgrade    — INCOMPLETE → OK when future bars arrive
6. Golden vector     — analytically computed expected values, exact match
"""

from __future__ import annotations

import math
import pytest
import asyncio
from typing import Dict, List

from src.core.outcome_engine import (
    BarData,
    OutcomeEngine,
    OutcomeResult,
    STATUS_GAP,
    STATUS_INCOMPLETE,
    STATUS_OK,
    _quantize,
)
from src.core.database import Database


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers / Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


BAR_DUR = 60  # 1-minute bars in seconds
BAR_DUR_MS = BAR_DUR * 1000
PROVIDER = "test"
SYMBOL = "TEST/USD"
VERSION = "TEST_V1"
QUANTIZE = 10

# Base timestamp: 2025-01-01 00:00:00 UTC in ms
BASE_TS_MS = 1_735_689_600_000


def _bar(index: int, close: float, high: float | None = None,
         low: float | None = None, open_: float | None = None,
         volume: float = 100.0) -> BarData:
    """Create a synthetic bar at BASE + index*60s.

    Defaults: open=close, high=close, low=close unless overridden.
    """
    return BarData(
        timestamp_ms=BASE_TS_MS + index * BAR_DUR_MS,
        open=open_ if open_ is not None else close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close,
        volume=volume,
    )


def _make_engine(config_overrides: dict | None = None) -> OutcomeEngine:
    """Build an OutcomeEngine with no DB (for sync tests)."""
    cfg: Dict = {
        "outcome_version": VERSION,
        "gap_tolerance_bars": 1,
        "quantize_decimals": QUANTIZE,
        "horizons_seconds_by_bar": {60: [300]},
    }
    if config_overrides:
        cfg.update(config_overrides)
    return OutcomeEngine(db=None, config=cfg)


def _make_linear_bars(n: int, start_close: float = 100.0,
                      step: float = 1.0) -> List[BarData]:
    """Create n bars with linearly increasing close prices.

    close[i] = start_close + i*step
    high[i]  = close[i] + 0.5
    low[i]   = close[i] - 0.5
    """
    return [
        _bar(i,
             close=start_close + i * step,
             high=start_close + i * step + 0.5,
             low=start_close + i * step - 0.5)
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Determinism
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Same bar data → identical outcomes across repeated runs."""

    def test_two_runs_identical(self):
        """Compute outcomes twice on the same bars; all fields must match."""
        engine = _make_engine()
        bars = _make_linear_bars(20)

        run1 = engine.compute_outcomes_from_bars_sync(
            bars, PROVIDER, SYMBOL, BAR_DUR, [300])
        run2 = engine.compute_outcomes_from_bars_sync(
            bars, PROVIDER, SYMBOL, BAR_DUR, [300])

        assert len(run1) == len(run2)
        for r1, r2 in zip(run1, run2):
            # Compare every deterministic field (exclude computed_at_ms)
            t1 = list(r1.to_tuple())
            t2 = list(r2.to_tuple())
            # Index 25 is computed_at_ms — allowed to differ
            t1[25] = None
            t2[25] = None
            assert t1 == t2, (
                f"Mismatch at ts_ms={r1.timestamp_ms}: {t1} != {t2}"
            )

    def test_determinism_multiple_horizons(self):
        """Multiple horizons produce identical results across runs."""
        cfg = {"horizons_seconds_by_bar": {60: [300, 600]}}
        engine = _make_engine(cfg)
        bars = _make_linear_bars(30)

        run1 = engine.compute_outcomes_from_bars_sync(
            bars, PROVIDER, SYMBOL, BAR_DUR, [300, 600])
        run2 = engine.compute_outcomes_from_bars_sync(
            bars, PROVIDER, SYMBOL, BAR_DUR, [300, 600])

        for r1, r2 in zip(run1, run2):
            t1 = list(r1.to_tuple())
            t2 = list(r2.to_tuple())
            t1[25] = None
            t2[25] = None
            assert t1 == t2


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Idempotency (via in-memory DB)
# ═══════════════════════════════════════════════════════════════════════════════


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


async def _insert_bars(db: Database, bars: List[BarData],
                       provider: str = PROVIDER,
                       symbol: str = SYMBOL,
                       bar_duration: int = BAR_DUR) -> None:
    """Insert synthetic bars into market_bars as ISO strings."""
    from datetime import datetime, timezone
    for b in bars:
        ts_dt = datetime.fromtimestamp(b.timestamp_ms / 1000, tz=timezone.utc)
        ts_iso = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            """INSERT OR IGNORE INTO market_bars
               (timestamp, symbol, source, open, high, low, close, volume,
                bar_duration)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts_iso, symbol, provider, b.open, b.high, b.low,
             b.close, b.volume, bar_duration),
        )


class TestIdempotency:
    """Run engine twice on the same data → same row count and values."""

    @pytest.mark.asyncio
    async def test_double_run_same_rows(self):
        db = await _make_db()
        try:
            bars = _make_linear_bars(15)
            await _insert_bars(db, bars)

            cfg = {
                "outcome_version": VERSION,
                "gap_tolerance_bars": 1,
                "quantize_decimals": QUANTIZE,
                "horizons_seconds_by_bar": {60: [300]},
            }
            engine = OutcomeEngine(db, cfg)

            start_ms = bars[0].timestamp_ms
            end_ms = bars[-1].timestamp_ms

            _, out1, ups1 = await engine.compute_outcomes_for_range(
                PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)
            _, out2, ups2 = await engine.compute_outcomes_for_range(
                PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)

            # Same number of outcomes computed
            assert out1 == out2

            # Check row count in DB hasn't doubled
            row = await db.fetch_one(
                "SELECT COUNT(*) as cnt FROM bar_outcomes")
            assert row["cnt"] == out1  # not out1 + out2
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_values_unchanged_on_rerun(self):
        db = await _make_db()
        try:
            bars = _make_linear_bars(12)
            await _insert_bars(db, bars)

            cfg = {
                "outcome_version": VERSION,
                "gap_tolerance_bars": 1,
                "quantize_decimals": QUANTIZE,
                "horizons_seconds_by_bar": {60: [300]},
            }
            engine = OutcomeEngine(db, cfg)

            start_ms = bars[0].timestamp_ms
            end_ms = bars[-1].timestamp_ms

            await engine.compute_outcomes_for_range(
                PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)
            snap1 = await db.fetch_all(
                "SELECT * FROM bar_outcomes ORDER BY timestamp_ms, horizon_seconds")

            await engine.compute_outcomes_for_range(
                PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)
            snap2 = await db.fetch_all(
                "SELECT * FROM bar_outcomes ORDER BY timestamp_ms, horizon_seconds")

            assert len(snap1) == len(snap2)
            for r1, r2 in zip(snap1, snap2):
                d1 = dict(r1)
                d2 = dict(r2)
                # Exclude computed_at_ms from comparison (wall-clock metadata)
                d1.pop("computed_at_ms", None)
                d2.pop("computed_at_ms", None)
                assert d1 == d2, f"Value drift at ts={d1.get('timestamp_ms')}"
        finally:
            await db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Gap Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestGapHandling:
    """Missing bars within the window → status=GAP, path metrics NULL."""

    def test_gap_beyond_tolerance(self):
        """Remove bars 3 and 4 from window → gap_count > 1 → GAP."""
        engine = _make_engine({"gap_tolerance_bars": 1})
        # 10 bars, but remove indices 3 and 4 (creating a 2-bar gap)
        bars = [_bar(i, close=100.0 + i) for i in range(10)]
        bars_with_gap = [b for b in bars if b.timestamp_ms not in (
            BASE_TS_MS + 3 * BAR_DUR_MS,
            BASE_TS_MS + 4 * BAR_DUR_MS,
        )]

        results = engine.compute_outcomes_from_bars_sync(
            bars_with_gap, PROVIDER, SYMBOL, BAR_DUR, [300])

        # Bar at index 0: window covers bars 1-5, but 3 and 4 are missing
        # bars_expected=5, bars_found=3, gap_count=2 > tolerance=1 → GAP
        r0 = results[0]
        assert r0.status == STATUS_GAP
        # Path metrics must be NULL for GAP
        assert r0.max_runup is None
        assert r0.max_drawdown is None
        assert r0.max_high_in_window is None
        assert r0.min_low_in_window is None
        assert r0.max_runup_ts_ms is None
        assert r0.max_drawdown_ts_ms is None
        # fwd_return can still be computed from last available bar
        # (or None if no future bars at all — here we do have some)

    def test_gap_within_tolerance(self):
        """Remove 1 bar when tolerance=1 → status=OK."""
        engine = _make_engine({"gap_tolerance_bars": 1})
        bars = [_bar(i, close=100.0 + i) for i in range(10)]
        # Remove bar at index 3 (1 bar missing in a 5-bar horizon window)
        bars_with_gap = [b for b in bars if b.timestamp_ms !=
                         BASE_TS_MS + 3 * BAR_DUR_MS]

        results = engine.compute_outcomes_from_bars_sync(
            bars_with_gap, PROVIDER, SYMBOL, BAR_DUR, [300])

        r0 = results[0]
        assert r0.status == STATUS_OK
        # Path metrics should be computed
        assert r0.max_runup is not None
        assert r0.max_drawdown is not None

    def test_no_future_bars(self):
        """Single bar with no future data → INCOMPLETE."""
        engine = _make_engine()
        bars = [_bar(0, close=100.0)]

        results = engine.compute_outcomes_from_bars_sync(
            bars, PROVIDER, SYMBOL, BAR_DUR, [300])

        assert len(results) == 1
        assert results[0].status == STATUS_INCOMPLETE
        assert results[0].close_at_horizon is None
        assert results[0].fwd_return is None


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Window Semantics
# ═══════════════════════════════════════════════════════════════════════════════


class TestWindowSemantics:
    """Verify reference point, window boundaries, and close_at_horizon."""

    def test_window_starts_at_bar_close(self):
        """close_ref_ms = bar.timestamp_ms + bar_duration_ms."""
        engine = _make_engine()
        bars = _make_linear_bars(10)

        results = engine.compute_outcomes_from_bars_sync(
            bars, PROVIDER, SYMBOL, BAR_DUR, [300])

        r0 = results[0]
        expected_close_ref = bars[0].timestamp_ms + BAR_DUR_MS
        assert r0.close_ref_ms == expected_close_ref
        assert r0.window_start_ms == expected_close_ref
        assert r0.window_end_ms == expected_close_ref + 300 * 1000

    def test_close_at_horizon_is_last_bar_in_window(self):
        """close_at_horizon should be the close of the last bar whose
        close_time <= window_end_ms."""
        engine = _make_engine()
        # 10 bars: close = 100, 101, 102, ..., 109
        bars = [_bar(i, close=100.0 + i) for i in range(10)]

        results = engine.compute_outcomes_from_bars_sync(
            bars, PROVIDER, SYMBOL, BAR_DUR, [300])

        # Bar 0 (close=100), horizon=300s → window covers bars 1-5
        # close_at_horizon = bar[5].close = 105
        r0 = results[0]
        assert r0.close_at_horizon == 105.0

    def test_bars_expected_count(self):
        """bars_expected = horizon_seconds / bar_duration_seconds."""
        engine = _make_engine()
        bars = _make_linear_bars(10)
        results = engine.compute_outcomes_from_bars_sync(
            bars, PROVIDER, SYMBOL, BAR_DUR, [300])
        assert results[0].bars_expected == 300 // BAR_DUR  # 5

    def test_future_bar_boundary_exclusive(self):
        """Bars exactly at the edge: a bar whose close_time == window_end
        should be included."""
        engine = _make_engine()
        # horizon=120s (2 bars), bar 0 window: [60000, 180000]
        # Bar 1 close at 120000ms (ts=60000 + 60000 = 120000) → ≤ 180000 → included
        # Bar 2 close at 180000ms (ts=120000 + 60000 = 180000) → ≤ 180000 → included
        # Bar 3 close at 240000ms → > 180000 → excluded
        bars = [_bar(i, close=100.0 + i) for i in range(5)]
        results = engine.compute_outcomes_from_bars_sync(
            bars, PROVIDER, SYMBOL, BAR_DUR, [120])

        r0 = results[0]
        assert r0.bars_expected == 2
        assert r0.bars_found == 2
        # close_at_horizon = bar[2].close = 102
        assert r0.close_at_horizon == 102.0


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Status Upgrade (INCOMPLETE → OK)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatusUpgrade:
    """INCOMPLETE records get upgraded to OK when future bars arrive."""

    @pytest.mark.asyncio
    async def test_incomplete_to_ok(self):
        """Insert only the anchor bar → INCOMPLETE (no future bars),
        then add future bars → upgrade to OK."""
        db = await _make_db()
        try:
            cfg = {
                "outcome_version": VERSION,
                "gap_tolerance_bars": 1,
                "quantize_decimals": QUANTIZE,
                "horizons_seconds_by_bar": {60: [300]},
            }
            engine = OutcomeEngine(db, cfg)

            # Phase 1: only bar 0 (no future data at all → INCOMPLETE)
            anchor = [_bar(0, close=100.0)]
            await _insert_bars(db, anchor)

            start_ms = anchor[0].timestamp_ms
            end_ms = anchor[0].timestamp_ms  # Just first bar

            await engine.compute_outcomes_for_range(
                PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)

            # Verify INCOMPLETE (bars_found == 0)
            row = await db.fetch_one(
                "SELECT status, fwd_return FROM bar_outcomes WHERE timestamp_ms = ?",
                (start_ms,))
            assert row["status"] == STATUS_INCOMPLETE
            assert row["fwd_return"] is None

            # Phase 2: add all future bars (indices 1-9)
            future_bars = [_bar(i, close=100.0 + i) for i in range(1, 10)]
            await _insert_bars(db, future_bars)

            await engine.compute_outcomes_for_range(
                PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)

            # Verify upgraded to OK
            row = await db.fetch_one(
                "SELECT status, fwd_return, close_at_horizon FROM bar_outcomes WHERE timestamp_ms = ?",
                (start_ms,))
            assert row["status"] == STATUS_OK
            assert row["fwd_return"] is not None
            assert row["close_at_horizon"] == 105.0

            # Row count should still be 1 (upsert, not insert)
            count_row = await db.fetch_one(
                "SELECT COUNT(*) as cnt FROM bar_outcomes WHERE timestamp_ms = ?",
                (start_ms,))
            assert count_row["cnt"] == 1
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_ok_not_overwritten(self):
        """Once a record is OK, re-running doesn't change it."""
        db = await _make_db()
        try:
            cfg = {
                "outcome_version": VERSION,
                "gap_tolerance_bars": 1,
                "quantize_decimals": QUANTIZE,
                "horizons_seconds_by_bar": {60: [300]},
            }
            engine = OutcomeEngine(db, cfg)

            bars = [_bar(i, close=100.0 + i) for i in range(10)]
            await _insert_bars(db, bars)

            start_ms = bars[0].timestamp_ms
            end_ms = bars[0].timestamp_ms

            await engine.compute_outcomes_for_range(
                PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)

            snap1 = await db.fetch_one(
                "SELECT * FROM bar_outcomes WHERE timestamp_ms = ?",
                (start_ms,))

            # Second run
            await engine.compute_outcomes_for_range(
                PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)

            snap2 = await db.fetch_one(
                "SELECT * FROM bar_outcomes WHERE timestamp_ms = ?",
                (start_ms,))

            d1 = dict(snap1)
            d2 = dict(snap2)
            d1.pop("computed_at_ms", None)
            d2.pop("computed_at_ms", None)
            assert d1 == d2
        finally:
            await db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Golden Test Vector
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoldenVector:
    """Analytically computed expected values — exact equality after quantization.

    Synthetic dataset (10 bars, 1-min, horizon=300s = 5 bars lookahead):

        Bar 0: ts=BASE, OHLC = (100, 102, 98,  100), vol=1000
        Bar 1: ts=BASE+60k, OHLC = (100, 103, 99,  101), vol=800
        Bar 2: ts=BASE+120k, OHLC = (101, 106, 100, 104), vol=1200  ← max high in window for bar 0
        Bar 3: ts=BASE+180k, OHLC = (104, 105, 97,  98),  vol=900   ← min low in window for bar 0
        Bar 4: ts=BASE+240k, OHLC = (98,  101, 96,  99),  vol=600
        Bar 5: ts=BASE+300k, OHLC = (99,  100, 95,  97),  vol=700   ← close_at_horizon for bar 0
        Bar 6: ts=BASE+360k, OHLC = (97,  98,  94,  96),  vol=500
        Bar 7: ts=BASE+420k, OHLC = (96,  97,  93,  95),  vol=400
        Bar 8: ts=BASE+480k, OHLC = (95,  96,  92,  94),  vol=350
        Bar 9: ts=BASE+540k, OHLC = (94,  95,  91,  93),  vol=300

    For bar 0 with horizon=300s:
        close_now = 100.0
        window: bars 1-5 (close times from BASE+120k to BASE+360k,
                          which is ≤ BASE+60k+300k = BASE+360k ✓)
        close_at_horizon = bar[5].close = 97.0
        fwd_return = (97/100) - 1 = -0.03
        max_high_in_window = 106.0 (bar 2)
        min_low_in_window  = 95.0 (bar 5)
        max_runup = (106/100) - 1 = 0.06
        max_drawdown = (95/100) - 1 = -0.05
        max_runup_ts_ms = bar[2].ts + 60000 = BASE + 180000
        max_drawdown_ts_ms = bar[5].ts + 60000 = BASE + 360000
        time_to_max_runup_ms = (BASE+180000) - (BASE+60000) = 120000
        time_to_max_drawdown_ms = (BASE+360000) - (BASE+60000) = 300000
        bars_expected = 5, bars_found = 5, gap_count = 0
        status = OK
    """

    GOLDEN_BARS = [
        BarData(BASE_TS_MS + 0 * BAR_DUR_MS, 100.0, 102.0, 98.0, 100.0, 1000.0),
        BarData(BASE_TS_MS + 1 * BAR_DUR_MS, 100.0, 103.0, 99.0, 101.0, 800.0),
        BarData(BASE_TS_MS + 2 * BAR_DUR_MS, 101.0, 106.0, 100.0, 104.0, 1200.0),
        BarData(BASE_TS_MS + 3 * BAR_DUR_MS, 104.0, 105.0, 97.0, 98.0, 900.0),
        BarData(BASE_TS_MS + 4 * BAR_DUR_MS, 98.0, 101.0, 96.0, 99.0, 600.0),
        BarData(BASE_TS_MS + 5 * BAR_DUR_MS, 99.0, 100.0, 95.0, 97.0, 700.0),
        BarData(BASE_TS_MS + 6 * BAR_DUR_MS, 97.0, 98.0, 94.0, 96.0, 500.0),
        BarData(BASE_TS_MS + 7 * BAR_DUR_MS, 96.0, 97.0, 93.0, 95.0, 400.0),
        BarData(BASE_TS_MS + 8 * BAR_DUR_MS, 95.0, 96.0, 92.0, 94.0, 350.0),
        BarData(BASE_TS_MS + 9 * BAR_DUR_MS, 94.0, 95.0, 91.0, 93.0, 300.0),
    ]

    def test_golden_bar0_horizon300(self):
        """Bar 0, horizon=300s — exact match of all fields."""
        engine = _make_engine()
        results = engine.compute_outcomes_from_bars_sync(
            self.GOLDEN_BARS, PROVIDER, SYMBOL, BAR_DUR, [300])

        # Find result for bar 0
        r = next(r for r in results if r.timestamp_ms == BASE_TS_MS)

        assert r.status == STATUS_OK
        assert r.close_now == 100.0
        assert r.close_at_horizon == 97.0
        assert r.fwd_return == _quantize((97.0 / 100.0) - 1, QUANTIZE)
        assert r.fwd_return == _quantize(-0.03, QUANTIZE)

        assert r.max_high_in_window == 106.0
        assert r.min_low_in_window == 95.0
        assert r.max_runup == _quantize((106.0 / 100.0) - 1, QUANTIZE)
        assert r.max_drawdown == _quantize((95.0 / 100.0) - 1, QUANTIZE)
        assert r.max_runup == _quantize(0.06, QUANTIZE)
        assert r.max_drawdown == _quantize(-0.05, QUANTIZE)

        # Timestamps of extrema
        assert r.max_runup_ts_ms == BASE_TS_MS + 2 * BAR_DUR_MS + BAR_DUR_MS
        assert r.max_drawdown_ts_ms == BASE_TS_MS + 5 * BAR_DUR_MS + BAR_DUR_MS

        # Time-to-extrema
        window_start = BASE_TS_MS + BAR_DUR_MS  # bar 0 close
        assert r.time_to_max_runup_ms == (BASE_TS_MS + 3 * BAR_DUR_MS) - window_start
        assert r.time_to_max_drawdown_ms == (BASE_TS_MS + 6 * BAR_DUR_MS) - window_start

        # Coverage
        assert r.bars_expected == 5
        assert r.bars_found == 5
        assert r.gap_count == 0

        # Window bounds
        assert r.close_ref_ms == BASE_TS_MS + BAR_DUR_MS
        assert r.window_start_ms == BASE_TS_MS + BAR_DUR_MS
        assert r.window_end_ms == BASE_TS_MS + BAR_DUR_MS + 300_000

    def test_golden_bar0_realized_vol(self):
        """Realized vol for bar 0 should match annualized stddev of log returns."""
        engine = _make_engine()
        results = engine.compute_outcomes_from_bars_sync(
            self.GOLDEN_BARS, PROVIDER, SYMBOL, BAR_DUR, [300])

        r = next(r for r in results if r.timestamp_ms == BASE_TS_MS)

        # Bars in vol computation: [bar0, bar1, bar2, bar3, bar4, bar5]
        # closes: [100, 101, 104, 98, 99, 97]
        closes = [100.0, 101.0, 104.0, 98.0, 99.0, 97.0]
        log_rets = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
        mean = sum(log_rets) / len(log_rets)
        var = sum((lr - mean) ** 2 for lr in log_rets) / (len(log_rets) - 1)
        vol_per_bar = math.sqrt(var)

        # Annualize: 252 trading days * 6.5h * 3600s / bar_duration_seconds
        periods_per_year = (252 * 6.5 * 3600) / BAR_DUR
        expected_vol = vol_per_bar * math.sqrt(periods_per_year)

        assert r.realized_vol == _quantize(expected_vol, QUANTIZE)

    def test_golden_last_bar_incomplete(self):
        """Last bar (index 9) has no future data → INCOMPLETE."""
        engine = _make_engine()
        results = engine.compute_outcomes_from_bars_sync(
            self.GOLDEN_BARS, PROVIDER, SYMBOL, BAR_DUR, [300])

        r_last = next(r for r in results
                      if r.timestamp_ms == BASE_TS_MS + 9 * BAR_DUR_MS)
        assert r_last.status == STATUS_INCOMPLETE

    def test_golden_bar4_partial_window(self):
        """Bar 4 (index=4) with horizon=300s: only bars 5-9 available,
        all 5 fit in window → OK."""
        engine = _make_engine()
        results = engine.compute_outcomes_from_bars_sync(
            self.GOLDEN_BARS, PROVIDER, SYMBOL, BAR_DUR, [300])

        r4 = next(r for r in results
                  if r.timestamp_ms == BASE_TS_MS + 4 * BAR_DUR_MS)

        # Bar 4 close = 99, window covers bars 5-9
        # close times: 360k, 420k, 480k, 540k, 600k
        # window_end = (4*60k + 60k) + 300k = BASE + 600k
        # bar 9 close_time = BASE + 9*60k + 60k = BASE + 600k → included
        assert r4.status == STATUS_OK
        assert r4.bars_found == 5
        assert r4.close_now == 99.0
        assert r4.close_at_horizon == 93.0
        assert r4.fwd_return == _quantize((93.0 / 99.0) - 1, QUANTIZE)

    def test_golden_with_gap(self):
        """Remove bar 3 from golden set → bar 0 should have gap_count=1.
        With tolerance=1, should still be OK."""
        engine = _make_engine({"gap_tolerance_bars": 1})
        bars_with_gap = [b for b in self.GOLDEN_BARS
                         if b.timestamp_ms != BASE_TS_MS + 3 * BAR_DUR_MS]

        results = engine.compute_outcomes_from_bars_sync(
            bars_with_gap, PROVIDER, SYMBOL, BAR_DUR, [300])

        r0 = next(r for r in results if r.timestamp_ms == BASE_TS_MS)
        assert r0.gap_count == 1
        assert r0.bars_found == 4
        assert r0.status == STATUS_OK

    def test_golden_with_gap_beyond_tolerance(self):
        """Remove bars 3 and 4 → gap_count=2 > tolerance=1 → GAP.
        Path metrics must be NULL."""
        engine = _make_engine({"gap_tolerance_bars": 1})
        bars_with_gap = [b for b in self.GOLDEN_BARS
                         if b.timestamp_ms not in (
                             BASE_TS_MS + 3 * BAR_DUR_MS,
                             BASE_TS_MS + 4 * BAR_DUR_MS,
                         )]

        results = engine.compute_outcomes_from_bars_sync(
            bars_with_gap, PROVIDER, SYMBOL, BAR_DUR, [300])

        r0 = next(r for r in results if r.timestamp_ms == BASE_TS_MS)
        assert r0.status == STATUS_GAP
        assert r0.gap_count == 2
        assert r0.max_runup is None
        assert r0.max_drawdown is None
        assert r0.max_high_in_window is None
        assert r0.min_low_in_window is None

    def test_golden_fwd_return_sign(self):
        """Sanity check: bar 0 is a decline (100→97), fwd_return < 0."""
        engine = _make_engine()
        results = engine.compute_outcomes_from_bars_sync(
            self.GOLDEN_BARS, PROVIDER, SYMBOL, BAR_DUR, [300])
        r0 = next(r for r in results if r.timestamp_ms == BASE_TS_MS)
        assert r0.fwd_return < 0

    def test_golden_runup_positive_drawdown_negative(self):
        """max_runup > 0 (max high > close_now), max_drawdown < 0."""
        engine = _make_engine()
        results = engine.compute_outcomes_from_bars_sync(
            self.GOLDEN_BARS, PROVIDER, SYMBOL, BAR_DUR, [300])
        r0 = next(r for r in results if r.timestamp_ms == BASE_TS_MS)
        assert r0.max_runup > 0
        assert r0.max_drawdown < 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Quantization helper
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuantize:
    """Verify the _quantize helper."""

    def test_basic(self):
        assert _quantize(0.123456789012345, 10) == round(0.123456789012345, 10)

    def test_none(self):
        assert _quantize(None, 10) is None

    def test_nan(self):
        assert _quantize(float("nan"), 10) is None

    def test_inf(self):
        assert _quantize(float("inf"), 10) is None

    def test_zero(self):
        assert _quantize(0.0, 10) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DB integration: full round-trip golden vector
# ═══════════════════════════════════════════════════════════════════════════════


class TestDBRoundTrip:
    """Full round-trip: insert bars → compute → read from DB → verify golden values."""

    @pytest.mark.asyncio
    async def test_golden_roundtrip(self):
        db = await _make_db()
        try:
            await _insert_bars(db, TestGoldenVector.GOLDEN_BARS)

            cfg = {
                "outcome_version": VERSION,
                "gap_tolerance_bars": 1,
                "quantize_decimals": QUANTIZE,
                "horizons_seconds_by_bar": {60: [300]},
            }
            engine = OutcomeEngine(db, cfg)

            start_ms = TestGoldenVector.GOLDEN_BARS[0].timestamp_ms
            end_ms = TestGoldenVector.GOLDEN_BARS[-1].timestamp_ms

            bars_proc, outcomes_computed, upserted = \
                await engine.compute_outcomes_for_range(
                    PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)

            assert bars_proc == 10
            assert outcomes_computed == 10  # 10 bars × 1 horizon

            # Read bar 0 outcome from DB
            row = await db.fetch_one(
                """SELECT * FROM bar_outcomes
                   WHERE timestamp_ms = ? AND horizon_seconds = 300""",
                (BASE_TS_MS,))

            assert row is not None
            assert row["status"] == STATUS_OK
            assert row["close_now"] == 100.0
            assert row["close_at_horizon"] == 97.0
            assert row["fwd_return"] == _quantize(-0.03, QUANTIZE)
            assert row["max_high_in_window"] == 106.0
            assert row["min_low_in_window"] == 95.0
            assert row["max_runup"] == _quantize(0.06, QUANTIZE)
            assert row["max_drawdown"] == _quantize(-0.05, QUANTIZE)
            assert row["bars_expected"] == 5
            assert row["bars_found"] == 5
            assert row["gap_count"] == 0
        finally:
            await db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  DB inventory methods (used by CLI `list` / `list-outcomes`)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarInventory:
    """Verify get_bar_inventory returns correct keys and counts."""

    @pytest.mark.asyncio
    async def test_empty_db(self):
        db = await _make_db()
        try:
            inv = await db.get_bar_inventory()
            assert inv == []
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_single_provider_symbol(self):
        db = await _make_db()
        try:
            bars = _make_linear_bars(5)
            await _insert_bars(db, bars, provider="bybit", symbol="BTCUSDT")

            inv = await db.get_bar_inventory()
            assert len(inv) == 1
            row = inv[0]
            assert row["source"] == "bybit"
            assert row["symbol"] == "BTCUSDT"
            assert row["bar_duration"] == BAR_DUR
            assert row["bar_count"] == 5
            assert row["min_ts"] is not None
            assert row["max_ts"] is not None
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_multiple_providers(self):
        db = await _make_db()
        try:
            bars = _make_linear_bars(3)
            await _insert_bars(db, bars, provider="bybit", symbol="BTCUSDT")
            await _insert_bars(db, bars, provider="yahoo", symbol="IBIT")

            inv = await db.get_bar_inventory()
            assert len(inv) == 2
            sources = {r["source"] for r in inv}
            assert sources == {"bybit", "yahoo"}
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_equity_inventory_multi_provider(self):
        db = await _make_db()
        try:
            bars = _make_linear_bars(2)
            await _insert_bars(db, bars, provider="yahoo", symbol="SPY")
            await _insert_bars(db, bars, provider="alpaca", symbol="SPY")

            inv = await db.get_bar_inventory()
            assert len(inv) == 2
            keys = {(row["source"], row["symbol"], row["bar_duration"]) for row in inv}
            assert ("yahoo", "SPY", BAR_DUR) in keys
            assert ("alpaca", "SPY", BAR_DUR) in keys
        finally:
            await db.close()


class TestOutcomeInventory:
    """Verify get_outcome_inventory returns correct keys and status counts."""

    @pytest.mark.asyncio
    async def test_empty_db(self):
        db = await _make_db()
        try:
            inv = await db.get_outcome_inventory()
            assert inv == []
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_after_backfill(self):
        db = await _make_db()
        try:
            bars = _make_linear_bars(10)
            await _insert_bars(db, bars)

            cfg = {
                "outcome_version": VERSION,
                "gap_tolerance_bars": 1,
                "quantize_decimals": QUANTIZE,
                "horizons_seconds_by_bar": {60: [300]},
            }
            engine = OutcomeEngine(db, cfg)
            start_ms = bars[0].timestamp_ms
            end_ms = bars[-1].timestamp_ms

            await engine.compute_outcomes_for_range(
                PROVIDER, SYMBOL, BAR_DUR, start_ms, end_ms)

            inv = await db.get_outcome_inventory()
            assert len(inv) == 1  # 1 unique (provider, symbol, bar_dur, horizon)
            row = inv[0]
            assert row["provider"] == PROVIDER
            assert row["symbol"] == SYMBOL
            assert row["bar_duration_seconds"] == BAR_DUR
            assert row["horizon_seconds"] == 300
            assert row["total"] == 10
            # Should have some OK and some INCOMPLETE
            assert row["ok_count"] + row["incomplete_count"] + row["gap_count"] == 10
            assert row["ok_count"] > 0
            assert row["incomplete_count"] > 0  # last bars lack future data
        finally:
            await db.close()


class TestBackfillKeyValidation:
    """The backfill CLI should catch wrong provider/symbol before computing."""

    @pytest.mark.asyncio
    async def test_wrong_provider_returns_zero(self):
        """Engine returns (0,0,0) when provider doesn't match any bars."""
        db = await _make_db()
        try:
            bars = _make_linear_bars(10)
            await _insert_bars(db, bars, provider="bybit", symbol="BTCUSDT")

            cfg = {
                "outcome_version": VERSION,
                "gap_tolerance_bars": 1,
                "quantize_decimals": QUANTIZE,
                "horizons_seconds_by_bar": {60: [300]},
            }
            engine = OutcomeEngine(db, cfg)
            start_ms = bars[0].timestamp_ms
            end_ms = bars[-1].timestamp_ms

            # Wrong provider
            b, o, u = await engine.compute_outcomes_for_range(
                "alpaca", "BTCUSDT", BAR_DUR, start_ms, end_ms)
            assert b == 0 and o == 0 and u == 0

            # Wrong symbol
            b, o, u = await engine.compute_outcomes_for_range(
                "bybit", "BTC/USDT:USDT", BAR_DUR, start_ms, end_ms)
            assert b == 0 and o == 0 and u == 0
        finally:
            await db.close()
