"""
Tests for session_regime on OutcomeResult.

Validates that the OutcomeEngine populates session_regime deterministically
and that session grouping works for analytics.
"""

from __future__ import annotations

import pytest

from src.core.outcome_engine import BarData, OutcomeEngine, OutcomeResult


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ts_for_hour_et(hour: int, minute: int = 0) -> int:
    """Build a UTC epoch-ms for a given Eastern Time hour (EST, no DST).

    In EST (UTC-5): hour 9 ET → 14 UTC.
    """
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    # Use a known EST date (January — no DST)
    dt_et = datetime(2025, 1, 15, hour, minute, 0, tzinfo=et)
    return int(dt_et.timestamp() * 1000)


def _make_bar(ts_ms: int, close: float = 100.0) -> BarData:
    return BarData(
        timestamp_ms=ts_ms,
        open=close - 0.5,
        high=close + 0.5,
        low=close - 1.0,
        close=close,
        volume=1000.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOutcomeSessionRegime:
    def test_rth_bar_gets_rth_session(self):
        """Bar at 10:00 ET should be RTH."""
        ts = _ts_for_hour_et(10, 0)
        bar = _make_bar(ts)
        engine = OutcomeEngine(db=None, config={
            "horizons_seconds_by_bar": {60: [300]},
        })
        # Use sync method with a small set of bars
        bars = [bar, _make_bar(ts + 60_000), _make_bar(ts + 120_000),
                _make_bar(ts + 180_000), _make_bar(ts + 240_000),
                _make_bar(ts + 300_000), _make_bar(ts + 360_000)]
        results = engine.compute_outcomes_from_bars_sync(
            bars=bars,
            provider="test",
            symbol="SPY",
            bar_duration_seconds=60,
            horizons=[300],
        )
        # First bar's outcome should have RTH
        first = results[0]
        assert first.session_regime == "RTH"

    def test_pre_market_bar_gets_pre_session(self):
        """Bar at 7:00 ET should be PRE."""
        ts = _ts_for_hour_et(7, 0)
        bar = _make_bar(ts)
        engine = OutcomeEngine(db=None, config={
            "horizons_seconds_by_bar": {60: [300]},
        })
        bars = [bar] + [_make_bar(ts + i * 60_000) for i in range(1, 7)]
        results = engine.compute_outcomes_from_bars_sync(
            bars=bars, provider="test", symbol="SPY",
            bar_duration_seconds=60, horizons=[300],
        )
        assert results[0].session_regime == "PRE"

    def test_post_market_bar_gets_post_session(self):
        """Bar at 17:00 ET should be POST."""
        ts = _ts_for_hour_et(17, 0)
        bar = _make_bar(ts)
        engine = OutcomeEngine(db=None, config={
            "horizons_seconds_by_bar": {60: [300]},
        })
        bars = [bar] + [_make_bar(ts + i * 60_000) for i in range(1, 7)]
        results = engine.compute_outcomes_from_bars_sync(
            bars=bars, provider="test", symbol="SPY",
            bar_duration_seconds=60, horizons=[300],
        )
        assert results[0].session_regime == "POST"

    def test_closed_session(self):
        """Bar at 2:00 ET should be CLOSED."""
        ts = _ts_for_hour_et(2, 0)
        bar = _make_bar(ts)
        engine = OutcomeEngine(db=None, config={
            "horizons_seconds_by_bar": {60: [300]},
        })
        bars = [bar] + [_make_bar(ts + i * 60_000) for i in range(1, 7)]
        results = engine.compute_outcomes_from_bars_sync(
            bars=bars, provider="test", symbol="SPY",
            bar_duration_seconds=60, horizons=[300],
        )
        assert results[0].session_regime == "CLOSED"

    def test_session_regime_deterministic(self):
        """Same bar → same session_regime every time."""
        ts = _ts_for_hour_et(10, 30)
        bar = _make_bar(ts)
        engine = OutcomeEngine(db=None, config={
            "horizons_seconds_by_bar": {60: [300]},
        })
        bars = [bar] + [_make_bar(ts + i * 60_000) for i in range(1, 7)]
        for _ in range(5):
            results = engine.compute_outcomes_from_bars_sync(
                bars=bars, provider="test", symbol="SPY",
                bar_duration_seconds=60, horizons=[300],
            )
            assert results[0].session_regime == "RTH"

    def test_session_not_in_to_tuple(self):
        """session_regime should NOT appear in to_tuple (not persisted to DB)."""
        ts = _ts_for_hour_et(10, 0)
        bar = _make_bar(ts)
        engine = OutcomeEngine(db=None, config={
            "horizons_seconds_by_bar": {60: [300]},
        })
        bars = [bar] + [_make_bar(ts + i * 60_000) for i in range(1, 7)]
        results = engine.compute_outcomes_from_bars_sync(
            bars=bars, provider="test", symbol="SPY",
            bar_duration_seconds=60, horizons=[300],
        )
        t = results[0].to_tuple()
        # to_tuple has exactly 26 elements (all the DB columns)
        assert len(t) == 26
        # session_regime string should not be in the tuple
        assert "RTH" not in t

    def test_grouping_by_session(self):
        """Analytics pattern: group outcomes by session_regime."""
        engine = OutcomeEngine(db=None, config={
            "horizons_seconds_by_bar": {60: [300]},
        })
        # Mix of RTH and PRE bars
        rth_ts = _ts_for_hour_et(11, 0)
        pre_ts = _ts_for_hour_et(6, 0)
        bars_rth = [_make_bar(rth_ts + i * 60_000) for i in range(7)]
        bars_pre = [_make_bar(pre_ts + i * 60_000) for i in range(7)]
        all_bars = bars_pre + bars_rth

        results = engine.compute_outcomes_from_bars_sync(
            bars=all_bars, provider="test", symbol="SPY",
            bar_duration_seconds=60, horizons=[300],
        )

        # Group by session
        by_session: dict = {}
        for r in results:
            by_session.setdefault(r.session_regime, []).append(r)

        assert "RTH" in by_session
        assert "PRE" in by_session
        assert len(by_session["RTH"]) > 0
        assert len(by_session["PRE"]) > 0
