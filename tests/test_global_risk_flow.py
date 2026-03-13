"""
Tests for GlobalRiskFlow computation.

Covers:
- Component return computation
- Weight redistribution with missing data
- Strict less-than lookahead prevention
- Full and partial signal computation
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List

from src.core.global_risk_flow import (
    compute_global_risk_flow,
    _latest_daily_return,
    _avg_return,
    ASIA_SYMBOLS,
    EUROPE_SYMBOLS,
    FX_RISK_SYMBOL,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_DAY_MS = 86_400_000  # 1 day in milliseconds


def _daily_bar(day_offset: int, close: float, symbol: str = "EWJ") -> Dict[str, Any]:
    """Create a daily bar dict at day_offset * 86400s from epoch."""
    return {
        "timestamp_ms": day_offset * _DAY_MS,
        "close": close,
        "symbol": symbol,
    }


def _make_bars(
    symbol: str,
    closes: List[float],
    start_day: int = 1,
) -> List[Dict[str, Any]]:
    """Create a sequence of daily bars for a symbol."""
    return [_daily_bar(start_day + i, c, symbol) for i, c in enumerate(closes)]


# ═══════════════════════════════════════════════════════════════════════════
# Tests for _latest_daily_return
# ═══════════════════════════════════════════════════════════════════════════


class TestLatestDailyReturn:
    """Tests for single-symbol daily return computation."""

    def test_basic_return(self):
        """Simple 1% return."""
        bars = _make_bars("EWJ", [100.0, 101.0])
        sim_time = 10 * _DAY_MS  # well after both bars
        ret = _latest_daily_return(bars, sim_time)
        assert ret is not None
        assert abs(ret - 0.01) < 1e-10

    def test_negative_return(self):
        """Negative 2% return."""
        bars = _make_bars("EWJ", [100.0, 98.0])
        sim_time = 10 * _DAY_MS
        ret = _latest_daily_return(bars, sim_time)
        assert ret is not None
        assert abs(ret - (-0.02)) < 1e-10

    def test_strict_less_than_lookahead(self):
        """Bars at exactly sim_time are excluded (strict <)."""
        bars = _make_bars("EWJ", [100.0, 101.0], start_day=5)
        # sim_time exactly equals the second bar's timestamp
        sim_time = 6 * _DAY_MS
        ret = _latest_daily_return(bars, sim_time)
        # Only one bar qualifies (the one at day 5), so return is None
        assert ret is None

    def test_insufficient_bars(self):
        """Returns None with < 2 eligible bars."""
        bars = _make_bars("EWJ", [100.0])
        ret = _latest_daily_return(bars, 10 * _DAY_MS)
        assert ret is None

    def test_empty_bars(self):
        """Returns None with no bars."""
        assert _latest_daily_return([], 10 * _DAY_MS) is None

    def test_zero_previous_close(self):
        """Returns None when previous close is zero (avoid division by zero)."""
        bars = _make_bars("EWJ", [0.0, 100.0])
        ret = _latest_daily_return(bars, 10 * _DAY_MS)
        assert ret is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests for _avg_return
# ═══════════════════════════════════════════════════════════════════════════


class TestAvgReturn:
    """Tests for multi-symbol average return."""

    def test_multiple_symbols(self):
        """Average of two symbols with different returns."""
        bars_by_symbol = {
            "EWJ": _make_bars("EWJ", [100.0, 102.0]),  # +2%
            "FXI": _make_bars("FXI", [100.0, 104.0]),  # +4%
        }
        ret = _avg_return(bars_by_symbol, ("EWJ", "FXI"), 10 * _DAY_MS)
        assert ret is not None
        assert abs(ret - 0.03) < 1e-10  # avg of 0.02 and 0.04

    def test_missing_symbol_skipped(self):
        """Missing symbols are skipped, average of remaining."""
        bars_by_symbol = {
            "EWJ": _make_bars("EWJ", [100.0, 102.0]),  # +2%
            # FXI missing
        }
        ret = _avg_return(bars_by_symbol, ("EWJ", "FXI"), 10 * _DAY_MS)
        assert ret is not None
        assert abs(ret - 0.02) < 1e-10  # only EWJ

    def test_all_missing(self):
        """Returns None when all symbols are missing."""
        ret = _avg_return({}, ASIA_SYMBOLS, 10 * _DAY_MS)
        assert ret is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests for compute_global_risk_flow
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeGlobalRiskFlow:
    """Tests for the main composite signal."""

    def test_all_components_positive(self):
        """All regions positive → positive risk flow."""
        bars_by_symbol = {}
        # Asia: all +1%
        for sym in ASIA_SYMBOLS:
            bars_by_symbol[sym] = _make_bars(sym, [100.0, 101.0])
        # Europe: all +2%
        for sym in EUROPE_SYMBOLS:
            bars_by_symbol[sym] = _make_bars(sym, [100.0, 102.0])
        # FX: +0.5% (yen weakening = risk-on)
        bars_by_symbol[FX_RISK_SYMBOL] = _make_bars(
            FX_RISK_SYMBOL, [100.0, 100.5],
        )

        flow = compute_global_risk_flow(bars_by_symbol, 10 * _DAY_MS)
        assert flow is not None
        assert flow > 0

        # Expected: 0.4*0.01 + 0.4*0.02 + 0.2*0.005 = 0.013
        assert abs(flow - 0.013) < 1e-10

    def test_all_components_negative(self):
        """All regions negative → negative risk flow."""
        bars_by_symbol = {}
        for sym in ASIA_SYMBOLS:
            bars_by_symbol[sym] = _make_bars(sym, [100.0, 99.0])
        for sym in EUROPE_SYMBOLS:
            bars_by_symbol[sym] = _make_bars(sym, [100.0, 98.0])
        bars_by_symbol[FX_RISK_SYMBOL] = _make_bars(
            FX_RISK_SYMBOL, [100.0, 99.5],
        )

        flow = compute_global_risk_flow(bars_by_symbol, 10 * _DAY_MS)
        assert flow is not None
        assert flow < 0

    def test_weight_redistribution_no_fx(self):
        """When FX is missing, weights redistribute: Asia=0.5, Europe=0.5."""
        bars_by_symbol = {}
        for sym in ASIA_SYMBOLS:
            bars_by_symbol[sym] = _make_bars(sym, [100.0, 101.0])  # +1%
        for sym in EUROPE_SYMBOLS:
            bars_by_symbol[sym] = _make_bars(sym, [100.0, 103.0])  # +3%
        # FX_RISK_SYMBOL intentionally missing

        flow = compute_global_risk_flow(bars_by_symbol, 10 * _DAY_MS)
        assert flow is not None
        # Total base weight = 0.4 + 0.4 = 0.8
        # Redistributed: Asia = 0.4/0.8 = 0.5, Europe = 0.5
        # Expected: 0.5*0.01 + 0.5*0.03 = 0.02
        assert abs(flow - 0.02) < 1e-10

    def test_only_asia_available(self):
        """Only Asia data → full weight on Asia."""
        bars_by_symbol = {}
        for sym in ASIA_SYMBOLS:
            bars_by_symbol[sym] = _make_bars(sym, [100.0, 102.0])  # +2%

        flow = compute_global_risk_flow(bars_by_symbol, 10 * _DAY_MS)
        assert flow is not None
        # All weight on Asia: 1.0 * 0.02 = 0.02
        assert abs(flow - 0.02) < 1e-10

    def test_no_data_returns_none(self):
        """No data at all → None."""
        flow = compute_global_risk_flow({}, 10 * _DAY_MS)
        assert flow is None

    def test_determinism(self):
        """Same inputs → same output."""
        bars_by_symbol = {}
        for sym in ASIA_SYMBOLS:
            bars_by_symbol[sym] = _make_bars(sym, [100.0, 101.5])
        for sym in EUROPE_SYMBOLS:
            bars_by_symbol[sym] = _make_bars(sym, [100.0, 100.75])

        sim_time = 10 * _DAY_MS
        flow_a = compute_global_risk_flow(bars_by_symbol, sim_time)
        flow_b = compute_global_risk_flow(bars_by_symbol, sim_time)
        assert flow_a == flow_b
