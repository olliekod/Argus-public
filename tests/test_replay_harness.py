"""
Tests for the Deterministic Replay Harness.

Validates the lookahead barrier, strategy interface, portfolio tracking,
session awareness, and end-to-end replay flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from src.analysis.execution_model import ExecutionConfig, ExecutionModel, Quote
from src.analysis.replay_harness import (
    ReplayConfig,
    ReplayHarness,
    ReplayStrategy,
    TradeIntent,
    VirtualPortfolio,
    Position,
    PortfolioSnapshot,
)
from src.core.outcome_engine import BarData, OutcomeResult


# ═══════════════════════════════════════════════════════════════════════════
# Test strategy implementations
# ═══════════════════════════════════════════════════════════════════════════

class NullStrategy(ReplayStrategy):
    """Does nothing — used to test the harness itself."""

    @property
    def strategy_id(self) -> str:
        return "NULL_STRATEGY"

    def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
        pass

    def generate_intents(self, sim_ts_ms):
        return []


class BuyOnSecondBarStrategy(ReplayStrategy):
    """Buys 1 contract on the 2nd bar, sells on the 4th."""

    def __init__(self):
        self._bar_count = 0
        self._holding = False
        self._fills = []

    @property
    def strategy_id(self) -> str:
        return "BUY_ON_SECOND_BAR"

    def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
        self._bar_count += 1

    def generate_intents(self, sim_ts_ms):
        if self._bar_count == 2 and not self._holding:
            self._holding = True
            return [TradeIntent(
                symbol="SPY", side="BUY", quantity=1,
                intent_type="OPEN", tag="test_entry",
            )]
        if self._bar_count == 4 and self._holding:
            self._holding = False
            return [TradeIntent(
                symbol="SPY", side="SELL", quantity=1,
                intent_type="CLOSE", tag="test_exit",
            )]
        return []

    def on_fill(self, intent, fill):
        self._fills.append(fill)

    def finalize(self):
        return {"bars_seen": self._bar_count, "fills": len(self._fills)}


class OutcomeLeakDetector(ReplayStrategy):
    """Records which outcomes are visible at each bar.

    Used to verify the lookahead barrier.
    """

    def __init__(self):
        self._observations: List[Dict[str, Any]] = []

    @property
    def strategy_id(self) -> str:
        return "LEAK_DETECTOR"

    def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
        self._observations.append({
            "bar_ts_ms": bar.timestamp_ms,
            "sim_ts_ms": sim_ts_ms,
            "outcome_count": len(visible_outcomes),
            "outcome_keys": list(visible_outcomes.keys()),
        })

    def generate_intents(self, sim_ts_ms):
        return []

    def finalize(self):
        return {"observations": self._observations}


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _make_bars(
    n: int = 10,
    start_ms: int = 1_700_000_000_000,
    interval_ms: int = 60_000,
    open_price: float = 100.0,
    drift: float = 0.5,
) -> List[BarData]:
    """Generate synthetic bars in ascending order."""
    bars = []
    price = open_price
    for i in range(n):
        ts = start_ms + i * interval_ms
        bars.append(BarData(
            timestamp_ms=ts,
            open=round(price, 2),
            high=round(price + 1.0, 2),
            low=round(price - 0.5, 2),
            close=round(price + drift, 2),
            volume=1000.0 + i * 100,
        ))
        price += drift
    return bars


def _make_outcomes(
    bars: List[BarData],
    horizon_seconds: int = 300,
    bar_duration_seconds: int = 60,
) -> List[Dict[str, Any]]:
    """Generate synthetic outcomes for each bar."""
    outcomes = []
    bar_dur_ms = bar_duration_seconds * 1000
    horizon_ms = horizon_seconds * 1000
    for bar in bars:
        close_ref_ms = bar.timestamp_ms + bar_dur_ms
        window_end_ms = close_ref_ms + horizon_ms
        outcomes.append({
            "provider": "test",
            "symbol": "SPY",
            "bar_duration_seconds": bar_duration_seconds,
            "timestamp_ms": bar.timestamp_ms,
            "horizon_seconds": horizon_seconds,
            "outcome_version": "TEST_V1",
            "close_now": bar.close,
            "close_at_horizon": bar.close + 1.0,
            "fwd_return": 0.01,
            "max_runup": 0.02,
            "max_drawdown": -0.005,
            "realized_vol": 0.15,
            "max_high_in_window": bar.high + 1.0,
            "min_low_in_window": bar.low,
            "max_runup_ts_ms": close_ref_ms + 120_000,
            "max_drawdown_ts_ms": close_ref_ms + 60_000,
            "time_to_max_runup_ms": 120_000,
            "time_to_max_drawdown_ms": 60_000,
            "status": "OK",
            "close_ref_ms": close_ref_ms,
            "window_start_ms": close_ref_ms,
            "window_end_ms": window_end_ms,
            "bars_expected": horizon_seconds // bar_duration_seconds,
            "bars_found": horizon_seconds // bar_duration_seconds,
            "gap_count": 0,
            "computed_at_ms": window_end_ms,
        })
    return outcomes


# ═══════════════════════════════════════════════════════════════════════════
# Lookahead barrier tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLookaheadBarrier:
    def test_outcomes_not_visible_before_window_ends(self):
        """The core invariant: no outcome is visible before its window_end_ms."""
        bars = _make_bars(n=10, interval_ms=60_000)
        outcomes = _make_outcomes(bars, horizon_seconds=300)
        strategy = OutcomeLeakDetector()
        exec_model = ExecutionModel()
        harness = ReplayHarness(
            bars=bars,
            outcomes=outcomes,
            strategy=strategy,
            execution_model=exec_model,
        )
        result = harness.run()
        observations = result.strategy_state["observations"]

        for obs in observations:
            sim_ts_ms = obs["sim_ts_ms"]
            for outcome_ts in obs["outcome_keys"]:
                # Find the outcome
                oc = next(o for o in outcomes if o["timestamp_ms"] == outcome_ts)
                # The invariant: sim_time >= window_end_ms
                assert sim_ts_ms >= oc["window_end_ms"], (
                    f"Lookahead leak! sim_time={sim_ts_ms} < "
                    f"window_end={oc['window_end_ms']} for outcome at {outcome_ts}"
                )

    def test_no_outcomes_visible_for_first_bars(self):
        """With a 5-minute horizon and 1-minute bars, the first 5 bars
        should not have any outcomes visible (window hasn't closed yet)."""
        bars = _make_bars(n=10, interval_ms=60_000)
        outcomes = _make_outcomes(bars, horizon_seconds=300)
        strategy = OutcomeLeakDetector()
        harness = ReplayHarness(
            bars=bars,
            outcomes=outcomes,
            strategy=strategy,
            execution_model=ExecutionModel(),
        )
        result = harness.run()
        obs = result.strategy_state["observations"]

        # First bar sim_ts = start + 60_000.  First outcome window_end = start + 60_000 + 300_000
        # So first 5 bars (sim_ts up to start + 5*60_000 = start + 300_000) should see 0 outcomes
        # Bar 6 sim_ts = start + 6*60_000 = start + 360_000 >= first window_end = start + 360_000 → visible
        for i in range(5):
            assert obs[i]["outcome_count"] == 0, f"Bar {i} should see 0 outcomes"

    def test_outcomes_accumulate_over_time(self):
        """Later bars should see more outcomes (all prior whose windows have closed)."""
        bars = _make_bars(n=15, interval_ms=60_000)
        outcomes = _make_outcomes(bars, horizon_seconds=300)
        strategy = OutcomeLeakDetector()
        harness = ReplayHarness(
            bars=bars,
            outcomes=outcomes,
            strategy=strategy,
            execution_model=ExecutionModel(),
        )
        result = harness.run()
        obs = result.strategy_state["observations"]

        counts = [o["outcome_count"] for o in obs]
        # Should be non-decreasing
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], "Outcome count should not decrease"

    def test_empty_outcomes_safe(self):
        """Replay works fine with no outcomes at all."""
        bars = _make_bars(n=5)
        strategy = NullStrategy()
        harness = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=strategy,
            execution_model=ExecutionModel(),
        )
        result = harness.run()
        assert result.bars_replayed == 5
        assert result.outcomes_used == 0


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVirtualPortfolio:
    def test_open_long_deducts_cash(self):
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        pos = portfolio.open_position(
            symbol="SPY", side="LONG", quantity=1,
            fill_price=2.00, ts_ms=1000, multiplier=100,
        )
        assert portfolio._cash == 10_000.0 - (2.00 * 1 * 100)

    def test_open_short_adds_cash(self):
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        pos = portfolio.open_position(
            symbol="SPY", side="SHORT", quantity=1,
            fill_price=2.00, ts_ms=1000, multiplier=100,
        )
        assert portfolio._cash == 10_000.0 + (2.00 * 1 * 100)

    def test_close_long_with_profit(self):
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        pos = portfolio.open_position(
            symbol="SPY", side="LONG", quantity=1,
            fill_price=2.00, ts_ms=1000, multiplier=100,
        )
        pnl = portfolio.close_position(pos, fill_price=3.00, ts_ms=2000, multiplier=100)
        assert pnl == (3.00 - 2.00) * 1 * 100  # +100
        assert pos.closed is True

    def test_close_short_with_profit(self):
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        pos = portfolio.open_position(
            symbol="SPY", side="SHORT", quantity=1,
            fill_price=2.00, ts_ms=1000, multiplier=100,
        )
        pnl = portfolio.close_position(pos, fill_price=1.50, ts_ms=2000, multiplier=100)
        assert pnl == (2.00 - 1.50) * 1 * 100  # +50
        assert pos.closed is True

    def test_close_short_with_loss(self):
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        pos = portfolio.open_position(
            symbol="SPY", side="SHORT", quantity=1,
            fill_price=2.00, ts_ms=1000, multiplier=100,
        )
        pnl = portfolio.close_position(pos, fill_price=2.50, ts_ms=2000, multiplier=100)
        assert pnl == (2.00 - 2.50) * 1 * 100  # -50

    def test_mark_to_market(self):
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        portfolio.open_position(
            symbol="SPY", side="LONG", quantity=1,
            fill_price=100.0, ts_ms=1000, multiplier=1,  # use multiplier=1 for simplicity
        )
        snap = portfolio.mark_to_market(
            prices={"SPY": 105.0},
            ts_ms=2000,
            multiplier=1,
        )
        assert snap.unrealized_pnl == 5.0
        assert snap.equity == 10_000.0 - 100.0 + 5.0  # cash - cost + unrealized

    def test_drawdown_tracking(self):
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        pos = portfolio.open_position(
            symbol="SPY", side="LONG", quantity=1,
            fill_price=100.0, ts_ms=1000, multiplier=1,
        )
        # Mark up: equity = 9900 + 10 = 9910 (still below peak of 10000)
        portfolio.mark_to_market({"SPY": 110.0}, ts_ms=2000, multiplier=1)
        # Mark down: equity = 9900 + (-5) = 9895
        portfolio.mark_to_market({"SPY": 95.0}, ts_ms=3000, multiplier=1)
        # Drawdown = peak(10000) - current(9895) = 105
        # Peak equity is starting_cash (10000) since buying at 100 with mult=1
        # already put equity below the initial peak.
        assert portfolio.max_drawdown == pytest.approx(105.0)

    def test_summary(self):
        portfolio = VirtualPortfolio(starting_cash=5_000.0)
        pos = portfolio.open_position("SPY", "SHORT", 1, 2.0, 1000, multiplier=100)
        portfolio.close_position(pos, 1.5, 2000, multiplier=100)
        s = portfolio.summary()
        assert s["total_trades"] == 1
        assert s["winners"] == 1
        assert s["total_realized_pnl"] == (2.0 - 1.5) * 100

    def test_commission_tracked(self):
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        portfolio.open_position("SPY", "LONG", 1, 100.0, 1000, commission=1.30, multiplier=1)
        portfolio.close_position(
            portfolio.open_positions[0], 105.0, 2000, commission=1.30, multiplier=1
        )
        assert portfolio.total_commission == 2.60


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end replay tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReplayEndToEnd:
    def test_null_strategy_runs(self):
        bars = _make_bars(n=5)
        harness = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=NullStrategy(),
            execution_model=ExecutionModel(),
        )
        result = harness.run()
        assert result.bars_replayed == 5
        assert result.portfolio_summary["total_trades"] == 0

    def test_buy_on_second_bar_strategy(self):
        bars = _make_bars(n=6, open_price=50.0, drift=1.0)
        # Use relaxed execution config so low/high bars don't get rejected
        cfg = ExecutionConfig(max_spread_pct=5.0, slippage_per_contract=0.0)
        exec_model = ExecutionModel(cfg)
        strategy = BuyOnSecondBarStrategy()
        harness = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=strategy,
            execution_model=exec_model,
        )
        result = harness.run()
        state = result.strategy_state
        assert state["bars_seen"] == 6
        assert state["fills"] == 2  # open + close
        assert result.portfolio_summary["total_trades"] == 1

    def test_session_distribution_populated(self):
        bars = _make_bars(n=10)
        harness = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=NullStrategy(),
            execution_model=ExecutionModel(),
        )
        result = harness.run()
        # Session distribution should have entries
        assert sum(result.session_distribution.values()) == 10

    def test_replay_result_summary(self):
        bars = _make_bars(n=3)
        harness = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=NullStrategy(),
            execution_model=ExecutionModel(),
        )
        result = harness.run()
        summary = result.summary()
        assert "strategy_id" in summary
        assert "bars_replayed" in summary
        assert "portfolio" in summary
        assert "execution" in summary

    def test_bars_processed_in_order(self):
        """Verify bars are seen in ascending timestamp order."""
        bars = _make_bars(n=5, interval_ms=60_000)
        # Shuffle to test sorting
        shuffled = [bars[3], bars[0], bars[4], bars[1], bars[2]]

        seen_ts: List[int] = []

        class OrderTracker(ReplayStrategy):
            @property
            def strategy_id(self):
                return "ORDER_TRACKER"
            def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
                seen_ts.append(bar.timestamp_ms)
            def generate_intents(self, sim_ts_ms):
                return []

        harness = ReplayHarness(
            bars=shuffled,
            outcomes=[],
            strategy=OrderTracker(),
            execution_model=ExecutionModel(),
        )
        harness.run()
        assert seen_ts == sorted(seen_ts), "Bars must be processed in chronological order"


# ═══════════════════════════════════════════════════════════════════════════
# TradeIntent tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTradeIntent:
    def test_intent_creation(self):
        intent = TradeIntent(
            symbol="SPY",
            side="SELL",
            quantity=2,
            intent_type="OPEN",
            tag="test",
        )
        assert intent.symbol == "SPY"
        assert intent.side == "SELL"
        assert intent.quantity == 2
        assert intent.intent_type == "OPEN"

    def test_intent_with_quote_meta(self):
        intent = TradeIntent(
            symbol="SPY",
            side="BUY",
            quantity=1,
            intent_type="OPEN",
            meta={"quote": {"bid": 1.0, "ask": 1.1}},
        )
        assert intent.meta["quote"]["bid"] == 1.0
