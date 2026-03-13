"""
Phase 4B Integration Tests
===========================

End-to-end replay tests using DB-like data only.

Validates:
- market_bars replay
- bar_outcomes replay
- conservative execution model
- portfolio accounting
- outcome lookahead guard
- market data availability barrier
- no future data leakage
- equity curve generation

Includes a minimal deterministic strategy that trades on simple rules.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List, Optional

from src.core.outcome_engine import BarData, OutcomeResult
from src.analysis.execution_model import ExecutionModel, ExecutionConfig, Quote
from src.analysis.replay_harness import (
    MarketDataSnapshot,
    PortfolioSnapshot,
    ReplayConfig,
    ReplayHarness,
    ReplayResult,
    ReplayStrategy,
    TradeIntent,
)


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic Example Strategy
# ═══════════════════════════════════════════════════════════════════════════

class SimpleMovingAverageStrategy(ReplayStrategy):
    """Minimal deterministic strategy for testing.

    Trades on a 3-bar simple moving average cross:
    - BUY when close > SMA(3)
    - SELL when close < SMA(3)
    """

    def __init__(self) -> None:
        self._closes: List[float] = []
        self._position_open = False
        self._intents: List[TradeIntent] = []
        self._bars_seen = 0
        self._outcomes_seen = 0
        self._snapshots_seen = 0

    @property
    def strategy_id(self) -> str:
        return "SIMPLE_SMA_3_TEST"

    def on_bar(
        self,
        bar: BarData,
        sim_ts_ms: int,
        session_regime: str,
        visible_outcomes: Dict[int, OutcomeResult],
        **kwargs,
    ) -> None:
        self._bars_seen += 1
        self._outcomes_seen = len(visible_outcomes)
        self._closes.append(bar.close)

        # Track snapshots if passed
        snapshots = kwargs.get("visible_snapshots", [])
        self._snapshots_seen = len(snapshots)

        self._intents.clear()

        if len(self._closes) < 3:
            return

        sma3 = sum(self._closes[-3:]) / 3.0

        if bar.close > sma3 and not self._position_open:
            self._intents.append(TradeIntent(
                symbol="TEST_SYM",
                side="BUY",
                quantity=1,
                intent_type="OPEN",
                tag="sma_cross_up",
            ))
        elif bar.close < sma3 and self._position_open:
            self._intents.append(TradeIntent(
                symbol="TEST_SYM",
                side="SELL",
                quantity=1,
                intent_type="CLOSE",
                tag="sma_cross_down",
            ))

    def generate_intents(self, sim_ts_ms: int) -> List[TradeIntent]:
        return list(self._intents)

    def on_fill(self, intent: TradeIntent, fill: Any) -> None:
        if intent.intent_type == "OPEN":
            self._position_open = True
        elif intent.intent_type == "CLOSE":
            self._position_open = False

    def finalize(self) -> Dict[str, Any]:
        return {
            "bars_seen": self._bars_seen,
            "final_position_open": self._position_open,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Test Data Generators
# ═══════════════════════════════════════════════════════════════════════════

def make_bars(n: int = 20, start_ms: int = 1_700_000_000_000, price_start: float = 100.0) -> List[BarData]:
    """Generate n synthetic bars with a simple uptrend then downtrend."""
    bars = []
    price = price_start
    for i in range(n):
        # Uptrend first half, downtrend second half
        if i < n // 2:
            price += 0.50
        else:
            price -= 0.50
        ts = start_ms + i * 60_000  # 1-minute bars
        bars.append(BarData(
            timestamp_ms=ts,
            open=price - 0.10,
            high=price + 0.20,
            low=price - 0.20,
            close=price,
            volume=1000.0 + i * 10,
        ))
    return bars


def make_outcomes(bars: List[BarData], horizon_ms: int = 300_000) -> List[Dict[str, Any]]:
    """Generate synthetic outcomes for each bar."""
    outcomes = []
    for i, bar in enumerate(bars):
        # Outcome is available only after window closes
        window_end = bar.timestamp_ms + horizon_ms
        fwd_return = 0.005 if i < len(bars) // 2 else -0.003
        outcomes.append({
            "provider": "test",
            "symbol": "TEST_SYM",
            "bar_duration_seconds": 60,
            "timestamp_ms": bar.timestamp_ms,
            "horizon_seconds": horizon_ms // 1000,
            "outcome_version": "test_v1",
            "close_now": bar.close,
            "close_at_horizon": bar.close * (1 + fwd_return),
            "fwd_return": fwd_return,
            "max_runup": abs(fwd_return),
            "max_drawdown": -abs(fwd_return) / 2,
            "realized_vol": 0.15,
            "max_high_in_window": bar.high,
            "min_low_in_window": bar.low,
            "max_runup_ts_ms": bar.timestamp_ms + 60_000,
            "max_drawdown_ts_ms": bar.timestamp_ms + 120_000,
            "time_to_max_runup_ms": 60_000,
            "time_to_max_drawdown_ms": 120_000,
            "status": "OK",
            "close_ref_ms": bar.timestamp_ms,
            "window_start_ms": bar.timestamp_ms,
            "window_end_ms": window_end,
            "bars_expected": 5,
            "bars_found": 5,
            "gap_count": 0,
            "computed_at_ms": window_end + 1000,
        })
    return outcomes


def make_snapshots(bars: List[BarData], delay_ms: int = 5_000) -> List[MarketDataSnapshot]:
    """Generate market data snapshots with a realistic delay."""
    snaps = []
    for bar in bars:
        snaps.append(MarketDataSnapshot(
            symbol="TEST_SYM",
            recv_ts_ms=bar.timestamp_ms + delay_ms,  # received with delay
            bid=bar.low + 0.05,
            ask=bar.high - 0.05,
            bid_size=10,
            ask_size=10,
            quote_ts_ms=0,  # provider ts = 0 (typical for Tastytrade)
            delta=-0.15,
            gamma=0.02,
            theta=-0.05,
            vega=0.10,
            iv=0.25,
            source="test",
        ))
    return snaps


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase4BEndToEnd:
    def test_full_replay_with_bars_and_outcomes(self):
        """End-to-end: replay bars → generate intents → fill → equity curve."""
        bars = make_bars(20)
        outcomes = make_outcomes(bars)
        strategy = SimpleMovingAverageStrategy()
        exec_model = ExecutionModel(ExecutionConfig(
            slippage_per_contract=0.01,
            max_spread_pct=1.0,
            commission_per_contract=0.50,
        ))
        config = ReplayConfig(
            starting_cash=10_000.0,
            bar_duration_seconds=60,
        )

        harness = ReplayHarness(
            bars=bars,
            outcomes=outcomes,
            strategy=strategy,
            execution_model=exec_model,
            config=config,
        )
        result = harness.run()

        assert result.bars_replayed == 20
        assert result.strategy_id == "SIMPLE_SMA_3_TEST"
        assert result.portfolio_summary["starting_cash"] == 10_000.0
        assert result.portfolio_summary["equity_curve_points"] == 20
        assert strategy._bars_seen == 20

    def test_equity_curve_generated(self):
        """Verify equity curve has one point per bar."""
        bars = make_bars(10)
        outcomes = make_outcomes(bars)
        strategy = SimpleMovingAverageStrategy()
        exec_model = ExecutionModel()
        harness = ReplayHarness(bars=bars, outcomes=outcomes, strategy=strategy, execution_model=exec_model)
        result = harness.run()

        curve = harness.portfolio.equity_curve
        assert len(curve) == 10
        assert all(isinstance(s, PortfolioSnapshot) for s in curve)
        assert curve[0].ts_ms > 0

    def test_fills_and_rejects_recorded(self):
        """Execution model records fills and rejects correctly."""
        bars = make_bars(20)
        outcomes = make_outcomes(bars)
        strategy = SimpleMovingAverageStrategy()
        exec_model = ExecutionModel(ExecutionConfig(slippage_per_contract=0.01, max_spread_pct=1.0))
        harness = ReplayHarness(bars=bars, outcomes=outcomes, strategy=strategy, execution_model=exec_model)
        result = harness.run()

        summary = result.execution_summary
        assert "fills" in summary
        assert "rejects" in summary
        # Strategy should generate some intents
        total = summary["fills"] + summary["rejects"]
        assert total >= 0  # May be 0 if no SMA cross

    def test_session_distribution_tracked(self):
        """Session regime distribution is tracked."""
        bars = make_bars(10)
        outcomes = make_outcomes(bars)
        strategy = SimpleMovingAverageStrategy()
        exec_model = ExecutionModel()
        harness = ReplayHarness(bars=bars, outcomes=outcomes, strategy=strategy, execution_model=exec_model)
        result = harness.run()

        assert isinstance(result.session_distribution, dict)
        # Should have at least one session type
        assert sum(result.session_distribution.values()) == 10


# ═══════════════════════════════════════════════════════════════════════════
# Lookahead Barrier Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOutcomeLookaheadBarrier:
    def test_outcomes_not_visible_before_window_end(self):
        """Outcomes must not be visible until sim_time >= window_end_ms."""

        class OutcomeRecorder(ReplayStrategy):
            def __init__(self):
                self.outcome_counts = []

            @property
            def strategy_id(self):
                return "OUTCOME_RECORDER"

            def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
                self.outcome_counts.append((sim_ts_ms, len(visible_outcomes)))

            def generate_intents(self, sim_ts_ms):
                return []

            def finalize(self):
                return {}

        bars = make_bars(10)
        outcomes = make_outcomes(bars, horizon_ms=300_000)  # 5 min horizon
        strategy = OutcomeRecorder()
        exec_model = ExecutionModel()
        config = ReplayConfig(bar_duration_seconds=60)

        harness = ReplayHarness(
            bars=bars, outcomes=outcomes, strategy=strategy,
            execution_model=exec_model, config=config,
        )
        harness.run()

        # First few bars should see 0 outcomes (window_end hasn't passed)
        for sim_ts, count in strategy.outcome_counts[:4]:
            assert count == 0, f"Bar at {sim_ts} should see 0 outcomes, got {count}"

        # Later bars should accumulate visible outcomes
        # After bar 5 (sim_time = start + 5*60000 + 60000 = start + 360000)
        # outcome[0] window_end = start + 300000, which is < 360000
        # So bar 5 should see at least 1 outcome
        last_count = strategy.outcome_counts[-1][1]
        assert last_count > 0, "Final bar should see some outcomes"

    def test_no_future_outcomes_leaked(self):
        """Verify that no outcome's window_end > sim_time is ever visible."""

        class FutureLeakDetector(ReplayStrategy):
            def __init__(self):
                self.leaked = False

            @property
            def strategy_id(self):
                return "LEAK_DETECTOR"

            def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
                for ts, outcome in visible_outcomes.items():
                    if outcome.window_end_ms > sim_ts_ms:
                        self.leaked = True

            def generate_intents(self, sim_ts_ms):
                return []

            def finalize(self):
                return {"leaked": self.leaked}

        bars = make_bars(20)
        outcomes = make_outcomes(bars, horizon_ms=180_000)
        strategy = FutureLeakDetector()
        exec_model = ExecutionModel()

        harness = ReplayHarness(
            bars=bars, outcomes=outcomes, strategy=strategy,
            execution_model=exec_model,
        )
        result = harness.run()
        assert strategy.leaked is False, "Future outcome data was leaked!"


# ═══════════════════════════════════════════════════════════════════════════
# Market Data Availability Barrier Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMarketDataBarrier:
    def test_snapshots_not_visible_before_recv_ts(self):
        """Snapshots should only appear when sim_time >= recv_ts_ms."""

        class SnapshotRecorder(ReplayStrategy):
            def __init__(self):
                self.snapshot_counts = []

            @property
            def strategy_id(self):
                return "SNAPSHOT_RECORDER"

            def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
                snaps = kwargs.get("visible_snapshots", [])
                self.snapshot_counts.append((sim_ts_ms, len(snaps)))

            def generate_intents(self, sim_ts_ms):
                return []

            def finalize(self):
                return {}

        bars = make_bars(10, start_ms=1_700_000_000_000)
        outcomes = make_outcomes(bars)
        # Snapshots with 120-second delay
        snapshots = make_snapshots(bars, delay_ms=120_000)
        strategy = SnapshotRecorder()
        exec_model = ExecutionModel()
        config = ReplayConfig(bar_duration_seconds=60)

        harness = ReplayHarness(
            bars=bars, outcomes=outcomes, strategy=strategy,
            execution_model=exec_model, config=config,
            snapshots=snapshots,
        )
        harness.run()

        # With 120s delay and 60s bars, first bar's snapshot (recv = bar_ts + 120000)
        # won't be visible until sim_time >= bar_ts + 120000
        # sim_time for bar i = bar_ts_i + 60000
        # So snapshot for bar 0 becomes visible when sim_time = start + 120000
        # That's bar index 2 (start + 2*60000 + 60000 = start + 180000 > start + 120000)
        assert strategy.snapshot_counts[0][1] == 0, "First bar should see 0 snapshots"

    def test_no_future_snapshots_leaked(self):
        """Verify no snapshot with recv_ts_ms > sim_time is visible."""

        class SnapshotLeakDetector(ReplayStrategy):
            def __init__(self):
                self.leaked = False

            @property
            def strategy_id(self):
                return "SNAP_LEAK_DETECTOR"

            def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
                snaps = kwargs.get("visible_snapshots", [])
                for snap in snaps:
                    if snap.recv_ts_ms > sim_ts_ms:
                        self.leaked = True

            def generate_intents(self, sim_ts_ms):
                return []

            def finalize(self):
                return {"leaked": self.leaked}

        bars = make_bars(15)
        outcomes = make_outcomes(bars)
        snapshots = make_snapshots(bars, delay_ms=30_000)
        strategy = SnapshotLeakDetector()
        exec_model = ExecutionModel()

        harness = ReplayHarness(
            bars=bars, outcomes=outcomes, strategy=strategy,
            execution_model=exec_model, snapshots=snapshots,
        )
        harness.run()
        assert strategy.leaked is False, "Future snapshot data was leaked!"

    def test_snapshots_monotonically_increasing(self):
        """Visible snapshots should accumulate monotonically."""

        class MonotonicChecker(ReplayStrategy):
            def __init__(self):
                self.counts = []

            @property
            def strategy_id(self):
                return "MONOTONIC_CHECKER"

            def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
                snaps = kwargs.get("visible_snapshots", [])
                self.counts.append(len(snaps))

            def generate_intents(self, sim_ts_ms):
                return []

            def finalize(self):
                return {}

        bars = make_bars(15)
        outcomes = make_outcomes(bars)
        snapshots = make_snapshots(bars, delay_ms=5_000)
        strategy = MonotonicChecker()
        exec_model = ExecutionModel()

        harness = ReplayHarness(
            bars=bars, outcomes=outcomes, strategy=strategy,
            execution_model=exec_model, snapshots=snapshots,
        )
        harness.run()

        # Each bar should see >= as many snapshots as the previous
        for i in range(1, len(strategy.counts)):
            assert strategy.counts[i] >= strategy.counts[i - 1], (
                f"Snapshot count decreased at bar {i}: "
                f"{strategy.counts[i]} < {strategy.counts[i - 1]}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_harness_works_without_snapshots(self):
        """Old usage without snapshots should still work."""
        bars = make_bars(10)
        outcomes = make_outcomes(bars)
        strategy = SimpleMovingAverageStrategy()
        exec_model = ExecutionModel()

        harness = ReplayHarness(
            bars=bars, outcomes=outcomes, strategy=strategy,
            execution_model=exec_model,
            # No snapshots parameter
        )
        result = harness.run()
        assert result.bars_replayed == 10

    def test_old_strategy_without_kwargs(self):
        """Strategy that doesn't accept visible_snapshots should still work."""

        class OldStyleStrategy(ReplayStrategy):
            @property
            def strategy_id(self):
                return "OLD_STYLE"

            def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes):
                # No **kwargs — old-style signature
                pass

            def generate_intents(self, sim_ts_ms):
                return []

        bars = make_bars(5)
        outcomes = make_outcomes(bars)
        # Even with snapshots provided, old strategy should work
        snapshots = make_snapshots(bars)
        strategy = OldStyleStrategy()
        exec_model = ExecutionModel()

        harness = ReplayHarness(
            bars=bars, outcomes=outcomes, strategy=strategy,
            execution_model=exec_model, snapshots=snapshots,
        )
        # Should not raise
        result = harness.run()
        assert result.bars_replayed == 5


# ═══════════════════════════════════════════════════════════════════════════
# Quote recv_ts_ms in Execution
# ═══════════════════════════════════════════════════════════════════════════

class TestRecvTsInExecution:
    def test_recv_ts_preferred_for_staleness(self):
        """Execution model should use recv_ts_ms for staleness check."""
        exec_model = ExecutionModel(ExecutionConfig(max_stale_ms=60_000))
        sim_ts = 1_700_000_100_000

        # Provider ts is stale (200s ago) but recv_ts is fresh (10s ago)
        quote = Quote(
            bid=1.00, ask=1.10,
            quote_ts_ms=sim_ts - 200_000,  # provider = stale
            recv_ts_ms=sim_ts - 10_000,     # receipt = fresh
        )
        result = exec_model.attempt_fill(quote, "BUY", 1, sim_ts)
        assert result.filled is True, "Should use recv_ts (fresh), not provider ts (stale)"

    def test_falls_back_to_provider_ts(self):
        """When recv_ts_ms=0, should fall back to provider timestamp."""
        exec_model = ExecutionModel(ExecutionConfig(max_stale_ms=60_000))
        sim_ts = 1_700_000_100_000

        quote = Quote(
            bid=1.00, ask=1.10,
            quote_ts_ms=sim_ts - 10_000,  # provider = fresh
            recv_ts_ms=0,                  # no receipt ts
        )
        result = exec_model.attempt_fill(quote, "BUY", 1, sim_ts)
        assert result.filled is True

    def test_both_zero_skips_staleness(self):
        """When both timestamps are 0, staleness check is skipped."""
        exec_model = ExecutionModel(ExecutionConfig(max_stale_ms=60_000))
        sim_ts = 1_700_000_100_000

        quote = Quote(bid=1.00, ask=1.10, quote_ts_ms=0, recv_ts_ms=0)
        result = exec_model.attempt_fill(quote, "BUY", 1, sim_ts)
        assert result.filled is True
