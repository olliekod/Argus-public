"""
Tests for ExecutionModel reset at start of replay run() (10.7).

Verifies:
- Replay harness calls execution_model.reset() at run() start
- Two consecutive runs produce identical metrics
- Ledger starts clean each run (no leakage between runs)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Direct file-level imports to bypass src.analysis.__init__ (which pulls
# yfinance via trade_calculator → ibit_options_client).
_ROOT = Path(__file__).resolve().parent.parent

def _import_from_file(module_name: str, file_path: Path):
    """Import a module directly from its file path, bypassing __init__."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_exec_mod = _import_from_file(
    "src.analysis.execution_model",
    _ROOT / "src" / "analysis" / "execution_model.py",
)
ExecutionConfig = _exec_mod.ExecutionConfig
ExecutionModel = _exec_mod.ExecutionModel
Quote = _exec_mod.Quote

_replay_mod = _import_from_file(
    "src.analysis.replay_harness",
    _ROOT / "src" / "analysis" / "replay_harness.py",
)
ReplayConfig = _replay_mod.ReplayConfig
ReplayHarness = _replay_mod.ReplayHarness
ReplayStrategy = _replay_mod.ReplayStrategy
TradeIntent = _replay_mod.TradeIntent

from src.core.outcome_engine import BarData


# ═══════════════════════════════════════════════════════════════════════════
# Test strategy that generates predictable trades
# ═══════════════════════════════════════════════════════════════════════════

class DeterministicTrader(ReplayStrategy):
    """Buys on bar 2, sells on bar 4. Predictable for testing."""

    def __init__(self):
        self._bar_count = 0
        self._holding = False

    @property
    def strategy_id(self) -> str:
        return "DETERMINISTIC_TRADER"

    def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
        self._bar_count += 1

    def generate_intents(self, sim_ts_ms):
        if self._bar_count == 2 and not self._holding:
            self._holding = True
            return [TradeIntent(
                symbol="TEST", side="BUY", quantity=1,
                intent_type="OPEN", tag="entry",
            )]
        if self._bar_count == 4 and self._holding:
            self._holding = False
            return [TradeIntent(
                symbol="TEST", side="SELL", quantity=1,
                intent_type="CLOSE", tag="exit",
            )]
        return []

    def finalize(self):
        return {"bars_seen": self._bar_count}


def _make_bars(n: int = 6) -> List[BarData]:
    """Generate synthetic bars."""
    bars = []
    price = 100.0
    for i in range(n):
        ts = 1_700_000_000_000 + i * 60_000
        bars.append(BarData(
            timestamp_ms=ts,
            open=round(price, 2),
            high=round(price + 1.0, 2),
            low=round(price - 0.5, 2),
            close=round(price + 0.5, 2),
            volume=1000.0,
        ))
        price += 0.5
    return bars


class TestReplayReset:
    """Verify execution_model.reset() is called at replay start."""

    def test_two_runs_identical_metrics(self):
        """Running the same replay twice produces identical results."""
        bars = _make_bars(6)
        cfg = ExecutionConfig(max_spread_pct=5.0, slippage_per_contract=0.0)
        exec_model = ExecutionModel(cfg)

        # Run 1
        strategy1 = DeterministicTrader()
        harness1 = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=strategy1,
            execution_model=exec_model,
        )
        result1 = harness1.run()

        # Run 2 — same exec_model instance, fresh strategy
        strategy2 = DeterministicTrader()
        harness2 = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=strategy2,
            execution_model=exec_model,
        )
        result2 = harness2.run()

        # Both runs should produce identical execution summaries
        assert result1.execution_summary == result2.execution_summary
        assert result1.portfolio_summary["total_trades"] == result2.portfolio_summary["total_trades"]
        assert result1.portfolio_summary["total_realized_pnl"] == result2.portfolio_summary["total_realized_pnl"]

    def test_ledger_starts_clean_each_run(self):
        """Ledger should be empty at the start of each run()."""
        bars = _make_bars(6)
        cfg = ExecutionConfig(max_spread_pct=5.0, slippage_per_contract=0.0)
        exec_model = ExecutionModel(cfg)

        # Run 1 — generates some fills
        strategy1 = DeterministicTrader()
        harness1 = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=strategy1,
            execution_model=exec_model,
        )
        result1 = harness1.run()
        fills_after_run1 = result1.execution_summary["fills"]
        assert fills_after_run1 > 0, "Run 1 should have fills"

        # Run 2 — the ledger should NOT carry over run1's fills
        strategy2 = DeterministicTrader()
        harness2 = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=strategy2,
            execution_model=exec_model,
        )
        result2 = harness2.run()
        fills_after_run2 = result2.execution_summary["fills"]

        assert fills_after_run2 == fills_after_run1, \
            f"Run2 fills ({fills_after_run2}) should equal Run1 ({fills_after_run1}), not accumulate"

    def test_no_accumulated_rejects(self):
        """Rejects from run 1 should not appear in run 2's ledger."""
        bars = _make_bars(6)
        # Tight spread filter will reject some fills
        cfg = ExecutionConfig(max_spread_pct=0.001, slippage_per_contract=0.0)
        exec_model = ExecutionModel(cfg)

        strategy1 = DeterministicTrader()
        harness1 = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=strategy1,
            execution_model=exec_model,
        )
        result1 = harness1.run()
        rejects1 = result1.execution_summary["rejects"]

        strategy2 = DeterministicTrader()
        harness2 = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=strategy2,
            execution_model=exec_model,
        )
        result2 = harness2.run()
        rejects2 = result2.execution_summary["rejects"]

        assert rejects2 == rejects1, "Rejects should not accumulate across runs"
