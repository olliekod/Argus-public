"""
Tests for Allocation Engine
==============================

Verifies:
- Per-play cap enforcement (7% max per strategy)
- Aggregate exposure cap
- Zero allocation for edge-less strategies
- Correct contract computation for options
- Equity update
"""

from __future__ import annotations

import pytest

from src.analysis.allocation_engine import (
    Allocation,
    AllocationConfig,
    AllocationEngine,
)
from src.analysis.sizing import Forecast


class TestAllocationEngine:
    def _make_forecast(
        self,
        strategy_id: str = "test",
        mu: float = 0.005,
        sigma: float = 0.15,
        cost: float = 0.001,
        confidence: float = 1.0,
    ) -> Forecast:
        return Forecast(
            strategy_id=strategy_id,
            instrument="SPY",
            mu=mu,
            sigma=sigma,
            cost=cost,
            confidence=confidence,
        )

    def test_single_forecast(self):
        engine = AllocationEngine(equity=100_000.0)
        forecasts = [self._make_forecast()]
        allocs = engine.allocate(forecasts)
        assert len(allocs) == 1
        assert allocs[0].strategy_id == "test"
        assert allocs[0].weight > 0

    def test_per_play_cap(self):
        """No allocation should exceed per_play_cap."""
        config = AllocationConfig(
            kelly_fraction=0.50,  # aggressive
            per_play_cap=0.07,
        )
        engine = AllocationEngine(config=config, equity=100_000.0)
        forecasts = [
            self._make_forecast("s1", mu=0.10, sigma=0.10),
            self._make_forecast("s2", mu=0.20, sigma=0.10),
        ]
        allocs = engine.allocate(forecasts)
        for a in allocs:
            assert abs(a.weight) <= 0.07 + 1e-8

    def test_aggregate_cap(self):
        """Sum of absolute weights should not exceed aggregate cap."""
        config = AllocationConfig(
            kelly_fraction=0.50,
            per_play_cap=0.07,
            aggregate_exposure_cap=0.10,  # tight cap
        )
        engine = AllocationEngine(config=config, equity=100_000.0)
        forecasts = [
            self._make_forecast("s1", mu=0.05, sigma=0.10),
            self._make_forecast("s2", mu=0.05, sigma=0.10),
            self._make_forecast("s3", mu=0.05, sigma=0.10),
        ]
        allocs = engine.allocate(forecasts)
        total = sum(abs(a.weight) for a in allocs)
        assert total <= 0.10 + 1e-6

    def test_zero_edge_strategy(self):
        """Strategy with mu <= cost should get 0 allocation."""
        engine = AllocationEngine(equity=100_000.0)
        forecasts = [
            self._make_forecast("edge", mu=0.005, sigma=0.15, cost=0.001),
            self._make_forecast("no_edge", mu=0.001, sigma=0.15, cost=0.001),
        ]
        allocs = engine.allocate(forecasts)
        for a in allocs:
            if a.strategy_id == "no_edge":
                assert a.weight == 0.0

    def test_contracts_computation(self):
        """Options strategies should get contract counts."""
        engine = AllocationEngine(equity=100_000.0)
        forecasts = [self._make_forecast("opt_strat", mu=0.005, sigma=0.15)]
        max_loss = {"opt_strat": 300.0}
        allocs = engine.allocate(forecasts, max_loss_per_contract=max_loss)
        assert len(allocs) == 1
        assert allocs[0].contracts >= 0

    def test_update_equity(self):
        engine = AllocationEngine(equity=100_000.0)
        assert engine.equity == 100_000.0
        engine.update_equity(150_000.0)
        assert engine.equity == 150_000.0

    def test_summary(self):
        engine = AllocationEngine(equity=100_000.0)
        forecasts = [self._make_forecast()]
        allocs = engine.allocate(forecasts)
        summary = engine.summary(allocs)
        assert "equity" in summary
        assert "n_active" in summary
        assert "gross_exposure" in summary
        assert "allocations" in summary

    def test_empty_forecasts(self):
        engine = AllocationEngine(equity=100_000.0)
        allocs = engine.allocate([])
        assert allocs == []

    def test_multiple_instruments(self):
        engine = AllocationEngine(equity=100_000.0)
        forecasts = [
            Forecast("s1", "SPY", mu=0.003, sigma=0.12, cost=0.001),
            Forecast("s2", "QQQ", mu=0.004, sigma=0.15, cost=0.001),
            Forecast("s3", "IBIT", mu=0.005, sigma=0.25, cost=0.002),
        ]
        allocs = engine.allocate(forecasts)
        assert len(allocs) == 3
        for a in allocs:
            assert abs(a.weight) <= 0.07 + 1e-8
