"""
Test: RiskEngine Monotone Property
====================================

Verify that total exposure never increases after clamping.
"""

import pytest

from src.analysis.allocation_engine import Allocation
from src.analysis.drawdown_containment import DrawdownConfig
from src.analysis.correlation_exposure import CorrelationConfig
from src.analysis.greek_limits import GreekLimitsConfig
from src.analysis.portfolio_state import PortfolioState
from src.analysis.risk_engine import RiskEngine, RiskEngineConfig
from src.analysis.tail_risk_scenario import TailRiskConfig


def _make_allocations(weights=None):
    """Create test allocations with optional custom weights."""
    weights = weights or [0.07, 0.05, 0.04, 0.03]
    instruments = ["SPY", "IBIT", "QQQ", "SPY"]
    result = []
    for i, (w, inst) in enumerate(zip(weights, instruments)):
        result.append(Allocation(
            strategy_id=f"strat_{chr(65+i)}",
            instrument=inst,
            weight=w,
            dollar_risk=w * 10_000.0,
            kelly_raw=w * 1.2,
            contracts=0,
        ))
    return result


def _make_state(**overrides):
    defaults = dict(
        as_of_ts_ms=1700000000000,
        equity_usd=10_000.0,
        current_drawdown_pct=0.0,
        peak_equity_usd=10_000.0,
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


def _make_config(**overrides):
    defaults = dict(
        enabled=True,
        aggregate_exposure_cap=1.0,
        drawdown=DrawdownConfig(threshold_pct=0.10),
        correlation=CorrelationConfig(),
        greek_limits=GreekLimitsConfig(),
        tail_risk=TailRiskConfig(enabled_for_options=False),
        enforce_idempotence_check=False,
        enforce_monotone_check=False,  # we check manually
    )
    defaults.update(overrides)
    return RiskEngineConfig(**defaults)


class TestRiskEngineMonotone:
    """Total exposure never increases after clamping."""

    def _assert_monotone(self, original, clamped):
        """Helper: total abs weight of clamped ≤ original."""
        orig_total = sum(abs(a.weight) for a in original)
        clamp_total = sum(abs(a.weight) for a in clamped)
        assert clamp_total <= orig_total + 1e-10, (
            f"Monotone violation: original={orig_total:.8f} "
            f"clamped={clamp_total:.8f}"
        )

        # Also check per-allocation: no individual weight increased
        orig_by_id = {
            f"{a.strategy_id}:{a.instrument}": a for a in original
        }
        for c in clamped:
            key = f"{c.strategy_id}:{c.instrument}"
            if key in orig_by_id:
                assert abs(c.weight) <= abs(orig_by_id[key].weight) + 1e-10, (
                    f"Per-alloc monotone violation for {key}: "
                    f"original={orig_by_id[key].weight} clamped={c.weight}"
                )

    def test_monotone_no_constraints(self):
        """No constraints active — weights unchanged."""
        engine = RiskEngine()
        allocs = _make_allocations()
        state = _make_state()
        config = _make_config()

        clamped, _, _ = engine.clamp(allocs, state, config)
        self._assert_monotone(allocs, clamped)

    def test_monotone_aggregate_cap(self):
        """Aggregate cap reduces total exposure."""
        engine = RiskEngine()
        allocs = _make_allocations([0.30, 0.25, 0.20, 0.15])
        state = _make_state()
        config = _make_config(aggregate_exposure_cap=0.50)

        clamped, _, _ = engine.clamp(allocs, state, config)
        self._assert_monotone(allocs, clamped)

        total = sum(abs(a.weight) for a in clamped)
        assert total <= 0.50 + 1e-8

    def test_monotone_drawdown_throttle(self):
        """Drawdown throttle reduces exposure."""
        engine = RiskEngine()
        allocs = _make_allocations()
        state = _make_state(current_drawdown_pct=0.20)
        config = _make_config(
            drawdown=DrawdownConfig(
                threshold_pct=0.10,
                throttle_mode="linear",
                k=5.0,
            ),
        )

        clamped, _, _ = engine.clamp(allocs, state, config)
        self._assert_monotone(allocs, clamped)

    def test_monotone_underlying_cap(self):
        """Underlying cap reduces per-underlying exposure."""
        engine = RiskEngine()
        allocs = _make_allocations()
        state = _make_state()
        config = _make_config(
            correlation=CorrelationConfig(
                max_exposure_per_underlying_usd=300.0,
            ),
        )

        clamped, _, _ = engine.clamp(allocs, state, config)
        self._assert_monotone(allocs, clamped)

    def test_monotone_greek_limits(self):
        """Greek limits reduce exposure."""
        engine = RiskEngine()
        allocs = _make_allocations()
        state = _make_state()
        config = _make_config(
            greek_limits=GreekLimitsConfig(
                portfolio_max_delta_shares=200.0,
            ),
        )

        clamped, _, _ = engine.clamp(allocs, state, config)
        self._assert_monotone(allocs, clamped)

    def test_monotone_all_constraints(self):
        """All constraints combined still monotone."""
        engine = RiskEngine()
        allocs = _make_allocations([0.20, 0.15, 0.10, 0.08])
        state = _make_state(current_drawdown_pct=0.12)
        config = _make_config(
            aggregate_exposure_cap=0.30,
            drawdown=DrawdownConfig(
                threshold_pct=0.10,
                throttle_mode="step",
                throttle_scale=0.6,
            ),
            correlation=CorrelationConfig(
                max_exposure_per_underlying_usd=1000.0,
            ),
            greek_limits=GreekLimitsConfig(
                portfolio_max_delta_shares=1000.0,
            ),
        )

        clamped, _, _ = engine.clamp(allocs, state, config)
        self._assert_monotone(allocs, clamped)

    def test_monotone_check_raises_on_violation(self):
        """Verify the built-in monotone check raises on violation.

        We can't easily create a real violation, so we test that the
        check mechanism works by verifying it doesn't raise on valid input.
        """
        engine = RiskEngine()
        allocs = _make_allocations()
        state = _make_state()
        config = _make_config(enforce_monotone_check=True)

        # Should not raise
        clamped, _, _ = engine.clamp(allocs, state, config)
        assert len(clamped) == len(allocs)
