"""
Test: RiskEngine Idempotence
=============================

Verify that clamp(clamp(x)) == clamp(x) for all constraint combinations.
"""

import pytest

from src.analysis.allocation_engine import Allocation
from src.analysis.drawdown_containment import DrawdownConfig
from src.analysis.correlation_exposure import CorrelationConfig
from src.analysis.greek_limits import GreekLimitsConfig
from src.analysis.portfolio_state import PortfolioState
from src.analysis.risk_engine import RiskEngine, RiskEngineConfig
from src.analysis.tail_risk_scenario import TailRiskConfig


def _make_allocations():
    """Create a set of test allocations."""
    return [
        Allocation(
            strategy_id="strat_A",
            instrument="SPY",
            weight=0.07,
            dollar_risk=700.0,
            kelly_raw=0.09,
            vol_adjusted=True,
            contracts=0,
        ),
        Allocation(
            strategy_id="strat_B",
            instrument="IBIT",
            weight=0.05,
            dollar_risk=500.0,
            kelly_raw=0.06,
            vol_adjusted=False,
            contracts=3,
        ),
        Allocation(
            strategy_id="strat_C",
            instrument="SPY",
            weight=0.03,
            dollar_risk=300.0,
            kelly_raw=0.04,
            vol_adjusted=True,
            contracts=0,
        ),
    ]


def _make_state():
    """Create a test portfolio state."""
    return PortfolioState(
        as_of_ts_ms=1700000000000,
        equity_usd=10_000.0,
        current_drawdown_pct=0.0,
        peak_equity_usd=10_000.0,
    )


def _make_config(**overrides):
    """Create a risk engine config with optional overrides."""
    defaults = dict(
        enabled=True,
        aggregate_exposure_cap=1.0,
        drawdown=DrawdownConfig(threshold_pct=0.10),
        correlation=CorrelationConfig(),
        greek_limits=GreekLimitsConfig(),
        tail_risk=TailRiskConfig(enabled_for_options=False),
        enforce_idempotence_check=False,  # disable in tests to avoid recursion
        enforce_monotone_check=True,
    )
    defaults.update(overrides)
    return RiskEngineConfig(**defaults)


class TestRiskEngineIdempotence:
    """clamp(clamp(x)) == clamp(x)"""

    def test_idempotence_no_constraints_active(self):
        """No constraints triggered — allocations pass through unchanged."""
        engine = RiskEngine()
        allocs = _make_allocations()
        state = _make_state()
        config = _make_config()

        first, _, _ = engine.clamp(allocs, state, config)
        second, _, _ = engine.clamp(first, state, config)

        for a, b in zip(
            sorted(first, key=lambda x: x.strategy_id),
            sorted(second, key=lambda x: x.strategy_id),
        ):
            assert abs(a.weight - b.weight) < 1e-8, (
                f"Weight mismatch for {a.strategy_id}: {a.weight} vs {b.weight}"
            )
            assert a.contracts == b.contracts, (
                f"Contracts mismatch for {a.strategy_id}: {a.contracts} vs {b.contracts}"
            )

    def test_idempotence_with_aggregate_cap(self):
        """Aggregate cap triggered — second pass should not change anything."""
        engine = RiskEngine()
        allocs = _make_allocations()
        state = _make_state()
        config = _make_config(aggregate_exposure_cap=0.10)

        first, reasons1, _ = engine.clamp(allocs, state, config)
        assert len(reasons1) > 0, "Expected clamp reasons from aggregate cap"

        second, reasons2, _ = engine.clamp(first, state, config)

        # Second pass should produce NO new clamp actions (already within limits)
        # (Some info-level reasons may still appear but weights should not change)
        for a, b in zip(
            sorted(first, key=lambda x: x.strategy_id),
            sorted(second, key=lambda x: x.strategy_id),
        ):
            assert abs(a.weight - b.weight) < 1e-8, (
                f"Idempotence violation for {a.strategy_id}: "
                f"first={a.weight} second={b.weight}"
            )

    def test_idempotence_with_drawdown_throttle(self):
        """Drawdown throttle active — second pass should be stable.

        The drawdown throttle works as a tighter aggregate cap:
        effective_cap = aggregate_cap * throttle.  For throttling to
        actually trigger, total exposure must exceed this effective cap.
        """
        engine = RiskEngine()
        # Use higher weights so total (0.90) exceeds effective_cap (0.75)
        allocs = [
            Allocation(
                strategy_id="strat_A", instrument="SPY",
                weight=0.40, dollar_risk=4000.0,
                kelly_raw=0.50, vol_adjusted=True, contracts=0,
            ),
            Allocation(
                strategy_id="strat_B", instrument="IBIT",
                weight=0.30, dollar_risk=3000.0,
                kelly_raw=0.35, vol_adjusted=False, contracts=0,
            ),
            Allocation(
                strategy_id="strat_C", instrument="QQQ",
                weight=0.20, dollar_risk=2000.0,
                kelly_raw=0.25, vol_adjusted=True, contracts=0,
            ),
        ]
        state = PortfolioState(
            as_of_ts_ms=1700000000000,
            equity_usd=10_000.0,
            current_drawdown_pct=0.15,
            peak_equity_usd=11_765.0,
        )
        config = _make_config(
            aggregate_exposure_cap=1.0,
            drawdown=DrawdownConfig(
                threshold_pct=0.10,
                throttle_mode="linear",
                k=5.0,
            ),
        )
        # throttle = 1 - 5*(0.15-0.10) = 0.75
        # effective_cap = 1.0 * 0.75 = 0.75
        # total = 0.90 > 0.75, so scale = 0.75/0.90 = 0.8333

        first, reasons1, _ = engine.clamp(allocs, state, config)
        assert any(r.constraint_id == "drawdown_throttle" for r in reasons1), \
            "Expected drawdown throttle to fire"

        second, _, _ = engine.clamp(first, state, config)

        for a, b in zip(
            sorted(first, key=lambda x: x.strategy_id),
            sorted(second, key=lambda x: x.strategy_id),
        ):
            assert abs(a.weight - b.weight) < 1e-8, (
                f"Idempotence violation with drawdown for {a.strategy_id}: "
                f"first={a.weight} second={b.weight}"
            )

    def test_idempotence_with_underlying_cap(self):
        """Underlying exposure cap — second pass stable."""
        engine = RiskEngine()
        allocs = _make_allocations()
        state = _make_state()
        config = _make_config(
            correlation=CorrelationConfig(
                max_exposure_per_underlying_usd=500.0,
            ),
        )

        first, _, _ = engine.clamp(allocs, state, config)
        second, _, _ = engine.clamp(first, state, config)

        for a, b in zip(
            sorted(first, key=lambda x: x.strategy_id),
            sorted(second, key=lambda x: x.strategy_id),
        ):
            assert abs(a.weight - b.weight) < 1e-8, (
                f"Idempotence violation with underlying cap for {a.strategy_id}"
            )

    def test_idempotence_with_all_constraints(self):
        """All constraints active simultaneously."""
        engine = RiskEngine()
        allocs = _make_allocations()
        state = PortfolioState(
            as_of_ts_ms=1700000000000,
            equity_usd=10_000.0,
            current_drawdown_pct=0.12,
            peak_equity_usd=11_364.0,
        )
        config = _make_config(
            aggregate_exposure_cap=0.12,
            drawdown=DrawdownConfig(
                threshold_pct=0.10,
                throttle_mode="step",
                throttle_scale=0.5,
            ),
            correlation=CorrelationConfig(
                max_exposure_per_underlying_usd=400.0,
            ),
            greek_limits=GreekLimitsConfig(
                portfolio_max_delta_shares=5000.0,
            ),
        )

        first, _, _ = engine.clamp(allocs, state, config)
        second, _, _ = engine.clamp(first, state, config)

        for a, b in zip(
            sorted(first, key=lambda x: x.strategy_id),
            sorted(second, key=lambda x: x.strategy_id),
        ):
            assert abs(a.weight - b.weight) < 1e-8, (
                f"Idempotence violation (all constraints) for {a.strategy_id}: "
                f"first={a.weight} second={b.weight}"
            )
            assert a.contracts == b.contracts
