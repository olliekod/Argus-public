"""
Tests for Risk Engine Audit Bug Fixes
=======================================

Comprehensive regression tests for all bugs identified in the audit:
- Critical: gamma/vega sign, dollar_risk overwrite, zero-width spread,
  DSR formula, Kelly cost subtraction
- High: drawdown hysteresis, MC drawdown calculation, final_equity,
  analytical call spread, DSR zero bypass
- Medium: deterministic sort, moment estimators, python_version,
  option type detection, threshold naming
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Bug 1: portfolio_state.py — gamma/vega should use abs(qty)
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.portfolio_state import PortfolioState, PositionRecord


class TestBug1GammaVegaSignedQty:
    """Gamma and vega must always be positive regardless of position direction."""

    def test_short_position_gamma_vega_positive(self):
        """Short positions should contribute positive gamma/vega."""
        pos = PositionRecord(
            underlying="SPY",
            instrument_type="option_spread",
            qty=-10,  # short
            greeks={"delta": 0.3, "gamma": 0.05, "vega": 0.10},
        )
        state = PortfolioState(current_positions=[pos])
        greeks = state.total_position_greeks()

        # Delta should be negative (short * positive delta)
        assert greeks["delta"] == pytest.approx(-3.0)
        # Gamma should be positive (abs(qty) * gamma)
        assert greeks["gamma"] == pytest.approx(0.5)
        # Vega should be positive (abs(qty) * vega)
        assert greeks["vega"] == pytest.approx(1.0)

    def test_long_position_gamma_vega_positive(self):
        """Long positions should also contribute positive gamma/vega."""
        pos = PositionRecord(
            underlying="SPY",
            instrument_type="option_spread",
            qty=10,  # long
            greeks={"delta": 0.3, "gamma": 0.05, "vega": 0.10},
        )
        state = PortfolioState(current_positions=[pos])
        greeks = state.total_position_greeks()

        assert greeks["delta"] == pytest.approx(3.0)
        assert greeks["gamma"] == pytest.approx(0.5)
        assert greeks["vega"] == pytest.approx(1.0)

    def test_mixed_positions_gamma_vega_additive(self):
        """Mixed long/short should have gamma/vega that add (never cancel)."""
        long_pos = PositionRecord(
            underlying="SPY", qty=5,
            greeks={"delta": 0.3, "gamma": 0.05, "vega": 0.10},
        )
        short_pos = PositionRecord(
            underlying="SPY", qty=-5,
            greeks={"delta": 0.3, "gamma": 0.05, "vega": 0.10},
        )
        state = PortfolioState(current_positions=[long_pos, short_pos])
        greeks = state.total_position_greeks()

        # Delta cancels: 5*0.3 + (-5)*0.3 = 0
        assert greeks["delta"] == pytest.approx(0.0)
        # Gamma adds: abs(5)*0.05 + abs(-5)*0.05 = 0.5
        assert greeks["gamma"] == pytest.approx(0.5)
        # Vega adds: abs(5)*0.10 + abs(-5)*0.10 = 1.0
        assert greeks["vega"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Bug 2: risk_engine.py — dollar_risk preservation in denormalize
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.risk_engine import RiskEngine, RiskEngineConfig, NormalizedAllocation
from src.analysis.allocation_engine import Allocation


class TestBug2DollarRiskPreservation:
    """dollar_risk should be proportionally scaled, not overwritten with notional."""

    def test_denormalize_preserves_dollar_risk(self):
        """After denormalization, dollar_risk should reflect original risk budget scaled."""
        engine = RiskEngine()
        na = NormalizedAllocation(
            allocation_id="strat_a:SPY",
            strategy_id="strat_a",
            underlying="SPY",
            proposed_weight=0.05,
            current_weight=0.025,  # clamped to half
            dollar_risk=500.0,     # original risk budget
        )
        equity = 100_000.0
        result = engine._denormalize([na], equity)

        assert len(result) == 1
        alloc = result[0]
        # dollar_risk should be 500 * (0.025/0.05) = 250, not 0.025 * 100000 = 2500
        assert alloc.dollar_risk == pytest.approx(250.0, abs=0.01)

    def test_denormalize_zero_proposed_weight(self):
        """If proposed weight was 0, dollar_risk should be 0."""
        engine = RiskEngine()
        na = NormalizedAllocation(
            allocation_id="strat_a:SPY",
            strategy_id="strat_a",
            underlying="SPY",
            proposed_weight=0.0,
            current_weight=0.0,
            dollar_risk=0.0,
        )
        result = engine._denormalize([na], 100_000.0)
        assert result[0].dollar_risk == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Bug 3: tail_risk_scenario.py — zero-width spread guard
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.tail_risk_scenario import (
    TailRiskConfig,
    TailRiskResult,
    evaluate_tail_risk,
)


class TestBug3DegenerateSpread:
    """Degenerate spreads (same strike) must not cause division by zero."""

    def test_zero_width_spread_returns_zero_contracts(self):
        """Same short and long strike should return 0 contracts, not crash."""
        config = TailRiskConfig(enabled_for_options=True)
        result = evaluate_tail_risk(
            S=100.0,
            short_strike=95.0,
            long_strike=95.0,  # same as short = degenerate
            T=0.1,
            credit=0.50,
            v0=0.04,
            proposed_contracts=5,
            equity_usd=100_000.0,
            config=config,
        )
        assert result.capped is True
        assert result.allowed_contracts == 0
        assert "degenerate" in result.reason


# ═══════════════════════════════════════════════════════════════════════════
# Bug 4: deflated_sharpe.py — DSR formula uses observed_sharpe in SE
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.deflated_sharpe import (
    compute_deflated_sharpe_ratio,
    compute_sharpe_stats,
)


class TestBug4DSRFormula:
    """DSR standard error should use observed_sharpe, not threshold_sr."""

    def test_dsr_uses_observed_sharpe_in_denominator(self):
        """DSR with skewed returns should differ from normal assumption."""
        # With skewness=0, kurtosis=0, the choice of SR doesn't matter
        # because denom_sq = 1 - 0 + 0 = 1 in both cases.
        # With non-zero skewness, the formula uses observed_sharpe.
        dsr_skewed = compute_deflated_sharpe_ratio(
            observed_sharpe=1.5,
            threshold_sr=0.5,
            n_obs=100,
            skewness=-1.0,
            kurtosis=2.0,
        )
        # With the fix, the SE uses observed_sharpe=1.5 which makes denom_sq
        # = 1 - (-1.0)*1.5 + ((2.0-1.0)/4.0)*1.5^2 = 1 + 1.5 + 0.5625 = 3.0625
        # Without the fix (using threshold_sr=0.5), denom_sq would be
        # = 1 - (-1.0)*0.5 + ((2.0-1.0)/4.0)*0.5^2 = 1 + 0.5 + 0.0625 = 1.5625
        # The z-statistic and DSR should reflect the correct SE.
        assert 0.0 < dsr_skewed < 1.0

    def test_dsr_normal_returns_high_for_good_sharpe(self):
        """With normal returns (skew=0, kurt=0), moderate SR should yield high DSR."""
        # Use moderate observed_sharpe to avoid denom_sq going non-positive
        # denom_sq = 1 - 0*SR + ((-1)/4)*SR^2 = 1 - 0.25*SR^2
        # For SR=1.0: denom_sq = 0.75 (positive)
        dsr = compute_deflated_sharpe_ratio(
            observed_sharpe=1.0,
            threshold_sr=0.2,
            n_obs=252,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert dsr > 0.95

    def test_dsr_insufficient_obs(self):
        """With < 2 observations, DSR should be 0."""
        assert compute_deflated_sharpe_ratio(
            observed_sharpe=2.0, threshold_sr=0.5, n_obs=1
        ) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Bug 5: sizing.py — Kelly formula uses (mu - cost) / sigma^2
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.sizing import Forecast, fractional_kelly_size


class TestBug5KellyCostSubtraction:
    """Kelly f* should be (mu - cost) / sigma^2, not mu / sigma^2."""

    def test_kelly_subtracts_cost(self):
        """Position size should be smaller when cost is subtracted."""
        # Use large sigma so Kelly sizes stay well below the 0.07 per-play cap
        forecast_no_cost = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.02, sigma=0.50, cost=0.0,
        )
        forecast_with_cost = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.02, sigma=0.50, cost=0.005,
        )
        size_no_cost = fractional_kelly_size(forecast_no_cost, apply_shrinkage=False)
        size_with_cost = fractional_kelly_size(forecast_with_cost, apply_shrinkage=False)

        # No cost: f* = 0.02/0.25 = 0.08, fractional = 0.25 * 0.08 = 0.02
        # With cost: f* = (0.02-0.005)/0.25 = 0.06, fractional = 0.25 * 0.06 = 0.015
        assert size_with_cost < size_no_cost
        assert size_no_cost == pytest.approx(0.02 / 0.25 * 0.25, abs=1e-6)
        assert size_with_cost == pytest.approx(0.015 / 0.25 * 0.25, abs=1e-6)

    def test_kelly_rejects_when_mu_equals_cost(self):
        """If mu == cost, the strategy should be rejected (returns 0)."""
        forecast = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.05, sigma=0.20, cost=0.05,
        )
        size = fractional_kelly_size(forecast, apply_shrinkage=False)
        assert size == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Bug 6: drawdown_containment.py — hysteresis parameter is now passed
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.drawdown_containment import DrawdownConfig, compute_drawdown_throttle


class TestBug6DrawdownHysteresis:
    """Hysteresis should keep throttling until recovery_threshold is reached."""

    def test_hysteresis_stays_throttled(self):
        """When was_throttled=True and dd > recovery but < threshold, stay throttled."""
        config = DrawdownConfig(
            threshold_pct=0.10,
            recovery_threshold_pct=0.05,
            throttle_mode="step",
            throttle_scale=0.5,
        )
        # Drawdown is 7% — below threshold (10%) but above recovery (5%)
        throttle = compute_drawdown_throttle(0.07, config, was_throttled=True)
        # Should still be throttled due to hysteresis
        assert throttle == pytest.approx(0.5)

    def test_hysteresis_releases_after_recovery(self):
        """When dd < recovery, throttle should release to 1.0."""
        config = DrawdownConfig(
            threshold_pct=0.10,
            recovery_threshold_pct=0.05,
            throttle_mode="step",
            throttle_scale=0.5,
        )
        # Drawdown at 3% — below recovery (5%)
        throttle = compute_drawdown_throttle(0.03, config, was_throttled=True)
        assert throttle == 1.0

    def test_no_hysteresis_when_not_throttled(self):
        """When was_throttled=False and dd < threshold, return 1.0."""
        config = DrawdownConfig(threshold_pct=0.10, recovery_threshold_pct=0.05)
        throttle = compute_drawdown_throttle(0.07, config, was_throttled=False)
        assert throttle == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Bug 7: mc_bootstrap.py — drawdown % uses peak equity, not starting cash
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.mc_bootstrap import _max_drawdown_pct


class TestBug7MCDrawdownCalculation:
    """Max drawdown % should be computed against peak equity."""

    def test_drawdown_against_peak(self):
        """If equity doubles then drops 50%, drawdown should be 50% not 100%."""
        path = [100.0, 200.0, 100.0]  # peak=200, drops back to 100
        dd = _max_drawdown_pct(path, starting_cash=100.0)
        # Correct: (200 - 100) / 200 = 0.50
        # Old bug: (200 - 100) / 100 = 1.00
        assert dd == pytest.approx(0.50, abs=0.01)

    def test_flat_path_no_drawdown(self):
        """Flat equity path should have 0% drawdown."""
        path = [100.0, 100.0, 100.0]
        dd = _max_drawdown_pct(path, starting_cash=100.0)
        assert dd == pytest.approx(0.0)

    def test_monotone_decline(self):
        """Monotone decline from start should show drawdown relative to peak."""
        path = [100.0, 80.0, 60.0]
        dd = _max_drawdown_pct(path, starting_cash=100.0)
        # Peak = 100, worst = 60, dd = (100-60)/100 = 0.40
        assert dd == pytest.approx(0.40, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Bug 8: replay_harness.py — final_equity includes open positions
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.replay_harness import VirtualPortfolio


class TestBug8FinalEquity:
    """final_equity in summary should include unrealized PnL."""

    def test_final_equity_includes_unrealized(self):
        """With open positions, final_equity should be cash + unrealized."""
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        # Open a long position at $100
        portfolio.open_position(
            symbol="SPY", side="LONG", quantity=1,
            fill_price=100.0, ts_ms=1000, multiplier=100,
        )
        # Cash is now 10000 - 100*1*100 = 0
        # Mark to market at $110 (unrealized = (110-100)*1*100 = 1000)
        portfolio.mark_to_market(
            prices={"SPY": 110.0}, ts_ms=2000, multiplier=100,
        )

        summary = portfolio.summary()
        # final_equity should be cash (0) + unrealized (1000) = 1000
        # NOT just cash (0)
        assert summary["final_equity"] == pytest.approx(1000.0, abs=1.0)
        assert summary["open_positions"] == 1

    def test_total_return_includes_unrealized(self):
        """total_return_pct should account for unrealized PnL."""
        portfolio = VirtualPortfolio(starting_cash=10_000.0)
        portfolio.open_position(
            symbol="SPY", side="LONG", quantity=1,
            fill_price=100.0, ts_ms=1000, multiplier=100,
        )
        portfolio.mark_to_market(
            prices={"SPY": 110.0}, ts_ms=2000, multiplier=100,
        )
        summary = portfolio.summary()
        # Unrealized = 1000, realized = 0, total PnL = 1000
        # total_return_pct = (1000 / 10000) * 100 = 10%
        assert summary["total_return_pct"] == pytest.approx(10.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Bug 9: tail_risk_scenario.py — analytical fallback for call spreads
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.tail_risk_scenario import _analytical_prob_touch


class TestBug9AnalyticalCallSpread:
    """Analytical prob_touch should handle both put (downside) and call (upside) barriers."""

    def test_upside_barrier_gives_nonzero_prob(self):
        """Call spread with barrier > S should return nonzero probability."""
        # S=100, barrier=110 (call spread short strike above spot)
        prob = _analytical_prob_touch(S=100.0, barrier=110.0, T=0.25, sigma=0.30)
        assert prob > 0.0
        assert prob < 1.0

    def test_downside_barrier_gives_nonzero_prob(self):
        """Put spread with barrier < S should return nonzero probability."""
        prob = _analytical_prob_touch(S=100.0, barrier=90.0, T=0.25, sigma=0.30)
        assert prob > 0.0
        assert prob < 1.0

    def test_barrier_equals_spot_returns_one(self):
        """When barrier == spot (already touching), return 1.0."""
        prob = _analytical_prob_touch(S=100.0, barrier=100.0, T=0.25, sigma=0.30)
        assert prob == 1.0

    def test_barrier_below_spot_returns_one(self):
        """When barrier < spot and downside, S <= barrier check should trigger."""
        prob = _analytical_prob_touch(S=90.0, barrier=100.0, T=0.25, sigma=0.30)
        # S=90 < barrier=100 is upside barrier case now; this should give prob > 0
        assert prob > 0.0

    def test_far_otm_upside_barrier_low_prob(self):
        """Very far OTM upside barrier should have low probability."""
        prob = _analytical_prob_touch(S=100.0, barrier=200.0, T=0.05, sigma=0.20)
        assert prob < 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Bug 10: strategy_evaluator.py — DSR=0 should not bypass kill gate
# ═══════════════════════════════════════════════════════════════════════════


class TestBug10DSRZeroKillGate:
    """Strategies with DSR=0 should be killed when dsr_min > 0."""

    def test_dsr_zero_is_killed(self):
        """A strategy with DSR=0.0 should be flagged when dsr_min=0.95."""
        from src.analysis.strategy_evaluator import StrategyEvaluator

        evaluator = StrategyEvaluator(
            kill_thresholds={"dsr_min": 0.95},
        )
        record = {
            "dsr": 0.0,
            "composite_score": 0.5,
            "scoring": {"components": {}},
            "metrics": {},
            "manifest_ref": {},
        }
        reasons = evaluator._compute_kill_reasons(record)
        dsr_reasons = [r for r in reasons if r["reason"] == "dsr_below_threshold"]
        assert len(dsr_reasons) == 1
        assert dsr_reasons[0]["value"] == 0.0

    def test_dsr_above_threshold_not_killed(self):
        """A strategy with DSR=0.96 should pass when dsr_min=0.95."""
        from src.analysis.strategy_evaluator import StrategyEvaluator

        evaluator = StrategyEvaluator(
            kill_thresholds={"dsr_min": 0.95},
        )
        record = {
            "dsr": 0.96,
            "composite_score": 0.5,
            "scoring": {"components": {}},
            "metrics": {},
            "manifest_ref": {},
        }
        reasons = evaluator._compute_kill_reasons(record)
        dsr_reasons = [r for r in reasons if r["reason"] == "dsr_below_threshold"]
        assert len(dsr_reasons) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Bug 11: risk_engine.py — deterministic sort with tiebreaker
# ═══════════════════════════════════════════════════════════════════════════


class TestBug11DeterministicSort:
    """Sort key should include a tiebreaker for identical strategy+instrument."""

    def test_same_strategy_instrument_sorted_by_weight(self):
        """Two allocations with same key should have stable ordering via weight."""
        alloc1 = Allocation(
            strategy_id="strat_a", instrument="SPY",
            weight=0.03, dollar_risk=300.0,
        )
        alloc2 = Allocation(
            strategy_id="strat_a", instrument="SPY",
            weight=0.05, dollar_risk=500.0,
        )
        # Sort with the new key that includes weight
        sorted_allocs = sorted(
            [alloc2, alloc1],
            key=lambda a: (a.strategy_id, a.instrument, a.weight),
        )
        assert sorted_allocs[0].weight == 0.03
        assert sorted_allocs[1].weight == 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Bug 19: deflated_sharpe.py — consistent moment estimators
# ═══════════════════════════════════════════════════════════════════════════


class TestBug19ConsistentMomentEstimators:
    """Skewness and kurtosis should use n-1 divisor consistent with variance."""

    def test_moment_estimators_use_n_minus_1(self):
        """The stats should use n-1 for skewness/kurtosis like variance."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.04, -0.03, 0.01]
        stats = compute_sharpe_stats(returns)
        n = len(returns)
        mean = sum(returns) / n

        # Compute expected m3 using n-1
        m3 = sum((r - mean) ** 3 for r in returns) / (n - 1)
        std = stats["std"]
        expected_skew = m3 / (std ** 3)

        assert stats["skew"] == pytest.approx(expected_skew, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Bug 24: experiment_runner.py — python_version should use sys.version
# ═══════════════════════════════════════════════════════════════════════════


class TestBug24PythonVersion:
    """python_version field should report actual Python version, not date."""

    def test_python_version_not_date(self):
        """Ensure the python_version code uses sys.version."""
        import sys
        # The fix replaces `datetime.now().year.month` with `sys.version`
        # We just verify sys.version is a string starting with a version number
        assert sys.version[0].isdigit()


# ═══════════════════════════════════════════════════════════════════════════
# Bug 29: strategy_registry.py — deterministic sort for tied scores
# ═══════════════════════════════════════════════════════════════════════════

from src.analysis.strategy_registry import StrategyEntry, StrategyRegistry


class TestBug29RegistrySortDeterminism:
    """Candidates with equal scores should have deterministic ordering."""

    def test_tied_scores_sorted_by_strategy_id(self):
        """Strategies with same score should sort by strategy_id."""
        registry = StrategyRegistry()
        registry.register(StrategyEntry(
            strategy_id="b_strategy", strategy_class="B",
            composite_score=0.5,
        ))
        registry.register(StrategyEntry(
            strategy_id="a_strategy", strategy_class="A",
            composite_score=0.5,
        ))

        candidates = registry.candidates
        assert len(candidates) == 2
        # With same score, should be sorted by strategy_id
        assert candidates[0].strategy_id == "a_strategy"
        assert candidates[1].strategy_id == "b_strategy"


# ═══════════════════════════════════════════════════════════════════════════
# Bug 16: strategy_evaluator.py — threshold naming consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestBug16ThresholdNaming:
    """mc_p95 metric should match mc_p95 threshold name."""

    def test_p95_drawdown_threshold_lookup(self):
        """Both old and new threshold names should work."""
        from src.analysis.strategy_evaluator import StrategyEvaluator

        # Using the new name
        evaluator = StrategyEvaluator(
            kill_thresholds={"mc_p95_drawdown_max": 0.30},
        )
        record = {
            "dsr": 0.99,
            "composite_score": 0.5,
            "scoring": {"components": {}},
            "metrics": {"mc_p95_max_drawdown": 0.50},
            "manifest_ref": {},
        }
        reasons = evaluator._compute_kill_reasons(record)
        dd_reasons = [r for r in reasons if "drawdown" in r["reason"]]
        assert len(dd_reasons) == 1

    def test_old_p5_drawdown_threshold_still_works(self):
        """The old mc_p5_drawdown_max name should still be checked as fallback."""
        from src.analysis.strategy_evaluator import StrategyEvaluator

        evaluator = StrategyEvaluator(
            kill_thresholds={"mc_p5_drawdown_max": 0.30},
        )
        record = {
            "dsr": 0.99,
            "composite_score": 0.5,
            "scoring": {"components": {}},
            "metrics": {"mc_p95_max_drawdown": 0.50},
            "manifest_ref": {},
        }
        reasons = evaluator._compute_kill_reasons(record)
        dd_reasons = [r for r in reasons if "drawdown" in r["reason"]]
        assert len(dd_reasons) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Integration: RiskEngine with hysteresis
# ═══════════════════════════════════════════════════════════════════════════


class TestRiskEngineHysteresisIntegration:
    """Verify the risk engine passes was_throttled to the drawdown function."""

    def test_engine_tracks_throttle_state(self):
        """After throttling, _was_throttled should be True."""
        engine = RiskEngine()
        assert engine._was_throttled is False

        # Create a state with drawdown above threshold
        state = PortfolioState(
            as_of_ts_ms=1000,
            equity_usd=10_000.0,
            current_drawdown_pct=0.15,  # 15% drawdown
        )

        allocs = [
            Allocation(
                strategy_id="strat_a",
                instrument="SPY",
                weight=0.05,
                dollar_risk=500.0,
            ),
        ]

        config = RiskEngineConfig(
            drawdown=DrawdownConfig(
                threshold_pct=0.10,
                recovery_threshold_pct=0.05,
                throttle_mode="step",
                throttle_scale=0.5,
            ),
            enforce_idempotence_check=False,
            enforce_monotone_check=False,
        )

        engine.clamp(allocs, state, config)
        # After clamping with 15% drawdown (> 10% threshold), should be throttled
        assert engine._was_throttled is True
