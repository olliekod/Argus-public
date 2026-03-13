"""
Tests for GreeksEngine hardening.

Validates illiquid rejection, solver convergence tracking,
quality scores, and fallback logic.
"""

from __future__ import annotations

import pytest
from src.analysis.greeks_engine import Greeks, GreeksEngine


@pytest.fixture
def engine() -> GreeksEngine:
    return GreeksEngine(risk_free_rate=0.045)


# ═══════════════════════════════════════════════════════════════════════════
# Illiquid Rejection
# ═══════════════════════════════════════════════════════════════════════════

class TestIlliquidRejection:
    def test_zero_bid_rejected(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            bid=0.0, ask=0.50,
        )
        assert g.source == "failed_illiquid"
        assert g.solver_converged is False
        assert g.delta == 0.0

    def test_near_zero_mid_rejected(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=60.0, T=5 / 365, option_type="put",
            bid=0.001, ask=0.005,
        )
        assert g.source == "failed_illiquid"

    def test_wide_spread_rejected(self, engine):
        """Spread > 50% of mid should be flagged illiquid."""
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            bid=0.50, ask=2.00,  # 120% spread
        )
        assert g.source == "failed_illiquid"

    def test_liquid_quote_accepted(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            bid=2.00, ask=2.20,  # tight spread
        )
        assert g.source in ("derived", "provider")
        assert g.delta != 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Solver Convergence Tracking
# ═══════════════════════════════════════════════════════════════════════════

class TestSolverConvergence:
    def test_derived_iv_marks_converged(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            bid=2.00, ask=2.20,
        )
        assert g.solver_converged is True
        assert g.source == "derived"
        assert g.iv_used is not None
        assert g.iv_used > 0

    def test_provider_iv_marks_none(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            provider_iv=0.30,
        )
        assert g.solver_converged is None  # solver not used
        assert g.source == "provider"

    def test_failed_solve_marks_false(self, engine):
        # Market price below intrinsic should fail
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            bid=0.0, ask=0.50,
        )
        assert g.solver_converged is False

    def test_no_data_marks_none(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
        )
        assert g.solver_converged is None
        assert g.source == "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Quote Quality Score
# ═══════════════════════════════════════════════════════════════════════════

class TestQuoteQualityScore:
    def test_zero_bid_score_zero(self):
        score = GreeksEngine.compute_quote_quality_score(0.0, 1.0)
        assert score == 0.0

    def test_tight_spread_high_score(self):
        score = GreeksEngine.compute_quote_quality_score(2.00, 2.05)
        assert score > 0.9

    def test_wide_spread_low_score(self):
        score = GreeksEngine.compute_quote_quality_score(0.50, 2.00)
        assert score < 0.3

    def test_moderate_spread(self):
        score = GreeksEngine.compute_quote_quality_score(1.00, 1.30)
        assert 0.3 < score < 0.9

    def test_near_zero_mid_score_zero(self):
        score = GreeksEngine.compute_quote_quality_score(0.001, 0.005)
        assert score == 0.0

    def test_greeks_from_quote_populates_score(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            bid=2.00, ask=2.10,
        )
        assert g.quote_quality_score > 0.5

    def test_provider_iv_still_gets_score(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            provider_iv=0.30, bid=2.00, ask=2.10,
        )
        assert g.quote_quality_score > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Solver Metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestSolverMetrics:
    def test_metrics_after_solves(self):
        engine = GreeksEngine(risk_free_rate=0.045)

        # Success path
        engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            bid=2.00, ask=2.20,
        )
        # Illiquid path
        engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            bid=0.0, ask=0.50,
        )
        # Provider IV (no solve)
        engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            provider_iv=0.30,
        )

        metrics = engine.solver_metrics()
        assert metrics["solve_attempts"] == 2  # only bid/ask paths attempt
        assert metrics["solve_successes"] == 1
        assert metrics["illiquid_rejections"] == 1
        assert metrics["success_rate"] > 0
        assert metrics["illiquid_rate"] > 0

    def test_initial_metrics_zero(self):
        engine = GreeksEngine()
        metrics = engine.solver_metrics()
        assert metrics["solve_attempts"] == 0
        assert metrics["solve_successes"] == 0
        assert metrics["solve_failures"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Fallback Logic
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackLogic:
    def test_provider_iv_preferred(self, engine):
        """When provider IV is given, it should be used even if bid/ask available."""
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            provider_iv=0.30, bid=2.00, ask=2.20,
        )
        assert g.source == "provider"
        assert g.iv_used == 0.30

    def test_derived_when_no_provider_iv(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
            bid=2.00, ask=2.20,
        )
        assert g.source == "derived"

    def test_unknown_when_nothing(self, engine):
        g = engine.greeks_from_quote(
            S=100.0, K=95.0, T=30 / 365, option_type="put",
        )
        assert g.source == "unknown"
        assert g.iv_used is None
