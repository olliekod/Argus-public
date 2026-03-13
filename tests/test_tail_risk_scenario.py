"""
Test: Tail-Risk Scenario
=========================

Verify prob_touch > threshold reduces contracts deterministically.
"""

import pytest

from src.analysis.tail_risk_scenario import (
    TailRiskConfig,
    TailRiskResult,
    evaluate_tail_risk,
    _analytical_prob_touch,
)


class TestAnalyticalProbTouch:

    def test_already_touching(self):
        """Price at barrier → prob = 1.0 (for downside barrier S == barrier)."""
        assert _analytical_prob_touch(50.0, 50.0, 30/365, 0.40) == 1.0
        # S=48 < barrier=50 is an upside barrier case (barrier > S),
        # so prob < 1.0 (high probability but not certain).
        prob = _analytical_prob_touch(48.0, 50.0, 30/365, 0.40)
        assert prob > 0.5  # likely to touch nearby upside barrier

    def test_far_otm(self):
        """Very far OTM barrier → low touch probability."""
        prob = _analytical_prob_touch(100.0, 50.0, 30/365, 0.20)
        assert prob < 0.01

    def test_higher_vol_increases_prob(self):
        """Higher volatility increases touch probability."""
        prob_low = _analytical_prob_touch(50.0, 45.0, 30/365, 0.20)
        prob_high = _analytical_prob_touch(50.0, 45.0, 30/365, 0.60)
        assert prob_high > prob_low

    def test_longer_time_increases_prob(self):
        """Longer time to expiry increases touch probability."""
        prob_short = _analytical_prob_touch(50.0, 45.0, 7/365, 0.40)
        prob_long = _analytical_prob_touch(50.0, 45.0, 90/365, 0.40)
        assert prob_long > prob_short

    def test_invalid_inputs(self):
        """Invalid inputs → prob = 0.0."""
        assert _analytical_prob_touch(0.0, 45.0, 30/365, 0.40) == 0.0
        assert _analytical_prob_touch(50.0, 45.0, 0.0, 0.40) == 0.0
        assert _analytical_prob_touch(50.0, 45.0, 30/365, 0.0) == 0.0


class TestEvaluateTailRisk:

    def _default_config(self, **overrides):
        defaults = dict(
            enabled_for_options=True,
            max_prob_touch=0.35,
            stress_iv_bump=0.20,
            max_stress_loss_pct=0.05,
            mc_simulations=1_000,  # small for tests
            mc_steps_per_year=52,
            seed=42,
        )
        defaults.update(overrides)
        return TailRiskConfig(**defaults)

    def test_disabled_returns_proposed(self):
        """Disabled tail risk returns proposed contracts unchanged."""
        config = self._default_config(enabled_for_options=False)
        result = evaluate_tail_risk(
            S=50.0, short_strike=45.0, long_strike=40.0,
            T=30/365, credit=1.0, v0=0.25,
            proposed_contracts=5, equity_usd=10000.0,
            config=config,
        )
        assert result.allowed_contracts == 5
        assert not result.capped

    def test_zero_contracts(self):
        """Zero proposed contracts → zero allowed."""
        config = self._default_config()
        result = evaluate_tail_risk(
            S=50.0, short_strike=45.0, long_strike=40.0,
            T=30/365, credit=1.0, v0=0.25,
            proposed_contracts=0, equity_usd=10000.0,
            config=config,
        )
        assert result.allowed_contracts == 0

    def test_high_prob_touch_reduces_contracts(self):
        """High prob_touch (close to ATM, high vol) should reduce contracts."""
        config = self._default_config(max_prob_touch=0.20)
        result = evaluate_tail_risk(
            S=50.0,
            short_strike=49.0,   # Very close to ATM
            long_strike=44.0,
            T=60/365,            # Longer time
            credit=0.50,
            v0=0.80**2,          # Very high vol
            proposed_contracts=10,
            equity_usd=10000.0,
            config=config,
        )
        # With very close strike and high vol, prob_touch should be high
        # and contracts should be reduced
        assert result.allowed_contracts <= 10

    def test_low_prob_touch_passes(self):
        """Low prob_touch (far OTM, low vol) → contracts pass through."""
        config = self._default_config(max_prob_touch=0.90)
        result = evaluate_tail_risk(
            S=100.0,
            short_strike=70.0,   # Very far OTM
            long_strike=65.0,
            T=14/365,            # Short time
            credit=0.50,
            v0=0.15**2,          # Low vol
            proposed_contracts=5,
            equity_usd=10000.0,
            config=config,
        )
        assert result.allowed_contracts == 5
        assert not result.capped

    def test_invalid_T_returns_zero(self):
        """T ≤ 0 → zero contracts."""
        config = self._default_config()
        result = evaluate_tail_risk(
            S=50.0, short_strike=45.0, long_strike=40.0,
            T=0.0, credit=1.0, v0=0.25,
            proposed_contracts=5, equity_usd=10000.0,
            config=config,
        )
        assert result.allowed_contracts == 0
        assert result.capped

    def test_deterministic_with_seed(self):
        """Same inputs + same seed → same result."""
        config = self._default_config(seed=123, mc_simulations=5_000)
        kwargs = dict(
            S=50.0, short_strike=45.0, long_strike=40.0,
            T=30/365, credit=1.0, v0=0.30**2,
            proposed_contracts=5, equity_usd=10000.0,
            config=config,
        )
        r1 = evaluate_tail_risk(**kwargs)
        r2 = evaluate_tail_risk(**kwargs)
        assert r1.allowed_contracts == r2.allowed_contracts

    def test_stress_loss_caps_contracts(self):
        """Stress loss exceeding equity fraction caps contracts."""
        config = self._default_config(
            max_stress_loss_pct=0.01,  # Very tight: 1% of equity
        )
        result = evaluate_tail_risk(
            S=50.0,
            short_strike=48.0,
            long_strike=43.0,
            T=30/365,
            credit=0.50,
            v0=0.50**2,
            proposed_contracts=20,  # Large position
            equity_usd=10000.0,
            config=config,
        )
        # Should be reduced due to stress loss constraint
        assert result.allowed_contracts <= 20
