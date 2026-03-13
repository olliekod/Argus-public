"""
Test: Greek Limits
===================
"""

import pytest

from src.analysis.greek_limits import (
    GreekLimitsConfig,
    compute_proposed_greeks,
    compute_greeks_for_underlying,
    compute_scale_factor_for_limit,
)


class MockNormalizedAlloc:
    """Mock normalized allocation for testing."""
    def __init__(self, underlying="SPY", delta=100.0, vega=50.0, gamma=5.0):
        self.underlying = underlying
        self.delta_shares_equiv = delta
        self.vega = vega
        self.gamma = gamma


class TestComputeScaleFactor:

    def test_within_limit(self):
        """Value within limit → scale = 1.0."""
        assert compute_scale_factor_for_limit(50.0, 100.0) == 1.0

    def test_exactly_at_limit(self):
        """Value at limit → scale = 1.0."""
        assert compute_scale_factor_for_limit(100.0, 100.0) == 1.0

    def test_exceeds_limit(self):
        """Value exceeds limit → scale < 1.0."""
        scale = compute_scale_factor_for_limit(200.0, 100.0)
        assert abs(scale - 0.5) < 1e-10

    def test_negative_value(self):
        """Negative value: uses absolute value."""
        scale = compute_scale_factor_for_limit(-200.0, 100.0)
        assert abs(scale - 0.5) < 1e-10

    def test_zero_value(self):
        """Zero current value → scale = 1.0."""
        assert compute_scale_factor_for_limit(0.0, 100.0) == 1.0

    def test_never_exceeds_one(self):
        """Scale factor never exceeds 1.0."""
        assert compute_scale_factor_for_limit(50.0, 100.0) <= 1.0
        assert compute_scale_factor_for_limit(100.0, 100.0) <= 1.0
        assert compute_scale_factor_for_limit(0.0, 100.0) <= 1.0


class TestComputeProposedGreeks:

    def test_basic_summation(self):
        """Sum greeks across allocations."""
        allocs = [
            MockNormalizedAlloc(delta=100, vega=50, gamma=5),
            MockNormalizedAlloc(delta=200, vega=30, gamma=3),
        ]
        existing = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
        total = compute_proposed_greeks(allocs, existing, enforce_existing=False)
        assert abs(total["delta"] - 300.0) < 1e-10
        assert abs(total["vega"] - 80.0) < 1e-10
        assert abs(total["gamma"] - 8.0) < 1e-10

    def test_includes_existing_when_enforced(self):
        """Existing position greeks added when enforce_existing=True."""
        allocs = [MockNormalizedAlloc(delta=100, vega=50, gamma=5)]
        existing = {"delta": 50.0, "gamma": 2.0, "vega": 20.0}
        total = compute_proposed_greeks(allocs, existing, enforce_existing=True)
        assert abs(total["delta"] - 150.0) < 1e-10
        assert abs(total["vega"] - 70.0) < 1e-10
        assert abs(total["gamma"] - 7.0) < 1e-10

    def test_excludes_existing_when_not_enforced(self):
        """Existing greeks excluded when enforce_existing=False."""
        allocs = [MockNormalizedAlloc(delta=100, vega=50, gamma=5)]
        existing = {"delta": 50.0, "gamma": 2.0, "vega": 20.0}
        total = compute_proposed_greeks(allocs, existing, enforce_existing=False)
        assert abs(total["delta"] - 100.0) < 1e-10

    def test_empty_allocations(self):
        """No allocations → zero greeks (plus existing if enforced)."""
        existing = {"delta": 50.0, "gamma": 2.0, "vega": 20.0}
        total = compute_proposed_greeks([], existing, enforce_existing=True)
        assert abs(total["delta"] - 50.0) < 1e-10


class TestComputeGreeksForUnderlying:

    def test_filters_by_underlying(self):
        """Only includes allocations for the specified underlying."""
        allocs = [
            MockNormalizedAlloc(underlying="SPY", delta=100, vega=50, gamma=5),
            MockNormalizedAlloc(underlying="IBIT", delta=200, vega=30, gamma=3),
            MockNormalizedAlloc(underlying="SPY", delta=50, vega=25, gamma=2),
        ]
        total = compute_greeks_for_underlying(
            allocs, "SPY", enforce_existing=False,
        )
        assert abs(total["delta"] - 150.0) < 1e-10
        assert abs(total["vega"] - 75.0) < 1e-10

    def test_includes_existing_per_underlying(self):
        """Includes existing greeks for the specific underlying."""
        allocs = [MockNormalizedAlloc(underlying="SPY", delta=100)]
        existing = {"delta": 30.0, "gamma": 1.0, "vega": 10.0}
        total = compute_greeks_for_underlying(
            allocs, "SPY", existing, enforce_existing=True,
        )
        assert abs(total["delta"] - 130.0) < 1e-10


class TestGreekLimitsConfig:

    def test_default_config(self):
        """Default config has inf limits."""
        config = GreekLimitsConfig()
        assert config.portfolio_max_delta_shares == float("inf")
        assert config.portfolio_max_vega == float("inf")
        assert config.portfolio_max_gamma == float("inf")
        assert config.enforce_existing_positions is True

    def test_per_underlying_config(self):
        """Per-underlying limits can be set."""
        config = GreekLimitsConfig(
            per_underlying={
                "SPY": {"max_delta_shares": 1000, "max_vega": 500},
                "IBIT": {"max_delta_shares": 500},
            },
        )
        assert config.per_underlying["SPY"]["max_delta_shares"] == 1000
        assert config.per_underlying["IBIT"]["max_delta_shares"] == 500
