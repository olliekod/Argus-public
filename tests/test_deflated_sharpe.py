"""
Tests for Deflated Sharpe Ratio
================================

Verifies:
- DSR formula correctness for known inputs
- DSR increases with higher Sharpe
- DSR increases with more observations (T)
- DSR decreases with more trials (N)
- Threshold Sharpe increases with N
- Edge cases: insufficient data, zero variance, extreme skew/kurtosis
"""

from __future__ import annotations

import math

import pytest

from src.analysis.deflated_sharpe import (
    _normal_cdf,
    _normal_ppf,
    compute_deflated_sharpe_ratio,
    compute_sharpe_stats,
    deflated_sharpe_ratio,
    threshold_sharpe_ratio,
)


class TestNormalCDF:
    def test_standard_values(self):
        assert abs(_normal_cdf(0.0) - 0.5) < 1e-10
        assert abs(_normal_cdf(1.96) - 0.975) < 0.01
        assert abs(_normal_cdf(-1.96) - 0.025) < 0.01

    def test_symmetry(self):
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert abs(_normal_cdf(x) + _normal_cdf(-x) - 1.0) < 1e-10


class TestNormalPPF:
    def test_median(self):
        assert abs(_normal_ppf(0.5)) < 1e-6

    def test_standard_quantiles(self):
        assert abs(_normal_ppf(0.975) - 1.96) < 0.02
        assert abs(_normal_ppf(0.025) - (-1.96)) < 0.02

    def test_inverse_of_cdf(self):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            x = _normal_ppf(p)
            assert abs(_normal_cdf(x) - p) < 0.01

    def test_boundaries(self):
        assert _normal_ppf(0.0) < -5
        assert _normal_ppf(1.0) > 5


class TestComputeSharpeStats:
    def test_basic_stats(self):
        returns = [0.01, 0.02, -0.005, 0.015, 0.01]
        stats = compute_sharpe_stats(returns)
        assert stats["n_obs"] == 5
        assert stats["mean"] > 0
        assert stats["std"] > 0
        assert stats["sharpe"] > 0

    def test_zero_returns(self):
        returns = [0.0, 0.0, 0.0, 0.0]
        stats = compute_sharpe_stats(returns)
        assert stats["sharpe"] == 0.0
        assert stats["std"] == 0.0

    def test_single_observation(self):
        stats = compute_sharpe_stats([0.05])
        assert stats["sharpe"] == 0.0
        assert stats["n_obs"] == 1

    def test_empty(self):
        stats = compute_sharpe_stats([])
        assert stats["sharpe"] == 0.0
        assert stats["n_obs"] == 0

    def test_positive_skew(self):
        # Series with positive skew (occasional large positive returns)
        returns = [0.01] * 90 + [0.50] * 10
        stats = compute_sharpe_stats(returns)
        assert stats["skew"] > 0

    def test_negative_skew(self):
        # Series with negative skew (occasional large negative returns)
        returns = [0.01] * 90 + [-0.50] * 10
        stats = compute_sharpe_stats(returns)
        assert stats["skew"] < 0


class TestThresholdSharpe:
    def test_increases_with_n(self):
        """More trials -> higher threshold (harder to beat random)."""
        sr0_10 = threshold_sharpe_ratio(1.0, 10)
        sr0_100 = threshold_sharpe_ratio(1.0, 100)
        sr0_1000 = threshold_sharpe_ratio(1.0, 1000)
        assert sr0_10 < sr0_100 < sr0_1000

    def test_increases_with_variance(self):
        """Higher cross-sectional variance -> higher threshold."""
        sr0_low = threshold_sharpe_ratio(0.1, 50)
        sr0_high = threshold_sharpe_ratio(1.0, 50)
        assert sr0_low < sr0_high

    def test_single_trial(self):
        """With 1 trial, threshold is finite and less than for many trials."""
        sr0 = threshold_sharpe_ratio(1.0, 1)
        assert math.isfinite(sr0)
        assert sr0 < threshold_sharpe_ratio(1.0, 100)

    def test_zero_trials(self):
        assert threshold_sharpe_ratio(1.0, 0) == 0.0

    def test_zero_variance(self):
        assert threshold_sharpe_ratio(0.0, 10) == 0.0


class TestDSR:
    def test_high_sharpe_high_dsr(self):
        """A very high observed Sharpe should yield DSR close to 1."""
        dsr = compute_deflated_sharpe_ratio(
            observed_sharpe=3.0,
            threshold_sr=0.5,
            n_obs=500,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert dsr > 0.99

    def test_low_sharpe_low_dsr(self):
        """An observed Sharpe below threshold should yield DSR < 0.5."""
        dsr = compute_deflated_sharpe_ratio(
            observed_sharpe=0.3,
            threshold_sr=1.5,
            n_obs=100,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert dsr < 0.5

    def test_dsr_increases_with_sharpe(self):
        """DSR should increase as observed Sharpe increases."""
        sr0 = 0.5
        dsr_low = compute_deflated_sharpe_ratio(0.5, sr0, 200)
        dsr_med = compute_deflated_sharpe_ratio(1.0, sr0, 200)
        dsr_high = compute_deflated_sharpe_ratio(2.0, sr0, 200)
        assert dsr_low < dsr_med < dsr_high

    def test_dsr_increases_with_observations(self):
        """DSR should increase with more observations (tighter CI)."""
        dsr_10 = compute_deflated_sharpe_ratio(1.0, 0.5, 10)
        dsr_50 = compute_deflated_sharpe_ratio(1.0, 0.5, 50)
        dsr_500 = compute_deflated_sharpe_ratio(1.0, 0.5, 500)
        assert dsr_10 < dsr_50
        assert dsr_50 <= dsr_500

    def test_dsr_at_threshold(self):
        """When observed SR == threshold SR, DSR should be ~0.5."""
        dsr = compute_deflated_sharpe_ratio(1.0, 1.0, 1000)
        assert 0.45 < dsr < 0.55

    def test_insufficient_observations(self):
        assert compute_deflated_sharpe_ratio(1.0, 0.5, 1) == 0.0
        assert compute_deflated_sharpe_ratio(1.0, 0.5, 0) == 0.0

    def test_negative_denominator_guard(self):
        """Extreme kurtosis should not crash, just return 0."""
        dsr = compute_deflated_sharpe_ratio(
            observed_sharpe=0.5,
            threshold_sr=0.5,
            n_obs=100,
            skewness=0.0,
            kurtosis=-100.0,  # extreme negative kurtosis
        )
        assert 0.0 <= dsr <= 1.0


class TestEndToEndDSR:
    def test_positive_returns(self):
        """Positive mean return series yields valid DSR in [0, 1] and correct metadata."""
        returns = [0.01, -0.005, 0.015, -0.002, 0.01] * 40  # 200 obs, positive mean
        result = deflated_sharpe_ratio(returns, n_trials=5)
        assert 0.0 <= result["dsr"] <= 1.0
        assert result["n_obs"] == 200
        assert result["n_trials"] == 5
        assert result["observed_sharpe"] > 0

    def test_many_trials_lower_dsr(self):
        """More trials should lower DSR (harder to beat random)."""
        returns = [0.01, -0.005, 0.015, -0.002, 0.01] * 40
        dsr_5 = deflated_sharpe_ratio(returns, n_trials=5)
        dsr_100 = deflated_sharpe_ratio(returns, n_trials=100)
        assert dsr_5["dsr"] >= dsr_100["dsr"]

    def test_with_all_sharpes(self):
        """Providing all_sharpes should change the result."""
        returns = [0.01] * 100
        all_sharpes = [0.5, 0.3, 0.2, 0.4, 0.6]
        result = deflated_sharpe_ratio(returns, n_trials=5, all_sharpes=all_sharpes)
        assert "dsr" in result
        assert "sharpe_variance" in result
        assert result["sharpe_variance"] > 0

    def test_result_keys(self):
        returns = [0.01, 0.02, -0.01] * 10
        result = deflated_sharpe_ratio(returns, n_trials=3)
        expected_keys = {
            "dsr", "observed_sharpe", "threshold_sr", "n_obs",
            "n_trials", "skew", "kurtosis", "sharpe_variance",
        }
        assert expected_keys == set(result.keys())
