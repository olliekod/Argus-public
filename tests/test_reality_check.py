"""
Tests for Reality Check / SPA Test
====================================

Verifies:
- Correct p-value computation
- Strong strategy gets low p-value
- Random strategies get high p-value
- Edge cases: empty input, single strategy, mismatched lengths
- Bootstrap reproducibility with seed
"""

from __future__ import annotations

import random

import pytest

from src.analysis.reality_check import (
    _compute_hac_variance,
    _stationary_bootstrap_indices,
    reality_check,
)


class TestStationaryBootstrap:
    def test_correct_length(self):
        rng = random.Random(42)
        indices = _stationary_bootstrap_indices(100, 10.0, rng)
        assert len(indices) == 100

    def test_all_valid_indices(self):
        rng = random.Random(42)
        indices = _stationary_bootstrap_indices(50, 5.0, rng)
        assert all(0 <= i < 50 for i in indices)

    def test_empty(self):
        rng = random.Random(42)
        assert _stationary_bootstrap_indices(0, 10.0, rng) == []

    def test_block_size_1_random(self):
        """Block size 1 should produce nearly i.i.d. draws."""
        rng = random.Random(42)
        indices = _stationary_bootstrap_indices(1000, 1.0, rng)
        assert len(indices) == 1000
        # With block_size=1, each draw is independent
        # So runs should be short on average
        unique = len(set(indices))
        assert unique > 100  # Should see many unique values


class TestHACVariance:
    def test_simple_variance(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        var = _compute_hac_variance(series, bandwidth=0)
        # gamma_0 / n: population variance of series = 2.0, so 2.0/5 = 0.4
        assert abs(var - 0.4) < 0.01

    def test_with_bandwidth(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        var_0 = _compute_hac_variance(series, bandwidth=0)
        var_2 = _compute_hac_variance(series, bandwidth=2)
        # HAC with positive bandwidth should differ
        assert var_2 != var_0

    def test_single_observation(self):
        var = _compute_hac_variance([5.0], bandwidth=0)
        assert var == 1.0  # returns 1.0 for n < 2

    def test_constant_series(self):
        var = _compute_hac_variance([3.0] * 100, bandwidth=0)
        assert var < 1e-10


class TestRealityCheck:
    def test_empty_strategies(self):
        result = reality_check({})
        assert result["p_value"] == 1.0
        assert result["n_strategies"] == 0

    def test_single_strong_strategy(self):
        """A strategy with consistently positive excess returns."""
        returns = {"alpha": [0.01] * 200}
        result = reality_check(returns, seed=42)
        assert result["p_value"] < 0.10
        assert result["best_strategy"] == "alpha"
        assert result["n_strategies"] == 1

    def test_single_weak_strategy(self):
        """A strategy with near-zero returns should not reject H0."""
        rng = random.Random(123)
        returns = {"weak": [rng.gauss(0.0, 0.01) for _ in range(200)]}
        result = reality_check(returns, seed=42, n_bootstrap=500)
        assert result["p_value"] > 0.10

    def test_multiple_strategies_one_strong(self):
        """Multiple strategies where one is clearly dominant."""
        rng = random.Random(42)
        strategies = {
            "noise_1": [rng.gauss(0.0, 0.01) for _ in range(200)],
            "noise_2": [rng.gauss(0.0, 0.01) for _ in range(200)],
            "noise_3": [rng.gauss(0.0, 0.01) for _ in range(200)],
            "strong": [0.02 + rng.gauss(0.0, 0.005) for _ in range(200)],
        }
        result = reality_check(strategies, seed=42)
        assert result["best_strategy"] == "strong"
        assert result["p_value"] < 0.10

    def test_with_benchmark(self):
        """Test with non-zero benchmark."""
        strategies = {"alpha": [0.02] * 100}
        benchmark = [0.01] * 100  # alpha beats benchmark by 1% per period
        result = reality_check(strategies, benchmark_returns=benchmark, seed=42)
        assert result["p_value"] < 0.10
        assert result["best_mean_excess"] > 0

    def test_reproducibility(self):
        """Same seed should produce same result."""
        rng = random.Random(99)
        strategies = {
            "s1": [rng.gauss(0.005, 0.01) for _ in range(100)],
            "s2": [rng.gauss(0.003, 0.01) for _ in range(100)],
        }
        r1 = reality_check(strategies, seed=42)
        r2 = reality_check(strategies, seed=42)
        assert r1["p_value"] == r2["p_value"]
        assert r1["test_statistic"] == r2["test_statistic"]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            reality_check({"a": [1, 2, 3], "b": [1, 2]})

    def test_mismatched_benchmark_raises(self):
        with pytest.raises(ValueError, match="Benchmark length"):
            reality_check(
                {"a": [1, 2, 3]},
                benchmark_returns=[1, 2],
            )

    def test_short_series(self):
        result = reality_check({"a": [0.01]}, seed=42)
        assert result["p_value"] == 1.0
        assert result["n_obs"] == 1

    def test_result_keys(self):
        result = reality_check({"a": [0.01] * 50}, seed=42)
        expected = {"p_value", "test_statistic", "best_strategy",
                    "best_mean_excess", "n_strategies", "n_obs"}
        assert expected == set(result.keys())
