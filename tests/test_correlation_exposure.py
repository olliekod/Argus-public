"""
Test: Correlation / Cluster Exposure Control
==============================================
"""

import math
import pytest

from src.analysis.correlation_exposure import (
    CorrelationConfig,
    build_clusters,
    compute_correlation_matrix,
    get_strategy_returns_for_correlation,
)


class TestCorrelationMatrix:

    def test_perfect_correlation(self):
        """Identical return series → correlation = 1.0."""
        aligned = {
            "A": [0.01, 0.02, -0.01, 0.03, 0.00] * 10,
            "B": [0.01, 0.02, -0.01, 0.03, 0.00] * 10,
        }
        matrix = compute_correlation_matrix(aligned, min_obs=10)
        assert abs(matrix[("A", "B")] - 1.0) < 1e-6

    def test_inverse_correlation(self):
        """Negated return series → correlation = -1.0."""
        base = [0.01, 0.02, -0.01, 0.03, 0.00] * 10
        aligned = {
            "A": base,
            "B": [-x for x in base],
        }
        matrix = compute_correlation_matrix(aligned, min_obs=10)
        assert abs(matrix[("A", "B")] - (-1.0)) < 1e-6

    def test_uncorrelated(self):
        """Unrelated series → correlation near 0."""
        import math
        n = 50
        aligned = {
            "A": [math.sin(i * 0.7) for i in range(n)],
            "B": [math.cos(i * 1.3 + 2.0) for i in range(n)],
        }
        matrix = compute_correlation_matrix(aligned, min_obs=10)
        # Should be weakly correlated
        assert abs(matrix[("A", "B")]) < 0.5

    def test_insufficient_observations(self):
        """Below min_obs → NaN."""
        aligned = {
            "A": [0.01, 0.02, 0.03],
            "B": [0.01, 0.02, 0.03],
        }
        matrix = compute_correlation_matrix(aligned, min_obs=10)
        assert math.isnan(matrix[("A", "B")])

    def test_empty_returns(self):
        """Empty returns → empty matrix."""
        matrix = compute_correlation_matrix({})
        assert matrix == {}

    def test_multiple_strategies(self):
        """Three strategies produce all pairwise correlations."""
        aligned = {
            "A": [0.01] * 50,
            "B": [0.01] * 50,
            "C": [-0.01] * 50,
        }
        matrix = compute_correlation_matrix(aligned, min_obs=10)
        assert ("A", "B") in matrix
        assert ("A", "C") in matrix
        assert ("B", "C") in matrix
        assert len(matrix) == 3


class TestBuildClusters:

    def test_all_correlated(self):
        """All highly correlated → single cluster."""
        corr = {("A", "B"): 0.9, ("A", "C"): 0.85, ("B", "C"): 0.9}
        clusters = build_clusters(["A", "B", "C"], corr, threshold=0.8)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == ["A", "B", "C"]

    def test_no_correlation(self):
        """No pair above threshold → each strategy is its own cluster."""
        corr = {("A", "B"): 0.3, ("A", "C"): 0.2, ("B", "C"): 0.1}
        clusters = build_clusters(["A", "B", "C"], corr, threshold=0.8)
        assert len(clusters) == 3

    def test_partial_clustering(self):
        """A and B correlated, C separate."""
        corr = {("A", "B"): 0.9, ("A", "C"): 0.3, ("B", "C"): 0.2}
        clusters = build_clusters(["A", "B", "C"], corr, threshold=0.8)
        assert len(clusters) == 2
        # Find the cluster containing A and B
        ab_cluster = [c for c in clusters if "A" in c][0]
        assert sorted(ab_cluster) == ["A", "B"]

    def test_deterministic_ordering(self):
        """Clusters are sorted deterministically."""
        corr = {("A", "B"): 0.9, ("C", "D"): 0.85}
        c1 = build_clusters(["A", "B", "C", "D"], corr, threshold=0.8)
        c2 = build_clusters(["D", "C", "B", "A"], corr, threshold=0.8)
        assert c1 == c2

    def test_nan_correlation_ignored(self):
        """NaN correlations don't cause clustering."""
        corr = {("A", "B"): float("nan"), ("A", "C"): 0.9}
        clusters = build_clusters(["A", "B", "C"], corr, threshold=0.8)
        # A and C should be in same cluster, B separate
        ac_cluster = [c for c in clusters if "A" in c][0]
        assert "C" in ac_cluster
        assert "B" not in ac_cluster


class TestGetStrategyReturns:

    def test_respects_as_of_ts(self):
        """Only returns data at or before as_of_ts_ms."""
        from datetime import datetime, timezone

        # as_of = 2024-01-05 midnight UTC
        as_of_ts = int(datetime(2024, 1, 5, tzinfo=timezone.utc).timestamp() * 1000)

        series = {
            "A": {
                "2024-01-01": 0.01,
                "2024-01-02": 0.02,
                "2024-01-03": 0.03,
                "2024-01-10": 0.10,  # future — should be excluded
                "2024-01-15": 0.15,  # future — should be excluded
            },
            "B": {
                "2024-01-01": -0.01,
                "2024-01-02": -0.02,
                "2024-01-03": -0.03,
                "2024-01-10": -0.10,
                "2024-01-15": -0.15,
            },
        }

        result = get_strategy_returns_for_correlation(
            series, ["A", "B"], as_of_ts, rolling_days=60,
        )

        assert len(result["A"]) == 3, "Should only include 3 dates ≤ as_of"
        assert len(result["B"]) == 3

    def test_empty_series(self):
        """Empty series → empty result."""
        result = get_strategy_returns_for_correlation({}, ["A"], 1700000000000)
        assert result == {}

    def test_no_common_dates(self):
        """Strategies with no overlapping dates → empty result."""
        series = {
            "A": {"2024-01-01": 0.01},
            "B": {"2024-01-02": 0.02},
        }
        result = get_strategy_returns_for_correlation(
            series, ["A", "B"], 1710000000000,
        )
        # No common dates
        assert result.get("A", []) == []

    def test_rolling_days_limit(self):
        """Only returns the most recent rolling_days entries."""
        dates = {f"2024-01-{d:02d}": 0.01 * d for d in range(1, 31)}
        series = {"A": dates, "B": dates}
        result = get_strategy_returns_for_correlation(
            series, ["A", "B"], 1710000000000, rolling_days=10,
        )
        assert len(result["A"]) == 10
