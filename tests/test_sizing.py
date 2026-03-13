"""
Tests for Position Sizing Module
==================================

Verifies:
- Fractional Kelly formula: c * mu / sigma^2
- Per-play cap enforcement (7% max)
- Zero sizing when edge <= cost
- Vol-target overlay
- Options spread sizing (contracts_from_risk_budget)
- Confidence shrinkage
- Full sizing pipeline
"""

from __future__ import annotations

import math

import pytest

from src.analysis.sizing import (
    Forecast,
    contracts_from_risk_budget,
    fractional_kelly_size,
    shrink_mu,
    size_position,
    vol_target_overlay,
)


class TestShrinkMu:
    def test_full_confidence(self):
        assert shrink_mu(0.05, 1.0) == 0.05

    def test_zero_confidence(self):
        assert shrink_mu(0.05, 0.0) == 0.0

    def test_half_confidence(self):
        assert abs(shrink_mu(0.10, 0.5) - 0.05) < 1e-10

    def test_clamp_above_1(self):
        assert shrink_mu(0.05, 1.5) == 0.05  # clamped to 1.0

    def test_clamp_below_0(self):
        assert shrink_mu(0.05, -0.5) == 0.0  # clamped to 0.0


class TestFractionalKellySize:
    def test_quarter_kelly_basic(self):
        """f = 0.25 * mu / sigma^2"""
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.05, sigma=0.10, confidence=1.0,
        )
        w = fractional_kelly_size(f, kelly_fraction=0.25)
        expected = 0.25 * 0.05 / (0.10 ** 2)  # = 0.25 * 5 = 1.25
        # But capped at 0.07
        assert w == 0.07

    def test_small_edge(self):
        """Small edge, no cap hit."""
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.001, sigma=0.10, confidence=1.0,
        )
        w = fractional_kelly_size(f, kelly_fraction=0.25)
        expected = 0.25 * 0.001 / 0.01  # = 0.025
        assert abs(w - expected) < 1e-6

    def test_cap_enforcement(self):
        """Kelly weight exceeding cap is clamped."""
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=1.0, sigma=0.10, confidence=1.0,
        )
        w = fractional_kelly_size(f, kelly_fraction=0.25, per_play_cap=0.07)
        assert w == 0.07

    def test_zero_when_edge_below_cost(self):
        """Size = 0 when mu <= cost."""
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.001, sigma=0.10, cost=0.002, confidence=1.0,
        )
        w = fractional_kelly_size(f)
        assert w == 0.0

    def test_zero_when_edge_equals_cost(self):
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.001, sigma=0.10, cost=0.001, confidence=1.0,
        )
        w = fractional_kelly_size(f)
        assert w == 0.0

    def test_zero_sigma(self):
        """Zero volatility should return 0."""
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.05, sigma=0.0, confidence=1.0,
        )
        w = fractional_kelly_size(f)
        assert w == 0.0

    def test_negative_sigma(self):
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.05, sigma=-0.10, confidence=1.0,
        )
        w = fractional_kelly_size(f)
        assert w == 0.0

    def test_with_shrinkage(self):
        """Low confidence should reduce position size."""
        # Use small mu/sigma to avoid hitting the cap
        f_high = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.002, sigma=0.20, confidence=1.0,
        )
        f_low = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.002, sigma=0.20, confidence=0.5,
        )
        w_high = fractional_kelly_size(f_high, apply_shrinkage=True)
        w_low = fractional_kelly_size(f_low, apply_shrinkage=True)
        assert w_low < w_high

    def test_no_shrinkage(self):
        """Without shrinkage, confidence doesn't affect sizing."""
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.002, sigma=0.20, confidence=0.5,
        )
        w_shrink = fractional_kelly_size(f, apply_shrinkage=True)
        w_no_shrink = fractional_kelly_size(f, apply_shrinkage=False)
        assert w_no_shrink > w_shrink

    def test_custom_kelly_fraction(self):
        """Half-Kelly vs quarter-Kelly."""
        # Use values small enough that neither hits the cap
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.001, sigma=0.20, confidence=1.0,
        )
        w_quarter = fractional_kelly_size(f, kelly_fraction=0.25)
        w_half = fractional_kelly_size(f, kelly_fraction=0.50)
        assert abs(w_half - 2.0 * w_quarter) < 1e-6

    def test_custom_per_play_cap(self):
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.10, sigma=0.10, confidence=1.0,
        )
        w_5pct = fractional_kelly_size(f, per_play_cap=0.05)
        w_10pct = fractional_kelly_size(f, per_play_cap=0.10)
        assert w_5pct == 0.05
        assert w_10pct == 0.10


class TestVolTargetOverlay:
    def test_basic_scaling(self):
        """If strategy vol = 20% and target = 10%, scale by 0.5."""
        w = vol_target_overlay(0.04, forecast_sigma=0.20, target_vol_annual=0.10)
        assert abs(w - 0.02) < 1e-6

    def test_high_vol_reduces(self):
        """High forecast vol should reduce weight."""
        w = vol_target_overlay(0.04, forecast_sigma=0.40, target_vol_annual=0.10)
        assert abs(w - 0.01) < 1e-6

    def test_low_vol_amplifies(self):
        """Low forecast vol should amplify weight."""
        w = vol_target_overlay(0.02, forecast_sigma=0.05, target_vol_annual=0.10)
        assert abs(w - 0.04) < 1e-6

    def test_zero_sigma(self):
        assert vol_target_overlay(0.04, 0.0, 0.10) == 0.0

    def test_negative_weight(self):
        """Negative weight should scale correctly."""
        w = vol_target_overlay(-0.04, 0.20, 0.10)
        assert abs(w - (-0.02)) < 1e-6


class TestContractsFromRiskBudget:
    def test_basic(self):
        """$700 budget, $100 max loss -> 7 contracts."""
        assert contracts_from_risk_budget(700.0, 100.0) == 7

    def test_fractional_floors(self):
        """$750 budget, $100 max loss -> 7 contracts (floor)."""
        assert contracts_from_risk_budget(750.0, 100.0) == 7

    def test_zero_max_loss(self):
        assert contracts_from_risk_budget(700.0, 0.0) == 0

    def test_negative_max_loss(self):
        assert contracts_from_risk_budget(700.0, -100.0) == 0

    def test_zero_budget(self):
        assert contracts_from_risk_budget(0.0, 100.0) == 0

    def test_negative_budget(self):
        assert contracts_from_risk_budget(-500.0, 100.0) == 0

    def test_budget_less_than_max_loss(self):
        assert contracts_from_risk_budget(50.0, 100.0) == 0


class TestSizePosition:
    def test_basic_pipeline(self):
        f = Forecast(
            strategy_id="VRP_v1", instrument="SPY",
            mu=0.003, sigma=0.15, cost=0.001, confidence=0.8,
        )
        result = size_position(f, equity=100_000.0)
        assert "weight" in result
        assert "dollar_risk" in result
        assert "kelly_raw" in result
        assert result["strategy_id"] == "VRP_v1"
        assert result["instrument"] == "SPY"

    def test_cap_enforced(self):
        """Weight should never exceed per_play_cap."""
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=1.0, sigma=0.10, confidence=1.0,
        )
        result = size_position(f, equity=100_000.0, per_play_cap=0.07)
        assert abs(result["weight"]) <= 0.07

    def test_vol_overlay_applied(self):
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.003, sigma=0.20, confidence=1.0,
        )
        result = size_position(f, equity=100_000.0, vol_target_annual=0.10)
        assert result["vol_adjusted"] is True

    def test_vol_overlay_skipped_when_none(self):
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.003, sigma=0.20, confidence=1.0,
        )
        result = size_position(f, equity=100_000.0, vol_target_annual=None)
        assert result["vol_adjusted"] is False

    def test_dollar_risk_correct(self):
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.002, sigma=0.10, confidence=1.0,
        )
        result = size_position(f, equity=100_000.0)
        expected_dollar = result["weight"] * 100_000.0
        assert abs(result["dollar_risk"] - expected_dollar) < 0.01

    def test_zero_edge_zero_size(self):
        f = Forecast(
            strategy_id="test", instrument="SPY",
            mu=0.001, sigma=0.10, cost=0.001, confidence=1.0,
        )
        result = size_position(f, equity=100_000.0)
        assert result["weight"] == 0.0
        assert result["dollar_risk"] == 0.0
