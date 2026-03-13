"""
Test: Drawdown Containment
============================
"""

import pytest

from src.analysis.drawdown_containment import DrawdownConfig, compute_drawdown_throttle


class TestDrawdownContainment:

    def test_no_drawdown_returns_one(self):
        """No drawdown → throttle = 1.0."""
        config = DrawdownConfig(threshold_pct=0.10)
        assert compute_drawdown_throttle(0.0, config) == 1.0

    def test_below_threshold_returns_one(self):
        """Drawdown below threshold → throttle = 1.0."""
        config = DrawdownConfig(threshold_pct=0.10)
        assert compute_drawdown_throttle(0.05, config) == 1.0

    def test_at_threshold_returns_one(self):
        """Drawdown exactly at threshold → throttle = 1.0 (threshold is inclusive boundary)."""
        config = DrawdownConfig(threshold_pct=0.10)
        assert compute_drawdown_throttle(0.10, config) == 1.0

    def test_linear_mode_basic(self):
        """Linear mode: throttle decreases linearly past threshold."""
        config = DrawdownConfig(
            threshold_pct=0.10,
            throttle_mode="linear",
            k=5.0,
            min_throttle=0.1,
        )
        # dd=0.15, excess=0.05, throttle = 1 - 5*0.05 = 0.75
        throttle = compute_drawdown_throttle(0.15, config)
        assert abs(throttle - 0.75) < 1e-6

    def test_linear_mode_floors_at_min_throttle(self):
        """Linear mode floors at min_throttle."""
        config = DrawdownConfig(
            threshold_pct=0.10,
            throttle_mode="linear",
            k=10.0,
            min_throttle=0.2,
        )
        # dd=0.30, excess=0.20, throttle = 1 - 10*0.20 = -1.0 → clamped to 0.2
        throttle = compute_drawdown_throttle(0.30, config)
        assert abs(throttle - 0.2) < 1e-6

    def test_step_mode(self):
        """Step mode: throttle jumps to throttle_scale above threshold."""
        config = DrawdownConfig(
            threshold_pct=0.10,
            throttle_mode="step",
            throttle_scale=0.5,
            min_throttle=0.1,
        )
        throttle = compute_drawdown_throttle(0.15, config)
        assert abs(throttle - 0.5) < 1e-6

    def test_step_mode_respects_min_throttle(self):
        """Step mode: if throttle_scale < min_throttle, floor applies."""
        config = DrawdownConfig(
            threshold_pct=0.10,
            throttle_mode="step",
            throttle_scale=0.05,
            min_throttle=0.2,
        )
        throttle = compute_drawdown_throttle(0.15, config)
        assert abs(throttle - 0.2) < 1e-6

    def test_hysteresis_still_throttled(self):
        """Hysteresis: was_throttled=True and dd > recovery → still throttled."""
        config = DrawdownConfig(
            threshold_pct=0.10,
            throttle_mode="step",
            throttle_scale=0.5,
            recovery_threshold_pct=0.05,
        )
        # dd=0.07, below threshold but above recovery and was_throttled
        throttle = compute_drawdown_throttle(
            0.07, config, was_throttled=True,
        )
        assert throttle < 1.0

    def test_hysteresis_recovered(self):
        """Hysteresis: dd below recovery → full exposure restored."""
        config = DrawdownConfig(
            threshold_pct=0.10,
            recovery_threshold_pct=0.05,
        )
        # dd=0.03, below recovery even if was_throttled
        throttle = compute_drawdown_throttle(
            0.03, config, was_throttled=True,
        )
        assert throttle == 1.0

    def test_negative_drawdown_treated_as_zero(self):
        """Negative drawdown values treated as zero."""
        config = DrawdownConfig(threshold_pct=0.10)
        assert compute_drawdown_throttle(-0.05, config) == 1.0

    def test_throttle_always_in_range(self):
        """Throttle is always in [min_throttle, 1.0] for any drawdown."""
        config = DrawdownConfig(
            threshold_pct=0.10,
            throttle_mode="linear",
            k=5.0,
            min_throttle=0.1,
        )
        for dd in [0.0, 0.05, 0.10, 0.15, 0.30, 0.50, 1.0]:
            t = compute_drawdown_throttle(dd, config)
            assert 0.1 <= t <= 1.0, f"Throttle {t} out of range for dd={dd}"
