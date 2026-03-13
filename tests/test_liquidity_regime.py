"""
Tests for liquidity regime classification.

Validates the LiquidityRegime enum, SymbolRegimeEvent serialization,
and the regime detector's liquidity classification.
"""

from __future__ import annotations

import pytest

from src.core.regimes import (
    LiquidityRegime,
    LIQUIDITY_REGIME_NAMES,
    SymbolRegimeEvent,
    symbol_regime_to_dict,
    dict_to_symbol_regime,
    DEFAULT_REGIME_THRESHOLDS,
    REGIME_SCHEMA_VERSION,
)


# ═══════════════════════════════════════════════════════════════════════════
# LiquidityRegime Enum
# ═══════════════════════════════════════════════════════════════════════════

class TestLiquidityRegimeEnum:
    def test_enum_values(self):
        assert LiquidityRegime.UNKNOWN == 0
        assert LiquidityRegime.LIQ_HIGH == 1
        assert LiquidityRegime.LIQ_NORMAL == 2
        assert LiquidityRegime.LIQ_LOW == 3
        assert LiquidityRegime.LIQ_DRIED == 4

    def test_name_mapping(self):
        assert LIQUIDITY_REGIME_NAMES[LiquidityRegime.LIQ_HIGH] == "LIQ_HIGH"
        assert LIQUIDITY_REGIME_NAMES[LiquidityRegime.LIQ_DRIED] == "LIQ_DRIED"


# ═══════════════════════════════════════════════════════════════════════════
# SymbolRegimeEvent with liquidity
# ═══════════════════════════════════════════════════════════════════════════

class TestSymbolRegimeEventLiquidity:
    def _make_event(self, liq_regime="LIQ_NORMAL", spread_pct=0.10, volume_pctile=50.0):
        return SymbolRegimeEvent(
            symbol="SPY",
            timeframe=60,
            timestamp_ms=1_700_000_000_000,
            vol_regime="VOL_NORMAL",
            trend_regime="RANGE",
            liquidity_regime=liq_regime,
            atr=1.5,
            atr_pct=0.003,
            vol_z=0.2,
            ema_fast=450.0,
            ema_slow=448.0,
            ema_slope=0.1,
            rsi=55.0,
            spread_pct=spread_pct,
            volume_pctile=volume_pctile,
            confidence=1.0,
            is_warm=True,
            data_quality_flags=0,
            config_hash="abc123",
        )

    def test_event_creation(self):
        e = self._make_event()
        assert e.liquidity_regime == "LIQ_NORMAL"
        assert e.spread_pct == 0.10
        assert e.volume_pctile == 50.0

    def test_serialization_roundtrip(self):
        e = self._make_event(liq_regime="LIQ_HIGH", spread_pct=0.02, volume_pctile=85.0)
        d = symbol_regime_to_dict(e)
        assert d["liquidity_regime"] == "LIQ_HIGH"
        assert d["spread_pct"] == pytest.approx(0.02, abs=1e-6)
        assert d["volume_pctile"] == pytest.approx(85.0, abs=1e-6)

        # Roundtrip
        e2 = dict_to_symbol_regime(d)
        assert e2.liquidity_regime == "LIQ_HIGH"
        assert e2.spread_pct == pytest.approx(0.02, abs=1e-6)

    def test_backward_compat_deserialization(self):
        """Dicts from before liquidity regime was added should still work."""
        old_dict = {
            "symbol": "SPY",
            "timeframe": 60,
            "timestamp_ms": 1_700_000_000_000,
            "vol_regime": "VOL_NORMAL",
            "trend_regime": "RANGE",
            # No liquidity_regime, spread_pct, volume_pctile
            "atr": 1.5,
            "atr_pct": 0.003,
            "vol_z": 0.2,
            "ema_fast": 450.0,
            "ema_slow": 448.0,
            "ema_slope": 0.1,
            "rsi": 55.0,
            "confidence": 1.0,
            "is_warm": True,
            "data_quality_flags": 0,
            "config_hash": "abc123",
        }
        e = dict_to_symbol_regime(old_dict)
        assert e.liquidity_regime == "UNKNOWN"
        assert e.spread_pct == 0.0
        assert e.volume_pctile == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Default Thresholds
# ═══════════════════════════════════════════════════════════════════════════

class TestDefaultThresholds:
    def test_liquidity_thresholds_present(self):
        assert "liq_spread_high_pct" in DEFAULT_REGIME_THRESHOLDS
        assert "liq_spread_low_pct" in DEFAULT_REGIME_THRESHOLDS
        assert "liq_spread_dried_pct" in DEFAULT_REGIME_THRESHOLDS
        assert "liq_volume_high_pctile" in DEFAULT_REGIME_THRESHOLDS
        assert "liq_volume_low_pctile" in DEFAULT_REGIME_THRESHOLDS

    def test_threshold_ordering(self):
        t = DEFAULT_REGIME_THRESHOLDS
        assert t["liq_spread_high_pct"] < t["liq_spread_low_pct"]
        assert t["liq_spread_low_pct"] < t["liq_spread_dried_pct"]


# ═══════════════════════════════════════════════════════════════════════════
# Regime Detector Liquidity Classification
# ═══════════════════════════════════════════════════════════════════════════

class TestRegimeDetectorLiquidity:
    """Test the _classify_liquidity_regime method directly."""

    def _make_detector(self):
        """Create a minimal RegimeDetector for testing classification."""
        from unittest.mock import MagicMock
        from src.core.regime_detector import RegimeDetector

        bus = MagicMock()
        bus.subscribe = MagicMock()
        detector = RegimeDetector(bus=bus)
        return detector

    def test_high_liquidity(self):
        d = self._make_detector()
        regime, _, _ = d._classify_liquidity_regime(0.02, 80.0)
        assert regime == "LIQ_HIGH"

    def test_normal_liquidity(self):
        d = self._make_detector()
        regime, _, _ = d._classify_liquidity_regime(0.10, 50.0)
        assert regime == "LIQ_NORMAL"

    def test_low_liquidity_wide_spread(self):
        d = self._make_detector()
        regime, _, _ = d._classify_liquidity_regime(0.25, 50.0)
        assert regime == "LIQ_LOW"

    def test_low_liquidity_low_volume(self):
        d = self._make_detector()
        regime, _, _ = d._classify_liquidity_regime(0.10, 20.0)
        assert regime == "LIQ_LOW"

    def test_dried_liquidity(self):
        d = self._make_detector()
        regime, _, _ = d._classify_liquidity_regime(0.60, 10.0)
        assert regime == "LIQ_DRIED"

    def test_deterministic_classification(self):
        """Same inputs always produce same regime."""
        d = self._make_detector()
        for _ in range(10):
            regime, _, _ = d._classify_liquidity_regime(0.15, 50.0)
            assert regime == "LIQ_NORMAL"
