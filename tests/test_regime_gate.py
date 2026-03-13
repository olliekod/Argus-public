"""
Tests for regime gate + Kalshi dispatch integration.

Covers:
  - RegimeGate scalp gating (VOL_SPIKE microstructure, LIQ_LOW/DRIED block)
  - RegimeGate hold gating (spike entry horizon, spike min edge, risk-off)
  - SharedFarmState regime cache updates
  - Conservative fallback when regime data is missing
  - Config defaults and validation
  - Diagnostics counters
"""

from __future__ import annotations

import time
from typing import Dict

import pytest

from argus_kalshi.config import KalshiConfig
from argus_kalshi.shared_state import SharedFarmState
from argus_kalshi.regime_gate import RegimeGate, GateResult


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_gate(
    enable: bool = True,
    fallback_mode: str = "conservative",
    vol: str = "",
    liq: str = "",
    risk: str = "UNKNOWN",
    last_update: float = 0.0,
    **cfg_overrides,
) -> tuple:
    """Create a RegimeGate with pre-populated SharedFarmState."""
    cfg = KalshiConfig(
        enable_regime_gating=enable,
        regime_fallback_mode=fallback_mode,
        **cfg_overrides,
    )
    shared = SharedFarmState()
    if vol:
        shared.regime_vol["BTC"] = vol
    if liq:
        shared.regime_liq["BTC"] = liq
    shared.regime_risk = risk
    if last_update > 0:
        shared.regime_last_update["BTC"] = last_update
    gate = RegimeGate(cfg, shared)
    return gate, shared


# ---------------------------------------------------------------------------
#  Scalp gating tests
# ---------------------------------------------------------------------------

class TestScalpGating:

    def test_gating_disabled_allows_all(self):
        gate, _ = _make_gate(enable=False, vol="VOL_SPIKE", liq="LIQ_DRIED")
        result = gate.gate_scalp("BTC")
        assert result.allowed
        assert result.reason == "gating_disabled"

    def test_liq_low_blocks_scalp(self):
        gate, _ = _make_gate(vol="VOL_NORMAL", liq="LIQ_LOW", last_update=time.monotonic())
        result = gate.gate_scalp("BTC")
        assert not result.allowed
        assert "regime_block_liquidity" in result.reason

    def test_liq_dried_blocks_scalp(self):
        gate, _ = _make_gate(vol="VOL_NORMAL", liq="LIQ_DRIED", last_update=time.monotonic())
        result = gate.gate_scalp("BTC")
        assert not result.allowed
        assert "LIQ_DRIED" in result.reason

    def test_vol_normal_liq_normal_allows_scalp(self):
        gate, _ = _make_gate(vol="VOL_NORMAL", liq="LIQ_NORMAL", last_update=time.monotonic())
        result = gate.gate_scalp("BTC")
        assert result.allowed
        assert result.reason == "regime_ok"
        assert result.qty_multiplier == 1.0

    def test_vol_high_liq_normal_allows_scalp(self):
        gate, _ = _make_gate(vol="VOL_HIGH", liq="LIQ_NORMAL", last_update=time.monotonic())
        result = gate.gate_scalp("BTC")
        assert result.allowed

    def test_vol_spike_fails_microstructure(self):
        """VOL_SPIKE with bad microstructure should block."""
        gate, _ = _make_gate(
            vol="VOL_SPIKE", liq="LIQ_NORMAL", last_update=time.monotonic(),
            scalp_spike_max_spread_cents=1,
            scalp_spike_depth_min=100,
            scalp_spike_reprice_min=6,
            scalp_spike_min_edge_cents=8,
        )
        result = gate.gate_scalp(
            "BTC",
            spread_cents=3,   # too wide
            depth=50,         # too shallow
            reprice_move_cents=2,   # too low
            projected_net_edge_cents=5,  # below spike min
        )
        assert not result.allowed
        assert "regime_spike_microstructure_fail" in result.reason

    def test_vol_spike_passes_strict_microstructure(self):
        """VOL_SPIKE with good microstructure should allow with reduced size."""
        gate, _ = _make_gate(
            vol="VOL_SPIKE", liq="LIQ_NORMAL", last_update=time.monotonic(),
            scalp_spike_max_spread_cents=2,
            scalp_spike_depth_min=50,
            scalp_spike_reprice_min=4,
            scalp_spike_min_edge_cents=6,
            scalp_spike_qty_multiplier=0.3,
            scalp_spike_max_hold_minutes=3.0,
        )
        result = gate.gate_scalp(
            "BTC",
            spread_cents=1,
            depth=100,
            reprice_move_cents=6,
            projected_net_edge_cents=10,
        )
        assert result.allowed
        assert result.reason == "spike_strict_pass"
        assert result.qty_multiplier == 0.3
        assert result.max_hold_override_s == 180.0  # 3 * 60


# ---------------------------------------------------------------------------
#  Hold gating tests
# ---------------------------------------------------------------------------

class TestHoldGating:

    def test_gating_disabled_allows_hold(self):
        gate, _ = _make_gate(enable=False, vol="VOL_SPIKE")
        result = gate.gate_hold("BTC", time_to_settle_s=600, edge=0.02)
        assert result.allowed
        assert result.reason == "gating_disabled"

    def test_vol_spike_blocks_early_entry(self):
        """VOL_SPIKE with too much time to settle should block."""
        gate, _ = _make_gate(
            vol="VOL_SPIKE", liq="LIQ_NORMAL",
            last_update=time.monotonic(),
            hold_spike_entry_horizon_s=300.0,
            hold_spike_min_edge=0.06,
        )
        result = gate.gate_hold("BTC", time_to_settle_s=600, edge=0.08)
        assert not result.allowed
        assert "spike_too_early" in result.reason

    def test_vol_spike_blocks_low_edge(self):
        """VOL_SPIKE near-expiry but low edge should block."""
        gate, _ = _make_gate(
            vol="VOL_SPIKE", liq="LIQ_NORMAL",
            last_update=time.monotonic(),
            hold_spike_entry_horizon_s=300.0,
            hold_spike_min_edge=0.06,
        )
        result = gate.gate_hold("BTC", time_to_settle_s=200, edge=0.03)
        assert not result.allowed
        assert "spike_low_edge" in result.reason

    def test_vol_spike_allows_near_expiry_high_edge(self):
        """VOL_SPIKE with near-expiry + high edge should allow."""
        gate, _ = _make_gate(
            vol="VOL_SPIKE", liq="LIQ_NORMAL",
            last_update=time.monotonic(),
            hold_spike_entry_horizon_s=300.0,
            hold_spike_min_edge=0.06,
        )
        result = gate.gate_hold("BTC", time_to_settle_s=200, edge=0.08)
        assert result.allowed

    def test_liq_low_reduces_exposure(self):
        gate, _ = _make_gate(
            vol="VOL_NORMAL", liq="LIQ_LOW",
            last_update=time.monotonic(),
        )
        result = gate.gate_hold("BTC", time_to_settle_s=600, edge=0.05)
        assert result.allowed
        assert result.qty_multiplier < 1.0
        assert "regime_reduced" in result.reason

    def test_risk_off_reduces_exposure(self):
        gate, _ = _make_gate(
            vol="VOL_NORMAL", liq="LIQ_NORMAL",
            risk="RISK_OFF",
            last_update=time.monotonic(),
            risk_off_qty_multiplier=0.5,
        )
        result = gate.gate_hold("BTC", time_to_settle_s=600, edge=0.05)
        assert result.allowed
        assert result.qty_multiplier == 0.5
        assert result.reason == "regime_risk_off"

    def test_normal_conditions_allow_full_size(self):
        gate, _ = _make_gate(
            vol="VOL_NORMAL", liq="LIQ_NORMAL",
            risk="NEUTRAL",
            last_update=time.monotonic(),
        )
        result = gate.gate_hold("BTC", time_to_settle_s=600, edge=0.05)
        assert result.allowed
        assert result.qty_multiplier == 1.0
        assert result.reason == "regime_ok"


# ---------------------------------------------------------------------------
#  Missing regime data (fallback modes)
# ---------------------------------------------------------------------------

class TestFallbackModes:

    def test_conservative_fallback_blocks_scalp(self):
        """Missing regime data with conservative fallback should block scalps."""
        gate, _ = _make_gate(fallback_mode="conservative")
        # No regime data set — last_update=0 means data is missing
        result = gate.gate_scalp("BTC")
        assert not result.allowed
        assert "LIQ_LOW" in result.reason or "regime_block" in result.reason

    def test_permissive_fallback_allows_scalp(self):
        """Missing regime data with permissive fallback should allow."""
        gate, _ = _make_gate(fallback_mode="permissive")
        result = gate.gate_scalp("BTC")
        assert result.allowed

    def test_conservative_fallback_blocks_hold(self):
        """Missing regime data with conservative fallback should block holds (spike logic)."""
        gate, _ = _make_gate(
            fallback_mode="conservative",
            hold_spike_entry_horizon_s=300.0,
        )
        # Time to settle > horizon → should be blocked
        result = gate.gate_hold("BTC", time_to_settle_s=600, edge=0.08)
        assert not result.allowed

    def test_missing_data_increments_counter(self):
        gate, _ = _make_gate(fallback_mode="permissive")
        gate.gate_scalp("BTC")
        assert gate.counters["regime_data_missing"] >= 1


# ---------------------------------------------------------------------------
#  SharedFarmState regime cache
# ---------------------------------------------------------------------------

class TestSharedFarmStateRegime:

    def test_regime_cache_fields_exist(self):
        shared = SharedFarmState()
        assert hasattr(shared, "regime_vol")
        assert hasattr(shared, "regime_liq")
        assert hasattr(shared, "regime_risk")
        assert hasattr(shared, "regime_session")
        assert hasattr(shared, "regime_last_update")

    def test_regime_cache_defaults(self):
        shared = SharedFarmState()
        assert shared.regime_vol == {}
        assert shared.regime_liq == {}
        assert shared.regime_risk == "UNKNOWN"
        assert shared.regime_session == {}
        assert shared.regime_last_update == {}

    def test_regime_cache_update(self):
        shared = SharedFarmState()
        shared.regime_vol["BTC"] = "VOL_HIGH"
        shared.regime_liq["BTC"] = "LIQ_NORMAL"
        shared.regime_risk = "RISK_ON"
        shared.regime_last_update["BTC"] = time.monotonic()

        assert shared.regime_vol["BTC"] == "VOL_HIGH"
        assert shared.regime_liq["BTC"] == "LIQ_NORMAL"
        assert shared.regime_risk == "RISK_ON"
        assert shared.regime_last_update["BTC"] > 0


# ---------------------------------------------------------------------------
#  Config defaults and validation
# ---------------------------------------------------------------------------

class TestConfigRegimeFields:

    def test_defaults_backward_compatible(self):
        """Default config must have regime gating disabled."""
        cfg = KalshiConfig()
        assert cfg.enable_regime_gating is False
        assert cfg.regime_fallback_mode == "conservative"

    def test_invalid_fallback_mode_raises(self):
        with pytest.raises(ValueError, match="regime_fallback_mode"):
            KalshiConfig(regime_fallback_mode="invalid")

    def test_invalid_spike_qty_multiplier_raises(self):
        with pytest.raises(ValueError, match="scalp_spike_qty_multiplier"):
            KalshiConfig(scalp_spike_qty_multiplier=0.0)

    def test_valid_regime_config(self):
        cfg = KalshiConfig(
            enable_regime_gating=True,
            scalp_spike_min_edge_cents=10,
            regime_fallback_mode="permissive",
            risk_off_qty_multiplier=0.3,
        )
        assert cfg.enable_regime_gating is True
        assert cfg.scalp_spike_min_edge_cents == 10


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:

    def test_diagnostics_contains_counters(self):
        gate, _ = _make_gate(vol="VOL_NORMAL", liq="LIQ_NORMAL", last_update=time.monotonic())
        gate.gate_scalp("BTC")
        diag = gate.get_diagnostics()
        assert "counters" in diag
        assert "regime_state" in diag
        assert diag["counters"]["regime_allowed"] > 0

    def test_diagnostics_regime_snapshot(self):
        gate, shared = _make_gate(vol="VOL_HIGH", liq="LIQ_LOW", risk="RISK_OFF", last_update=time.monotonic())
        diag = gate.get_diagnostics()
        assert "BTC" in diag["regime_state"]
        assert diag["regime_state"]["BTC"]["vol"] == "VOL_HIGH"
        assert diag["regime_state"]["BTC"]["liq"] == "LIQ_LOW"
        assert diag["regime_state"]["BTC"]["risk"] == "RISK_OFF"
