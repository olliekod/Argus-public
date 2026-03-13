"""Tests for the Kalshi farm optimization redesign components.

Covers: decision_context v2, context_policy, edge_tracker, population_scaler,
backward compatibility, and bounded data structures.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import pytest

from argus_kalshi.config import KalshiConfig
from argus_kalshi.decision_context import (
    _edge_bucket,
    _liq_bucket,
    _price_bucket,
    _spread_bucket,
    _strike_distance_bucket,
    _strike_distance_pct,
    _tts_bucket,
    build_decision_context,
)
from argus_kalshi.models import MarketMetadata, OrderbookState, TradeSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides):
    defaults = {
        "bot_id": "test",
        "bankroll_usd": 5000.0,
        "max_fraction_per_market": 0.10,
        "daily_drawdown_limit": 0.05,
        "risk_fraction_per_trade": 0.001,
        "sizing_risk_fraction": 0.001,
        "fallback_activation_s": 5.0,
        "truth_feed_stale_timeout_s": 30.0,
    }
    defaults.update(overrides)
    return KalshiConfig(**defaults)


def _make_signal(**overrides):
    defaults = {
        "market_ticker": "KXBTC-26MAR-90000",
        "side": "yes",
        "action": "buy",
        "limit_price_cents": 35,
        "quantity_contracts": 2,
        "edge": 0.08,
        "p_yes": 0.45,
        "timestamp": 1_700_000_000.0,
    }
    defaults.update(overrides)
    return TradeSignal(**defaults)


def _make_metadata(**overrides):
    defaults = {
        "market_ticker": "KXBTC-26MAR-90000",
        "strike_price": 90_000.0,
        "settlement_time_iso": "2026-03-26T12:00:00+00:00",
        "last_trade_time_iso": "2026-03-26T11:50:00+00:00",
        "is_range": False,
        "strike_floor": None,
        "strike_cap": None,
        "asset": "BTC",
        "window_minutes": 15,
    }
    defaults.update(overrides)
    return MarketMetadata(**defaults)


def _make_orderbook(**overrides):
    defaults = {
        "market_ticker": "KXBTC-26MAR-90000",
        "best_yes_bid_cents": 33,
        "best_no_bid_cents": 63,
        "implied_yes_ask_cents": 37,
        "implied_no_ask_cents": 67,
        "seq": 1,
        "valid": True,
        "obi": 0.15,
        "micro_price_cents": 35.0,
        "best_yes_depth": 2500,
        "best_no_depth": 5000,
    }
    defaults.update(overrides)
    return OrderbookState(**defaults)


# ===========================================================================
# 1. Decision context v2 fields
# ===========================================================================

class TestStrikeDistancePct:
    def test_binary_market(self):
        meta = _make_metadata(strike_price=90_000.0)
        result = _strike_distance_pct(88_000.0, meta)
        assert result == pytest.approx(abs(88_000 - 90_000) / 88_000, rel=1e-6)

    def test_range_market(self):
        meta = _make_metadata(is_range=True, strike_floor=85_000.0, strike_cap=95_000.0, strike_price=0.0)
        midpoint = (85_000 + 95_000) / 2.0
        result = _strike_distance_pct(88_000.0, meta)
        assert result == pytest.approx(abs(88_000 - midpoint) / 88_000, rel=1e-6)

    def test_spot_zero_returns_none(self):
        meta = _make_metadata()
        assert _strike_distance_pct(0.0, meta) is None

    def test_no_metadata_returns_none(self):
        assert _strike_distance_pct(90_000.0, None) is None

    def test_zero_strike_returns_none(self):
        meta = _make_metadata(strike_price=0.0)
        assert _strike_distance_pct(90_000.0, meta) is None


class TestStrikeDistanceBucket:
    @pytest.mark.parametrize("sd_pct, expected", [
        (None, "na"),
        (0.003, "lt_0.005"),
        (0.007, "0.005_0.01"),
        (0.015, "0.01_0.02"),
        (0.035, "0.02_0.05"),
        (0.10, "ge_0.05"),
    ])
    def test_default_edges(self, sd_pct, expected):
        assert _strike_distance_bucket(sd_pct) == expected

    def test_custom_edges(self):
        assert _strike_distance_bucket(0.015, edges=[0.01, 0.05]) == "0.01_0.05"
        assert _strike_distance_bucket(0.005, edges=[0.01, 0.05]) == "lt_0.01"
        assert _strike_distance_bucket(0.10, edges=[0.01, 0.05]) == "ge_0.05"


class TestSpreadBucket:
    @pytest.mark.parametrize("spread, expected", [
        (None, "na"),
        (0, "tight"),
        (1, "tight"),
        (2, "normal"),
        (3, "normal"),
        (5, "wide"),
        (6, "wide"),
        (10, "very_wide"),
    ])
    def test_spread_classification(self, spread, expected):
        assert _spread_bucket(spread) == expected


class TestLiqBucket:
    @pytest.mark.parametrize("depth, expected", [
        (None, "na"),
        (100, "deep"),
        (50, "deep"),
        (30, "normal"),
        (20, "normal"),
        (10, "thin"),
        (5, "thin"),
        (3, "dry"),
        (0, "dry"),
    ])
    def test_liquidity_classification(self, depth, expected):
        assert _liq_bucket(depth) == expected


class TestBuildDecisionContext:
    def test_v2_fields_present(self):
        signal = _make_signal()
        meta = _make_metadata(strike_price=90_000.0)
        ob = _make_orderbook()
        ctx = build_decision_context(
            signal,
            family="BTC 15m",
            source="strategy",
            profile_name="base",
            now_ts=1_700_000_000.0,
            orderbook=ob,
            metadata=meta,
            spot_price=89_000.0,
        )
        assert ctx["v"] == 2
        assert "sdp" in ctx
        assert "sdb" in ctx
        assert "nm" in ctx
        assert "spb" in ctx
        assert "lb" in ctx

    def test_sdp_correct_value(self):
        signal = _make_signal()
        meta = _make_metadata(strike_price=90_000.0)
        ctx = build_decision_context(
            signal,
            family="BTC 15m",
            source="strategy",
            profile_name="base",
            now_ts=1_700_000_000.0,
            metadata=meta,
            spot_price=89_000.0,
        )
        expected_sdp = abs(89_000 - 90_000) / 89_000
        assert ctx["sdp"] == pytest.approx(round(expected_sdp, 4), rel=1e-4)

    def test_near_money_true_when_close(self):
        signal = _make_signal()
        # strike=90000, spot=89500 => sd_pct ~ 0.56% which is < 8% default
        meta = _make_metadata(strike_price=90_000.0)
        ctx = build_decision_context(
            signal,
            family="BTC 15m",
            source="strategy",
            profile_name="base",
            now_ts=1_700_000_000.0,
            metadata=meta,
            spot_price=89_500.0,
        )
        assert ctx["nm"] is True

    def test_near_money_false_when_far(self):
        signal = _make_signal()
        # strike=90000, spot=50000 => sd_pct = 80% which is > 8%
        meta = _make_metadata(strike_price=90_000.0)
        ctx = build_decision_context(
            signal,
            family="BTC 15m",
            source="strategy",
            profile_name="base",
            now_ts=1_700_000_000.0,
            metadata=meta,
            spot_price=50_000.0,
        )
        assert ctx.get("nm") is False

    def test_spot_zero_sdp_absent(self):
        """When spot_price=0, sdp/sdb/nm should be None-stripped or defaults."""
        signal = _make_signal()
        meta = _make_metadata()
        ctx = build_decision_context(
            signal,
            family="BTC 15m",
            source="strategy",
            profile_name="base",
            now_ts=1_700_000_000.0,
            metadata=meta,
            spot_price=0.0,
        )
        # sdp is None → stripped by the None filter
        assert "sdp" not in ctx
        assert ctx["sdb"] == "na"

    def test_spread_and_liq_buckets_with_orderbook(self):
        signal = _make_signal()
        ob = _make_orderbook()
        ctx = build_decision_context(
            signal,
            family="BTC 15m",
            source="strategy",
            profile_name="base",
            now_ts=1_700_000_000.0,
            orderbook=ob,
        )
        assert ctx["spb"] in ("tight", "normal", "wide", "very_wide")
        assert ctx["lb"] in ("deep", "normal", "thin", "dry")

    def test_no_orderbook_spread_liq_na(self):
        signal = _make_signal()
        ctx = build_decision_context(
            signal,
            family="BTC 15m",
            source="strategy",
            profile_name="base",
            now_ts=1_700_000_000.0,
        )
        assert ctx["spb"] == "na"
        assert ctx["lb"] == "na"


# ===========================================================================
# 2. Context policy engine
# ===========================================================================

from argus_kalshi.context_policy import (
    AdaptiveCapEngine,
    ContextPolicyEngine,
    DriftGuard,
    build_context_key,
)


class TestBuildContextKey:
    def test_output_format(self):
        key = build_context_key("BTC 15m", "yes", "0.05_0.10", "lt_40", "lt_0.005", True)
        assert key == "BTC 15m|yes|0.05_0.10|lt_40|lt_0.005|nm|flat"

    def test_far_money(self):
        key = build_context_key("ETH 60m", "no", "ge_0.20", "ge_78", "ge_0.05", False)
        assert key == "ETH 60m|no|ge_0.20|ge_78|ge_0.05|far|flat"

    def test_default_strike_distance(self):
        key = build_context_key("BTC 15m", "yes", "lt_0.05", "lt_40")
        assert key == "BTC 15m|yes|lt_0.05|lt_40|na|far|flat"


class TestContextPolicyEngine:
    def _engine(self, **overrides):
        return ContextPolicyEngine(_make_cfg(
            enable_context_policy=True,
            context_policy_window_settles=100,
            context_policy_min_samples=5,
            context_policy_shrinkage=0.7,
            context_policy_promote_threshold_usd=0.50,
            context_policy_demote_threshold_usd=-0.50,
            context_policy_core_weight_max=1.5,
            context_policy_explore_weight=0.5,
            **overrides,
        ))

    def test_record_and_get_weight(self):
        engine = self._engine()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        # Seed enough positive settlements to promote
        for _ in range(10):
            engine.record_settlement(key, 1.0)
        w = engine.get_weight(key)
        # With 10 samples >= min_samples=5, core lane, should return 1.5
        assert w == pytest.approx(1.5)

    def test_classification_core(self):
        engine = self._engine()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(10):
            engine.record_settlement(key, 1.0)
        assert engine.classify_context(key) == "core"

    def test_classification_explore(self):
        engine = self._engine()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(10):
            engine.record_settlement(key, -1.0)
        assert engine.classify_context(key) == "explore"

    def test_shrinkage_blends_toward_one(self):
        # Use min_samples=10 so we can have enough to classify but test shrinkage
        engine = ContextPolicyEngine(_make_cfg(
            enable_context_policy=True,
            context_policy_window_settles=100,
            context_policy_min_samples=10,
            context_policy_shrinkage=0.7,
            context_policy_promote_threshold_usd=0.50,
            context_policy_demote_threshold_usd=-0.50,
            context_policy_core_weight_max=1.5,
            context_policy_explore_weight=0.5,
        ))
        key = "fam|yes|lt_0.05|lt_40|na|far"
        # Seed enough to classify as core (>= old min but < new min won't work)
        # Instead: manually set the lane, then check shrinkage on low samples
        # Approach: use min_samples=10, feed 10 positives to classify as core,
        # then create a fresh engine, load the lane, and test with fewer samples.
        # Simpler: just verify that unknown keys with few samples return 1.0
        # (shrinkage only matters for classified contexts)
        for _ in range(10):
            engine.record_settlement(key, 1.0)
        assert engine.classify_context(key) == "core"
        assert engine.get_weight(key) == pytest.approx(1.5)

        # Now test a key that's classified as core but has few samples via policy load
        engine2 = ContextPolicyEngine(_make_cfg(
            enable_context_policy=True,
            context_policy_window_settles=100,
            context_policy_min_samples=10,
            context_policy_shrinkage=0.7,
            context_policy_core_weight_max=1.5,
        ))
        # Manually inject a core lane with only 3 samples
        with engine2._lock:
            engine2._lanes[key] = "core"
            engine2._windows[key] = deque([1.0, 1.0, 1.0], maxlen=100)
        w = engine2.get_weight(key)
        # 3 samples, min=10, shrinkage=0.7: alpha = (3/10)*(1-0.7) = 0.09
        # weight = 0.09*1.5 + 0.91*1.0 = 1.045
        assert 1.0 < w < 1.5

    def test_unknown_key_returns_one(self):
        engine = self._engine()
        assert engine.get_weight("nonexistent") == 1.0

    def test_save_load_roundtrip(self):
        engine = self._engine()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(10):
            engine.record_settlement(key, 1.0)
        Path("logs").mkdir(parents=True, exist_ok=True)
        path_obj = Path("logs/test_context_policy_roundtrip.json")
        if path_obj.exists():
            path_obj.unlink()
        path = str(path_obj)
        engine.save_policy(path)

        engine2 = self._engine()
        assert engine2.load_policy(path) is True
        assert engine2.classify_context(key) == "core"
        assert engine2.get_weight(key) == pytest.approx(1.5)

        if path_obj.exists():
            path_obj.unlink()

    def test_load_missing_file_returns_false(self):
        engine = self._engine()
        assert engine.load_policy(str(Path("tmp") / "nope_policy_context.json")) is False

    def test_disabled_mode_returns_one(self):
        engine = ContextPolicyEngine(_make_cfg(enable_context_policy=False))
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(100):
            engine.record_settlement(key, 1.0)
        assert engine.get_weight(key) == 1.0


class TestDriftGuard:
    def _guard(self, **overrides):
        return DriftGuard(_make_cfg(
            enable_drift_guard=True,
            drift_guard_window_settles=5,
            drift_guard_consecutive_negative=2,
            drift_guard_negative_threshold_usd=-0.10,
            drift_guard_demote_multiplier=0.5,
            **overrides,
        ))

    def test_demote_after_consecutive_negative(self):
        guard = self._guard()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        # Fill the window (size=5) with losses
        for _ in range(5):
            guard.record_settlement(key, -0.50)
        # Each check_drift sees negative expectancy → consecutive_neg increments
        r1 = guard.check_drift(key)
        # First check: consecutive_neg=1, need 2
        assert r1 is None
        # Second check
        r2 = guard.check_drift(key)
        assert r2 == "demote"

    def test_demote_multiplier_after_demote(self):
        guard = self._guard()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(5):
            guard.record_settlement(key, -0.50)
        guard.check_drift(key)
        guard.check_drift(key)
        assert guard.get_demote_multiplier(key) == pytest.approx(0.5)

    def test_no_demote_with_positive(self):
        guard = self._guard()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(5):
            guard.record_settlement(key, 1.0)
        assert guard.check_drift(key) is None
        assert guard.get_demote_multiplier(key) == 1.0

    def test_disabled_returns_none(self):
        guard = DriftGuard(_make_cfg(enable_drift_guard=False))
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(10):
            guard.record_settlement(key, -1.0)
        assert guard.check_drift(key) is None
        assert guard.get_demote_multiplier(key) == 1.0


class TestAdaptiveCapEngine:
    def _engine(self, **overrides):
        return AdaptiveCapEngine(_make_cfg(
            enable_adaptive_caps=True,
            adaptive_cap_min_samples=3,
            adaptive_cap_negative_threshold_usd=-0.10,
            adaptive_cap_tightening_mult=0.5,
            adaptive_cap_cooldown_minutes=1.0,
            **overrides,
        ))

    def test_cap_tightens_on_negative_concentration(self):
        engine = self._engine()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(3):
            engine.record_settlement(key, -0.50, concentration_share=0.3)
        now = time.time()
        mult = engine.get_cap_multiplier(key, now)
        assert mult == pytest.approx(0.5)

    def test_no_tighten_with_positive(self):
        engine = self._engine()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(3):
            engine.record_settlement(key, 1.0, concentration_share=0.3)
        now = time.time()
        assert engine.get_cap_multiplier(key, now) == 1.0

    def test_cooldown_expires(self):
        engine = self._engine()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(3):
            engine.record_settlement(key, -0.50, concentration_share=0.3)
        # After cooldown expires (1 minute), multiplier should be 1.0
        future_ts = time.time() + 120  # 2 minutes later
        assert engine.get_cap_multiplier(key, future_ts) == 1.0

    def test_disabled_returns_one(self):
        engine = AdaptiveCapEngine(_make_cfg(enable_adaptive_caps=False))
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(100):
            engine.record_settlement(key, -1.0, concentration_share=0.5)
        assert engine.get_cap_multiplier(key, time.time()) == 1.0


# ===========================================================================
# 3. Edge tracker
# ===========================================================================

from argus_kalshi.edge_tracker import EdgeTracker


class TestEdgeTracker:
    def _tracker(self, **overrides):
        return EdgeTracker(_make_cfg(
            enable_edge_tracking=True,
            edge_tracking_window_settles=50,
            edge_tracking_min_samples=3,
            edge_retention_decay_threshold=0.3,
            edge_retention_decay_multiplier=0.7,
            **overrides,
        ))

    def test_record_entry_and_settlement_retention(self):
        tracker = self._tracker()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        # Record entries (expected edge) and settlements (realized)
        for i in range(5):
            tracker.record_entry(key, expected_edge=0.10, entry_price_cents=40)
            # net_pnl = realized_edge * cost; cost = (40/100)*1 = 0.4
            # realized_edge = pnl / cost = 0.04 / 0.4 = 0.10
            tracker.record_settlement(key, net_pnl_usd=0.04, entry_price_cents=40, quantity=1)
        ratio = tracker.get_retention_ratio(key)
        assert ratio == pytest.approx(1.0, abs=0.1)

    def test_weight_multiplier_good_retention(self):
        tracker = self._tracker()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(5):
            tracker.record_entry(key, expected_edge=0.10, entry_price_cents=40)
            tracker.record_settlement(key, net_pnl_usd=0.04, entry_price_cents=40, quantity=1)
        assert tracker.get_weight_multiplier(key) == 1.0

    def test_weight_multiplier_poor_retention(self):
        tracker = self._tracker()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(5):
            tracker.record_entry(key, expected_edge=0.10, entry_price_cents=40)
            # Very small realized = low retention
            tracker.record_settlement(key, net_pnl_usd=0.001, entry_price_cents=40, quantity=1)
        assert tracker.get_weight_multiplier(key) == pytest.approx(0.7)

    def test_insufficient_samples_returns_one(self):
        tracker = self._tracker()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        tracker.record_entry(key, expected_edge=0.10, entry_price_cents=40)
        tracker.record_settlement(key, net_pnl_usd=0.001, entry_price_cents=40, quantity=1)
        # Only 1 sample, min_samples=3
        assert tracker.get_retention_ratio(key) == 1.0
        assert tracker.get_weight_multiplier(key) == 1.0

    def test_disabled_always_one(self):
        tracker = EdgeTracker(_make_cfg(enable_edge_tracking=False))
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(10):
            tracker.record_entry(key, expected_edge=0.10, entry_price_cents=40)
            tracker.record_settlement(key, net_pnl_usd=0.0, entry_price_cents=40, quantity=1)
        assert tracker.get_weight_multiplier(key) == 1.0

    def test_clamp_extreme_realized(self):
        """Extreme PnL values should not blow up retention ratio."""
        tracker = self._tracker()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        for _ in range(5):
            tracker.record_entry(key, expected_edge=0.10, entry_price_cents=40)
            # Extremely large PnL → clamped to max 2.0
            tracker.record_settlement(key, net_pnl_usd=100.0, entry_price_cents=40, quantity=1)
        ratio = tracker.get_retention_ratio(key)
        # realized clamped to 2.0 each, expected=0.10 each → ratio = 2.0/0.10 = 20.0
        # But this should still be a finite number, not blow up
        assert ratio > 0
        assert ratio < 100.0

    def test_zero_entry_price_skipped(self):
        tracker = self._tracker()
        key = "fam|yes|lt_0.05|lt_40|na|far"
        tracker.record_entry(key, expected_edge=0.10, entry_price_cents=40)
        tracker.record_settlement(key, net_pnl_usd=0.04, entry_price_cents=0, quantity=1)
        # settlement was skipped, so insufficient samples
        assert tracker.get_retention_ratio(key) == 1.0


# ===========================================================================
# 4. Population scaler
# ===========================================================================

from argus_kalshi.population_scaler import PopulationScaler, ScaleEvent, ScaleGateResult


class TestPopulationScaler:
    def _scaler(self, **overrides):
        return PopulationScaler(_make_cfg(
            bot_population_scale_enabled=True,
            bot_population_scale_schedule=[1.0, 1.15, 1.30],
            bot_population_scale_require_passes=2,
            bot_population_scale_cooldown_hours=0.0,  # no cooldown for tests
            bot_population_scale_max_step=0.20,
            bot_population_scale_min_window_hours=0.0,
            **overrides,
        ))

    def _passing_gate(self, scaler, now_ts=1_000_000.0):
        return scaler.evaluate_gate(
            concentration_share=0.1,
            edge_retention=0.9,
            expectancy_usd=1.0,
            drawdown_pct=0.01,
            crowding_stable=True,
            dispatch_p95_ms=5.0,
            now_ts=now_ts,
        )

    def _failing_gate(self, scaler, now_ts=1_000_000.0):
        return scaler.evaluate_gate(
            concentration_share=0.5,  # exceeds max 0.25 default → fail
            edge_retention=0.9,
            expectancy_usd=1.0,
            drawdown_pct=0.01,
            crowding_stable=True,
            dispatch_p95_ms=5.0,
            now_ts=now_ts,
        )

    def test_gate_all_pass(self):
        scaler = self._scaler()
        result = self._passing_gate(scaler)
        assert result.passed is True
        assert result.concentration_ok is True
        assert result.edge_retention_ok is True

    def test_gate_fail_concentration(self):
        scaler = self._scaler()
        result = self._failing_gate(scaler)
        assert result.passed is False
        assert result.concentration_ok is False

    def test_scale_up_after_passes(self):
        scaler = self._scaler()
        now = 1_000_000.0
        gate = self._passing_gate(scaler, now)
        r1 = scaler.attempt_scale(gate, now)
        assert r1 is None  # 1 pass, need 2
        gate = self._passing_gate(scaler, now + 1)
        r2 = scaler.attempt_scale(gate, now + 1)
        assert r2 is not None
        assert r2.action == "scale_up"
        assert r2.from_stage == 0
        assert r2.to_stage == 1
        assert scaler.current_stage == 1

    def test_gate_fail_resets_pass_counter(self):
        scaler = self._scaler()
        now = 1_000_000.0
        gate = self._passing_gate(scaler, now)
        scaler.attempt_scale(gate, now)  # 1 pass
        gate = self._failing_gate(scaler, now + 1)
        scaler.attempt_scale(gate, now + 1)  # resets
        gate = self._passing_gate(scaler, now + 2)
        r = scaler.attempt_scale(gate, now + 2)
        assert r is None  # only 1 pass since reset

    def test_scale_down_after_failures(self):
        scaler = self._scaler()
        # First scale up to stage 1
        now = 1_000_000.0
        for i in range(2):
            gate = self._passing_gate(scaler, now + i)
            scaler.attempt_scale(gate, now + i)
        assert scaler.current_stage == 1

        # 3 consecutive failures → scale down
        for i in range(3):
            gate = self._failing_gate(scaler, now + 100 + i)
            r = scaler.attempt_scale(gate, now + 100 + i)
        assert r is not None
        assert r.action == "scale_down"
        assert scaler.current_stage == 0

    def test_cannot_scale_below_zero(self):
        scaler = self._scaler()
        assert scaler.current_stage == 0
        # Try many failures at stage 0
        now = 1_000_000.0
        for i in range(10):
            gate = self._failing_gate(scaler, now + i)
            scaler.attempt_scale(gate, now + i)
        assert scaler.current_stage == 0

    def test_cooldown_prevents_re_attempt(self):
        scaler = PopulationScaler(_make_cfg(
            bot_population_scale_enabled=True,
            bot_population_scale_schedule=[1.0, 1.15, 1.30],
            bot_population_scale_require_passes=2,
            bot_population_scale_cooldown_hours=1.0,
            bot_population_scale_max_step=0.20,
            bot_population_scale_min_window_hours=0.0,
        ))
        now = 1_000_000.0
        # Scale up
        for i in range(2):
            gate = self._passing_gate(scaler, now + i)
            scaler.attempt_scale(gate, now + i)
        assert scaler.current_stage == 1
        # Immediately try again: cooldown blocks
        gate = self._passing_gate(scaler, now + 10)
        r = scaler.attempt_scale(gate, now + 10)
        assert r is None
        gate = self._passing_gate(scaler, now + 11)
        r2 = scaler.attempt_scale(gate, now + 11)
        assert r2 is None

    def test_disabled_returns_none(self):
        scaler = PopulationScaler(_make_cfg(bot_population_scale_enabled=False))
        gate = ScaleGateResult(
            passed=True,
            concentration_ok=True,
            edge_retention_ok=True,
            expectancy_ok=True,
            drawdown_ok=True,
            crowding_ok=True,
            runtime_ok=True,
            details={},
        )
        assert scaler.attempt_scale(gate, time.time()) is None

    def test_force_stage(self):
        scaler = self._scaler()
        scaler.force_stage(2)
        assert scaler.current_stage == 2
        assert scaler.current_multiplier == pytest.approx(1.30)

    def test_force_stage_invalid_raises(self):
        scaler = self._scaler()
        with pytest.raises(ValueError):
            scaler.force_stage(10)
        with pytest.raises(ValueError):
            scaler.force_stage(-1)


# ===========================================================================
# 5. Backward compatibility
# ===========================================================================

class TestBackwardCompatibility:
    def test_default_config_no_new_fields_needed(self):
        """KalshiConfig with only required fields still works."""
        cfg = _make_cfg()
        assert cfg.bot_id == "test"

    def test_new_toggles_default_disabled(self):
        cfg = _make_cfg()
        assert cfg.enable_context_policy is False
        assert cfg.enable_drift_guard is False
        assert cfg.enable_adaptive_caps is False
        assert cfg.enable_edge_tracking is False
        assert cfg.bot_population_scale_enabled is False

    def test_edge_bucket_unchanged(self):
        assert _edge_bucket(0.03) == "lt_0.05"
        assert _edge_bucket(0.07) == "0.05_0.10"
        assert _edge_bucket(0.15) == "0.10_0.20"
        assert _edge_bucket(0.25) == "ge_0.20"

    def test_price_bucket_unchanged(self):
        assert _price_bucket(30) == "lt_40"
        assert _price_bucket(45) == "40_55"
        assert _price_bucket(60) == "55_70"
        assert _price_bucket(75) == "70_78"
        assert _price_bucket(80) == "ge_78"

    def test_tts_bucket_unchanged(self):
        assert _tts_bucket(None) == "unknown"
        assert _tts_bucket(30) == "lt_1m"
        assert _tts_bucket(120) == "1_3m"
        assert _tts_bucket(300) == "3_10m"
        assert _tts_bucket(1000) == "10_30m"
        assert _tts_bucket(3600) == "ge_30m"


# ===========================================================================
# 6. Bounded data structures
# ===========================================================================

class TestBoundedStructures:
    def test_context_policy_window_bounded(self):
        engine = ContextPolicyEngine(_make_cfg(
            enable_context_policy=True,
            context_policy_window_settles=10,
        ))
        key = "test|yes|lt_0.05|lt_40|na|far"
        for i in range(50):
            engine.record_settlement(key, float(i))
        # Internal window should be capped at 10
        with engine._lock:
            assert len(engine._windows[key]) == 10

    def test_drift_guard_window_bounded(self):
        guard = DriftGuard(_make_cfg(
            enable_drift_guard=True,
            drift_guard_window_settles=10,
        ))
        key = "test|yes|lt_0.05|lt_40|na|far"
        for i in range(50):
            guard.record_settlement(key, -1.0)
        with guard._lock:
            assert len(guard._windows[key]) == 10

    def test_edge_tracker_window_bounded(self):
        tracker = EdgeTracker(_make_cfg(
            enable_edge_tracking=True,
            edge_tracking_window_settles=10,
            edge_tracking_min_samples=3,
        ))
        key = "test|yes|lt_0.05|lt_40|na|far"
        for i in range(50):
            tracker.record_entry(key, expected_edge=0.10, entry_price_cents=40)
            tracker.record_settlement(key, net_pnl_usd=0.04, entry_price_cents=40, quantity=1)
        with tracker._lock:
            assert len(tracker._expected[key]) == 10
            assert len(tracker._realized[key]) == 10

    def test_population_scaler_events_bounded(self):
        scaler = PopulationScaler(_make_cfg(
            bot_population_scale_enabled=True,
            bot_population_scale_schedule=[1.0, 1.15, 1.30],
            bot_population_scale_require_passes=1,
            bot_population_scale_cooldown_hours=0.0,
            bot_population_scale_max_step=0.20,
            bot_population_scale_min_window_hours=0.0,
        ))
        # The events deque has maxlen=50
        assert scaler._events.maxlen == 50

    def test_adaptive_cap_window_bounded(self):
        engine = AdaptiveCapEngine(_make_cfg(
            enable_adaptive_caps=True,
            adaptive_cap_min_samples=5,
        ))
        key = "test|yes|lt_0.05|lt_40|na|far"
        for i in range(50):
            engine.record_settlement(key, -1.0, concentration_share=0.3)
        with engine._lock:
            assert len(engine._windows[key]) == 5
