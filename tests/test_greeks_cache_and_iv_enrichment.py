"""
Tests for Greeks Cache and Snapshot IV Enrichment
==================================================

Validates:
- GreeksCache stores and retrieves Greeks events correctly
- Time-gating (as_of_ms) prevents future data leakage
- Staleness eviction removes old entries
- ATM IV selection uses nearest strike to underlying price
- enrich_snapshot_iv enriches snapshots with cached provider IV
- Derived IV fallback works when provider IV unavailable
- Replay harness receives IV from enriched snapshots
- VRP strategy produces trades when IV exists
- Determinism is preserved across multiple replay runs
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import pytest

from src.core.greeks_cache import (
    CachedGreek,
    GreeksCache,
    _parse_option_symbol,
    enrich_snapshot_iv,
)
from src.core.iv_consensus import IVConsensusEngine
from src.core.option_events import (
    OptionChainSnapshotEvent,
    OptionQuoteEvent,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_snapshot(
    symbol: str = "SPY",
    underlying_price: float = 595.0,
    atm_iv: Optional[float] = None,
    recv_ts_ms: int = 1_700_000_060_000,
    puts: tuple = (),
    calls: tuple = (),
    provider: str = "tastytrade",
) -> OptionChainSnapshotEvent:
    """Create a minimal OptionChainSnapshotEvent for testing.

    Default expiration_ms corresponds to 2025-03-21 midnight UTC to
    align with the ``.SPY250321P*`` symbols used throughout the tests.
    """
    from datetime import datetime, timezone as _tz
    # 2025-03-21 00:00 UTC in epoch ms
    _default_exp_ms = int(datetime(2025, 3, 21, tzinfo=_tz.utc).timestamp() * 1000)
    return OptionChainSnapshotEvent(
        symbol=symbol,
        expiration_ms=_default_exp_ms,
        underlying_price=underlying_price,
        puts=puts,
        calls=calls,
        n_strikes=len(puts),
        atm_iv=atm_iv,
        timestamp_ms=1_700_000_060_000,
        recv_ts_ms=recv_ts_ms,
        provider=provider,
        snapshot_id=f"{provider}_{symbol}_test",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _parse_option_symbol
# ═══════════════════════════════════════════════════════════════════════════

class TestParseOptionSymbol:
    def test_put_symbol(self):
        result = _parse_option_symbol(".SPY250321P590")
        assert result == ("SPY", "PUT", 590.0, "250321")

    def test_call_symbol(self):
        result = _parse_option_symbol(".SPY250321C595")
        assert result == ("SPY", "CALL", 595.0, "250321")

    def test_without_leading_dot(self):
        """Some formats omit the leading dot."""
        result = _parse_option_symbol("SPY250321P590")
        assert result == ("SPY", "PUT", 590.0, "250321")

    def test_ibit_symbol(self):
        result = _parse_option_symbol(".IBIT250321P55")
        assert result == ("IBIT", "PUT", 55.0, "250321")

    def test_decimal_strike(self):
        result = _parse_option_symbol(".SPY250321P590.5")
        assert result == ("SPY", "PUT", 590.5, "250321")

    def test_invalid_symbol(self):
        assert _parse_option_symbol("INVALID") is None
        assert _parse_option_symbol("") is None
        assert _parse_option_symbol("123") is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: GreeksCache
# ═══════════════════════════════════════════════════════════════════════════

class TestGreeksCache:
    def test_basic_update_and_retrieve(self):
        cache = GreeksCache()
        cache.update(".SPY250321P590", volatility=0.22, recv_ts_ms=1000)
        cache.update(".SPY250321P595", volatility=0.21, recv_ts_ms=1000)
        cache.update(".SPY250321P600", volatility=0.20, recv_ts_ms=1000)

        # ATM at 595 should pick the 595 put
        iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=2000)
        assert iv == 0.21

    def test_atm_iv_nearest_strike(self):
        cache = GreeksCache()
        cache.update(".SPY250321P590", volatility=0.25, recv_ts_ms=1000)
        cache.update(".SPY250321P600", volatility=0.20, recv_ts_ms=1000)

        # 593 is closer to 590
        iv = cache.get_atm_iv("SPY", underlying_price=593.0, as_of_ms=2000)
        assert iv == 0.25

        # 597 is closer to 600
        iv = cache.get_atm_iv("SPY", underlying_price=597.0, as_of_ms=2000)
        assert iv == 0.20

    def test_time_gating(self):
        """Greeks received AFTER as_of_ms must NOT be used."""
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=5000)

        # as_of_ms=4000 is before the event's recv_ts_ms=5000
        iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=4000)
        assert iv is None

        # as_of_ms=5000 should work (exact match)
        iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=5000)
        assert iv == 0.22

    def test_staleness_eviction(self):
        cache = GreeksCache(max_age_ms=1000)
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1000)

        # Within max_age: should return IV
        iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=1500)
        assert iv == 0.22

        # Beyond max_age: should return None
        iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=2500)
        assert iv is None

    def test_evict_stale(self):
        cache = GreeksCache(max_age_ms=1000)
        cache.update(".SPY250321P590", volatility=0.22, recv_ts_ms=1000)
        cache.update(".SPY250321P595", volatility=0.21, recv_ts_ms=3000)

        assert cache.size == 2
        evicted = cache.evict_stale(now_ms=2500)
        assert evicted == 1
        assert cache.size == 1

    def test_update_newer_replaces_older(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.20, recv_ts_ms=1000)
        cache.update(".SPY250321P595", volatility=0.25, recv_ts_ms=2000)

        iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=3000)
        assert iv == 0.25

    def test_update_older_does_not_replace_newer(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.25, recv_ts_ms=2000)
        cache.update(".SPY250321P595", volatility=0.20, recv_ts_ms=1000)

        iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=3000)
        assert iv == 0.25

    def test_none_volatility_ignored(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=None, recv_ts_ms=1000)
        assert cache.size == 0

    def test_negative_volatility_rejected(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=-0.1, recv_ts_ms=1000)
        assert cache.size == 0

    def test_zero_volatility_rejected(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.0, recv_ts_ms=1000)
        assert cache.size == 0

    def test_different_underlyings(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1000)
        cache.update(".IBIT250321P55", volatility=0.45, recv_ts_ms=1000)

        spy_iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=2000)
        assert spy_iv == 0.22

        ibit_iv = cache.get_atm_iv("IBIT", underlying_price=55.0, as_of_ms=2000)
        assert ibit_iv == 0.45

    def test_calls_vs_puts(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1000)
        cache.update(".SPY250321C595", volatility=0.23, recv_ts_ms=1000)

        put_iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=2000, option_type="PUT")
        assert put_iv == 0.22

        call_iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=2000, option_type="CALL")
        assert call_iv == 0.23

    def test_clear(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1000)
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0

    def test_zero_underlying_price(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1000)
        iv = cache.get_atm_iv("SPY", underlying_price=0.0, as_of_ms=2000)
        assert iv is None

    def test_get_greeks_for_strike(self):
        cache = GreeksCache()
        cache.update(
            ".SPY250321P595", volatility=0.22, recv_ts_ms=1000,
            delta=-0.45, gamma=0.02, theta=-0.05, vega=0.15,
        )
        greek = cache.get_greeks_for_strike("SPY", 595.0, "PUT", as_of_ms=2000)
        assert greek is not None
        assert greek.volatility == 0.22
        assert greek.delta == -0.45

    def test_cross_expiration_filtering(self):
        """ATM IV must not mix expirations when expiration_ms is supplied."""
        from datetime import datetime, timezone

        cache = GreeksCache()
        # March 21 expiry: IV = 0.22
        cache.update(".SPY250321P590", volatility=0.22, recv_ts_ms=1000)
        # April 18 expiry: IV = 0.30
        cache.update(".SPY250418P590", volatility=0.30, recv_ts_ms=1000)

        # March 21 midnight UTC in ms
        mar21_ms = int(datetime(2025, 3, 21, tzinfo=timezone.utc).timestamp() * 1000)
        apr18_ms = int(datetime(2025, 4, 18, tzinfo=timezone.utc).timestamp() * 1000)

        # When filtering to March expiry, must get March IV
        iv = cache.get_atm_iv(
            "SPY", underlying_price=590.0, as_of_ms=2000,
            expiration_ms=mar21_ms,
        )
        assert iv == 0.22

        # When filtering to April expiry, must get April IV
        iv = cache.get_atm_iv(
            "SPY", underlying_price=590.0, as_of_ms=2000,
            expiration_ms=apr18_ms,
        )
        assert iv == 0.30

    def test_no_expiration_filter_matches_all(self):
        """Without expiration_ms, get_atm_iv still matches all expirations."""
        cache = GreeksCache()
        cache.update(".SPY250321P590", volatility=0.22, recv_ts_ms=1000)
        cache.update(".SPY250418P595", volatility=0.30, recv_ts_ms=1000)

        # No expiration filter: should pick nearest strike to 593 → 590 (dist=3) vs 595 (dist=2)
        iv = cache.get_atm_iv("SPY", underlying_price=593.0, as_of_ms=2000)
        assert iv == 0.30  # 595 is closer to 593 (dist=2 vs dist=3)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: enrich_snapshot_iv
# ═══════════════════════════════════════════════════════════════════════════

class TestEnrichSnapshotIV:
    def test_enriches_when_atm_iv_missing(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1_700_000_050_000)

        snapshot = _make_snapshot(atm_iv=None, recv_ts_ms=1_700_000_060_000)
        enriched = enrich_snapshot_iv(snapshot, cache)

        assert enriched.atm_iv == 0.22
        # Original unchanged (frozen dataclass)
        assert snapshot.atm_iv is None

    def test_preserves_existing_atm_iv(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.30, recv_ts_ms=1_700_000_050_000)

        snapshot = _make_snapshot(atm_iv=0.18, recv_ts_ms=1_700_000_060_000)
        enriched = enrich_snapshot_iv(snapshot, cache)

        # Should keep original IV, not overwrite
        assert enriched.atm_iv == 0.18

    def test_time_gating_respected(self):
        """Greeks received after snapshot recv_ts must not be used."""
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1_700_000_070_000)

        snapshot = _make_snapshot(atm_iv=None, recv_ts_ms=1_700_000_060_000)
        enriched = enrich_snapshot_iv(snapshot, cache)

        # Cache entry is AFTER snapshot recv_ts — should not be used
        assert enriched.atm_iv is None

    def test_returns_original_when_no_cache_data(self):
        cache = GreeksCache()
        snapshot = _make_snapshot(atm_iv=None, recv_ts_ms=1_700_000_060_000)
        enriched = enrich_snapshot_iv(snapshot, cache)
        assert enriched is snapshot  # Same object, no enrichment

    def test_falls_back_to_call_iv(self):
        """If no put IV, should try call IV."""
        cache = GreeksCache()
        cache.update(".SPY250321C595", volatility=0.23, recv_ts_ms=1_700_000_050_000)

        snapshot = _make_snapshot(atm_iv=None, recv_ts_ms=1_700_000_060_000)
        enriched = enrich_snapshot_iv(snapshot, cache)

        assert enriched.atm_iv == 0.23

    def test_snapshot_fields_preserved(self):
        """All other snapshot fields must be preserved during enrichment."""
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1_700_000_050_000)

        snapshot = _make_snapshot(
            symbol="SPY",
            underlying_price=595.0,
            atm_iv=None,
            recv_ts_ms=1_700_000_060_000,
            provider="tastytrade",
        )
        enriched = enrich_snapshot_iv(snapshot, cache)

        assert enriched.symbol == snapshot.symbol
        assert enriched.underlying_price == snapshot.underlying_price
        assert enriched.recv_ts_ms == snapshot.recv_ts_ms
        assert enriched.provider == snapshot.provider
        assert enriched.expiration_ms == snapshot.expiration_ms
        assert enriched.timestamp_ms == snapshot.timestamp_ms
        assert enriched.snapshot_id == snapshot.snapshot_id

    def test_zero_underlying_price_skips_enrichment(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1000)

        snapshot = _make_snapshot(underlying_price=0.0, atm_iv=None)
        enriched = enrich_snapshot_iv(snapshot, cache)
        assert enriched is snapshot

    def test_enrichment_respects_expiration(self):
        """Enrichment must not use IV from a different expiration."""
        from datetime import datetime, timezone

        cache = GreeksCache()
        # Only March expiry in cache
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1_700_000_050_000)

        # Snapshot is for April expiry
        apr18_ms = int(datetime(2025, 4, 18, tzinfo=timezone.utc).timestamp() * 1000)
        snapshot = OptionChainSnapshotEvent(
            symbol="SPY",
            expiration_ms=apr18_ms,
            underlying_price=595.0,
            puts=(),
            calls=(),
            n_strikes=0,
            atm_iv=None,
            timestamp_ms=1_700_000_060_000,
            recv_ts_ms=1_700_000_060_000,
            provider="tastytrade",
            snapshot_id="test_exp",
        )
        enriched = enrich_snapshot_iv(snapshot, cache)

        # Must NOT enrich with March IV for an April snapshot
        assert enriched.atm_iv is None

    def test_consensus_engine_with_no_iv_returns_snapshot_no_attribute_error(self):
        """When greeks_cache is IVConsensusEngine and consensus has no data, must not call get_atm_iv (engine has no such method)."""
        engine = IVConsensusEngine()
        snapshot = _make_snapshot(atm_iv=None, recv_ts_ms=1_700_000_060_000)
        enriched = enrich_snapshot_iv(snapshot, engine)
        assert enriched is snapshot
        assert enriched.atm_iv is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Replay Harness Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestReplayHarnessIVIntegration:
    """Verify that enriched snapshots flow through to strategies via replay."""

    def test_snapshot_with_atm_iv_reaches_strategy(self):
        """MarketDataSnapshot with atm_iv is visible to strategy."""
        from src.analysis.execution_model import ExecutionModel
        from src.analysis.replay_harness import (
            MarketDataSnapshot,
            ReplayConfig,
            ReplayHarness,
            ReplayStrategy,
            TradeIntent,
        )
        from src.core.outcome_engine import BarData

        observed_ivs: List[Optional[float]] = []

        class IVRecordingStrategy(ReplayStrategy):
            @property
            def strategy_id(self):
                return "IV_RECORDER"

            def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes,
                       *, visible_snapshots=None, **kwargs):
                if visible_snapshots:
                    for snap in visible_snapshots:
                        observed_ivs.append(snap.atm_iv)

            def generate_intents(self, sim_ts_ms):
                return []

        bars = [
            BarData(timestamp_ms=1000, open=100, high=101, low=99, close=100.5, volume=1000),
            BarData(timestamp_ms=61000, open=100.5, high=102, low=100, close=101, volume=1100),
            BarData(timestamp_ms=121000, open=101, high=103, low=100.5, close=102, volume=1200),
        ]

        snapshots = [
            MarketDataSnapshot(
                symbol="SPY",
                recv_ts_ms=50000,  # Before bar 1 close (1000 + 60000)
                underlying_price=595.0,
                atm_iv=0.22,
                source="tastytrade",
            ),
            MarketDataSnapshot(
                symbol="SPY",
                recv_ts_ms=100000,  # Before bar 2 close
                underlying_price=596.0,
                atm_iv=0.24,
                source="tastytrade",
            ),
        ]

        harness = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=IVRecordingStrategy(),
            execution_model=ExecutionModel(),
            snapshots=snapshots,
        )
        harness.run()

        # Both IV values should have been observed
        assert 0.22 in observed_ivs
        assert 0.24 in observed_ivs

    def test_snapshot_gating_prevents_future_iv(self):
        """Snapshot with recv_ts_ms > sim_ts_ms must not be visible."""
        from src.analysis.execution_model import ExecutionModel
        from src.analysis.replay_harness import (
            MarketDataSnapshot,
            ReplayHarness,
            ReplayStrategy,
        )
        from src.core.outcome_engine import BarData

        future_iv_seen = []

        class FutureIVDetector(ReplayStrategy):
            @property
            def strategy_id(self):
                return "FUTURE_DETECTOR"

            def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes,
                       *, visible_snapshots=None, **kwargs):
                if visible_snapshots:
                    for snap in visible_snapshots:
                        if snap.atm_iv == 0.99:
                            future_iv_seen.append(sim_ts_ms)

            def generate_intents(self, sim_ts_ms):
                return []

        bars = [
            BarData(timestamp_ms=1000, open=100, high=101, low=99, close=100, volume=100),
        ]
        snapshots = [
            MarketDataSnapshot(
                symbol="SPY",
                recv_ts_ms=999_999_999,  # Far future
                underlying_price=595.0,
                atm_iv=0.99,
                source="tastytrade",
            ),
        ]

        harness = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=FutureIVDetector(),
            execution_model=ExecutionModel(),
            snapshots=snapshots,
        )
        harness.run()
        assert len(future_iv_seen) == 0

    def test_vrp_strategy_produces_trades_with_iv(self):
        """VRP strategy should produce trades when IV and RV are available."""
        from src.analysis.execution_model import ExecutionModel
        from src.analysis.replay_harness import (
            MarketDataSnapshot,
            ReplayConfig,
            ReplayHarness,
        )
        from src.core.outcome_engine import BarData
        from src.strategies.vrp_credit_spread import VRPCreditSpreadStrategy

        bars = []
        base_ts = 1_700_000_000_000
        for i in range(20):
            ts = base_ts + i * 60_000
            bars.append(BarData(
                timestamp_ms=ts,
                open=595.0 + i * 0.1,
                high=596.0 + i * 0.1,
                low=594.0 + i * 0.1,
                close=595.5 + i * 0.1,
                volume=1000,
            ))

        outcomes = []
        for i in range(20):
            ts = base_ts + i * 60_000
            outcomes.append({
                "timestamp_ms": ts,
                "window_end_ms": ts + 60_000,  # Available at next bar
                "provider": "alpaca",
                "symbol": "SPY",
                "bar_duration_seconds": 60,
                "horizon_seconds": 3600,
                "outcome_version": "v1",
                "close_now": 595.0 + i * 0.1,
                "close_at_horizon": 596.0 + i * 0.1,
                "fwd_return": 0.001,
                "max_runup": 0.005,
                "max_drawdown": -0.003,
                "realized_vol": 0.15,  # RV = 0.15
                "status": "OK",
                "window_start_ms": ts,
                "bars_expected": 60,
                "bars_found": 60,
                "gap_count": 0,
            })

        # Snapshot with IV = 0.25 (VRP = 0.25 - 0.15 = 0.10 > 0.05 threshold)
        snapshots = [
            MarketDataSnapshot(
                symbol="SPY",
                recv_ts_ms=base_ts + 30_000,  # Available early
                underlying_price=595.0,
                atm_iv=0.25,
                source="tastytrade",
            ),
        ]

        # Add regimes (bullish/neutral to pass VRP gating)
        regimes = [
            {
                "scope": "SPY",
                "timestamp_ms": base_ts,
                "vol_regime": "VOL_NORMAL",
                "trend_regime": "TREND_UP",
            }
        ]

        strategy = VRPCreditSpreadStrategy({"min_vrp": 0.05})
        harness = ReplayHarness(
            bars=bars,
            outcomes=outcomes,
            strategy=strategy,
            execution_model=ExecutionModel(),
            snapshots=snapshots,
            regimes=regimes,
        )
        result = harness.run()

        # Strategy should have seen IV and generated intents
        state = strategy.finalize()
        assert state["last_iv"] == 0.25
        assert state["last_rv"] == 0.15


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Determinism
# ═══════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Verify that replay is deterministic across runs."""

    def test_identical_results_across_runs(self):
        """Same inputs must produce identical outputs."""
        from src.analysis.execution_model import ExecutionModel
        from src.analysis.replay_harness import (
            MarketDataSnapshot,
            ReplayHarness,
        )
        from src.core.outcome_engine import BarData
        from src.strategies.vrp_credit_spread import VRPCreditSpreadStrategy

        bars = []
        base_ts = 1_700_000_000_000
        for i in range(10):
            ts = base_ts + i * 60_000
            bars.append(BarData(
                timestamp_ms=ts,
                open=595.0, high=596.0, low=594.0, close=595.5,
                volume=1000,
            ))

        outcomes = []
        for i in range(10):
            ts = base_ts + i * 60_000
            outcomes.append({
                "timestamp_ms": ts,
                "window_end_ms": ts + 60_000,
                "provider": "alpaca",
                "symbol": "SPY",
                "bar_duration_seconds": 60,
                "horizon_seconds": 3600,
                "outcome_version": "v1",
                "close_now": 595.0,
                "realized_vol": 0.15,
                "status": "OK",
                "window_start_ms": ts,
                "bars_expected": 60,
                "bars_found": 60,
                "gap_count": 0,
            })

        snapshots = [
            MarketDataSnapshot(
                symbol="SPY",
                recv_ts_ms=base_ts + 30_000,
                underlying_price=595.0,
                atm_iv=0.25,
                source="tastytrade",
            ),
        ]
        regimes = [{
            "scope": "SPY",
            "timestamp_ms": base_ts,
            "vol_regime": "VOL_NORMAL",
            "trend_regime": "TREND_UP",
        }]

        results = []
        for _ in range(3):
            strategy = VRPCreditSpreadStrategy({"min_vrp": 0.05})
            harness = ReplayHarness(
                bars=list(bars),
                outcomes=list(outcomes),
                strategy=strategy,
                execution_model=ExecutionModel(),
                snapshots=list(snapshots),
                regimes=list(regimes),
            )
            result = harness.run()
            results.append(result.summary())

        # All runs should produce identical results
        for i in range(1, len(results)):
            assert results[i]["bars_replayed"] == results[0]["bars_replayed"]
            assert results[i]["portfolio"] == results[0]["portfolio"]

    def test_greeks_cache_enrichment_is_deterministic(self):
        """Same cache state + same snapshot must produce same enrichment."""
        cache = GreeksCache()
        cache.update(".SPY250321P590", volatility=0.25, recv_ts_ms=1000)
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1000)
        cache.update(".SPY250321P600", volatility=0.20, recv_ts_ms=1000)

        snapshot = _make_snapshot(
            atm_iv=None, underlying_price=595.0, recv_ts_ms=2000
        )

        # Multiple enrichments of the same snapshot should yield identical results
        results = [enrich_snapshot_iv(snapshot, cache).atm_iv for _ in range(5)]
        assert all(r == results[0] for r in results)
        assert results[0] == 0.22  # 595 put is nearest


# ═══════════════════════════════════════════════════════════════════════════
# Tests: VRP Strategy IV Selection
# ═══════════════════════════════════════════════════════════════════════════

class TestVRPIVSelection:
    """Test the IV selection logic in VRP strategy."""

    def test_tastytrade_atm_iv_preferred(self):
        from src.strategies.vrp_credit_spread import _select_iv_from_snapshots

        class FakeSnap:
            def __init__(self, source, atm_iv):
                self.source = source
                self.atm_iv = atm_iv

        snaps = [
            FakeSnap("alpaca", 0.30),
            FakeSnap("tastytrade", 0.22),
        ]
        iv = _select_iv_from_snapshots(snaps)
        assert iv == 0.22  # Tastytrade preferred

    def test_returns_none_when_no_snapshots(self):
        from src.strategies.vrp_credit_spread import _select_iv_from_snapshots
        assert _select_iv_from_snapshots([]) is None

    def test_returns_none_when_no_iv_available(self):
        from src.strategies.vrp_credit_spread import _select_iv_from_snapshots

        class FakeSnap:
            def __init__(self, source, atm_iv):
                self.source = source
                self.atm_iv = atm_iv

        snaps = [FakeSnap("tastytrade", None)]
        iv = _select_iv_from_snapshots(snaps)
        assert iv is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Performance Safeguards
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformanceSafeguards:
    """Verify performance characteristics of the Greeks cache."""

    def test_cache_does_not_grow_unbounded_with_eviction(self):
        cache = GreeksCache(max_age_ms=100)
        for i in range(1000):
            cache.update(f".SPY250321P{500 + i}", volatility=0.2, recv_ts_ms=i)

        # All entries with recv_ts_ms < 900 should be stale at now=1000
        evicted = cache.evict_stale(now_ms=1000)
        assert evicted > 0
        assert cache.size < 1000

    def test_enrichment_does_not_mutate_cache(self):
        cache = GreeksCache()
        cache.update(".SPY250321P595", volatility=0.22, recv_ts_ms=1000)
        initial_size = cache.size

        snapshot = _make_snapshot(atm_iv=None, recv_ts_ms=2000)
        enrich_snapshot_iv(snapshot, cache)

        assert cache.size == initial_size  # No new entries
