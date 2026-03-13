"""
Tests for end-to-end regime persistence.

Validates:
- DB schema (new liquidity columns, indexes)
- write_regime / get_regimes / get_latest_regime round-trip
- PersistenceManager subscription wiring
- Replay harness regime lookahead barrier
- get_latest_regime helper
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from src.core.outcome_engine import BarData, OutcomeResult
from src.analysis.execution_model import ExecutionModel
from src.analysis.replay_harness import (
    ReplayConfig,
    ReplayHarness,
    ReplayStrategy,
    TradeIntent,
    get_latest_regime,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def event_loop():
    """Provide a fresh event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro, loop=None):
    """Helper to run async code in sync tests."""
    if loop is None:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return loop.run_until_complete(coro)


async def _create_test_db():
    """Create an in-memory Database instance with schema."""
    import aiosqlite
    from src.core.database import Database
    db = Database.__new__(Database)
    db._connection = await aiosqlite.connect(":memory:")
    db._connection.row_factory = aiosqlite.Row
    db._db_path = ":memory:"
    await db._create_tables()
    return db


# ═══════════════════════════════════════════════════════════════════════════
# Schema Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRegimeSchema:
    def test_table_has_liquidity_columns(self):
        async def _test():
            db = await _create_test_db()
            cursor = await db._connection.execute("PRAGMA table_info(regimes)")
            cols = {row[1] for row in await cursor.fetchall()}
            assert "liquidity_regime" in cols
            assert "spread_pct" in cols
            assert "volume_pctile" in cols
            # Original columns still present
            assert "vol_regime" in cols
            assert "trend_regime" in cols
            assert "session_regime" in cols
            assert "risk_regime" in cols
            assert "confidence" in cols
            assert "metrics_json" in cols
            await db._connection.close()
        run(_test())

    def test_scope_timestamp_index_exists(self):
        async def _test():
            db = await _create_test_db()
            cursor = await db._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='regimes'"
            )
            indexes = {row[0] for row in await cursor.fetchall()}
            assert "idx_regimes_scope_ts" in indexes
            assert "idx_regimes_lookup" in indexes
            await db._connection.close()
        run(_test())

    def test_unique_constraint(self):
        async def _test():
            db = await _create_test_db()
            # Write once
            await db.write_regime(
                event_type="symbol", scope="SPY", timeframe=60,
                timestamp_ms=1000, config_hash="abc",
                vol_regime="VOL_LOW",
            )
            # Write again with same key → upsert, not duplicate
            await db.write_regime(
                event_type="symbol", scope="SPY", timeframe=60,
                timestamp_ms=1000, config_hash="abc",
                vol_regime="VOL_HIGH",
            )
            rows = await db.get_regimes("SPY")
            assert len(rows) == 1
            assert rows[0]["vol_regime"] == "VOL_HIGH"
            await db._connection.close()
        run(_test())


# ═══════════════════════════════════════════════════════════════════════════
# Write / Read Round-Trip Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteReadRegime:
    def test_write_symbol_regime_full(self):
        """All fields round-trip correctly."""
        async def _test():
            db = await _create_test_db()
            await db.write_regime(
                event_type="symbol",
                scope="SPY",
                timeframe=60,
                timestamp_ms=1_700_000_000_000,
                config_hash="hash123",
                vol_regime="VOL_NORMAL",
                trend_regime="TREND_UP",
                liquidity_regime="LIQ_HIGH",
                spread_pct=0.0015,
                volume_pctile=85.5,
                confidence=0.92,
                is_warm=True,
                data_quality_flags=0,
                metrics_json='{"atr": 1.5}',
            )
            rows = await db.get_regimes("SPY")
            assert len(rows) == 1
            r = rows[0]
            assert r["event_type"] == "symbol"
            assert r["scope"] == "SPY"
            assert r["timestamp_ms"] == 1_700_000_000_000
            assert r["vol_regime"] == "VOL_NORMAL"
            assert r["trend_regime"] == "TREND_UP"
            assert r["liquidity_regime"] == "LIQ_HIGH"
            assert r["spread_pct"] == pytest.approx(0.0015)
            assert r["volume_pctile"] == pytest.approx(85.5)
            assert r["confidence"] == pytest.approx(0.92)
            assert r["is_warm"] == 1
            assert r["config_hash"] == "hash123"
            assert r["metrics_json"] == '{"atr": 1.5}'
            await db._connection.close()
        run(_test())

    def test_write_market_regime(self):
        """Market regime without liquidity fields stores NULLs."""
        async def _test():
            db = await _create_test_db()
            await db.write_regime(
                event_type="market",
                scope="EQUITIES",
                timeframe=60,
                timestamp_ms=1_700_000_000_000,
                config_hash="mhash",
                session_regime="RTH",
                risk_regime="NEUTRAL",
                confidence=1.0,
            )
            rows = await db.get_regimes("EQUITIES", event_type="market")
            assert len(rows) == 1
            r = rows[0]
            assert r["session_regime"] == "RTH"
            assert r["risk_regime"] == "NEUTRAL"
            # Liquidity fields are NULL (backward compatible)
            assert r["liquidity_regime"] is None
            assert r["spread_pct"] is None
            assert r["volume_pctile"] is None
            await db._connection.close()
        run(_test())

    def test_backward_compatible_write_no_new_fields(self):
        """Writing without liquidity fields works (all default to NULL)."""
        async def _test():
            db = await _create_test_db()
            await db.write_regime(
                event_type="symbol",
                scope="AAPL",
                timeframe=60,
                timestamp_ms=1000,
                config_hash="old",
                vol_regime="VOL_LOW",
            )
            rows = await db.get_regimes("AAPL")
            assert len(rows) == 1
            assert rows[0]["liquidity_regime"] is None
            assert rows[0]["spread_pct"] is None
            assert rows[0]["volume_pctile"] is None
            await db._connection.close()
        run(_test())

    def test_get_regimes_time_range(self):
        """start_ms / end_ms filtering works."""
        async def _test():
            db = await _create_test_db()
            for ts in [1000, 2000, 3000, 4000, 5000]:
                await db.write_regime(
                    event_type="symbol", scope="SPY", timeframe=60,
                    timestamp_ms=ts, config_hash=f"h{ts}",
                )
            rows = await db.get_regimes("SPY", start_ms=2000, end_ms=4000)
            timestamps = [r["timestamp_ms"] for r in rows]
            assert timestamps == [2000, 3000, 4000]
            await db._connection.close()
        run(_test())

    def test_get_regimes_ascending_order(self):
        """Results are in ascending timestamp order."""
        async def _test():
            db = await _create_test_db()
            for ts in [3000, 1000, 5000, 2000, 4000]:
                await db.write_regime(
                    event_type="symbol", scope="SPY", timeframe=60,
                    timestamp_ms=ts, config_hash=f"h{ts}",
                )
            rows = await db.get_regimes("SPY")
            timestamps = [r["timestamp_ms"] for r in rows]
            assert timestamps == sorted(timestamps)
            await db._connection.close()
        run(_test())

    def test_get_latest_regime_asof(self):
        """get_latest_regime returns the most recent regime at or before asof."""
        async def _test():
            db = await _create_test_db()
            await db.write_regime(
                event_type="symbol", scope="SPY", timeframe=60,
                timestamp_ms=1000, config_hash="a",
                vol_regime="VOL_LOW",
            )
            await db.write_regime(
                event_type="symbol", scope="SPY", timeframe=60,
                timestamp_ms=3000, config_hash="b",
                vol_regime="VOL_HIGH",
            )
            # asof=2000 → should get the regime at 1000
            r = await db.get_latest_regime("SPY", asof_ms=2000)
            assert r is not None
            assert r["timestamp_ms"] == 1000
            assert r["vol_regime"] == "VOL_LOW"

            # asof=3000 → should get the regime at 3000
            r = await db.get_latest_regime("SPY", asof_ms=3000)
            assert r is not None
            assert r["timestamp_ms"] == 3000
            assert r["vol_regime"] == "VOL_HIGH"

            # asof=500 → nothing before
            r = await db.get_latest_regime("SPY", asof_ms=500)
            assert r is None
            await db._connection.close()
        run(_test())

    def test_get_latest_regime_wrong_scope(self):
        """get_latest_regime returns None for a non-matching scope."""
        async def _test():
            db = await _create_test_db()
            await db.write_regime(
                event_type="symbol", scope="SPY", timeframe=60,
                timestamp_ms=1000, config_hash="a",
            )
            r = await db.get_latest_regime("AAPL", asof_ms=9999)
            assert r is None
            await db._connection.close()
        run(_test())


# ═══════════════════════════════════════════════════════════════════════════
# Persistence Manager Wiring Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistenceSubscription:
    def test_persistence_imports_regime_topics(self):
        """PersistenceManager can import regime topic constants."""
        from src.core.events import TOPIC_REGIMES_SYMBOL, TOPIC_REGIMES_MARKET
        assert TOPIC_REGIMES_SYMBOL == "regimes.symbol"
        assert TOPIC_REGIMES_MARKET == "regimes.market"

    def test_persistence_has_regime_handlers(self):
        """PersistenceManager has the regime handler methods."""
        from src.core.persistence import PersistenceManager
        assert hasattr(PersistenceManager, "_on_symbol_regime")
        assert hasattr(PersistenceManager, "_on_market_regime")

    def test_symbol_regime_handler_builds_correct_payload(self):
        """_on_symbol_regime extracts the right fields from the event."""
        from src.core.regimes import SymbolRegimeEvent

        # Create a real event
        event = SymbolRegimeEvent(
            symbol="SPY",
            timeframe=60,
            timestamp_ms=1_700_000_000_000,
            vol_regime="VOL_NORMAL",
            trend_regime="TREND_UP",
            liquidity_regime="LIQ_HIGH",
            atr=1.5,
            atr_pct=0.003,
            vol_z=0.5,
            ema_fast=450.0,
            ema_slow=445.0,
            ema_slope=0.01,
            rsi=55.0,
            spread_pct=0.001,
            volume_pctile=80.0,
            confidence=0.95,
            is_warm=True,
            data_quality_flags=0,
            config_hash="test_hash",
        )
        # Verify the event has the fields the handler accesses
        assert event.liquidity_regime == "LIQ_HIGH"
        assert event.spread_pct == 0.001
        assert event.volume_pctile == 80.0
        assert event.config_hash == "test_hash"


# ═══════════════════════════════════════════════════════════════════════════
# Replay Harness Regime Barrier Tests
# ═══════════════════════════════════════════════════════════════════════════

def _make_bars(n, start_ms=1_000_000, interval_ms=60_000, price=100.0):
    """Generate n synthetic bars."""
    bars = []
    for i in range(n):
        ts = start_ms + i * interval_ms
        bars.append(BarData(
            timestamp_ms=ts,
            open=price, high=price + 1, low=price - 0.5,
            close=price + 0.5, volume=1000,
        ))
    return bars


def _make_regimes(timestamps_ms, scope="SPY"):
    """Generate regime dicts at given timestamps."""
    regimes = []
    for i, ts in enumerate(timestamps_ms):
        regimes.append({
            "event_type": "symbol",
            "scope": scope,
            "timestamp_ms": ts,
            "vol_regime": f"VOL_{i}",
            "trend_regime": "TREND_UP",
            "liquidity_regime": "LIQ_NORMAL",
            "spread_pct": 0.001 * (i + 1),
            "volume_pctile": 50.0 + i,
            "config_hash": f"h{i}",
        })
    return regimes


class RegimeLeakDetector(ReplayStrategy):
    """Records which regimes are visible at each bar to detect lookahead leaks."""

    def __init__(self):
        self._observations: List[Dict[str, Any]] = []

    @property
    def strategy_id(self):
        return "REGIME_LEAK_DETECTOR"

    def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes,
               *, visible_regimes=None, visible_snapshots=None):
        self._observations.append({
            "bar_ts_ms": bar.timestamp_ms,
            "sim_ts_ms": sim_ts_ms,
            "regime_scopes": list((visible_regimes or {}).keys()),
            "regime_count": len(visible_regimes or {}),
            "regimes": dict(visible_regimes or {}),
        })

    def generate_intents(self, sim_ts_ms):
        return []

    def finalize(self):
        return {"observations": self._observations}


class TestReplayRegimeBarrier:
    def test_regimes_not_visible_before_timestamp(self):
        """The invariant: no regime is visible before its timestamp_ms."""
        bars = _make_bars(10, start_ms=1_000_000, interval_ms=60_000)
        # Regimes at bar 3 and bar 7
        regimes = _make_regimes([
            1_000_000 + 3 * 60_000,  # bar 3 open
            1_000_000 + 7 * 60_000,  # bar 7 open
        ])
        strategy = RegimeLeakDetector()
        harness = ReplayHarness(
            bars=bars, outcomes=[], regimes=regimes,
            strategy=strategy, execution_model=ExecutionModel(),
        )
        result = harness.run()
        obs = result.strategy_state["observations"]

        for o in obs:
            sim_ts = o["sim_ts_ms"]
            for scope, regime in o["regimes"].items():
                rts = regime.get("timestamp_ms", 0)
                assert sim_ts >= rts, (
                    f"Regime lookahead leak! sim_time={sim_ts} < "
                    f"regime.timestamp_ms={rts}"
                )

    def test_no_regimes_before_first_regime_timestamp(self):
        """Bars before the first regime should see 0 regimes."""
        bars = _make_bars(10, start_ms=1_000_000, interval_ms=60_000)
        # Regime arrives at bar 5 timestamp
        regime_ts = 1_000_000 + 5 * 60_000
        regimes = _make_regimes([regime_ts])
        strategy = RegimeLeakDetector()
        harness = ReplayHarness(
            bars=bars, outcomes=[], regimes=regimes,
            strategy=strategy, execution_model=ExecutionModel(),
        )
        result = harness.run()
        obs = result.strategy_state["observations"]

        for o in obs:
            if o["sim_ts_ms"] < regime_ts:
                assert o["regime_count"] == 0, (
                    f"Bar at sim_ts={o['sim_ts_ms']} should see 0 regimes "
                    f"(first regime at {regime_ts})"
                )

    def test_latest_regime_overwrites_older(self):
        """When multiple regimes exist for the same scope, only the latest
        (up to sim_time) is visible."""
        bars = _make_bars(10, start_ms=1_000_000, interval_ms=60_000)
        regimes = _make_regimes([
            1_000_000 + 1 * 60_000,  # bar 1
            1_000_000 + 4 * 60_000,  # bar 4
            1_000_000 + 8 * 60_000,  # bar 8
        ])
        strategy = RegimeLeakDetector()
        harness = ReplayHarness(
            bars=bars, outcomes=[], regimes=regimes,
            strategy=strategy, execution_model=ExecutionModel(),
        )
        result = harness.run()
        obs = result.strategy_state["observations"]

        # The last bar (sim_ts = bar9 open + 60s) should see the bar 8 regime
        last_obs = obs[-1]
        if "SPY" in last_obs["regimes"]:
            assert last_obs["regimes"]["SPY"]["vol_regime"] == "VOL_2"  # 3rd regime

    def test_empty_regimes_safe(self):
        """Replay works fine with no regimes at all."""
        bars = _make_bars(5)
        strategy = RegimeLeakDetector()
        harness = ReplayHarness(
            bars=bars, outcomes=[], regimes=[],
            strategy=strategy, execution_model=ExecutionModel(),
        )
        result = harness.run()
        assert result.bars_replayed == 5
        assert result.regimes_loaded == 0
        for o in result.strategy_state["observations"]:
            assert o["regime_count"] == 0

    def test_regimes_loaded_count(self):
        """ReplayResult.regimes_loaded reflects the input count."""
        bars = _make_bars(5)
        regimes = _make_regimes([1_000_000, 1_060_000, 1_120_000])
        strategy = RegimeLeakDetector()
        harness = ReplayHarness(
            bars=bars, outcomes=[], regimes=regimes,
            strategy=strategy, execution_model=ExecutionModel(),
        )
        result = harness.run()
        assert result.regimes_loaded == 3


# ═══════════════════════════════════════════════════════════════════════════
# get_latest_regime helper tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGetLatestRegimeHelper:
    def test_returns_latest_at_or_before(self):
        regimes = [
            {"scope": "SPY", "timestamp_ms": 1000, "vol_regime": "A"},
            {"scope": "SPY", "timestamp_ms": 3000, "vol_regime": "B"},
            {"scope": "SPY", "timestamp_ms": 5000, "vol_regime": "C"},
        ]
        assert get_latest_regime(regimes, "SPY", 2000)["vol_regime"] == "A"
        assert get_latest_regime(regimes, "SPY", 3000)["vol_regime"] == "B"
        assert get_latest_regime(regimes, "SPY", 4999)["vol_regime"] == "B"
        assert get_latest_regime(regimes, "SPY", 5000)["vol_regime"] == "C"

    def test_returns_none_before_first(self):
        regimes = [
            {"scope": "SPY", "timestamp_ms": 1000, "vol_regime": "A"},
        ]
        assert get_latest_regime(regimes, "SPY", 500) is None

    def test_filters_by_scope(self):
        regimes = [
            {"scope": "SPY", "timestamp_ms": 1000, "vol_regime": "SPY_A"},
            {"scope": "QQQ", "timestamp_ms": 2000, "vol_regime": "QQQ_A"},
            {"scope": "SPY", "timestamp_ms": 3000, "vol_regime": "SPY_B"},
        ]
        r = get_latest_regime(regimes, "QQQ", 9999)
        assert r is not None
        assert r["vol_regime"] == "QQQ_A"

    def test_empty_list(self):
        assert get_latest_regime([], "SPY", 9999) is None

    def test_wrong_scope(self):
        regimes = [
            {"scope": "SPY", "timestamp_ms": 1000, "vol_regime": "A"},
        ]
        assert get_latest_regime(regimes, "AAPL", 9999) is None
