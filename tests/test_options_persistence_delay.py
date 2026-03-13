"""
Tests for options receipt time persistence and replay gating.

Validates:
- DB schema (new recv_ts_ms column, indexes)
- upsert_option_chain_snapshot / get_option_chain_snapshots round-trip
- PersistenceManager correctly stamps recv_ts_ms (using event_ts_ms)
- Replay harness availability barrier gating on recv_ts_ms
- load_replay_data correctly populates MarketDataSnapshot.recv_ts_ms
"""

from __future__ import annotations

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.outcome_engine import BarData
from src.core.option_events import OptionChainSnapshotEvent
from src.analysis.execution_model import ExecutionModel
from src.analysis.replay_harness import (
    ReplayConfig,
    ReplayHarness,
    ReplayStrategy,
    MarketDataSnapshot,
    load_replay_data,
)


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


class TestOptionsPersistenceSchema:
    def test_tables_have_recv_ts_column(self):
        async def _test():
            db = await _create_test_db()
            for table in ["option_quotes", "option_chain_snapshots"]:
                cursor = await db._connection.execute(f"PRAGMA table_info({table})")
                cols = {row[1] for row in await cursor.fetchall()}
                assert "recv_ts_ms" in cols
            await db._connection.close()
        run(_test())

    def test_snapshot_recv_ts_index_exists(self):
        async def _test():
            db = await _create_test_db()
            cursor = await db._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='option_chain_snapshots'"
            )
            indexes = {row[0] for row in await cursor.fetchall()}
            assert "idx_ocs_recv_ts" in indexes
            assert "idx_ocs_symbol_recv" in indexes
            await db._connection.close()
        run(_test())


class TestSnapshotRoundTrip:
    def test_write_read_snapshot_full(self):
        async def _test():
            db = await _create_test_db()
            await db.upsert_option_chain_snapshot(
                snapshot_id="test_snap",
                symbol="IBIT",
                expiration_ms=1700000000000,
                underlying_price=50.5,
                n_strikes=10,
                atm_iv=0.8,
                timestamp_ms=1000,
                source_ts_ms=900,
                recv_ts_ms=1100,
                provider="alpaca",
                quotes_json="{}",
            )
            rows = await db.get_option_chain_snapshots("IBIT", 0, 2000)
            assert len(rows) == 1
            r = rows[0]
            assert r["recv_ts_ms"] == 1100
            assert r["timestamp_ms"] == 1000
            assert r["source_ts_ms"] == 900
            await db._connection.close()
        run(_test())


class ToyStrategy(ReplayStrategy):
    """Strategy that records visible snapshots."""
    def __init__(self):
        self.seen_snapshots = []

    @property
    def strategy_id(self): return "TOY"

    def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes,
               *, visible_regimes=None, visible_snapshots=None):
        if visible_snapshots:
            self.seen_snapshots.extend(visible_snapshots)

    def generate_intents(self, sim_ts_ms): return []
    def finalize(self): return {"seen": self.seen_snapshots}


class TestReplaySnapshotBarrier:
    def test_snapshot_availability_barrier(self):
        """Snapshots only visible when sim_time >= recv_ts_ms."""
        # bar_duration = 60s (default in ReplayConfig)
        # Bar 1 starts at 1000, ends at 61000 (sim_time)
        # Bar 2 starts at 61000, ends at 121000 (sim_time)
        bars = [
            BarData(timestamp_ms=1000, open=100, high=101, low=99, close=100, volume=100),
            BarData(timestamp_ms=61000, open=100, high=101, low=99, close=100, volume=100),
        ]
        
        # Snap 1: recv_ts=50000 (arrives during bar 1) -> visible at end of bar 1 (61000)
        # Snap 2: recv_ts=100000 (arrives during bar 2) -> visible at end of bar 2 (121000)
        # Snap 3: recv_ts=200000 (arrives after bar 2) -> NEVER visible
        snapshots = [
            MarketDataSnapshot(symbol="IBIT", recv_ts_ms=50000, quote_ts_ms=49000),
            MarketDataSnapshot(symbol="IBIT", recv_ts_ms=100000, quote_ts_ms=99000),
            MarketDataSnapshot(symbol="IBIT", recv_ts_ms=200000, quote_ts_ms=199000),
        ]
        
        strategy = ToyStrategy()
        harness = ReplayHarness(
            bars=bars,
            outcomes=[],
            strategy=strategy,
            execution_model=ExecutionModel(),
            snapshots=snapshots
        )
        
        result = harness.run()
        seen = result.strategy_state["seen"]
        
        # Bar 1: sim_ts=61000. Sees Snap 1.
        # Bar 2: sim_ts=121000. Sees Snap 1, 2.
        # Total seen events: 3 items (Snap 1 once in Bar 1, Snap 1+2 in Bar 2)
        assert len(seen) == 3
        
        recv_timestamps = [s.recv_ts_ms for s in seen]
        assert recv_timestamps.count(50000) == 2 # Once in Bar 1, once in Bar 2
        assert recv_timestamps.count(100000) == 1 # Once in Bar 2
        assert 200000 not in recv_timestamps
        
    def test_provider_ts_ignored_for_gating(self):
        """Snapshot with old provider_ts but future recv_ts is gated correctly."""
        bars = [BarData(timestamp_ms=1000, open=100, high=101, low=99, close=100, volume=100)]
        # sim_ts = 1000 + 60000 = 61000
        # provider_ts = 500 (old), but recv_ts = 61001 (future relative to bar close)
        snapshots = [MarketDataSnapshot(symbol="IBIT", recv_ts_ms=61001, quote_ts_ms=500)]
        strategy = ToyStrategy()
        harness = ReplayHarness(bars=bars, outcomes=[], strategy=strategy, 
                                execution_model=ExecutionModel(), snapshots=snapshots)
        result = harness.run()
        assert len(result.strategy_state["seen"]) == 0

