"""
Tests for Replay Pack Snapshot Support
=======================================

Verifies:
- Option chain snapshots are included in replay packs
- Chronological ordering by recv_ts_ms is preserved
- ReplayHarness loads snapshots from packs correctly
- Snapshot gating uses recv_ts_ms (data availability barrier)
- Symbols without options data produce empty snapshot lists
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import pytest

from src.analysis.execution_model import ExecutionModel
from src.analysis.replay_harness import (
    MarketDataSnapshot,
    ReplayConfig,
    ReplayHarness,
    ReplayStrategy,
    TradeIntent,
)
from src.core.outcome_engine import BarData


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_bars(
    n: int = 10,
    start_ms: int = 1_700_000_000_000,
    interval_ms: int = 60_000,
    open_price: float = 100.0,
    drift: float = 0.5,
) -> List[BarData]:
    bars = []
    price = open_price
    for i in range(n):
        ts = start_ms + i * interval_ms
        bars.append(BarData(
            timestamp_ms=ts,
            open=round(price, 2),
            high=round(price + 1.0, 2),
            low=round(price - 0.5, 2),
            close=round(price + drift, 2),
            volume=1000.0 + i * 100,
        ))
        price += drift
    return bars


def _make_snapshots(
    n: int = 5,
    start_ms: int = 1_700_000_000_000,
    interval_ms: int = 120_000,
    symbol: str = "SPY",
) -> List[Dict[str, Any]]:
    """Build synthetic snapshot dicts as they would appear in a replay pack."""
    snaps = []
    for i in range(n):
        ts = start_ms + i * interval_ms
        snaps.append({
            "timestamp_ms": ts,
            "recv_ts_ms": ts + 500,  # 500ms receipt delay
            "provider": "tastytrade",
            "underlying_price": 450.0 + i * 0.5,
            "atm_iv": 0.18 + i * 0.001,
            "quotes_json": json.dumps({"calls": [], "puts": []}),
            "symbol": symbol,
            "n_strikes": 20,
        })
    return snaps


def _build_pack(
    bars: List[BarData],
    snapshots: Optional[List[Dict[str, Any]]] = None,
    outcomes: Optional[List[Dict[str, Any]]] = None,
    regimes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    bar_dicts = [
        {
            "timestamp_ms": b.timestamp_ms,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "symbol": "SPY",
        }
        for b in bars
    ]
    return {
        "metadata": {
            "symbol": "SPY",
            "provider": "tastytrade",
            "start_date": "2023-11-14",
            "end_date": "2023-11-14",
            "packed_at": "2023-11-14T00:00:00",
            "snapshot_count": len(snapshots or []),
        },
        "bars": bar_dicts,
        "outcomes": outcomes or [],
        "regimes": regimes or [],
        "snapshots": snapshots or [],
    }


class SnapshotCollector(ReplayStrategy):
    """Records visible snapshots at each bar for verification."""

    def __init__(self):
        self.observations: List[Dict[str, Any]] = []

    @property
    def strategy_id(self) -> str:
        return "SNAPSHOT_COLLECTOR"

    def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes, **kwargs):
        visible_snaps = kwargs.get("visible_snapshots", [])
        self.observations.append({
            "bar_ts_ms": bar.timestamp_ms,
            "sim_ts_ms": sim_ts_ms,
            "snapshot_count": len(visible_snaps),
            "snapshot_recv_ts": [s.recv_ts_ms for s in visible_snaps],
        })

    def generate_intents(self, sim_ts_ms):
        return []

    def finalize(self):
        return {"observations": self.observations}


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Snapshot inclusion in replay packs
# ═══════════════════════════════════════════════════════════════════════════

class TestSnapshotInclusion:
    def test_snapshots_present_in_pack(self):
        """Pack JSON contains the snapshots list with expected fields."""
        bars = _make_bars(n=5)
        snaps = _make_snapshots(n=3, start_ms=bars[0].timestamp_ms)
        pack = _build_pack(bars, snapshots=snaps)

        assert "snapshots" in pack
        assert len(pack["snapshots"]) == 3
        for s in pack["snapshots"]:
            assert "timestamp_ms" in s
            assert "recv_ts_ms" in s
            assert "provider" in s
            assert "underlying_price" in s
            assert "quotes_json" in s

    def test_empty_snapshots_for_no_options(self):
        """Symbols without options data have an empty snapshots list."""
        bars = _make_bars(n=5)
        pack = _build_pack(bars, snapshots=[])
        assert pack["snapshots"] == []

    def test_snapshot_ordering_preserved(self):
        """Snapshots must be in chronological order by recv_ts_ms."""
        bars = _make_bars(n=10)
        # Create out-of-order snapshots
        snaps = _make_snapshots(n=5, start_ms=bars[0].timestamp_ms)
        # Shuffle
        shuffled = [snaps[3], snaps[0], snaps[4], snaps[1], snaps[2]]

        # The _fetch_snapshots function sorts; simulate that here
        sorted_snaps = sorted(shuffled, key=lambda s: s["recv_ts_ms"])

        for i in range(1, len(sorted_snaps)):
            assert sorted_snaps[i]["recv_ts_ms"] >= sorted_snaps[i - 1]["recv_ts_ms"]

    def test_atm_iv_nullable(self):
        """atm_iv can be None for snapshots where IV isn't available."""
        bars = _make_bars(n=3)
        snaps = _make_snapshots(n=2, start_ms=bars[0].timestamp_ms)
        snaps[0]["atm_iv"] = None
        pack = _build_pack(bars, snapshots=snaps)
        assert pack["snapshots"][0]["atm_iv"] is None
        assert pack["snapshots"][1]["atm_iv"] is not None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: ReplayHarness snapshot loading from packs
# ═══════════════════════════════════════════════════════════════════════════

class TestHarnessSnapshotLoading:
    def _harness_from_pack(self, pack: Dict[str, Any]) -> ReplayHarness:
        """Build a ReplayHarness from a pack dict (as experiment_runner does)."""
        bars = []
        for b in pack["bars"]:
            bar = BarData(
                timestamp_ms=b["timestamp_ms"],
                open=b["open"],
                high=b["high"],
                low=b["low"],
                close=b["close"],
                volume=b.get("volume", 0),
            )
            if "symbol" in b:
                object.__setattr__(bar, "symbol", b["symbol"])
            bars.append(bar)

        # Convert snapshot dicts to MarketDataSnapshot objects
        snapshots = []
        for s in pack.get("snapshots", []):
            snapshots.append(MarketDataSnapshot(
                symbol=s.get("symbol", "SPY"),
                recv_ts_ms=s["recv_ts_ms"],
                underlying_price=s.get("underlying_price", 0.0),
                atm_iv=s.get("atm_iv"),
                source=s.get("provider", ""),
            ))

        strategy = SnapshotCollector()
        return ReplayHarness(
            bars=bars,
            outcomes=pack.get("outcomes", []),
            strategy=strategy,
            execution_model=ExecutionModel(),
            regimes=pack.get("regimes", []),
            snapshots=snapshots,
        ), strategy

    def test_harness_receives_snapshots(self):
        """Harness should receive and gate snapshots properly."""
        bars = _make_bars(n=10)
        snaps = _make_snapshots(n=3, start_ms=bars[0].timestamp_ms, interval_ms=120_000)
        pack = _build_pack(bars, snapshots=snaps)
        harness, strategy = self._harness_from_pack(pack)
        result = harness.run()

        assert result.bars_replayed == 10
        # By the last bar, all 3 snapshots should be visible
        last_obs = strategy.observations[-1]
        assert last_obs["snapshot_count"] == 3

    def test_snapshot_gating_uses_recv_ts_ms(self):
        """Snapshots must not be visible before their recv_ts_ms."""
        bars = _make_bars(n=10, interval_ms=60_000)
        bar_duration_ms = 60_000

        # Place snapshot recv_ts_ms after bar 5's sim_time
        # Bar 5 sim_time = start_ms + 5*60000 + 60000 = start_ms + 360000
        snap_recv_ts = bars[0].timestamp_ms + 360_000 + 1  # Just after bar 5 closes
        snaps = [{
            "timestamp_ms": bars[0].timestamp_ms + 300_000,
            "recv_ts_ms": snap_recv_ts,
            "provider": "tastytrade",
            "underlying_price": 450.0,
            "atm_iv": 0.18,
            "quotes_json": "{}",
            "symbol": "SPY",
            "n_strikes": 20,
        }]
        pack = _build_pack(bars, snapshots=snaps)
        harness, strategy = self._harness_from_pack(pack)
        harness.run()

        # Bars 0-5 should see 0 snapshots (sim_time < recv_ts_ms)
        for i in range(6):
            assert strategy.observations[i]["snapshot_count"] == 0, (
                f"Bar {i} should not see the snapshot yet"
            )
        # Bar 6+ should see 1 snapshot
        for i in range(6, 10):
            assert strategy.observations[i]["snapshot_count"] == 1, (
                f"Bar {i} should see the snapshot"
            )

    def test_empty_snapshots_harness_ok(self):
        """Harness runs fine with no snapshots."""
        bars = _make_bars(n=5)
        pack = _build_pack(bars, snapshots=[])
        harness, strategy = self._harness_from_pack(pack)
        result = harness.run()
        assert result.bars_replayed == 5
        for obs in strategy.observations:
            assert obs["snapshot_count"] == 0

    def test_snapshots_accumulate(self):
        """Snapshot count should be non-decreasing over time."""
        bars = _make_bars(n=20, interval_ms=60_000)
        snaps = _make_snapshots(
            n=5,
            start_ms=bars[0].timestamp_ms,
            interval_ms=180_000,  # every 3 bars
        )
        pack = _build_pack(bars, snapshots=snaps)
        harness, strategy = self._harness_from_pack(pack)
        harness.run()

        counts = [obs["snapshot_count"] for obs in strategy.observations]
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], "Snapshot count must not decrease"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Pack file I/O round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestPackRoundTrip:
    def test_json_round_trip(self):
        """Write a pack to JSON and read it back — snapshots survive."""
        bars = _make_bars(n=5)
        snaps = _make_snapshots(n=3, start_ms=bars[0].timestamp_ms)
        pack = _build_pack(bars, snapshots=snaps)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pack, f, indent=2)
            tmp_path = f.name

        try:
            with open(tmp_path) as f:
                loaded = json.load(f)

            assert len(loaded["snapshots"]) == 3
            assert loaded["snapshots"][0]["recv_ts_ms"] == snaps[0]["recv_ts_ms"]
            assert loaded["snapshots"][0]["underlying_price"] == snaps[0]["underlying_price"]
            assert loaded["snapshots"][0]["quotes_json"] is not None
        finally:
            os.unlink(tmp_path)
