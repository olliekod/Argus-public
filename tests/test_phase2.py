"""
Phase 2 comprehensive tests — Streams 1-4.

Tests cover:
* Stream 1: Bar provenance fields, invariant enforcement, close reasons
* Stream 2: Heartbeat emission, lag tracking, equity market-close continuity
* Stream 3: Polymarket watchlist filtering
* Stream 4: FeatureBuilder metrics, RegimeDetector transitions

Run with:  python -m pytest tests/test_phase2.py -v
"""

import math
import time
from collections import deque
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock

from src.core.bar_builder import BarBuilder, _BarAccumulator, _minute_floor
from src.core.bus import EventBus
from src.core.events import (
    BarEvent,
    CloseReason,
    ComponentHeartbeatEvent,
    MetricEvent,
    MinuteTickEvent,
    QuoteEvent,
    RegimeChangeEvent,
    SCHEMA_VERSION,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_METRICS,
    TOPIC_REGIMES_SYMBOL,
    TOPIC_SYSTEM_COMPONENT_HEARTBEAT,
)
from src.core.feature_builder import FeatureBuilder
from src.core.regime_detector import RegimeDetector
from src.core.regimes import (
    VolRegime,
    TrendRegime,
    VOL_REGIME_NAMES,
    TREND_REGIME_NAMES,
)


def _quote(symbol, price, volume_24h, ts, source="test", source_ts=0.0):
    return QuoteEvent(
        symbol=symbol, bid=price - 0.01, ask=price + 0.01, mid=price,
        last=price, timestamp=ts, source=source, volume_24h=volume_24h,
        source_ts=source_ts if source_ts else ts,
        event_ts=ts, receive_time=ts,
    )


def _drain(bus, timeout=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        depths = bus.get_queue_depths()
        if all(d == 0 for d in depths.values()):
            return
        time.sleep(0.01)


BASE = 1_700_000_000.0
M0 = _minute_floor(BASE)
M1 = M0 + 60
M2 = M0 + 120


# ═══════════════════════════════════════════════════════════
#  STREAM 1 — Bar Provenance
# ═══════════════════════════════════════════════════════════


class TestBarProvenance:
    """Provenance fields are correctly populated in emitted bars."""

    def test_n_ticks_equals_tick_count(self):
        bus = EventBus()
        emitted: List[BarEvent] = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda b: emitted.append(b))
        bb = BarBuilder(bus)
        bus.start()
        try:
            bb._on_quote(_quote("BTC", 100, 1000, M0 + 1, source_ts=M0 + 0.5))
            bb._on_quote(_quote("BTC", 101, 1010, M0 + 15, source_ts=M0 + 14))
            bb._on_quote(_quote("BTC", 102, 1020, M1 + 1, source_ts=M1 + 0.3))
            _drain(bus)
            assert len(emitted) == 1
            bar = emitted[0]
            assert bar.n_ticks == 2
            assert bar.n_ticks == bar.tick_count
        finally:
            bus.stop()

    def test_first_last_source_ts(self):
        bus = EventBus()
        emitted: List[BarEvent] = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda b: emitted.append(b))
        bb = BarBuilder(bus)
        bus.start()
        try:
            bb._on_quote(_quote("BTC", 100, 1000, M0 + 1, source_ts=M0 + 0.5))
            bb._on_quote(_quote("BTC", 101, 1010, M0 + 30, source_ts=M0 + 29.9))
            bb._on_quote(_quote("BTC", 102, 1020, M1 + 1, source_ts=M1 + 0.1))
            _drain(bus)
            bar = emitted[0]
            assert bar.first_source_ts == M0 + 0.5
            assert bar.last_source_ts == M0 + 29.9
        finally:
            bus.stop()

    def test_close_reason_new_tick(self):
        bus = EventBus()
        emitted: List[BarEvent] = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda b: emitted.append(b))
        bb = BarBuilder(bus)
        bus.start()
        try:
            bb._on_quote(_quote("BTC", 100, 1000, M0 + 1))
            bb._on_quote(_quote("BTC", 101, 1010, M1 + 1))
            _drain(bus)
            assert emitted[0].close_reason == int(CloseReason.NEW_TICK)
        finally:
            bus.stop()

    def test_close_reason_minute_tick(self):
        bus = EventBus()
        emitted: List[BarEvent] = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda b: emitted.append(b))
        bb = BarBuilder(bus)
        bus.start()
        try:
            bb._on_quote(_quote("BTC", 100, 1000, M0 + 1))
            bb._on_minute_tick(MinuteTickEvent(timestamp=M1))
            _drain(bus)
            assert emitted[0].close_reason == int(CloseReason.MINUTE_TICK)
        finally:
            bus.stop()

    def test_close_reason_shutdown_flush(self):
        bus = EventBus()
        bb = BarBuilder(bus)
        bb._on_quote(_quote("BTC", 100, 1000, M0 + 1))
        bars = bb.flush()
        assert len(bars) == 1
        assert bars[0].close_reason == int(CloseReason.SHUTDOWN_FLUSH)

    def test_schema_version(self):
        bus = EventBus()
        emitted: List[BarEvent] = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda b: emitted.append(b))
        bb = BarBuilder(bus)
        bus.start()
        try:
            bb._on_quote(_quote("BTC", 100, 1000, M0 + 1))
            bb._on_quote(_quote("BTC", 101, 1010, M1 + 1))
            _drain(bus)
            assert emitted[0].v == SCHEMA_VERSION
        finally:
            bus.stop()

    def test_source_ts_matches_first_source_ts(self):
        bus = EventBus()
        emitted: List[BarEvent] = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda b: emitted.append(b))
        bb = BarBuilder(bus)
        bus.start()
        try:
            bb._on_quote(_quote("BTC", 100, 1000, M0 + 1, source_ts=M0 + 0.8))
            bb._on_quote(_quote("BTC", 101, 1010, M1 + 1))
            _drain(bus)
            assert emitted[0].source_ts == emitted[0].first_source_ts
        finally:
            bus.stop()


class TestBarInvariants:
    """Bar invariant enforcement repairs bad data without crashing."""

    def test_high_enforced(self):
        acc = _BarAccumulator(100.0, 0.0, M0, "test")
        acc.high = 99.0  # Force invalid state
        bb = BarBuilder(EventBus())
        valid = bb._enforce_invariants(acc)
        assert not valid
        assert acc.high >= acc.open

    def test_low_enforced(self):
        acc = _BarAccumulator(100.0, 0.0, M0, "test")
        acc.close = 95.0
        acc.low = 101.0  # Force invalid: low > min(open, close)
        bb = BarBuilder(EventBus())
        valid = bb._enforce_invariants(acc)
        assert not valid
        assert acc.low <= min(acc.open, acc.close)

    def test_volume_enforced(self):
        acc = _BarAccumulator(100.0, 0.0, M0, "test")
        acc.volume = -5.0
        bb = BarBuilder(EventBus())
        valid = bb._enforce_invariants(acc)
        assert not valid
        assert acc.volume == 0.0

    def test_valid_bar_passes(self):
        acc = _BarAccumulator(100.0, 10.0, M0, "test")
        acc.update(105.0, 5.0)
        acc.update(98.0, 3.0)
        bb = BarBuilder(EventBus())
        valid = bb._enforce_invariants(acc)
        assert valid


class TestLateTicks:
    """Late ticks are counted per-symbol and never mutate emitted bars."""

    def test_late_tick_count_in_provenance(self):
        bus = EventBus()
        emitted: List[BarEvent] = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda b: emitted.append(b))
        bb = BarBuilder(bus)
        bus.start()
        try:
            bb._on_quote(_quote("BTC", 100, 1000, M0 + 1))
            bb._on_quote(_quote("BTC", 101, 1010, M1 + 1))
            _drain(bus)
            # Minute-0 bar closed. Send late tick for minute 0.
            bb._on_quote(_quote("BTC", 50, 1020, M0 + 30))
            # Now close minute 1 by sending minute 2 tick
            bb._on_quote(_quote("BTC", 102, 1030, M2 + 1))
            _drain(bus)
            # Second bar should record that 1 late tick was dropped
            bar1 = emitted[1]
            assert bar1.late_ticks_dropped >= 1
        finally:
            bus.stop()


# ═══════════════════════════════════════════════════════════
#  STREAM 2 — Heartbeats & Lag
# ═══════════════════════════════════════════════════════════


class TestComponentHeartbeats:
    """Component heartbeat emission."""

    def test_bar_builder_heartbeat(self):
        bus = EventBus()
        heartbeats = []
        bus.subscribe(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, lambda h: heartbeats.append(h))
        bb = BarBuilder(bus)
        bus.start()
        try:
            bb._on_quote(_quote("BTC", 100, 1000, M0 + 1))
            hb = bb.emit_heartbeat()
            _drain(bus)
            assert hb.component == "bar_builder"
            assert hb.events_processed >= 1
            assert hb.health in ("ok", "degraded", "down")
            assert hb.v == SCHEMA_VERSION
        finally:
            bus.stop()


class TestEquityContinuity:
    """QueryLayer recognizes market-close gaps for equity symbols."""

    def test_is_equity_market_open_weekend(self):
        from src.core.query_layer import _is_equity_market_open
        # 2026-02-01 is a Sunday
        sunday_utc = datetime(2026, 2, 1, 15, 0, 0, tzinfo=timezone.utc)
        assert not _is_equity_market_open(sunday_utc)

    def test_is_equity_market_open_weekday(self):
        from src.core.query_layer import _is_equity_market_open
        # 2026-02-02 is a Monday, 15:00 UTC = 10:00 ET (market open)
        monday_utc = datetime(2026, 2, 2, 15, 0, 0, tzinfo=timezone.utc)
        assert _is_equity_market_open(monday_utc)


# ═══════════════════════════════════════════════════════════
#  STREAM 4 — FeatureBuilder
# ═══════════════════════════════════════════════════════════


class TestFeatureBuilder:
    """FeatureBuilder computes rolling metrics from bars."""

    def _make_bar(self, symbol, close, ts, source="test"):
        return BarEvent(
            symbol=symbol, open=close, high=close, low=close, close=close,
            volume=100, timestamp=ts, source=source, event_ts=ts, receive_time=ts,
        )

    def test_first_bar_produces_no_metrics(self):
        bus = EventBus()
        metrics = []
        bus.subscribe(TOPIC_MARKET_METRICS, lambda m: metrics.append(m))
        fb = FeatureBuilder(bus)
        bus.start()
        try:
            fb._on_bar(self._make_bar("BTC", 100.0, M0))
            _drain(bus)
            assert len(metrics) == 0, "First bar should produce no metrics (no prior close)"
        finally:
            bus.stop()

    def test_second_bar_produces_log_return(self):
        bus = EventBus()
        metrics = []
        bus.subscribe(TOPIC_MARKET_METRICS, lambda m: metrics.append(m))
        fb = FeatureBuilder(bus)
        bus.start()
        try:
            fb._on_bar(self._make_bar("BTC", 100.0, M0))
            fb._on_bar(self._make_bar("BTC", 105.0, M1))
            _drain(bus)
            ret_metrics = [m for m in metrics if m.metric == "log_return"]
            assert len(ret_metrics) == 1
            expected = math.log(105.0 / 100.0)
            assert abs(ret_metrics[0].value - expected) < 1e-9
        finally:
            bus.stop()

    def test_realized_vol_after_enough_bars(self):
        bus = EventBus()
        metrics = []
        bus.subscribe(TOPIC_MARKET_METRICS, lambda m: metrics.append(m))
        fb = FeatureBuilder(bus)
        bus.start()
        try:
            # Feed 31 bars (30 returns needed for _VOL_WINDOW)
            for i in range(31):
                price = 100.0 + i * 0.1
                fb._on_bar(self._make_bar("BTC", price, M0 + i * 60))
            _drain(bus)
            vol_metrics = [m for m in metrics if m.metric == "realized_vol"]
            assert len(vol_metrics) >= 1, "Should produce realized_vol after enough bars"
            assert vol_metrics[-1].value > 0
        finally:
            bus.stop()

    def test_jump_score_emitted(self):
        bus = EventBus()
        metrics = []
        bus.subscribe(TOPIC_MARKET_METRICS, lambda m: metrics.append(m))
        fb = FeatureBuilder(bus)
        bus.start()
        try:
            # Normal bars
            for i in range(31):
                fb._on_bar(self._make_bar("BTC", 100.0 + i * 0.01, M0 + i * 60))
            # Spike
            fb._on_bar(self._make_bar("BTC", 110.0, M0 + 31 * 60))
            _drain(bus)
            jumps = [m for m in metrics if m.metric == "jump_score"]
            assert len(jumps) >= 1
            assert jumps[-1].value > 1.0  # Should be a significant jump
        finally:
            bus.stop()


# ═══════════════════════════════════════════════════════════
#  STREAM 4 — RegimeDetector
# ═══════════════════════════════════════════════════════════


class TestRegimeDetector:
    """RegimeDetector classifies market regimes from bars."""

    def _make_bar(self, symbol: str, close: float, ts: float) -> BarEvent:
        return BarEvent(
            symbol=symbol,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100,
            timestamp=ts,
            source="test",
            bar_duration=60,
        )

    def test_initial_regime_is_unknown(self):
        bus = EventBus()
        rd = RegimeDetector(bus)
        assert rd.get_symbol_regime("BTC") is None

    def test_regime_transition_on_data(self):
        bus = EventBus()
        regime_events = []
        bus.subscribe(TOPIC_REGIMES_SYMBOL, lambda e: regime_events.append(e))
        rd = RegimeDetector(bus)
        bus.start()
        try:
            for i in range(40):
                close = 100.0 + i * 0.1
                rd._on_bar(self._make_bar("BTC", close, M0 + i * 60))
            _drain(bus)

            regime = rd.get_symbol_regime("BTC")
            assert regime in VOL_REGIME_NAMES.values()
            assert len(regime_events) >= 1
            assert regime_events[-1].vol_regime in VOL_REGIME_NAMES.values()
            assert regime_events[-1].trend_regime in TREND_REGIME_NAMES.values()
        finally:
            bus.stop()

    def test_crash_detection(self):
        bus = EventBus()
        regime_events = []
        bus.subscribe(TOPIC_REGIMES_SYMBOL, lambda e: regime_events.append(e))
        rd = RegimeDetector(
            bus,
            thresholds={
                "vol_spike_z": -1.0,
                "vol_high_z": 0.0,
                "vol_low_z": -2.0,
                "trend_slope_threshold": 0.5,
                "trend_strength_threshold": 1.0,
                "atr_epsilon": 1e-8,
                "gap_tolerance_bars": 1,
                "gap_flag_duration_bars": 2,
                "warmup_bars": 10,
            },
        )
        bus.start()
        try:
            for i in range(15):
                close = 100.0 + i * 0.5
                rd._on_bar(self._make_bar("BTC", close, M0 + i * 60))
            _drain(bus)

            assert regime_events
            assert regime_events[-1].vol_regime == VOL_REGIME_NAMES[VolRegime.VOL_SPIKE]
        finally:
            bus.stop()

    def test_no_regime_without_bars(self):
        bus = EventBus()
        rd = RegimeDetector(bus)
        assert rd.get_symbol_regime("BTC") is None


# ═══════════════════════════════════════════════════════════
#  STREAM 3 — Polymarket Watchlist
# ═══════════════════════════════════════════════════════════


class TestPolymarketWatchlist:
    """Watchlist service filters and syncs markets to CLOB."""

    def test_volume_filter(self):
        # Import directly using importlib.util to avoid triggering __init__.py
        import importlib.util, sys
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "polymarket_watchlist",
            str(Path(__file__).resolve().parents[1] / "src/connectors/polymarket_watchlist.py"),
        )
        watchlist_mod = importlib.util.module_from_spec(spec)
        sys.modules["polymarket_watchlist"] = watchlist_mod
        spec.loader.exec_module(watchlist_mod)
        WatchlistService = watchlist_mod.PolymarketWatchlistService

        gamma_mock = MagicMock()
        gamma_mock.get_cached_markets.return_value = {
            "c1": {"active": True, "volume": 50000, "question": "Will X?",
                    "tokens": [{"token_id": "tok1"}]},
            "c2": {"active": True, "volume": 500, "question": "Will Y?",
                    "tokens": [{"token_id": "tok2"}]},
            "c3": {"active": False, "volume": 100000, "question": "Will Z?",
                    "tokens": [{"token_id": "tok3"}]},
        }
        clob_mock = MagicMock()

        import asyncio
        ws = WatchlistService(
            gamma_client=gamma_mock,
            clob_client=clob_mock,
            min_volume=10000,
        )

        tokens = asyncio.run(ws.sync())
        # Only c1 passes (active=True, volume=50000 > 10000)
        assert "tok1" in tokens
        assert "tok2" not in tokens  # too low volume
        assert "tok3" not in tokens  # inactive
        clob_mock.set_watchlist.assert_called_once()

    def test_keyword_filter(self):
        import importlib.util, sys
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "polymarket_watchlist",
            str(Path(__file__).resolve().parents[1] / "src/connectors/polymarket_watchlist.py"),
        )
        watchlist_mod = importlib.util.module_from_spec(spec)
        sys.modules["polymarket_watchlist"] = watchlist_mod
        spec.loader.exec_module(watchlist_mod)
        WatchlistService = watchlist_mod.PolymarketWatchlistService

        gamma_mock = MagicMock()
        gamma_mock.get_cached_markets.return_value = {
            "c1": {"active": True, "volume": 100000, "question": "Will Bitcoin hit 100k?",
                    "tokens": [{"token_id": "tok1"}]},
            "c2": {"active": True, "volume": 100000, "question": "Will the election be close?",
                    "tokens": [{"token_id": "tok2"}]},
        }
        clob_mock = MagicMock()

        import asyncio
        ws = WatchlistService(
            gamma_client=gamma_mock, clob_client=clob_mock,
            min_volume=1000, keywords=["bitcoin"],
        )
        tokens = asyncio.run(ws.sync())
        assert "tok1" in tokens
        assert "tok2" not in tokens


# ═══════════════════════════════════════════════════════════
#  Event schema versioning
# ═══════════════════════════════════════════════════════════


class TestSchemaVersioning:
    """All event types carry the schema version field."""

    def test_quote_event_has_version(self):
        q = QuoteEvent(symbol="X", bid=1, ask=2, mid=1.5, last=1.5,
                       timestamp=M0, source="test")
        assert q.v == SCHEMA_VERSION

    def test_metric_event_has_version(self):
        m = MetricEvent(symbol="X", metric="m", value=1.0,
                        timestamp=M0, source="test")
        assert m.v == SCHEMA_VERSION

    def test_bar_event_has_version(self):
        b = BarEvent(symbol="X", open=1, high=2, low=0.5, close=1.5,
                     volume=100, timestamp=M0, source="test")
        assert b.v == SCHEMA_VERSION

    def test_component_heartbeat_has_version(self):
        h = ComponentHeartbeatEvent(component="test")
        assert h.v == SCHEMA_VERSION

    def test_regime_change_has_version(self):
        r = RegimeChangeEvent(symbol="BTC", old_regime="A", new_regime="B")
        assert r.v == SCHEMA_VERSION
