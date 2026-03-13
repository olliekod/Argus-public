"""
Signal Schema and Router Tests
==============================

Tests for Phase 3 signal infrastructure.
"""

import json
import pytest
from unittest.mock import MagicMock

from src.core.signals import (
    SignalEvent,
    RankedSignalEvent,
    SignalOutcomeEvent,
    signal_to_dict,
    dict_to_signal,
    ranked_signal_to_dict,
    dict_to_ranked_signal,
    outcome_to_dict,
    dict_to_outcome,
    compute_signal_id,
    compute_config_hash,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    SIGNAL_TYPE_ENTRY,
)


# ═══════════════════════════════════════════════════════════════════════════
# SignalEvent Schema Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalEventSchema:
    """Test SignalEvent serialization and determinism."""

    def test_signal_round_trip(self):
        """SignalEvent → dict → SignalEvent preserves all fields."""
        signal = SignalEvent(
            timestamp_ms=1700000000000,
            strategy_id="FVG_BREAKOUT_V1",
            config_hash="abc123",
            symbol="BTC",
            direction=DIRECTION_LONG,
            signal_type=SIGNAL_TYPE_ENTRY,
            timeframe=60,
            entry_price=40000.0,
            stop_price=39500.0,
            tp_price=41000.0,
            confidence=0.85,
            quality_score=75,
            data_quality_flags=0,
            regime_snapshot={"vol": "VOL_NORMAL", "trend": "TREND_UP"},
            features_snapshot={"atr": 100.5, "rsi": 65.0},
            explain="FVG detected at 39800, breakout confirmed",
            idempotency_key="test123",
        )
        
        d = signal_to_dict(signal)
        restored = dict_to_signal(d)
        
        assert restored.timestamp_ms == signal.timestamp_ms
        assert restored.strategy_id == signal.strategy_id
        assert restored.symbol == signal.symbol
        assert restored.direction == signal.direction
        assert abs(restored.entry_price - signal.entry_price) < 1e-7
        assert restored.regime_snapshot == signal.regime_snapshot
        assert restored.idempotency_key == signal.idempotency_key

    def test_signal_json_determinism(self):
        """Same signal produces identical JSON."""
        signal = SignalEvent(
            timestamp_ms=1700000000000,
            strategy_id="TEST_V1",
            config_hash="abc",
            symbol="ETH",
            direction=DIRECTION_SHORT,
            signal_type=SIGNAL_TYPE_ENTRY,
            timeframe=60,
            confidence=0.9,
            idempotency_key="key1",
        )
        
        d1 = signal_to_dict(signal)
        d2 = signal_to_dict(signal)
        
        json1 = json.dumps(d1, sort_keys=True)
        json2 = json.dumps(d2, sort_keys=True)
        
        assert json1 == json2

    def test_signal_snapshot_normalization(self):
        """Snapshot values are JSON-safe and deterministically ordered."""
        signal = SignalEvent(
            timestamp_ms=1700000000000,
            strategy_id="TEST_V1",
            config_hash="abc",
            symbol="BTC",
            direction=DIRECTION_LONG,
            signal_type=SIGNAL_TYPE_ENTRY,
            timeframe=60,
            confidence=0.9,
            regime_snapshot={"trend": "TREND_UP", "vol": "VOL_NORMAL"},
            features_snapshot={
                "gate_name": "SELL_PUT_SPREAD",
                "gate_score": 0.123456789,
                "gate_allow": 1.0,
            },
            idempotency_key="key1",
        )

        d = signal_to_dict(signal)

        assert list(d.keys()) == sorted(d.keys())
        assert d["features_snapshot"]["gate_name"] == "SELL_PUT_SPREAD"
        assert d["features_snapshot"]["gate_score"] == 0.12345679

    def test_idempotency_key_determinism(self):
        """Same inputs produce same idempotency key."""
        key1 = compute_signal_id("STRAT", "config", "BTC", 1700000000000)
        key2 = compute_signal_id("STRAT", "config", "BTC", 1700000000000)
        assert key1 == key2
        
        # Different inputs produce different key
        key3 = compute_signal_id("STRAT", "config", "ETH", 1700000000000)
        assert key1 != key3


# ═══════════════════════════════════════════════════════════════════════════
# RankedSignalEvent Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRankedSignalEvent:
    """Test RankedSignalEvent serialization."""

    def test_ranked_signal_round_trip(self):
        """RankedSignalEvent → dict → RankedSignalEvent preserves fields."""
        signal = SignalEvent(
            timestamp_ms=1700000000000,
            strategy_id="TEST",
            config_hash="abc",
            symbol="BTC",
            direction=DIRECTION_LONG,
            signal_type=SIGNAL_TYPE_ENTRY,
            timeframe=60,
            idempotency_key="key1",
        )
        
        ranked = RankedSignalEvent(
            signal=signal,
            rank=1,
            final_score=85.5,
            score_breakdown={"base": 50, "confidence": 30, "bonus_trend": 5.5},
            suppressed=False,
            suppression_reason="",
        )
        
        d = ranked_signal_to_dict(ranked)
        restored = dict_to_ranked_signal(d)
        
        assert restored.rank == ranked.rank
        assert abs(restored.final_score - ranked.final_score) < 1e-7
        assert restored.suppressed == ranked.suppressed
        assert restored.signal.symbol == signal.symbol


# ═══════════════════════════════════════════════════════════════════════════
# SignalRouter Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalRouter:
    """Test SignalRouter scoring and ranking."""

    def _create_mock_bus(self):
        """Create mock bus that captures published events."""
        bus = MagicMock()
        bus.published = []
        bus.publish.side_effect = lambda t, e: bus.published.append((t, e))
        return bus

    def test_router_scores_signals(self):
        """Router scores and ranks incoming signals."""
        from src.strategies.router import SignalRouter
        
        bus = self._create_mock_bus()
        router = SignalRouter(bus)
        
        # Emit a signal
        signal = SignalEvent(
            timestamp_ms=1700000000000,
            strategy_id="TEST",
            config_hash="abc",
            symbol="BTC",
            direction=DIRECTION_LONG,
            signal_type=SIGNAL_TYPE_ENTRY,
            timeframe=60,
            confidence=0.9,
            regime_snapshot={"trend": "TREND_UP", "vol": "VOL_NORMAL"},
            idempotency_key="key1",
        )
        
        router._on_raw_signal(signal)
        router.flush()
        
        # Check ranked output
        ranked_events = [e for t, e in bus.published if t == "signals.ranked"]
        assert len(ranked_events) == 1
        
        ranked = ranked_events[0]
        assert ranked.rank == 1
        assert ranked.final_score > 50  # Should have bonuses
        assert not ranked.suppressed

    def test_router_suppresses_low_quality(self):
        """Router suppresses signals below threshold."""
        from src.strategies.router import SignalRouter
        
        bus = self._create_mock_bus()
        config = {"min_score_threshold": 80, "base_score": 50, "max_signals_per_bucket": 5}
        router = SignalRouter(bus, config=config)
        
        # Low confidence signal
        signal = SignalEvent(
            timestamp_ms=1700000000000,
            strategy_id="TEST",
            config_hash="abc",
            symbol="BTC",
            direction=DIRECTION_LONG,
            signal_type=SIGNAL_TYPE_ENTRY,
            timeframe=60,
            confidence=0.1,  # Low confidence
            data_quality_flags=3,  # Repaired + gap
            idempotency_key="key1",
        )
        
        router._on_raw_signal(signal)
        router.flush()
        
        ranked_events = [e for t, e in bus.published if t == "signals.ranked"]
        assert len(ranked_events) == 1
        assert ranked_events[0].suppressed  # Should be suppressed

    def test_router_determinism(self):
        """Same signals produce identical rankings."""
        from src.strategies.router import SignalRouter
        
        signal = SignalEvent(
            timestamp_ms=1700000000000,
            strategy_id="TEST",
            config_hash="abc",
            symbol="BTC",
            direction=DIRECTION_LONG,
            signal_type=SIGNAL_TYPE_ENTRY,
            timeframe=60,
            confidence=0.8,
            regime_snapshot={"trend": "TREND_UP"},
            idempotency_key="key1",
        )
        
        # Run 1
        bus1 = self._create_mock_bus()
        router1 = SignalRouter(bus1)
        router1._on_raw_signal(signal)
        router1.flush()
        
        # Run 2
        bus2 = self._create_mock_bus()
        router2 = SignalRouter(bus2)
        router2._on_raw_signal(signal)
        router2.flush()
        
        ranked1 = [e for t, e in bus1.published if t == "signals.ranked"][0]
        ranked2 = [e for t, e in bus2.published if t == "signals.ranked"][0]
        
        assert ranked1.final_score == ranked2.final_score
        assert ranked1.rank == ranked2.rank
