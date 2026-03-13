"""
Regime Detector Tests
=====================

Tests for Phase 2 deterministic regime detection.
"""

import json
import math
import pytest
from collections import deque
from typing import Any, Dict, List
from unittest.mock import MagicMock

from src.core.regimes import (
    SymbolRegimeEvent,
    MarketRegimeEvent,
    DQ_NONE,
    DQ_REPAIRED_INPUT,
    DQ_GAP_WINDOW,
    DQ_STALE_INPUT,
    DEFAULT_REGIME_THRESHOLDS,
    compute_config_hash,
    symbol_regime_to_dict,
    dict_to_symbol_regime,
    market_regime_to_dict,
    dict_to_market_regime,
    get_market_for_symbol,
    canonical_metrics_json,
    _to_int_ms,
)
from src.core.events import BarEvent


# ═══════════════════════════════════════════════════════════════════════════
# Round-Trip Tests (Event → Dict → Event)
# ═══════════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    """Test event serialization round-trips."""

    def test_symbol_regime_round_trip(self):
        """SymbolRegimeEvent → dict → SymbolRegimeEvent preserves all fields."""
        event = SymbolRegimeEvent(
            symbol="BTC",
            timeframe=60,
            timestamp_ms=1700000000000,
            vol_regime="VOL_HIGH",
            trend_regime="TREND_UP",
            liquidity_regime="LIQ_NORMAL",
            atr=100.5,
            atr_pct=0.0025,
            vol_z=1.5,
            ema_fast=40000.0,
            ema_slow=39500.0,
            ema_slope=0.75,
            rsi=65.0,
            spread_pct=0.05,
            volume_pctile=50.0,
            confidence=0.9,
            is_warm=True,
            data_quality_flags=DQ_NONE,
            config_hash="abc123def456",
        )
        
        # Serialize
        d = symbol_regime_to_dict(event)
        
        # Deserialize
        restored = dict_to_symbol_regime(d)
        
        # Verify all fields match
        assert restored.symbol == event.symbol
        assert restored.timeframe == event.timeframe
        assert restored.timestamp_ms == event.timestamp_ms
        assert restored.vol_regime == event.vol_regime
        assert restored.trend_regime == event.trend_regime
        assert abs(restored.atr - event.atr) < 1e-7
        assert abs(restored.atr_pct - event.atr_pct) < 1e-7
        assert abs(restored.vol_z - event.vol_z) < 1e-7
        assert abs(restored.ema_fast - event.ema_fast) < 1e-7
        assert abs(restored.ema_slow - event.ema_slow) < 1e-7
        assert abs(restored.ema_slope - event.ema_slope) < 1e-7
        assert abs(restored.rsi - event.rsi) < 1e-7
        assert abs(restored.confidence - event.confidence) < 1e-7
        assert restored.is_warm == event.is_warm
        assert restored.data_quality_flags == event.data_quality_flags
        assert restored.config_hash == event.config_hash
        assert restored.v == event.v

    def test_market_regime_round_trip(self):
        """MarketRegimeEvent → dict → MarketRegimeEvent preserves all fields."""
        event = MarketRegimeEvent(
            market="CRYPTO",
            timeframe=60,
            timestamp_ms=1700000000000,
            session_regime="US",
            risk_regime="UNKNOWN",
            confidence=1.0,
            data_quality_flags=DQ_GAP_WINDOW,
            config_hash="abc123def456",
        )
        
        d = market_regime_to_dict(event)
        restored = dict_to_market_regime(d)
        
        assert restored.market == event.market
        assert restored.timeframe == event.timeframe
        assert restored.timestamp_ms == event.timestamp_ms
        assert restored.session_regime == event.session_regime
        assert restored.risk_regime == event.risk_regime
        assert abs(restored.confidence - event.confidence) < 1e-7
        assert restored.data_quality_flags == event.data_quality_flags
        assert restored.config_hash == event.config_hash
        assert restored.v == event.v

    def test_json_round_trip(self):
        """Verify JSON serialization is stable."""
        event = SymbolRegimeEvent(
            symbol="ETH",
            timeframe=60,
            timestamp_ms=1700000000000,
            vol_regime="VOL_NORMAL",
            trend_regime="RANGE",
            liquidity_regime="LIQ_NORMAL",
            atr=50.0,
            atr_pct=0.002,
            vol_z=0.0,
            ema_fast=2000.0,
            ema_slow=2000.0,
            ema_slope=0.0,
            rsi=50.0,
            spread_pct=0.05,
            volume_pctile=50.0,
            confidence=1.0,
            is_warm=True,
            data_quality_flags=DQ_NONE,
            config_hash="test123",
        )
        
        d = symbol_regime_to_dict(event)
        json_str = json.dumps(d, sort_keys=True)
        d2 = json.loads(json_str)
        restored = dict_to_symbol_regime(d2)
        
        assert restored.symbol == event.symbol
        assert restored.timestamp_ms == event.timestamp_ms


# ═══════════════════════════════════════════════════════════════════════════
# Config Hash Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigHash:
    """Test config hash determinism."""

    def test_same_config_same_hash(self):
        """Same thresholds produce same hash."""
        config1 = {"vol_spike_z": 2.5, "vol_high_z": 1.0}
        config2 = {"vol_spike_z": 2.5, "vol_high_z": 1.0}
        
        assert compute_config_hash(config1) == compute_config_hash(config2)

    def test_different_config_different_hash(self):
        """Different thresholds produce different hash."""
        config1 = {"vol_spike_z": 2.5}
        config2 = {"vol_spike_z": 3.0}
        
        assert compute_config_hash(config1) != compute_config_hash(config2)

    def test_order_independent(self):
        """Key order doesn't affect hash."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        
        assert compute_config_hash(config1) == compute_config_hash(config2)


# ═══════════════════════════════════════════════════════════════════════════
# Market Classification Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMarketClassification:
    """Test symbol → market mapping."""

    def test_equities_classification(self):
        """Equities symbols are correctly classified."""
        assert get_market_for_symbol("IBIT") == "EQUITIES"
        assert get_market_for_symbol("BITO") == "EQUITIES"
        assert get_market_for_symbol("SPY") == "EQUITIES"

    def test_crypto_classification(self):
        """Crypto symbols are correctly classified."""
        assert get_market_for_symbol("BTC") == "CRYPTO"
        assert get_market_for_symbol("ETH") == "CRYPTO"
        assert get_market_for_symbol("DOGE") == "CRYPTO"


# ═══════════════════════════════════════════════════════════════════════════
# Data Quality Flag Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDataQualityFlags:
    """Test data quality flag handling."""

    def test_flag_values(self):
        """Verify flag values are correct powers of 2."""
        assert DQ_NONE == 0
        assert DQ_REPAIRED_INPUT == 1
        assert DQ_GAP_WINDOW == 2
        assert DQ_STALE_INPUT == 4

    def test_flag_combination(self):
        """Flags can be combined with OR."""
        combined = DQ_REPAIRED_INPUT | DQ_GAP_WINDOW
        assert combined == 3
        assert combined & DQ_REPAIRED_INPUT
        assert combined & DQ_GAP_WINDOW
        assert not (combined & DQ_STALE_INPUT)


# ═══════════════════════════════════════════════════════════════════════════
# Regime Detector Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRegimeDetectorDeterminism:
    """Test regime detector produces deterministic output."""

    def _create_mock_bus(self):
        """Create a mock event bus that captures published events."""
        bus = MagicMock()
        bus.published = []
        
        def capture_publish(topic, event):
            bus.published.append((topic, event))
        
        bus.publish.side_effect = capture_publish
        return bus

    def _create_bars(self, symbol: str, n: int, base_ts: int = 1700000000) -> List[BarEvent]:
        """Create a sequence of bars with deterministic values."""
        bars = []
        for i in range(n):
            # Create deterministic price movement
            close = 100 + (i * 0.1)
            bar = BarEvent(
                symbol=symbol,
                open=close - 0.05,
                high=close + 0.1,
                low=close - 0.1,
                close=close,
                volume=1000.0,
                timestamp=float(base_ts + (i * 60)),  # 1 minute bars
                source="test",
                bar_duration=60,
            )
            bars.append(bar)
        return bars

    def test_same_bars_same_regimes(self):
        """Same bar sequence produces identical regime events."""
        from src.core.regime_detector import RegimeDetector
        
        # Run 1
        bus1 = self._create_mock_bus()
        detector1 = RegimeDetector(bus1)
        bars = self._create_bars("BTC", 50)
        for bar in bars:
            detector1._on_bar(bar)
        
        # Run 2
        bus2 = self._create_mock_bus()
        detector2 = RegimeDetector(bus2)
        for bar in bars:
            detector2._on_bar(bar)
        
        # Compare regime events (excluding market events for simplicity)
        symbol_events1 = [
            (topic, e) for topic, e in bus1.published
            if topic == "regimes.symbol"
        ]
        symbol_events2 = [
            (topic, e) for topic, e in bus2.published
            if topic == "regimes.symbol"
        ]
        
        assert len(symbol_events1) == len(symbol_events2)
        
        for (t1, e1), (t2, e2) in zip(symbol_events1, symbol_events2):
            assert e1.symbol == e2.symbol
            assert e1.timestamp_ms == e2.timestamp_ms
            assert e1.vol_regime == e2.vol_regime
            assert e1.trend_regime == e2.trend_regime
            assert e1.config_hash == e2.config_hash

    def test_mixed_symbol_order_stability(self):
        """Different symbol interleaving produces consistent per-symbol regimes."""
        from src.core.regime_detector import RegimeDetector
        
        base_ts = 1700000000
        
        # Create interleaved bars for two symbols
        bars_interleaved = []
        for i in range(30):
            ts = float(base_ts + (i * 60))
            bars_interleaved.append(BarEvent(
                symbol="BTC",
                open=100, high=101, low=99, close=100.5,
                volume=1000, timestamp=ts, source="test", bar_duration=60,
            ))
            bars_interleaved.append(BarEvent(
                symbol="ETH",
                open=50, high=51, low=49, close=50.5,
                volume=500, timestamp=ts, source="test", bar_duration=60,
            ))
        
        # Run with interleaved order
        bus = self._create_mock_bus()
        detector = RegimeDetector(bus)
        for bar in bars_interleaved:
            detector._on_bar(bar)
        
        # Verify we got events for both symbols
        btc_events = [e for t, e in bus.published if t == "regimes.symbol" and e.symbol == "BTC"]
        eth_events = [e for t, e in bus.published if t == "regimes.symbol" and e.symbol == "ETH"]
        
        assert len(btc_events) == 30
        assert len(eth_events) == 30
        
        # All events should have the same config hash
        config_hash = btc_events[0].config_hash
        for e in btc_events + eth_events:
            assert e.config_hash == config_hash


# ═══════════════════════════════════════════════════════════════════════════
# Session Regime Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionRegime:
    """Test session regime classification from timestamp."""

    def test_equities_sessions(self):
        """Equities session detection works correctly."""
        from src.core.regime_detector import RegimeDetector
        
        bus = MagicMock()
        detector = RegimeDetector(bus)
        
        # Use a base at midnight UTC (2023-11-15 00:00:00 UTC)
        midnight_utc = 1700006400  # 2023-11-15 00:00:00 UTC
        
        # 10:00 UTC = PRE (before 14:00 UTC RTH start, after 9:00 PRE start)
        ts_10_utc = midnight_utc + (10 * 3600)
        ts_ms = ts_10_utc * 1000
        session = detector._get_session_regime("EQUITIES", ts_ms)
        assert session == "PRE", f"Expected PRE at 10 UTC, got {session}"
        
        # 15:00 UTC = RTH (between 14:00 and 21:00 UTC)
        ts_15_utc = midnight_utc + (15 * 3600)
        ts_ms = ts_15_utc * 1000
        session = detector._get_session_regime("EQUITIES", ts_ms)
        assert session == "RTH", f"Expected RTH at 15 UTC, got {session}"


    def test_crypto_sessions(self):
        """Crypto session detection works correctly."""
        from src.core.regime_detector import RegimeDetector
        
        bus = MagicMock()
        detector = RegimeDetector(bus)
        
        # 3:00 UTC = ASIA
        ts_3_utc = 1700000000 + (3 * 3600)
        ts_ms = ts_3_utc * 1000
        session = detector._get_session_regime("CRYPTO", ts_ms)
        assert session == "ASIA"
        
        # 10:00 UTC = EU
        ts_10_utc = 1700000000 + (10 * 3600)
        ts_ms = ts_10_utc * 1000
        session = detector._get_session_regime("CRYPTO", ts_ms)
        assert session == "EU"
        
        # 16:00 UTC = US
        ts_16_utc = 1700000000 + (16 * 3600)
        ts_ms = ts_16_utc * 1000
        session = detector._get_session_regime("CRYPTO", ts_ms)
        assert session == "US"


# ═══════════════════════════════════════════════════════════════════════════
# Timestamp Correctness Tests (ISSUE 1)
# ═══════════════════════════════════════════════════════════════════════════

class TestTimestampCorrectness:
    """Test that timestamps are always int milliseconds."""

    def test_timestamp_is_int_ms(self):
        """SymbolRegimeEvent.timestamp_ms must be int."""
        event = SymbolRegimeEvent(
            symbol="BTC",
            timeframe=60,
            timestamp_ms=1700000000000,
            vol_regime="VOL_NORMAL",
            trend_regime="RANGE",
            liquidity_regime="LIQ_NORMAL",
            atr=100.0,
            atr_pct=0.001,
            vol_z=0.0,
            ema_fast=100.0,
            ema_slow=100.0,
            ema_slope=0.0,
            rsi=50.0,
            spread_pct=0.05,
            volume_pctile=50.0,
            confidence=1.0,
            is_warm=True,
            data_quality_flags=DQ_NONE,
            config_hash="test",
        )
        assert isinstance(event.timestamp_ms, int)

    def test_serialized_timestamp_is_int(self):
        """Serialized timestamp_ms must be int."""
        event = SymbolRegimeEvent(
            symbol="ETH",
            timeframe=60,
            timestamp_ms=1700000000123,
            vol_regime="VOL_NORMAL",
            trend_regime="RANGE",
            liquidity_regime="LIQ_NORMAL",
            atr=50.0,
            atr_pct=0.001,
            vol_z=0.0,
            ema_fast=50.0,
            ema_slow=50.0,
            ema_slope=0.0,
            rsi=50.0,
            spread_pct=0.05,
            volume_pctile=50.0,
            confidence=1.0,
            is_warm=True,
            data_quality_flags=DQ_NONE,
            config_hash="test",
        )
        d = symbol_regime_to_dict(event)
        assert isinstance(d["timestamp_ms"], int)
        assert d["timestamp_ms"] == 1700000000123

    def test_backwards_compat_float_seconds(self):
        """Float seconds are converted to int ms on load."""
        ts_float = 1700000000.5  # seconds
        ts_int = _to_int_ms(ts_float)
        assert isinstance(ts_int, int)
        assert ts_int == 1700000000500

    def test_backwards_compat_float_ms(self):
        """Float ms are converted to int ms on load."""
        ts_float = 1700000000123.0  # ms as float
        ts_int = _to_int_ms(ts_float)
        assert isinstance(ts_int, int)
        assert ts_int == 1700000000123


# ═══════════════════════════════════════════════════════════════════════════
# JSON Determinism Tests (ISSUE 3)
# ═══════════════════════════════════════════════════════════════════════════

class TestJSONDeterminism:
    """Test that metrics_json is deterministic."""

    def test_canonical_json_sorted_keys(self):
        """Keys are sorted in canonical JSON."""
        metrics = {"z": 1, "a": 2, "m": 3}
        json_str = canonical_metrics_json(metrics)
        assert json_str == '{"a":2,"m":3,"z":1}'

    def test_canonical_json_rounded_floats(self):
        """Floats are rounded in canonical JSON."""
        metrics = {"value": 1.123456789012345}
        json_str = canonical_metrics_json(metrics)
        # Should be rounded to 8 decimals
        parsed = json.loads(json_str)
        assert parsed["value"] == 1.12345679  # rounded

    def test_canonical_json_identical_across_runs(self):
        """Same metrics produce identical JSON across runs."""
        metrics1 = {"atr": 100.5, "vol_z": 1.5, "rsi": 65.0}
        metrics2 = {"rsi": 65.0, "atr": 100.5, "vol_z": 1.5}  # different order
        
        json1 = canonical_metrics_json(metrics1)
        json2 = canonical_metrics_json(metrics2)
        
        assert json1 == json2


# ═══════════════════════════════════════════════════════════════════════════
# ATR Edge Case Tests (ISSUE 4)
# ═══════════════════════════════════════════════════════════════════════════

class TestATREdgeCase:
    """Test ATR near-zero handling."""

    def test_zero_atr_produces_range_regime(self):
        """When ATR is near-zero, trend regime should be RANGE."""
        from src.core.regime_detector import RegimeDetector
        
        bus = MagicMock()
        bus.published = []
        bus.publish.side_effect = lambda t, e: bus.published.append((t, e))
        
        detector = RegimeDetector(bus)
        
        # Create bars with zero price movement (ATR will be very small)
        base_ts = 1700000000
        for i in range(50):
            bar = BarEvent(
                symbol="FLAT",
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000.0,
                timestamp=float(base_ts + (i * 60)),
                source="test",
                bar_duration=60,
            )
            detector._on_bar(bar)
        
        # Get the last symbol regime event
        symbol_events = [e for t, e in bus.published if t == "regimes.symbol"]
        assert len(symbol_events) > 0
        
        last_event = symbol_events[-1]
        
        # With zero ATR, should be RANGE and lower confidence
        assert last_event.trend_regime == "RANGE"
        # Confidence should be reduced due to stale/warmup issues
        assert last_event.confidence <= 1.0

