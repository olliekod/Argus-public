"""
Tests for Tape Ordering and Replay.

Verifies:
1. Faithful replay: sequence_id order preserved exactly
2. Canonical replay: stable ordering independent of arrival
3. Sequence_id monotonicity
4. Replay determinism

Run with: pytest tests/test_tape_ordering.py -v
"""

import pytest
from typing import List, Dict, Any

from src.soak.tape import (
    TapeRecorder,
    PROVIDER_PRIORITY,
    EVENT_TYPE_PRIORITY,
    _faithful_sort_key,
    _canonical_sort_key,
    _quote_to_dict,
    _bar_to_dict,
    _tick_to_dict,
    _dict_to_event,
)
from src.core.events import QuoteEvent, BarEvent, MinuteTickEvent


# ═══════════════════════════════════════════════════════════════════════════
# Fixture Data
# ═══════════════════════════════════════════════════════════════════════════

def create_quote(symbol: str, price: float, source: str, ts: float) -> QuoteEvent:
    """Create a test QuoteEvent."""
    return QuoteEvent(
        symbol=symbol,
        bid=price - 0.01,
        ask=price + 0.01,
        mid=price,
        last=price,
        timestamp=ts,
        source=source,
        volume_24h=1000.0,
        source_ts=ts,
        event_ts=ts,
        receive_time=ts,
    )


def create_bar(symbol: str, close: float, source: str, ts: float) -> BarEvent:
    """Create a test BarEvent."""
    return BarEvent(
        symbol=symbol,
        open=close - 0.1,
        high=close + 0.2,
        low=close - 0.2,
        close=close,
        volume=1000.0,
        timestamp=ts,
        source=source,
        bar_duration=60,
        n_ticks=10,
        first_source_ts=ts,
        last_source_ts=ts,
        source_ts=ts,
    )


def create_minute_tick(ts: float) -> MinuteTickEvent:
    """Create a test MinuteTickEvent."""
    return MinuteTickEvent(timestamp=ts)


# ═══════════════════════════════════════════════════════════════════════════
# Sequence ID Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSequenceId:
    def test_monotonic_sequence_id(self):
        """Sequence IDs should be strictly increasing."""
        recorder = TapeRecorder(enabled=True, maxlen=100)
        
        # Simulate events by calling serialization with incrementing seq_id
        seq_ids = []
        for i in range(10):
            seq_id = recorder._get_next_sequence_id()
            seq_ids.append(seq_id)
        
        # Verify monotonic
        for i in range(1, len(seq_ids)):
            assert seq_ids[i] > seq_ids[i - 1], f"seq_id[{i}] not monotonic"
    
    def test_sequence_id_in_serialized_event(self):
        """Serialized events must include sequence_id."""
        seq_id = 42
        
        quote = create_quote("IBIT", 100.0, "alpaca", 1700000000.0)
        d = _quote_to_dict(quote, seq_id)
        assert d["sequence_id"] == seq_id
        
        bar = create_bar("BITO", 50.0, "yahoo", 1700000060.0)
        d = _bar_to_dict(bar, seq_id)
        assert d["sequence_id"] == seq_id
        
        tick = create_minute_tick(1700000120.0)
        d = _tick_to_dict(tick, seq_id)
        assert d["sequence_id"] == seq_id


# ═══════════════════════════════════════════════════════════════════════════
# Faithful Replay Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFaithfulReplay:
    def test_faithful_replay_preserves_order(self):
        """Faithful replay outputs events in exact sequence_id order."""
        # Create tape entries with out-of-order sequence_ids
        tape = [
            {"sequence_id": 5, "event_ts": 1700000000000, "provider": "alpaca", "event_type": "quote", "symbol": "IBIT", "timestamp": 1700000000000, "bid": 99, "ask": 101, "mid": 100, "last": 100, "source": "alpaca", "volume_24h": 1000, "source_ts": 1700000000000, "receive_time": 1700000000000, "timeframe": 0},
            {"sequence_id": 2, "event_ts": 1699999900000, "provider": "yahoo", "event_type": "quote", "symbol": "BITO", "timestamp": 1699999900000, "bid": 49, "ask": 51, "mid": 50, "last": 50, "source": "yahoo", "volume_24h": 500, "source_ts": 1699999900000, "receive_time": 1699999900000, "timeframe": 0},
            {"sequence_id": 8, "event_ts": 1700000100000, "provider": "alpaca", "event_type": "quote", "symbol": "IBIT", "timestamp": 1700000100000, "bid": 100, "ask": 102, "mid": 101, "last": 101, "source": "alpaca", "volume_24h": 1000, "source_ts": 1700000100000, "receive_time": 1700000100000, "timeframe": 0},
        ]
        
        # Faithful sort: by sequence_id
        sorted_tape = sorted(tape, key=_faithful_sort_key)
        
        assert sorted_tape[0]["sequence_id"] == 2
        assert sorted_tape[1]["sequence_id"] == 5
        assert sorted_tape[2]["sequence_id"] == 8
    
    def test_faithful_replay_identical_twice(self):
        """Replaying same tape faithfully twice yields identical order."""
        tape = [
            {"sequence_id": 3, "event_ts": 1700000000000, "provider": "alpaca", "event_type": "quote", "symbol": "IBIT", "timestamp": 1700000000000, "bid": 99, "ask": 101, "mid": 100, "last": 100, "source": "alpaca", "volume_24h": 1000, "source_ts": 1700000000000, "receive_time": 1700000000000, "timeframe": 0},
            {"sequence_id": 1, "event_ts": 1699999900000, "provider": "yahoo", "event_type": "quote", "symbol": "BITO", "timestamp": 1699999900000, "bid": 49, "ask": 51, "mid": 50, "last": 50, "source": "yahoo", "volume_24h": 500, "source_ts": 1699999900000, "receive_time": 1699999900000, "timeframe": 0},
            {"sequence_id": 2, "event_ts": 1700000050000, "provider": "bybit", "event_type": "quote", "symbol": "BTCUSDT", "timestamp": 1700000050000, "bid": 40000, "ask": 40001, "mid": 40000.5, "last": 40000, "source": "bybit", "volume_24h": 10000, "source_ts": 1700000050000, "receive_time": 1700000050000, "timeframe": 0},
        ]
        
        sorted1 = sorted(tape, key=_faithful_sort_key)
        sorted2 = sorted(tape, key=_faithful_sort_key)
        
        for i, (e1, e2) in enumerate(zip(sorted1, sorted2)):
            assert e1["sequence_id"] == e2["sequence_id"], f"Mismatch at {i}"


# ═══════════════════════════════════════════════════════════════════════════
# Canonical Replay Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCanonicalReplay:
    def test_canonical_sort_key_order(self):
        """Canonical sort: (event_ts, provider_priority, event_type_priority, symbol, sequence_id)."""
        # Same event_ts, different providers
        tape = [
            {"sequence_id": 1, "event_ts": 1700000000000, "provider": "bybit", "event_type": "quote", "symbol": "BTCUSDT", "timeframe": 0},
            {"sequence_id": 2, "event_ts": 1700000000000, "provider": "alpaca", "event_type": "quote", "symbol": "IBIT", "timeframe": 0},
        ]
        
        sorted_tape = sorted(tape, key=_canonical_sort_key)
        
        # alpaca (priority 1) should come before bybit (priority 3)
        assert sorted_tape[0]["provider"] == "alpaca"
        assert sorted_tape[1]["provider"] == "bybit"
    
    def test_canonical_stable_across_runs(self):
        """Canonical ordering must be stable across multiple runs."""
        tape = [
            {"sequence_id": 5, "event_ts": 1700000000000, "provider": "yahoo", "event_type": "bar", "symbol": "IBIT", "timeframe": 60},
            {"sequence_id": 1, "event_ts": 1700000000000, "provider": "alpaca", "event_type": "bar", "symbol": "IBIT", "timeframe": 60},
            {"sequence_id": 3, "event_ts": 1700000000000, "provider": "alpaca", "event_type": "quote", "symbol": "IBIT", "timeframe": 0},
            {"sequence_id": 2, "event_ts": 1699999940000, "provider": "bybit", "event_type": "quote", "symbol": "BTCUSDT", "timeframe": 0},
        ]
        
        # Run canonical sort 3 times
        sorted1 = sorted(tape, key=_canonical_sort_key)
        sorted2 = sorted(tape, key=_canonical_sort_key)
        sorted3 = sorted(tape, key=_canonical_sort_key)
        
        for i in range(len(tape)):
            assert sorted1[i] == sorted2[i] == sorted3[i], f"Not stable at {i}"
    
    def test_canonical_event_type_priority(self):
        """Canonical sort should order by event_type_priority within same timestamp/provider."""
        tape = [
            {"sequence_id": 1, "event_ts": 1700000000000, "provider": "alpaca", "event_type": "quote", "symbol": "IBIT", "timeframe": 0},
            {"sequence_id": 2, "event_ts": 1700000000000, "provider": "alpaca", "event_type": "bar", "symbol": "IBIT", "timeframe": 60},
        ]
        
        sorted_tape = sorted(tape, key=_canonical_sort_key)
        
        # bar (priority 1) should come before quote (priority 2)
        assert sorted_tape[0]["event_type"] == "bar"
        assert sorted_tape[1]["event_type"] == "quote"
    
    def test_canonical_sequence_id_tiebreaker(self):
        """sequence_id is the final tiebreaker in canonical sort."""
        tape = [
            {"sequence_id": 10, "event_ts": 1700000000000, "provider": "alpaca", "event_type": "bar", "symbol": "IBIT", "timeframe": 60},
            {"sequence_id": 5, "event_ts": 1700000000000, "provider": "alpaca", "event_type": "bar", "symbol": "IBIT", "timeframe": 60},
        ]
        
        sorted_tape = sorted(tape, key=_canonical_sort_key)
        
        # Lower sequence_id should come first as tiebreaker
        assert sorted_tape[0]["sequence_id"] == 5
        assert sorted_tape[1]["sequence_id"] == 10


# ═══════════════════════════════════════════════════════════════════════════
# Priority Table Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPriorityTables:
    def test_provider_priority_values(self):
        """Verify provider priority table values."""
        # Code is source of truth - updated to match actual tape.py
        assert PROVIDER_PRIORITY["alpaca"] == 1
        assert PROVIDER_PRIORITY["tradier"] == 2
        assert PROVIDER_PRIORITY["yahoo"] == 3
        assert PROVIDER_PRIORITY["bybit"] == 4
        assert PROVIDER_PRIORITY["binance"] == 5
        assert PROVIDER_PRIORITY["deribit"] == 6
        assert PROVIDER_PRIORITY["polymarket"] == 7
        assert PROVIDER_PRIORITY["unknown"] == 99
    
    def test_event_type_priority_values(self):
        """Verify event type priority table values."""
        # Code is source of truth - updated to match actual tape.py
        # Options events ordered by data dependency: contract → quote → chain
        assert EVENT_TYPE_PRIORITY["bar"] == 1
        assert EVENT_TYPE_PRIORITY["quote"] == 2
        assert EVENT_TYPE_PRIORITY["option_contract"] == 3  # Static metadata first
        assert EVENT_TYPE_PRIORITY["option_quote"] == 4     # Live quotes per contract
        assert EVENT_TYPE_PRIORITY["option_chain"] == 5     # Aggregate snapshot last
        assert EVENT_TYPE_PRIORITY["metric"] == 6
        assert EVENT_TYPE_PRIORITY["minute_tick"] == 7
        assert EVENT_TYPE_PRIORITY["signal"] == 8
        assert EVENT_TYPE_PRIORITY["heartbeat"] == 9


# ═══════════════════════════════════════════════════════════════════════════
# Envelope Schema Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvelopeSchema:
    def test_quote_envelope_fields(self):
        """Quote envelope must include all required fields."""
        quote = create_quote("IBIT", 100.0, "alpaca", 1700000000.0)
        d = _quote_to_dict(quote, 42)
        
        required = ["sequence_id", "event_ts", "provider", "event_type", "symbol", "timeframe"]
        for field in required:
            assert field in d, f"Missing envelope field: {field}"
        
        assert d["event_type"] == "quote"
        assert d["timeframe"] == 0  # N/A for quotes
    
    def test_bar_envelope_fields(self):
        """Bar envelope must include all required fields including timeframe."""
        bar = create_bar("IBIT", 100.0, "alpaca", 1700000000.0)
        d = _bar_to_dict(bar, 42)
        
        required = ["sequence_id", "event_ts", "provider", "event_type", "symbol", "timeframe"]
        for field in required:
            assert field in d, f"Missing envelope field: {field}"
        
        assert d["event_type"] == "bar"
        assert d["timeframe"] == 60  # bar_duration
    
    def test_minute_tick_envelope_fields(self):
        """Minute tick envelope must include all required fields."""
        tick = create_minute_tick(1700000000.0)
        d = _tick_to_dict(tick, 42)
        
        required = ["sequence_id", "event_ts", "provider", "event_type", "symbol"]
        for field in required:
            assert field in d, f"Missing envelope field: {field}"
        
        assert d["event_type"] == "minute_tick"
        assert d["provider"] == "system"


# ═══════════════════════════════════════════════════════════════════════════
# Timestamp Conversion Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTimestampConversion:
    def test_timestamps_are_int_ms(self):
        """Serialized timestamps should be int milliseconds."""
        quote = create_quote("IBIT", 100.0, "alpaca", 1700000000.0)
        d = _quote_to_dict(quote, 42)
        
        assert isinstance(d["event_ts"], int)
        assert isinstance(d["timestamp"], int)
        assert isinstance(d["source_ts"], int)
        
        # Should be in ms range (> 1e12)
        assert d["event_ts"] > 1e12
    
    def test_deserialize_preserves_event(self):
        """Deserialize should reconstruct the event correctly."""
        original_quote = create_quote("IBIT", 100.0, "alpaca", 1700000000.0)
        d = _quote_to_dict(original_quote, 42)
        
        reconstructed = _dict_to_event(d)
        
        assert reconstructed.symbol == original_quote.symbol
        assert reconstructed.source == original_quote.source
        assert abs(reconstructed.mid - original_quote.mid) < 0.01
