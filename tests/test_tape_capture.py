"""
Tests for TapeRecorder boundedness and determinism replay.

Run with:  python -m pytest tests/test_tape_capture.py -v
"""

import json
import os
import tempfile
import time

from src.core.bar_builder import _minute_floor
from src.core.bus import EventBus
from src.core.events import (
    MinuteTickEvent,
    QuoteEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_QUOTES,
    TOPIC_SYSTEM_MINUTE_TICK,
)
from src.soak.tape import TapeRecorder, _quote_to_dict, _dict_to_quote, _dict_to_event


def _quote(symbol, price, volume, ts, source_ts=None):
    return QuoteEvent(
        symbol=symbol,
        bid=price - 0.01,
        ask=price + 0.01,
        mid=price,
        last=price,
        timestamp=ts,
        source="test",
        volume_24h=volume,
        source_ts=source_ts or ts,
        event_ts=ts,
        receive_time=ts,
    )


class TestTapeRecorderBoundedness:
    def test_disabled_by_default(self):
        t = TapeRecorder()
        assert not t.enabled
        # Operations are no-ops
        t._on_quote(_quote("BTC", 100, 1000, 1_700_000_001))
        assert t.get_status()["tape_size"] == 0

    def test_maxlen_enforced(self):
        t = TapeRecorder(enabled=True, maxlen=10)
        for i in range(20):
            t._on_quote(_quote("BTC", 100 + i, 1000, 1_700_000_000 + i))
        status = t.get_status()
        assert status["tape_size"] == 10
        assert status["events_captured"] == 20
        assert status["events_evicted"] == 10

    def test_symbol_filter(self):
        t = TapeRecorder(enabled=True, symbols={"BTC"})
        t._on_quote(_quote("BTC", 100, 1000, 1_700_000_001))
        t._on_quote(_quote("ETH", 200, 500, 1_700_000_002))
        assert t.get_status()["tape_size"] == 1

    def test_minute_tick_captured(self):
        t = TapeRecorder(enabled=True)
        t._on_minute_tick(MinuteTickEvent(timestamp=1_700_000_060))
        assert t.get_status()["tape_size"] == 1


class TestTapeSerialization:
    def test_quote_roundtrip(self):
        q = _quote("BTC", 100.5, 1000.0, 1_700_000_001.5, source_ts=1_700_000_001.0)
        d = _quote_to_dict(q)
        assert d["type"] == "quote"
        assert d["symbol"] == "BTC"

        q2 = _dict_to_quote(d)
        assert q2.symbol == q.symbol
        assert q2.mid == q.mid
        assert q2.source_ts == q.source_ts

    def test_event_dispatch(self):
        qd = {"type": "quote", "symbol": "BTC", "bid": 99.99, "ask": 100.01,
               "mid": 100.0, "last": 100.0, "timestamp": 1_700_000_001,
               "source": "test", "volume_24h": 1000.0, "source_ts": 1_700_000_001}
        td = {"type": "minute_tick", "timestamp": 1_700_000_060}

        assert isinstance(_dict_to_event(qd), QuoteEvent)
        assert isinstance(_dict_to_event(td), MinuteTickEvent)


class TestTapeExport:
    def test_export_and_load(self):
        t = TapeRecorder(enabled=True, maxlen=100)
        for i in range(5):
            t._on_quote(_quote("BTC", 100 + i, 1000, 1_700_000_000 + i))
        t._on_minute_tick(MinuteTickEvent(timestamp=1_700_000_060))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            count = t.export_jsonl(path)
            assert count == 6

            loaded = TapeRecorder.load_tape(path)
            assert len(loaded) == 6
            assert loaded[0]["type"] == "quote"
            assert loaded[-1]["type"] == "minute_tick"
        finally:
            os.unlink(path)

    def test_export_with_time_filter(self):
        now = time.time()
        t = TapeRecorder(enabled=True, maxlen=100)
        # Old event
        t._on_quote(_quote("BTC", 100, 1000, now - 7200))
        # Recent event
        t._on_quote(_quote("BTC", 101, 1010, now - 60))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            count = t.export_jsonl(path, last_n_minutes=30)
            assert count == 1  # Only the recent one
        finally:
            os.unlink(path)


class TestTapeReplay:
    def test_replay_determinism(self):
        """Two replays of the same tape produce identical bars."""
        BASE = 1_700_000_000.0
        M0 = _minute_floor(BASE)
        M1 = M0 + 60

        tape = [
            _quote_to_dict(_quote("BTC", 100, 1000, M0 + 1, source_ts=M0 + 0.5)),
            _quote_to_dict(_quote("BTC", 101, 1020, M0 + 15, source_ts=M0 + 14.8)),
            _quote_to_dict(_quote("BTC", 102, 1040, M1 + 1, source_ts=M1 + 0.5)),
            {"type": "minute_tick", "timestamp": M1 + 60},
        ]

        bars1 = TapeRecorder.replay_tape(tape)
        bars2 = TapeRecorder.replay_tape(tape)

        assert len(bars1) == len(bars2)
        assert len(bars1) >= 1

        for b1, b2 in zip(bars1, bars2):
            assert b1.symbol == b2.symbol
            assert b1.open == b2.open
            assert b1.high == b2.high
            assert b1.low == b2.low
            assert b1.close == b2.close
            assert b1.volume == b2.volume
            assert b1.timestamp == b2.timestamp


class TestTapeStatus:
    def test_status_shape(self):
        t = TapeRecorder(enabled=True, symbols={"BTC", "ETH"}, maxlen=50_000)
        status = t.get_status()
        assert status["enabled"] is True
        assert status["maxlen"] == 50_000
        assert status["tape_size"] == 0
        assert status["events_captured"] == 0
        assert status["events_evicted"] == 0
        assert sorted(status["symbols"]) == ["BTC", "ETH"]
