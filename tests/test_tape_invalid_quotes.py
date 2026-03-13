"""
Tape replay regression for invalid quote filtering.

Run with: python -m pytest tests/test_tape_invalid_quotes.py -v
"""

import time

from src.core.bar_builder import BarBuilder, _minute_floor
from src.core.bus import EventBus
from src.core.events import QuoteEvent, MinuteTickEvent, TOPIC_MARKET_BARS
from src.soak.tape import _dict_to_event


def _drain(bus, timeout=0.5):
    """Wait until all bus queues are empty or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        depths = bus.get_queue_depths()
        if all(d == 0 for d in depths.values()):
            return
        time.sleep(0.01)


class TestTapeReplayInvalidQuotes:
    def test_invalid_quote_is_filtered_during_replay(self):
        bus = EventBus()
        emitted = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            base = 1_700_000_000.0
            minute0 = _minute_floor(base)
            minute1 = minute0 + 60

            tape = [
                {
                    "type": "quote",
                    "symbol": "BTCUSDT",
                    "bid": 0.0,
                    "ask": 0.0,
                    "mid": 0.0,
                    "last": 0.0,
                    "timestamp": minute0 + 1,
                    "source": "bybit",
                    "volume_24h": 0.0,
                    "source_ts": minute0 + 1,
                    "event_ts": minute0 + 1,
                    "receive_time": minute0 + 1,
                },
                {
                    "type": "quote",
                    "symbol": "BTCUSDT",
                    "bid": 100.0,
                    "ask": 101.0,
                    "mid": 100.5,
                    "last": 100.5,
                    "timestamp": minute0 + 10,
                    "source": "bybit",
                    "volume_24h": 1000.0,
                    "source_ts": minute0 + 10,
                    "event_ts": minute0 + 10,
                    "receive_time": minute0 + 10,
                },
                {
                    "type": "quote",
                    "symbol": "BTCUSDT",
                    "bid": 102.0,
                    "ask": 103.0,
                    "mid": 102.5,
                    "last": 102.5,
                    "timestamp": minute1 + 1,
                    "source": "bybit",
                    "volume_24h": 1010.0,
                    "source_ts": minute1 + 1,
                    "event_ts": minute1 + 1,
                    "receive_time": minute1 + 1,
                },
            ]

            for entry in tape:
                event = _dict_to_event(entry)
                if isinstance(event, QuoteEvent):
                    bb._on_quote(event)
                elif isinstance(event, MinuteTickEvent):
                    bb._on_minute_tick(event)

            bb.flush()
            _drain(bus)

            assert len(emitted) == 2
            status = bb.get_status()
            extras = status["extras"]
            assert extras["quotes_rejected_invalid_price_total"] == 1
            assert extras["quotes_rejected_invalid_price_by_symbol"]["BTCUSDT"] == 1
        finally:
            bus.stop()
