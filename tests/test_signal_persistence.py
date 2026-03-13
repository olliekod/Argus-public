"""
Signal Persistence Tests
========================

Ensures raw signals are persisted exactly once.
"""

import asyncio
import threading
import time

from src.core.bus import EventBus
from src.core.events import TOPIC_SIGNALS_RAW
from src.core.persistence import PersistenceManager
from src.core.signals import SignalEvent, compute_signal_id, DIRECTION_LONG, SIGNAL_TYPE_ENTRY


class _DummyDB:
    def __init__(self) -> None:
        self.signal_calls = []

    async def write_signal(self, **kwargs) -> None:
        self.signal_calls.append(kwargs)
        await asyncio.sleep(0)

    async def execute_many(self, sql: str, params_list):
        await asyncio.sleep(0)

    async def execute(self, sql: str, params=()):
        await asyncio.sleep(0)

    async def insert_market_metric(self, **kwargs) -> None:
        await asyncio.sleep(0)

    async def insert_component_heartbeat(self, **kwargs) -> None:
        await asyncio.sleep(0)


def _start_loop(loop: asyncio.AbstractEventLoop) -> threading.Thread:
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return thread


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_signal_raw_persists_once():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)
    bus = EventBus()
    db = _DummyDB()
    pm = PersistenceManager(bus, db, loop)
    bus.start()
    pm.start()

    try:
        timestamp_ms = 1_700_000_000_000
        signal = SignalEvent(
            timestamp_ms=timestamp_ms,
            strategy_id="TEST_STRAT",
            config_hash="cfg123",
            symbol="BTC",
            direction=DIRECTION_LONG,
            signal_type=SIGNAL_TYPE_ENTRY,
            timeframe=60,
            confidence=0.9,
            idempotency_key=compute_signal_id(
                "TEST_STRAT", "cfg123", "BTC", timestamp_ms
            ),
        )

        bus.publish(TOPIC_SIGNALS_RAW, signal)

        assert _wait_for(lambda: len(db.signal_calls) == 1)
        assert len(db.signal_calls) == 1
    finally:
        pm.shutdown()
        bus.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
