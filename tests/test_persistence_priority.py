"""
Tests for persistence priority behavior: bars never dropped, retry on failure.
Also tests bounded bar buffer with disk spool overflow.

Run with:  python -m pytest tests/test_persistence_priority.py -v
"""

import asyncio
import json
import os
import tempfile
import threading
import time

from src.core.bus import EventBus
from src.core.events import BarEvent, MetricEvent, SignalEvent
from src.core.persistence import PersistenceManager


class _DummyDB:
    """DB stub that tracks calls and can simulate failures."""

    def __init__(self, fail_count=0):
        self.bar_batches = []
        self._fail_count = fail_count
        self._call_count = 0

    async def execute_many(self, sql: str, params_list):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise RuntimeError("Simulated DB failure")
        self.bar_batches.append(params_list)
        await asyncio.sleep(0)

    async def execute(self, sql: str, params=()):
        await asyncio.sleep(0)

    async def insert_market_metric(self, **kwargs):
        await asyncio.sleep(0)


def _start_loop(loop: asyncio.AbstractEventLoop) -> threading.Thread:
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return thread


def _make_bar(symbol="BTC", ts=None):
    return BarEvent(
        symbol=symbol,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
        timestamp=ts or time.time(),
        source="test",
        bar_duration=60,
        tick_count=1,
    )


class TestBarNeverDropped:
    def test_bars_buffered_within_limit(self):
        """Bars fit in bounded buffer when under max."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td,
                                        bar_buffer_max=5000)

                # Add many bars (under limit)
                for i in range(1000):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))

                assert len(pm._bar_buffer) == 1000
                assert not pm._spool_active
                pm.shutdown()  # Close spool file before temp dir cleanup
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


    def test_flush_success_clears_buffer(self):
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)

                pm._on_bar(_make_bar(ts=1_700_000_000.0))
                pm._on_bar(_make_bar(ts=1_700_000_060.0))
                assert len(pm._bar_buffer) == 2

                pm._do_flush()

                assert len(pm._bar_buffer) == 0
                assert pm._bars_dropped_total == 0
                assert pm._bar_flush_success_total == 1
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_flush_failure_returns_bars_to_buffer(self):
        """On total flush failure, bars go back to the buffer."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(fail_count=999)  # Always fail
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)

                pm._on_bar(_make_bar(ts=1_700_000_000.0))
                pm._on_bar(_make_bar(ts=1_700_000_060.0))
                pm._do_flush()

                assert len(pm._bar_buffer) == 2
                assert pm._bars_dropped_total == 0
                assert pm._bar_flush_failure_total == 1
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_flush_retry_succeeds_on_second_attempt(self):
        """Bars are flushed on retry after initial failure."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(fail_count=1)  # Fail first attempt, succeed on retry
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)

                pm._on_bar(_make_bar(ts=1_700_000_000.0))
                pm._do_flush()

                assert len(pm._bar_buffer) == 0
                assert pm._bars_dropped_total == 0
                assert pm._bar_retry_count >= 1
                assert pm._bar_flush_success_total == 1
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_bar_drop_count_always_zero(self):
        """bars_dropped_total must remain zero in all normal paths."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)

                for i in range(100):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                pm._do_flush()

                assert pm._bars_dropped_total == 0
                status = pm.get_status()
                assert status["extras"]["bars_dropped_total"] == 0
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


class TestSpoolOverflow:
    def test_overflow_spools_to_disk(self):
        """When buffer is full, bars spill to disk spool."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td,
                                        bar_buffer_max=10)

                # Fill buffer to max
                for i in range(10):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                assert len(pm._bar_buffer) == 10
                assert not pm._spool_active

                # Next bar should go to spool
                pm._on_bar(_make_bar(ts=1_700_000_600.0))
                assert pm._spool_active
                assert pm._spool_bars_pending == 1
                assert pm._bars_spooled_total == 1
                assert len(pm._bar_buffer) == 10  # buffer unchanged

                pm._on_bar(_make_bar(ts=1_700_000_660.0))
                assert pm._spool_bars_pending == 2
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_spool_drained_after_flush(self):
        """After buffer flush, spool is drained to DB."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td,
                                        bar_buffer_max=5)

                # Fill buffer + overflow to spool
                for i in range(8):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))

                assert len(pm._bar_buffer) == 5
                assert pm._spool_bars_pending == 3

                # Flush should drain buffer then spool
                pm._do_flush()

                assert len(pm._bar_buffer) == 0
                assert pm._bars_dropped_total == 0
                assert not pm._spool_active
                assert pm._spool_bars_pending == 0
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_no_bars_dropped_during_overflow(self):
        """Zero bars are dropped even during overflow."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td,
                                        bar_buffer_max=5)

                for i in range(20):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))

                # All bars accounted for: buffer + spool
                total = len(pm._bar_buffer) + pm._spool_bars_pending
                assert total == 20
                assert pm._bars_dropped_total == 0

                # Flush everything
                pm._do_flush()
                assert pm._bars_dropped_total == 0
                # All bars written to DB
                total_written = sum(len(b) for b in db.bar_batches)
                assert total_written == 20
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


    def test_spool_recovery_on_restart(self):
        """Leftover spool from previous crash is detected."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                # Simulate a leftover spool file
                spool_path = os.path.join(td, "bar_spool.jsonl")
                bar = _make_bar(ts=1_700_000_000.0)
                from dataclasses import asdict
                with open(spool_path, "w") as f:
                    f.write(json.dumps(asdict(bar)) + "\n")
                    f.write(json.dumps(asdict(bar)) + "\n")

                pm = PersistenceManager(bus, db, loop, spool_dir=td)
                assert pm._spool_active
                assert pm._spool_bars_pending == 2

                pm._do_flush()
                assert not pm._spool_active
                assert pm._spool_bars_pending == 0
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_status_includes_spool_metrics(self):
        """Status dict exposes spool state."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td,
                                        bar_buffer_max=5)
                # Trigger spool
                for i in range(7):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))

                status = pm.get_status()
                extras = status["extras"]
                assert extras["spool_active"] is True
                assert extras["spool_bars_pending"] == 2
                assert extras["spool_file_size"] > 0
                assert extras["bar_buffer_max"] == 5
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


class TestPriorityQueues:
    def test_signal_and_telemetry_queues_separate(self):
        """Signals and metrics use different queues."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)

                assert pm._signal_queue is not pm._telemetry_queue
                assert pm._signal_queue.maxsize > 0
                assert pm._telemetry_queue.maxsize > 0
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_signal_dropped_counter(self):
        """Signals dropped when signal queue is full."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)

                # Fill the signal queue manually
                from queue import Full
                for _ in range(pm._signal_queue.maxsize):
                    pm._signal_queue.put_nowait(("signal", None))

                # This should trigger a drop
                signal = SignalEvent(
                    detector="test",
                    symbol="BTC",
                    signal_type="test",
                    priority=1,
                    data={},
                    timestamp=time.time(),
                )
                pm._on_signal(signal)
                assert pm._signals_dropped_total == 1
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_metric_dropped_counter(self):
        """Metrics dropped when telemetry queue is full."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)

                # Fill the telemetry queue
                for _ in range(pm._telemetry_queue.maxsize):
                    pm._telemetry_queue.put_nowait(("metric", None))

                metric = MetricEvent(
                    symbol="BTC",
                    metric="test",
                    value=1.0,
                    source="test",
                    timestamp=time.time(),
                )
                pm._on_metric(metric)
                assert pm._metrics_dropped_total == 1
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


class TestStatusCounters:
    def test_status_includes_new_counters(self):
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)
                status = pm.get_status()

                extras = status["extras"]
                assert "bars_dropped_total" in extras
                assert "bar_flush_success_total" in extras
                assert "bar_flush_failure_total" in extras
                assert "bar_retry_count" in extras
                assert "signal_queue_depth" in extras
                assert "telemetry_queue_depth" in extras
                assert extras["bars_dropped_total"] == 0
                assert "spool_active" in extras
                assert "spool_bars_pending" in extras
                assert "bars_spooled_total" in extras
                pm.shutdown()  # Close spool file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
