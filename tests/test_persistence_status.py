import asyncio
import tempfile
import threading
import time

from src.core.bus import EventBus
from src.core.events import BarEvent
from src.core.persistence import PersistenceManager, _SOURCE_TS_LOG_SYMBOL_MAX


class _DummyDB:
    async def execute_many(self, sql: str, params_list):
        await asyncio.sleep(0)

    async def execute(self, sql: str, params=()):
        await asyncio.sleep(0)

    async def insert_market_metric(self, **kwargs):
        await asyncio.sleep(0)


def _start_loop(loop: asyncio.AbstractEventLoop) -> threading.Thread:
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return thread


def test_persistence_flush_timing_updates_status():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            bar = BarEvent(
                symbol="BTC",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=time.time(),
                source="test",
                bar_duration=60,
                tick_count=1,
            )
            pm._on_bar(bar)
            pm._do_flush()
            status = pm.get_status()

            assert status["timing"]["last_latency_ms"] is not None
            assert status["extras"]["bars_writes_total"] == 1
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_persist_lag_ms_uses_sane_source_ts():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            bar = BarEvent(
                symbol="BTC",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="test",
                bar_duration=60,
                tick_count=1,
                source_ts=now - 0.2,
            )
            pm._on_bar(bar)
            pm._do_flush()
            status = pm.get_status()

            lag_ms = status["extras"]["persist_lag_ms"]
            assert lag_ms is not None
            assert lag_ms < 2_000
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_persist_lag_ignores_ms_source_ts():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            bar = BarEvent(
                symbol="BTC",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="test",
                bar_duration=60,
                tick_count=1,
                source_ts=now * 1000.0,
            )
            pm._on_bar(bar)
            pm._do_flush()
            status = pm.get_status()

            assert status["extras"]["persist_lag_ms"] is None
            assert status["extras"]["source_ts_units_discarded_total"] == 1
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_persist_lag_ema_ignores_stale_then_updates():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            stale_bar = BarEvent(
                symbol="BTC",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="test",
                bar_duration=60,
                tick_count=1,
                source_ts=now - 10_000,
                last_source_ts=now - 10_000,
            )
            pm._on_bar(stale_bar)
            pm._do_flush()
            status = pm.get_status()
            assert status["extras"]["persist_lag_ema_ms"] is None
            assert status["extras"]["source_ts_stale_ignored_total"] == 1

            fresh_bar = BarEvent(
                symbol="BTC",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="test",
                bar_duration=60,
                tick_count=1,
                source_ts=now - 0.1,
                last_source_ts=now - 0.1,
            )
            pm._on_bar(fresh_bar)
            pm._do_flush()
            status = pm.get_status()
            assert status["extras"]["persist_lag_ema_ms"] is not None
            assert status["extras"]["persist_lag_ema_ms"] < 2_000
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_persist_lag_future_ts_clamps_to_now():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            bar = BarEvent(
                symbol="BTCUSDT",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="bybit",
                bar_duration=60,
                tick_count=1,
                source_ts=now + 10.0,
                last_source_ts=now + 10.0,
            )
            pm._on_bar(bar)
            pm._do_flush()
            status = pm.get_status()

            assert status["extras"]["source_ts_future_clamped_total"] == 1
            assert status["extras"]["persist_lag_ms"] == 0.0
            assert status["extras"]["persist_lag_ema_ms"] is not None
            assert status["extras"]["persist_lag_ema_ms"] <= 5.0
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_persist_lag_small_future_skew_does_not_mark_stale():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            bar = BarEvent(
                symbol="BTCUSDT",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="bybit",
                bar_duration=60,
                tick_count=1,
                source_ts=now + 0.4,
                last_source_ts=now + 0.4,
            )
            pm._on_bar(bar)
            pm._do_flush()
            status = pm.get_status()

            assert status["extras"]["source_ts_future_clamped_total"] == 1
            assert status["extras"]["source_ts_stale_ignored_total"] == 0
            assert status["extras"]["persist_lag_ms"] == 0.0
            assert status["extras"]["persist_lag_ema_ms"] is not None
            assert status["extras"]["persist_lag_ema_ms"] >= 0.0
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_persist_lag_future_within_skew_clamps_and_counts():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            bar = BarEvent(
                symbol="BTCUSDT",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="bybit",
                bar_duration=60,
                tick_count=1,
                source_ts=now + 1.0,
                last_source_ts=now + 1.0,
            )
            pm._on_bar(bar)
            pm._do_flush()
            status = pm.get_status()

            assert status["extras"]["source_ts_future_clamped_total"] == 1
            assert status["extras"]["source_ts_future_clamped_by_symbol"]["BTCUSDT"] == 1
            assert status["extras"]["source_ts_stale_ignored_total"] == 0
            assert status["extras"]["persist_lag_ms"] == 0.0
            assert status["extras"]["persist_lag_ema_ms"] is not None
            assert status["extras"]["persist_lag_ema_ms"] >= 0.0
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_future_log_tracking_is_bounded():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            for i in range(_SOURCE_TS_LOG_SYMBOL_MAX + 10):
                bar = BarEvent(
                    symbol=f"SYM{i}",
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=10.0,
                    timestamp=now,
                    source="bybit",
                    bar_duration=60,
                    tick_count=1,
                    source_ts=now + 10.0,
                    last_source_ts=now + 10.0,
                )
                pm._extract_source_ts_for_lag(bar, now)

            assert len(pm._source_ts_future_log_ts_by_symbol) <= _SOURCE_TS_LOG_SYMBOL_MAX
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_persist_lag_ema_separates_crypto_and_deribit():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            crypto_bar = BarEvent(
                symbol="BTCUSDT",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="bybit",
                bar_duration=60,
                tick_count=1,
                source_ts=now - 0.1,
                last_source_ts=now - 0.1,
            )
            deribit_bar = BarEvent(
                symbol="BTC-INDEX",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="deribit",
                bar_duration=60,
                tick_count=1,
                source_ts=now - 20.0,
                last_source_ts=now - 20.0,
            )
            pm._on_bar(crypto_bar)
            pm._on_bar(deribit_bar)
            pm._do_flush()
            status = pm.get_status()

            crypto_ema = status["extras"]["persist_lag_crypto_ema_ms"]
            deribit_ema = status["extras"]["persist_lag_deribit_ema_ms"]
            overall_ema = status["extras"]["persist_lag_ema_ms"]
            assert crypto_ema is not None
            assert crypto_ema < 2_000
            assert deribit_ema is not None
            assert deribit_ema > 10_000
            assert overall_ema is not None
            assert overall_ema > crypto_ema
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
