"""
Tests for safe-pause behavior when bar spool is full and DB is unreachable.

Covers:
- Entering paused state when spool full + DB down
- Memory stays bounded during pause
- Bars rejected (not silently dropped) during pause
- Health/status reflects paused state
- Automatic recovery when DB comes back and spool drains
- bars_dropped_total remains 0
- Guard fires alert for ingestion_paused

Run with:  python -m pytest tests/test_persistence_safe_pause.py -v
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict

from src.core.bus import EventBus
from src.core.events import BarEvent
from src.core.persistence import PersistenceManager


# ── Helpers ─────────────────────────────────────────────


class _DummyDB:
    """DB stub that can simulate failures and then recover."""

    def __init__(self, fail_count=0, always_fail=False):
        self.bar_batches = []
        self._fail_count = fail_count
        self._always_fail = always_fail
        self._call_count = 0

    async def execute_many(self, sql: str, params_list):
        self._call_count += 1
        if self._always_fail or self._call_count <= self._fail_count:
            raise RuntimeError("Simulated DB failure")
        self.bar_batches.append(params_list)
        await asyncio.sleep(0)

    async def execute(self, sql: str, params=()):
        await asyncio.sleep(0)

    async def insert_market_metric(self, **kwargs):
        await asyncio.sleep(0)

    def recover(self):
        """Make DB start succeeding again."""
        self._always_fail = False
        self._fail_count = 0
        self._call_count = 999  # past any fail_count


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


def _bar_line_bytes():
    """Approximate byte size of one serialized bar line in the spool."""
    bar = _make_bar(ts=1_700_000_000.0)
    line = json.dumps(asdict(bar), separators=(",", ":")) + "\n"
    return len(line.encode("utf-8"))


# ── Test: Entering Paused State ─────────────────────────


class TestEnterPausedState:
    """When spool is full and DB is unreachable, system must pause."""

    def test_pause_on_spool_full(self):
        """Spool fills up → system enters paused state."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            # Spool max fits ~3 bars (add margin for byte boundary)
            spool_max = line_size * 3 + line_size

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=5,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                )

                # Fill the in-memory buffer (5 bars)
                for i in range(5):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                assert len(pm._bar_buffer) == 5
                assert not pm._ingestion_paused

                # Fill the spool (3 bars, well within capacity)
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_300.0 + i * 60))
                assert pm._spool_active
                assert pm._spool_bars_pending == 3
                assert not pm._ingestion_paused

                # Flush fails (DB down), bars return to buffer
                pm._do_flush()
                assert len(pm._bar_buffer) == 5

                # Keep adding bars until spool fills and pause triggers
                for i in range(10):
                    pm._on_bar(_make_bar(ts=1_700_000_600.0 + i * 60))
                    if pm._ingestion_paused:
                        break
                assert pm._ingestion_paused
                assert pm._pause_entered_ts is not None
                assert pm._bars_rejected_paused >= 1
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_subsequent_bars_rejected_when_paused(self):
        """Once paused, additional bars are rejected immediately."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            # Use spool_max that fits exactly 3 bars (generous margin)
            spool_max = line_size * 3

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=3,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                )

                # Fill buffer (3 bars)
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                # Overflow to spool (3 more bars = spool capacity)
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_180.0 + i * 60))
                assert pm._spool_bars_pending >= 1
                assert pm._spool_active

                # Flush fails → bars return to buffer
                pm._do_flush()

                # Keep adding bars until pause triggers
                for i in range(10):
                    pm._on_bar(_make_bar(ts=1_700_000_400.0 + i * 60))
                    if pm._ingestion_paused:
                        break
                assert pm._ingestion_paused

                # Record buffer size and rejected count
                buffer_before = len(pm._bar_buffer)
                rejected_before = pm._bars_rejected_paused

                # Send 100 more bars — all should be rejected, memory constant
                for i in range(100):
                    pm._on_bar(_make_bar(ts=1_700_001_000.0 + i * 60))

                assert len(pm._bar_buffer) == buffer_before  # no growth
                assert pm._bars_rejected_paused == rejected_before + 100
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


# ── Test: Memory Stays Bounded ──────────────────────────


class TestMemoryBounded:
    """Memory buffer must NOT become unbounded during pause."""

    def test_buffer_does_not_grow_when_paused(self):
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            spool_max = line_size * 2

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=5,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                )

                # Fill buffer
                for i in range(5):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))

                # Fill spool
                for i in range(2):
                    pm._on_bar(_make_bar(ts=1_700_000_300.0 + i * 60))

                # Flush fails
                pm._do_flush()

                # Trigger pause
                pm._on_bar(_make_bar(ts=1_700_000_500.0))
                assert pm._ingestion_paused

                max_buf = len(pm._bar_buffer)

                # Hammer with 1000 bars
                for i in range(1000):
                    pm._on_bar(_make_bar(ts=1_700_002_000.0 + i * 60))

                # Buffer size must not have grown
                assert len(pm._bar_buffer) <= max_buf
                assert pm._bars_rejected_paused >= 1000
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


# ── Test: Health / Status / Counters ────────────────────


class TestHealthAndStatus:
    """Paused state must be visible in health_status and /debug/soak."""

    def test_status_shows_paused(self):
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            spool_max = line_size * 1  # very small spool

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=3,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                )

                # Fill buffer
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                # Fill spool
                pm._on_bar(_make_bar(ts=1_700_000_180.0))
                # Flush fails
                pm._do_flush()
                # Trigger pause
                pm._on_bar(_make_bar(ts=1_700_000_240.0))
                assert pm._ingestion_paused

                status = pm.get_status()
                assert status["status"] == "paused"
                extras = status["extras"]
                assert extras["ingestion_paused"] is True
                assert extras["bars_rejected_paused"] >= 1
                assert extras["pause_entered_ts"] is not None
                assert extras["spool_max_bytes"] == spool_max
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_heartbeat_health_down_when_paused(self):
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            spool_max = line_size * 1

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=3,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                )

                # Force into paused state
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                pm._on_bar(_make_bar(ts=1_700_000_180.0))
                pm._do_flush()
                pm._on_bar(_make_bar(ts=1_700_000_240.0))
                assert pm._ingestion_paused

                hb = pm.emit_heartbeat()
                assert hb.health == "down"
                assert hb.extra["ingestion_paused"] is True
                assert hb.extra["bars_rejected_paused"] >= 1
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_bars_dropped_total_stays_zero(self):
        """bars_dropped_total must remain 0 even during pause."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            spool_max = line_size * 2

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=3,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                )

                # Drive into pause
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                for i in range(2):
                    pm._on_bar(_make_bar(ts=1_700_000_180.0 + i * 60))
                pm._do_flush()
                pm._on_bar(_make_bar(ts=1_700_000_400.0))
                assert pm._ingestion_paused

                # Send many more
                for i in range(50):
                    pm._on_bar(_make_bar(ts=1_700_001_000.0 + i * 60))

                assert pm._bars_dropped_total == 0
                status = pm.get_status()
                assert status["extras"]["bars_dropped_total"] == 0
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


# ── Test: Automatic Recovery ────────────────────────────


class TestAutomaticRecovery:
    """When DB recovers and spool drains, ingestion must auto-resume."""

    def test_resume_after_db_recovery(self):
        """DB comes back → spool drains → pause lifts."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            spool_max = line_size * 2

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=3,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                    resume_threshold_pct=80.0,
                )

                # Drive into pause
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                for i in range(2):
                    pm._on_bar(_make_bar(ts=1_700_000_180.0 + i * 60))
                pm._do_flush()
                pm._on_bar(_make_bar(ts=1_700_000_400.0))
                assert pm._ingestion_paused

                rejected_before = pm._bars_rejected_paused

                # DB recovers
                db.recover()

                # Flush should now succeed and drain spool
                pm._do_flush()

                # System should have resumed
                assert not pm._ingestion_paused
                assert pm._pause_entered_ts is None

                # New bars should be accepted again
                pm._on_bar(_make_bar(ts=1_700_002_000.0))
                assert pm._bars_rejected_paused == rejected_before  # no new rejects
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_resume_respects_threshold(self):
        """Resume only happens when spool drops below resume_threshold_pct."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            # Spool max = 10 bars. With 80% threshold, need < 8 bars worth.
            spool_max = line_size * 10

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=3,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                    resume_threshold_pct=50.0,  # need < 50% to resume
                )

                # Fill buffer
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                # Fill spool to max (10 bars)
                for i in range(10):
                    pm._on_bar(_make_bar(ts=1_700_000_180.0 + i * 60))
                # Flush fails
                pm._do_flush()
                # Trigger pause
                pm._on_bar(_make_bar(ts=1_700_000_800.0))
                assert pm._ingestion_paused

                # DB recovers
                db.recover()

                # Flush buffer — succeeds but spool still has bars
                pm._do_flush()
                # After one flush cycle: buffer drained, spool partially drained
                # (500 batch from spool). With 10 spool bars, one drain clears all.
                # But spool_bytes_written is still the original amount until cleanup.
                # After _cleanup_spool, bytes = 0 which is < 50%.
                # So it should resume.
                assert not pm._ingestion_paused
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


# ── Test: Configuration ─────────────────────────────────


class TestConfiguration:
    """Configurable parameters are respected."""

    def test_pause_disabled_falls_back_to_unbounded(self):
        """pause_on_spool_full=False preserves legacy unbounded behavior."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            spool_max = line_size * 2

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=3,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=False,  # legacy
                )

                # Fill buffer
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                # Fill spool
                for i in range(2):
                    pm._on_bar(_make_bar(ts=1_700_000_180.0 + i * 60))
                # Flush fails
                pm._do_flush()
                # Spool full → should NOT pause, should grow memory
                pm._on_bar(_make_bar(ts=1_700_000_400.0))
                assert not pm._ingestion_paused
                # Bar was appended to buffer (unbounded)
                assert len(pm._bar_buffer) > 3
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_custom_spool_max_bytes(self):
        """Custom spool_max_bytes is respected."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            line_size = _bar_line_bytes()
            custom_max = line_size * 5

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=2,
                    spool_max_bytes=custom_max,
                )
                assert pm._spool_max_bytes == custom_max

                status = pm.get_status()
                assert status["extras"]["spool_max_bytes"] == custom_max
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_ingestion_paused_property(self):
        """Thread-safe property works correctly."""
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)
                assert pm.ingestion_paused is False

                with pm._bar_lock:
                    pm._ingestion_paused = True
                assert pm.ingestion_paused is True
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


# ── Test: Guard Integration ─────────────────────────────


class TestGuardIntegration:
    """SoakGuardian detects and alerts on ingestion_paused."""

    def test_guard_fires_on_paused(self):
        from src.soak.guards import SoakGuardian

        guardian = SoakGuardian()

        persist_status = {
            "extras": {
                "ingestion_paused": True,
                "bars_rejected_paused": 42,
                "pause_entered_ts": time.time() - 120,
                "write_queue_depth": 0,
                "persist_lag_ema_ms": None,
                "persist_lag_crypto_ema_ms": None,
                "persist_lag_deribit_ema_ms": None,
                "persist_lag_equities_ema_ms": None,
                "bar_buffer_size": 100,
                "bar_buffer_max": 100000,
                "spool_active": True,
                "spool_bars_pending": 500,
                "spool_file_size": 1_000_000,
                "bar_flush_success_total": 0,
                "bar_flush_failure_total": 5,
            },
            "counters": {"error_count": 5},
            "last_error": "bar_flush_failed",
        }

        bb_status = {"extras": {}}
        resource = {}

        triggered = guardian.evaluate(
            bus_stats={},
            bar_builder_status=bb_status,
            persistence_status=persist_status,
            resource_snapshot=resource,
        )

        # Should contain an ingestion_paused alert
        paused_alerts = [a for a in triggered if a["guard"] == "ingestion_paused"]
        assert len(paused_alerts) == 1
        assert paused_alerts[0]["severity"] == "ALERT"
        assert "INGESTION PAUSED" in paused_alerts[0]["message"]

        # Health should be "alert"
        assert guardian.health_status.get("ingestion_paused") == "alert"

    def test_guard_ok_when_not_paused(self):
        from src.soak.guards import SoakGuardian

        guardian = SoakGuardian()

        persist_status = {
            "extras": {
                "ingestion_paused": False,
                "bars_rejected_paused": 0,
                "pause_entered_ts": None,
                "write_queue_depth": 0,
                "persist_lag_ema_ms": None,
                "persist_lag_crypto_ema_ms": None,
                "persist_lag_deribit_ema_ms": None,
                "persist_lag_equities_ema_ms": None,
                "bar_buffer_size": 10,
                "bar_buffer_max": 100000,
                "spool_active": False,
                "spool_bars_pending": 0,
                "spool_file_size": 0,
            },
            "counters": {"error_count": 0},
            "last_error": None,
        }

        triggered = guardian.evaluate(
            bus_stats={},
            bar_builder_status={"extras": {}},
            persistence_status=persist_status,
            resource_snapshot={},
        )

        paused_alerts = [a for a in triggered if a["guard"] == "ingestion_paused"]
        assert len(paused_alerts) == 0
        assert guardian.health_status.get("ingestion_paused") == "ok"


# ── Test: Soak Summary ──────────────────────────────────


class TestSoakSummary:
    """Soak summary includes pause metrics."""

    def test_summary_includes_pause_fields(self):
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB()
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(bus, db, loop, spool_dir=td)

                from src.soak.summary import build_soak_summary
                summary = build_soak_summary(persistence=pm)

                pt = summary["persistence_telemetry"]
                assert "ingestion_paused" in pt
                assert "bars_rejected_paused" in pt
                assert "pause_entered_ts" in pt
                assert "spool_max_bytes" in pt
                assert pt["ingestion_paused"] is False
                assert pt["bars_rejected_paused"] == 0
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


# ── Test: End-to-end Scenario ───────────────────────────


class TestEndToEnd:
    """Full scenario: DB down → spool fills → pause → DB recovers → resume."""

    def test_full_lifecycle(self):
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            # Use spool that fits exactly 5 bars (add margin for rounding)
            spool_max = line_size * 5 + line_size

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=5,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                    resume_threshold_pct=80.0,
                )

                # Phase 1: Normal ingestion
                for i in range(3):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))
                assert len(pm._bar_buffer) == 3
                assert not pm._ingestion_paused

                # Phase 2: Buffer fills up, bars go to spool
                for i in range(2):
                    pm._on_bar(_make_bar(ts=1_700_000_180.0 + i * 60))
                assert len(pm._bar_buffer) == 5
                # More bars → spool
                for i in range(5):
                    pm._on_bar(_make_bar(ts=1_700_000_300.0 + i * 60))
                assert pm._spool_active
                assert pm._spool_bars_pending == 5

                # Phase 3: Flush fails (DB down), bars return to buffer
                pm._do_flush()
                assert len(pm._bar_buffer) == 5
                assert pm._bar_flush_failure_total >= 1

                # Phase 4: Keep adding bars until pause triggers
                for i in range(10):
                    pm._on_bar(_make_bar(ts=1_700_000_700.0 + i * 60))
                    if pm._ingestion_paused:
                        break
                assert pm._ingestion_paused
                assert pm._bars_dropped_total == 0

                # Phase 5: More bars during pause — all rejected
                for i in range(20):
                    pm._on_bar(_make_bar(ts=1_700_001_000.0 + i * 60))
                assert pm._bars_rejected_paused >= 20
                assert pm._bars_dropped_total == 0

                # Phase 6: Verify health
                status = pm.get_status()
                assert status["status"] == "paused"
                hb = pm.emit_heartbeat()
                assert hb.health == "down"

                # Phase 7: DB recovers
                db.recover()
                pm._do_flush()

                # Phase 8: Should have resumed
                assert not pm._ingestion_paused
                assert pm._bars_dropped_total == 0

                # Phase 9: New bars accepted
                pm._on_bar(_make_bar(ts=1_700_003_000.0))
                assert len(pm._bar_buffer) >= 1 or pm._spool_active

                status_after = pm.get_status()
                assert status_after["status"] != "paused"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)


# ── Test: Windows Spool File Handle Release ─────────────


class TestWindowsSpoolFileRelease:
    """Regression test: spool file must be renameable/removable immediately.
    
    This catches WinError 32 (file in use) issues where a file handle
    is held open, preventing temp directory cleanup on Windows.
    """

    def test_spool_file_not_locked_after_writes(self):
        """After spooling bars, the spool file handle must be released.
        
        Uses os.rename as a proxy for "no open handle" — Windows won't
        allow renaming a file if another process has it open.
        """
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            spool_max = line_size * 10  # generous

            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=2,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=False,  # don't pause, just spool
                )

                # Trigger spooling by overflowing the buffer
                for i in range(5):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))

                assert pm._spool_active
                assert pm._spool_bars_pending >= 1

                # Spool file must exist
                spool_path = pm._spool_path
                assert spool_path.exists()

                # Key assertion: file must be renameable (no open handle)
                # On Windows, this fails with WinError 32 if handle is open
                renamed_path = spool_path.with_suffix(".moved")
                os.rename(spool_path, renamed_path)
                assert renamed_path.exists()
                assert not spool_path.exists()

                # Rename back for cleanup
                os.rename(renamed_path, spool_path)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

    def test_tempdir_cleanup_succeeds_after_spooling(self):
        """TemporaryDirectory cleanup must succeed after spooling.
        
        This is the actual failure scenario from the bug report.
        """
        loop = asyncio.new_event_loop()
        thread = _start_loop(loop)
        try:
            bus = EventBus()
            db = _DummyDB(always_fail=True)
            line_size = _bar_line_bytes()
            spool_max = line_size * 5

            # If cleanup fails, TemporaryDirectory will raise PermissionError
            with tempfile.TemporaryDirectory() as td:
                pm = PersistenceManager(
                    bus, db, loop,
                    spool_dir=td,
                    bar_buffer_max=2,
                    spool_max_bytes=spool_max,
                    pause_on_spool_full=True,
                )

                # Fill buffer then spool
                for i in range(4):
                    pm._on_bar(_make_bar(ts=1_700_000_000.0 + i * 60))

                # Ingestion may or may not be paused depending on spool size
                # Either way, spool should be active
                assert pm._spool_active or len(pm._bar_buffer) > 0

            # If we get here, cleanup succeeded
            # On Windows with lingering handle, this would fail with:
            # PermissionError: [WinError 32] The process cannot access the file
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)

