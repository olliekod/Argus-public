"""
Argus Persistence Module
========================

Subscribes to bus topics and writes data to the database.

Priority rules
--------------
* **Market bars** (HIGH priority) are batched and flushed every 1 second
  (or on heartbeat / shutdown).  Bars are **never dropped**: on flush
  failure they are returned to the buffer for retry.
* **SignalEvents** (MEDIUM priority) are persisted via a dedicated queue.
* **Metrics / Heartbeats** (LOW priority) share a separate queue and
  are the first to be dropped under overload.

Flush triggers
--------------
1. Periodic 1-second timer inside the bar-batch writer.
2. ``system.heartbeat`` events (flush on heartbeat boundaries).
3. ``SIGINT / Ctrl-C`` — the orchestrator calls :meth:`shutdown` which
   flushes all remaining buffered bars.

Storage optimisation
--------------------
Only 1-minute bars are logged.

Lag tracking (Stream 2.1)
-------------------------
Computes ``persist_lag_ms = now - source_ts`` for every bar flush and
exposes it via status / heartbeat telemetry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional

from .bus import EventBus
from .bar_builder import _ts_sane
from .events import (
    BarEvent,
    ComponentHeartbeatEvent,
    HeartbeatEvent,
    MetricEvent,
    SignalEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_METRICS,
    TOPIC_OPTIONS_CHAINS,
    TOPIC_REGIMES_SYMBOL,
    TOPIC_REGIMES_MARKET,
    TOPIC_SIGNALS,
    TOPIC_SIGNALS_RAW,
    TOPIC_SYSTEM_HEARTBEAT,
    TOPIC_SYSTEM_COMPONENT_HEARTBEAT,
)
from .liquid_etf_universe import LIQUID_ETF_UNIVERSE
from .option_events import OptionChainSnapshotEvent, option_chain_to_dict
from .regimes import SymbolRegimeEvent, MarketRegimeEvent
from .signals import SignalEvent as Phase3SignalEvent, normalize_snapshot

logger = logging.getLogger("argus.persistence")

# How often the bar-batch writer flushes (seconds)
_FLUSH_INTERVAL = 1.0
_SIGNAL_QUEUE_MAXSIZE = 5_000     # Medium priority (signals)
_TELEMETRY_QUEUE_MAXSIZE = 5_000  # Low priority (metrics / heartbeats)
_SIGNAL_PUT_TIMEOUT_S = 0.5      # Block up to 0.5s before dropping signal
_BAR_FLUSH_MAX_RETRIES = 3        # Retry failed bar flushes
_WRITE_STOP = object()

# ── Bar buffer / spool limits ────────────────────────────────
_BAR_BUFFER_MAX = 100_000         # Max bars held in memory
_SPOOL_MAX_BYTES = 1 << 30        # 1 GB max spool file size
_SPOOL_DRAIN_BATCH = 500          # Bars to drain from spool per flush cycle
_PAUSE_ON_SPOOL_FULL = True       # Pause ingestion when spool full (safe failure)
_RESUME_THRESHOLD_PCT = 80.0      # Resume when spool below this % of max
_PAUSE_LOG_INTERVAL = 60.0        # Rate-limit pause log messages (seconds)
_PERSIST_LAG_MAX_AGE_SECONDS = 300.0  # Ignore bar timestamps older than this
_PERSIST_LAG_MAX_FUTURE_SKEW_SECONDS = 2.0  # Clamp if timestamp too far in future
_SOURCE_TS_LOG_INTERVAL_SECONDS = 60.0
_SOURCE_TS_SYMBOL_MAX = 200
_SOURCE_TS_LOG_SYMBOL_MAX = 200

_CRYPTO_SOURCES = {"bybit", "binance", "coinbase", "okx"}
_DERIBIT_SOURCES = {"deribit"}
_EQUITIES_SOURCES = {"yahoo", "alpaca"}

_CRYPTO_LAG_CLASS = "crypto"
_DERIBIT_LAG_CLASS = "deribit"
_EQUITIES_LAG_CLASS = "equities"


def _normalize_source(source: Optional[str]) -> Optional[str]:
    if not source:
        return None
    return str(source).strip().lower()


def _infer_source_class_from_symbol(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    symbol_upper = symbol.upper()
    if symbol_upper.endswith("-INDEX") or symbol_upper.endswith("INDEX"):
        return _DERIBIT_LAG_CLASS
    if symbol_upper in (set(LIQUID_ETF_UNIVERSE) | {"IBIT", "BITO", "NVDA"}):
        return _EQUITIES_LAG_CLASS
    crypto_markers = ("USDT", "USDC", "PERP")
    if any(marker in symbol_upper for marker in crypto_markers) or symbol_upper.endswith("USD"):
        return _CRYPTO_LAG_CLASS
    return None


def _classify_bar_source(bar: BarEvent) -> Optional[str]:
    source_hint = None
    for attr in ("connector", "provider", "source"):
        candidate = getattr(bar, attr, None)
        if candidate:
            source_hint = candidate
            break
    source_hint = _normalize_source(source_hint)
    if source_hint in _CRYPTO_SOURCES:
        return _CRYPTO_LAG_CLASS
    if source_hint in _DERIBIT_SOURCES:
        return _DERIBIT_LAG_CLASS
    if source_hint in _EQUITIES_SOURCES:
        return _EQUITIES_LAG_CLASS
    return _infer_source_class_from_symbol(getattr(bar, "symbol", None))


def _select_source_ts(bar: BarEvent) -> Optional[float]:
    last_source_ts = getattr(bar, "last_source_ts", 0.0)
    first_source_ts = getattr(bar, "first_source_ts", 0.0)
    bar_source_ts = getattr(bar, "source_ts", 0.0)
    if _ts_sane(last_source_ts):
        return last_source_ts
    if _ts_sane(first_source_ts):
        return first_source_ts
    if _ts_sane(bar_source_ts):
        return bar_source_ts
    if last_source_ts > 0:
        return last_source_ts
    if first_source_ts > 0:
        return first_source_ts
    if bar_source_ts > 0:
        return bar_source_ts
    return None


def _compute_persist_lag_ms(source_ts_values: List[float], now: float) -> Optional[float]:
    if not source_ts_values:
        return None
    avg_source_ts = sum(source_ts_values) / len(source_ts_values)
    return (now - avg_source_ts) * 1000


class PersistenceManager:
    """Async-safe persistence subscriber for the event bus.

    Parameters
    ----------
    bus : EventBus
        Shared event bus.
    db : Database
        The Argus async SQLite database instance.
    loop : asyncio.AbstractEventLoop
        The running asyncio loop (needed to bridge bus worker threads
        into the async database layer).
    """

    def __init__(
        self,
        bus: EventBus,
        db: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        spool_dir: Optional[str] = None,
        bar_buffer_max: int = _BAR_BUFFER_MAX,
        spool_max_bytes: int = _SPOOL_MAX_BYTES,
        pause_on_spool_full: bool = _PAUSE_ON_SPOOL_FULL,
        resume_threshold_pct: float = _RESUME_THRESHOLD_PCT,
    ) -> None:
        self._bus = bus
        self._db = db
        self._loop = loop

        # Bar buffer — bounded list guarded by lock.
        # Bars are NEVER dropped: overflow spills to disk spool.
        self._bar_buffer: List[BarEvent] = []
        self._bar_buffer_max: int = bar_buffer_max
        self._bar_lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._signal_write_thread: Optional[threading.Thread] = None
        self._telemetry_write_thread: Optional[threading.Thread] = None

        # Disk spool for bar overflow (JSONL append-only file)
        self._spool_dir = Path(spool_dir) if spool_dir else Path("data")
        self._spool_path = self._spool_dir / "bar_spool.jsonl"
        self._spool_max_bytes: int = spool_max_bytes
        self._spool_active: bool = False
        # Note: per-append pattern used — no persistent file handle stored
        self._spool_read_offset: int = 0
        self._spool_bytes_written: int = 0
        self._bars_spooled_total: int = 0
        self._spool_write_errors: int = 0
        self._spool_bars_pending: int = 0

        # Safe-pause: pause ingestion when spool full + DB unreachable
        self._pause_on_spool_full: bool = pause_on_spool_full
        self._resume_threshold_pct: float = resume_threshold_pct
        self._ingestion_paused: bool = False
        self._pause_entered_ts: Optional[float] = None
        self._bars_rejected_paused: int = 0
        self._pause_log_ts: Optional[float] = None

        # Priority queues: signals (medium) vs metrics/heartbeats (low)
        self._signal_queue: Queue = Queue(maxsize=_SIGNAL_QUEUE_MAXSIZE)
        self._telemetry_queue: Queue = Queue(maxsize=_TELEMETRY_QUEUE_MAXSIZE)

        self._running = False
        self._status_lock = threading.Lock()
        self._last_flush_ts: Optional[float] = None
        self._last_flush_ms: Optional[float] = None
        self._flush_latency_ema: Optional[float] = None
        self._last_write_ts: Optional[float] = None
        self._db_write_errors: int = 0
        self._last_error: Optional[str] = None
        self._metrics_writes_total: int = 0
        self._bars_writes_total: int = 0
        self._signals_writes_total: int = 0
        self._consecutive_failures: int = 0
        self._start_time = time.monotonic()
        self._signals_dropped_total: int = 0
        self._metrics_dropped_total: int = 0
        self._heartbeats_dropped_total: int = 0

        # Bars must never be dropped — this counter must stay zero
        self._bars_dropped_total: int = 0
        self._bar_flush_success_total: int = 0
        self._bar_flush_failure_total: int = 0
        self._bar_retry_count: int = 0

        # Lag tracking (Stream 2.1)
        self._last_persist_lag_ms: Optional[float] = None
        self._persist_lag_ema: Optional[float] = None
        self._persist_lag_crypto_ema: Optional[float] = None
        self._persist_lag_deribit_ema: Optional[float] = None
        self._persist_lag_equities_ema: Optional[float] = None
        self._source_ts_future_clamped_total: int = 0
        self._source_ts_stale_ignored_total: int = 0
        self._source_ts_units_discarded_total: int = 0
        self._flush_count: int = 0
        self._source_ts_missing_total: int = 0
        self._source_ts_future_clamped_by_symbol: "OrderedDict[str, int]" = OrderedDict()
        self._source_ts_stale_ignored_by_symbol: "OrderedDict[str, int]" = OrderedDict()
        self._source_ts_units_discarded_by_symbol: "OrderedDict[str, int]" = OrderedDict()
        self._source_ts_missing_by_symbol: "OrderedDict[str, int]" = OrderedDict()
        self._source_ts_future_log_ts_by_symbol: "OrderedDict[str, float]" = OrderedDict()

        # Recover spool from previous crash (if any)
        self._recover_spool()

        # Subscribe to relevant topics
        bus.subscribe(TOPIC_MARKET_BARS, self._on_bar)
        bus.subscribe(TOPIC_SIGNALS, self._on_signal)
        bus.subscribe(TOPIC_SIGNALS_RAW, self._on_signal_raw)
        bus.subscribe(TOPIC_SYSTEM_HEARTBEAT, self._on_heartbeat)
        bus.subscribe(TOPIC_MARKET_METRICS, self._on_metric)
        bus.subscribe(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, self._on_component_heartbeat)
        # Phase 3B: Options chain snapshots (idempotent upsert)
        bus.subscribe(TOPIC_OPTIONS_CHAINS, self._on_option_chain)
        # Phase 4B: Regime events (symbol + market)
        bus.subscribe(TOPIC_REGIMES_SYMBOL, self._on_symbol_regime)
        bus.subscribe(TOPIC_REGIMES_MARKET, self._on_market_regime)

        logger.info("PersistenceManager initialised (bar_buffer_max=%d)", self._bar_buffer_max)

    # ── lifecycle ───────────────────────────────────────────

    def start(self) -> None:
        """Start the background flush and write threads."""
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="persistence-flush",
            daemon=True,
        )
        self._flush_thread.start()
        self._signal_write_thread = threading.Thread(
            target=self._write_loop,
            args=(self._signal_queue, "signal"),
            name="persistence-signal-writes",
            daemon=True,
        )
        self._signal_write_thread.start()
        self._telemetry_write_thread = threading.Thread(
            target=self._write_loop,
            args=(self._telemetry_queue, "telemetry"),
            name="persistence-telemetry-writes",
            daemon=True,
        )
        self._telemetry_write_thread.start()
        logger.info("PersistenceManager flush + write threads started")

    def shutdown(self) -> None:
        """Flush all remaining bars and stop the flush thread.

        Called on SIGINT / Ctrl-C from the orchestrator.
        Idempotent: safe to call multiple times.
        """
        self._running = False
        # Final flush
        self._do_flush()
        # Note: spool file handles are closed after each write (per-append pattern)
        # so no file handle cleanup needed here.
        for q in (self._signal_queue, self._telemetry_queue):
            try:
                q.put_nowait(_WRITE_STOP)
            except Exception:
                pass
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        if self._signal_write_thread and self._signal_write_thread.is_alive():
            self._signal_write_thread.join(timeout=5.0)
        if self._telemetry_write_thread and self._telemetry_write_thread.is_alive():
            self._telemetry_write_thread.join(timeout=5.0)
        logger.info("PersistenceManager shut down (all bars flushed)")

    # ── spool management ────────────────────────────────────

    def _recover_spool(self) -> None:
        """Detect and recover a leftover spool file from a previous crash."""
        try:
            if self._spool_path.exists():
                size = self._spool_path.stat().st_size
                if size > 0:
                    count = 0
                    with open(self._spool_path, "r") as f:
                        for _ in f:
                            count += 1
                    self._spool_active = True
                    self._spool_bytes_written = size
                    self._spool_bars_pending = count
                    self._bars_spooled_total = count
                    logger.warning(
                        "Recovered bar spool from previous session: "
                        "%d bars, %d bytes",
                        count, size,
                    )
                else:
                    # Empty spool file — remove it
                    self._spool_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to recover spool: %s", e)

    def _spool_bar(self, bar: BarEvent) -> None:
        """Write a single bar to the disk spool (JSONL). Called under _bar_lock.
        
        Uses per-append open/write/close pattern for Windows compatibility.
        This ensures no file handle is held open between calls, preventing
        WinError 32 (file in use) during temp directory cleanup in tests.
        """
        try:
            self._spool_dir.mkdir(parents=True, exist_ok=True)

            line = json.dumps(asdict(bar), separators=(",", ":")) + "\n"
            line_bytes = len(line.encode("utf-8"))

            if self._spool_bytes_written + line_bytes > self._spool_max_bytes:
                if self._pause_on_spool_full:
                    # SAFE PAUSE: stop accepting bars instead of OOM risk
                    self._enter_pause("spool_full")
                    self._bars_rejected_paused += 1
                    return
                # Legacy fallback (pause_on_spool_full=False): unbounded memory
                self._bar_buffer.append(bar)
                logger.critical(
                    "Bar spool FULL (%d bytes, %d bars pending). "
                    "Bar kept in memory (buffer=%d). "
                    "DB recovery urgently needed!",
                    self._spool_bytes_written,
                    self._spool_bars_pending,
                    len(self._bar_buffer),
                )
                return

            # Per-append open/write/close — Windows-safe (no lingering handle)
            with open(self._spool_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

            self._spool_bytes_written += line_bytes
            self._spool_bars_pending += 1
            self._bars_spooled_total += 1
            self._spool_active = True
        except OSError as e:
            self._spool_write_errors += 1
            if self._pause_on_spool_full and self._spool_write_errors >= 3:
                # Repeated disk failures — pause to prevent OOM
                self._enter_pause("spool_write_error")
                self._bars_rejected_paused += 1
                return
            # Isolated disk write failure — fall back to memory
            self._bar_buffer.append(bar)
            logger.error(
                "Failed to spool bar to disk: %s — kept in memory (buffer=%d)",
                e, len(self._bar_buffer),
            )

    def _drain_spool_batch(self) -> List[BarEvent]:
        """Read a batch of bars from the spool file. Called under _bar_lock."""
        if not self._spool_active or self._spool_bars_pending <= 0:
            return []

        bars: List[BarEvent] = []
        new_offset = self._spool_read_offset
        try:
            with open(self._spool_path, "r") as f:
                f.seek(self._spool_read_offset)
                for _ in range(_SPOOL_DRAIN_BATCH):
                    line = f.readline()
                    if not line or not line.strip():
                        break
                    data = json.loads(line)
                    bars.append(BarEvent(**data))
                new_offset = f.tell()
        except json.JSONDecodeError as e:
            # Advance past the corrupt line to prevent infinite loop
            logger.error("Corrupt line in bar spool at offset %d, skipping: %s",
                        self._spool_read_offset, e)
            try:
                with open(self._spool_path, "r") as f:
                    f.seek(self._spool_read_offset)
                    f.readline()  # Skip the corrupt line
                    new_offset = f.tell()
                self._spool_read_offset = new_offset
                self._spool_bars_pending = max(0, self._spool_bars_pending - 1)
            except OSError:
                pass
        except (OSError, TypeError) as e:
            logger.error("Failed to read bar spool: %s", e)

        if bars:
            self._spool_read_offset = new_offset
            self._spool_bars_pending -= len(bars)

            if self._spool_bars_pending <= 0:
                self._cleanup_spool()

        return bars

    def _drain_spool_to_db(self) -> None:
        """Attempt to drain one batch from the disk spool to the database."""
        with self._bar_lock:
            spool_batch = self._drain_spool_batch()

        if not spool_batch:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._write_bars(spool_batch), self._loop
            )
            future.result(timeout=30.0)
            logger.info(
                "Drained %d bars from spool to DB (%d still pending)",
                len(spool_batch), self._spool_bars_pending,
            )
            # Successful drain may allow pause recovery
            if self._ingestion_paused:
                self._check_pause_recovery()
        except Exception:
            logger.warning(
                "Failed to drain spool batch (%d bars) — returned to buffer",
                len(spool_batch),
            )
            with self._bar_lock:
                self._bar_buffer = spool_batch + self._bar_buffer

    def _cleanup_spool(self) -> None:
        """Remove the spool file after full drain. Called under _bar_lock.
        
        Note: spool file handles are closed after each write (per-append pattern)
        so no file handle cleanup needed here.
        """
        try:
            if self._spool_path.exists():
                self._spool_path.unlink()
        except OSError as e:
            logger.warning("Failed to cleanup spool file: %s", e)

        self._spool_active = False
        self._spool_read_offset = 0
        self._spool_bytes_written = 0
        self._spool_bars_pending = 0
        logger.info("Bar spool fully drained and cleaned up")

    # ── safe-pause management ─────────────────────────────

    def _enter_pause(self, reason: str) -> None:
        """Enter the ingestion-paused state.  Called under _bar_lock."""
        if not self._ingestion_paused:
            self._ingestion_paused = True
            self._pause_entered_ts = time.monotonic()
            self._pause_log_ts = None  # allow immediate first log
            logger.critical(
                "INGESTION PAUSED (%s): spool=%d bytes, %d bars pending, "
                "buffer=%d/%d. DB recovery urgently needed!",
                reason,
                self._spool_bytes_written,
                self._spool_bars_pending,
                len(self._bar_buffer),
                self._bar_buffer_max,
            )

    def _check_pause_recovery(self) -> None:
        """Resume ingestion if spool has drained below the resume threshold.

        Called after a successful DB write (buffer flush or spool drain).
        Must NOT hold _bar_lock when called.
        """
        with self._bar_lock:
            if not self._ingestion_paused:
                return

            threshold_bytes = self._spool_max_bytes * (
                self._resume_threshold_pct / 100.0
            )
            spool_ok = self._spool_bytes_written < threshold_bytes
            buffer_ok = len(self._bar_buffer) < self._bar_buffer_max

            if spool_ok and buffer_ok:
                pause_duration = time.monotonic() - (
                    self._pause_entered_ts or time.monotonic()
                )
                rejected = self._bars_rejected_paused
                self._ingestion_paused = False
                self._pause_entered_ts = None
                logger.warning(
                    "INGESTION RESUMED: spool at %.1f%% (threshold=%.0f%%), "
                    "buffer=%d/%d. Paused for %.1fs, "
                    "%d bars rejected during pause.",
                    (self._spool_bytes_written / max(1, self._spool_max_bytes))
                    * 100,
                    self._resume_threshold_pct,
                    len(self._bar_buffer),
                    self._bar_buffer_max,
                    pause_duration,
                    rejected,
                )

    def _increment_symbol_counter(
        self, store: "OrderedDict[str, int]", symbol: str
    ) -> None:
        if not symbol:
            symbol = "unknown"
        if symbol in store:
            store[symbol] += 1
            store.move_to_end(symbol)
            return
        if len(store) >= _SOURCE_TS_SYMBOL_MAX:
            store.popitem(last=False)
        store[symbol] = 1

    def _record_future_log_ts(self, symbol: str, now: float) -> None:
        if not symbol:
            symbol = "unknown"
        store = self._source_ts_future_log_ts_by_symbol
        if symbol in store:
            store[symbol] = now
            store.move_to_end(symbol)
            return
        if len(store) >= _SOURCE_TS_LOG_SYMBOL_MAX:
            store.popitem(last=False)
        store[symbol] = now

    def _extract_source_ts_for_lag(
        self, bar: BarEvent, now: float
    ) -> Optional[float]:
        symbol = getattr(bar, "symbol", "unknown")
        source_ts = _select_source_ts(bar)
        if source_ts is None or source_ts <= 0:
            with self._status_lock:
                self._source_ts_missing_total += 1
                self._increment_symbol_counter(
                    self._source_ts_missing_by_symbol, symbol
                )
            return None
        if not _ts_sane(source_ts):
            with self._status_lock:
                self._source_ts_units_discarded_total += 1
                self._increment_symbol_counter(
                    self._source_ts_units_discarded_by_symbol, symbol
                )
            logger.warning(
                "Ignoring bar source_ts with units mismatch: %r (symbol=%s)",
                source_ts,
                symbol,
            )
            return None
        if source_ts > (now + _PERSIST_LAG_MAX_FUTURE_SKEW_SECONDS):
            with self._status_lock:
                self._source_ts_future_clamped_total += 1
                self._increment_symbol_counter(
                    self._source_ts_future_clamped_by_symbol, symbol
                )
                last_log = self._source_ts_future_log_ts_by_symbol.get(symbol, 0.0)
                if (now - last_log) >= _SOURCE_TS_LOG_INTERVAL_SECONDS:
                    self._record_future_log_ts(symbol, now)
                    logger.info(
                        "Clamped future bar source_ts for %s: %.3f > now+%.1fs",
                        symbol,
                        source_ts,
                        _PERSIST_LAG_MAX_FUTURE_SKEW_SECONDS,
                    )
            return now

        age_seconds = now - source_ts
        if age_seconds < 0:
            with self._status_lock:
                self._source_ts_future_clamped_total += 1
                self._increment_symbol_counter(
                    self._source_ts_future_clamped_by_symbol, symbol
                )
            return now
        if age_seconds > _PERSIST_LAG_MAX_AGE_SECONDS:
            with self._status_lock:
                self._source_ts_stale_ignored_total += 1
                self._increment_symbol_counter(
                    self._source_ts_stale_ignored_by_symbol, symbol
                )
            return None

        return source_ts

    @property
    def ingestion_paused(self) -> bool:
        """Whether bar ingestion is currently paused (thread-safe read)."""
        with self._bar_lock:
            return self._ingestion_paused

    # ── handlers (run on bus worker threads) ────────────────

    def _on_bar(self, event: BarEvent) -> None:
        """Buffer bar for batched write. Bounded with disk spool overflow.

        When ingestion is paused (spool full + DB unreachable), bars are
        rejected at this boundary to keep memory bounded.  The paused state
        is highly visible in health/status and triggers alerts.
        """
        with self._bar_lock:
            if self._ingestion_paused:
                self._bars_rejected_paused += 1
                now = time.time()
                if (self._pause_log_ts is None
                        or now - self._pause_log_ts >= _PAUSE_LOG_INTERVAL):
                    self._pause_log_ts = now
                    logger.critical(
                        "INGESTION PAUSED: bar rejected (total rejected=%d). "
                        "Spool full + DB unreachable. "
                        "Waiting for DB recovery.",
                        self._bars_rejected_paused,
                    )
                return
            if self._spool_active or len(self._bar_buffer) >= self._bar_buffer_max:
                self._spool_bar(event)
            else:
                self._bar_buffer.append(event)

    def _on_signal(self, event: SignalEvent) -> None:
        """Persist signal via medium-priority queue.

        Uses a blocking put with a short timeout so that transient queue
        pressure doesn't silently discard signals.  The signal is only
        dropped after the timeout expires.
        """
        try:
            self._signal_queue.put(("signal", event), timeout=_SIGNAL_PUT_TIMEOUT_S)
        except Full:
            with self._status_lock:
                self._signals_dropped_total += 1
            logger.warning(
                "Signal queue full after %.1fs backpressure — dropping signal event "
                "(total dropped=%d)",
                _SIGNAL_PUT_TIMEOUT_S,
                self._signals_dropped_total,
            )

    def _on_signal_raw(self, event: Phase3SignalEvent) -> None:
        """Persist Phase 3 signals via medium-priority queue.

        Same backpressure strategy as ``_on_signal``.
        """
        try:
            self._signal_queue.put(("signal_raw", event), timeout=_SIGNAL_PUT_TIMEOUT_S)
        except Full:
            with self._status_lock:
                self._signals_dropped_total += 1
            logger.warning(
                "Signal queue full after %.1fs backpressure — dropping raw signal event "
                "(total dropped=%d)",
                _SIGNAL_PUT_TIMEOUT_S,
                self._signals_dropped_total,
            )

    def _on_heartbeat(self, event: HeartbeatEvent) -> None:
        """Flush buffered bars on heartbeat boundary."""
        self._do_flush()

    def _on_metric(self, event: MetricEvent) -> None:
        """Persist market metric via low-priority queue."""
        try:
            self._telemetry_queue.put_nowait(("metric", event))
        except Full:
            with self._status_lock:
                self._metrics_dropped_total += 1
            logger.debug("Telemetry queue full — dropping metric event")

    def _on_component_heartbeat(self, event: ComponentHeartbeatEvent) -> None:
        """Persist component heartbeat via low-priority queue."""
        try:
            self._telemetry_queue.put_nowait(("heartbeat", event))
        except Full:
            with self._status_lock:
                self._heartbeats_dropped_total += 1
            logger.debug("Telemetry queue full — dropping heartbeat event")

    def _on_option_chain(self, event: OptionChainSnapshotEvent) -> None:
        """Persist option chain snapshot via idempotent upsert.

        Writes directly to DB (async-safe via futures).
        Uses upsert_option_chain_snapshot for restart idempotency.
        Retries up to 3 times with exponential backoff on failure.
        """
        if not self._running:
            return

        # Serialize quotes to JSON for storage
        import json
        quotes_json = json.dumps(option_chain_to_dict(event), sort_keys=True)

        # Fill atm_iv from quotes when provider did not supply it (e.g. Tastytrade early snapshots)
        atm_iv = event.atm_iv
        if (atm_iv is None or (isinstance(atm_iv, (int, float)) and atm_iv <= 0)) and quotes_json:
            from src.tools.replay_pack import _atm_iv_from_quotes_json
            derived = _atm_iv_from_quotes_json(quotes_json, event.underlying_price or 0)
            if derived is not None and derived > 0:
                atm_iv = derived

        max_retries = 3
        for attempt in range(max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._db.upsert_option_chain_snapshot(
                        snapshot_id=event.snapshot_id,
                        symbol=event.symbol,
                        expiration_ms=event.expiration_ms,
                        underlying_price=event.underlying_price,
                        n_strikes=event.n_strikes,
                        atm_iv=atm_iv,
                        timestamp_ms=event.timestamp_ms,
                        source_ts_ms=event.source_ts_ms,
                        recv_ts_ms=event.recv_ts_ms,
                        provider=event.provider,
                        quotes_json=quotes_json,
                    ),
                    self._loop,
                )
                future.result(timeout=15.0)
                return  # success
            except Exception:
                if attempt < max_retries - 1:
                    backoff = 0.5 * (2 ** attempt)
                    logger.warning(
                        "Option chain write attempt %d/%d failed for %s; "
                        "retrying in %.1fs",
                        attempt + 1, max_retries, event.symbol, backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "Option chain write failed after %d retries for %s",
                        max_retries, event.symbol,
                    )

    def _on_symbol_regime(self, event: SymbolRegimeEvent) -> None:
        """Persist a symbol regime event via fire-and-forget async write.

        Extracts all fields from the event — including the new liquidity
        fields (``liquidity_regime``, ``spread_pct``, ``volume_pctile``) —
        and packs remaining numeric metrics into ``metrics_json``.
        """
        if not self._running:
            return
        metrics = {
            "atr": event.atr,
            "atr_pct": event.atr_pct,
            "vol_z": event.vol_z,
            "ema_fast": event.ema_fast,
            "ema_slow": event.ema_slow,
            "ema_slope": event.ema_slope,
            "rsi": event.rsi,
            "trend_accel": event.trend_accel,
        }
        future = asyncio.run_coroutine_threadsafe(
            self._db.write_regime(
                event_type="symbol",
                scope=event.symbol,
                timeframe=event.timeframe,
                timestamp_ms=event.timestamp_ms,
                config_hash=event.config_hash,
                vol_regime=event.vol_regime,
                trend_regime=event.trend_regime,
                liquidity_regime=event.liquidity_regime,
                spread_pct=event.spread_pct,
                volume_pctile=event.volume_pctile,
                confidence=event.confidence,
                is_warm=event.is_warm,
                data_quality_flags=event.data_quality_flags,
                metrics_json=json.dumps(metrics, sort_keys=True),
            ),
            self._loop,
        )
        future.add_done_callback(lambda f: f.exception() if f.exception() else None)

    def _on_market_regime(self, event: MarketRegimeEvent) -> None:
        """Persist a market regime event via fire-and-forget async write."""
        if not self._running:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._db.write_regime(
                event_type="market",
                scope=event.market,
                timeframe=event.timeframe,
                timestamp_ms=event.timestamp_ms,
                config_hash=event.config_hash,
                session_regime=event.session_regime,
                risk_regime=event.risk_regime,
                confidence=event.confidence,
                data_quality_flags=event.data_quality_flags,
                metrics_json=event.metrics_json or None,
            ),
            self._loop,
        )
        future.add_done_callback(lambda f: f.exception() if f.exception() else None)

    # ── flush logic ─────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Background thread: flush bar buffer every _FLUSH_INTERVAL seconds."""
        while self._running:
            time.sleep(_FLUSH_INTERVAL)
            self._do_flush()

    def _write_loop(self, queue: Queue, label: str) -> None:
        """Background thread: serialize async DB writes for one priority queue."""
        while self._running or not queue.empty():
            try:
                item = queue.get(timeout=0.5)
            except Empty:
                continue
            if item is _WRITE_STOP:
                return
            kind, event = item
            try:
                if kind == "signal":
                    future = asyncio.run_coroutine_threadsafe(
                        self._write_signal(event), self._loop
                    )
                elif kind == "signal_raw":
                    future = asyncio.run_coroutine_threadsafe(
                        self._write_phase3_signal(event), self._loop
                    )
                elif kind == "metric":
                    future = asyncio.run_coroutine_threadsafe(
                        self._write_metric(event), self._loop
                    )
                else:
                    future = asyncio.run_coroutine_threadsafe(
                        self._write_component_heartbeat(event), self._loop
                    )
                future.result(timeout=10.0)
            except Exception:
                with self._status_lock:
                    self._db_write_errors += 1
                    self._consecutive_failures += 1
                    self._last_error = f"{kind}_write_failed"
                logger.exception("Failed to persist %s event via %s queue", kind, label)

    def _do_flush(self) -> None:
        """Drain the bar buffer and write to DB.

        On failure, bars are returned to the buffer for retry so that
        bars are **never silently dropped**.

        After a successful buffer flush, drains a batch from the disk
        spool (if active) to steadily recover spooled bars.
        """
        # Periodic WAL checkpoint (every ~5 minutes if _FLUSH_INTERVAL=1s)
        self._flush_count += 1
        if self._flush_count >= 300:
            self._flush_count = 0
            asyncio.run_coroutine_threadsafe(self._db.checkpoint(), self._loop)

        with self._bar_lock:
            if not self._bar_buffer and not self._spool_active:
                return
            batch = list(self._bar_buffer)
            self._bar_buffer.clear()

        if not batch:
            # Buffer was empty but spool is active — drain spool directly
            self._drain_spool_to_db()
            return

        start = time.perf_counter()
        success = False
        last_exc = None
        for attempt in range(_BAR_FLUSH_MAX_RETRIES):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._write_bars(batch), self._loop
                )
                future.result(timeout=30.0)
                success = True
                break
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Bar flush attempt %d/%d failed for %d bars: %s: %s",
                    attempt + 1, _BAR_FLUSH_MAX_RETRIES, len(batch),
                    type(e).__name__, e,
                    exc_info=True,
                )
                logger.debug(
                    "Bar flush failure detail: attempt=%d batch_sample=%s",
                    attempt + 1,
                    [(b.symbol, b.source, b.timestamp) for b in batch[:5]],
                )
                if attempt < _BAR_FLUSH_MAX_RETRIES - 1:
                    with self._status_lock:
                        self._bar_retry_count += 1
                    time.sleep(0.5 * (attempt + 1))  # brief backoff

        if success:
            duration_ms = (time.perf_counter() - start) * 1000
            now = time.time()

            # Compute persist_lag_ms from source timestamps
            source_ts_values: List[float] = []
            source_ts_by_class: Dict[str, List[float]] = {
                _CRYPTO_LAG_CLASS: [],
                _DERIBIT_LAG_CLASS: [],
                _EQUITIES_LAG_CLASS: [],
            }
            for b in batch:
                source_ts = self._extract_source_ts_for_lag(b, now)
                if source_ts is None:
                    continue
                source_ts_values.append(source_ts)
                source_class = _classify_bar_source(b)
                if source_class in source_ts_by_class:
                    source_ts_by_class[source_class].append(source_ts)
            persist_lag = _compute_persist_lag_ms(source_ts_values, now)
            persist_lag_crypto = _compute_persist_lag_ms(
                source_ts_by_class[_CRYPTO_LAG_CLASS], now
            )
            persist_lag_deribit = _compute_persist_lag_ms(
                source_ts_by_class[_DERIBIT_LAG_CLASS], now
            )
            persist_lag_equities = _compute_persist_lag_ms(
                source_ts_by_class[_EQUITIES_LAG_CLASS], now
            )

            with self._status_lock:
                self._last_flush_ts = now
                self._last_flush_ms = duration_ms
                self._flush_latency_ema = (
                    duration_ms if self._flush_latency_ema is None
                    else (duration_ms * 0.2) + (self._flush_latency_ema * 0.8)
                )
                self._last_write_ts = now
                self._bar_flush_success_total += 1
                self._consecutive_failures = 0
                if persist_lag is not None:
                    self._last_persist_lag_ms = persist_lag
                    self._persist_lag_ema = (
                        persist_lag if self._persist_lag_ema is None
                        else (persist_lag * 0.2) + (self._persist_lag_ema * 0.8)
                    )
                if persist_lag_crypto is not None:
                    self._persist_lag_crypto_ema = (
                        persist_lag_crypto if self._persist_lag_crypto_ema is None
                        else (persist_lag_crypto * 0.2) + (self._persist_lag_crypto_ema * 0.8)
                    )
                if persist_lag_deribit is not None:
                    self._persist_lag_deribit_ema = (
                        persist_lag_deribit if self._persist_lag_deribit_ema is None
                        else (persist_lag_deribit * 0.2) + (self._persist_lag_deribit_ema * 0.8)
                    )
                if persist_lag_equities is not None:
                    self._persist_lag_equities_ema = (
                        persist_lag_equities if self._persist_lag_equities_ema is None
                        else (persist_lag_equities * 0.2) + (self._persist_lag_equities_ema * 0.8)
                    )

            # After successful buffer flush, drain spool if active
            if self._spool_active:
                self._drain_spool_to_db()

            # Check if we can resume from pause
            if self._ingestion_paused:
                self._check_pause_recovery()
        else:
            # All retries failed — return bars to buffer (never drop)
            with self._bar_lock:
                self._bar_buffer = batch + self._bar_buffer
            with self._status_lock:
                self._db_write_errors += 1
                self._consecutive_failures += 1
                self._bar_flush_failure_total += 1
                self._last_error = "bar_flush_failed"
            exc_type = type(last_exc).__name__ if last_exc else "Unknown"
            exc_msg = str(last_exc) if last_exc else "no exception captured"
            logger.error(
                "All %d bar flush retries exhausted for %d bars — "
                "returned to buffer for next cycle. Last error: %s: %s",
                _BAR_FLUSH_MAX_RETRIES, len(batch), exc_type, exc_msg,
            )
            sample = [(b.symbol, b.source, b.timestamp) for b in batch[:10]]
            logger.debug(
                "Bar flush exhausted: batch_sample (symbol, source, timestamp)=%s",
                sample,
            )
            if last_exc is not None and getattr(last_exc, "__traceback__", None) is not None:
                import traceback as _tb
                logger.debug(
                    "Bar flush last exception traceback:\n%s",
                    "".join(_tb.format_exception(type(last_exc), last_exc, last_exc.__traceback__)),
                )

    # ── async DB writers ────────────────────────────────────

    async def _write_bars(self, bars: List[BarEvent]) -> None:
        """Batch-insert 1-minute bars into the ``market_bars`` table.
        
        Uses INSERT OR REPLACE for idempotent upsert behavior. Unique constraint
        is on (source, symbol, bar_duration, timestamp).
        """
        rows = [
            (
                datetime.fromtimestamp(b.timestamp, tz=timezone.utc).isoformat(),
                b.symbol,
                b.source,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                b.tick_count,
                getattr(b, 'n_ticks', b.tick_count),
                getattr(b, 'first_source_ts', None),
                getattr(b, 'last_source_ts', None),
                getattr(b, 'late_ticks_dropped', 0),
                getattr(b, 'close_reason', 0),
                getattr(b, 'bar_duration', 60),
            )
            for b in bars
            if b.bar_duration == 60  # only 1m bars
        ]
        if not rows:
            return
        try:
            await self._db.execute_many(
                """INSERT OR REPLACE INTO market_bars
                   (timestamp, symbol, source, open, high, low, close, volume,
                    tick_count, n_ticks, first_source_ts, last_source_ts,
                    late_ticks_dropped, close_reason, bar_duration)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        except Exception as e:
            logger.debug(
                "Bar write failed: %s: %s; row_count=%d; first_row_symbol=%s",
                type(e).__name__, e, len(rows),
                rows[0][1] if rows else None,
                exc_info=True,
            )
            raise
        now = time.time()
        with self._status_lock:
            self._bars_writes_total += len(rows)
            self._last_write_ts = now
            self._consecutive_failures = 0
        logger.debug("Flushed %d bars to market_bars", len(rows))


    async def _write_signal(self, event: SignalEvent) -> None:
        """Write a signal event immediately."""
        await self._db.execute(
            """INSERT OR IGNORE INTO signal_events
               (timestamp, detector, symbol, signal_type, priority, data)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat(),
                event.detector,
                event.symbol,
                event.signal_type,
                int(event.priority),
                str(event.data),
            ),
        )
        now = time.time()
        with self._status_lock:
            self._signals_writes_total += 1
            self._last_write_ts = now
            self._consecutive_failures = 0
        logger.debug("Persisted signal: %s %s", event.detector, event.signal_type)

    async def _write_phase3_signal(self, event: Phase3SignalEvent) -> None:
        """Write a Phase 3 signal event immediately."""
        regime_snapshot = (
            normalize_snapshot(event.regime_snapshot)
            if event.regime_snapshot
            else None
        )
        features_snapshot = (
            normalize_snapshot(event.features_snapshot)
            if event.features_snapshot
            else None
        )
        regime_json = (
            json.dumps(regime_snapshot, sort_keys=True, separators=(",", ":"))
            if regime_snapshot
            else None
        )
        features_json = (
            json.dumps(features_snapshot, sort_keys=True, separators=(",", ":"))
            if features_snapshot
            else None
        )
        await self._db.write_signal(
            idempotency_key=event.idempotency_key,
            timestamp_ms=event.timestamp_ms,
            strategy_id=event.strategy_id,
            config_hash=event.config_hash,
            symbol=event.symbol,
            direction=event.direction,
            signal_type=event.signal_type,
            timeframe=event.timeframe,
            entry_type=event.entry_type,
            entry_price=event.entry_price,
            stop_price=event.stop_price,
            tp_price=event.tp_price,
            horizon=event.horizon,
            confidence=event.confidence,
            quality_score=event.quality_score,
            data_quality_flags=event.data_quality_flags,
            regime_snapshot_json=regime_json,
            features_snapshot_json=features_json,
            explain=event.explain,
        )
        now = time.time()
        with self._status_lock:
            self._signals_writes_total += 1
            self._last_write_ts = now
            self._consecutive_failures = 0
        logger.debug("Persisted Phase 3 signal: %s", event.strategy_id)

    async def _write_metric(self, event: MetricEvent) -> None:
        """Generic DB writer for market metrics."""
        import json
        ts_iso = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()

        # metadata_json handling
        meta = None
        if event.extra:
            try:
                meta = json.dumps(event.extra)
            except (TypeError, ValueError):
                meta = str(event.extra)

        await self._db.insert_market_metric(
            timestamp=ts_iso,
            source=event.source,
            symbol=event.symbol,
            metric=event.metric,
            value=event.value,
            metadata_json=meta
        )
        now = time.time()
        with self._status_lock:
            self._metrics_writes_total += 1
            self._last_write_ts = now
            self._consecutive_failures = 0
        logger.debug("Persisted metric: %s:%s=%s", event.symbol, event.metric, event.value)

    async def _write_component_heartbeat(self, event: ComponentHeartbeatEvent) -> None:
        """Write a structured component heartbeat to the DB."""
        import json
        ts_iso = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()
        extra_json = json.dumps(event.extra) if event.extra else None
        await self._db.execute(
            """INSERT INTO component_heartbeats
               (timestamp, component, uptime_seconds, events_processed,
                latest_lag_ms, health, extra_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ts_iso,
                event.component,
                event.uptime_seconds,
                event.events_processed,
                event.latest_lag_ms,
                event.health,
                extra_json,
            ),
        )

    def emit_heartbeat(self) -> ComponentHeartbeatEvent:
        """Create and publish a structured heartbeat for PersistenceManager."""
        now = time.monotonic()
        with self._status_lock:
            total_writes = self._bars_writes_total + self._metrics_writes_total + self._signals_writes_total
            lag_ms = self._last_persist_lag_ms
            persist_lag_crypto = self._persist_lag_crypto_ema
            persist_lag_deribit = self._persist_lag_deribit_ema
            persist_lag_equities = self._persist_lag_equities_ema
            source_ts_future_clamped_total = self._source_ts_future_clamped_total
            source_ts_stale_ignored_total = self._source_ts_stale_ignored_total
            source_ts_units_discarded_total = self._source_ts_units_discarded_total
            source_ts_missing_total = self._source_ts_missing_total

        health = "ok"
        if self._consecutive_failures > 0:
            health = "degraded"
        if self._db_write_errors > 5:
            health = "down"

        with self._bar_lock:
            ingestion_paused = self._ingestion_paused
            bars_rejected_paused = self._bars_rejected_paused
            spool_active = self._spool_active
            spool_bars_pending = self._spool_bars_pending
        if ingestion_paused:
            health = "down"

        hb = ComponentHeartbeatEvent(
            component="persistence",
            uptime_seconds=round(now - self._start_time, 1),
            events_processed=total_writes,
            latest_lag_ms=round(lag_ms, 1) if lag_ms is not None else None,
            health=health,
            extra={
                "bars_writes_total": self._bars_writes_total,
                "bars_dropped_total": self._bars_dropped_total,
                "bar_flush_success_total": self._bar_flush_success_total,
                "bar_flush_failure_total": self._bar_flush_failure_total,
                "bar_retry_count": self._bar_retry_count,
                "metrics_writes_total": self._metrics_writes_total,
                "signals_writes_total": self._signals_writes_total,
                "signals_dropped_total": self._signals_dropped_total,
                "metrics_dropped_total": self._metrics_dropped_total,
                "heartbeats_dropped_total": self._heartbeats_dropped_total,
                "signal_queue_depth": self._signal_queue.qsize(),
                "telemetry_queue_depth": self._telemetry_queue.qsize(),
                "persist_lag_ema_ms": round(self._persist_lag_ema, 1) if self._persist_lag_ema else None,
                "persist_lag_crypto_ema_ms": (
                    round(persist_lag_crypto, 1) if persist_lag_crypto is not None else None
                ),
                "persist_lag_deribit_ema_ms": (
                    round(persist_lag_deribit, 1) if persist_lag_deribit is not None else None
                ),
                "persist_lag_equities_ema_ms": (
                    round(persist_lag_equities, 1) if persist_lag_equities is not None else None
                ),
                "source_ts_future_clamped_total": source_ts_future_clamped_total,
                "source_ts_stale_ignored_total": source_ts_stale_ignored_total,
                "source_ts_units_discarded_total": source_ts_units_discarded_total,
                "source_ts_missing_total": source_ts_missing_total,
                "spool_active": spool_active,
                "spool_bars_pending": spool_bars_pending,
                "ingestion_paused": ingestion_paused,
                "bars_rejected_paused": bars_rejected_paused,
            },
        )
        self._bus.publish(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, hb)
        return hb

    def get_status(self) -> Dict[str, Any]:
        with self._bar_lock:
            bar_buffer_size = len(self._bar_buffer)
            spool_active = self._spool_active
            spool_bars_pending = self._spool_bars_pending
            spool_bytes_written = self._spool_bytes_written
            bars_spooled_total = self._bars_spooled_total
            spool_write_errors = self._spool_write_errors
            ingestion_paused = self._ingestion_paused
            bars_rejected_paused = self._bars_rejected_paused
            pause_entered_ts = self._pause_entered_ts
        with self._status_lock:
            last_flush_ts = self._last_flush_ts
            last_flush_ms = self._last_flush_ms
            flush_ema = self._flush_latency_ema
            last_write_ts = self._last_write_ts
            db_write_errors = self._db_write_errors
            last_error = self._last_error
            metrics_writes_total = self._metrics_writes_total
            bars_writes_total = self._bars_writes_total
            signals_writes_total = self._signals_writes_total
            consecutive_failures = self._consecutive_failures
            persist_lag_ms = self._last_persist_lag_ms
            persist_lag_ema = self._persist_lag_ema
            persist_lag_crypto = self._persist_lag_crypto_ema
            persist_lag_deribit = self._persist_lag_deribit_ema
            persist_lag_equities = self._persist_lag_equities_ema
            source_ts_future_clamped_total = self._source_ts_future_clamped_total
            source_ts_stale_ignored_total = self._source_ts_stale_ignored_total
            source_ts_units_discarded_total = self._source_ts_units_discarded_total
            source_ts_missing_total = self._source_ts_missing_total
            source_ts_future_clamped_by_symbol = dict(self._source_ts_future_clamped_by_symbol)
            source_ts_stale_ignored_by_symbol = dict(self._source_ts_stale_ignored_by_symbol)
            source_ts_units_discarded_by_symbol = dict(self._source_ts_units_discarded_by_symbol)
            source_ts_missing_by_symbol = dict(self._source_ts_missing_by_symbol)
            signals_dropped_total = self._signals_dropped_total
            metrics_dropped_total = self._metrics_dropped_total
            heartbeats_dropped_total = self._heartbeats_dropped_total
            bars_dropped_total = self._bars_dropped_total
            bar_flush_success = self._bar_flush_success_total
            bar_flush_failure = self._bar_flush_failure_total
            bar_retry_count = self._bar_retry_count

        now = time.time()
        age_seconds = (now - last_write_ts) if last_write_ts else None
        status = "ok"
        if consecutive_failures > 0 or db_write_errors > 0:
            status = "degraded"
        if last_write_ts is None:
            status = "unknown"
        if ingestion_paused:
            status = "paused"

        from .status import build_status

        return build_status(
            name="persistence",
            type="internal",
            status=status,
            last_success_ts=last_write_ts,
            last_error=last_error,
            consecutive_failures=consecutive_failures,
            request_count=metrics_writes_total + bars_writes_total + signals_writes_total,
            error_count=db_write_errors,
            avg_latency_ms=round(flush_ema, 2) if flush_ema is not None else None,
            last_latency_ms=round(last_flush_ms, 2) if last_flush_ms is not None else None,
            last_poll_ts=last_flush_ts,
            age_seconds=round(age_seconds, 1) if age_seconds is not None else None,
            extras={
                "bar_buffer_size": bar_buffer_size,
                "bar_buffer_max": self._bar_buffer_max,
                "last_flush_ts": (
                    datetime.fromtimestamp(last_flush_ts, tz=timezone.utc).isoformat()
                    if last_flush_ts
                    else None
                ),
                "metrics_writes_total": metrics_writes_total,
                "bars_writes_total": bars_writes_total,
                "signals_writes_total": signals_writes_total,
                "signals_dropped_total": signals_dropped_total,
                "metrics_dropped_total": metrics_dropped_total,
                "heartbeats_dropped_total": heartbeats_dropped_total,
                "bars_dropped_total": bars_dropped_total,
                "bar_flush_success_total": bar_flush_success,
                "bar_flush_failure_total": bar_flush_failure,
                "bar_retry_count": bar_retry_count,
                "signal_queue_depth": self._signal_queue.qsize(),
                "telemetry_queue_depth": self._telemetry_queue.qsize(),
                "write_queue_depth": self._signal_queue.qsize() + self._telemetry_queue.qsize(),
                "persist_lag_ms": round(persist_lag_ms, 1) if persist_lag_ms is not None else None,
                "persist_lag_ema_ms": round(persist_lag_ema, 1) if persist_lag_ema is not None else None,
                "persist_lag_crypto_ema_ms": (
                    round(persist_lag_crypto, 1) if persist_lag_crypto is not None else None
                ),
                "persist_lag_deribit_ema_ms": (
                    round(persist_lag_deribit, 1) if persist_lag_deribit is not None else None
                ),
                "persist_lag_equities_ema_ms": (
                    round(persist_lag_equities, 1) if persist_lag_equities is not None else None
                ),
                "source_ts_future_clamped_total": source_ts_future_clamped_total,
                "source_ts_stale_ignored_total": source_ts_stale_ignored_total,
                "source_ts_units_discarded_total": source_ts_units_discarded_total,
                "source_ts_missing_total": source_ts_missing_total,
                "source_ts_future_clamped_by_symbol": source_ts_future_clamped_by_symbol,
                "source_ts_stale_ignored_by_symbol": source_ts_stale_ignored_by_symbol,
                "source_ts_units_discarded_by_symbol": source_ts_units_discarded_by_symbol,
                "source_ts_missing_by_symbol": source_ts_missing_by_symbol,
                # Spool metrics
                "spool_active": spool_active,
                "spool_bars_pending": spool_bars_pending,
                "spool_file_size": spool_bytes_written,
                "spool_max_bytes": self._spool_max_bytes,
                "bars_spooled_total": bars_spooled_total,
                "spool_write_errors": spool_write_errors,
                # Safe-pause metrics
                "ingestion_paused": ingestion_paused,
                "bars_rejected_paused": bars_rejected_paused,
                "pause_entered_ts": pause_entered_ts,
            },
        )
