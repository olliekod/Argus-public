"""
Resource Monitor — Process / Disk / Log Health
===============================================

Tracks process RSS, CPU, file descriptors, disk space, WAL size,
and log error/warn counts for soak-run leak/thrash detection.

Uses ``psutil`` if available; degrades gracefully without it.
"""

from __future__ import annotations

import logging
import os
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger("argus.soak.resource")

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


class ResourceMonitor:
    """Lightweight resource tracker for soak runs.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (for WAL size check).
    log_ring : deque or None
        The orchestrator's recent-log ring buffer (for error counting).
    """

    def __init__(
        self,
        db_path: str = "data/argus.db",
        log_ring: Optional[Deque[str]] = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._log_ring = log_ring
        self._process: Any = None
        if _HAS_PSUTIL:
            try:
                self._process = psutil.Process(os.getpid())
            except Exception:
                pass

        # Error/warn tracking via a logging handler
        self._error_total: int = 0
        self._warn_total: int = 0
        self._boot_ts = time.time()
        self._hourly_errors: Deque[Tuple[float, str]] = deque(maxlen=5000)
        self._hourly_warns: Deque[Tuple[float, str]] = deque(maxlen=5000)
        self._lock = threading.Lock()

        # Install a counting handler on the root argus logger
        self._handler = _CountingHandler(self)
        root = logging.getLogger("argus")
        root.addHandler(self._handler)

    # ── Snapshots ─────────────────────────────────────────

    def get_process_snapshot(self) -> Dict[str, Any]:
        """Return process-level metrics (RSS, CPU, FDs)."""
        result: Dict[str, Any] = {"psutil_available": _HAS_PSUTIL}
        if not self._process:
            return result
        try:
            mem = self._process.memory_info()
            result["rss_mb"] = round(mem.rss / (1024 * 1024), 1)
        except Exception:
            result["rss_mb"] = None
        try:
            result["cpu_percent"] = self._process.cpu_percent(interval=0)
        except Exception:
            result["cpu_percent"] = None
        try:
            result["open_fds"] = self._process.num_fds()
        except Exception:
            # num_fds() not available on all platforms
            try:
                # Fallback: count /proc/self/fd on Linux
                fd_dir = Path("/proc/self/fd")
                if fd_dir.exists():
                    result["open_fds"] = len(list(fd_dir.iterdir()))
                else:
                    result["open_fds"] = None
            except Exception:
                result["open_fds"] = None
        return result

    def get_storage_snapshot(self) -> Dict[str, Any]:
        """Return disk / WAL health metrics."""
        result: Dict[str, Any] = {}

        # Disk free space on the partition hosting the DB
        try:
            if self._db_path.exists():
                if _HAS_PSUTIL:
                    usage = psutil.disk_usage(str(self._db_path.parent))
                    result["disk_free_gb"] = round(usage.free / (1024 ** 3), 2)
                    result["disk_total_gb"] = round(usage.total / (1024 ** 3), 2)
                else:
                    st = os.statvfs(str(self._db_path.parent))
                    result["disk_free_gb"] = round(
                        (st.f_bavail * st.f_frsize) / (1024 ** 3), 2
                    )
            else:
                result["disk_free_gb"] = None
        except Exception:
            result["disk_free_gb"] = None

        # DB file size
        try:
            if self._db_path.exists():
                result["db_size_mb"] = round(
                    self._db_path.stat().st_size / (1024 * 1024), 1
                )
            else:
                result["db_size_mb"] = None
        except Exception:
            result["db_size_mb"] = None

        # WAL file size
        wal_path = Path(str(self._db_path) + "-wal")
        try:
            if wal_path.exists():
                result["wal_size_mb"] = round(
                    wal_path.stat().st_size / (1024 * 1024), 1
                )
            else:
                result["wal_size_mb"] = 0.0
        except Exception:
            result["wal_size_mb"] = None

        return result

    def get_log_entropy(self) -> Dict[str, Any]:
        """Return error/warn counts + top unique error messages."""
        now = time.time()
        one_hour_ago = now - 3600

        with self._lock:
            # Prune entries older than 1 hour
            while self._hourly_errors and self._hourly_errors[0][0] < one_hour_ago:
                self._hourly_errors.popleft()
            while self._hourly_warns and self._hourly_warns[0][0] < one_hour_ago:
                self._hourly_warns.popleft()

            errors_last_hour = len(self._hourly_errors)
            warns_last_hour = len(self._hourly_warns)

            # Top N unique errors in last hour
            error_counts: Dict[str, int] = {}
            for _, msg in self._hourly_errors:
                # Truncate long messages for grouping
                key = msg[:120]
                error_counts[key] = error_counts.get(key, 0) + 1

            top_errors = sorted(
                error_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]

        return {
            "errors_total": self._error_total,
            "warns_total": self._warn_total,
            "errors_last_hour": errors_last_hour,
            "warns_last_hour": warns_last_hour,
            "top_errors_last_hour": [
                {"message": msg, "count": cnt} for msg, cnt in top_errors
            ],
        }

    def get_full_snapshot(self) -> Dict[str, Any]:
        """Combined resource snapshot."""
        return {
            "process": self.get_process_snapshot(),
            "storage": self.get_storage_snapshot(),
            "log_entropy": self.get_log_entropy(),
        }

    # ── Internal: logging handler ────────────────────────

    def _record_log(self, level: int, message: str) -> None:
        now = time.time()
        with self._lock:
            if level >= logging.ERROR:
                self._error_total += 1
                self._hourly_errors.append((now, message))
            elif level >= logging.WARNING:
                self._warn_total += 1
                self._hourly_warns.append((now, message))


class _CountingHandler(logging.Handler):
    """Lightweight handler that increments ResourceMonitor counters."""

    def __init__(self, monitor: ResourceMonitor) -> None:
        super().__init__(level=logging.WARNING)
        self._monitor = monitor

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._monitor._record_log(record.levelno, record.getMessage())
        except Exception:
            pass
