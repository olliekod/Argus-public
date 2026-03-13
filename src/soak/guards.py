"""
Soak Guardian — Fail-Fast / Fail-Loud Threshold Guards
=======================================================

Periodically evaluates system health metrics against configurable
thresholds and fires rate-limited alerts when correctness is
threatened.

All guards update a ``health_status`` dict that the soak summary
and dashboard consume.  Alerts go through a callback (typically
wired to Telegram).

Guards never crash ingestion — they only observe and alert.
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.soak.guards")

# Guard severity levels
WARN = "WARN"
ALERT = "ALERT"

# Default thresholds (all overridable via config)
_DEFAULTS: Dict[str, Any] = {
    # EventBus quote drops
    "quote_drops_alert_window_s": 300,
    "quote_drops_alert_threshold": 1,

    # Rejected quotes (missing source_ts)
    "rejected_quotes_per_min_threshold": 5,

    # Heartbeat staleness
    "heartbeat_missing_intervals": 2,
    "heartbeat_interval_s": 60,

    # Persistence write-queue depth
    "persist_queue_depth_warn": 5000,
    "persist_queue_depth_alert": 8000,
    "persist_queue_sustained_s": 30,

    # Persist lag
    "persist_lag_p95_warn_ms": 5000,
    "persist_lag_p95_alert_ms": 15000,
    "persist_lag_sustained_s": 60,
    "persist_lag_use_crypto_only": True,
    "persist_lag_deribit_enabled": False,
    "persist_lag_equities_enabled": False,

    # Bar flush failures
    "bar_flush_failure_threshold": 1,

    # Log flood (rate = errors_last_hour / 60; keep threshold; fix source errors in hot paths)
    "log_error_flood_threshold_per_min": 50,
    "log_error_flood_window_min": 5,

    # Bar liveness (no bar emitted for a monitored symbol)
    "bar_liveness_symbols": ["BTC/USDT:USDT"],
    "bar_liveness_timeout_s": 180,

    # Disk fatigue
    "disk_free_warn_gb": 5.0,
    "disk_free_alert_gb": 2.0,
    "wal_size_warn_mb": 1024,
    "wal_size_alert_mb": 2048,

    # Bar buffer pressure / spool
    "bar_buffer_pressure_warn_pct": 70,   # % of buffer max
    "bar_buffer_pressure_alert_pct": 90,  # % of buffer max

    # Alert rate limiting
    "alert_cooldown_s": 300,
}


class SoakGuardian:
    """Evaluates soak guards and fires rate-limited alerts.

    Parameters
    ----------
    config : dict
        Threshold overrides (merged with _DEFAULTS).
    alert_callback : callable or None
        ``async def(severity, guard_name, message) -> None``
        Called on threshold breach (rate-limited).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        alert_callback: Optional[Callable] = None,
    ) -> None:
        raw_cfg = config or {}
        self._cfg = {**_DEFAULTS, **raw_cfg}
        persist_cfg = raw_cfg.get("persist_lag", {}) if isinstance(raw_cfg, dict) else {}
        if isinstance(persist_cfg, dict):
            use_crypto_only = persist_cfg.get("use_crypto_only", True)
            self._cfg["persist_lag_use_crypto_only"] = bool(use_crypto_only)
            if "deribit_enabled" in persist_cfg:
                self._cfg["persist_lag_deribit_enabled"] = bool(
                    persist_cfg.get("deribit_enabled")
                )
            if "equities_enabled" in persist_cfg:
                self._cfg["persist_lag_equities_enabled"] = bool(
                    persist_cfg.get("equities_enabled")
                )
        self._alert_cb = alert_callback
        self._lock = threading.Lock()

        # Last alert timestamps per guard (for rate limiting)
        self._last_alert_ts: Dict[str, float] = {}

        # Guard state
        self._health: Dict[str, str] = {}  # guard_name → "ok" | "warn" | "alert"
        self._guard_messages: Dict[str, str] = {}  # guard_name → last message

        # Rolling windows for sustained checks
        self._persist_queue_high_since: Optional[float] = None
        self._persist_lag_high_since: Dict[str, Optional[float]] = {}

        # Previous bus stats for delta computation
        self._prev_bus_stats: Dict[str, Dict[str, int]] = {}
        self._prev_check_ts: float = time.time()

        # Bars dropped counter (set by persistence manager)
        self._bars_dropped_count: int = 0

    @property
    def health_status(self) -> Dict[str, str]:
        """Current health per guard."""
        with self._lock:
            return dict(self._health)

    @property
    def guard_messages(self) -> Dict[str, str]:
        """Last message per guard."""
        with self._lock:
            return dict(self._guard_messages)

    def get_overall_health(self) -> str:
        """Return worst-case health across all guards."""
        with self._lock:
            statuses = list(self._health.values())
        if any(s == "alert" for s in statuses):
            return "alert"
        if any(s == "warn" for s in statuses):
            return "warn"
        return "ok"

    # ── Evaluate all guards ──────────────────────────────

    def evaluate(
        self,
        bus_stats: Dict[str, Dict[str, Any]],
        bar_builder_status: Dict[str, Any],
        persistence_status: Dict[str, Any],
        resource_snapshot: Dict[str, Any],
        component_heartbeats: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Run all guards and return list of triggered alerts.

        Parameters
        ----------
        bus_stats
            From ``EventBus.get_status_summary()``.
        bar_builder_status
            From ``BarBuilder.get_status()``.
        persistence_status
            From ``PersistenceManager.get_status()``.
        resource_snapshot
            From ``ResourceMonitor.get_full_snapshot()``.
        component_heartbeats
            ``{component_name: last_heartbeat_epoch}`` or None.

        Returns
        -------
        list of dicts
            Each dict: ``{"severity", "guard", "message"}``.
        """
        triggered: List[Dict[str, Any]] = []
        now = time.time()

        triggered += self._check_quote_drops(bus_stats, now)
        triggered += self._check_rejected_quotes(bar_builder_status, now)
        triggered += self._check_heartbeat_staleness(component_heartbeats, now)
        triggered += self._check_persist_queue(persistence_status, now)
        triggered += self._check_persist_lag(persistence_status, now)
        triggered += self._check_bar_flush_failures(persistence_status, now)
        triggered += self._check_log_flood(resource_snapshot, now)
        triggered += self._check_bar_liveness(bar_builder_status, now)
        triggered += self._check_disk_fatigue(resource_snapshot, now)
        triggered += self._check_bars_dropped(now)
        triggered += self._check_bar_buffer_pressure(persistence_status, now)
        triggered += self._check_ingestion_paused(persistence_status, now)

        # Update previous state
        self._prev_bus_stats = {
            topic: dict(stats) for topic, stats in bus_stats.items()
        }
        self._prev_check_ts = now

        return triggered

    # ── Individual guard checks ──────────────────────────

    def _check_quote_drops(
        self, bus_stats: Dict, now: float
    ) -> List[Dict[str, Any]]:
        guard = "quote_drops"
        quotes_stats = bus_stats.get("market.quotes", {})
        dropped = quotes_stats.get("dropped_events", 0)
        prev = self._prev_bus_stats.get("market.quotes", {}).get("dropped_events", 0)
        new_drops = dropped - prev

        if new_drops > self._cfg["quote_drops_alert_threshold"]:
            return self._fire(
                ALERT, guard,
                f"market.quotes dropped {new_drops} events in last check interval "
                f"(total: {dropped})",
                now,
            )
        self._set_health(guard, "ok")
        return []

    def _check_rejected_quotes(
        self, bb_status: Dict, now: float
    ) -> List[Dict[str, Any]]:
        guard = "rejected_quotes"
        extras = bb_status.get("extras", {})
        rejected = extras.get("quotes_rejected_total", 0)
        by_symbol = extras.get("quotes_rejected_by_symbol", {})

        threshold = self._cfg["rejected_quotes_per_min_threshold"]
        # Simple rate: total / uptime_minutes
        uptime_s = bb_status.get("extras", {}).get("uptime_s", 0)
        if not uptime_s:
            # Estimate from staleness
            uptime_s = max(1, time.time() - self._prev_check_ts)
        rate_per_min = (rejected / max(1, uptime_s)) * 60

        if rate_per_min > threshold:
            detail = ", ".join(f"{s}={c}" for s, c in by_symbol.items()) if by_symbol else ""
            return self._fire(
                ALERT, guard,
                f"quotes_rejected rate={rate_per_min:.1f}/min > {threshold}/min "
                f"(total={rejected}, by_symbol: {detail})",
                now,
            )
        self._set_health(guard, "ok")
        return []

    def _check_heartbeat_staleness(
        self, heartbeats: Optional[Dict[str, float]], now: float
    ) -> List[Dict[str, Any]]:
        guard = "heartbeat_missing"
        if not heartbeats:
            self._set_health(guard, "ok")
            return []

        max_age = (
            self._cfg["heartbeat_missing_intervals"]
            * self._cfg["heartbeat_interval_s"]
        )
        triggered = []
        stale = []
        for comp, last_ts in heartbeats.items():
            age = now - last_ts
            if age > max_age:
                stale.append(f"{comp}={age:.0f}s")
        if stale:
            triggered = self._fire(
                ALERT, guard,
                f"Component heartbeat missing > {max_age}s: {', '.join(stale)}",
                now,
            )
        else:
            self._set_health(guard, "ok")
        return triggered

    def _check_persist_queue(
        self, persist_status: Dict, now: float
    ) -> List[Dict[str, Any]]:
        guard = "persist_queue_depth"
        extras = persist_status.get("extras", {})
        depth = extras.get("write_queue_depth", 0)

        warn_thresh = self._cfg["persist_queue_depth_warn"]
        alert_thresh = self._cfg["persist_queue_depth_alert"]
        sustained_s = self._cfg["persist_queue_sustained_s"]

        if depth >= alert_thresh:
            if self._persist_queue_high_since is None:
                self._persist_queue_high_since = now
            if now - self._persist_queue_high_since >= sustained_s:
                return self._fire(
                    ALERT, guard,
                    f"Write-queue depth={depth} >= {alert_thresh} "
                    f"for {sustained_s}s",
                    now,
                )
            self._set_health(guard, "warn")
            return []
        elif depth >= warn_thresh:
            if self._persist_queue_high_since is None:
                self._persist_queue_high_since = now
            if now - self._persist_queue_high_since >= sustained_s:
                return self._fire(
                    WARN, guard,
                    f"Write-queue depth={depth} >= {warn_thresh} "
                    f"for {sustained_s}s",
                    now,
                )
            self._set_health(guard, "warn")
            return []
        else:
            self._persist_queue_high_since = None
            self._set_health(guard, "ok")
            return []

    def _check_persist_lag(
        self, persist_status: Dict, now: float
    ) -> List[Dict[str, Any]]:
        extras = persist_status.get("extras", {})
        triggered: List[Dict[str, Any]] = []

        if self._cfg.get("persist_lag_use_crypto_only", True):
            primary_lag = extras.get("persist_lag_crypto_ema_ms")
            primary_label = "persist_lag_crypto_ema"
        else:
            primary_lag = extras.get("persist_lag_ema_ms")
            primary_label = "persist_lag_ema"

        triggered += self._check_persist_lag_metric(
            guard="persist_lag",
            lag_ema=primary_lag,
            now=now,
            label=primary_label,
        )

        if self._cfg.get("persist_lag_deribit_enabled", False):
            triggered += self._check_persist_lag_metric(
                guard="persist_lag_deribit",
                lag_ema=extras.get("persist_lag_deribit_ema_ms"),
                now=now,
                label="persist_lag_deribit_ema",
            )

        if self._cfg.get("persist_lag_equities_enabled", False):
            triggered += self._check_persist_lag_metric(
                guard="persist_lag_equities",
                lag_ema=extras.get("persist_lag_equities_ema_ms"),
                now=now,
                label="persist_lag_equities_ema",
            )

        return triggered

    def _check_persist_lag_metric(
        self,
        *,
        guard: str,
        lag_ema: Optional[float],
        now: float,
        label: str,
    ) -> List[Dict[str, Any]]:
        if lag_ema is None:
            self._set_health(guard, "ok")
            self._persist_lag_high_since.pop(guard, None)
            return []

        warn_ms = self._cfg["persist_lag_p95_warn_ms"]
        alert_ms = self._cfg["persist_lag_p95_alert_ms"]
        sustained_s = self._cfg["persist_lag_sustained_s"]
        high_since = self._persist_lag_high_since.get(guard)

        if lag_ema >= alert_ms:
            if high_since is None:
                high_since = now
                self._persist_lag_high_since[guard] = high_since
            if now - high_since >= sustained_s:
                return self._fire(
                    ALERT, guard,
                    f"{label}={lag_ema:.0f}ms >= {alert_ms}ms "
                    f"for {sustained_s}s",
                    now,
                )
            self._set_health(guard, "warn")
            return []
        if lag_ema >= warn_ms:
            if high_since is None:
                self._persist_lag_high_since[guard] = now
            self._set_health(guard, "warn")
            return []

        self._persist_lag_high_since.pop(guard, None)
        self._set_health(guard, "ok")
        return []

    def _check_bar_flush_failures(
        self, persist_status: Dict, now: float
    ) -> List[Dict[str, Any]]:
        """Alert when persistence has bar flush failures. error_count is cumulative (lifetime)."""
        guard = "bar_flush_failures"
        error_count = persist_status.get("counters", {}).get("error_count", 0)
        last_error = persist_status.get("last_error")

        if error_count > 0 and last_error == "bar_flush_failed":
            return self._fire(
                ALERT, guard,
                f"Bar flush failures detected (error_count={error_count})",
                now,
            )
        self._set_health(guard, "ok")
        return []

    def _check_log_flood(
        self, resource: Dict, now: float
    ) -> List[Dict[str, Any]]:
        """Alert when ERROR log rate (last hour / 60) exceeds threshold.
        Recoverable/hot-path failures should be logged as WARNING to avoid flooding."""
        guard = "log_flood"
        log = resource.get("log_entropy", {})
        errors_last_hour = log.get("errors_last_hour", 0)
        threshold = self._cfg["log_error_flood_threshold_per_min"]
        window_min = self._cfg["log_error_flood_window_min"]

        # Rate = errors in last hour / 60 (per-minute average)
        rate_per_min = errors_last_hour / 60.0
        if rate_per_min > threshold:
            return self._fire(
                ALERT, guard,
                f"ERROR log flood: {rate_per_min:.1f}/min > {threshold}/min "
                f"(last hour: {errors_last_hour})",
                now,
            )
        self._set_health(guard, "ok")
        return []

    def _check_bar_liveness(
        self, bb_status: Dict, now: float
    ) -> List[Dict[str, Any]]:
        guard = "bar_liveness"
        extras = bb_status.get("extras", {})
        last_ts_by_symbol = extras.get("last_bar_ts_by_symbol_epoch", {})
        timeout = self._cfg["bar_liveness_timeout_s"]
        monitored = self._cfg.get("bar_liveness_symbols", [])

        stale = []
        for sym in monitored:
            last_ts = last_ts_by_symbol.get(sym)
            if last_ts is None:
                continue  # Not yet producing bars — skip
            age = now - last_ts
            if age > timeout:
                stale.append(f"{sym}={age:.0f}s")

        if stale:
            return self._fire(
                ALERT, guard,
                f"No bars for {timeout}s: {', '.join(stale)}",
                now,
            )
        self._set_health(guard, "ok")
        return []

    def _check_disk_fatigue(
        self, resource: Dict, now: float
    ) -> List[Dict[str, Any]]:
        guard = "disk_fatigue"
        storage = resource.get("storage", {})
        triggered = []
        any_bad = False

        disk_free = storage.get("disk_free_gb")
        if disk_free is not None:
            if disk_free < self._cfg["disk_free_alert_gb"]:
                any_bad = True
                triggered += self._fire(
                    ALERT, guard,
                    f"Disk free={disk_free:.1f}GB < {self._cfg['disk_free_alert_gb']}GB",
                    now,
                )
            elif disk_free < self._cfg["disk_free_warn_gb"]:
                any_bad = True
                triggered += self._fire(
                    WARN, guard,
                    f"Disk free={disk_free:.1f}GB < {self._cfg['disk_free_warn_gb']}GB",
                    now,
                )

        wal_mb = storage.get("wal_size_mb")
        if wal_mb is not None:
            if wal_mb > self._cfg["wal_size_alert_mb"]:
                any_bad = True
                triggered += self._fire(
                    ALERT, f"{guard}_wal",
                    f"WAL size={wal_mb:.0f}MB > {self._cfg['wal_size_alert_mb']}MB",
                    now,
                )
            elif wal_mb > self._cfg["wal_size_warn_mb"]:
                any_bad = True
                triggered += self._fire(
                    WARN, f"{guard}_wal",
                    f"WAL size={wal_mb:.0f}MB > {self._cfg['wal_size_warn_mb']}MB",
                    now,
                )

        if not any_bad:
            self._set_health(guard, "ok")
        return triggered

    def _check_bars_dropped(self, now: float) -> List[Dict[str, Any]]:
        guard = "bars_dropped"
        if self._bars_dropped_count > 0:
            return self._fire(
                ALERT, guard,
                f"BARS DROPPED: {self._bars_dropped_count} bars lost! "
                f"This should never happen.",
                now,
            )
        self._set_health(guard, "ok")
        return []

    def _check_bar_buffer_pressure(
        self, persist_status: Dict, now: float
    ) -> List[Dict[str, Any]]:
        guard = "bar_buffer_pressure"
        extras = persist_status.get("extras", {})
        buf_size = extras.get("bar_buffer_size", 0)
        buf_max = extras.get("bar_buffer_max", 1)
        spool_active = extras.get("spool_active", False)
        spool_pending = extras.get("spool_bars_pending", 0)
        spool_bytes = extras.get("spool_file_size", 0)

        # Spool active is an immediate alert
        if spool_active:
            return self._fire(
                ALERT, guard,
                f"Bar spool ACTIVE — {spool_pending} bars pending on disk "
                f"({spool_bytes / (1024*1024):.1f}MB). "
                f"In-memory buffer={buf_size}/{buf_max}",
                now,
            )

        pct = (buf_size / max(1, buf_max)) * 100
        alert_pct = self._cfg["bar_buffer_pressure_alert_pct"]
        warn_pct = self._cfg["bar_buffer_pressure_warn_pct"]

        if pct >= alert_pct:
            return self._fire(
                ALERT, guard,
                f"Bar buffer at {pct:.0f}% ({buf_size}/{buf_max}) — "
                f"approaching spool threshold",
                now,
            )
        if pct >= warn_pct:
            return self._fire(
                WARN, guard,
                f"Bar buffer at {pct:.0f}% ({buf_size}/{buf_max})",
                now,
            )

        self._set_health(guard, "ok")
        return []

    def _check_ingestion_paused(
        self, persist_status: Dict, now: float
    ) -> List[Dict[str, Any]]:
        guard = "ingestion_paused"
        extras = persist_status.get("extras", {})
        paused = extras.get("ingestion_paused", False)
        rejected = extras.get("bars_rejected_paused", 0)
        pause_ts = extras.get("pause_entered_ts")

        if paused:
            duration = f"{now - pause_ts:.0f}s" if pause_ts else "unknown"
            return self._fire(
                ALERT, guard,
                f"INGESTION PAUSED — bars rejected={rejected}, "
                f"duration={duration}. "
                f"Spool full + DB unreachable. "
                f"System will auto-resume when DB recovers.",
                now,
            )
        self._set_health(guard, "ok")
        return []

    # ── Alert plumbing ───────────────────────────────────

    def _fire(
        self, severity: str, guard: str, message: str, now: float
    ) -> List[Dict[str, Any]]:
        """Fire an alert if cooldown allows."""
        cooldown = self._cfg["alert_cooldown_s"]
        last = self._last_alert_ts.get(guard, 0)

        health = "alert" if severity == ALERT else "warn"
        self._set_health(guard, health)

        if now - last < cooldown:
            return []  # Rate-limited

        self._last_alert_ts[guard] = now
        logger.warning("[GUARD:%s] %s — %s", severity, guard, message)

        alert = {"severity": severity, "guard": guard, "message": message}

        # Fire async callback if wired
        if self._alert_cb:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        self._alert_cb(severity, guard, message)
                    )
            except Exception:
                logger.debug("Guard alert callback failed", exc_info=True)

        return [alert]

    def _set_health(self, guard: str, status: str) -> None:
        with self._lock:
            self._health[guard] = status
