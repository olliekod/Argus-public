"""
Activity status tracker for providers and detectors.

Includes per-symbol tick tracking for fine-grained staleness detection.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional


_DEFAULT_BOOT_GRACE_S = 120
_DEFAULT_WARN_AFTER_S = 60
_DEFAULT_ALERT_AFTER_S = 300
_DEFAULT_SYMBOL_STALE_S = 120   # Per-symbol staleness threshold
_MAX_SYMBOL_ENTRIES = 500       # Cap per-symbol tracking to bound memory


class ActivityStatusTracker:
    """Track provider + detector activity based on bus events.

    Now includes per-symbol tick tracking: each ``record_provider_event``
    call with a ``symbol`` kwarg updates a per-symbol last-seen timestamp.
    ``get_stale_symbols`` returns symbols that haven't been seen within
    ``symbol_stale_s`` seconds.
    """

    def __init__(
        self,
        *,
        provider_names: Optional[list[str]] = None,
        detector_names: Optional[list[str]] = None,
        boot_ts: Optional[float] = None,
        boot_grace_s: int = _DEFAULT_BOOT_GRACE_S,
        warn_after_s: int = _DEFAULT_WARN_AFTER_S,
        alert_after_s: int = _DEFAULT_ALERT_AFTER_S,
        symbol_stale_s: float = _DEFAULT_SYMBOL_STALE_S,
        max_entries: int = 200,
        max_symbol_entries: int = _MAX_SYMBOL_ENTRIES,
    ) -> None:
        self._boot_ts = boot_ts or time.time()
        self._boot_grace_s = boot_grace_s
        self._warn_after_s = warn_after_s
        self._alert_after_s = alert_after_s
        self._symbol_stale_s = symbol_stale_s
        self._max_entries = max_entries
        self._max_symbol_entries = max_symbol_entries
        self._providers: Dict[str, Dict[str, Any]] = {}
        self._detectors: Dict[str, Dict[str, Any]] = {}

        # Per-symbol tracking:  { "provider::symbol" -> last_event_ts }
        self._symbol_ticks: OrderedDict[str, float] = OrderedDict()

        for name in provider_names or []:
            self.register_provider(name, configured=True)
        for name in detector_names or []:
            self.register_detector(name)

    def register_provider(self, name: str, *, configured: bool) -> None:
        if name not in self._providers:
            if len(self._providers) >= self._max_entries:
                return
            self._providers[name] = {
                "configured": configured,
                "last_msg_ts": None,
                "last_source_ts": None,
                "last_event_ts": None,
                "counters": {
                    "messages_total": 0,
                    "quotes_total": 0,
                    "bars_total": 0,
                    "metrics_total": 0,
                },
            }
        else:
            self._providers[name]["configured"] = configured

    def register_detector(self, name: str) -> None:
        if name not in self._detectors:
            if len(self._detectors) >= self._max_entries:
                return
            self._detectors[name] = {
                "last_event_ts": None,
                "last_signal_ts": None,
                "counters": {
                    "events_total": 0,
                    "bars_total": 0,
                    "metrics_total": 0,
                    "signals_total": 0,
                },
            }

    def record_provider_event(
        self,
        name: str,
        *,
        event_ts: Optional[float] = None,
        source_ts: Optional[float] = None,
        kind: str = "message",
        symbol: Optional[str] = None,
    ) -> None:
        if name not in self._providers and len(self._providers) >= self._max_entries:
            return
        entry = self._providers.setdefault(
            name,
            {
                "configured": True,
                "last_msg_ts": None,
                "last_source_ts": None,
                "last_event_ts": None,
                "counters": {
                    "messages_total": 0,
                    "quotes_total": 0,
                    "bars_total": 0,
                    "metrics_total": 0,
                },
            },
        )
        ts = event_ts or time.time()
        entry["last_msg_ts"] = ts
        entry["last_event_ts"] = ts
        if source_ts:
            entry["last_source_ts"] = source_ts
        entry["counters"]["messages_total"] += 1
        if kind == "quote":
            entry["counters"]["quotes_total"] += 1
        elif kind == "bar":
            entry["counters"]["bars_total"] += 1
        elif kind == "metric":
            entry["counters"]["metrics_total"] += 1

        # Per-symbol tick tracking
        if symbol:
            key = f"{name}::{symbol}"
            self._symbol_ticks[key] = ts
            self._symbol_ticks.move_to_end(key)
            # Evict oldest if over capacity
            while len(self._symbol_ticks) > self._max_symbol_entries:
                self._symbol_ticks.popitem(last=False)

    def record_detector_event(
        self,
        name: str,
        *,
        event_ts: Optional[float] = None,
        kind: str = "event",
    ) -> None:
        if name not in self._detectors and len(self._detectors) >= self._max_entries:
            return
        entry = self._detectors.setdefault(
            name,
            {
                "last_event_ts": None,
                "last_signal_ts": None,
                "counters": {
                    "events_total": 0,
                    "bars_total": 0,
                    "metrics_total": 0,
                    "signals_total": 0,
                },
            },
        )
        ts = event_ts or time.time()
        entry["last_event_ts"] = ts
        entry["counters"]["events_total"] += 1
        if kind == "bar":
            entry["counters"]["bars_total"] += 1
        elif kind == "metric":
            entry["counters"]["metrics_total"] += 1

    def record_detector_signal(
        self, name: str, *, event_ts: Optional[float] = None
    ) -> None:
        if name not in self._detectors and len(self._detectors) >= self._max_entries:
            return
        entry = self._detectors.setdefault(
            name,
            {
                "last_event_ts": None,
                "last_signal_ts": None,
                "counters": {
                    "events_total": 0,
                    "bars_total": 0,
                    "metrics_total": 0,
                    "signals_total": 0,
                },
            },
        )
        ts = event_ts or time.time()
        entry["last_signal_ts"] = ts
        entry["last_event_ts"] = ts
        entry["counters"]["signals_total"] += 1
        entry["counters"]["events_total"] += 1

    def _health_from_age(
        self, *, age_s: Optional[float], now: float, configured: bool
    ) -> str:
        if not configured:
            return "unknown"
        if age_s is None:
            if (now - self._boot_ts) < self._boot_grace_s:
                return "unknown"
            return "alert"
        if age_s >= self._alert_after_s:
            return "alert"
        if age_s >= self._warn_after_s:
            return "warn"
        return "ok"

    def get_provider_statuses(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        statuses: Dict[str, Any] = {}
        for name, entry in self._providers.items():
            age_s = (now - entry["last_msg_ts"]) if entry["last_msg_ts"] else None
            statuses[name] = {
                "health": self._health_from_age(
                    age_s=age_s, now=now, configured=entry.get("configured", True)
                ),
                "last_msg_age_s": round(age_s, 1) if age_s is not None else None,
                "last_source_ts": entry["last_source_ts"],
                "last_event_ts": entry["last_event_ts"],
                "counters": dict(entry["counters"]),
            }
        return statuses

    def get_detector_statuses(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        statuses: Dict[str, Any] = {}
        for name, entry in self._detectors.items():
            age_s = (now - entry["last_event_ts"]) if entry["last_event_ts"] else None
            statuses[name] = {
                "health": self._health_from_age(
                    age_s=age_s, now=now, configured=True
                ),
                "last_event_age_s": round(age_s, 1) if age_s is not None else None,
                "last_signal_ts": entry["last_signal_ts"],
                "counters": dict(entry["counters"]),
            }
        return statuses

    # ------------------------------------------------------------------
    # Per-symbol staleness
    # ------------------------------------------------------------------

    def get_stale_symbols(
        self,
        now: Optional[float] = None,
        stale_after_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return symbols whose last tick is older than *stale_after_s*.

        Each entry is ``{"provider": ..., "symbol": ..., "age_s": ..., "last_ts": ...}``.
        """
        now = now or time.time()
        threshold = stale_after_s if stale_after_s is not None else self._symbol_stale_s
        stale: List[Dict[str, Any]] = []
        for key, last_ts in self._symbol_ticks.items():
            age = now - last_ts
            if age > threshold:
                provider, symbol = key.split("::", 1)
                stale.append({
                    "provider": provider,
                    "symbol": symbol,
                    "age_s": round(age, 1),
                    "last_ts": last_ts,
                })
        return stale

    def get_symbol_statuses(
        self,
        provider: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return per-symbol health statuses, optionally filtered by provider.

        Returns ``{ "provider::symbol": {"health": ..., "age_s": ...} }``.
        """
        now = now or time.time()
        result: Dict[str, Dict[str, Any]] = {}
        for key, last_ts in self._symbol_ticks.items():
            if provider and not key.startswith(f"{provider}::"):
                continue
            age = now - last_ts
            health = "ok"
            if age >= self._alert_after_s:
                health = "alert"
            elif age >= self._symbol_stale_s:
                health = "stale"
            elif age >= self._warn_after_s:
                health = "warn"
            result[key] = {"health": health, "age_s": round(age, 1), "last_ts": last_ts}
        return result
