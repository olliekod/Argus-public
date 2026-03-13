"""
Structured logging utilities for the Argus Kalshi module.

Produces JSON-line log records so that they are machine-parseable while
still readable in a terminal.  Every record carries a ``component`` field
that identifies the subsystem (``auth``, ``rest``, ``ws``, ``orderbook``,
``strategy``, ``execution``, …).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
#  Structured formatter
# ---------------------------------------------------------------------------

class StructuredFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "msg": record.getMessage(),
        }
        # Merge any extra dict passed via `extra={"data": {...}}`
        data = getattr(record, "data", None)
        if data:
            payload["data"] = data
        return json.dumps(payload, default=str, separators=(",", ":"))


# ---------------------------------------------------------------------------
#  Module-level helpers
# ---------------------------------------------------------------------------

_LOG_SETUP_DONE = False


def setup_logging(level: int = logging.INFO) -> None:
    """Attach the structured formatter to the root ``argus_kalshi`` logger.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _LOG_SETUP_DONE
    if _LOG_SETUP_DONE:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root = logging.getLogger("argus_kalshi")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _LOG_SETUP_DONE = True


def get_logger(component: str) -> logging.Logger:
    """Return a child logger under ``argus_kalshi`` with *component* baked in."""
    setup_logging()
    logger = logging.getLogger(f"argus_kalshi.{component}")
    return logger


class ComponentLogger:
    """Convenience wrapper that injects *component* and optional *data* into
    every log call without requiring ``extra=`` at each call site.
    """

    def __init__(self, component: str) -> None:
        self._logger = get_logger(component)
        self._component = component

    # -- level methods -------------------------------------------------------
    # *args forwarded for %-formatting (e.g. log.warning("Error: %s", exc, data={...})).

    def debug(self, msg: str, *args: Any, data: Optional[Dict[str, Any]] = None) -> None:
        self._logger.debug(msg, *args, extra={"component": self._component, "data": data})

    def info(self, msg: str, *args: Any, data: Optional[Dict[str, Any]] = None) -> None:
        self._logger.info(msg, *args, extra={"component": self._component, "data": data})

    def warning(self, msg: str, *args: Any, data: Optional[Dict[str, Any]] = None) -> None:
        self._logger.warning(msg, *args, extra={"component": self._component, "data": data})

    def error(self, msg: str, *args: Any, data: Optional[Dict[str, Any]] = None) -> None:
        self._logger.error(msg, *args, extra={"component": self._component, "data": data})

    def critical(self, msg: str, *args: Any, data: Optional[Dict[str, Any]] = None) -> None:
        self._logger.critical(msg, *args, extra={"component": self._component, "data": data})


# ---------------------------------------------------------------------------
#  Latency helper
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Lightweight wall-clock latency measurement."""

    def __init__(self, label: str, logger: ComponentLogger) -> None:
        self._label = label
        self._logger = logger
        self._start: float = 0.0

    def __enter__(self) -> "LatencyTracker":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        elapsed_ms = (time.monotonic() - self._start) * 1000
        self._logger.debug(
            f"{self._label} completed",
            data={"latency_ms": round(elapsed_ms, 2)},
        )
