"""
Status helpers for Argus observability (Status v2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def update_ema(prev: Optional[float], value: float, alpha: float = 0.2) -> float:
    if prev is None:
        return value
    return (value * alpha) + (prev * (1 - alpha))


@dataclass(slots=True)
class StatusSnapshot:
    """Standardized status payload for connectors and internal components."""

    name: str
    type: str
    status: str = "unknown"
    last_success_ts: Optional[str] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    reconnect_attempts: int = 0
    request_count: int = 0
    error_count: int = 0
    avg_latency_ms: Optional[float] = None
    last_latency_ms: Optional[float] = None
    last_message_ts: Optional[str] = None
    last_poll_ts: Optional[str] = None
    age_seconds: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "status": self.status,
            "last_success_ts": self.last_success_ts,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
            "counters": {
                "reconnect_attempts": self.reconnect_attempts,
                "request_count": self.request_count,
                "error_count": self.error_count,
            },
            "timing": {
                "avg_latency_ms": self.avg_latency_ms,
                "last_latency_ms": self.last_latency_ms,
            },
            "staleness": {
                "last_message_ts": self.last_message_ts,
                "last_poll_ts": self.last_poll_ts,
                "age_seconds": self.age_seconds,
            },
            "extras": self.extras,
        }


def build_status(
    *,
    name: str,
    type: str,
    status: str = "unknown",
    last_success_ts: Optional[float] = None,
    last_error: Optional[str] = None,
    consecutive_failures: int = 0,
    reconnect_attempts: int = 0,
    request_count: int = 0,
    error_count: int = 0,
    avg_latency_ms: Optional[float] = None,
    last_latency_ms: Optional[float] = None,
    last_message_ts: Optional[float] = None,
    last_poll_ts: Optional[float] = None,
    age_seconds: Optional[float] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = StatusSnapshot(
        name=name,
        type=type,
        status=status,
        last_success_ts=utc_iso(last_success_ts),
        last_error=last_error,
        consecutive_failures=consecutive_failures,
        reconnect_attempts=reconnect_attempts,
        request_count=request_count,
        error_count=error_count,
        avg_latency_ms=avg_latency_ms,
        last_latency_ms=last_latency_ms,
        last_message_ts=utc_iso(last_message_ts),
        last_poll_ts=utc_iso(last_poll_ts),
        age_seconds=age_seconds,
        extras=extras or {},
    )
    return snapshot.to_dict()

