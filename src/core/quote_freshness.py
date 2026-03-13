"""
Quote Freshness Utilities
=========================

Receipt-time-based freshness checks for market data.

Tastytrade/DXLink often delivers option quotes with ``timestamp=0``.
Greeks events carry timestamps but are delayed ~50-120 seconds.
Therefore, freshness **must** be based on receipt time (``recv_ts_ms``),
not provider timestamps.

Usage::

    from src.core.quote_freshness import is_quote_fresh, effective_greeks_timestamp

    if is_quote_fresh(quote.recv_ts_ms, max_age_ms=120_000):
        # quote is usable
        ...

    ts = effective_greeks_timestamp(greeks_event)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("argus.quote_freshness")


# ═══════════════════════════════════════════════════════════════════════════
# Quote Freshness
# ═══════════════════════════════════════════════════════════════════════════

def now_ms() -> int:
    """Current UTC time in epoch milliseconds."""
    return int(time.time() * 1000)


def is_quote_fresh(
    recv_ts_ms: int,
    max_age_ms: int = 120_000,
    *,
    reference_ms: Optional[int] = None,
) -> bool:
    """Check whether a quote is fresh based on **receipt** timestamp.

    Parameters
    ----------
    recv_ts_ms : int
        Epoch ms when the quote was received locally.
    max_age_ms : int
        Maximum allowed age in ms (default 120 000 = 2 minutes).
    reference_ms : int, optional
        Reference time for comparison.  Defaults to ``now_ms()``.
        Use this in replay/simulation to pass ``sim_ts_ms``.

    Returns
    -------
    bool
        True if the quote is within the freshness window.
    """
    if recv_ts_ms <= 0:
        return False
    ref = reference_ms if reference_ms is not None else now_ms()
    age = ref - recv_ts_ms
    return age <= max_age_ms


def quote_age_ms(
    recv_ts_ms: int,
    *,
    reference_ms: Optional[int] = None,
) -> int:
    """Return the age of a quote in milliseconds.

    Returns -1 if recv_ts_ms is invalid (zero or negative).
    """
    if recv_ts_ms <= 0:
        return -1
    ref = reference_ms if reference_ms is not None else now_ms()
    return ref - recv_ts_ms


# ═══════════════════════════════════════════════════════════════════════════
# Greeks Freshness
# ═══════════════════════════════════════════════════════════════════════════

def effective_greeks_timestamp(
    greeks_event: Dict[str, Any],
    *,
    event_ts_key: str = "event_ts_ms",
    recv_ts_key: str = "recv_ts_ms",
) -> tuple[int, str]:
    """Determine the best available timestamp for a Greeks event.

    Priority:
    1. Event timestamp (from provider) — if present and positive.
    2. Receipt timestamp — fallback when provider timestamp is missing.

    Returns
    -------
    (timestamp_ms, source) where source is ``"event"`` or ``"receipt"``.
    """
    event_ts = greeks_event.get(event_ts_key, 0) or 0
    recv_ts = greeks_event.get(recv_ts_key, 0) or 0

    if isinstance(event_ts, (int, float)) and event_ts > 0:
        return int(event_ts), "event"
    if isinstance(recv_ts, (int, float)) and recv_ts > 0:
        return int(recv_ts), "receipt"

    # Nothing usable
    return 0, "none"


def is_greeks_fresh(
    greeks_event: Dict[str, Any],
    max_age_ms: int = 120_000,
    *,
    reference_ms: Optional[int] = None,
) -> bool:
    """Check whether a Greeks event is fresh.

    Uses :func:`effective_greeks_timestamp` to pick the best timestamp,
    then applies the same age check as :func:`is_quote_fresh`.
    """
    ts, _source = effective_greeks_timestamp(greeks_event)
    if ts <= 0:
        return False
    return is_quote_fresh(ts, max_age_ms, reference_ms=reference_ms)


# ═══════════════════════════════════════════════════════════════════════════
# Staleness Summary (for health audits / dashboards)
# ═══════════════════════════════════════════════════════════════════════════

def freshness_summary(
    recv_ts_list: list[int],
    *,
    reference_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute freshness statistics over a collection of recv timestamps.

    Returns a dict with p50, p95, max age, and count of stale entries.
    """
    ref = reference_ms if reference_ms is not None else now_ms()
    ages = [ref - ts for ts in recv_ts_list if ts > 0]
    if not ages:
        return {
            "count": 0,
            "valid_count": 0,
            "age_p50_ms": None,
            "age_p95_ms": None,
            "age_max_ms": None,
            "stale_count_120s": 0,
        }
    ages.sort()
    n = len(ages)
    return {
        "count": len(recv_ts_list),
        "valid_count": n,
        "age_p50_ms": ages[n // 2],
        "age_p95_ms": ages[min(n - 1, int(0.95 * n))],
        "age_max_ms": ages[-1],
        "stale_count_120s": sum(1 for a in ages if a > 120_000),
    }
