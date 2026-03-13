"""
Session schedule helpers.

Provides deterministic session classification and timing utilities
based solely on timestamps (no wall clock).

DST handling
------------
Equities session boundaries are defined in US/Eastern time.  This module
uses ``zoneinfo`` (stdlib ≥ 3.9) to convert UTC timestamps to Eastern
time before comparing against session hours so the boundaries shift
automatically with spring-forward / fall-back.

Crypto boundaries remain fixed UTC because crypto markets are 24/7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Timezone constants
# ---------------------------------------------------------------------------
_ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Equities session boundaries (Eastern Time, not UTC)
# ---------------------------------------------------------------------------
_EQ_PRE_OPEN_HOUR = 4        # Pre-market opens 04:00 ET
_EQ_PRE_OPEN_MINUTE = 0
_EQ_RTH_OPEN_HOUR = 9        # RTH opens 09:30 ET
_EQ_RTH_OPEN_MINUTE = 30
_EQ_RTH_CLOSE_HOUR = 16      # RTH closes 16:00 ET
_EQ_RTH_CLOSE_MINUTE = 0
_EQ_POST_CLOSE_HOUR = 20     # Post-market closes 20:00 ET
_EQ_POST_CLOSE_MINUTE = 0

# ---------------------------------------------------------------------------
# Crypto session boundaries (fixed UTC — no DST adjustment needed)
# ---------------------------------------------------------------------------
CRYPTO_ASIA_START = 0       # 00:00 UTC
CRYPTO_ASIA_END = 8         # 08:00 UTC
CRYPTO_EU_START = 8         # 08:00 UTC
CRYPTO_EU_END = 14          # 14:00 UTC
CRYPTO_US_START = 14        # 14:00 UTC
CRYPTO_US_END = 22          # 22:00 UTC

# ---------------------------------------------------------------------------
# Legacy hardcoded UTC constants — kept as *approximate EST defaults* for
# any code that imports them directly.  New callers should use the
# timezone-aware helpers below.
# ---------------------------------------------------------------------------
EQUITIES_PRE_START = 9      # ≈ 4 AM ET in EST
EQUITIES_RTH_START = 14     # ≈ 9:30 AM ET in EST (simplified to hour)
EQUITIES_RTH_END = 21       # ≈ 4 PM ET in EST (simplified to hour)
EQUITIES_POST_END = 1       # ≈ 8 PM ET in EST = 1 UTC next day


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """Session window bounds in minutes from UTC day start."""

    start_minute: int
    end_minute: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _et_from_ts_ms(ts_ms: int) -> datetime:
    """Convert epoch-ms to an aware Eastern-Time datetime."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(_ET)


def _et_minute_of_day(dt_et: datetime) -> int:
    """Minutes since midnight in Eastern Time."""
    return dt_et.hour * 60 + dt_et.minute


def _hour_from_ts_ms(ts_ms: int) -> int:
    """UTC hour from epoch-ms (used for crypto boundaries)."""
    seconds = ts_ms // 1000
    return (seconds // 3600) % 24


def _minute_of_day_from_ts_ms(ts_ms: int) -> int:
    """UTC minute-of-day from epoch-ms."""
    seconds = ts_ms // 1000
    return (seconds // 60) % (24 * 60)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_session_regime(market: str, ts_ms: int) -> str:
    """Determine session regime from timestamp (no wall clock).

    For equities the timestamp is converted to Eastern Time so DST
    transitions are handled automatically.  Crypto boundaries remain
    fixed UTC.
    """
    if market == "EQUITIES":
        dt_et = _et_from_ts_ms(ts_ms)
        minute = _et_minute_of_day(dt_et)

        rth_open = _EQ_RTH_OPEN_HOUR * 60 + _EQ_RTH_OPEN_MINUTE   # 09:30 → 570
        rth_close = _EQ_RTH_CLOSE_HOUR * 60 + _EQ_RTH_CLOSE_MINUTE  # 16:00 → 960
        pre_open = _EQ_PRE_OPEN_HOUR * 60 + _EQ_PRE_OPEN_MINUTE   # 04:00 → 240
        post_close = _EQ_POST_CLOSE_HOUR * 60 + _EQ_POST_CLOSE_MINUTE  # 20:00 → 1200

        if rth_open <= minute < rth_close:
            return "RTH"
        if pre_open <= minute < rth_open:
            return "PRE"
        if rth_close <= minute < post_close:
            return "POST"
        return "CLOSED"

    # CRYPTO — fixed UTC
    hour = _hour_from_ts_ms(ts_ms)
    if CRYPTO_ASIA_START <= hour < CRYPTO_ASIA_END:
        return "ASIA"
    if CRYPTO_EU_START <= hour < CRYPTO_EU_END:
        return "EU"
    if CRYPTO_US_START <= hour < CRYPTO_US_END:
        return "US"
    return "OFFPEAK"


def get_equities_rth_window_minutes(ts_ms: int | None = None) -> SessionWindow:
    """Return RTH window in minutes from UTC day start.

    When *ts_ms* is provided the boundaries are computed from the actual
    Eastern Time offset on that date (i.e. DST-aware).  When omitted the
    function falls back to the legacy EST approximation for backward
    compatibility.
    """
    if ts_ms is not None:
        dt_et = _et_from_ts_ms(ts_ms)
        utc_offset_minutes = int(dt_et.utcoffset().total_seconds() // 60)
        # Convert ET boundaries to UTC minutes-of-day
        start_et = _EQ_RTH_OPEN_HOUR * 60 + _EQ_RTH_OPEN_MINUTE    # 570
        end_et = _EQ_RTH_CLOSE_HOUR * 60 + _EQ_RTH_CLOSE_MINUTE    # 960
        return SessionWindow(
            start_minute=start_et - utc_offset_minutes,
            end_minute=end_et - utc_offset_minutes,
        )
    # Legacy fallback (approximate EST)
    return SessionWindow(
        start_minute=EQUITIES_RTH_START * 60,
        end_minute=EQUITIES_RTH_END * 60,
    )


def minutes_from_session_end(ts_ms: int, window: SessionWindow) -> int:
    """
    Minutes from timestamp to session end. Returns negative if after end.
    Assumes window does not cross midnight.
    """
    minute_of_day = _minute_of_day_from_ts_ms(ts_ms)
    return window.end_minute - minute_of_day


def is_within_last_n_minutes(ts_ms: int, window: SessionWindow, n_minutes: int) -> bool:
    """Return True if timestamp is within the last N minutes of the window."""
    if n_minutes <= 0:
        return False
    remaining = minutes_from_session_end(ts_ms, window)
    return 0 <= remaining <= n_minutes

