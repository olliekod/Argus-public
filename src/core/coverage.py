"""
Argus Coverage Diagnostics
==========================

Pure-function module for computing uptime, gap analysis, and coverage
diagnostics.  All functions operate on plain lists of int timestamps (ms)
so they can be tested without DB access.

Phase 4A.1+ — does NOT modify outcome computation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE


# ═══════════════════════════════════════════════════════════════════════════════
#  A) Uptime from heartbeats
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class UptimeResult:
    """Result of uptime computation from heartbeat timestamps."""
    wall_span_s: int          # total wall-clock span in seconds
    uptime_s: int             # estimated seconds the system was ON
    downtime_s: int           # estimated seconds the system was OFF
    heartbeat_count: int      # number of heartbeat records
    gap_threshold_ms: int     # threshold used to detect OFF intervals
    median_cadence_ms: int = 0  # median delta between consecutive heartbeats
    off_intervals: List[Tuple[int, int]] = field(default_factory=list)
    """List of (start_ms, end_ms) intervals where system was OFF."""


def compute_uptime(
    heartbeat_ms: List[int],
    start_ms: int,
    end_ms: int,
    gap_threshold_ms: int = 120_000,  # 2× default 60s heartbeat cadence
) -> UptimeResult:
    """Compute uptime from a sorted list of heartbeat timestamps.

    Algorithm:
    - Walk consecutive heartbeat pairs.
    - If delta <= gap_threshold_ms: the system was ON for that interval.
    - If delta > gap_threshold_ms: the system was OFF.  The OFF interval
      starts at ``hb[i] + gap_threshold_ms`` (allow for one missed beat)
      and ends at ``hb[i+1]``.

    Parameters
    ----------
    heartbeat_ms : list of int
        Sorted heartbeat timestamps in epoch milliseconds.
    start_ms, end_ms : int
        The wall-clock range to report on.
    gap_threshold_ms : int
        Maximum gap between heartbeats before declaring the system OFF.

    Returns
    -------
    UptimeResult
    """
    wall_span_s = max(0, (end_ms - start_ms)) // 1000

    if len(heartbeat_ms) < 2:
        # Cannot determine uptime from < 2 heartbeats
        return UptimeResult(
            wall_span_s=wall_span_s,
            uptime_s=0,
            downtime_s=wall_span_s,
            heartbeat_count=len(heartbeat_ms),
            gap_threshold_ms=gap_threshold_ms,
            median_cadence_ms=0,
            off_intervals=[(start_ms, end_ms)] if wall_span_s > 0 else [],
        )

    off_intervals: List[Tuple[int, int]] = []
    uptime_ms = 0
    deltas: List[int] = []

    for i in range(len(heartbeat_ms) - 1):
        delta = heartbeat_ms[i + 1] - heartbeat_ms[i]
        deltas.append(delta)
        if delta <= gap_threshold_ms:
            uptime_ms += delta
        else:
            # Count up to threshold as ON, rest is OFF
            uptime_ms += gap_threshold_ms
            off_start = heartbeat_ms[i] + gap_threshold_ms
            off_end = heartbeat_ms[i + 1]
            off_intervals.append((off_start, off_end))

    # Compute median cadence from deltas that are within threshold (normal beats)
    normal_deltas = sorted(d for d in deltas if d <= gap_threshold_ms)
    if normal_deltas:
        mid = len(normal_deltas) // 2
        median_cadence = normal_deltas[mid]
    else:
        # All gaps exceed threshold; use raw median of all deltas
        sorted_deltas = sorted(deltas)
        mid = len(sorted_deltas) // 2
        median_cadence = sorted_deltas[mid]

    uptime_s = uptime_ms // 1000
    downtime_s = max(0, wall_span_s - uptime_s)

    return UptimeResult(
        wall_span_s=wall_span_s,
        uptime_s=uptime_s,
        downtime_s=downtime_s,
        heartbeat_count=len(heartbeat_ms),
        gap_threshold_ms=gap_threshold_ms,
        median_cadence_ms=median_cadence,
        off_intervals=off_intervals,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  B) Bar gap analysis
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GapInfo:
    """A single gap between consecutive bars."""
    start_ms: int   # timestamp of bar before gap
    end_ms: int     # timestamp of bar after gap
    gap_seconds: int


@dataclass
class BarContinuityResult:
    """Continuity statistics for a bar series."""
    bar_count: int
    span_seconds: int
    expected_bars: int
    coverage_pct: float
    max_gap_seconds: int
    gap_count_above_threshold: int
    gap_threshold_seconds: int
    top_gaps: List[GapInfo] = field(default_factory=list)


def analyze_bar_continuity(
    bar_timestamps_ms: List[int],
    bar_duration_seconds: int,
    gap_threshold_seconds: int = 300,
    top_n: int = 10,
) -> BarContinuityResult:
    """Analyze bar timestamp continuity.

    Parameters
    ----------
    bar_timestamps_ms : list of int
        Sorted bar open timestamps in epoch ms.
    bar_duration_seconds : int
        Expected cadence between bars (e.g. 60 for 1m).
    gap_threshold_seconds : int
        Threshold above which gaps are counted / flagged.
    top_n : int
        How many largest gaps to keep.

    Returns
    -------
    BarContinuityResult
    """
    n = len(bar_timestamps_ms)
    if n < 2:
        return BarContinuityResult(
            bar_count=n,
            span_seconds=0,
            expected_bars=n,
            coverage_pct=100.0 if n else 0.0,
            max_gap_seconds=0,
            gap_count_above_threshold=0,
            gap_threshold_seconds=gap_threshold_seconds,
        )

    span_ms = bar_timestamps_ms[-1] - bar_timestamps_ms[0]
    span_s = span_ms // 1000
    expected = (span_s // bar_duration_seconds) + 1
    coverage = (n / expected * 100.0) if expected > 0 else 0.0

    # Compute all gaps
    gaps: List[GapInfo] = []
    max_gap_s = 0
    big_gap_count = 0

    for i in range(n - 1):
        delta_ms = bar_timestamps_ms[i + 1] - bar_timestamps_ms[i]
        delta_s = delta_ms // 1000
        if delta_s > bar_duration_seconds:
            gap = GapInfo(
                start_ms=bar_timestamps_ms[i],
                end_ms=bar_timestamps_ms[i + 1],
                gap_seconds=delta_s,
            )
            gaps.append(gap)
            if delta_s > max_gap_s:
                max_gap_s = delta_s
            if delta_s >= gap_threshold_seconds:
                big_gap_count += 1

    # Keep top N by size
    gaps.sort(key=lambda g: g.gap_seconds, reverse=True)
    top_gaps = gaps[:top_n]

    return BarContinuityResult(
        bar_count=n,
        span_seconds=span_s,
        expected_bars=expected,
        coverage_pct=round(coverage, 2),
        max_gap_seconds=max_gap_s,
        gap_count_above_threshold=big_gap_count,
        gap_threshold_seconds=gap_threshold_seconds,
        top_gaps=top_gaps,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  C) Diagnose: uptime-aware bar coverage
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DiagnoseResult:
    """Full diagnosis combining uptime + bar coverage."""
    wall_span_s: int
    uptime_s: int
    downtime_s: int
    uptime_pct: float
    bars_total: int
    bars_during_uptime: int
    bars_during_downtime: int
    expected_bars_during_uptime: int
    uptime_coverage_pct: float
    missing_during_uptime: int        # actionable: feed/ingestion issue
    missing_during_downtime: int      # expected: Argus was OFF
    off_intervals: List[Tuple[int, int]] = field(default_factory=list)
    top_uptime_gaps: List[GapInfo] = field(default_factory=list)
    is_equity: bool = False
    equity_session_note: str = ""


def _intervals_overlap(a_start: int, a_end: int,
                       b_start: int, b_end: int) -> int:
    """Return overlap in ms between two intervals, or 0."""
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    return max(0, overlap_end - overlap_start)


def diagnose_coverage(
    bar_timestamps_ms: List[int],
    bar_duration_seconds: int,
    uptime: UptimeResult,
    start_ms: int,
    end_ms: int,
    is_equity: bool = False,
) -> DiagnoseResult:
    """Combine uptime info with bar timestamps to produce a diagnosis.

    Bars that fall inside OFF intervals are attributed to "Argus was OFF"
    rather than "feed missing while ON".
    """
    wall_span_s = max(0, (end_ms - start_ms)) // 1000
    bar_dur_ms = bar_duration_seconds * 1000

    # Count bars that fall during uptime (i.e. NOT inside any OFF interval)
    off_intervals = uptime.off_intervals

    def _is_during_off(ts_ms: int) -> bool:
        for off_start, off_end in off_intervals:
            if off_start <= ts_ms < off_end:
                return True
        return False

    bars_during_uptime = 0
    bars_during_downtime = 0
    for ts in bar_timestamps_ms:
        if _is_during_off(ts):
            bars_during_downtime += 1
        else:
            bars_during_uptime += 1

    # Expected bars during uptime: uptime_s / bar_duration_seconds
    expected_during_uptime = uptime.uptime_s // bar_duration_seconds if bar_duration_seconds > 0 else 0
    uptime_cov = (bars_during_uptime / expected_during_uptime * 100.0) if expected_during_uptime > 0 else 0.0
    missing_during_uptime = max(0, expected_during_uptime - bars_during_uptime)

    # Expected bars during downtime (for informational purposes)
    expected_during_downtime = uptime.downtime_s // bar_duration_seconds if bar_duration_seconds > 0 else 0
    missing_during_downtime = max(0, expected_during_downtime - bars_during_downtime)

    # Find gaps that occur during uptime
    uptime_gaps: List[GapInfo] = []
    n = len(bar_timestamps_ms)
    for i in range(n - 1):
        delta_ms = bar_timestamps_ms[i + 1] - bar_timestamps_ms[i]
        delta_s = delta_ms // 1000
        if delta_s > bar_duration_seconds:
            gap_start = bar_timestamps_ms[i]
            gap_end = bar_timestamps_ms[i + 1]
            # Check how much of this gap overlaps with OFF time
            off_overlap_ms = 0
            for off_start, off_end in off_intervals:
                off_overlap_ms += _intervals_overlap(gap_start, gap_end,
                                                     off_start, off_end)
            # If more than half the gap is during uptime, count it
            gap_during_uptime_ms = (gap_end - gap_start) - off_overlap_ms
            if gap_during_uptime_ms > bar_dur_ms:
                uptime_gaps.append(GapInfo(
                    start_ms=gap_start,
                    end_ms=gap_end,
                    gap_seconds=delta_s,
                ))

    uptime_gaps.sort(key=lambda g: g.gap_seconds, reverse=True)

    uptime_pct = (uptime.uptime_s / wall_span_s * 100.0) if wall_span_s > 0 else 0.0

    # Equity session note
    note = ""
    if is_equity:
        note = ("Equity/ETF: overnight + weekend gaps are expected session "
                "closures (US equities 9:30-16:00 ET weekdays). "
                "Missing-during-uptime metrics may overcount due to "
                "non-trading hours.")

    return DiagnoseResult(
        wall_span_s=wall_span_s,
        uptime_s=uptime.uptime_s,
        downtime_s=uptime.downtime_s,
        uptime_pct=round(uptime_pct, 1),
        bars_total=len(bar_timestamps_ms),
        bars_during_uptime=bars_during_uptime,
        bars_during_downtime=bars_during_downtime,
        expected_bars_during_uptime=expected_during_uptime,
        uptime_coverage_pct=round(uptime_cov, 1),
        missing_during_uptime=missing_during_uptime,
        missing_during_downtime=missing_during_downtime,
        off_intervals=off_intervals,
        top_uptime_gaps=uptime_gaps[:10],
        is_equity=is_equity,
        equity_session_note=note,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  D) Equity session heuristic
# ═══════════════════════════════════════════════════════════════════════════════

# Known equity/ETF providers whose symbols trade US market hours only
EQUITY_PROVIDERS = {"yahoo", "alpaca"}

# Known equity/ETF symbols (extend as needed)
EQUITY_SYMBOLS = set(LIQUID_ETF_UNIVERSE) | {"IBIT", "BITO", "NVDA"}


def is_likely_equity(provider: str, symbol: str) -> bool:
    """Heuristic: return True if this provider/symbol pair is likely
    an equity/ETF with limited trading hours."""
    if provider.lower() in EQUITY_PROVIDERS:
        return True
    if symbol.upper() in EQUITY_SYMBOLS:
        return True
    return False
