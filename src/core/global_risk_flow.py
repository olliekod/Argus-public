"""
Global Risk Flow
================

Composite global risk appetite signal computed from international
ETF daily returns and FX movements.  Used by the Overnight Session
Strategy as an optional gating signal.

The metric is a weighted average of regional return signals:

    GlobalRiskFlow = 0.4 * AsiaReturn + 0.4 * EuropeReturn + 0.2 * FXRiskSignal

Components
----------
- **AsiaReturn**: Average daily return of EWJ, FXI, EWT, EWY, INDA
- **EuropeReturn**: Average daily return of EWG, EWU, FEZ, EWL
- **FXRiskSignal**: USDJPY daily return (positive = risk-on, yen weakening)

Timestamping
------------
Daily bars are normalized to 00:00 UTC of the trading date.  In replay,
only bars with ``bar_ts < sim_time`` are used (strictly less-than — most
recent *completed* daily bar, no lookahead into today).

Weights redistribute proportionally when components are missing.
Returns ``None`` if no components are available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.global_risk_flow")

# ═══════════════════════════════════════════════════════════════════════════
# Symbol groupings
# ═══════════════════════════════════════════════════════════════════════════

ASIA_SYMBOLS = ("EWJ", "FXI", "EWT", "EWY", "INDA")
EUROPE_SYMBOLS = ("EWG", "EWU", "FEZ", "EWL")
FX_RISK_SYMBOL = "FX:USDJPY"  # positive return = yen weakening = risk-on

# Base weights (must sum to 1.0)
_W_ASIA = 0.4
_W_EUROPE = 0.4
_W_FX = 0.2


# ═══════════════════════════════════════════════════════════════════════════
# Daily return computation
# ═══════════════════════════════════════════════════════════════════════════


def _latest_daily_return(
    bars: List[Dict[str, Any]],
    sim_time_ms: int,
) -> Optional[float]:
    """Compute the most recent completed daily return from a bar list.

    Bars are assumed to have ``timestamp_ms`` (00:00 UTC of the day)
    and ``close``.  Only bars with ``timestamp_ms < sim_time_ms``
    are considered (strict less-than: no lookahead).

    Returns the daily return (close-to-close) of the most recent
    completed pair, or ``None`` if fewer than 2 qualifying bars exist.
    """
    # Filter and sort eligible bars (strictly before sim_time)
    eligible = [
        b for b in bars
        if b.get("timestamp_ms", 0) < sim_time_ms
    ]
    if len(eligible) < 2:
        return None

    # Sort by timestamp descending, take the two most recent
    eligible.sort(key=lambda b: b["timestamp_ms"], reverse=True)
    latest = eligible[0]
    prev = eligible[1]

    close_now = latest.get("close", 0)
    close_prev = prev.get("close", 0)

    if close_prev == 0:
        return None

    return (close_now - close_prev) / close_prev


def _avg_return(
    bars_by_symbol: Dict[str, List[Dict[str, Any]]],
    symbols: Tuple[str, ...],
    sim_time_ms: int,
) -> Optional[float]:
    """Average daily return across a set of symbols.

    Skips symbols with no available data.  Returns ``None`` if
    no symbols have valid returns.
    """
    returns = []
    for sym in symbols:
        sym_bars = bars_by_symbol.get(sym, [])
        ret = _latest_daily_return(sym_bars, sim_time_ms)
        if ret is not None:
            returns.append(ret)

    if not returns:
        return None
    return sum(returns) / len(returns)


# ═══════════════════════════════════════════════════════════════════════════
# Main computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_global_risk_flow(
    bars_by_symbol: Dict[str, List[Dict[str, Any]]],
    sim_time_ms: int,
) -> Optional[float]:
    """Compute the GlobalRiskFlow composite signal.

    Args:
        bars_by_symbol: Maps symbol → list of daily bar dicts.
            Each bar dict must have ``timestamp_ms`` and ``close``.
        sim_time_ms: Current simulation timestamp (ms).  Only bars
            with ``timestamp_ms < sim_time_ms`` are used.

    Returns:
        The weighted risk-flow signal, or ``None`` if no components
        are available.  Positive = risk-on, negative = risk-off.
    """
    asia_ret = _avg_return(bars_by_symbol, ASIA_SYMBOLS, sim_time_ms)
    europe_ret = _avg_return(bars_by_symbol, EUROPE_SYMBOLS, sim_time_ms)
    fx_ret = _latest_daily_return(
        bars_by_symbol.get(FX_RISK_SYMBOL, []),
        sim_time_ms,
    )

    # Collect available components with their base weights
    components: List[Tuple[float, float]] = []
    if asia_ret is not None:
        components.append((_W_ASIA, asia_ret))
    if europe_ret is not None:
        components.append((_W_EUROPE, europe_ret))
    if fx_ret is not None:
        components.append((_W_FX, fx_ret))

    if not components:
        logger.debug(
            "No risk-flow components available at sim_time=%d",
            sim_time_ms,
        )
        return None

    # Redistribute weights proportionally
    total_weight = sum(w for w, _ in components)
    risk_flow = sum((w / total_weight) * ret for w, ret in components)

    logger.debug(
        "GlobalRiskFlow = %.6f (asia=%s europe=%s fx=%s) at %d",
        risk_flow,
        f"{asia_ret:.6f}" if asia_ret is not None else "N/A",
        f"{europe_ret:.6f}" if europe_ret is not None else "N/A",
        f"{fx_ret:.6f}" if fx_ret is not None else "N/A",
        sim_time_ms,
    )
    return risk_flow
