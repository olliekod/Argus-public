"""
Portfolio State
================

Concrete datamodel representing the portfolio's state at a specific point
in time.  Used by the RiskEngine to make deterministic clamping decisions
without lookahead.

All fields must be populated using data available at ``as_of_ts_ms``.

References
----------
- MASTER_PLAN.md §9 — Phase 5: Portfolio Risk Engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("argus.portfolio_state")


@dataclass
class PositionRecord:
    """A single existing position in the portfolio.

    Attributes
    ----------
    underlying : str
        Ticker / underlying symbol.
    instrument_type : str
        ``"equity"`` | ``"option_spread"`` | ``"option_single"``
    qty : float
        Number of shares or contracts.  Positive = long, negative = short.
    avg_price : float
        Average entry price per share/contract (0.0 if unknown).
    strategy_id : str
        Strategy that owns this position (empty if manual/legacy).
    greeks : dict
        Optional cached greeks for this position.
        Expected keys (when available): ``delta``, ``gamma``, ``vega``.
    meta : dict
        Arbitrary metadata (strike info, expiry, credit, etc.).
    """
    underlying: str
    instrument_type: str = "equity"
    qty: float = 0.0
    avg_price: float = 0.0
    strategy_id: str = ""
    greeks: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a specific point in time.

    Every field must reflect data available **at or before** ``as_of_ts_ms``.
    No future data may be included.

    Attributes
    ----------
    as_of_ts_ms : int
        Epoch milliseconds timestamp for this snapshot.
    equity_usd : float
        Total portfolio equity in USD.
    current_positions : list of PositionRecord
        Existing open positions.
    current_drawdown_pct : float
        Current drawdown from peak as a fraction (e.g., 0.05 = 5%).
    peak_equity_usd : float
        High-water mark equity (used to derive drawdown).
    rolling_portfolio_returns : dict
        Optional mapping of ``{date_str: return_pct}`` or similar
        structure for correlation / drawdown calculations.
        Only entries with date ≤ as_of_ts_ms should be included.
    strategy_return_series : dict
        Optional mapping ``{strategy_id: {date_str: return_pct}}``.
        Used for correlation-based cluster exposure control.
        Only entries with date ≤ as_of_ts_ms should be included.
    notes : dict
        Debug / metadata fields.  Must be deterministic.
    """
    as_of_ts_ms: int = 0
    equity_usd: float = 10_000.0
    current_positions: List[PositionRecord] = field(default_factory=list)
    current_drawdown_pct: float = 0.0
    peak_equity_usd: float = 10_000.0
    rolling_portfolio_returns: Dict[str, float] = field(default_factory=dict)
    strategy_return_series: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def positions_for_underlying(self, underlying: str) -> List[PositionRecord]:
        """Return positions matching the given underlying."""
        return [p for p in self.current_positions if p.underlying == underlying]

    def total_position_greeks(self) -> Dict[str, float]:
        """Aggregate greeks across all current positions.

        Returns dict with keys ``delta``, ``gamma``, ``vega`` (summed).
        Missing greeks on individual positions are treated as 0.
        """
        totals: Dict[str, float] = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
        for pos in self.current_positions:
            # Delta is sign-dependent (short = negative contribution).
            # Gamma and vega are always positive regardless of direction.
            totals["delta"] += pos.greeks.get("delta", 0.0) * pos.qty
            totals["gamma"] += pos.greeks.get("gamma", 0.0) * abs(pos.qty)
            totals["vega"] += pos.greeks.get("vega", 0.0) * abs(pos.qty)
        return totals


def build_portfolio_state_from_context(
    *,
    as_of_ts_ms: int,
    equity_usd: float,
    positions: Optional[List[Dict[str, Any]]] = None,
    peak_equity_usd: Optional[float] = None,
    rolling_returns: Optional[Dict[str, float]] = None,
    strategy_returns: Optional[Dict[str, Dict[str, float]]] = None,
) -> PortfolioState:
    """Convenience builder for PortfolioState from raw dicts.

    Parameters
    ----------
    as_of_ts_ms : int
        Epoch ms for the snapshot.
    equity_usd : float
        Current equity.
    positions : list of dict, optional
        Each dict should have at least ``underlying`` and ``qty``.
    peak_equity_usd : float, optional
        High-water mark.  Defaults to ``equity_usd``.
    rolling_returns : dict, optional
        ``{date_str: return_pct}``.
    strategy_returns : dict, optional
        ``{strategy_id: {date_str: return_pct}}``.

    Returns
    -------
    PortfolioState
    """
    peak = peak_equity_usd if peak_equity_usd is not None else equity_usd
    dd_pct = max(0.0, (peak - equity_usd) / peak) if peak > 0 else 0.0

    pos_records: List[PositionRecord] = []
    for raw in (positions or []):
        pos_records.append(PositionRecord(
            underlying=raw.get("underlying", "UNKNOWN"),
            instrument_type=raw.get("instrument_type", "equity"),
            qty=float(raw.get("qty", 0)),
            avg_price=float(raw.get("avg_price", 0)),
            strategy_id=raw.get("strategy_id", ""),
            greeks=raw.get("greeks", {}),
            meta=raw.get("meta", {}),
        ))

    return PortfolioState(
        as_of_ts_ms=as_of_ts_ms,
        equity_usd=equity_usd,
        current_positions=pos_records,
        current_drawdown_pct=dd_pct,
        peak_equity_usd=peak,
        rolling_portfolio_returns=dict(rolling_returns or {}),
        strategy_return_series=dict(strategy_returns or {}),
    )
