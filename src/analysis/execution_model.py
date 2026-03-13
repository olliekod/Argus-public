"""
Conservative Execution Model
=============================

Honest fill simulation based on recorded liquidity and spreads.

Philosophy
----------
Every backtest fill should be *worse* than what you'd get live so that
any edge that survives is real.

Fill rules
----------
- **Shorts (sell-to-open)**: Filled at ``bid − slippage``.
- **Longs  (buy-to-open)**:  Filled at ``ask + slippage``.
- **Close short (buy-to-close)**: Filled at ``ask + slippage``.
- **Close long  (sell-to-close)**: Filled at ``bid − slippage``.

Rejection reasons
-----------------
- ``ILLIQUID``    — ``bid_size`` or ``ask_size`` below minimum.
- ``STALE_QUOTE`` — quote timestamp too far from simulation time.
- ``ZERO_BID``    — bid is 0 (no market).
- ``CROSSED``     — ask < bid (quote error).
- ``SPREAD_WIDE`` — spread exceeds maximum fraction of mid.

All fills are recorded with a ``source="simulated"`` tag so they are
never confused with live executions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger("argus.execution_model")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExecutionConfig:
    """Tunable knobs for the conservative execution model.

    All thresholds are intentionally pessimistic to bias toward
    underestimating live performance.
    """

    # Fixed slippage in dollars per contract (additive after bid/ask)
    slippage_per_contract: float = 0.02

    # Minimum quote size (shares/contracts) to accept a fill
    min_bid_size: int = 1
    min_ask_size: int = 1

    # Maximum quote staleness relative to sim time (ms)
    max_stale_ms: int = 120_000   # 2 minutes

    # Maximum spread as fraction of mid-price to accept a fill
    max_spread_pct: float = 0.50  # 50 %

    # Commission per contract (one leg)
    commission_per_contract: float = 0.65

    # Whether to allow partial fills (False = all-or-nothing)
    allow_partial_fills: bool = False

    # Cost multiplier for slippage-sensitivity sweeps.
    # 1.0 = baseline; 1.25 = +25% costs; 1.50 = +50% costs.
    # Scales slippage_per_contract and commission_per_contract.
    cost_multiplier: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Fill / Rejection types
# ═══════════════════════════════════════════════════════════════════════════

class RejectReason(Enum):
    ILLIQUID = auto()
    STALE_QUOTE = auto()
    ZERO_BID = auto()
    CROSSED = auto()
    SPREAD_WIDE = auto()
    INSUFFICIENT_SIZE = auto()


@dataclass(frozen=True)
class Quote:
    """Point-in-time quote snapshot used for fill simulation.

    All prices are per-share / per-contract (not notional).

    .. note::

        ``recv_ts_ms`` is the local receipt timestamp and is the
        **preferred** freshness reference.  Provider timestamps
        (``quote_ts_ms``) are often zero from Tastytrade/DXLink.
    """
    bid: float
    ask: float
    bid_size: int = 0          # 0 = unknown
    ask_size: int = 0          # 0 = unknown
    quote_ts_ms: int = 0       # UTC epoch ms of this quote (provider)
    symbol: str = ""
    source: str = ""           # e.g. "alpaca", "tastytrade"
    recv_ts_ms: int = 0        # local receipt timestamp (epoch ms)


@dataclass(frozen=True)
class FillResult:
    """Result of an execution attempt.

    ``filled`` is True when the order was accepted.
    When rejected, ``reject_reason`` and ``reject_detail`` explain why.
    """
    filled: bool
    fill_price: float = 0.0       # effective fill (after slippage)
    raw_price: float = 0.0        # bid or ask before slippage
    slippage: float = 0.0
    commission: float = 0.0
    quantity: int = 0
    side: str = ""                # "BUY" or "SELL"
    reject_reason: Optional[RejectReason] = None
    reject_detail: str = ""
    sim_ts_ms: int = 0            # simulation time at fill
    quote_ts_ms: int = 0          # quote time used
    source: str = "simulated"


@dataclass
class ExecutionLedger:
    """Accumulates fill / reject statistics for a simulation run."""
    fills: List[FillResult] = field(default_factory=list)
    rejects: List[FillResult] = field(default_factory=list)

    # Running totals
    total_commission: float = 0.0
    total_slippage: float = 0.0
    fills_count: int = 0
    rejects_count: int = 0

    def record(self, result: FillResult) -> None:
        if result.filled:
            self.fills.append(result)
            self.fills_count += 1
            self.total_commission += result.commission
            self.total_slippage += abs(result.slippage) * result.quantity
        else:
            self.rejects.append(result)
            self.rejects_count += 1

    @property
    def fill_rate(self) -> float:
        total = self.fills_count + self.rejects_count
        return self.fills_count / total if total > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "fills": self.fills_count,
            "rejects": self.rejects_count,
            "fill_rate": round(self.fill_rate, 4),
            "total_commission": round(self.total_commission, 2),
            "total_slippage": round(self.total_slippage, 4),
            "reject_reasons": self._reject_breakdown(),
        }

    def _reject_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.rejects:
            key = r.reject_reason.name if r.reject_reason else "UNKNOWN"
            counts[key] = counts.get(key, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════

class ExecutionModel:
    """Conservative fill simulator for honest backtesting.

    Usage::

        model = ExecutionModel()
        result = model.attempt_fill(
            quote=Quote(bid=1.20, ask=1.35, bid_size=10, ask_size=12,
                        quote_ts_ms=sim_ts),
            side="SELL",
            quantity=2,
            sim_ts_ms=sim_ts,
        )
        if result.filled:
            effective_credit = result.fill_price * result.quantity * 100
    """

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        self._cfg = config or ExecutionConfig()
        self.ledger = ExecutionLedger()

    @property
    def config(self) -> ExecutionConfig:
        return self._cfg

    def reset(self) -> None:
        """Reset the ledger for a new simulation run."""
        self.ledger = ExecutionLedger()

    # ------------------------------------------------------------------
    # Core fill logic
    # ------------------------------------------------------------------

    def attempt_fill(
        self,
        quote: Quote,
        side: Literal["BUY", "SELL"],
        quantity: int,
        sim_ts_ms: int,
        *,
        multiplier: int = 100,
    ) -> FillResult:
        """Attempt to fill an order against *quote*.

        Parameters
        ----------
        quote : Quote
            Current best bid/ask.
        side : "BUY" | "SELL"
            Direction of the fill.
        quantity : int
            Number of contracts.
        sim_ts_ms : int
            Current simulation time in epoch ms.
        multiplier : int
            Contract multiplier (default 100 for equity options).

        Returns
        -------
        FillResult
        """
        # ── Pre-flight checks ────────────────────────────────────────
        reject = self._validate_quote(quote, side, quantity, sim_ts_ms)
        if reject is not None:
            self.ledger.record(reject)
            return reject

        # ── Compute fill price ───────────────────────────────────────
        # Apply cost_multiplier for slippage-sensitivity sweeps
        cm = self._cfg.cost_multiplier
        slip = self._cfg.slippage_per_contract * cm
        if side == "SELL":
            raw_price = quote.bid
            fill_price = raw_price - slip
        else:
            raw_price = quote.ask
            fill_price = raw_price + slip

        # Floor at zero (can't pay negative)
        fill_price = max(fill_price, 0.0)

        commission = self._cfg.commission_per_contract * cm * quantity

        result = FillResult(
            filled=True,
            fill_price=round(fill_price, 4),
            raw_price=round(raw_price, 4),
            slippage=round(slip, 4),
            commission=round(commission, 4),
            quantity=quantity,
            side=side,
            sim_ts_ms=sim_ts_ms,
            quote_ts_ms=quote.quote_ts_ms,
            source="simulated",
        )
        self.ledger.record(result)
        logger.debug(
            "FILL %s %d @ %.4f (raw=%.4f slip=%.4f) %s",
            side, quantity, fill_price, raw_price, slip, quote.symbol,
        )
        return result

    # ------------------------------------------------------------------
    # Spread fill helpers
    # ------------------------------------------------------------------

    def fill_spread(
        self,
        short_quote: Quote,
        long_quote: Quote,
        quantity: int,
        sim_ts_ms: int,
        *,
        multiplier: int = 100,
    ) -> Dict[str, Any]:
        """Fill a vertical spread (sell short leg, buy long leg).

        Returns a dict with ``"short_fill"``, ``"long_fill"``,
        ``"net_credit"``, and ``"filled"`` keys.

        If either leg is rejected the entire spread is rejected.
        """
        short_fill = self.attempt_fill(
            short_quote, "SELL", quantity, sim_ts_ms, multiplier=multiplier,
        )
        if not short_fill.filled:
            return {
                "filled": False,
                "short_fill": short_fill,
                "long_fill": None,
                "net_credit": 0.0,
                "reject_reason": short_fill.reject_detail or "short_leg_rejected",
            }

        long_fill = self.attempt_fill(
            long_quote, "BUY", quantity, sim_ts_ms, multiplier=multiplier,
        )
        if not long_fill.filled:
            return {
                "filled": False,
                "short_fill": short_fill,
                "long_fill": long_fill,
                "net_credit": 0.0,
                "reject_reason": long_fill.reject_detail or "long_leg_rejected",
            }

        net_credit = (short_fill.fill_price - long_fill.fill_price) * quantity * multiplier
        total_commission = short_fill.commission + long_fill.commission

        return {
            "filled": True,
            "short_fill": short_fill,
            "long_fill": long_fill,
            "net_credit": round(net_credit, 2),
            "total_commission": round(total_commission, 2),
            "net_credit_after_commission": round(net_credit - total_commission, 2),
        }

    def close_spread(
        self,
        short_quote: Quote,
        long_quote: Quote,
        quantity: int,
        sim_ts_ms: int,
        *,
        multiplier: int = 100,
    ) -> Dict[str, Any]:
        """Close a vertical spread (buy-to-close short, sell-to-close long).

        Returns a dict with ``"short_fill"``, ``"long_fill"``,
        ``"net_debit"``, and ``"filled"`` keys.
        """
        # Buy to close the short leg
        short_fill = self.attempt_fill(
            short_quote, "BUY", quantity, sim_ts_ms, multiplier=multiplier,
        )
        if not short_fill.filled:
            return {
                "filled": False,
                "short_fill": short_fill,
                "long_fill": None,
                "net_debit": 0.0,
                "reject_reason": short_fill.reject_detail or "short_close_rejected",
            }

        # Sell to close the long leg
        long_fill = self.attempt_fill(
            long_quote, "SELL", quantity, sim_ts_ms, multiplier=multiplier,
        )
        if not long_fill.filled:
            return {
                "filled": False,
                "short_fill": short_fill,
                "long_fill": long_fill,
                "net_debit": 0.0,
                "reject_reason": long_fill.reject_detail or "long_close_rejected",
            }

        net_debit = (short_fill.fill_price - long_fill.fill_price) * quantity * multiplier
        total_commission = short_fill.commission + long_fill.commission

        return {
            "filled": True,
            "short_fill": short_fill,
            "long_fill": long_fill,
            "net_debit": round(net_debit, 2),
            "total_commission": round(total_commission, 2),
            "net_debit_after_commission": round(net_debit + total_commission, 2),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_quote(
        self,
        quote: Quote,
        side: str,
        quantity: int,
        sim_ts_ms: int,
    ) -> Optional[FillResult]:
        """Return a rejected FillResult if the quote fails checks, else None."""

        def _reject(reason: RejectReason, detail: str) -> FillResult:
            logger.debug("REJECT %s %d %s: %s", side, quantity, quote.symbol, detail)
            return FillResult(
                filled=False,
                side=side,
                quantity=quantity,
                reject_reason=reason,
                reject_detail=detail,
                sim_ts_ms=sim_ts_ms,
                quote_ts_ms=quote.quote_ts_ms,
                source="simulated",
            )

        # Zero bid
        if quote.bid <= 0:
            return _reject(RejectReason.ZERO_BID, f"bid={quote.bid}")

        # Crossed market
        if quote.ask < quote.bid:
            return _reject(
                RejectReason.CROSSED,
                f"ask={quote.ask} < bid={quote.bid}",
            )

        # Spread width
        mid = (quote.bid + quote.ask) / 2.0
        if mid > 0:
            spread_pct = (quote.ask - quote.bid) / mid
            if spread_pct > self._cfg.max_spread_pct:
                return _reject(
                    RejectReason.SPREAD_WIDE,
                    f"spread={spread_pct:.2%} > max={self._cfg.max_spread_pct:.0%}",
                )

        # Staleness — prefer recv_ts_ms (receipt time), fall back to
        # provider timestamp.  Provider timestamps are often zero from
        # Tastytrade/DXLink, so receipt time is the reliable reference.
        freshness_ts = quote.recv_ts_ms if quote.recv_ts_ms > 0 else quote.quote_ts_ms
        if freshness_ts > 0 and sim_ts_ms > 0:
            age_ms = sim_ts_ms - freshness_ts
            if age_ms > self._cfg.max_stale_ms:
                return _reject(
                    RejectReason.STALE_QUOTE,
                    f"age={age_ms}ms > max={self._cfg.max_stale_ms}ms (ts_source={'recv' if quote.recv_ts_ms > 0 else 'provider'})",
                )

        # Liquidity (size checks)
        if side == "SELL" and quote.bid_size > 0:
            if quote.bid_size < self._cfg.min_bid_size:
                return _reject(
                    RejectReason.INSUFFICIENT_SIZE,
                    f"bid_size={quote.bid_size} < min={self._cfg.min_bid_size}",
                )
            if not self._cfg.allow_partial_fills and quote.bid_size < quantity:
                return _reject(
                    RejectReason.INSUFFICIENT_SIZE,
                    f"bid_size={quote.bid_size} < quantity={quantity}",
                )
        if side == "BUY" and quote.ask_size > 0:
            if quote.ask_size < self._cfg.min_ask_size:
                return _reject(
                    RejectReason.INSUFFICIENT_SIZE,
                    f"ask_size={quote.ask_size} < min={self._cfg.min_ask_size}",
                )
            if not self._cfg.allow_partial_fills and quote.ask_size < quantity:
                return _reject(
                    RejectReason.INSUFFICIENT_SIZE,
                    f"ask_size={quote.ask_size} < quantity={quantity}",
                )

        return None  # All checks passed
