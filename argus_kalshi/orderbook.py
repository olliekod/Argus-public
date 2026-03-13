"""
Kalshi orderbook maintained from WebSocket snapshots + deltas.

Design
------
Kalshi contracts have two sides: **yes** and **no**.  Each side has a
separate order-book of bids.  There are no explicit asks — instead:

    implied yes_ask = 100 - best_no_bid   (cents)
    implied no_ask  = 100 - best_yes_bid  (cents)

Data structure
--------------
Each side stores price levels in a ``dict[int, int]`` mapping
``price_cents → quantity`` (centi-contracts) for O(1) lookup/update, plus
a **sorted list** of active prices maintained with ``bisect`` for O(1)
best-price retrieval (best = last element for bids).

Kalshi orderbook depth is capped at ~50 levels, so the ``bisect`` O(log n)
insert/remove on a list of ≤50 elements is effectively O(1) in practice
and avoids the overhead of heap-based structures (which don't support
efficient arbitrary deletion).

Sequence handling
-----------------
Kalshi snapshots and deltas carry a ``seq``. In practice, on multi-market
subscriptions the observed sequence appears to be stream-wide rather than
strictly per-market, so per-ticker updates can legitimately jump forward as
other tickers interleave between them. We therefore require deltas to be
strictly monotonic for a given ticker (``seq > last_seq``), not consecutive.
Only duplicate or out-of-order deltas invalidate the book.

Fixed-point (``*_fp``) fields
-----------------------------
Quantities arrive as ``"123.00"`` strings.  We convert to integer
*centi-contracts* on ingest (``int(round(float(s) * 100))``).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .kalshi_subpenny import (
    parse_count_centicx,
    parse_level_price,
    parse_price_cents,
    parse_qty_centicx,
    parse_snapshot_level,
)
from .logging_utils import ComponentLogger

log = ComponentLogger("orderbook")


def _fp_to_centicx(fp_str: str) -> int:
    """Convert a Kalshi ``*_fp`` string to an integer centi-contract count."""
    return int(round(float(fp_str) * 100))


def _centicx_to_fp(val: int) -> str:
    """Convert centi-contracts back to ``"X.XX"`` format."""
    return f"{val / 100:.2f}"


# ---------------------------------------------------------------------------
#  One side of the book (yes bids or no bids)
# ---------------------------------------------------------------------------

class _BookSide:
    """Bid side for one contract side (yes or no).

    Prices are in integer cents [1, 99].
    Quantities are in integer centi-contracts.
    """

    __slots__ = ("_levels", "_sorted_prices")

    def __init__(self) -> None:
        self._levels: Dict[int, int] = {}       # price_cents → qty (centi-cx)
        self._sorted_prices: List[int] = []      # ascending; best bid = last

    def clear(self) -> None:
        self._levels.clear()
        self._sorted_prices.clear()

    def set_level(self, price_cents: int, qty_centicx: int) -> None:
        """Set quantity at *price_cents*.  If *qty_centicx* is 0, remove."""
        if qty_centicx <= 0:
            self.remove_level(price_cents)
            return

        if price_cents in self._levels:
            self._levels[price_cents] = qty_centicx
        else:
            self._levels[price_cents] = qty_centicx
            bisect.insort(self._sorted_prices, price_cents)

    def remove_level(self, price_cents: int) -> None:
        if price_cents in self._levels:
            del self._levels[price_cents]
            idx = bisect.bisect_left(self._sorted_prices, price_cents)
            if idx < len(self._sorted_prices) and self._sorted_prices[idx] == price_cents:
                self._sorted_prices.pop(idx)

    @property
    def best_bid_cents(self) -> int:
        """Highest bid price, or 0 if empty."""
        return self._sorted_prices[-1] if self._sorted_prices else 0

    @property
    def best_bid_qty(self) -> int:
        """Quantity (centi-contracts) at the best bid, or 0 if empty."""
        if not self._sorted_prices:
            return 0
        return self._levels.get(self._sorted_prices[-1], 0)

    @property
    def depth(self) -> int:
        return len(self._sorted_prices)

    def levels_snapshot(self) -> List[Tuple[int, int]]:
        """Return ``[(price_cents, qty_centicx), ...]`` sorted descending."""
        return [(p, self._levels[p]) for p in reversed(self._sorted_prices)]


# ---------------------------------------------------------------------------
#  Full orderbook for one market
# ---------------------------------------------------------------------------

@dataclass
class OrderBook:
    """Orderbook for a single Kalshi market ticker."""

    market_ticker: str
    yes_bids: _BookSide = field(default_factory=_BookSide)
    no_bids: _BookSide = field(default_factory=_BookSide)
    last_seq: int = -1
    valid: bool = False
    # True only after at least one snapshot has been applied.
    # Distinguishes "never initialised" from "was valid, then invalidated".
    has_snapshot: bool = False

    # -- snapshot / delta interface -----------------------------------------

    def apply_snapshot(self, snapshot: Dict, seq: int) -> None:
        """Reset the book from a full snapshot payload.

        Expected *snapshot* shape (Kalshi WS)::

            {
                "yes": [[price_cents_or_dollars, qty_fp_str], ...],
                "no":  [[price_cents_or_dollars, qty_fp_str], ...],
            }

        Price may be int (legacy cents), str (price_dollars), or dict with
        "price" / "price_dollars" (subpenny migration as of March 2026).
        """
        self.yes_bids.clear()
        self.no_bids.clear()

        yes_parsed = 0
        for level in snapshot.get("yes", []):
            parsed = parse_snapshot_level(level)
            if parsed is not None:
                price_cents, qty = parsed
                self.yes_bids.set_level(price_cents, qty)
                yes_parsed += 1

        no_parsed = 0
        for level in snapshot.get("no", []):
            parsed = parse_snapshot_level(level)
            if parsed is not None:
                price_cents, qty = parsed
                self.no_bids.set_level(price_cents, qty)
                no_parsed += 1

        if (yes_parsed == 0 and len(snapshot.get("yes", [])) > 0) or (
            no_parsed == 0 and len(snapshot.get("no", [])) > 0
        ):
            log.warning(
                "Orderbook snapshot parse mismatch — some levels dropped",
                data={
                    "ticker": self.market_ticker,
                    "yes_raw": len(snapshot.get("yes", [])),
                    "yes_parsed": yes_parsed,
                    "no_raw": len(snapshot.get("no", [])),
                    "no_parsed": no_parsed,
                    "first_yes": str(snapshot.get("yes", [None])[0])[:120],
                },
            )

        self.last_seq = seq
        self.valid = True
        self.has_snapshot = True

    def apply_delta(self, delta: Dict, seq: int) -> bool:
        """Apply an incremental delta. Returns False on duplicate/out-of-order seq.

        Expected *delta* shape::

            {
                "price": <int cents> or "price_dollars": "<str>",
                "delta": <qty> or "delta_fp": "<str>",  (signed: + add, - remove)
                "side": "yes" | "no",
            }

        Or a list of such entries. Prefers *_fp when present (March 2026).
        """
        if not self.valid:
            return False

        # Kalshi seq appears to be subscription-stream scoped, not per ticker.
        # Other markets can legitimately consume intermediate seq values, so
        # only reject non-monotonic updates for this ticker.
        if seq <= self.last_seq:
            self.valid = False
            return False

        entries = delta if isinstance(delta, list) else [delta]
        for entry in entries:
            side_str = entry.get("side", "yes")
            price_cents = parse_price_cents(entry, "price", "price_dollars", default=0)
            delta_qty = parse_count_centicx(entry, "delta", "delta_fp", default=0)
            book_side = self.yes_bids if side_str == "yes" else self.no_bids

            current = book_side._levels.get(price_cents, 0)
            new_qty = current + delta_qty
            book_side.set_level(price_cents, max(new_qty, 0))

        self.last_seq = seq
        return True

    # -- derived prices -----------------------------------------------------

    @property
    def best_yes_bid_cents(self) -> int:
        return self.yes_bids.best_bid_cents

    @property
    def best_no_bid_cents(self) -> int:
        return self.no_bids.best_bid_cents

    @property
    def implied_yes_ask_cents(self) -> int:
        """Best price to buy YES = 100 - best NO bid."""
        nb = self.best_no_bid_cents
        return (100 - nb) if nb > 0 else 100

    @property
    def implied_no_ask_cents(self) -> int:
        """Best price to buy NO = 100 - best YES bid."""
        yb = self.best_yes_bid_cents
        return (100 - yb) if yb > 0 else 100

    @property
    def spread_cents(self) -> int:
        """Bid-ask spread on the YES side."""
        return self.implied_yes_ask_cents - self.best_yes_bid_cents

    # -- microstructure metrics (Phase 2) ------------------------------------

    @property
    def order_book_imbalance(self) -> float:
        """Order-book imbalance (OBI) at the best level.

        Computed as ``(Q_yes_bid - Q_no_bid) / (Q_yes_bid + Q_no_bid)``.
        Positive values indicate buying pressure on YES; negative on NO.
        Returns 0.0 if both sides are empty.
        """
        q_yes = self.yes_bids.best_bid_qty
        q_no = self.no_bids.best_bid_qty
        total = q_yes + q_no
        if total == 0:
            return 0.0
        return (q_yes - q_no) / total

    @property
    def micro_price_cents(self) -> float:
        """Volume-weighted micro-price (YES side, in cents).

        Leans the mid-point between best YES bid and implied YES ask
        toward the side with more quantity at the touch::

            micro = yes_ask * Q_yes_bid / (Q_yes_bid + Q_no_bid)
                  + yes_bid * Q_no_bid  / (Q_yes_bid + Q_no_bid)

        Falls back to the simple midpoint if either side is empty.
        """
        yb = self.best_yes_bid_cents
        ya = self.implied_yes_ask_cents
        q_yes = self.yes_bids.best_bid_qty
        q_no = self.no_bids.best_bid_qty
        total = q_yes + q_no
        if total == 0 or yb == 0:
            return (yb + ya) / 2.0
        return (ya * q_yes + yb * q_no) / total

    @property
    def best_yes_bid_depth(self) -> int:
        """Quantity (centi-contracts) available at the best YES bid."""
        return self.yes_bids.best_bid_qty

    @property
    def best_no_bid_depth(self) -> int:
        """Quantity (centi-contracts) available at the best NO bid."""
        return self.no_bids.best_bid_qty

    def invalidate(self) -> None:
        self.valid = False

    def summary(self) -> Dict:
        return {
            "ticker": self.market_ticker,
            "yes_bid": self.best_yes_bid_cents,
            "no_bid": self.best_no_bid_cents,
            "yes_ask": self.implied_yes_ask_cents,
            "no_ask": self.implied_no_ask_cents,
            "obi": round(self.order_book_imbalance, 4),
            "micro_price": round(self.micro_price_cents, 2),
            "seq": self.last_seq,
            "valid": self.valid,
            "has_snapshot": self.has_snapshot,
        }
