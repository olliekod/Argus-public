"""
Bus message dataclasses for the Argus Kalshi module.

Every message on the internal pub/sub bus is an instance of one of these
dataclasses.  Fields are restricted to JSON-serializable primitives
(str, int, float, bool, dict, list) — no raw aiohttp objects, no
datetime (use ISO-8601 strings or epoch floats instead).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_json(obj: object) -> str:
    """Serialize any of our dataclasses to a JSON string."""
    return json.dumps(asdict(obj), separators=(",", ":"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
#  Market metadata
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MarketMetadata:
    market_ticker: str
    strike_price: float          # e.g. 65000.0
    settlement_time_iso: str     # ISO-8601 UTC
    last_trade_time_iso: str     # ISO-8601 UTC
    index_name: str = ""         # "CF Benchmarks BRTI" when available
    contract_type: str = "binary"
    series_ticker: str = ""
    event_ticker: str = ""
    status: str = ""
    asset: str = "BTC"
    window_minutes: int = 15
    is_range: bool = False
    strike_floor: Optional[float] = None
    strike_cap: Optional[float] = None
    is_directional: bool = False  # True for "BTC price up?" style contracts (no dollar strike)
    bot_id: str = "default"

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  BTC truth feed
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BtcMidPrice:
    """Single price tick from the truth feed."""
    price: float
    timestamp: float  # epoch seconds (float for sub-second precision)
    source: str = "unknown"
    asset: str = "BTC"

    def to_json(self) -> str:
        return _to_json(self)


@dataclass(slots=True)
class BtcWindowState:
    """Rolling 60-second window state for BTC mid price."""
    last_60_sum: float
    last_60_avg: float
    count: int                     # how many valid seconds in the window
    timestamp: float               # epoch seconds
    asset: str = "BTC"
    last_60_values: List[float] = field(default_factory=list)  # optional debug

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Probability
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FairProbability:
    market_ticker: str
    p_yes: float
    drift: float = 0.0  # per-second log-drift used in computation
    model_inputs: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Orderbook snapshot (bus message form)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OrderbookState:
    market_ticker: str
    best_yes_bid_cents: int
    best_no_bid_cents: int
    implied_yes_ask_cents: int     # 100 - best_no_bid_cents
    implied_no_ask_cents: int      # 100 - best_yes_bid_cents
    seq: int
    valid: bool
    # Phase 2 microstructure fields (optional — default to neutral values).
    obi: float = 0.0                # order-book imbalance at best level
    depth_pressure: float = 0.0     # weighted multi-level imbalance [-1, +1]
    micro_price_cents: float = 0.0  # volume-weighted micro-price (YES side)
    best_yes_depth: int = 0         # centi-contracts at best YES bid
    best_no_depth: int = 0          # centi-contracts at best NO bid

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Trade signal & execution
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TradeSignal:
    market_ticker: str
    side: str           # "yes" or "no"
    action: str         # "buy"
    limit_price_cents: int
    quantity_contracts: int  # in whole contracts
    edge: float
    p_yes: float
    timestamp: float
    order_style: str = "aggressive"  # "aggressive" (cross spread) or "passive" (post at bid)
    source: str = ""  # "mispricing_scalp" when emitted by MispricingScalper
    bot_id: str = "default"

    def to_json(self) -> str:
        return _to_json(self)


@dataclass(slots=True)
class OrderUpdate:
    """Reported by the execution layer after placing / cancelling / filling."""
    market_ticker: str
    order_id: str
    status: str          # "placed", "cancelled", "filled", "partial_fill", "error"
    side: str
    price_cents: int
    quantity_contracts: int
    filled_contracts: int
    remaining_contracts: int
    timestamp: float
    error_detail: str = ""
    bot_id: str = "default"

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Ticker feed (from WS ticker channel)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TickerUpdate:
    market_ticker: str
    yes_bid_cents: int
    yes_ask_cents: int
    last_price_cents: int
    volume: int
    timestamp: float

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  WebSocket lifecycle
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WsConnectionEvent:
    status: str        # "connected", "disconnected", "reconnecting", "error"
    detail: str = ""
    timestamp: float = 0.0

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Fill (from WS fill channel)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FillEvent:
    market_ticker: str
    order_id: str
    side: str
    price_cents: int
    count: int          # centi-contracts filled
    is_taker: bool
    timestamp: float
    fee_usd: float = 0.0
    source: str = ""    # e.g. "mispricing_scalp" for attribution to settlement/PnL
    action: str = "buy" # "buy" or "sell" — sell means early exit of an open position
    bot_id: str = "default"
    family: str = ""
    scenario_profile: str = "base"
    decision_context: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Trade tape (executed trades from Kalshi WS trade channel)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class KalshiTradeEvent:
    """A single executed trade on a Kalshi market."""
    market_ticker: str
    taker_side: str    # "yes" or "no"
    count: int         # contracts traded
    ts: float          # unix timestamp seconds

    def to_json(self) -> str:
        return _to_json(self)


@dataclass(slots=True)
class KalshiOrderDeltaEvent:
    """Orderbook delta-flow event derived from WS orderbook_delta updates."""
    market_ticker: str
    side: str          # "yes" or "no"
    is_add: bool       # True=order added, False=order cancelled/reduced
    qty: int
    ts: float

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Risk / kill switch
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RiskEvent:
    event_type: str      # "drawdown_breach", "disconnect_halt", "orderbook_invalid"
    detail: str = ""
    timestamp: float = 0.0

    def to_json(self) -> str:
        return _to_json(self)


@dataclass(slots=True)
class SettlementOutcome:
    """Published when a position settles; used for UI PnL, win rate, and event log."""
    market_ticker: str
    side: str            # "yes" or "no"
    won: bool
    pnl: float           # net $ profit/loss
    quantity_centicx: int
    entry_price_cents: float
    final_avg: float
    strike: float
    timestamp: float
    bot_id: str = "default"  # for multi-agent paper farm attribution
    source: str = ""     # e.g. "mispricing_scalp" so UI counts scalps only on settled trades
    gross_pnl: float = 0.0
    fees_usd: float = 0.0
    family: str = ""
    scenario_profile: str = "base"
    decision_context: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Market discovery
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SelectedMarkets:
    """Published when market discovery selects or refreshes the active ticker set."""
    tickers: List[str]
    timestamp: float

    def to_json(self) -> str:
        return _to_json(self)


@dataclass(slots=True)
class ActiveTicker:
    """Published by the strategy to tell the UI which single ticker to focus on."""
    ticker: str
    timestamp: float

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Account balance
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AccountBalance:
    """Published periodically by the balance poller in runner.py."""
    balance_cents: int      # raw API value in cents
    timestamp: float

    @property
    def balance_usd(self) -> float:
        return self.balance_cents / 100.0

    def to_json(self) -> str:
        return _to_json(self)


@dataclass(slots=True)
class KalshiRtt:
    """Kalshi round-trip time in ms. source is 'rest' (REST /exchange/status) or 'ws' (WebSocket subscribe→subscribed)."""
    rtt_ms: float
    timestamp: float = 0.0  # epoch seconds when measured
    source: Optional[str] = None  # "rest" | "ws"; None for backward compat

    def to_json(self) -> str:
        return _to_json(self)


# ---------------------------------------------------------------------------
#  Logging and Analytics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TerminalEvent:
    """A log message intended for the UI terminal and/or database tracking."""
    level: str           # "INFO", "WARN", "ALERT", "ERROR"
    message: str
    timestamp: float
    bot_id: str = "default"

    def to_json(self) -> str:
        return _to_json(self)


@dataclass(slots=True)
class StrategyDecision:
    """A record of a strategy tick evaluation, even if no trade was taken."""
    market_ticker: str
    p_yes: float
    yes_ask: int
    no_ask: int
    action_taken: str    # "buy_yes", "buy_no", "pass"
    reason: str          # e.g. "no_edge", "drawdown_breach", "max_position_reached"
    timestamp: float
    bot_id: str = "default"

    def to_json(self) -> str:
        return _to_json(self)
