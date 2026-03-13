"""
Argus Option Event Types
========================

Deterministic event schemas for options data ingestion.
All timestamps are int milliseconds (UTC epoch).

Schema versioning: every event carries ``v`` (schema version).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Schema version for options events
OPTIONS_SCHEMA_VERSION = 1


def _now_ms() -> int:
    """Current time as int milliseconds."""
    return int(time.time() * 1000)


def _to_ms(ts: float) -> int:
    """Convert float seconds to int milliseconds."""
    if ts < 2e10:  # Seconds
        return int(ts * 1000)
    return int(ts)


# ═══════════════════════════════════════════════════════════════════════════
# Option Contract Event (Static Metadata)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class OptionContractEvent:
    """Static options contract metadata.
    
    Published to: options.contracts
    Changes infrequently — only on new listings or contract updates.
    """
    symbol: str              # Underlying: "IBIT", "BITO"
    contract_id: str         # OCC/Exchange standardized ID
    option_symbol: str       # Full option symbol (OCC format)
    strike: float            # Strike price
    expiration_ms: int       # Expiration date (UTC midnight, ms)
    option_type: str         # "PUT" | "CALL"
    multiplier: int = 100    # Contract multiplier
    style: str = "american"  # "american" | "european"
    provider: str = ""       # "alpaca" | "tradier" | "yahoo"
    timestamp_ms: int = 0    # When this snapshot was taken
    source_ts_ms: int = 0    # Provider timestamp
    v: int = OPTIONS_SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════════════════════
# Option Quote Event (Live Quote + Greeks)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class OptionQuoteEvent:
    """Real-time options quote with optional Greeks.
    
    Published to: options.quotes
    Contains bid/ask/last with optional IV and Greeks.
    """
    contract_id: str         # FK to OptionContractEvent
    symbol: str              # Underlying: "IBIT", "BITO"
    strike: float
    expiration_ms: int
    option_type: str         # "PUT" | "CALL"
    
    # Prices
    bid: float
    ask: float
    last: float = 0.0
    mid: float = 0.0         # Computed: (bid + ask) / 2
    
    # Volume / OI
    volume: int = 0
    open_interest: int = 0
    
    # Greeks (optional — may be missing from some providers)
    iv: Optional[float] = None     # Implied volatility
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    
    # Timestamps (all int milliseconds UTC)
    timestamp_ms: int = 0          # Event logical time
    source_ts_ms: int = 0          # Provider timestamp
    recv_ts_ms: int = field(default_factory=_now_ms)  # Local arrival time
    
    provider: str = ""             # "alpaca" | "tradier"
    sequence_id: int = 0           # Monotonic for tape ordering
    v: int = OPTIONS_SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════════════════════
# Option Chain Snapshot Event (Atomic Chain)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class OptionChainSnapshotEvent:
    """Atomic snapshot of an options chain for one expiration.
    
    Published to: options.chains
    Contains entire put/call chain as frozen tuples for determinism.
    """
    symbol: str                    # "IBIT" | "BITO"
    expiration_ms: int             # Expiration as UTC ms
    underlying_price: float        # Spot price at snapshot time
    underlying_bid: float = 0.0
    underlying_ask: float = 0.0
    
    # Quotes indexed by strike (frozen for hashability)
    puts: Tuple[OptionQuoteEvent, ...] = ()   # Sorted by strike ascending
    calls: Tuple[OptionQuoteEvent, ...] = ()  # Sorted by strike ascending
    
    # Chain metadata
    n_strikes: int = 0
    atm_iv: Optional[float] = None       # ATM put IV
    
    # Timestamps
    timestamp_ms: int = 0                # Snapshot logical time
    source_ts_ms: int = 0                # Provider chain timestamp
    recv_ts_ms: int = field(default_factory=_now_ms)  # Local arrival time
    
    provider: str = ""
    snapshot_id: str = ""                # Deterministic: f"{symbol}_{expiration}_{timestamp_ms}"
    sequence_id: int = 0
    v: int = OPTIONS_SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════════════════════
# Serialization Helpers
# ═══════════════════════════════════════════════════════════════════════════


def option_contract_to_dict(c: OptionContractEvent, sequence_id: int = 0) -> Dict[str, Any]:
    """Serialize OptionContractEvent to dict for tape/persistence."""
    return {
        "event_type": "option_contract",
        "type": "option_contract",  # Alias for backward compat
        "symbol": c.symbol,
        "contract_id": c.contract_id,
        "option_symbol": c.option_symbol,
        "strike": c.strike,
        "expiration_ms": c.expiration_ms,
        "option_type": c.option_type,
        "multiplier": c.multiplier,
        "style": c.style,
        "provider": c.provider,
        "timestamp_ms": c.timestamp_ms,
        "source_ts_ms": c.source_ts_ms,
        "sequence_id": sequence_id,
        "v": c.v,
    }


def dict_to_option_contract(d: Dict[str, Any]) -> OptionContractEvent:
    """Deserialize dict to OptionContractEvent."""
    return OptionContractEvent(
        symbol=d["symbol"],
        contract_id=d["contract_id"],
        option_symbol=d["option_symbol"],
        strike=d["strike"],
        expiration_ms=d["expiration_ms"],
        option_type=d["option_type"],
        multiplier=d.get("multiplier", 100),
        style=d.get("style", "american"),
        provider=d.get("provider", ""),
        timestamp_ms=d.get("timestamp_ms", 0),
        source_ts_ms=d.get("source_ts_ms", 0),
        v=d.get("v", 1),
    )


def option_quote_to_dict(q: OptionQuoteEvent, sequence_id: int = 0) -> Dict[str, Any]:
    """Serialize OptionQuoteEvent to dict for tape/persistence."""
    return {
        "event_type": "option_quote",
        "type": "option_quote",  # Alias for backward compat
        "contract_id": q.contract_id,
        "symbol": q.symbol,
        "strike": q.strike,
        "expiration_ms": q.expiration_ms,
        "option_type": q.option_type,
        "bid": q.bid,
        "ask": q.ask,
        "last": q.last,
        "mid": q.mid,
        "volume": q.volume,
        "open_interest": q.open_interest,
        "iv": q.iv,
        "delta": q.delta,
        "gamma": q.gamma,
        "theta": q.theta,
        "vega": q.vega,
        "timestamp_ms": q.timestamp_ms,
        "source_ts_ms": q.source_ts_ms,
        "recv_ts_ms": q.recv_ts_ms,
        "provider": q.provider,
        "sequence_id": sequence_id if sequence_id else q.sequence_id,
        "v": q.v,
    }


def dict_to_option_quote(d: Dict[str, Any]) -> OptionQuoteEvent:
    """Deserialize dict to OptionQuoteEvent."""
    return OptionQuoteEvent(
        contract_id=d["contract_id"],
        symbol=d["symbol"],
        strike=d["strike"],
        expiration_ms=d["expiration_ms"],
        option_type=d["option_type"],
        bid=d["bid"],
        ask=d["ask"],
        last=d.get("last", 0.0),
        mid=d.get("mid", 0.0),
        volume=d.get("volume", 0),
        open_interest=d.get("open_interest", 0),
        iv=d.get("iv"),
        delta=d.get("delta"),
        gamma=d.get("gamma"),
        theta=d.get("theta"),
        vega=d.get("vega"),
        timestamp_ms=d.get("timestamp_ms", 0),
        source_ts_ms=d.get("source_ts_ms", 0),
        recv_ts_ms=d.get("recv_ts_ms", 0),
        provider=d.get("provider", ""),
        sequence_id=d.get("sequence_id", 0),
        v=d.get("v", 1),
    )


def option_chain_to_dict(s: OptionChainSnapshotEvent, sequence_id: int = 0) -> Dict[str, Any]:
    """Serialize OptionChainSnapshotEvent to dict for tape/persistence."""
    return {
        "event_type": "option_chain",
        "type": "option_chain",  # Alias for backward compat
        "symbol": s.symbol,
        "expiration_ms": s.expiration_ms,
        "underlying_price": s.underlying_price,
        "underlying_bid": s.underlying_bid,
        "underlying_ask": s.underlying_ask,
        "n_strikes": s.n_strikes,
        "atm_iv": s.atm_iv,
        "timestamp_ms": s.timestamp_ms,
        "source_ts_ms": s.source_ts_ms,
        "recv_ts_ms": s.recv_ts_ms,
        "provider": s.provider,
        "snapshot_id": s.snapshot_id,
        "sequence_id": sequence_id if sequence_id else s.sequence_id,
        "puts": [option_quote_to_dict(q) for q in s.puts],
        "calls": [option_quote_to_dict(q) for q in s.calls],
        "v": s.v,
    }


def dict_to_option_chain(d: Dict[str, Any]) -> OptionChainSnapshotEvent:
    """Deserialize dict to OptionChainSnapshotEvent."""
    puts = tuple(dict_to_option_quote(q) for q in d.get("puts", []))
    calls = tuple(dict_to_option_quote(q) for q in d.get("calls", []))
    
    return OptionChainSnapshotEvent(
        symbol=d["symbol"],
        expiration_ms=d["expiration_ms"],
        underlying_price=d["underlying_price"],
        underlying_bid=d.get("underlying_bid", 0.0),
        underlying_ask=d.get("underlying_ask", 0.0),
        puts=puts,
        calls=calls,
        n_strikes=d.get("n_strikes", len(puts)),
        atm_iv=d.get("atm_iv"),
        timestamp_ms=d.get("timestamp_ms", 0),
        source_ts_ms=d.get("source_ts_ms", 0),
        recv_ts_ms=d.get("recv_ts_ms", 0),
        provider=d.get("provider", ""),
        snapshot_id=d.get("snapshot_id", ""),
        sequence_id=d.get("sequence_id", 0),
        v=d.get("v", 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_snapshot_id(symbol: str, expiration_ms: int, timestamp_ms: int) -> str:
    """Compute deterministic snapshot ID."""
    return f"{symbol}_{expiration_ms}_{timestamp_ms}"


def compute_chain_hash(snapshot: OptionChainSnapshotEvent) -> str:
    """Compute deterministic hash of chain snapshot for deduplication."""
    key_data = {
        "symbol": snapshot.symbol,
        "expiration_ms": snapshot.expiration_ms,
        "underlying_price": snapshot.underlying_price,
        "n_puts": len(snapshot.puts),
        "n_calls": len(snapshot.calls),
        "timestamp_ms": snapshot.timestamp_ms,
    }
    key_str = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Topic Constants
# ═══════════════════════════════════════════════════════════════════════════


TOPIC_OPTIONS_CONTRACTS = "options.contracts"
TOPIC_OPTIONS_QUOTES = "options.quotes"
TOPIC_OPTIONS_CHAINS = "options.chains"
