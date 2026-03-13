"""
Argus Signal Types
==================

Deterministic signal event schemas for Phase 3.

All timestamps are int milliseconds (UTC epoch).
All signals are produced from BarEvents + RegimeEvents only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ═══════════════════════════════════════════════════════════════════════════
# Schema Version
# ═══════════════════════════════════════════════════════════════════════════

SIGNAL_SCHEMA_VERSION = 1

# ═══════════════════════════════════════════════════════════════════════════
# Signal Direction
# ═══════════════════════════════════════════════════════════════════════════

DIRECTION_LONG = "LONG"
DIRECTION_SHORT = "SHORT"
DIRECTION_NEUTRAL = "NEUTRAL"

# ═══════════════════════════════════════════════════════════════════════════
# Entry Types
# ═══════════════════════════════════════════════════════════════════════════

ENTRY_MARKET = "MARKET"
ENTRY_LIMIT = "LIMIT"
ENTRY_STOP = "STOP"
ENTRY_CONDITIONAL = "CONDITIONAL"

# ═══════════════════════════════════════════════════════════════════════════
# Signal Types (for filtering)
# ═══════════════════════════════════════════════════════════════════════════

SIGNAL_TYPE_ENTRY = "ENTRY"
SIGNAL_TYPE_EXIT = "EXIT"
SIGNAL_TYPE_FILTER = "FILTER"  # e.g., "market window open"


# ═══════════════════════════════════════════════════════════════════════════
# Serialization Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _round_float(val: float, decimals: int = 8) -> float:
    """Round float to fixed decimals for stable serialization."""
    if not isinstance(val, (int, float)):
        return val
    return round(val, decimals)


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _round_float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {
            str(k): _normalize_json_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(v) for v in value]
    raise TypeError(f"Unsupported snapshot value type: {type(value)}")


def _normalize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(k): _normalize_json_value(v)
        for k, v in sorted(snapshot.items(), key=lambda item: str(item[0]))
    }


def _sorted_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: data[k] for k in sorted(data)}


def normalize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize snapshot data to JSON-safe deterministic values."""
    return _normalize_snapshot(snapshot)


def _to_int_ms(ts: Any) -> int:
    """Convert timestamp to int milliseconds (backwards compat)."""
    if isinstance(ts, int):
        return ts
    if isinstance(ts, float):
        if ts < 2e10:
            return int(ts * 1000)
        return int(ts)
    raise ValueError(f"Invalid timestamp type: {type(ts)}")


def compute_signal_id(
    strategy_id: str,
    config_hash: str,
    symbol: str,
    timestamp_ms: int,
) -> str:
    """
    Compute deterministic idempotency key for a signal.
    
    Returns first 16 chars of SHA256 hex digest.
    """
    payload = f"{strategy_id}|{config_hash}|{symbol}|{timestamp_ms}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _normalize_for_hash(obj: Any) -> Any:
    """Normalize values for stable hashing (round floats to 8 decimals)."""
    if isinstance(obj, float):
        return round(obj, 8)
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(v) for v in obj]
    return obj


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of strategy configuration."""
    normalized = _normalize_for_hash(config)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════
# Signal Event Dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SignalEvent:
    """
    Deterministic trading signal event.
    
    Produced by strategy modules, consumed by the Signal Router/Ranker.
    Persisted and tape-recorded for replay and analysis.
    
    Published to: signals.raw
    """
    # ─── Attribution ───────────────────────────────────────────────────────
    timestamp_ms: int           # bar timestamp triggering signal
    strategy_id: str            # e.g., "OPTIONS_SPREAD_V1", "FVG_BREAKOUT_V1"
    config_hash: str            # hash of strategy parameters
    
    # ─── Intent ────────────────────────────────────────────────────────────
    symbol: str                 # underlying or contract symbol
    direction: str              # LONG | SHORT | NEUTRAL
    signal_type: str            # ENTRY | EXIT | FILTER
    timeframe: int              # bar_duration in seconds
    
    # ─── Entry/Exit (Optional) ─────────────────────────────────────────────
    entry_type: str = ENTRY_MARKET
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None
    horizon: str = "intraday"   # e.g., "intraday", "DTE=7"
    
    # ─── Quality & Provenance ──────────────────────────────────────────────
    confidence: float = 1.0     # 0-1 (from strategy)
    quality_score: int = 50     # 0-100 (from ranker)
    data_quality_flags: int = 0 # propagated from regimes/bars
    
    # ─── Context Snapshots (Compact, Deterministic) ────────────────────────
    regime_snapshot: Dict[str, str] = field(default_factory=dict)
    features_snapshot: Dict[str, float] = field(default_factory=dict)
    
    # ─── Research Provenance ─────────────────────────────────────────────
    case_id: str = ""           # Pantheon research case (e.g., "research_1708970820_3847")

    # ─── Identification ────────────────────────────────────────────────────
    explain: str = ""           # short deterministic explanation
    idempotency_key: str = ""   # computed from strategy+symbol+ts

    v: int = SIGNAL_SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════════════════════
# Ranked Signal (Output of Ranker)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class RankedSignalEvent:
    """
    A SignalEvent that has been scored and ranked.
    
    Published to: signals.ranked
    """
    signal: SignalEvent
    rank: int                   # 1 = top signal
    final_score: float          # composite score (higher = better)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    suppressed: bool = False
    suppression_reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Signal Outcome (Markout)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SignalOutcomeEvent:
    """
    Forward returns after a signal was emitted.
    
    Computed by the Markout Harness and persisted for analysis.
    """
    idempotency_key: str        # links back to SignalEvent
    timestamp_ms: int           # signal timestamp
    symbol: str
    strategy_id: str
    case_id: str = ""           # Pantheon research case provenance

    # Markouts (relative returns)
    ret_1bar: Optional[float] = None
    ret_5bar: Optional[float] = None
    ret_10bar: Optional[float] = None
    ret_60bar: Optional[float] = None
    
    # For options: mark-to-market P&L
    pnl_1bar: Optional[float] = None
    pnl_5bar: Optional[float] = None
    
    v: int = SIGNAL_SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════════════════════
# Serialization: SignalEvent
# ═══════════════════════════════════════════════════════════════════════════

def signal_to_dict(event: SignalEvent) -> Dict[str, Any]:
    """Serialize SignalEvent to dict for tape/persistence."""
    data = {
        "event_type": "signal",
        "timestamp_ms": event.timestamp_ms,
        "strategy_id": event.strategy_id,
        "config_hash": event.config_hash,
        "symbol": event.symbol,
        "direction": event.direction,
        "signal_type": event.signal_type,
        "timeframe": event.timeframe,
        "entry_type": event.entry_type,
        "entry_price": (
            _round_float(event.entry_price)
            if event.entry_price is not None
            else None
        ),
        "stop_price": (
            _round_float(event.stop_price)
            if event.stop_price is not None
            else None
        ),
        "tp_price": (
            _round_float(event.tp_price)
            if event.tp_price is not None
            else None
        ),
        "horizon": event.horizon,
        "confidence": _round_float(event.confidence),
        "quality_score": event.quality_score,
        "data_quality_flags": event.data_quality_flags,
        "regime_snapshot": _normalize_snapshot(event.regime_snapshot),
        "features_snapshot": _normalize_snapshot(event.features_snapshot),
        "case_id": event.case_id,
        "explain": event.explain,
        "idempotency_key": event.idempotency_key,
        "v": event.v,
    }
    return _sorted_dict(data)


def dict_to_signal(d: Dict[str, Any]) -> SignalEvent:
    """Deserialize dict to SignalEvent (backwards compat)."""
    return SignalEvent(
        timestamp_ms=_to_int_ms(d["timestamp_ms"]),
        strategy_id=str(d["strategy_id"]),
        config_hash=str(d["config_hash"]),
        symbol=str(d["symbol"]),
        direction=str(d["direction"]),
        signal_type=str(d["signal_type"]),
        timeframe=int(d["timeframe"]),
        entry_type=str(d.get("entry_type", ENTRY_MARKET)),
        entry_price=(
            float(d["entry_price"]) if d.get("entry_price") is not None else None
        ),
        stop_price=(
            float(d["stop_price"]) if d.get("stop_price") is not None else None
        ),
        tp_price=(
            float(d["tp_price"]) if d.get("tp_price") is not None else None
        ),
        horizon=str(d.get("horizon", "intraday")),
        confidence=float(d.get("confidence", 1.0)),
        quality_score=int(d.get("quality_score", 50)),
        data_quality_flags=int(d.get("data_quality_flags", 0)),
        regime_snapshot=dict(d.get("regime_snapshot", {})),
        features_snapshot={
            k: float(v) for k, v in d.get("features_snapshot", {}).items()
        },
        case_id=str(d.get("case_id", "")),
        explain=str(d.get("explain", "")),
        idempotency_key=str(d.get("idempotency_key", "")),
        v=int(d.get("v", SIGNAL_SCHEMA_VERSION)),
    )


def ranked_signal_to_dict(event: RankedSignalEvent) -> Dict[str, Any]:
    """Serialize RankedSignalEvent to dict."""
    data = {
        "event_type": "ranked_signal",
        "signal": signal_to_dict(event.signal),
        "rank": event.rank,
        "final_score": _round_float(event.final_score),
        "score_breakdown": _normalize_snapshot(event.score_breakdown),
        "suppressed": event.suppressed,
        "suppression_reason": event.suppression_reason,
    }
    return _sorted_dict(data)


def dict_to_ranked_signal(d: Dict[str, Any]) -> RankedSignalEvent:
    """Deserialize dict to RankedSignalEvent."""
    return RankedSignalEvent(
        signal=dict_to_signal(d["signal"]),
        rank=int(d["rank"]),
        final_score=float(d["final_score"]),
        score_breakdown={
            k: float(v) for k, v in d.get("score_breakdown", {}).items()
        },
        suppressed=bool(d.get("suppressed", False)),
        suppression_reason=str(d.get("suppression_reason", "")),
    )


def outcome_to_dict(event: SignalOutcomeEvent) -> Dict[str, Any]:
    """Serialize SignalOutcomeEvent to dict."""
    data = {
        "event_type": "signal_outcome",
        "idempotency_key": event.idempotency_key,
        "timestamp_ms": event.timestamp_ms,
        "symbol": event.symbol,
        "strategy_id": event.strategy_id,
        "case_id": event.case_id,
        "ret_1bar": _round_float(event.ret_1bar) if event.ret_1bar is not None else None,
        "ret_5bar": _round_float(event.ret_5bar) if event.ret_5bar is not None else None,
        "ret_10bar": _round_float(event.ret_10bar) if event.ret_10bar is not None else None,
        "ret_60bar": _round_float(event.ret_60bar) if event.ret_60bar is not None else None,
        "pnl_1bar": _round_float(event.pnl_1bar) if event.pnl_1bar is not None else None,
        "pnl_5bar": _round_float(event.pnl_5bar) if event.pnl_5bar is not None else None,
        "v": event.v,
    }
    return _sorted_dict(data)


def dict_to_outcome(d: Dict[str, Any]) -> SignalOutcomeEvent:
    """Deserialize dict to SignalOutcomeEvent."""
    return SignalOutcomeEvent(
        idempotency_key=str(d["idempotency_key"]),
        timestamp_ms=_to_int_ms(d["timestamp_ms"]),
        symbol=str(d["symbol"]),
        strategy_id=str(d["strategy_id"]),
        case_id=str(d.get("case_id", "")),
        ret_1bar=float(d["ret_1bar"]) if d.get("ret_1bar") is not None else None,
        ret_5bar=float(d["ret_5bar"]) if d.get("ret_5bar") is not None else None,
        ret_10bar=float(d["ret_10bar"]) if d.get("ret_10bar") is not None else None,
        ret_60bar=float(d["ret_60bar"]) if d.get("ret_60bar") is not None else None,
        pnl_1bar=float(d["pnl_1bar"]) if d.get("pnl_1bar") is not None else None,
        pnl_5bar=float(d["pnl_5bar"]) if d.get("pnl_5bar") is not None else None,
        v=int(d.get("v", SIGNAL_SCHEMA_VERSION)),
    )
