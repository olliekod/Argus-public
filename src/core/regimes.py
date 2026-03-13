"""
Argus Regime Types
==================

Deterministic regime event schemas for Phase 2.

All timestamps are int milliseconds (UTC epoch).
All regimes are computed from BarEvents only (no wall-clock).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict

from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE

# ═══════════════════════════════════════════════════════════════════════════
# Schema Version
# ═══════════════════════════════════════════════════════════════════════════

REGIME_SCHEMA_VERSION = 2


# ═══════════════════════════════════════════════════════════════════════════
# Data Quality Flags (bitmask)
# ═══════════════════════════════════════════════════════════════════════════

DQ_NONE = 0
DQ_REPAIRED_INPUT = 1 << 0   # input bar was repaired
DQ_GAP_WINDOW = 1 << 1       # gap detected in bar sequence
DQ_STALE_INPUT = 1 << 2      # indicator not fully warm


# ═══════════════════════════════════════════════════════════════════════════
# Regime Enums
# ═══════════════════════════════════════════════════════════════════════════

class VolRegime(IntEnum):
    """Volatility regime classification."""
    UNKNOWN = 0
    VOL_LOW = 1
    VOL_NORMAL = 2
    VOL_HIGH = 3
    VOL_SPIKE = 4


class TrendRegime(IntEnum):
    """Trend regime classification."""
    UNKNOWN = 0
    RANGE = 1
    TREND_UP = 2
    TREND_DOWN = 3


class SessionRegime(IntEnum):
    """Session regime for market hours."""
    UNKNOWN = 0
    # Equities
    PRE = 1
    RTH = 2
    POST = 3
    CLOSED = 4
    # Crypto
    ASIA = 10
    EU = 11
    US = 12
    OFFPEAK = 13


class LiquidityRegime(IntEnum):
    """Liquidity regime based on spread and volume."""
    UNKNOWN = 0
    LIQ_HIGH = 1     # tight spreads, high volume
    LIQ_NORMAL = 2   # normal conditions
    LIQ_LOW = 3      # wide spreads or low volume
    LIQ_DRIED = 4    # extremely poor liquidity


class RiskRegime(IntEnum):
    """Global risk regime (stub for future)."""
    UNKNOWN = 0
    RISK_ON = 1
    RISK_OFF = 2
    NEUTRAL = 3


# ═══════════════════════════════════════════════════════════════════════════
# String Constants (for serialization)
# ═══════════════════════════════════════════════════════════════════════════

VOL_REGIME_NAMES = {
    VolRegime.UNKNOWN: "UNKNOWN",
    VolRegime.VOL_LOW: "VOL_LOW",
    VolRegime.VOL_NORMAL: "VOL_NORMAL",
    VolRegime.VOL_HIGH: "VOL_HIGH",
    VolRegime.VOL_SPIKE: "VOL_SPIKE",
}

TREND_REGIME_NAMES = {
    TrendRegime.UNKNOWN: "UNKNOWN",
    TrendRegime.RANGE: "RANGE",
    TrendRegime.TREND_UP: "TREND_UP",
    TrendRegime.TREND_DOWN: "TREND_DOWN",
}

SESSION_REGIME_NAMES = {
    SessionRegime.UNKNOWN: "UNKNOWN",
    SessionRegime.PRE: "PRE",
    SessionRegime.RTH: "RTH",
    SessionRegime.POST: "POST",
    SessionRegime.CLOSED: "CLOSED",
    SessionRegime.ASIA: "ASIA",
    SessionRegime.EU: "EU",
    SessionRegime.US: "US",
    SessionRegime.OFFPEAK: "OFFPEAK",
}

LIQUIDITY_REGIME_NAMES = {
    LiquidityRegime.UNKNOWN: "UNKNOWN",
    LiquidityRegime.LIQ_HIGH: "LIQ_HIGH",
    LiquidityRegime.LIQ_NORMAL: "LIQ_NORMAL",
    LiquidityRegime.LIQ_LOW: "LIQ_LOW",
    LiquidityRegime.LIQ_DRIED: "LIQ_DRIED",
}

RISK_REGIME_NAMES = {
    RiskRegime.UNKNOWN: "UNKNOWN",
    RiskRegime.RISK_ON: "RISK_ON",
    RiskRegime.RISK_OFF: "RISK_OFF",
    RiskRegime.NEUTRAL: "NEUTRAL",
}


# ═══════════════════════════════════════════════════════════════════════════
# Default Thresholds
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_REGIME_THRESHOLDS = {
    # Volatility regime thresholds (vol_z)
    "vol_spike_z": 2.5,
    "vol_high_z": 1.0,
    "vol_low_z": -0.5,

    # Trend regime thresholds
    "trend_slope_threshold": 0.5,
    "trend_strength_threshold": 1.0,

    # Liquidity regime thresholds
    "liq_spread_high_pct": 0.05,    # < 5% spread = high liquidity
    "liq_spread_low_pct": 0.20,     # > 20% spread = low liquidity
    "liq_spread_dried_pct": 0.50,   # > 50% spread = dried up
    "liq_volume_high_pctile": 75,   # above 75th percentile = high
    "liq_volume_low_pctile": 25,    # below 25th percentile = low

    # ATR guard
    "atr_epsilon": 1e-8,

    # Gap detection
    "gap_tolerance_bars": 1,
    "gap_flag_duration_bars": 2,

    # Warmup requirements
    "warmup_bars": 30,

    # Robustness controls (disabled by default for backward compatibility)
    "vol_hysteresis_enabled": False,
    "vol_hysteresis_band": 0.0,
    "trend_hysteresis_enabled": False,
    "trend_hysteresis_slope_band": 0.0,
    "trend_hysteresis_strength_band": 0.0,
    "min_dwell_bars": 0,

    # Gap-aware confidence/warmup behavior (neutral defaults preserve old behavior)
    "gap_confidence_decay_threshold_ms": 0,
    "gap_confidence_decay_multiplier": 1.0,
    "gap_reset_window_threshold_ms": 0,

    # Quote-derived liquidity (disabled by default)
    "quote_liquidity_enabled": False,
    "quote_history_maxlen": 512,

    # Trend acceleration metric (classification impact disabled by default)
    "trend_accel_classification_enabled": False,
}


def compute_config_hash(thresholds: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of regime configuration.
    
    Returns first 12 chars of SHA256 hex digest.
    """
    # Sort keys for determinism
    canonical = json.dumps(thresholds, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════
# Event Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SymbolRegimeEvent:
    """
    Per-symbol regime classification.

    Emitted on every bar for the symbol. Contains volatility,
    trend, and liquidity regimes computed from incremental indicators.

    Published to: regimes.symbol
    """
    symbol: str
    timeframe: int              # bar_duration in seconds
    timestamp_ms: int           # bar timestamp (UTC epoch ms)

    # Regime classifications (string for serialization)
    vol_regime: str             # VOL_LOW | VOL_NORMAL | VOL_HIGH | VOL_SPIKE
    trend_regime: str           # TREND_UP | TREND_DOWN | RANGE
    liquidity_regime: str       # LIQ_HIGH | LIQ_NORMAL | LIQ_LOW | LIQ_DRIED

    # Core metrics
    atr: float
    atr_pct: float              # ATR / close (normalized)
    vol_z: float                # volatility z-score
    ema_fast: float
    ema_slow: float
    ema_slope: float            # (ema_fast - prev) / ATR
    rsi: float

    # Liquidity metrics
    spread_pct: float           # current bar spread as fraction of mid
    volume_pctile: float        # volume percentile (0-100)

    # Quality & confidence
    confidence: float           # 0-1
    is_warm: bool
    data_quality_flags: int     # bitmask (DQ_*)

    # Traceability
    config_hash: str
    trend_accel: float = 0.0    # optional curvature/acceleration metric

    v: int = REGIME_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MarketRegimeEvent:
    """
    Per-market regime classification.
    
    Covers session timing and global risk state.
    Session is derived purely from bar timestamp (no wall-clock).
    
    Published to: regimes.market
    """
    market: str                 # CRYPTO | EQUITIES
    timeframe: int              # bar_duration triggering update
    timestamp_ms: int           # bar timestamp (UTC epoch ms)
    
    # Session regime (derived from timestamp)
    session_regime: str         # PRE | RTH | POST | CLOSED (equities)
                                # ASIA | EU | US | OFFPEAK (crypto)
    
    # Global risk (stub for future)
    risk_regime: str            # RISK_ON | RISK_OFF | NEUTRAL | UNKNOWN
    
    # Quality
    confidence: float
    data_quality_flags: int
    
    # Traceability
    config_hash: str
    metrics_json: str = ""

    v: int = REGIME_SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════════════════════
# Serialization Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _round_float(val: float, decimals: int = 8) -> float:
    """Round float to fixed decimals for stable serialization."""
    return round(val, decimals)


def _to_int_ms(ts: Any) -> int:
    """
    Convert timestamp to int milliseconds.
    
    Handles backwards compatibility:
    - int: return as-is (assumed ms)
    - float < 2e10: treat as seconds, convert to ms
    - float >= 2e10: treat as ms, convert to int
    """
    if isinstance(ts, int):
        return ts
    if isinstance(ts, float):
        if ts < 2e10:  # seconds
            return int(ts * 1000)
        return int(ts)
    raise ValueError(f"Invalid timestamp type: {type(ts)}")


def canonical_metrics_json(metrics: Dict[str, Any]) -> str:
    """
    Serialize metrics dict to canonical JSON string.
    
    Deterministic: sorted keys, compact separators, rounded floats.
    """
    # Round all float values
    rounded = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            rounded[k] = _round_float(v)
        else:
            rounded[k] = v
    
    return json.dumps(rounded, sort_keys=True, separators=(",", ":"))




def symbol_regime_to_dict(event: SymbolRegimeEvent) -> Dict[str, Any]:
    """Serialize SymbolRegimeEvent to dict for tape/persistence."""
    return {
        "event_type": "symbol_regime",
        "symbol": event.symbol,
        "timeframe": event.timeframe,
        "timestamp_ms": event.timestamp_ms,
        "vol_regime": event.vol_regime,
        "trend_regime": event.trend_regime,
        "liquidity_regime": event.liquidity_regime,
        "atr": _round_float(event.atr),
        "atr_pct": _round_float(event.atr_pct),
        "vol_z": _round_float(event.vol_z),
        "ema_fast": _round_float(event.ema_fast),
        "ema_slow": _round_float(event.ema_slow),
        "ema_slope": _round_float(event.ema_slope),
        "rsi": _round_float(event.rsi),
        "spread_pct": _round_float(event.spread_pct),
        "volume_pctile": _round_float(event.volume_pctile),
        "trend_accel": _round_float(event.trend_accel),
        "confidence": _round_float(event.confidence),
        "is_warm": event.is_warm,
        "data_quality_flags": event.data_quality_flags,
        "config_hash": event.config_hash,
        "v": event.v,
    }


def dict_to_symbol_regime(d: Dict[str, Any]) -> SymbolRegimeEvent:
    """Deserialize dict to SymbolRegimeEvent.

    Backwards compatible: accepts float timestamps and converts to int ms.
    New fields (liquidity_regime, spread_pct, volume_pctile) default
    gracefully for data persisted before they existed.
    """
    return SymbolRegimeEvent(
        symbol=d["symbol"],
        timeframe=int(d["timeframe"]),
        timestamp_ms=_to_int_ms(d["timestamp_ms"]),
        vol_regime=d["vol_regime"],
        trend_regime=d["trend_regime"],
        liquidity_regime=d.get("liquidity_regime", "UNKNOWN"),
        atr=float(d["atr"]),
        atr_pct=float(d["atr_pct"]),
        vol_z=float(d["vol_z"]),
        ema_fast=float(d["ema_fast"]),
        ema_slow=float(d["ema_slow"]),
        ema_slope=float(d["ema_slope"]),
        rsi=float(d["rsi"]),
        spread_pct=float(d.get("spread_pct", 0.0)),
        volume_pctile=float(d.get("volume_pctile", 0.0)),
        trend_accel=float(d.get("trend_accel", 0.0)),
        confidence=float(d["confidence"]),
        is_warm=bool(d["is_warm"]),
        data_quality_flags=int(d["data_quality_flags"]),
        config_hash=str(d["config_hash"]),
        v=int(d.get("v", REGIME_SCHEMA_VERSION)),
    )


def market_regime_to_dict(event: MarketRegimeEvent) -> Dict[str, Any]:
    """Serialize MarketRegimeEvent to dict for tape/persistence."""
    return {
        "event_type": "market_regime",
        "market": event.market,
        "timeframe": event.timeframe,
        "timestamp_ms": event.timestamp_ms,
        "session_regime": event.session_regime,
        "risk_regime": event.risk_regime,
        "confidence": _round_float(event.confidence),
        "data_quality_flags": event.data_quality_flags,
        "config_hash": event.config_hash,
        "metrics_json": event.metrics_json,
        "v": event.v,
    }


def dict_to_market_regime(d: Dict[str, Any]) -> MarketRegimeEvent:
    """Deserialize dict to MarketRegimeEvent.

    Backwards compatible: accepts float timestamps and converts to int ms.
    """
    return MarketRegimeEvent(
        market=d["market"],
        timeframe=int(d["timeframe"]),
        timestamp_ms=_to_int_ms(d["timestamp_ms"]),
        session_regime=d["session_regime"],
        risk_regime=d["risk_regime"],
        confidence=float(d["confidence"]),
        data_quality_flags=int(d["data_quality_flags"]),
        config_hash=str(d["config_hash"]),
        metrics_json=str(d.get("metrics_json", "")),
        v=int(d.get("v", REGIME_SCHEMA_VERSION)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Market Classification
# ═══════════════════════════════════════════════════════════════════════════

# Global ETF proxies for overnight strategy risk-flow feature (daily bars only)
_GLOBAL_ETF_PROXIES = {"EWJ", "FXI", "EWT", "EWY", "INDA", "EWG", "EWU", "FEZ", "EWL", "EEM"}

# Symbols that belong to EQUITIES market
EQUITIES_SYMBOLS = set(LIQUID_ETF_UNIVERSE) | {"IBIT", "BITO", "NVDA"} | _GLOBAL_ETF_PROXIES

# All other symbols default to CRYPTO


def get_market_for_symbol(symbol: str) -> str:
    """Determine market scope for a symbol."""
    # Check if symbol starts with any equities prefix
    for eq_sym in EQUITIES_SYMBOLS:
        if symbol.upper().startswith(eq_sym):
            return "EQUITIES"
    return "CRYPTO"
