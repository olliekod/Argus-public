"""
Argus Event Types
=================

Dataclass-based events for the Pub/Sub event bus.
All events are immutable after creation.

Schema versioning: every event carries ``v`` (schema version).
Consumers should check ``v`` before decoding to handle evolution.
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional


# ─── Schema version ─────────────────────────────────────────
SCHEMA_VERSION = 1


class Priority(IntEnum):
    """Signal priority / severity levels."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class CloseReason(IntEnum):
    """Why a bar was closed / emitted."""
    MINUTE_BOUNDARY = 0   # normal: a new-minute tick arrived
    NEW_TICK = 1          # a tick for a later minute closed the prior bar
    SHUTDOWN_FLUSH = 2    # graceful shutdown flushed in-progress bars
    MINUTE_TICK = 3       # system.minute_tick event triggered close


@dataclass(frozen=True, slots=True)
class QuoteEvent:
    """Real-time price quote from a connector.

    Published to: market.quotes
    Contains price-related fields only.  Non-price metrics
    (IV, funding, open interest) belong in :class:`MetricEvent`.
    """
    symbol: str
    bid: float
    ask: float
    mid: float
    last: float
    timestamp: float          # exchange epoch seconds (UTC)
    source: str               # 'bybit', 'deribit', 'yahoo'
    volume_24h: float = 0.0
    source_ts: float = 0.0   # upstream data-source timestamp
    event_ts: float = field(default_factory=time.time)  # when Argus emits
    receive_time: float = field(default_factory=time.time)
    v: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MetricEvent:
    """Non-price market metric from a connector.

    Published to: market.metrics
    Carries funding rates, open interest, IV, or other
    auxiliary data that should not pollute the price path.
    """
    symbol: str
    metric: str               # 'funding_rate', 'open_interest', 'atm_iv', …
    value: float
    timestamp: float
    source: str
    extra: Dict[str, Any] = field(default_factory=dict)
    source_ts: float = 0.0
    event_ts: float = field(default_factory=time.time)
    receive_time: float = field(default_factory=time.time)
    v: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class BarEvent:
    """One-minute OHLCV bar aligned to UTC minute boundaries.

    Published to: market.bars

    Provenance fields (Stream 1.1)
    ------------------------------
    * ``n_ticks`` — number of source ticks aggregated into this bar.
    * ``first_source_ts`` / ``last_source_ts`` — earliest / latest
      exchange timestamps from the source quotes in this bar.
    * ``late_ticks_dropped`` — ticks discarded because they arrived
      after this bar's minute had already been emitted.
    * ``close_reason`` — why this bar was closed (see :class:`CloseReason`).
    * ``repaired`` — True if invariants were repaired deterministically.
    """
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float          # UTC epoch of the bar open (minute-aligned)
    source: str               # originating connector
    bar_duration: int = 60    # seconds
    tick_count: int = 0       # quotes aggregated (alias kept for compat)
    n_ticks: int = 0          # provenance: number of ticks used
    first_source_ts: float = 0.0   # earliest source quote timestamp
    last_source_ts: float = 0.0    # latest source quote timestamp
    late_ticks_dropped: int = 0    # ticks dropped for this symbol since last bar
    close_reason: int = 0          # CloseReason enum value
    source_ts: float = 0.0        # bar-level source timestamp (= first_source_ts)
    repaired: bool = False
    event_ts: float = field(default_factory=time.time)
    receive_time: float = field(default_factory=time.time)
    v: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """Detection / trading signal from a detector.

    Published to: signals.detections
    MUST include priority / severity.
    """
    detector: str             # 'ibit', 'bito', 'options_iv', 'volatility'
    symbol: str               # tradeable instrument (IBIT / BITO only)
    signal_type: str          # 'put_spread', 'iv_spike', 'regime_change'
    priority: Priority
    timestamp: float          # UTC epoch
    data: Dict[str, Any] = field(default_factory=dict)
    source_ts: float = 0.0
    event_ts: float = field(default_factory=time.time)
    receive_time: float = field(default_factory=time.time)
    v: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class StatusEvent:
    """Component health / status beacon.

    Published to: system.status
    """
    component: str
    status: str               # 'ok', 'degraded', 'error'
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)
    v: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class HeartbeatEvent:
    """Periodic heartbeat for flush / bookkeeping.

    Published to: system.heartbeat
    """
    sequence: int = 0
    timestamp: float = field(default_factory=time.time)
    v: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MinuteTickEvent:
    """UTC minute-boundary tick event.

    Published to: system.minute_tick
    """
    timestamp: float = field(default_factory=time.time)
    v: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ComponentHeartbeatEvent:
    """Structured heartbeat emitted periodically by each component.

    Published to: system.component_heartbeat
    Carries SRE-grade telemetry for dashboards and alerting.
    """
    component: str
    uptime_seconds: float = 0.0
    events_processed: int = 0
    latest_lag_ms: Optional[float] = None
    health: str = "ok"          # 'ok' | 'degraded' | 'down'
    extra: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    v: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class RegimeChangeEvent:
    """Regime transition detected by the RegimeDetector.

    Published to: signals.regime (legacy compatibility)
    """
    symbol: str
    old_regime: str           # e.g. 'LO_VOL_TREND'
    new_regime: str           # e.g. 'HI_VOL_RANGE'
    confidence: float = 0.0   # 0-1
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    v: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ExternalMetricEvent:
    """External metric injected into the regime pipeline.

    Published to: regimes.external_metrics
    Consumed by RegimeDetector to enrich ``metrics_json``.
    """
    key: str                  # e.g. 'global_risk_flow'
    value: Any                # JSON-serialisable metric value
    timestamp_ms: int         # epoch milliseconds
    v: int = SCHEMA_VERSION


# ─── Topic constants ────────────────────────────────────────
TOPIC_MARKET_QUOTES = "market.quotes"
TOPIC_MARKET_BARS = "market.bars"
TOPIC_MARKET_METRICS = "market.metrics"
TOPIC_SIGNALS = "signals.detections"
TOPIC_SIGNALS_REGIME = "signals.regime"
TOPIC_REGIMES_SYMBOL = "regimes.symbol"
TOPIC_REGIMES_MARKET = "regimes.market"
TOPIC_SYSTEM_STATUS = "system.status"
TOPIC_SYSTEM_HEARTBEAT = "system.heartbeat"
TOPIC_SYSTEM_MINUTE_TICK = "system.minute_tick"
TOPIC_SYSTEM_COMPONENT_HEARTBEAT = "system.component_heartbeat"

# Signal Topics
TOPIC_SIGNALS_RAW = "signals.raw"           # Raw strategy output
TOPIC_SIGNALS_RANKED = "signals.ranked"     # After ranker scoring
TOPIC_SIGNALS_OUTCOME = "signals.outcome"   # Markout results

# Options Topics
TOPIC_OPTIONS_CONTRACTS = "options.contracts"   # Static contract metadata
TOPIC_OPTIONS_QUOTES = "options.quotes"         # Live option quotes
TOPIC_OPTIONS_CHAINS = "options.chains"         # Atomic chain snapshots
TOPIC_OPTIONS_SPREADS = "options.spreads"       # Put spread candidates

# External Metrics Topics
TOPIC_EXTERNAL_METRICS = "regimes.external_metrics"  # GlobalRiskFlow etc.


