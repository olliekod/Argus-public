"""
Tape Recorder — Deterministic Event Capture & Replay
=====================================================

Captures a bounded rolling window of QuoteEvents, BarEvents, and MinuteTickEvents
for a configurable subset of symbols. The tape is replayable for determinism proof.

REPLAY MODES
------------
Two distinct replay modes with clear semantics:

**Faithful Mode (DEFAULT)**:
  - Append-only with monotonic `sequence_id` at record time.
  - Replay outputs events in EXACT recorded order (by sequence_id).
  - REQUIRED for deterministic strategy evaluation.

**Canonical Mode (OPTIONAL)**:
  - Produces stable order independent of arrival timing.
  - Sort key: (event_ts, provider_priority, event_type_priority, symbol, sequence_id)
  - Logs warning that it is NOT faithful to arrival order.
  - Use only for analysis/comparison, NOT primary evaluation.

TIMESTAMP CONVENTION
--------------------
All timestamps are stored as **int milliseconds** (UTC epoch ms).
This is enforced at record time and validated at replay.

ENVELOPE SCHEMA
---------------
Every taped record includes:
  - sequence_id: int (monotonic, assigned at record time)
  - event_ts: int (ms, arrival time)
  - provider: str
  - event_type: str ("bar" | "quote" | "minute_tick")
  - symbol: str
  - timeframe: int (bar_duration in seconds, for bars)

PRIORITY TABLES
---------------
Provider Priority (lower = higher priority):
  alpaca=1, yahoo=2, bybit=3, binance=4, deribit=5, polymarket=6, unknown=99

Event Type Priority (lower = higher priority):
  bar=1, quote=2, metric=3, minute_tick=4, signal=5, heartbeat=6
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Set, Tuple, Union

from ..core.events import (
    BarEvent,
    MinuteTickEvent,
    QuoteEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_QUOTES,
    TOPIC_SYSTEM_MINUTE_TICK,
    TOPIC_REGIMES_SYMBOL,
    TOPIC_REGIMES_MARKET,
    TOPIC_SIGNALS_RAW,
    TOPIC_SIGNALS_RANKED,
)
from ..core.option_events import (
    OptionContractEvent,
    OptionQuoteEvent,
    OptionChainSnapshotEvent,
    option_contract_to_dict,
    option_quote_to_dict,
    option_chain_to_dict,
    dict_to_option_contract,
    dict_to_option_quote,
    dict_to_option_chain,
    TOPIC_OPTIONS_CHAINS,
    TOPIC_OPTIONS_QUOTES,
    TOPIC_OPTIONS_CONTRACTS,
)
from ..core.regimes import (
    SymbolRegimeEvent,
    MarketRegimeEvent,
    symbol_regime_to_dict,
    market_regime_to_dict,
    dict_to_symbol_regime,
    dict_to_market_regime,
)
from ..core.signals import (
    SignalEvent as Phase3SignalEvent,
    RankedSignalEvent,
    signal_to_dict,
    dict_to_signal,
    ranked_signal_to_dict,
    dict_to_ranked_signal,
)

logger = logging.getLogger("argus.soak.tape")

# ══════════════════════════════════════════════════════════════
# PRIORITY TABLES
# ══════════════════════════════════════════════════════════════

# Provider priority for canonical tape ordering (lower = higher priority)
PROVIDER_PRIORITY: Dict[str, int] = {
    "alpaca": 1,
    "tradier": 2,
    "yahoo": 3,
    "bybit": 4,
    "binance": 5,
    "deribit": 6,
    "polymarket": 7,
    "unknown": 99,
}

# Event type priority for canonical tape ordering (lower = higher priority)
# 
# Ordering rationale:
#   1. bar         - Core market data, highest priority for replay
#   2. quote       - Equity quotes feed bar builder
#   3. option_contract - Static metadata, defines chain structure
#   4. option_quote    - Live option quotes per contract
#   5. option_chain    - Aggregate snapshot built from quotes
#   6. metric      - Derived metrics from market data
#   7. minute_tick - Time boundary marker
#   8. signal      - Strategy outputs from processed data
#   9. heartbeat   - System health (lowest data priority)
#   10-11. regime  - Market state classifications
#
EVENT_TYPE_PRIORITY: Dict[str, int] = {
    "bar": 1,
    "quote": 2,
    "option_contract": 3,  # Static metadata first
    "option_quote": 4,     # Live quotes per contract
    "option_chain": 5,     # Aggregate snapshot last
    "metric": 6,
    "minute_tick": 7,
    "signal": 8,
    "heartbeat": 9,
    "symbol_regime": 10,
    "market_regime": 11,
}


def _to_ms(ts: Union[int, float]) -> int:
    """Convert timestamp to int milliseconds.
    
    Handles:
      - Already int ms: return as-is
      - Float seconds (epoch < 2e10): convert to ms
      - Float ms: round to int
    """
    if isinstance(ts, int):
        return ts
    # Heuristic: if ts < 2e10, it's seconds; convert to ms
    if ts < 2e10:
        return int(ts * 1000)
    return int(ts)


def _validate_ms(ts: int, field_name: str) -> None:
    """Validate timestamp is a sane int milliseconds value."""
    if not isinstance(ts, int):
        raise ValueError(f"{field_name} must be int milliseconds, got {type(ts)}")
    # Sanity bounds: 2020-01-01 to 2035-01-01 in ms
    if not (1_577_836_800_000 <= ts <= 2_051_222_400_000):
        raise ValueError(f"{field_name}={ts} outside sane range [2020, 2035] in ms")


# ══════════════════════════════════════════════════════════════
# SERIALIZATION
# ══════════════════════════════════════════════════════════════

def _quote_to_dict(q: QuoteEvent, sequence_id: int = 0) -> Dict[str, Any]:
    """Serialize a QuoteEvent to a tape envelope.
    
    Args:
        q: QuoteEvent to serialize
        sequence_id: Monotonic sequence ID (default 0 for backward compat)
    """
    event_ts_ms = _to_ms(getattr(q, 'event_ts', 0) or q.timestamp)
    return {
        # Envelope fields (required for all events)
        "sequence_id": sequence_id,
        "event_ts": event_ts_ms,
        "provider": q.source or "unknown",
        "event_type": "quote",
        "type": "quote",  # Backward compat alias
        "symbol": q.symbol,
        "timeframe": 0,  # N/A for quotes
        # Quote-specific fields
        "bid": q.bid,
        "ask": q.ask,
        "mid": q.mid,
        "last": q.last,
        "timestamp": _to_ms(q.timestamp),
        "source": q.source,
        "volume_24h": q.volume_24h,
        "source_ts": _to_ms(q.source_ts) if q.source_ts else 0,
        "receive_time": _to_ms(getattr(q, 'receive_time', 0) or q.timestamp),
    }


def _tick_to_dict(t: MinuteTickEvent, sequence_id: int = 0) -> Dict[str, Any]:
    """Serialize a MinuteTickEvent to a tape envelope.
    
    Args:
        t: MinuteTickEvent to serialize
        sequence_id: Monotonic sequence ID (default 0 for backward compat)
    """
    return {
        # Envelope fields
        "sequence_id": sequence_id,
        "event_ts": _to_ms(t.timestamp),
        "provider": "system",
        "event_type": "minute_tick",
        "type": "minute_tick",  # Backward compat alias
        "symbol": "",
        "timeframe": 0,
        # Tick-specific fields
        "timestamp": _to_ms(t.timestamp),
    }


def _bar_to_dict(b: BarEvent, sequence_id: int = 0) -> Dict[str, Any]:
    """Serialize a BarEvent to a tape envelope.
    
    Args:
        b: BarEvent to serialize
        sequence_id: Monotonic sequence ID (default 0 for backward compat)
    """
    event_ts_ms = _to_ms(getattr(b, 'event_ts', 0) or b.timestamp)
    return {
        # Envelope fields (required for all events)
        "sequence_id": sequence_id,
        "event_ts": event_ts_ms,
        "provider": b.source or "unknown",
        "event_type": "bar",
        "type": "bar",  # Backward compat alias
        "symbol": b.symbol,
        "timeframe": getattr(b, 'bar_duration', 60),
        # Bar-specific fields
        "open": b.open,
        "high": b.high,
        "low": b.low,
        "close": b.close,
        "volume": b.volume,
        "timestamp": _to_ms(b.timestamp),
        "source": b.source,
        "bar_duration": getattr(b, 'bar_duration', 60),
        "n_ticks": getattr(b, 'n_ticks', 1),
        "first_source_ts": _to_ms(getattr(b, 'first_source_ts', b.timestamp)),
        "last_source_ts": _to_ms(getattr(b, 'last_source_ts', b.timestamp)),
        "source_ts": _to_ms(getattr(b, 'source_ts', b.timestamp)),
    }


def _from_tape_ts(ts: float) -> float:
    """Convert tape timestamp to float seconds.
    
    Handles both legacy tapes (seconds) and new tapes (ms).
    Same heuristic as _to_ms: if ts < 2e10, it's seconds.
    """
    if ts < 2e10:
        return ts  # Already seconds
    return ts / 1000.0  # Convert ms to seconds


def _dict_to_quote(d: Dict[str, Any]) -> QuoteEvent:
    """Deserialize a dict back to a QuoteEvent.
    
    Handles both legacy tapes (seconds) and new tapes (ms).
    """
    ts = d["timestamp"]
    return QuoteEvent(
        symbol=d["symbol"],
        bid=d["bid"],
        ask=d["ask"],
        mid=d["mid"],
        last=d["last"],
        timestamp=_from_tape_ts(ts),
        source=d["source"],
        volume_24h=d.get("volume_24h", 0.0),
        source_ts=_from_tape_ts(d.get("source_ts", 0)) if d.get("source_ts") else 0.0,
        event_ts=_from_tape_ts(d.get("event_ts", ts)),
        receive_time=_from_tape_ts(d.get("receive_time", ts)),
    )



def _dict_to_bar(d: Dict[str, Any]) -> BarEvent:
    """Deserialize a dict back to a BarEvent.
    
    Handles both legacy tapes (seconds) and new tapes (ms).
    """
    ts = d["timestamp"]
    return BarEvent(
        symbol=d["symbol"],
        open=d["open"],
        high=d["high"],
        low=d["low"],
        close=d["close"],
        volume=d["volume"],
        timestamp=_from_tape_ts(ts),
        source=d["source"],
        bar_duration=d.get("bar_duration", 60),
        n_ticks=d.get("n_ticks", 1),
        first_source_ts=_from_tape_ts(d.get("first_source_ts", ts)),
        last_source_ts=_from_tape_ts(d.get("last_source_ts", ts)),
        source_ts=_from_tape_ts(d.get("source_ts", ts)),
        event_ts=_from_tape_ts(d.get("event_ts", ts)),
    )


def _dict_to_event(d: Dict[str, Any]):
    """Deserialize a dict to the appropriate event type.
    
    Handles both legacy tapes (seconds) and new tapes (ms).
    """
    event_type = d.get("event_type") or d.get("type")
    if event_type == "quote":
        return _dict_to_quote(d)
    elif event_type == "minute_tick":
        ts = d["timestamp"]
        return MinuteTickEvent(timestamp=_from_tape_ts(ts))
    elif event_type == "bar":
        return _dict_to_bar(d)
    elif event_type == "option_chain":
        return dict_to_option_chain(d)
    elif event_type == "option_quote":
        return dict_to_option_quote(d)
    elif event_type == "option_contract":
        return dict_to_option_contract(d)
    raise ValueError(f"Unknown event type: {event_type}")


# ══════════════════════════════════════════════════════════════
# SORT KEYS
# ══════════════════════════════════════════════════════════════

def _faithful_sort_key(entry: Dict[str, Any]) -> int:
    """Sort key for faithful replay: sequence_id only."""
    return entry.get("sequence_id", 0)


def _canonical_sort_key(entry: Dict[str, Any]) -> Tuple[int, int, int, str, int]:
    """
    Sort key for canonical ordering.
    
    Key: (event_ts, provider_priority, event_type_priority, symbol, sequence_id)
    
    This produces a stable order independent of arrival timing.
    sequence_id is the final tiebreaker for determinism.
    """
    event_ts = entry.get("event_ts", 0)
    provider = entry.get("provider", "unknown")
    event_type = entry.get("event_type") or entry.get("type", "unknown")
    symbol = entry.get("symbol", "")
    sequence_id = entry.get("sequence_id", 0)
    
    provider_priority = PROVIDER_PRIORITY.get(provider, 99)
    event_type_priority = EVENT_TYPE_PRIORITY.get(event_type, 99)
    
    return (event_ts, provider_priority, event_type_priority, symbol, sequence_id)


def _bar_sort_key(bar: BarEvent) -> Tuple[int, str, str, int]:
    """Deterministic sort key for emitted bars."""
    return (
        _to_ms(bar.timestamp),
        bar.symbol,
        bar.source or "unknown",
        getattr(bar, "bar_duration", 60),
    )


class _ReplayBus:
    """Synchronous event bus for deterministic tape replay."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List] = {}

    def subscribe(self, topic: str, handler) -> None:
        self._subscribers.setdefault(topic, []).append(handler)

    def publish(self, topic: str, event: Any) -> None:
        for handler in self._subscribers.get(topic, []):
            handler(event)

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def get_queue_depths(self) -> Dict[str, int]:
        return {topic: 0 for topic in self._subscribers}


# ══════════════════════════════════════════════════════════════
# TAPE RECORDER
# ══════════════════════════════════════════════════════════════

class TapeRecorder:
    """Bounded rolling tape capture for determinism proof.

    Parameters
    ----------
    enabled : bool
        If False, all operations are no-ops.
    symbols : set of str or None
        Symbols to capture.  None = capture all.
    maxlen : int
        Maximum events in the rolling buffer.
    """

    def __init__(
        self,
        enabled: bool = False,
        symbols: Optional[Set[str]] = None,
        maxlen: int = 100_000,
    ) -> None:
        self._enabled = enabled
        self._symbols = symbols
        self._maxlen = maxlen
        self._tape: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._events_captured: int = 0
        self._events_evicted: int = 0
        # Monotonic sequence counter for faithful replay
        self._next_sequence_id: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_next_sequence_id(self) -> int:
        """Get next monotonic sequence ID (must hold lock)."""
        seq_id = self._next_sequence_id
        self._next_sequence_id += 1
        return seq_id

    def attach(self, bus) -> None:
        """Subscribe to event bus topics if enabled."""
        if not self._enabled:
            logger.info("TapeRecorder disabled — not subscribing")
            return
        bus.subscribe(TOPIC_MARKET_QUOTES, self._on_quote)
        bus.subscribe(TOPIC_MARKET_BARS, self._on_bar)
        bus.subscribe(TOPIC_SYSTEM_MINUTE_TICK, self._on_minute_tick)
        bus.subscribe(TOPIC_REGIMES_SYMBOL, self._on_symbol_regime)
        bus.subscribe(TOPIC_REGIMES_MARKET, self._on_market_regime)
        bus.subscribe(TOPIC_SIGNALS_RAW, self._on_raw_signal)
        bus.subscribe(TOPIC_SIGNALS_RANKED, self._on_ranked_signal)
        # Phase 3B: Options chain snapshots
        bus.subscribe(TOPIC_OPTIONS_CHAINS, self._on_option_chain)
        logger.info(
            "TapeRecorder attached (maxlen=%d, symbols=%s)",
            self._maxlen,
            self._symbols or "ALL",

        )

    def _on_quote(self, event: QuoteEvent) -> None:
        if not self._enabled:
            return
        if self._symbols and event.symbol not in self._symbols:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = _quote_to_dict(event, seq_id)
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_minute_tick(self, event: MinuteTickEvent) -> None:
        if not self._enabled:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = _tick_to_dict(event, seq_id)
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_bar(self, event: BarEvent) -> None:
        if not self._enabled:
            return
        if self._symbols and event.symbol not in self._symbols:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = _bar_to_dict(event, seq_id)
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_symbol_regime(self, event: SymbolRegimeEvent) -> None:
        if not self._enabled:
            return
        if self._symbols and event.symbol not in self._symbols:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = symbol_regime_to_dict(event)
            d["sequence_id"] = seq_id
            d["event_ts"] = event.timestamp_ms
            d["provider"] = "regime_detector"
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_market_regime(self, event: MarketRegimeEvent) -> None:
        if not self._enabled:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = market_regime_to_dict(event)
            d["sequence_id"] = seq_id
            d["event_ts"] = event.timestamp_ms
            d["provider"] = "regime_detector"
            d["symbol"] = event.market  # Use market as "symbol" for sorting
            d["timeframe"] = event.timeframe
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_option_chain(self, event: OptionChainSnapshotEvent) -> None:
        """Record option chain snapshot to tape."""
        if not self._enabled:
            return
        if self._symbols and event.symbol not in self._symbols:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = option_chain_to_dict(event)
            d["sequence_id"] = seq_id
            d["event_ts"] = event.recv_ts_ms
            d["provider"] = event.provider
            d["event_type"] = "option_chain"
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_raw_signal(self, event: Phase3SignalEvent) -> None:
        if not self._enabled:
            return
        if self._symbols and event.symbol not in self._symbols:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = signal_to_dict(event)
            d["sequence_id"] = seq_id
            d["event_ts"] = event.timestamp_ms
            d["provider"] = event.strategy_id
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_ranked_signal(self, event: RankedSignalEvent) -> None:
        if not self._enabled:
            return
        if self._symbols and event.signal.symbol not in self._symbols:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = ranked_signal_to_dict(event)
            d["sequence_id"] = seq_id
            d["event_ts"] = event.signal.timestamp_ms
            d["provider"] = event.signal.strategy_id
            d["symbol"] = event.signal.symbol
            d["timeframe"] = event.signal.timeframe
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def get_status(self) -> Dict[str, Any]:


        """Status snapshot for soak summary."""
        with self._lock:
            size = len(self._tape)
            next_seq = self._next_sequence_id
        return {
            "enabled": self._enabled,
            "tape_size": size,
            "maxlen": self._maxlen,
            "events_captured": self._events_captured,
            "events_evicted": self._events_evicted,
            "next_sequence_id": next_seq,
            "symbols": sorted(self._symbols) if self._symbols else None,
        }

    def export_jsonl(self, path: str, last_n_minutes: Optional[int] = None) -> int:
        """Export tape to a JSONL file.

        Parameters
        ----------
        path : str
            Output file path.
        last_n_minutes : int or None
            If set, only export events from the last N minutes.

        Returns
        -------
        int
            Number of events written.
        """
        if not self._enabled:
            logger.warning("TapeRecorder is disabled — nothing to export")
            return 0

        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (last_n_minutes * 60_000) if last_n_minutes else 0

        with self._lock:
            snapshot = list(self._tape)

        count = 0
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for entry in snapshot:
                ts_ms = entry.get("event_ts", entry.get("timestamp", 0))
                if ts_ms >= cutoff_ms:
                    f.write(json.dumps(entry) + "\n")
                    count += 1
        logger.info("Exported %d events to %s", count, path)
        return count

    @staticmethod
    def load_tape(path: str) -> List[Dict[str, Any]]:
        """Load a JSONL tape file."""
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @staticmethod
    def replay_tape(
        tape: List[Dict[str, Any]],
        mode: str = "faithful",
    ) -> List:
        """Replay a tape through a fresh BarBuilder and return emitted bars.

        Parameters
        ----------
        tape : list of dict
            Tape entries (quote, bar, minute_tick events).
        mode : str
            Replay mode:
              - "faithful" (DEFAULT): replay in exact recorded order (sequence_id).
                Required for deterministic strategy evaluation.
              - "canonical": replay in stable sorted order independent of arrival.
                NOT faithful to live arrival order. Use only for analysis.

        Returns
        -------
        list
            Emitted BarEvents from replay.
        """
        if mode not in ("faithful", "canonical"):
            raise ValueError(f"mode must be 'faithful' or 'canonical', got {mode!r}")
        
        if mode == "canonical":
            logger.warning(
                "CANONICAL replay mode: NOT faithful to arrival order. "
                "DO NOT use for strategy evaluation or determinism proofs. "
                "Use mode='faithful' for all strategy backtesting."
            )
            tape = sorted(tape, key=_canonical_sort_key)
        else:
            # Faithful mode: sort by sequence_id
            tape = sorted(tape, key=_faithful_sort_key)

        from ..core.bar_builder import BarBuilder
        from ..core.events import BarEvent as BE, TOPIC_MARKET_BARS

        bus = _ReplayBus()
        emitted: list = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)

        try:
            for entry in tape:
                event = _dict_to_event(entry)
                if isinstance(event, QuoteEvent):
                    bb._on_quote(event)
                elif isinstance(event, MinuteTickEvent):
                    bb._on_minute_tick(event)
                elif isinstance(event, BE):
                    # Pre-aggregated bars are passed through directly
                    bus.publish(TOPIC_MARKET_BARS, event)

            bb.flush()
        finally:
            bus.stop()

        return sorted(emitted, key=_bar_sort_key)

    @staticmethod
    def replay_tape_events(
        tape: List[Dict[str, Any]],
        mode: str = "faithful",
    ) -> Iterator[Tuple[int, Any]]:
        """Iterate tape events in replay order without processing.
        
        Yields (sequence_id, event) tuples for testing/inspection.
        
        Parameters
        ----------
        tape : list of dict
            Tape entries.
        mode : str
            "faithful" or "canonical".
            
        Yields
        ------
        tuple of (int, event)
            Sequence ID and deserialized event.
        """
        if mode not in ("faithful", "canonical"):
            raise ValueError(f"mode must be 'faithful' or 'canonical', got {mode!r}")
        
        if mode == "canonical":
            logger.warning(
                "CANONICAL replay is NOT faithful to arrival order; use only for analysis."
            )
            tape = sorted(tape, key=_canonical_sort_key)
        else:
            tape = sorted(tape, key=_faithful_sort_key)
        
        for entry in tape:
            seq_id = entry.get("sequence_id", 0)
            event = _dict_to_event(entry)
            yield seq_id, event
