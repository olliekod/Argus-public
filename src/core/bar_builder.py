"""
Argus Bar Builder
=================

Subscribes to ``market.quotes`` and aggregates tick data into
1-minute OHLCV bars aligned to UTC minute boundaries.

Rules
-----
* Use exchange ``timestamp`` only (no wall-clock fallback).
* Reject quotes missing a valid ``source_ts``.
* Bars are aligned to the **start** of each UTC minute
  (e.g. 12:03:00.000 – 12:03:59.999 → bar timestamp 12:03:00).
* When a new minute begins, the completed bar is published to
  ``market.bars`` via the event bus.

Volume handling
---------------
``QuoteEvent.volume_24h`` is **cumulative** exchange volume.  Summing
it directly would inflate bar volume by orders of magnitude.  Instead
we track the last-seen cumulative value per symbol and only add the
*delta* (current − previous).  A negative delta (exchange reset /
rollover) is treated as zero to avoid corrupting the bar.

Late-tick policy
----------------
A tick whose minute-floor falls **before** the active bar's open
timestamp is silently discarded.  Once a bar is emitted it is
immutable.

Bar invariants (Stream 1.1)
---------------------------
Before publishing, bars are validated against OHLCV invariants:
* high >= max(open, close)
* low  <= min(open, close)
* volume >= 0
On violation the bar is repaired, a warning is logged, and a
``bar_invariant_violations`` counter is incremented.

Close reasons
-------------
Every bar carries a deterministic ``close_reason``:
* ``NEW_TICK``       — a tick for a later minute closed the prior bar
* ``MINUTE_TICK``    — ``system.minute_tick`` triggered close
* ``SHUTDOWN_FLUSH`` — graceful shutdown flushed in-progress bars
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Deque, Dict

from .bus import EventBus
from .events import (
    BarEvent,
    CloseReason,
    ComponentHeartbeatEvent,
    MinuteTickEvent,
    QuoteEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_QUOTES,
    TOPIC_SYSTEM_MINUTE_TICK,
    TOPIC_SYSTEM_COMPONENT_HEARTBEAT,
)

logger = logging.getLogger("argus.bar_builder")

# Timestamp sanity bounds (epoch seconds).
# Anything outside this range is almost certainly a unit error (e.g. milliseconds).
# 2020-01-01 00:00:00 UTC
_TS_MIN = 1_577_836_800.0
# 2035-01-01 00:00:00 UTC (generous upper bound)
_TS_MAX = 2_051_222_400.0


def _ts_sane(ts: float) -> bool:
    """Return True if *ts* looks like a plausible epoch-seconds value."""
    return _TS_MIN <= ts <= _TS_MAX


class _BarAccumulator:
    """Mutable accumulator for a single in-progress bar."""

    __slots__ = (
        "open", "high", "low", "close", "volume",
        "ts_open", "source", "tick_count",
        "first_source_ts", "last_source_ts",
    )

    def __init__(self, price: float, volume_delta: float, ts_open: float,
                 source: str, source_ts: float = 0.0) -> None:
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume_delta
        self.ts_open = ts_open
        self.source = source
        self.tick_count = 1
        self.first_source_ts = source_ts if source_ts > 0 else ts_open
        self.last_source_ts = source_ts if source_ts > 0 else ts_open

    def update(self, price: float, volume_delta: float,
               source_ts: float = 0.0) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume_delta
        self.tick_count += 1
        ts = source_ts if source_ts > 0 else 0.0
        if ts > 0:
            if ts < self.first_source_ts or self.first_source_ts <= 0:
                self.first_source_ts = ts
            if ts > self.last_source_ts:
                self.last_source_ts = ts


def _minute_floor(epoch: float) -> float:
    """Round *epoch* down to the start of its UTC minute."""
    return float(int(epoch) // 60 * 60)


class BarBuilder:
    """Aggregates :class:`QuoteEvent` ticks into 1-minute :class:`BarEvent`.

    Parameters
    ----------
    bus : EventBus
        The shared event bus.  BarBuilder will subscribe to
        ``market.quotes`` and publish completed bars on ``market.bars``.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._bars: Dict[str, _BarAccumulator] = {}   # symbol → accumulator
        self._last_cum_vol: Dict[str, float] = {}      # symbol → last cumulative volume_24h
        self._lock = threading.Lock()
        self._bars_emitted_total = 0
        self._bars_emitted_by_symbol: Dict[str, int] = {}
        self._last_bar_ts_by_symbol: Dict[str, float] = {}
        self._late_ticks_dropped_total = 0
        self._late_ticks_dropped_by_symbol: Dict[str, int] = {}
        self._quotes_received_by_symbol: Dict[str, int] = {}
        self._bars_emitted_window_minutes = 60
        self._bars_emitted_timestamps_by_symbol: Dict[str, Deque[float]] = {}
        self._invariant_violation_count = 0
        self._quotes_rejected_total = 0
        self._quotes_rejected_by_symbol: Dict[str, int] = {}
        self._quotes_rejected_invalid_price_total = 0
        self._quotes_rejected_invalid_price_by_symbol: "OrderedDict[str, int]" = OrderedDict()
        self._quotes_rejected_invalid_price_cap = 100
        self._start_time = time.time()
        
        # Rate limiting for rejection logs (wall-clock only for logging, not event processing)
        self._reject_log_last_ts: Dict[str, float] = {}  # key → last wall-clock log time
        self._reject_log_interval = 60.0  # seconds between identical rejection logs
        
        bus.subscribe(TOPIC_MARKET_QUOTES, self._on_quote)
        bus.subscribe(TOPIC_SYSTEM_MINUTE_TICK, self._on_minute_tick)
        logger.info("BarBuilder initialised — subscribed to %s", TOPIC_MARKET_QUOTES)

    # ── volume delta helper ─────────────────────────────────

    def _volume_delta(self, symbol: str, cum_vol: float) -> float:
        """Compute the volume delta since last tick for *symbol*.

        * First tick for a symbol → delta = 0 (no prior reference).
        * Negative delta (exchange reset / rollover) → 0.
        * Otherwise → ``cum_vol - last_cum_vol``.
        """
        prev = self._last_cum_vol.get(symbol)
        self._last_cum_vol[symbol] = cum_vol

        if prev is None:
            return 0.0

        delta = cum_vol - prev
        if delta < 0:
            # Exchange reset / rollover — ignore this tick's volume
            return 0.0
        return delta

    # ── handler (called from the bus worker thread) ─────────

    def _on_quote(self, event: QuoteEvent) -> None:
        """Ingest a quote and build / emit bars."""
        if not event.timestamp or event.timestamp <= 0:
            self._reject_quote(event, "missing/invalid timestamp")
            return
        if not event.source_ts or event.source_ts <= 0:
            reason = "missing source_ts" if not event.source_ts else "non-positive source_ts"
            self._reject_quote(event, reason)
            return

        # Sanity-check timestamp units (detect ms-vs-seconds confusion)
        if not _ts_sane(event.timestamp):
            if event.timestamp > _TS_MAX:
                # Likely milliseconds — auto-correct would hide the bug
                self._reject_quote(
                    event,
                    f"timestamp={event.timestamp:.0f} looks like milliseconds "
                    f"(expected seconds)",
                )
            else:
                self._reject_quote(
                    event,
                    f"timestamp={event.timestamp:.0f} outside sane range "
                    f"[{_TS_MIN:.0f}, {_TS_MAX:.0f}]",
                )
            return

        if not _ts_sane(event.source_ts):
            if event.source_ts > _TS_MAX:
                self._reject_quote(
                    event,
                    f"source_ts={event.source_ts:.0f} looks like milliseconds "
                    f"(expected seconds)",
                )
            else:
                self._reject_quote(
                    event,
                    f"source_ts={event.source_ts:.0f} outside sane range",
                )
            return

        if event.bid <= 0 or event.ask <= 0:
            self._reject_invalid_price(event, "non-positive bid/ask")
            return
        if event.bid > event.ask:
            self._reject_invalid_price(event, "crossed bid/ask")
            return

        ts = event.timestamp
        minute = _minute_floor(ts)
        price = event.last if event.last else event.mid
        if price <= 0:
            return

        source_ts = event.source_ts

        self._quotes_received_by_symbol[event.symbol] = (
            self._quotes_received_by_symbol.get(event.symbol, 0) + 1
        )
        vol_delta = self._volume_delta(event.symbol, event.volume_24h)

        with self._lock:
            acc = self._bars.get(event.symbol)

            if acc is None:
                # First tick for this symbol — start a new bar
                self._bars[event.symbol] = _BarAccumulator(
                    price, vol_delta, minute, event.source, source_ts
                )
                return

            # ── Late-tick guard ─────────────────────────────
            # Discard ticks older than the active bar window.
            # Once a bar is emitted it must never change.
            if minute < acc.ts_open:
                self._late_ticks_dropped_total += 1
                self._late_ticks_dropped_by_symbol[event.symbol] = (
                    self._late_ticks_dropped_by_symbol.get(event.symbol, 0) + 1
                )
                return

            if minute > acc.ts_open:
                # New minute — emit the completed bar and start fresh
                self._emit_bar(event.symbol, acc, CloseReason.NEW_TICK)

                # Reset accumulator for the new minute
                self._bars[event.symbol] = _BarAccumulator(
                    price, vol_delta, minute, event.source, source_ts
                )
            else:
                # Same minute — update accumulator
                acc.update(price, vol_delta, source_ts)

    def _on_minute_tick(self, event: MinuteTickEvent) -> None:
        """Flush any bars whose minute has closed at a boundary tick."""
        tick_minute = _minute_floor(event.timestamp)
        with self._lock:
            to_remove = []
            for symbol, acc in self._bars.items():
                if tick_minute > acc.ts_open:
                    self._emit_bar(symbol, acc, CloseReason.MINUTE_TICK)
                    to_remove.append(symbol)
            for symbol in to_remove:
                self._bars.pop(symbol, None)

    # ── bar invariant enforcement ─────────────────────────

    def _enforce_invariants(self, acc: _BarAccumulator) -> bool:
        """Validate and repair OHLCV invariants. Returns True if bar was valid."""
        valid = True

        # high must be >= max(open, close)
        expected_high = max(acc.open, acc.close)
        if acc.high < expected_high:
            logger.warning(
                "Bar invariant violation: high (%.6f) < max(open, close) (%.6f), repairing",
                acc.high, expected_high,
            )
            acc.high = expected_high
            valid = False

        # low must be <= min(open, close)
        expected_low = min(acc.open, acc.close)
        if acc.low > expected_low:
            logger.warning(
                "Bar invariant violation: low (%.6f) > min(open, close) (%.6f), repairing",
                acc.low, expected_low,
            )
            acc.low = expected_low
            valid = False

        # volume must be >= 0
        if acc.volume < 0:
            logger.warning(
                "Bar invariant violation: volume (%.6f) < 0, setting to 0",
                acc.volume,
            )
            acc.volume = 0.0
            valid = False

        if not valid:
            self._invariant_violation_count += 1

        return valid

    # ── utility ─────────────────────────────────────────────

    def _emit_bar(self, symbol: str, acc: _BarAccumulator,
                  close_reason: CloseReason = CloseReason.MINUTE_BOUNDARY) -> BarEvent:
        # Enforce invariants before publishing
        valid = self._enforce_invariants(acc)

        # Collect late-tick count for this symbol and reset
        late_dropped = self._late_ticks_dropped_by_symbol.get(symbol, 0)
        self._late_ticks_dropped_by_symbol[symbol] = 0

        now = time.time()
        bar = BarEvent(
            symbol=symbol,
            open=acc.open,
            high=acc.high,
            low=acc.low,
            close=acc.close,
            volume=acc.volume,
            timestamp=acc.ts_open,
            source=acc.source,
            bar_duration=60,
            tick_count=acc.tick_count,
            n_ticks=acc.tick_count,
            first_source_ts=acc.first_source_ts,
            last_source_ts=acc.last_source_ts,
            late_ticks_dropped=late_dropped,
            close_reason=int(close_reason),
            source_ts=acc.first_source_ts,
            repaired=not valid,
            event_ts=now,
        )
        self._bus.publish(TOPIC_MARKET_BARS, bar)
        self._bars_emitted_total += 1
        self._bars_emitted_by_symbol[symbol] = self._bars_emitted_by_symbol.get(symbol, 0) + 1
        self._last_bar_ts_by_symbol[symbol] = acc.ts_open

        window_seconds = self._bars_emitted_window_minutes * 60
        ts_deque = self._bars_emitted_timestamps_by_symbol.setdefault(symbol, deque())
        ts_deque.append(acc.ts_open)
        while ts_deque and (now - ts_deque[0]) > window_seconds:
            ts_deque.popleft()
        return bar

    def flush(self) -> list[BarEvent]:
        """Flush all in-progress bars (e.g. on shutdown).

        Returns the list of emitted bars.
        """
        emitted: list[BarEvent] = []
        with self._lock:
            for symbol, acc in self._bars.items():
                emitted.append(self._emit_bar(symbol, acc, CloseReason.SHUTDOWN_FLUSH))
            self._bars.clear()
        logger.info("BarBuilder flushed %d partial bars", len(emitted))
        return emitted

    def emit_heartbeat(self) -> ComponentHeartbeatEvent:
        """Create and publish a structured heartbeat for this component."""
        now = time.time()
        with self._lock:
            total_quotes = sum(self._quotes_received_by_symbol.values())
            latest_lag = None
            if self._last_bar_ts_by_symbol:
                most_recent = max(self._last_bar_ts_by_symbol.values())
                latest_lag = (now - most_recent) * 1000  # ms

        health = "ok"
        if latest_lag is not None and latest_lag > 300_000:
            health = "degraded"
        if not self._bars_emitted_total:
            health = "down" if (now - self._start_time) > 120 else "ok"

        hb = ComponentHeartbeatEvent(
            component="bar_builder",
            uptime_seconds=round(now - self._start_time, 1),
            events_processed=total_quotes,
            latest_lag_ms=round(latest_lag, 1) if latest_lag is not None else None,
            health=health,
            extra={
                "bars_emitted_total": self._bars_emitted_total,
                "bar_invariant_violations": self._invariant_violation_count,
                "invariant_violation_count": self._invariant_violation_count,
                "quotes_rejected_total": self._quotes_rejected_total,
                "quotes_rejected_invalid_price_total": self._quotes_rejected_invalid_price_total,
            },
        )
        self._bus.publish(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, hb)
        return hb

    def get_status(self) -> Dict[str, object]:
        now = time.time()
        with self._lock:
            last_bar_ts = dict(self._last_bar_ts_by_symbol)
            active_symbols = list(self._bars.keys())
            bars_emitted_total = self._bars_emitted_total
            bars_emitted_by_symbol = dict(self._bars_emitted_by_symbol)
            late_ticks_total = self._late_ticks_dropped_total
            late_ticks_by_symbol = dict(self._late_ticks_dropped_by_symbol)
            quotes_received_by_symbol = dict(self._quotes_received_by_symbol)
            invariant_violations = self._invariant_violation_count
            bars_emitted_recent_by_symbol = {
                symbol: sum(1 for ts in timestamps if (now - ts) <= (self._bars_emitted_window_minutes * 60))
                for symbol, timestamps in self._bars_emitted_timestamps_by_symbol.items()
            }
            # Rolling bars/minute (last 5 minutes)
            _bpm_window = 300  # 5 minutes
            bars_per_minute_by_symbol = {}
            for symbol, timestamps in self._bars_emitted_timestamps_by_symbol.items():
                recent = sum(1 for ts in timestamps if (now - ts) <= _bpm_window)
                bars_per_minute_by_symbol[symbol] = round(recent / 5.0, 2)
            # Aggregate stats across symbols
            all_rates = sorted(bars_per_minute_by_symbol.values()) if bars_per_minute_by_symbol else []
            bars_per_minute_p50 = all_rates[len(all_rates) // 2] if all_rates else 0.0
            bars_per_minute_p95 = (
                all_rates[min(int(len(all_rates) * 0.95), len(all_rates) - 1)]
                if all_rates else 0.0
            )

        ages = {
            symbol: round(now - ts, 1)
            for symbol, ts in last_bar_ts.items()
            if ts is not None
        }
        max_age = max(ages.values()) if ages else None

        if not last_bar_ts:
            status = "unknown"
        elif any(age < 300 for age in ages.values()):
            # If at least one symbol is ticking (e.g. crypto), pipeline is OK
            status = "ok"
        elif max_age is not None and max_age > 300:
            status = "degraded"
        else:
            status = "ok"

        from .status import build_status

        return build_status(
            name="bar_builder",
            type="internal",
            status=status,
            last_success_ts=max(last_bar_ts.values()) if last_bar_ts else None,
            consecutive_failures=0,
            request_count=bars_emitted_total,
            error_count=late_ticks_total,
            last_message_ts=max(last_bar_ts.values()) if last_bar_ts else None,
            age_seconds=max_age,
            extras={
                "bars_emitted_total": bars_emitted_total,
                "bars_emitted_by_symbol": bars_emitted_by_symbol,
                "bars_emitted_recent_by_symbol": bars_emitted_recent_by_symbol,
                "bars_emitted_recent_window_minutes": self._bars_emitted_window_minutes,
                "bars_per_minute_by_symbol": bars_per_minute_by_symbol,
                "bars_per_minute_p50": bars_per_minute_p50,
                "bars_per_minute_p95": bars_per_minute_p95,
                "last_bar_ts_by_symbol": {
                    symbol: datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    for symbol, ts in last_bar_ts.items()
                },
                "last_bar_ts_by_symbol_epoch": dict(last_bar_ts),
                "late_ticks_dropped_total": late_ticks_total,
                "late_ticks_dropped_by_symbol": late_ticks_by_symbol,
                "quotes_received_by_symbol": quotes_received_by_symbol,
                "quotes_rejected_total": self._quotes_rejected_total,
                "quotes_rejected_by_symbol": dict(self._quotes_rejected_by_symbol),
                "quotes_rejected_invalid_price_total": self._quotes_rejected_invalid_price_total,
                "quotes_rejected_invalid_price_by_symbol": dict(
                    self._quotes_rejected_invalid_price_by_symbol
                ),
                "active_symbols_count": len(active_symbols),
                "bar_invariant_violations": invariant_violations,
                "invariant_violation_count": invariant_violations,
            },
        )

    def _reject_quote(self, event: QuoteEvent, reason: str) -> None:
        self._quotes_rejected_total += 1
        self._quotes_rejected_by_symbol[event.symbol] = (
            self._quotes_rejected_by_symbol.get(event.symbol, 0) + 1
        )
        # Rate-limited logging: only log if interval elapsed (wall-clock only for logging hygiene)
        log_key = f"reject:{event.symbol}:{reason}"
        now_wall = time.time()  # wall-clock for log throttling ONLY
        last_log = self._reject_log_last_ts.get(log_key, 0)
        if (now_wall - last_log) >= self._reject_log_interval:
            self._reject_log_last_ts[log_key] = now_wall
            total = self._quotes_rejected_by_symbol.get(event.symbol, 0)
            logger.warning(
                "Rejected quote for %s [connector=%s]: %s "
                "(timestamp=%r, source_ts=%r) [%d total for symbol]",
                event.symbol,
                event.source,
                reason,
                event.timestamp,
                event.source_ts,
                total,
            )

    def _reject_invalid_price(self, event: QuoteEvent, reason: str) -> None:
        self._quotes_rejected_invalid_price_total += 1
        self._bump_bounded_counter(
            self._quotes_rejected_invalid_price_by_symbol,
            event.symbol,
            self._quotes_rejected_invalid_price_cap,
        )
        # Rate-limited logging: only log if interval elapsed (wall-clock only for logging hygiene)
        log_key = f"reject_price:{event.symbol}:{reason}"
        now_wall = time.time()  # wall-clock for log throttling ONLY
        last_log = self._reject_log_last_ts.get(log_key, 0)
        if (now_wall - last_log) >= self._reject_log_interval:
            self._reject_log_last_ts[log_key] = now_wall
            total = self._quotes_rejected_invalid_price_by_symbol.get(event.symbol, 0)
            logger.warning(
                "Rejected quote for %s [connector=%s]: %s "
                "(bid=%r, ask=%r) [%d total for symbol]",
                event.symbol,
                event.source,
                reason,
                event.bid,
                event.ask,
                total,
            )

    @staticmethod
    def _bump_bounded_counter(
        store: "OrderedDict[str, int]",
        key: str,
        cap: int,
    ) -> None:
        if key in store:
            store[key] += 1
            store.move_to_end(key)
            return
        if len(store) >= cap:
            store.popitem(last=False)
        store[key] = 1
