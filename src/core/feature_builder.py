# Created by Oliver Meihls

# Argus Feature Builder (Stream 4)
#
# Consumes ``market.bars`` and computes rolling metrics:
#
# * **returns** — 1-bar log returns
# * **realized_vol** — rolling realized volatility (annualised)
# * **jump_score** — detects sudden price jumps (|return| / rolling σ)
#
# Publishes computed metrics to ``market.metrics`` and persists via the
# existing MetricEvent pipeline.
#
# Safety constraints
# * Downstream-only: subscribes to ``market.bars``, publishes to
# ``market.metrics``.  Never mutates upstream state.
# * Bounded internal state: rolling windows are capped per symbol.
# * Overload: if the bar queue backs up, old bars are dropped (deque
# maxlen), never blocking the ingestion path.

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from .bus import EventBus
from .events import (
    BarEvent,
    ComponentHeartbeatEvent,
    MetricEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_METRICS,
    TOPIC_SYSTEM_COMPONENT_HEARTBEAT,
)

logger = logging.getLogger("argus.feature_builder")

# Rolling window sizes
_RETURNS_WINDOW = 60       # bars (= 1 hour of 1-minute bars)
_VOL_WINDOW = 30           # bars for realized vol calculation
_JUMP_THRESHOLD = 3.0      # σ multiples to flag a jump
_ANNUALISE_FACTOR = math.sqrt(365.25 * 24 * 60)  # for 1-minute bars


class FeatureBuilder:
    # Computes rolling market features from 1-minute bars.
    #
    # Parameters
    # bus : EventBus
    # Shared event bus.

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Per-symbol rolling state
        self._last_close: Dict[str, float] = {}
        self._returns: Dict[str, Deque[float]] = {}
        self._bars_processed = 0
        self._metrics_emitted = 0

        bus.subscribe(TOPIC_MARKET_BARS, self._on_bar)
        logger.info("FeatureBuilder initialised — subscribed to %s", TOPIC_MARKET_BARS)

    # ── bar handler ──────────────────────────────────────────

    def _on_bar(self, event: BarEvent) -> None:
        # Compute features from an incoming bar.
        symbol = event.symbol
        close = event.close
        ts = event.timestamp
        bar_duration = getattr(event, 'bar_duration', 60)  # seconds, default 1-min
        ts_close = ts + bar_duration
        source = event.source

        with self._lock:
            self._bars_processed += 1

            prev_close = self._last_close.get(symbol)
            self._last_close[symbol] = close

            if prev_close is None or prev_close <= 0 or close <= 0:
                return

            # Log return
            log_ret = math.log(close / prev_close)

            if not math.isfinite(log_ret):
                logger.warning("Non-finite log return for %s: %s, skipping", symbol, log_ret)
                return

            # Maintain rolling returns buffer
            if symbol not in self._returns:
                self._returns[symbol] = deque(maxlen=_RETURNS_WINDOW)
            self._returns[symbol].append(log_ret)

            # Publish return metric
            self._publish_metric(symbol, "log_return", log_ret, ts_close, source)

            # Realized volatility (need enough data)
            ret_buf = self._returns[symbol]
            if len(ret_buf) >= _VOL_WINDOW:
                recent = list(ret_buf)[-_VOL_WINDOW:]
                mean = sum(recent) / len(recent)
                variance = sum((r - mean) ** 2 for r in recent) / (len(recent) - 1)
                realised_vol = math.sqrt(variance) * _ANNUALISE_FACTOR
                self._publish_metric(symbol, "realized_vol", realised_vol, ts_close, source)

                # Jump score
                if variance > 0:
                    sigma = math.sqrt(variance)
                    jump_score = abs(log_ret) / sigma
                    self._publish_metric(symbol, "jump_score", jump_score, ts_close, source)

    # ── metric publishing ────────────────────────────────────

    def _publish_metric(
        self,
        symbol: str,
        metric: str,
        value: float,
        timestamp: float,
        source: str,
    ) -> None:
        # Emit a MetricEvent onto the bus.
        evt = MetricEvent(
            symbol=symbol,
            metric=metric,
            value=value,
            timestamp=timestamp,
            source=f"feature_builder:{source}",
        )
        self._bus.publish(TOPIC_MARKET_METRICS, evt)
        self._metrics_emitted += 1

    # ── heartbeat / status ───────────────────────────────────

    def emit_heartbeat(self) -> ComponentHeartbeatEvent:
        now = time.time()
        with self._lock:
            processed = self._bars_processed
            emitted = self._metrics_emitted
            symbols_tracked = len(self._last_close)

        health = "ok"
        if not processed and (now - self._start_time) > 120:
            health = "down"

        hb = ComponentHeartbeatEvent(
            component="feature_builder",
            uptime_seconds=round(now - self._start_time, 1),
            events_processed=processed,
            health=health,
            extra={
                "metrics_emitted": emitted,
                "symbols_tracked": symbols_tracked,
            },
        )
        self._bus.publish(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, hb)
        return hb

    def get_status(self) -> Dict[str, Any]:
        from .status import build_status
        
        now = time.time()
        with self._lock:
            bars_processed = self._bars_processed
            metrics_emitted = self._metrics_emitted
            symbols_tracked = len(self._last_close)
        
        # Determine health status
        uptime = now - self._start_time
        if bars_processed > 0:
            status = "ok"
        elif uptime > 120:
            status = "down"  # No data after 2 minutes
        else:
            status = "ok"  # Still starting up
        
        return build_status(
            name="feature_builder",
            type="internal",
            status=status,
            request_count=bars_processed,
            extras={
                "bars_processed": bars_processed,
                "metrics_emitted": metrics_emitted,
                "symbols_tracked": symbols_tracked,
                "uptime_seconds": round(uptime, 1),
            },
        )
