"""
GlobalRiskFlow Updater
======================

Periodically fetches Alpha Vantage daily bars, computes the
``global_risk_flow`` metric, and publishes it to the event bus
as an :class:`ExternalMetricEvent`.

The regime detector subscribes to these events and merges the
metric into ``metrics_json`` on subsequent market regime emissions.

This module is a no-op when ``alphavantage.enabled`` is false in config.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .bus import EventBus
from .events import ExternalMetricEvent, TOPIC_EXTERNAL_METRICS
from .global_risk_flow import (
    ASIA_SYMBOLS,
    EUROPE_SYMBOLS,
    FX_RISK_SYMBOL,
    compute_global_risk_flow,
)

logger = logging.getLogger("argus.global_risk_flow_updater")

# All symbols needed for the risk-flow computation
_ALL_DAILY_SYMBOLS = list(ASIA_SYMBOLS) + list(EUROPE_SYMBOLS)
_ALL_FX_PAIRS = [FX_RISK_SYMBOL]  # stored as "FX:USDJPY" in AV bars


class GlobalRiskFlowUpdater:
    """Fetch AV daily bars from DB, compute risk flow, publish to bus.

    DB-only (budget-safe): reads from market_bars seeded by the
    Alpha Vantage backfill/collector; never calls the Alpha Vantage API.

    Parameters
    ----------
    bus : EventBus
        Event bus for publishing ``ExternalMetricEvent``.
    db : Database
        Database instance for ``get_bars_daily_for_risk_flow``.
    config : dict
        Full Argus config.  Reads ``exchanges.alphavantage``.
    """

    def __init__(
        self,
        bus: EventBus,
        db: Any,
        config: Dict[str, Any],
    ) -> None:
        self._bus = bus
        self._db = db

        av_cfg = (config.get("exchanges") or {}).get("alphavantage") or {}
        self._enabled: bool = bool(av_cfg.get("enabled", False))
        self._daily_symbols: List[str] = list(av_cfg.get("daily_symbols") or _ALL_DAILY_SYMBOLS)
        self._fx_pairs: List[str] = list(av_cfg.get("fx_pairs") or [])

        self._last_value: Optional[float] = None
        self._last_load_ms: int = 0
        self._bars_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Throttling/Cache settings
        self._cache_duration_ms = 300_000  # 5 minutes
        self._max_bars_per_symbol = 500

        if self._enabled:
            logger.info(
                "GlobalRiskFlowUpdater (DB-only) enabled — %d equity + %d FX symbols",
                len(self._daily_symbols), len(self._fx_pairs),
            )
        else:
            logger.info("GlobalRiskFlowUpdater disabled (alphavantage.enabled=false)")

    async def update(self) -> Optional[float]:
        """Fetch bars from DB, compute risk flow, publish event.

        Returns the computed risk-flow value, or None if skipped/unavailable.
        """
        if not self._enabled:
            return None

        now_ms = int(time.time() * 1000)

        try:
            bars_by_symbol = await self._fetch_all_bars(now_ms)
        except Exception as exc:
            logger.warning("Failed to load risk-flow bars from DB: %s", exc)
            return None

        if not bars_by_symbol:
            logger.info("GlobalRiskFlow: [UNAVAILABLE] - No daily bars in DB")
            return None

        value = compute_global_risk_flow(bars_by_symbol, now_ms)

        if value is None:
            logger.info("GlobalRiskFlow: [UNAVAILABLE] - Insufficient bars for return calculation")
            return None

        # Always log at INFO for operational visibility
        logger.info("GlobalRiskFlow: %.6f (sim_time_ms=%d)", value, now_ms)

        # Publish ExternalMetricEvent
        self._last_value = value
        event = ExternalMetricEvent(
            key="global_risk_flow",
            value=round(value, 8),
            timestamp_ms=now_ms,
        )
        self._bus.publish(TOPIC_EXTERNAL_METRICS, event)
        return value

    async def _fetch_all_bars(self, now_ms: int) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch daily bars from DB for all configured symbols."""
        # Use cache if fresh enough (protects DB during stress tests)
        if now_ms - self._last_load_ms < self._cache_duration_ms and self._bars_cache:
            return self._bars_cache

        equity_syms = list(self._daily_symbols)
        fx_syms = []
        for pair in self._fx_pairs:
            parts = pair.replace("/", "")
            fx_syms.append(f"FX:{parts[:3]}{parts[3:6]}")

        all_needed = equity_syms + fx_syms
        
        # Fetch from DB (Budget Safe - no AV call)
        # Using lookback_days=500 and the DB logic will cap it if we wanted, 
        # but the query itself is fast on indexed timestamp.
        bars_by_symbol = await self._db.get_bars_daily_for_risk_flow(
            source="alphavantage",
            symbols=all_needed,
            end_ms=now_ms,
            lookback_days=self._max_bars_per_symbol  # using this as a proxy for lookback limit
        )

        self._bars_cache = bars_by_symbol
        self._last_load_ms = now_ms
        return bars_by_symbol
