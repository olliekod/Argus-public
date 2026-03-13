"""
Polymarket Gamma REST Connector (Stream 3)
==========================================

Discovers and fetches market metadata from the Polymarket Gamma API.

The Gamma API provides:
* Market listings (condition IDs, question text, outcomes)
* Token metadata for each outcome
* Resolution status and end dates

This connector is **read-only / poll-based** — it does not place orders
or interact with the CLOB.

Safety constraints
------------------
* No trade execution — read-only discovery.
* Rate-limited polling (configurable interval, default 60s).
* Publishes MarketMetadataEvent to the event bus for downstream use.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from ..core.bus import EventBus
from ..core.events import (
    MetricEvent,
    TOPIC_MARKET_METRICS,
    SCHEMA_VERSION,
)

logger = logging.getLogger("argus.polymarket.gamma")

# Polymarket Gamma API base URL
_GAMMA_BASE = "https://gamma-api.polymarket.com"
_DEFAULT_POLL_INTERVAL = 60  # seconds
_REQUEST_TIMEOUT = 15.0


class PolymarketGammaClient:
    """REST client for the Polymarket Gamma (discovery) API.

    Parameters
    ----------
    event_bus : EventBus, optional
        If provided, discovered markets are published as MetricEvents.
    poll_interval : float
        Seconds between discovery polls.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._bus = event_bus
        self._poll_interval = poll_interval
        self._client: Optional[httpx.AsyncClient] = None
        self._running = False
        self._markets: Dict[str, Dict[str, Any]] = {}  # condition_id → metadata
        self._start_time = time.time()
        self._polls_total = 0
        self._errors_total = 0
        self._last_poll_ts: Optional[float] = None

    # ── lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Start the async HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=_GAMMA_BASE,
            timeout=_REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        self._running = True
        logger.info("PolymarketGammaClient started")

    async def stop(self) -> None:
        self._running = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("PolymarketGammaClient stopped")

    # ── polling loop (called by orchestrator) ────────────────

    async def poll_loop(self) -> None:
        """Continuous polling loop — intended to run as an asyncio.Task."""
        while self._running:
            try:
                await self.fetch_markets()
            except Exception:
                self._errors_total += 1
                logger.exception("Gamma poll failed")
            await asyncio.sleep(self._poll_interval)

    # ── API calls ────────────────────────────────────────────

    async def fetch_markets(
        self,
        limit: int = 50,
        active: bool = True,
        closed: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch market listings from Gamma API.

        Returns a list of market metadata dicts.
        """
        if not self._client:
            return []

        params: Dict[str, Any] = {
            "limit": limit,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }

        resp = await self._client.get("/markets", params=params)
        resp.raise_for_status()
        markets = resp.json()

        now = time.time()
        self._polls_total += 1
        self._last_poll_ts = now

        # Index by condition_id and publish
        for mkt in markets:
            cid = mkt.get("condition_id") or mkt.get("id", "")
            if not cid:
                continue

            self._markets[cid] = {
                "condition_id": cid,
                "question": mkt.get("question", ""),
                "slug": mkt.get("slug", ""),
                "active": mkt.get("active", False),
                "closed": mkt.get("closed", False),
                "end_date_iso": mkt.get("end_date_iso"),
                "tokens": mkt.get("tokens", []),
                "volume": mkt.get("volume", 0),
                "liquidity": mkt.get("liquidity", 0),
                "fetched_at": now,
            }

            # Publish as a MetricEvent if bus is available
            if self._bus:
                slug = mkt.get("slug", cid[:16])
                self._bus.publish(TOPIC_MARKET_METRICS, MetricEvent(
                    symbol=f"PM:{slug}",
                    metric="polymarket_volume",
                    value=float(mkt.get("volume", 0) or 0),
                    timestamp=now,
                    source="polymarket_gamma",
                    extra={
                        "condition_id": cid,
                        "question": mkt.get("question", "")[:200],
                        "liquidity": mkt.get("liquidity", 0),
                        "active": mkt.get("active", False),
                    },
                ))

        logger.debug("Gamma poll fetched %d markets", len(markets))
        return markets

    async def fetch_market(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single market by condition ID."""
        if not self._client:
            return None
        resp = await self._client.get(f"/markets/{condition_id}")
        resp.raise_for_status()
        return resp.json()

    # ── accessors ────────────────────────────────────────────

    def get_cached_markets(self) -> Dict[str, Dict[str, Any]]:
        """Return the latest cached market metadata."""
        return dict(self._markets)

    def get_market(self, condition_id: str) -> Optional[Dict[str, Any]]:
        return self._markets.get(condition_id)

    # ── health status ────────────────────────────────────────

    def get_health_status(self) -> Dict[str, Any]:
        from ..core.status import build_status
        now = time.time()
        age = (now - self._last_poll_ts) if self._last_poll_ts else None
        status = "ok"
        if self._last_poll_ts is None:
            status = "unknown"
        elif age and age > self._poll_interval * 3:
            status = "degraded"
        if self._errors_total > 5:
            status = "degraded"

        return build_status(
            name="polymarket_gamma",
            type="rest",
            status=status,
            last_success_ts=self._last_poll_ts,
            request_count=self._polls_total,
            error_count=self._errors_total,
            age_seconds=round(age, 1) if age else None,
            extras={
                "markets_cached": len(self._markets),
                "poll_interval": self._poll_interval,
            },
        )
