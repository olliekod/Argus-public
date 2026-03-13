"""
Polymarket CLOB Connector (Stream 3)
====================================

Fetches order-book snapshots and probability data from the
Polymarket CLOB (Central Limit Order Book) REST API.

Provides:
* Order-book snapshots (bids/asks) for tracked tokens
* Mid-price / probability extraction
* Publishes OrderBookEvent-style MetricEvents to the bus

A future iteration can upgrade to WebSocket streaming; the REST
poller is simpler and sufficient for the watchlist use-case.

Safety constraints
------------------
* Read-only — no order placement.
* Rate-limited polling with configurable interval.
* Bounded internal state (only tracks watchlisted tokens).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from ..core.bus import EventBus
from ..core.events import (
    MetricEvent,
    TOPIC_MARKET_METRICS,
    SCHEMA_VERSION,
)

logger = logging.getLogger("argus.polymarket.clob")

_CLOB_BASE = "https://clob.polymarket.com"
_DEFAULT_POLL_INTERVAL = 30  # seconds
_REQUEST_TIMEOUT = 15.0


class PolymarketCLOBClient:
    """REST client for the Polymarket CLOB order-book API.

    Parameters
    ----------
    event_bus : EventBus, optional
        If provided, price / probability updates are published.
    poll_interval : float
        Seconds between order-book polls.
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
        self._token_ids: List[str] = []          # active watchlist
        self._books: Dict[str, Dict[str, Any]] = {}  # token_id → latest book
        self._start_time = time.time()
        self._polls_total = 0
        self._errors_total = 0
        self._last_poll_ts: Optional[float] = None

    # ── lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=_CLOB_BASE,
            timeout=_REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        self._running = True
        logger.info("PolymarketCLOBClient started")

    async def stop(self) -> None:
        self._running = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("PolymarketCLOBClient stopped")

    # ── watchlist management ─────────────────────────────────

    def set_watchlist(self, token_ids: List[str]) -> None:
        """Set the list of token IDs to track."""
        self._token_ids = list(token_ids)
        logger.info("CLOB watchlist updated: %d tokens", len(self._token_ids))

    def add_token(self, token_id: str) -> None:
        if token_id not in self._token_ids:
            self._token_ids.append(token_id)

    def remove_token(self, token_id: str) -> None:
        if token_id in self._token_ids:
            self._token_ids.remove(token_id)

    # ── polling loop ─────────────────────────────────────────

    async def poll_loop(self) -> None:
        """Continuous polling loop — run as asyncio.Task."""
        while self._running:
            try:
                await self.fetch_books()
            except Exception:
                self._errors_total += 1
                logger.exception("CLOB poll failed")
            await asyncio.sleep(self._poll_interval)

    async def fetch_books(self) -> Dict[str, Dict[str, Any]]:
        """Fetch order books for all watchlisted tokens."""
        if not self._client or not self._token_ids:
            return {}

        now = time.time()
        results: Dict[str, Dict[str, Any]] = {}

        for token_id in self._token_ids:
            try:
                book = await self._fetch_book(token_id)
                if book:
                    results[token_id] = book
                    self._books[token_id] = book

                    # Extract mid-price as probability
                    best_bid = self._best_price(book.get("bids", []))
                    best_ask = self._best_price(book.get("asks", []))

                    if best_bid is not None and best_ask is not None:
                        mid = (best_bid + best_ask) / 2.0
                        spread = best_ask - best_bid

                        if self._bus:
                            # Probability metric
                            self._bus.publish(TOPIC_MARKET_METRICS, MetricEvent(
                                symbol=f"PM_TOKEN:{token_id[:16]}",
                                metric="probability",
                                value=mid,
                                timestamp=now,
                                source="polymarket_clob",
                                extra={
                                    "token_id": token_id,
                                    "best_bid": best_bid,
                                    "best_ask": best_ask,
                                    "spread": round(spread, 4),
                                    "bid_depth": len(book.get("bids", [])),
                                    "ask_depth": len(book.get("asks", [])),
                                },
                            ))
            except Exception:
                self._errors_total += 1
                logger.debug("Failed to fetch book for token %s", token_id[:16])

        self._polls_total += 1
        self._last_poll_ts = now
        return results

    async def _fetch_book(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single token's order book."""
        if not self._client:
            return None
        resp = await self._client.get("/book", params={"token_id": token_id})
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _best_price(side: List[Dict[str, Any]]) -> Optional[float]:
        """Extract the best price from an order-book side."""
        if not side:
            return None
        try:
            return float(side[0].get("price", 0))
        except (ValueError, TypeError, IndexError):
            return None

    # ── accessors ────────────────────────────────────────────

    def get_cached_book(self, token_id: str) -> Optional[Dict[str, Any]]:
        return self._books.get(token_id)

    def get_probability(self, token_id: str) -> Optional[float]:
        """Return last mid-price probability for a token."""
        book = self._books.get(token_id)
        if not book:
            return None
        bid = self._best_price(book.get("bids", []))
        ask = self._best_price(book.get("asks", []))
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    # ── health status ────────────────────────────────────────

    def get_health_status(self) -> Dict[str, Any]:
        from ..core.status import build_status
        now = time.time()
        age = (now - self._last_poll_ts) if self._last_poll_ts else None
        status = "ok"
        if not self._token_ids:
            status = "unknown"
        elif self._last_poll_ts is None:
            status = "unknown"
        elif age and age > self._poll_interval * 3:
            status = "degraded"

        return build_status(
            name="polymarket_clob",
            type="rest",
            status=status,
            last_success_ts=self._last_poll_ts,
            request_count=self._polls_total,
            error_count=self._errors_total,
            age_seconds=round(age, 1) if age else None,
            extras={
                "watchlist_size": len(self._token_ids),
                "books_cached": len(self._books),
                "poll_interval": self._poll_interval,
            },
        )
