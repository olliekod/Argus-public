"""
Polymarket Watchlist Service (Stream 3)
=======================================

Periodically polls the Gamma API and syncs a watchlist of active,
high-volume Polymarket markets to the CLOB connector for order-book
tracking.

Workflow:
1. Fetch markets from Gamma (discovery).
2. Filter by volume / liquidity / keywords.
3. Extract token IDs from matching markets.
4. Push token IDs to the CLOB client's watchlist.

Safety constraints
------------------
* Read-only — no order placement or trade execution.
* Bounded watchlist size (configurable max).
* Runs on its own poll cadence separate from Gamma/CLOB.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("argus.polymarket.watchlist")

_DEFAULT_SYNC_INTERVAL = 300  # seconds (5 minutes)
_DEFAULT_MAX_WATCHLIST = 50
_DEFAULT_MIN_VOLUME = 10_000  # USD volume minimum


class PolymarketWatchlistService:
    """Syncs high-interest Polymarket markets to the CLOB tracker.

    Parameters
    ----------
    gamma_client
        PolymarketGammaClient instance (for discovery).
    clob_client
        PolymarketCLOBClient instance (watchlist target).
    sync_interval : float
        Seconds between watchlist syncs.
    max_watchlist : int
        Maximum number of tokens to track.
    min_volume : float
        Minimum market volume to qualify for watchlist.
    keywords : list of str, optional
        Only include markets whose question contains one of these
        keywords (case-insensitive).  Empty list = no keyword filter.
    """

    def __init__(
        self,
        gamma_client: Any,
        clob_client: Any,
        sync_interval: float = _DEFAULT_SYNC_INTERVAL,
        max_watchlist: int = _DEFAULT_MAX_WATCHLIST,
        min_volume: float = _DEFAULT_MIN_VOLUME,
        keywords: Optional[List[str]] = None,
    ) -> None:
        self._gamma = gamma_client
        self._clob = clob_client
        self._sync_interval = sync_interval
        self._max_watchlist = max_watchlist
        self._min_volume = min_volume
        self._keywords = [kw.lower() for kw in (keywords or [])]
        self._running = False
        self._last_sync_ts: Optional[float] = None
        self._syncs_total = 0
        self._current_watchlist: List[str] = []

    # ── lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        logger.info(
            "PolymarketWatchlistService started (interval=%ds, max=%d, min_vol=%.0f)",
            self._sync_interval, self._max_watchlist, self._min_volume,
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("PolymarketWatchlistService stopped")

    async def sync_loop(self) -> None:
        """Continuous sync loop — run as asyncio.Task."""
        while self._running:
            try:
                await self.sync()
            except Exception:
                logger.exception("Watchlist sync failed")
            await asyncio.sleep(self._sync_interval)

    # ── sync logic ───────────────────────────────────────────

    async def sync(self) -> List[str]:
        """Run one sync cycle: discover → filter → push to CLOB."""
        cached = self._gamma.get_cached_markets()
        if not cached:
            logger.debug("No cached markets — skipping sync")
            return self._current_watchlist

        # Filter and rank markets
        candidates = []
        for cid, meta in cached.items():
            if not meta.get("active", False):
                continue
            vol = float(meta.get("volume", 0) or 0)
            if vol < self._min_volume:
                continue
            question = (meta.get("question") or "").lower()
            if self._keywords and not any(kw in question for kw in self._keywords):
                continue
            candidates.append((vol, cid, meta))

        # Sort by volume descending, take top N
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:self._max_watchlist]

        # Extract token IDs from market metadata
        token_ids: List[str] = []
        for _, cid, meta in top:
            tokens = meta.get("tokens", [])
            for tok in tokens:
                tid = tok.get("token_id") or tok.get("id", "")
                if tid and tid not in token_ids:
                    token_ids.append(tid)
                    if len(token_ids) >= self._max_watchlist:
                        break
            if len(token_ids) >= self._max_watchlist:
                break

        # Push to CLOB client
        self._clob.set_watchlist(token_ids)
        self._current_watchlist = token_ids
        self._last_sync_ts = time.time()
        self._syncs_total += 1

        logger.info(
            "Watchlist synced: %d markets → %d tokens",
            len(top), len(token_ids),
        )
        return token_ids

    # ── accessors ────────────────────────────────────────────

    def get_watchlist(self) -> List[str]:
        return list(self._current_watchlist)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "polymarket_watchlist",
            "running": self._running,
            "syncs_total": self._syncs_total,
            "last_sync_ts": self._last_sync_ts,
            "watchlist_size": len(self._current_watchlist),
            "max_watchlist": self._max_watchlist,
            "min_volume": self._min_volume,
            "keywords": self._keywords,
        }
