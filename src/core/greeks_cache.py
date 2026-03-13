"""
Greeks Cache
============

Thread-safe in-memory cache for DXLink Greeks events.

Stores the latest implied volatility and Greeks per option symbol,
enabling snapshot IV enrichment without persisting tick-level data.

The cache is keyed by option symbol (e.g. ``.SPY250321P590``) and
stores the most recent ``(volatility, recv_ts_ms)`` tuple for each.

Usage::

    cache = GreeksCache()
    cache.update(".SPY250321P590", volatility=0.22, recv_ts_ms=1700000000000)
    iv = cache.get_atm_iv("SPY", underlying_price=595.0, as_of_ms=1700000060000)
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.greeks_cache")


@dataclass(frozen=True, slots=True)
class CachedGreek:
    """Single cached Greeks observation."""
    event_symbol: str       # e.g. ".SPY250321P590"
    volatility: float       # Implied volatility (annualized decimal)
    recv_ts_ms: int         # Local receipt time (epoch ms)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


def _parse_option_symbol(event_symbol: str) -> Optional[Tuple[str, str, float, str]]:
    """Parse a DXLink-style option symbol into (underlying, option_type, strike, expiry).

    Supports formats like:
    - ``.SPY250321P590``   → ("SPY", "PUT", 590.0, "250321")
    - ``.SPY250321C595``   → ("SPY", "CALL", 595.0, "250321")
    - ``.IBIT250321P55``   → ("IBIT", "PUT", 55.0, "250321")

    The fourth element is the raw YYMMDD expiration string.

    Returns None if the symbol cannot be parsed.
    """
    # Pattern: .UNDERLYING YYMMDD [CP] STRIKE
    m = re.match(
        r"^\.?([A-Z]+)(\d{6})([CP])(\d+(?:\.\d+)?)$",
        event_symbol.strip(),
    )
    if not m:
        return None
    underlying = m.group(1)
    expiry = m.group(2)
    opt_type = "PUT" if m.group(3) == "P" else "CALL"
    strike = float(m.group(4))
    return underlying, opt_type, strike, expiry


def _yymmdd_to_epoch_ms(yymmdd: str) -> int:
    """Convert a YYMMDD string to midnight-UTC epoch milliseconds."""
    from datetime import datetime, timezone
    dt = datetime.strptime(yymmdd, "%y%m%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# One calendar day in milliseconds — used for date-level matching.
_ONE_DAY_MS = 86_400_000


class GreeksCache:
    """Thread-safe in-memory cache of the latest DXLink Greeks per option symbol.

    Performance characteristics:
    - O(1) update per Greeks event
    - O(N) ATM IV lookup where N = number of cached symbols for the underlying
    - No DB writes; memory only
    - Configurable max age for stale eviction
    """

    def __init__(self, max_age_ms: int = 600_000) -> None:
        """
        Args:
            max_age_ms: Maximum age (in ms) before a cached entry is
                considered stale and ignored during lookups. Default 10 min.
        """
        self._lock = threading.Lock()
        self._cache: Dict[str, CachedGreek] = {}
        self._max_age_ms = max_age_ms
        self._last_update_ms: int = 0

    def update(
        self,
        event_symbol: str,
        volatility: Optional[float],
        recv_ts_ms: Optional[int] = None,
        *,
        delta: Optional[float] = None,
        gamma: Optional[float] = None,
        theta: Optional[float] = None,
        vega: Optional[float] = None,
    ) -> None:
        """Store or update a Greeks observation for an option symbol.

        Only updates if the new observation is more recent than the
        existing one (based on ``recv_ts_ms``).

        Args:
            event_symbol: DXLink option symbol (e.g. ``.SPY250321P590``).
            volatility: Implied volatility (annualized decimal). Ignored if None/NaN.
            recv_ts_ms: Local receipt time. Defaults to now if not provided.
            delta, gamma, theta, vega: Optional Greeks values.
        """
        if volatility is None or (isinstance(volatility, float) and math.isnan(volatility)):
            return
        if volatility <= 0 or volatility > 10.0:
            return  # Reject nonsensical IV values

        if recv_ts_ms is None:
            recv_ts_ms = int(time.time() * 1000)

        entry = CachedGreek(
            event_symbol=event_symbol,
            volatility=volatility,
            recv_ts_ms=recv_ts_ms,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
        )

        with self._lock:
            existing = self._cache.get(event_symbol)
            if existing is None or recv_ts_ms >= existing.recv_ts_ms:
                self._cache[event_symbol] = entry
                if recv_ts_ms > self._last_update_ms:
                    self._last_update_ms = recv_ts_ms

    def get_atm_iv(
        self,
        underlying: str,
        underlying_price: float,
        as_of_ms: int,
        *,
        option_type: str = "PUT",
        expiration_ms: Optional[int] = None,
    ) -> Optional[float]:
        """Find ATM implied volatility from cached Greeks.

        Searches all cached entries for the given underlying, filters to
        those with ``recv_ts_ms <= as_of_ms``, and returns the IV of the
        strike nearest to ``underlying_price``.

        When ``expiration_ms`` is supplied, only Greeks whose YYMMDD
        expiration matches the same calendar day (UTC) are considered.
        This prevents cross-expiration IV contamination when the cache
        holds multiple expirations for the same underlying.

        Args:
            underlying: Underlying symbol (e.g. ``"SPY"``).
            underlying_price: Current underlying price for ATM determination.
            as_of_ms: Only consider Greeks received at or before this time.
            option_type: ``"PUT"`` or ``"CALL"``. Defaults to ``"PUT"``.
            expiration_ms: If provided, restrict to options expiring on this
                date (midnight-UTC epoch ms).

        Returns:
            ATM implied volatility (annualized decimal), or None if no
            suitable cached entry exists.
        """
        if underlying_price <= 0:
            return None

        target_type = option_type.upper()
        target_day: Optional[int] = None
        if expiration_ms is not None:
            target_day = expiration_ms // _ONE_DAY_MS

        best_iv: Optional[float] = None
        best_dist = float("inf")

        with self._lock:
            for sym, entry in self._cache.items():
                # Time gating: only use entries received at or before as_of_ms
                if entry.recv_ts_ms > as_of_ms:
                    continue
                # Staleness check
                if as_of_ms - entry.recv_ts_ms > self._max_age_ms:
                    continue

                parsed = _parse_option_symbol(sym)
                if parsed is None:
                    continue
                sym_underlying, opt_type, strike, expiry = parsed
                if sym_underlying != underlying:
                    continue
                if opt_type != target_type:
                    continue

                # Expiration filter: match on calendar day
                if target_day is not None:
                    sym_day = _yymmdd_to_epoch_ms(expiry) // _ONE_DAY_MS
                    if sym_day != target_day:
                        continue

                dist = abs(strike - underlying_price)
                if dist < best_dist:
                    best_dist = dist
                    best_iv = entry.volatility

        return best_iv

    def get_greeks_for_strike(
        self,
        underlying: str,
        strike: float,
        option_type: str,
        as_of_ms: int,
        *,
        expiration_ms: Optional[int] = None,
    ) -> Optional[CachedGreek]:
        """Get the cached Greeks entry for a specific strike.

        Args:
            underlying: Underlying symbol.
            strike: Strike price.
            option_type: ``"PUT"`` or ``"CALL"``.
            as_of_ms: Time gate.
            expiration_ms: If provided, restrict to options expiring on
                this date (midnight-UTC epoch ms).

        Returns:
            CachedGreek or None.
        """
        target_type = option_type.upper()
        target_day: Optional[int] = None
        if expiration_ms is not None:
            target_day = expiration_ms // _ONE_DAY_MS

        best: Optional[CachedGreek] = None
        best_dist = float("inf")

        with self._lock:
            for sym, entry in self._cache.items():
                if entry.recv_ts_ms > as_of_ms:
                    continue
                if as_of_ms - entry.recv_ts_ms > self._max_age_ms:
                    continue

                parsed = _parse_option_symbol(sym)
                if parsed is None:
                    continue
                sym_underlying, opt_type, sym_strike, expiry = parsed
                if sym_underlying != underlying or opt_type != target_type:
                    continue

                if target_day is not None:
                    sym_day = _yymmdd_to_epoch_ms(expiry) // _ONE_DAY_MS
                    if sym_day != target_day:
                        continue

                dist = abs(sym_strike - strike)
                if dist < best_dist:
                    best_dist = dist
                    best = entry

        return best

    @property
    def size(self) -> int:
        """Number of cached entries."""
        with self._lock:
            return len(self._cache)

    @property
    def last_update_ms(self) -> int:
        """Epoch ms of the most recent cache update (0 if never updated)."""
        with self._lock:
            return self._last_update_ms

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._cache.clear()

    def evict_stale(self, now_ms: Optional[int] = None) -> int:
        """Remove entries older than ``max_age_ms``.

        Returns the number of entries evicted.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        evicted = 0
        with self._lock:
            stale_keys = [
                k for k, v in self._cache.items()
                if now_ms - v.recv_ts_ms > self._max_age_ms
            ]
            for k in stale_keys:
                del self._cache[k]
                evicted += 1

        if evicted:
            logger.debug("Evicted %d stale Greeks entries", evicted)
        return evicted


def enrich_snapshot_iv(
    snapshot: "OptionChainSnapshotEvent",
    greeks_cache: Any,
) -> "OptionChainSnapshotEvent":
    """Enrich an option chain snapshot with ATM IV from the Greeks cache.

    If the snapshot already has a valid ``atm_iv``, it is returned unchanged.
    Otherwise, attempts to find ATM IV from the cache using:
    1. Provider IV from cached Greeks (nearest put to underlying price).
    2. Falls back to None if no cached Greeks are available.

    The lookup is gated by ``recv_ts_ms``: only Greeks received at or
    before the snapshot's receipt time are considered.

    This function does NOT mutate the input; it returns a new snapshot
    (``OptionChainSnapshotEvent`` is frozen).

    Args:
        snapshot: The option chain snapshot to enrich.
        greeks_cache: In-memory Greeks cache populated by DXLink events.

    Returns:
        A new OptionChainSnapshotEvent with ``atm_iv`` populated, or the
        original snapshot if enrichment was not needed or not possible.
    """
    from .option_events import OptionChainSnapshotEvent

    # Preferred path: IVConsensusEngine-compatible object
    if hasattr(greeks_cache, "get_atm_consensus"):
        put_res = greeks_cache.get_atm_consensus(
            underlying=snapshot.symbol,
            option_type="PUT",
            expiration_ms=snapshot.expiration_ms,
            as_of_ms=snapshot.recv_ts_ms,
        )
        call_res = greeks_cache.get_atm_consensus(
            underlying=snapshot.symbol,
            option_type="CALL",
            expiration_ms=snapshot.expiration_ms,
            as_of_ms=snapshot.recv_ts_ms,
        )
        chosen = put_res if put_res.consensus_iv is not None else call_res
        if chosen.consensus_iv is not None:
            return OptionChainSnapshotEvent(
                symbol=snapshot.symbol,
                expiration_ms=snapshot.expiration_ms,
                underlying_price=snapshot.underlying_price,
                underlying_bid=snapshot.underlying_bid,
                underlying_ask=snapshot.underlying_ask,
                puts=snapshot.puts,
                calls=snapshot.calls,
                n_strikes=snapshot.n_strikes,
                atm_iv=chosen.consensus_iv,
                timestamp_ms=snapshot.timestamp_ms,
                source_ts_ms=snapshot.source_ts_ms,
                recv_ts_ms=snapshot.recv_ts_ms,
                provider=snapshot.provider,
                snapshot_id=snapshot.snapshot_id,
                sequence_id=snapshot.sequence_id,
                v=snapshot.v,
            )

    # Already has valid IV — no enrichment needed
    if snapshot.atm_iv is not None and snapshot.atm_iv > 0:
        return snapshot

    if snapshot.underlying_price <= 0:
        logger.debug(
            "Cannot enrich IV for %s: no underlying price",
            snapshot.symbol,
        )
        return snapshot

    # Legacy path: only when caller passed a GreeksCache (has get_atm_iv).
    # When greeks_cache is IVConsensusEngine-only, we already tried consensus above.
    if not hasattr(greeks_cache, "get_atm_iv"):
        logger.debug(
            "No consensus IV for %s at recv_ts=%d; no legacy cache, skipping enrichment",
            snapshot.symbol, snapshot.recv_ts_ms,
        )
        return snapshot

    # Try provider IV from DXLink Greeks cache (match expiration to prevent
    # cross-expiration contamination)
    atm_iv = greeks_cache.get_atm_iv(
        underlying=snapshot.symbol,
        underlying_price=snapshot.underlying_price,
        as_of_ms=snapshot.recv_ts_ms,
        option_type="PUT",
        expiration_ms=snapshot.expiration_ms,
    )

    if atm_iv is None:
        # Try calls as secondary (same expiration)
        atm_iv = greeks_cache.get_atm_iv(
            underlying=snapshot.symbol,
            underlying_price=snapshot.underlying_price,
            as_of_ms=snapshot.recv_ts_ms,
            option_type="CALL",
            expiration_ms=snapshot.expiration_ms,
        )

    if atm_iv is None:
        logger.debug(
            "No cached Greeks IV for %s at recv_ts=%d (cache_size=%d)",
            snapshot.symbol, snapshot.recv_ts_ms, greeks_cache.size,
        )
        return snapshot

    logger.info(
        "Enriched %s snapshot with ATM IV=%.4f from DXLink Greeks cache",
        snapshot.symbol, atm_iv,
    )

    # Create new frozen snapshot with enriched atm_iv
    return OptionChainSnapshotEvent(
        symbol=snapshot.symbol,
        expiration_ms=snapshot.expiration_ms,
        underlying_price=snapshot.underlying_price,
        underlying_bid=snapshot.underlying_bid,
        underlying_ask=snapshot.underlying_ask,
        puts=snapshot.puts,
        calls=snapshot.calls,
        n_strikes=snapshot.n_strikes,
        atm_iv=atm_iv,
        timestamp_ms=snapshot.timestamp_ms,
        source_ts_ms=snapshot.source_ts_ms,
        recv_ts_ms=snapshot.recv_ts_ms,
        provider=snapshot.provider,
        snapshot_id=snapshot.snapshot_id,
        sequence_id=snapshot.sequence_id,
        v=snapshot.v,
    )
