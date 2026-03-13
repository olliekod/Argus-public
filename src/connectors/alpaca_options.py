"""
Alpaca Options Connector
========================

Fetches options chain data from Alpaca Markets API.
Primary provider for IBIT/BITO options data.

Uses Alpaca's Options Data API which provides:
- Real-time and delayed quotes
- Greeks (when available)
- Contract metadata
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None

from ..core.option_events import (
    OptionContractEvent,
    OptionQuoteEvent,
    OptionChainSnapshotEvent,
)

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Current time as int milliseconds."""
    return int(time.time() * 1000)


def _poll_time_ms() -> int:
    """Poll time normalized to minute boundary (for timestamp_ms uniqueness).
    Use this for timestamp_ms so Alpaca and Tastytrade share the same
    granularity and ON CONFLICT(provider, symbol, timestamp_ms) behaves
    predictably. Keep recv_ts_ms as _now_ms() for accurate receipt time."""
    return (_now_ms() // 60_000) * 60_000


def _date_to_ms(date_str: str) -> int:
    """Convert date string (YYYY-MM-DD) to UTC midnight milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _compute_contract_id(option_symbol: str) -> str:
    """Compute deterministic contract ID from OCC option symbol."""
    return hashlib.sha256(option_symbol.encode()).hexdigest()[:16]


def _expiration_from_occ_symbol(occ_symbol: str) -> Optional[str]:
    """Derive YYYY-MM-DD expiration from OCC option symbol.
    Format: ROOT + YYMMDD + C|P + STRIKE(8). Example: SPY250221C00450000 -> 2025-02-21.
    """
    if not occ_symbol or len(occ_symbol) < 15:
        return None
    for i in range(len(occ_symbol) - 9, 5, -1):
        if occ_symbol[i] in ("C", "P") and occ_symbol[i + 1 : i + 9].isdigit():
            yy = int(occ_symbol[i - 6 : i - 4])
            mm = int(occ_symbol[i - 4 : i - 2])
            dd = int(occ_symbol[i - 2 : i])
            year = 2000 + yy if yy < 100 else yy
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                return f"{year:04d}-{mm:02d}-{dd:02d}"
            return None
    return None


@dataclass
class AlpacaOptionsConfig:
    """Configuration for Alpaca options connector."""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://data.alpaca.markets"
    paper: bool = True
    timeout_seconds: float = 30.0
    rate_limit_per_min: int = 200
    cache_ttl_seconds: int = 60
    feed: str = "indicative"  # "indicative" | "opra" (requires subscription)


class AlpacaOptionsConnector:
    """Alpaca options data connector.
    
    Provides deterministic options chain snapshots for tape recording.
    """
    
    PROVIDER = "alpaca"
    
    def __init__(self, config: Optional[AlpacaOptionsConfig] = None) -> None:
        self._config = config or AlpacaOptionsConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._sequence_id = 0
        self._cache: Dict[str, Tuple[int, Any]] = {}  # key -> (expire_ms, data)
        
        # Health metrics
        self._request_count = 0
        self._error_count = 0
        self._last_request_ms = 0
        self._last_latency_ms = 0.0
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if aiohttp is None:
            raise ImportError("aiohttp required: pip install aiohttp")
        if self._session is None or self._session.closed:
            headers = {
                "APCA-API-KEY-ID": self._config.api_key,
                "APCA-API-SECRET-KEY": self._config.api_secret,
            }
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session
    
    def _next_sequence_id(self) -> int:
        """Get next monotonic sequence ID."""
        self._sequence_id += 1
        return self._sequence_id
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            expire_ms, data = self._cache[key]
            if _now_ms() < expire_ms:
                return data
            del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any) -> None:
        """Set cache with TTL."""
        expire_ms = _now_ms() + (self._config.cache_ttl_seconds * 1000)
        self._cache[key] = (expire_ms, data)
    
    async def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to Alpaca API."""
        session = await self._get_session()
        url = f"{self._config.base_url}{endpoint}"
        
        start_ms = _now_ms()
        self._request_count += 1
        
        try:
            async with session.get(url, params=params) as resp:
                self._last_latency_ms = _now_ms() - start_ms
                self._last_request_ms = _now_ms()
                
                if resp.status != 200:
                    self._error_count += 1
                    text = await resp.text()
                    logger.warning("Alpaca API error %d: %s", resp.status, text[:200])
                    return {}
                
                return await resp.json()
        except Exception as e:
            self._error_count += 1
            logger.error("Alpaca request failed: %s", e)
            return {}
    
    async def get_expirations(self, symbol: str) -> List[str]:
        """Get available expiration dates for a symbol.
        
        Returns:
            List of expiration dates as YYYY-MM-DD strings, sorted ascending.
        """
        cache_key = f"exp:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Note: Alpaca doesn't have a dedicated expirations endpoint
        # We get this from the options chain snapshot
        data = await self._request(
            f"/v1beta1/options/snapshots/{symbol}",
            {"feed": self._config.feed, "limit": 1000}
        )

        if not data:
            logger.warning(
                "Expiration lookup failed for %s: empty API response (feed=%s)",
                symbol, self._config.feed,
            )
            return []

        expirations = set()
        snapshots_dict = data.get("snapshots", {})
        for opt_symbol, snapshot in snapshots_dict.items():
            # API may return expiration_date (snake_case) or expirationDate (camelCase)
            exp = snapshot.get("expiration_date") or snapshot.get("expirationDate")
            if exp:
                expirations.add(exp if isinstance(exp, str) else str(exp))
                continue
            # Indicative (and some) feeds often omit expiration; derive from OCC symbol
            try:
                exp_str = _expiration_from_occ_symbol(opt_symbol)
                if exp_str:
                    expirations.add(exp_str)
            except (ValueError, IndexError):
                pass

        if not expirations:
            logger.warning(
                "Expiration lookup for %s returned no expirations (snapshots=%d, feed=%s)",
                symbol, len(snapshots_dict), self._config.feed,
            )

        result = sorted(expirations)
        self._set_cache(cache_key, result)
        return result
    
    async def get_chain_raw(
        self,
        symbol: str,
        expiration: str,
    ) -> Dict[str, Any]:
        """Get raw options chain data from Alpaca.
        
        Args:
            symbol: Underlying symbol (IBIT, BITO)
            expiration: Expiration date (YYYY-MM-DD)
            
        Returns:
            Raw API response with snapshots.
        """
        # Alpaca snapshot endpoint
        data = await self._request(
            f"/v1beta1/options/snapshots/{symbol}",
            {
                "feed": self._config.feed,
                "expiration_date": expiration,
                "limit": 500,
            }
        )
        return data
    
    async def get_underlying_quote(self, symbol: str) -> Tuple[float, float, float]:
        """Get underlying stock quote.
        
        Returns:
            Tuple of (price, bid, ask)
        """
        # Use stocks snapshot endpoint
        data = await self._request(
            f"/v2/stocks/{symbol}/snapshot",
        )
        
        latest_trade = data.get("latestTrade", {})
        latest_quote = data.get("latestQuote", {})
        
        price = latest_trade.get("p", 0.0)
        bid = latest_quote.get("bp", 0.0)
        ask = latest_quote.get("ap", 0.0)
        
        return price, bid, ask
    
    async def build_chain_snapshot(
        self,
        symbol: str,
        expiration: str,
    ) -> Optional[OptionChainSnapshotEvent]:
        """Build deterministic chain snapshot for an expiration.
        
        Args:
            symbol: Underlying symbol
            expiration: Expiration date (YYYY-MM-DD)
            
        Returns:
            OptionChainSnapshotEvent or None if chain unavailable.
        """
        recv_ts_ms = _now_ms()  # Actual receipt time (for replay gating)
        timestamp_ms = _poll_time_ms()  # Minute-aligned for DB uniqueness
        expiration_ms = _date_to_ms(expiration)

        # Get underlying quote
        underlying_price, underlying_bid, underlying_ask = await self.get_underlying_quote(symbol)
        if underlying_price <= 0:
            logger.warning(
                "No underlying price for %s — cannot build snapshot (provider=%s)",
                symbol, self.PROVIDER,
            )
            return None

        # Get options chain
        raw_data = await self.get_chain_raw(symbol, expiration)
        snapshots = raw_data.get("snapshots", {})

        if not snapshots:
            logger.warning(
                "Empty options chain for %s exp=%s (feed=%s, provider=%s)",
                symbol, expiration, self._config.feed, self.PROVIDER,
            )
            return None

        puts: List[OptionQuoteEvent] = []
        calls: List[OptionQuoteEvent] = []
        skipped = 0
        missing_greeks = 0

        for option_symbol, snap in snapshots.items():
            quote_data = snap.get("latestQuote", {})
            greeks = snap.get("greeks", {})

            if not quote_data:
                logger.debug(
                    "Snapshot for %s missing latestQuote field (provider=%s)",
                    option_symbol, self.PROVIDER,
                )
                skipped += 1
                continue

            if not greeks:
                missing_greeks += 1

            # Parse OCC option symbol for strike and type
            # Format: IBIT250221P00045000
            try:
                strike, option_type = self._parse_occ_symbol(option_symbol)
            except ValueError:
                logger.debug("Cannot parse OCC symbol: %s", option_symbol)
                skipped += 1
                continue

            contract_id = _compute_contract_id(option_symbol)

            bid = quote_data.get("bp", 0.0)
            ask = quote_data.get("ap", 0.0)
            mid = (bid + ask) / 2 if bid and ask else 0.0

            quote = OptionQuoteEvent(
                contract_id=contract_id,
                symbol=symbol,
                strike=strike,
                expiration_ms=expiration_ms,
                option_type=option_type,
                bid=bid,
                ask=ask,
                last=quote_data.get("c", 0.0),
                mid=mid,
                volume=snap.get("dailyVolume", 0),
                open_interest=snap.get("openInterest", 0),
                iv=greeks.get("impliedVolatility"),
                delta=greeks.get("delta"),
                gamma=greeks.get("gamma"),
                theta=greeks.get("theta"),
                vega=greeks.get("vega"),
                timestamp_ms=timestamp_ms,
                source_ts_ms=quote_data.get("t", recv_ts_ms),
                recv_ts_ms=recv_ts_ms,
                provider=self.PROVIDER,
                sequence_id=self._next_sequence_id(),
            )

            if option_type == "PUT":
                puts.append(quote)
            else:
                calls.append(quote)

        if skipped:
            logger.debug(
                "%s chain %s exp=%s: skipped %d contracts (missing quote or unparseable)",
                self.PROVIDER, symbol, expiration, skipped,
            )
        if missing_greeks:
            logger.debug(
                "%s chain %s exp=%s: %d contracts missing greeks",
                self.PROVIDER, symbol, expiration, missing_greeks,
            )

        # Sort by strike for determinism
        puts.sort(key=lambda q: q.strike)
        calls.sort(key=lambda q: q.strike)

        if not puts and not calls:
            logger.warning(
                "Chain for %s exp=%s has no valid puts or calls after parsing (provider=%s)",
                symbol, expiration, self.PROVIDER,
            )
            return None

        # Compute ATM IV (from put closest to underlying price)
        atm_iv = None
        if puts:
            atm_put = min(puts, key=lambda q: abs(q.strike - underlying_price))
            atm_iv = atm_put.iv

        snapshot_id = f"{self.PROVIDER}_{symbol}_{expiration_ms}_{timestamp_ms}"

        return OptionChainSnapshotEvent(
            symbol=symbol,
            expiration_ms=expiration_ms,
            underlying_price=underlying_price,
            underlying_bid=underlying_bid,
            underlying_ask=underlying_ask,
            puts=tuple(puts),
            calls=tuple(calls),
            n_strikes=len(puts),
            atm_iv=atm_iv,
            timestamp_ms=timestamp_ms,
            source_ts_ms=recv_ts_ms,
            recv_ts_ms=recv_ts_ms,
            provider=self.PROVIDER,
            snapshot_id=snapshot_id,
            sequence_id=self._next_sequence_id(),
        )
    
    def _parse_occ_symbol(self, occ_symbol: str) -> Tuple[float, str]:
        """Parse OCC option symbol to extract strike and type.
        
        Format: SYMBOL + YYMMDD + C/P + STRIKE(8 digits, 3 decimals)
        Example: IBIT250221P00045000 → strike=45.0, type=PUT
        
        Returns:
            Tuple of (strike, option_type)
        """
        # Find the C or P character
        cp_idx = None
        for i in range(len(occ_symbol) - 8, 0, -1):
            if occ_symbol[i] in ("C", "P"):
                cp_idx = i
                break
        
        if cp_idx is None:
            raise ValueError(f"Cannot parse OCC symbol: {occ_symbol}")
        
        option_type = "CALL" if occ_symbol[cp_idx] == "C" else "PUT"
        strike_str = occ_symbol[cp_idx + 1:]
        
        # Strike is 8 digits with 3 implied decimals
        strike = int(strike_str) / 1000.0
        
        return strike, option_type
    
    async def get_expirations_in_range(
        self,
        symbol: str,
        min_dte: int = 7,
        max_dte: int = 21,
    ) -> List[Tuple[str, int]]:
        """Get expirations within DTE range.
        
        Returns:
            List of (expiration_date, dte) tuples.
        """
        expirations = await self.get_expirations(symbol)
        today = datetime.now(timezone.utc).date()
        
        results = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                results.append((exp_str, dte))
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get connector health metrics."""
        return {
            "provider": self.PROVIDER,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "last_request_ms": self._last_request_ms,
            "last_latency_ms": self._last_latency_ms,
            "sequence_id": self._sequence_id,
            "health": "ok" if self._error_count < max(1, self._request_count) * 0.1 else "degraded",
        }
    
    async def close(self) -> None:
        """Close the connector and release resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
