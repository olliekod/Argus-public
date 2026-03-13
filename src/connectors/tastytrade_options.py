"""
Tastytrade Options Snapshot Connector
======================================

REST-based snapshot polling for option chain data via Tastytrade API.
Uses TastytradeRestClient for authentication and nested chain fetching,
then normalizes into OptionChainSnapshotEvents compatible with the
existing schema used by ReplayHarness.

This does NOT replace DXLink streaming — it is for snapshot polling only.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..connectors.tastytrade_oauth import TastytradeOAuthClient
from ..connectors.tastytrade_rest import (
    RetryConfig,
    TastytradeError,
    TastytradeRestClient,
)
from ..core.options_normalize import normalize_tastytrade_nested_chain
from ..core.option_events import (
    OptionQuoteEvent,
    OptionChainSnapshotEvent,
)

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Current time as int milliseconds."""
    return int(time.time() * 1000)


def _poll_time_ms() -> int:
    """Poll time normalized to minute boundary (for timestamp_ms uniqueness).
    Matches Alpaca so both providers use the same granularity for
    ON CONFLICT(provider, symbol, timestamp_ms). recv_ts_ms stays _now_ms()."""
    return (_now_ms() // 60_000) * 60_000


def _date_to_ms(date_str: str) -> int:
    """Convert date string (YYYY-MM-DD) to UTC midnight milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _compute_contract_id(option_symbol: str) -> str:
    """Compute deterministic contract ID from option symbol."""
    return hashlib.sha256(option_symbol.encode()).hexdigest()[:16]


def _underlying_price_from_raw_chain(raw: Dict[str, Any]) -> float:
    """Extract underlying price from Tastytrade nested chain response if present.
    Tries common keys in data or first item. Returns 0.0 if not found.
    """
    if not raw or not isinstance(raw, dict):
        return 0.0
    data = raw.get("data", raw)
    if not isinstance(data, dict):
        return 0.0
    for key in ("underlying-price", "underlying_price", "spot-price", "spot_price", "mark", "last", "close"):
        val = data.get(key)
        if val is not None and val != "":
            try:
                p = float(val)
                if p > 0:
                    return p
            except (TypeError, ValueError):
                pass
    items = data.get("items") or data.get("item")
    if isinstance(items, list) and items:
        first = items[0] if isinstance(items[0], dict) else None
        if first:
            for key in ("underlying-price", "underlying_price", "spot-price", "mark", "last", "close"):
                val = first.get(key)
                if val is not None and val != "":
                    try:
                        p = float(val)
                        if p > 0:
                            return p
                    except (TypeError, ValueError):
                        pass
    return 0.0


@dataclass
class TastytradeOptionsConfig:
    """Configuration for Tastytrade options snapshot connector."""
    username: str = ""
    password: str = ""
    environment: str = "live"
    timeout_seconds: float = 20.0
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    symbols: Optional[List[str]] = None
    min_dte: int = 7
    max_dte: int = 21
    poll_interval_seconds: int = 60
    # OAuth 2.0 (preferred): Tastytrade deprecated session auth; use these when set
    oauth_client_id: str = ""
    oauth_client_secret: str = ""
    oauth_refresh_token: str = ""


class TastytradeOptionsConnector:
    """Tastytrade options snapshot connector.

    Fetches nested option chains via REST, normalizes them, and builds
    OptionChainSnapshotEvents compatible with option_chain_snapshots table
    and ReplayHarness.

    Thread-safety: all methods are synchronous (the REST client is sync).
    The orchestrator wraps calls in ``asyncio.to_thread`` or similar.
    """

    PROVIDER = "tastytrade"

    # Cooldown after login failure to avoid spamming the API
    _LOGIN_COOLDOWN_MS = 300_000  # 5 minutes

    def __init__(self, config: TastytradeOptionsConfig) -> None:
        self._config = config
        self._client: Optional[TastytradeRestClient] = None
        self._oauth_client: Optional[TastytradeOAuthClient] = None
        self._sequence_id = 0
        self._authenticated = False

        # OAuth: prefer when configured (Tastytrade deprecated session auth)
        if (
            config.oauth_client_id
            and config.oauth_client_secret
            and config.oauth_refresh_token
        ):
            self._oauth_client = TastytradeOAuthClient(
                client_id=config.oauth_client_id,
                client_secret=config.oauth_client_secret,
                refresh_token=config.oauth_refresh_token,
                timeout_s=config.timeout_seconds,
            )

        # Login failure tracking — prevents login spam on repeated poll cycles
        self._last_login_failure_ms: int = 0
        self._login_failure_reason: str = ""

        # Health metrics
        self._request_count = 0
        self._error_count = 0
        self._last_request_ms = 0
        self._last_latency_ms = 0.0

    def _next_sequence_id(self) -> int:
        """Get next monotonic sequence ID."""
        self._sequence_id += 1
        return self._sequence_id

    def _ensure_client(self) -> TastytradeRestClient:
        """Create and authenticate client if needed.

        Prefers OAuth 2.0 when oauth_client_id/secret/refresh_token are set
        (Tastytrade deprecated session auth; option-chain endpoints require OAuth).
        Respects a cooldown period after auth failures to avoid spamming the API.
        """
        now = _now_ms()

        # OAuth path: refresh token and ensure client has valid Bearer token
        if self._oauth_client is not None:
            if self._client is not None and self._authenticated:
                try:
                    result = self._oauth_client.refresh_access_token()
                    self._client.set_oauth_token(result.access_token)
                    return self._client
                except Exception as exc:
                    logger.warning("Tastytrade OAuth refresh failed, will recreate client: %s", exc)
                    self._authenticated = False
                    self._client = None
            # Enforce cooldown after prior OAuth failure
            if self._last_login_failure_ms > 0:
                elapsed = now - self._last_login_failure_ms
                if elapsed < self._LOGIN_COOLDOWN_MS:
                    remaining_s = (self._LOGIN_COOLDOWN_MS - elapsed) / 1000
                    raise TastytradeError(
                        f"Login cooldown active ({remaining_s:.0f}s remaining). "
                        f"Last failure: {self._login_failure_reason}"
                    )
            try:
                result = self._oauth_client.refresh_access_token()
                retry = RetryConfig(
                    max_attempts=self._config.max_attempts,
                    backoff_seconds=self._config.backoff_seconds,
                    backoff_multiplier=self._config.backoff_multiplier,
                )
                self._client = TastytradeRestClient(
                    environment=self._config.environment,
                    timeout_seconds=self._config.timeout_seconds,
                    retries=retry,
                    oauth_access_token=result.access_token,
                )
                self._authenticated = True
                self._last_login_failure_ms = 0
                self._login_failure_reason = ""
                logger.info(
                    "Tastytrade options connector authenticated via OAuth (env=%s)",
                    self._config.environment,
                )
                return self._client
            except Exception as exc:
                self._error_count += 1
                self._authenticated = False
                self._last_login_failure_ms = _now_ms()
                self._login_failure_reason = str(exc)
                logger.error(
                    "Tastytrade OAuth refresh failed (cooldown %ds): %s. "
                    "Check tastytrade_oauth2 client_id, client_secret, refresh_token in secrets.yaml.",
                    self._LOGIN_COOLDOWN_MS // 1000, exc,
                )
                raise TastytradeError(str(exc)) from exc

        # Session (legacy) path
        if self._client is not None and self._authenticated:
            return self._client

        # Enforce cooldown after login failure
        if self._last_login_failure_ms > 0:
            elapsed = now - self._last_login_failure_ms
            if elapsed < self._LOGIN_COOLDOWN_MS:
                remaining_s = (self._LOGIN_COOLDOWN_MS - elapsed) / 1000
                raise TastytradeError(
                    f"Login cooldown active ({remaining_s:.0f}s remaining). "
                    f"Last failure: {self._login_failure_reason}"
                )

        retry = RetryConfig(
            max_attempts=self._config.max_attempts,
            backoff_seconds=self._config.backoff_seconds,
            backoff_multiplier=self._config.backoff_multiplier,
        )
        self._client = TastytradeRestClient(
            username=self._config.username,
            password=self._config.password,
            environment=self._config.environment,
            timeout_seconds=self._config.timeout_seconds,
            retries=retry,
        )
        try:
            self._client.login()
            self._authenticated = True
            self._last_login_failure_ms = 0
            self._login_failure_reason = ""
            logger.info("Tastytrade options connector authenticated (session, env=%s)", self._config.environment)
        except TastytradeError as exc:
            self._error_count += 1
            self._authenticated = False
            self._last_login_failure_ms = _now_ms()
            self._login_failure_reason = str(exc)
            exc_str = str(exc)
            if "missing_request_token" in exc_str or "HTTP 400" in exc_str or "HTTP 401" in exc_str:
                logger.error(
                    "Tastytrade login failed (session auth deprecated? cooldown %ds): %s. "
                    "Use tastytrade_oauth2 in secrets.yaml and run OAuth bootstrap.",
                    self._LOGIN_COOLDOWN_MS // 1000, exc,
                )
            else:
                logger.error("Tastytrade login failed (transient, cooldown %ds): %s",
                             self._LOGIN_COOLDOWN_MS // 1000, exc)
            raise

        return self._client

    def fetch_nested_chain(self, symbol: str) -> Dict[str, Any]:
        """Fetch raw nested option chain for a symbol.

        Returns raw API response dict (or empty dict on failure).
        """
        start_ms = _now_ms()
        self._request_count += 1
        try:
            client = self._ensure_client()
            data = client.get_nested_option_chains(symbol)
            self._last_latency_ms = _now_ms() - start_ms
            self._last_request_ms = _now_ms()
            return data
        except TastytradeError as exc:
            self._error_count += 1
            self._last_latency_ms = _now_ms() - start_ms
            logger.error(
                "Tastytrade nested chain fetch failed for %s: %s",
                symbol, exc,
            )
            # Mark as unauthenticated so next call re-logins
            self._authenticated = False
            return {}
        except Exception as exc:
            self._error_count += 1
            logger.error(
                "Unexpected error fetching Tastytrade chain for %s: %s",
                symbol, exc,
            )
            self._authenticated = False
            return {}

    def _fetch_underlying_spot(self, symbol: str) -> Optional[float]:
        """Fetch underlying price from Tastytrade market-data snapshot. Returns None on failure."""
        try:
            client = self._ensure_client()
            snap = client.get_equity_snapshot(symbol)
            if not snap or not isinstance(snap, dict):
                return None
            for key in ("mark", "last", "close", "last-trade-price"):
                val = snap.get(key)
                if val is not None and val != "":
                    try:
                        p = float(val)
                        if p > 0:
                            return p
                    except (TypeError, ValueError):
                        pass
            return None
        except Exception:
            return None

    def get_dxlink_option_symbols(
        self,
        underlyings: List[str],
        min_dte: int = 7,
        max_dte: int = 21,
        max_total: int = 80,
    ) -> List[str]:
        """Return option symbols in DXLink streamer format for Greeks subscription.

        DXLink sends Greeks per option contract; subscribing to underlying symbols
        (e.g. SPY) does not yield Greeks. This fetches chains, normalizes, and
        samples streamer_symbol (e.g. .SPY250321P590) for near-ATM options in
        the given DTE range.

        max_total is an application cap only. dxFeed allows up to 100,000
        concurrent subscriptions; typical use is a few hundred symbols.

        Returns:
            List of symbols suitable for TastytradeStreamer(symbols=..., event_types=["Greeks"]).
        """
        result: List[str] = []
        seen: set = set()
        today = datetime.now(timezone.utc).date()

        for symbol in underlyings:
            if len(result) >= max_total:
                break
            raw = self.fetch_nested_chain(symbol)
            if not raw:
                continue
            normalized = normalize_tastytrade_nested_chain(raw)
            if not normalized:
                continue

            # Filter to DTE range and pick streamer symbol
            in_range: List[Dict[str, Any]] = []
            for c in normalized:
                exp = c.get("expiry")
                if not exp:
                    continue
                try:
                    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                except ValueError:
                    continue
                dte = (exp_date - today).days
                if min_dte <= dte <= max_dte:
                    in_range.append(c)

            if not in_range:
                continue

            # Prefer puts for ATM IV; sample by unique (expiry, strike) and streamer symbol
            # Sort by expiry then strike so we get front month and a spread of strikes
            in_range.sort(key=lambda x: (x.get("expiry") or "", x.get("strike") or 0))
            # Distribute max_total equally among underlyings
            n_underlyings = len(underlyings)
            per_underlying = max(4, max_total // max(1, n_underlyings))
            added = 0
            for c in in_range:
                if len(result) >= max_total or added >= per_underlying:
                    break
                streamer = (c.get("meta") or {}).get("streamer_symbol") or c.get("option_symbol")
                if not streamer or streamer in seen:
                    continue
                seen.add(streamer)
                result.append(streamer)
                added += 1

        return result

    def get_expirations_in_range(
        self,
        normalized: List[Dict[str, Any]],
        min_dte: int = 7,
        max_dte: int = 21,
    ) -> List[Tuple[str, int]]:
        """Filter normalized contracts to expirations within DTE range.

        Args:
            normalized: Output from normalize_tastytrade_nested_chain.
            min_dte: Minimum days to expiration.
            max_dte: Maximum days to expiration.

        Returns:
            List of (expiry_date_str, dte) tuples.
        """
        today = datetime.now(timezone.utc).date()
        expirations = sorted({
            c["expiry"] for c in normalized
            if isinstance(c, dict) and c.get("expiry")
        })
        results = []
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                results.append((exp_str, dte))
        return results

    def build_chain_snapshot(
        self,
        symbol: str,
        expiration: str,
        normalized: List[Dict[str, Any]],
        underlying_price: float = 0.0,
    ) -> Optional[OptionChainSnapshotEvent]:
        """Build an OptionChainSnapshotEvent from normalized contracts.

        Args:
            symbol: Underlying symbol (e.g. "SPY").
            expiration: Expiration date string (YYYY-MM-DD).
            normalized: Full normalized chain from normalize_tastytrade_nested_chain.
            underlying_price: Current underlying price (0.0 if unavailable).

        Returns:
            OptionChainSnapshotEvent or None if chain is empty.
        """
        recv_ts_ms = _now_ms()  # Actual receipt time (for replay gating)
        timestamp_ms = _poll_time_ms()  # Minute-aligned for DB uniqueness
        expiration_ms = _date_to_ms(expiration)

        # Filter to this expiration
        exp_contracts = [
            c for c in normalized
            if c.get("expiry") == expiration
        ]

        if not exp_contracts:
            logger.debug(
                "No contracts for %s exp=%s after filtering (provider=%s)",
                symbol, expiration, self.PROVIDER,
            )
            return None

        puts: List[OptionQuoteEvent] = []
        calls: List[OptionQuoteEvent] = []

        for contract in exp_contracts:
            option_symbol = contract.get("option_symbol", "")
            if not option_symbol:
                continue

            strike = contract.get("strike")
            if strike is None:
                continue

            right = contract.get("right", "")
            option_type = "CALL" if right == "C" else "PUT"

            contract_id = _compute_contract_id(str(option_symbol))

            # Tastytrade nested chains don't include live quotes
            # (quotes come from DXLink streaming). We create zero-quote
            # entries so the snapshot structure is valid for replay.
            quote = OptionQuoteEvent(
                contract_id=contract_id,
                symbol=symbol,
                strike=float(strike),
                expiration_ms=expiration_ms,
                option_type=option_type,
                bid=0.0,
                ask=0.0,
                last=0.0,
                mid=0.0,
                volume=0,
                open_interest=0,
                iv=None,
                delta=None,
                gamma=None,
                theta=None,
                vega=None,
                timestamp_ms=timestamp_ms,
                source_ts_ms=recv_ts_ms,
                recv_ts_ms=recv_ts_ms,
                provider=self.PROVIDER,
                sequence_id=self._next_sequence_id(),
            )

            if option_type == "PUT":
                puts.append(quote)
            else:
                calls.append(quote)

        # Sort by strike for determinism
        puts.sort(key=lambda q: q.strike)
        calls.sort(key=lambda q: q.strike)

        if not puts and not calls:
            logger.warning(
                "Chain for %s exp=%s has no valid puts or calls (provider=%s)",
                symbol, expiration, self.PROVIDER,
            )
            return None

        # Compute ATM IV (from put closest to underlying price, if available)
        atm_iv = None
        if puts and underlying_price > 0:
            atm_put = min(puts, key=lambda q: abs(q.strike - underlying_price))
            atm_iv = atm_put.iv

        snapshot_id = f"{self.PROVIDER}_{symbol}_{expiration_ms}_{timestamp_ms}"

        return OptionChainSnapshotEvent(
            symbol=symbol,
            expiration_ms=expiration_ms,
            underlying_price=underlying_price,
            underlying_bid=0.0,
            underlying_ask=0.0,
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

    def build_snapshots_for_symbol(
        self,
        symbol: str,
        min_dte: int = 7,
        max_dte: int = 21,
        underlying_price: float = 0.0,
    ) -> List[OptionChainSnapshotEvent]:
        """Fetch chain and build all snapshots within DTE range for a symbol.

        This is the main entry point for the orchestrator.

        Args:
            symbol: Underlying symbol.
            min_dte: Minimum days to expiration.
            max_dte: Maximum days to expiration.
            underlying_price: Current underlying price.

        Returns:
            List of OptionChainSnapshotEvents (may be empty on failure).
        """
        raw = self.fetch_nested_chain(symbol)
        if not raw:
            logger.warning(
                "Empty response from Tastytrade for %s — skipping snapshot build",
                symbol,
            )
            return []

        # Prefer orchestrator-passed price (e.g. from Alpaca); fill from chain or spot when 0
        from_chain = _underlying_price_from_raw_chain(raw)
        if underlying_price <= 0 and from_chain > 0:
            underlying_price = from_chain
            logger.debug(
                "Using underlying price from Tastytrade chain for %s: %.2f",
                symbol, underlying_price,
            )
        if underlying_price <= 0:
            try:
                spot = self._fetch_underlying_spot(symbol)
                if spot and spot > 0:
                    underlying_price = spot
                    logger.debug(
                        "Using underlying price from Tastytrade market-data for %s: %.2f",
                        symbol, underlying_price,
                    )
            except Exception as exc:
                logger.debug(
                    "Tastytrade spot fetch for %s failed (non-fatal): %s",
                    symbol, exc,
                )

        normalized = normalize_tastytrade_nested_chain(raw)
        if not normalized:
            logger.warning(
                "Normalization returned empty list for %s (provider=%s)",
                symbol, self.PROVIDER,
            )
            return []

        expirations = self.get_expirations_in_range(normalized, min_dte, max_dte)
        if not expirations:
            logger.debug(
                "No expirations in DTE range [%d, %d] for %s (provider=%s)",
                min_dte, max_dte, symbol, self.PROVIDER,
            )
            return []

        snapshots = []
        for exp_date, dte in expirations:
            snapshot = self.build_chain_snapshot(
                symbol, exp_date, normalized, underlying_price
            )
            if snapshot:
                snapshots.append(snapshot)
                logger.debug(
                    "Tastytrade snapshot: %s exp=%s DTE=%d puts=%d calls=%d",
                    symbol, exp_date, dte,
                    len(snapshot.puts), len(snapshot.calls),
                )

        return snapshots

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
            "authenticated": self._authenticated,
            "health": "ok" if self._error_count < max(1, self._request_count) * 0.3 else "degraded",
        }

    def close(self) -> None:
        """Close the connector and release resources."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.debug("Error closing Tastytrade client: %s", exc)
            self._client = None
            self._authenticated = False
