"""
Async REST client for the Kalshi Trade API v2.

Features
--------
* **Token-bucket rate limiting** — separate buckets for reads (GET) and
  writes (POST / DELETE).
* **Automatic retries** with exponential back-off + jitter for transient
  HTTP errors (429, 500, 502, 503, 504) and connection errors.
* **Idempotency** — POST requests carry an ``Idempotency-Key`` header so
  retried order submissions are safe.
* **Cursor pagination** helper for list endpoints (``/markets``, etc.).
* **Auth header injection** via ``kalshi_auth.build_headers``.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp

from .config import KalshiConfig
from .kalshi_auth import build_headers, calibrate_clock_offset, load_private_key
from .logging_utils import ComponentLogger, LatencyTracker

log = ComponentLogger("rest")

# HTTP status codes that trigger an automatic retry.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 4
_BASE_BACKOFF_S = 0.5


# ---------------------------------------------------------------------------
#  Async token-bucket rate limiter
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Simple async token-bucket that refills at a fixed rate.

    One token = one request.  ``acquire()`` awaits until a token is available.
    """

    def __init__(self, rate_per_sec: float) -> None:
        self._rate = rate_per_sec
        self._max_tokens = rate_per_sec  # burst size = 1 second of capacity
        self._tokens = rate_per_sec
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + elapsed * self._rate,
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Sleep until at least one token is available.
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
#  REST client
# ---------------------------------------------------------------------------

class KalshiRestClient:
    """Authenticated, rate-limited, retrying async client for the Kalshi API."""

    def __init__(self, config: KalshiConfig) -> None:
        self._cfg = config
        self._base = config.base_url_rest.rstrip("/")
        self._key_id = config.kalshi_key_id
        self._pk = load_private_key(config.kalshi_private_key_path)
        self._offset_ms: int = 0

        self._read_bucket = _TokenBucket(config.rate_limit_read_per_sec)
        self._write_bucket = _TokenBucket(config.rate_limit_write_per_sec)

        self._session: Optional[aiohttp.ClientSession] = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        # Default timeout 30s; market discovery can use longer via _request timeout param
        self._default_timeout = aiohttp.ClientTimeout(total=30, connect=15)
        self._session = aiohttp.ClientSession(timeout=self._default_timeout)
        if self._cfg.enable_clock_offset_calibration:
            self._offset_ms = await calibrate_clock_offset(
                self._session,
                self._base,
                max_offset_ms=self._cfg.max_clock_offset_ms,
            )

    async def measure_rtt_ms(self, samples: int = 5) -> float:
        """Ping the Kalshi REST endpoint and return median round-trip time in ms.

        Uses the unauthenticated /exchange/status endpoint so no API key is
        needed.  Takes *samples* measurements and returns the median to avoid
        outliers from cold-start TCP setup or transient congestion.
        """
        url = f"{self._base}/exchange/status"
        times: list[float] = []
        for _ in range(samples):
            t0 = time.monotonic()
            try:
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    await r.read()
                times.append((time.monotonic() - t0) * 1000)
            except Exception:
                pass
        if not times:
            return 200.0  # conservative fallback
        times.sort()
        return times[len(times) // 2]  # median

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # -- internal request wrapper -------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        idempotent: bool = False,
        timeout: Optional[aiohttp.ClientTimeout] = None,
        ignore_404: bool = False,
    ) -> Dict[str, Any]:
        """Execute a single API request with auth, rate limiting, and retries.

        Pass timeout=ClientTimeout(total=60) for discovery/pagination requests
        that may be slow under load. If ignore_404 is True, a 404 response
        is treated as success (return {}) and not logged as error — use for
        idempotent operations like cancel order where "not found" is acceptable.
        """
        assert self._session is not None, "call start() first"

        # Pick the right bucket.
        bucket = self._write_bucket if method in ("POST", "DELETE") else self._read_bucket
        url = f"{self._base}{path}"

        idempotency_key = str(uuid.uuid4()) if (method == "POST" or idempotent) else None

        last_exc: Optional[BaseException] = None

        for attempt in range(_MAX_RETRIES + 1):
            await bucket.acquire()

            # Kalshi signs the full path including /trade-api/v2 (see docs).
            sign_path = urlparse(url).path
            headers = build_headers(
                self._key_id, self._pk, method, sign_path, self._offset_ms,
            )
            headers["Accept"] = "application/json"
            headers["Content-Type"] = "application/json"
            if idempotency_key:
                headers["Idempotency-Key"] = idempotency_key

            try:
                req_timeout = timeout or self._default_timeout
                with LatencyTracker(f"{method} {path}", log):
                    async with self._session.request(
                        method,
                        url,
                        headers=headers,
                        params=params,
                        json=json_body,
                        timeout=req_timeout,
                    ) as resp:
                        body = await resp.json(content_type=None)

                        if resp.status < 300:
                            return body  # type: ignore[return-value]

                        if resp.status in _RETRYABLE_STATUSES:
                            log.warning(
                                f"Retryable HTTP {resp.status} on {method} {path}",
                                data={"attempt": attempt, "body": str(body)[:200]},
                            )
                            last_exc = aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=str(body),
                            )
                        elif resp.status == 404 and ignore_404:
                            # Idempotent success: resource already gone (e.g. order cancelled/filled).
                            log.debug(
                                "HTTP 404 (ignored)",
                                data={"method": method, "path": path},
                            )
                            return {}
                        else:
                            # Non-retryable error.
                            log.error(
                                f"HTTP {resp.status} on {method} {path}",
                                data={"body": str(body)[:500]},
                            )
                            raise aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=str(body),
                            )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                # Include type and repr when str(exc) is empty (e.g. some TimeoutError)
                err_msg = str(exc).strip()
                if not err_msg:
                    err_msg = f"{type(exc).__name__}({repr(exc)})"
                log.warning(
                    "Connection error on %s %s: %s",
                    method, path, err_msg,
                    data={"attempt": attempt, "exc_type": type(exc).__name__},
                )
                last_exc = exc

            # Exponential back-off with jitter.
            backoff = _BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 0.3)
            await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Exhausted {_MAX_RETRIES + 1} attempts for {method} {path}"
        ) from last_exc  # type: ignore[arg-type]

    # -- public API wrappers ------------------------------------------------

    async def get_markets(
        self,
        *,
        cursor: Optional[str] = None,
        limit: int = 100,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        return await self._request("GET", "/markets", params=params, timeout=timeout)

    async def get_market(self, ticker: str) -> Dict[str, Any]:
        return await self._request("GET", f"/markets/{ticker}")

    async def get_market_orderbook(self, ticker: str, depth: int = 10) -> Dict[str, Any]:
        return await self._request(
            "GET", f"/markets/{ticker}/orderbook", params={"depth": depth}
        )

    async def get_orders(
        self, *, cursor: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/portfolio/orders", params=params)

    async def get_fills(
        self, *, cursor: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/portfolio/fills", params=params)

    async def create_order(
        self,
        *,
        ticker: str,
        action: str,      # "buy" or "sell"
        side: str,         # "yes" or "no"
        order_type: str,   # "limit" or "market"
        count: int,        # number of contracts (whole)
        yes_price: Optional[int] = None,   # cents 1-99
        no_price: Optional[int] = None,    # cents 1-99
        client_order_id: Optional[str] = None,
        expiration_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "type": order_type,
            "count": count,
            # Send count_fp alongside count for forward-compatibility
            # with Kalshi's *_fp migration (March 5 2026 cutoff).
            "count_fp": f"{count}.00",
        }
        # Send both legacy cents and _dollars for March 2026 subpenny migration.
        # Legacy yes_price/no_price deprecated March 5, 2026; _dollars preferred.
        if yes_price is not None:
            body["yes_price"] = yes_price
            body["yes_price_dollars"] = f"{yes_price / 100:.4f}"
        if no_price is not None:
            body["no_price"] = no_price
            body["no_price_dollars"] = f"{no_price / 100:.4f}"
        if client_order_id:
            body["client_order_id"] = client_order_id
        if expiration_ts is not None:
            body["expiration_ts"] = expiration_ts
        return await self._request("POST", "/portfolio/orders", json_body=body)

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        # 404 = order already filled, expired, or cancelled — treat as success.
        return await self._request(
            "DELETE", f"/portfolio/orders/{order_id}", ignore_404=True
        )

    async def get_balance(self) -> int:
        """Return account balance in cents (divide by 100 for dollars).

        Prefers balance_dollars (fixed-point string) when present (March 2026
        subpenny migration); falls back to legacy balance (integer cents).
        """
        resp = await self._request("GET", "/portfolio/balance")
        # Prefer _dollars for March 2026 migration.
        if "balance_dollars" in resp:
            return round(float(resp["balance_dollars"].strip()) * 100)
        inner = resp.get("balance", 0)
        if isinstance(inner, dict):
            if "balance_dollars" in inner:
                return round(float(inner["balance_dollars"].strip()) * 100)
            inner = inner.get("balance", 0)
        return int(inner)

    # -- pagination helper --------------------------------------------------

    async def paginate(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        items_key: str = "markets",
        limit: int = 100,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield all items across paginated responses.

        Uses cursor-based pagination.  *items_key* is the JSON key under
        which the result list lives (``"markets"``, ``"orders"``, …).
        Pass timeout for discovery (e.g. 60s) when API may be slow.
        """
        cursor: Optional[str] = None
        base_params = dict(params or {})
        base_params["limit"] = limit

        while True:
            req_params = dict(base_params)
            if cursor:
                req_params["cursor"] = cursor

            resp = await self._request(method, path, params=req_params, timeout=timeout)
            items = resp.get(items_key, [])
            for item in items:
                yield item

            cursor = resp.get("cursor")
            if not cursor or not items:
                break
