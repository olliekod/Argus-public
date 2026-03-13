"""Public.com API client for options greeks.

Uses Public quickstart auth: exchange secret key for an access token, then use
Bearer access_token for all API requests. See https://public.com/api/docs/quickstart
"""

from __future__ import annotations

import logging
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    import aiohttp
except ImportError:  # pragma: no cover - tested via orchestrator guards
    aiohttp = None

logger = logging.getLogger(__name__)


class PublicAPIError(RuntimeError):
    """Raised on Public API request failures."""


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class PublicAPIConfig:
    api_secret: str  # Secret key from Public settings; exchanged for access token
    account_id: str = ""
    base_url: str = "https://api.public.com"
    timeout_seconds: float = 20.0
    rate_limit_rps: int = 10
    access_token_validity_minutes: int = 60  # Requested validity when exchanging secret for token


class PublicAPIClient:
    """Minimal async client for Public options greeks endpoint."""

    MAX_GREEKS_SYMBOLS = 250
    _TOKEN_PATH = "/userapiauthservice/personal/access-tokens"

    def __init__(self, config: PublicAPIConfig) -> None:
        if not config.api_secret:
            raise PublicAPIError("public.api_secret is required")
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._access_token: Optional[str] = None
        self._resolved_account_id: str = config.account_id or ""
        self._rate_limit_lock = asyncio.Lock()
        self._request_timestamps: deque[float] = deque()
        self._token_lock = asyncio.Lock()

    async def _acquire_rate_limit(self) -> None:
        """Enforce Public REST request budget (default: 10 req/s)."""
        rate_limit = max(int(self._config.rate_limit_rps or 10), 1)
        async with self._rate_limit_lock:
            while True:
                now = time.monotonic()
                while self._request_timestamps and (now - self._request_timestamps[0]) >= 1.0:
                    self._request_timestamps.popleft()

                if len(self._request_timestamps) < rate_limit:
                    self._request_timestamps.append(now)
                    return

                wait_seconds = max(0.0, 1.0 - (now - self._request_timestamps[0]))
                await asyncio.sleep(wait_seconds)

    async def _refresh_access_token(self) -> None:
        """Exchange secret key for access token. Public quickstart: POST .../access-tokens."""
        validity = max(1, min(self._config.access_token_validity_minutes, 1440))
        payload = {"validityInMinutes": validity, "secret": self._config.api_secret}
        url = f"{self._config.base_url.rstrip('/')}{self._TOKEN_PATH}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self._config.timeout_seconds),
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise PublicAPIError(
                        f"Public token exchange failed {resp.status}: {text[:300]}"
                    )
                data = await resp.json()
        token = (data.get("accessToken") or data.get("access_token") or "").strip()
        if not token:
            raise PublicAPIError("Public token response missing accessToken")
        self._access_token = token
        logger.debug("Public access token obtained (validity=%dm)", validity)

    async def _ensure_access_token(self) -> None:
        """Ensure we have a valid access token; exchange secret if needed."""
        async with self._token_lock:
            if self._access_token:
                return
            await self._refresh_access_token()

    async def _get_session(self) -> aiohttp.ClientSession:
        if aiohttp is None:
            raise ImportError("aiohttp required: pip install aiohttp")
        await self._ensure_access_token()
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
            }
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Union[Dict[str, Any], List[tuple]]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        retry_401: bool = True,
    ) -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self._config.base_url.rstrip('/')}{path}"
        await self._acquire_rate_limit()
        async with session.request(
            method, url, params=params, json=json_body
        ) as resp:
            if resp.status == 401 and retry_401:
                logger.warning("Public API 401 for %s; refreshing token and retrying once", path)
                async with self._token_lock:
                    self._access_token = None
                if self._session and not self._session.closed:
                    await self._session.close()
                self._session = None
                await self._ensure_access_token()
                return await self._request(
                    method, path, params=params, json_body=json_body, retry_401=False
                )
            if resp.status >= 400:
                text = await resp.text()
                raise PublicAPIError(f"Public API error {resp.status}: {text[:300]}")
            data = await resp.json(content_type=None)
            return data if isinstance(data, dict) else {"data": data}

    async def get_account_id(self) -> str:
        """Return account_id from config. Public API requires it; no accounts listing endpoint."""
        if self._resolved_account_id:
            return self._resolved_account_id
        account_id = (self._config.account_id or "").strip()
        if not account_id:
            raise PublicAPIError(
                "public.account_id is required. Set it in config (public.account_id) or "
                "secrets (public.account_id). The Public API does not expose an accounts listing endpoint."
            )
        self._resolved_account_id = account_id
        return account_id

    async def get_option_greeks(self, osi_symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch greeks for up to 250 OSI symbols."""
        if len(osi_symbols) > self.MAX_GREEKS_SYMBOLS:
            raise PublicAPIError(f"Public greeks limit exceeded: {len(osi_symbols)} > {self.MAX_GREEKS_SYMBOLS}")
        if not osi_symbols:
            return []
        account_id = await self.get_account_id()
        cleaned = [str(s).strip() for s in osi_symbols if s]
        if not cleaned:
            return []
        # Try comma-separated; some gateways expect osiSymbols=SYM1,SYM2,SYM3
        osi_param = ",".join(cleaned)
        logger.debug(
            "Public greeks request: %d symbols, sample: %s",
            len(cleaned),
            cleaned[:3] if len(cleaned) >= 3 else cleaned,
        )
        payload = await self._request(
            "GET",
            f"/userapigateway/option-details/{account_id}/greeks",
            params={"osiSymbols": osi_param},
        )
        if not isinstance(payload, dict):
            logger.warning("Public greeks response was not a dict: %s", type(payload).__name__)
            return []
        greeks = payload.get("greeks") or payload.get("data") or []
        if isinstance(greeks, dict):
            greeks = [greeks]
        return [g for g in (greeks if isinstance(greeks, list) else []) if isinstance(g, dict)]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
