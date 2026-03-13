"""Tastytrade OAuth helpers for bootstrap flows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urlencode

from aiohttp import ClientSession, ClientTimeout
import requests

TASTYTRADE_AUTH_URL = "https://my.tastytrade.com/auth.html"
TASTYTRADE_TOKEN_URL = "https://api.tastyworks.com/oauth/token"


@dataclass(frozen=True)
class TastytradeOAuthConfig:
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: tuple[str, ...]
    auth_url: str = TASTYTRADE_AUTH_URL
    token_url: str = TASTYTRADE_TOKEN_URL


def parse_scopes(raw_scopes: Optional[Iterable[str] | str]) -> tuple[str, ...]:
    if raw_scopes is None:
        return ("offline_access",)
    if isinstance(raw_scopes, str):
        parts = [s for s in raw_scopes.replace(",", " ").split() if s]
        return tuple(parts) if parts else ("offline_access",)
    scopes = [s for s in raw_scopes if s]
    return tuple(scopes) if scopes else ("offline_access",)


def build_authorize_url(config: TastytradeOAuthConfig, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "scope": " ".join(config.scopes),
        "state": state,
    }
    return f"{config.auth_url}?{urlencode(params)}"


async def exchange_code_for_tokens(
    config: TastytradeOAuthConfig,
    code: str,
    *,
    timeout_s: float = 15.0,
) -> dict:
    payload = {
        "grant_type": "authorization_code",
        "client_id": config.client_id,
        "client_secret": config.client_secret,
        "redirect_uri": config.redirect_uri,
        "code": code,
    }
    timeout = ClientTimeout(total=timeout_s)
    async with ClientSession(timeout=timeout) as session:
        async with session.post(config.token_url, data=payload) as response:
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError(
                    f"Tastytrade token exchange failed ({response.status}): {text[:200]}"
                )
            try:
                return await response.json()
            except Exception as exc:
                raise RuntimeError("Invalid JSON response from token endpoint.") from exc


def prune_states(states: dict[str, float], *, ttl_s: float = 900.0) -> None:
    cutoff = time.time() - ttl_s
    for key, ts in list(states.items()):
        if ts < cutoff:
            states.pop(key, None)


@dataclass(frozen=True)
class OAuthTokenResult:
    access_token: str
    refresh_token: Optional[str]
    expires_in: Optional[int]
    token_type: Optional[str]


class TastytradeOAuthClient:
    """Synchronous OAuth client for refreshing access tokens."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        token_url: str = TASTYTRADE_TOKEN_URL,
        timeout_s: float = 15.0,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._token_url = token_url
        self._timeout_s = timeout_s

    def refresh_access_token(self) -> OAuthTokenResult:
        payload = {
            "grant_type": "refresh_token",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": self._refresh_token,
        }
        response = requests.post(self._token_url, data=payload, timeout=self._timeout_s)
        text = response.text
        if response.status_code >= 400:
            raise RuntimeError(
                f"Tastytrade refresh failed ({response.status_code}): {text[:200]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Invalid JSON response from token endpoint.") from exc

        if isinstance(data.get("data"), dict):
            data = data["data"]

        access_token = data.get("access_token")
        if not access_token:
            raise RuntimeError("No access_token returned from refresh.")
        return OAuthTokenResult(
            access_token=access_token,
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in"),
            token_type=data.get("token_type"),
        )
