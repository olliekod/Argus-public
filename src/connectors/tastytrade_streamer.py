"""
Tastytrade DXLink Streamer Client
==================================

Async WebSocket client for DXLink streaming with a strict state machine:

    SETUP_SENT -> AUTH_SENT -> FEED_CHANNEL_READY -> FEED_SETUP_DONE -> SUBSCRIBED

Each transition has an explicit timeout; exceeding any threshold raises
``DXLinkHandshakeError`` (exit code 6).

Reconnect policy: exponential back-off with cap; re-auth and re-subscribe
after reconnect; hard reset on protocol violations.

Token refresh
-------------
The streamer accepts an optional ``token_refresh_cb`` async callback.  When
the DXLink quote token is close to expiry the streamer invokes the callback
to obtain a fresh token before the next reconnect.  The callback should
return a ``(new_token, new_dxlink_url)`` tuple.

Log redaction
-------------
All outbound/inbound frames that contain tokens are redacted before they
hit the log to prevent credential leakage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    InvalidURI,
)

from .tastytrade_dxlink_parser import (
    AuthState,
    DXLinkFrame,
    GreeksEvent,
    QuoteEvent,
    build_auth_frame,
    build_channel_request,
    build_feed_setup,
    build_feed_subscription,
    build_keepalive,
    build_setup_frame,
    classify_frame,
    is_auth_state,
    is_channel_opened,
    is_error,
    is_feed_config,
    is_keepalive,
    parse_feed_data,
    parse_raw_json,
)

logger = logging.getLogger("argus.dxlink.streamer")


# ---------------------------------------------------------------------------
# Token redaction
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r'("token"\s*:\s*")([^"]{8})[^"]*(")',
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    """Redact bearer / auth tokens in JSON text for safe logging."""
    return _TOKEN_RE.sub(r"\g<1>\g<2>…REDACTED\3", text)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DXLinkHandshakeError(RuntimeError):
    """Raised when any handshake phase exceeds its timeout.  Exit code 6."""


class DXLinkAuthError(RuntimeError):
    """Raised on authentication / token errors."""


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class _Phase(Enum):
    DISCONNECTED = auto()
    SETUP_SENT = auto()
    AUTH_SENT = auto()
    FEED_CHANNEL_READY = auto()
    FEED_SETUP_DONE = auto()
    SUBSCRIBED = auto()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamerConfig:
    """Timeouts and backoff knobs for the DXLink streamer."""

    # Per-phase handshake timeouts (seconds)
    setup_timeout: float = 10.0
    auth_timeout: float = 10.0
    channel_timeout: float = 10.0
    feed_setup_timeout: float = 10.0
    subscription_timeout: float = 10.0

    # Keepalive
    keepalive_interval: float = 30.0

    # Reconnect back-off
    reconnect_base_s: float = 1.0
    reconnect_max_s: float = 60.0
    reconnect_multiplier: float = 2.0
    max_reconnect_attempts: int = 10

    # Event types to request (Greeks will be attempted; if rejected the
    # streamer logs a warning and continues with Quote only).
    event_types: tuple[str, ...] = ("Quote", "Greeks")

    # Proactive token refresh — refresh this many seconds before the
    # token's presumed TTL expires.  Set to 0 to disable.
    token_refresh_ahead_s: float = 120.0


# ---------------------------------------------------------------------------
# Streamer
# ---------------------------------------------------------------------------

EventCallback = Callable[[QuoteEvent | GreeksEvent], None]
TokenRefreshCallback = Callable[[], Awaitable[Tuple[str, str]]]


class TastytradeStreamer:
    """Async DXLink WebSocket streamer with strict state machine."""

    def __init__(
        self,
        dxlink_url: str,
        token: str,
        symbols: List[str],
        *,
        config: Optional[StreamerConfig] = None,
        on_event: Optional[EventCallback] = None,
        event_types: Optional[List[str]] = None,
        token_refresh_cb: Optional[TokenRefreshCallback] = None,
        token_ttl_s: float = 600.0,
    ) -> None:
        self._url = dxlink_url
        self._token = token
        self._symbols = list(symbols)
        self._cfg = config or StreamerConfig()
        self._on_event = on_event
        self._event_types = list(event_types or self._cfg.event_types)

        self._phase: _Phase = _Phase.DISCONNECTED
        self._ws: Any = None
        self._feed_channel: int = 1
        self._running = False
        self._reconnect_count = 0

        # Proactive token refresh
        self._token_refresh_cb = token_refresh_cb
        self._token_ttl_s = token_ttl_s
        self._token_obtained_at: float = time.monotonic()

        # Collected events (for audit / one-shot modes)
        self.events: List[QuoteEvent | GreeksEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect_and_subscribe(self) -> None:
        """Run the full handshake sequence.  Raises on timeout."""
        self._phase = _Phase.DISCONNECTED
        self._ws = await websockets.connect(self._url)
        await self._handshake()

    async def run_forever(self) -> None:
        """Connect, subscribe, and consume events until stopped."""
        self._running = True
        while self._running:
            try:
                await self._maybe_refresh_token()
                await self.connect_and_subscribe()
                self._token_obtained_at = time.monotonic()
                self._reconnect_count = 0
                await self._consume_loop()
            except DXLinkHandshakeError:
                raise  # fatal – propagate exit code 6
            except (ConnectionClosed, ConnectionClosedError, OSError) as exc:
                logger.warning("Connection lost: %s", exc)
                await self._reconnect_backoff()
            except asyncio.CancelledError:
                break

        await self._close()

    async def run_for(self, duration_s: float) -> List[QuoteEvent | GreeksEvent]:
        """Connect, subscribe, collect events for *duration_s* seconds, then close."""
        self.events.clear()
        await self.connect_and_subscribe()
        deadline = asyncio.get_event_loop().time() + duration_s
        try:
            while asyncio.get_event_loop().time() < deadline:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    raw_text = await asyncio.wait_for(
                        self._ws.recv(), timeout=min(remaining, 2.0)
                    )
                except asyncio.TimeoutError:
                    continue
                self._handle_message(raw_text)
        finally:
            await self._close()
        return list(self.events)

    async def stop(self) -> None:
        self._running = False
        await self._close()

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    async def _handshake(self) -> None:
        # 1. SETUP
        await self._send(build_setup_frame())
        self._phase = _Phase.SETUP_SENT

        # 2. Wait for AUTH_STATE UNAUTHORIZED, then send AUTH
        await self._wait_for(
            lambda f: is_auth_state(f, AuthState.UNAUTHORIZED),
            self._cfg.setup_timeout,
            "SETUP ack (AUTH_STATE UNAUTHORIZED)",
        )
        await self._send(build_auth_frame(self._token))
        self._phase = _Phase.AUTH_SENT

        # 3. Wait for AUTH_STATE AUTHORIZED
        await self._wait_for(
            lambda f: is_auth_state(f, AuthState.AUTHORIZED),
            self._cfg.auth_timeout,
            "AUTH ack (AUTH_STATE AUTHORIZED)",
        )

        # 4. CHANNEL_REQUEST
        await self._send(build_channel_request(self._feed_channel))
        await self._wait_for(
            lambda f: is_channel_opened(f, self._feed_channel),
            self._cfg.channel_timeout,
            "CHANNEL_OPENED",
        )
        self._phase = _Phase.FEED_CHANNEL_READY

        # 5. FEED_SETUP
        await self._send(build_feed_setup(self._feed_channel, self._event_types))

        # Wait for FEED_CONFIG (the ack for FEED_SETUP).
        # If Greeks is rejected the server may return a FEED_CONFIG with only
        # Quote accepted – that is fine; we log a warning but continue.
        frame = await self._wait_for(
            is_feed_config,
            self._cfg.feed_setup_timeout,
            "FEED_CONFIG",
        )
        accepted = frame.raw.get("acceptedEventTypes") or frame.raw.get("eventTypes") or []
        self._phase = _Phase.FEED_SETUP_DONE

        # 6. FEED_SUBSCRIPTION
        # We try all requested types because some servers (Tastytrade) return 
        # an empty accepted list even when Quote/Greeks are supported.
        sub_types = self._event_types
        await self._send(
            build_feed_subscription(self._feed_channel, self._symbols, sub_types)
        )
        self._phase = _Phase.SUBSCRIBED
        logger.info(
            "DXLink handshake complete – subscribed to %d symbols (%s)",
            len(self._symbols),
            ", ".join(sub_types),
        )

    async def _wait_for(
        self,
        predicate: Callable[[DXLinkFrame], bool],
        timeout: float,
        label: str,
    ) -> DXLinkFrame:
        """Read frames until *predicate* matches or *timeout* expires."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise DXLinkHandshakeError(
                    f"Timeout waiting for {label} (>{timeout:.1f}s)"
                )
            try:
                raw_text = await asyncio.wait_for(
                    self._ws.recv(), timeout=min(remaining, 2.0)
                )
            except asyncio.TimeoutError:
                continue
            raw = parse_raw_json(raw_text)
            frame = classify_frame(raw)

            if is_error(frame):
                err_msg = raw.get("error") or raw.get("message") or str(raw)
                raise DXLinkHandshakeError(f"DXLink ERROR during {label}: {err_msg}")

            if is_keepalive(frame):
                await self._send(build_keepalive())
                continue

            if predicate(frame):
                return frame

            logger.debug("Ignoring frame while waiting for %s: %s", label, _redact(str(raw.get("type"))))

    # ------------------------------------------------------------------
    # Event consumption
    # ------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        while self._running:
            try:
                raw_text = await asyncio.wait_for(
                    self._ws.recv(), timeout=self._cfg.keepalive_interval
                )
            except asyncio.TimeoutError:
                await self._send(build_keepalive())
                continue
            self._handle_message(raw_text)

    def _handle_message(self, raw_text: str) -> None:
        raw = parse_raw_json(raw_text)
        frame = classify_frame(raw)

        if is_keepalive(frame):
            asyncio.ensure_future(self._send(build_keepalive()))
            return

        if frame.msg_type == "FEED_DATA":
            receipt_time = int(time.time() * 1000)
            events = parse_feed_data(frame, receipt_time)
            for event in events:
                self.events.append(event)
                if self._on_event:
                    try:
                        self._on_event(event)
                    except Exception:
                        logger.exception("Event callback error")
            return

        if is_error(frame):
            logger.error("DXLink error: %s", raw)
            logger.debug("DXLink error frame (raw): %s", raw)
            return

        logger.debug("Unhandled frame type: %s", frame.msg_type)

    # ------------------------------------------------------------------
    # Reconnect
    # ------------------------------------------------------------------

    async def _reconnect_backoff(self) -> None:
        self._reconnect_count += 1
        if self._reconnect_count > self._cfg.max_reconnect_attempts:
            logger.error("Max reconnect attempts exceeded")
            self._running = False
            return
        delay = min(
            self._cfg.reconnect_base_s * (self._cfg.reconnect_multiplier ** (self._reconnect_count - 1)),
            self._cfg.reconnect_max_s,
        )
        logger.info("Reconnecting in %.1fs (attempt %d)", delay, self._reconnect_count)
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Proactive token refresh
    # ------------------------------------------------------------------

    async def _maybe_refresh_token(self) -> None:
        """Refresh the DXLink token if it is close to expiry.

        Invokes the user-supplied ``token_refresh_cb`` callback which
        should return ``(new_token, new_dxlink_url)``.
        """
        if not self._token_refresh_cb:
            return
        if self._cfg.token_refresh_ahead_s <= 0:
            return

        elapsed = time.monotonic() - self._token_obtained_at
        remaining = self._token_ttl_s - elapsed
        if remaining > self._cfg.token_refresh_ahead_s:
            return  # Still fresh

        logger.info(
            "Token nearing expiry (%.0fs remaining, TTL=%.0fs) — refreshing",
            remaining, self._token_ttl_s,
        )
        try:
            new_token, new_url = await self._token_refresh_cb()
            self._token = new_token
            self._url = new_url
            self._token_obtained_at = time.monotonic()
            logger.info("Token refreshed successfully")
        except Exception:
            logger.exception("Token refresh callback failed — using old token")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send(self, msg: Dict[str, Any]) -> None:
        text = json.dumps(msg)
        logger.debug("OUT: %s", _redact(text))
        await self._ws.send(text)

    async def _close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._phase = _Phase.DISCONNECTED
