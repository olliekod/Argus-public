# Created by Oliver Meihls

# Async WebSocket client for the Kalshi streaming API.
#
# Responsibilities
# 1. **Authenticated handshake** — auth headers are included during the
# initial HTTP upgrade, using the same RSA-PSS signing scheme as REST.
# 2. **Heartbeat** — respond to server ``Ping`` frames (payload ``heartbeat``)
# with ``Pong``.  aiohttp handles this at the transport level, but we
# also detect application-layer heartbeat messages.
# 3. **Subscribe / update_subscription** — manage channel subscriptions for
# ``ticker``, ``orderbook_delta``, ``market_lifecycle_v2`` (public) and
# ``fill``, ``user_orders``, ``market_positions`` (private).
# 4. **Message dispatch** — decode JSON frames, wrap them in typed dataclass
# messages, and publish on the internal bus.
# 5. **Reconnect** — on disconnect, reconnect with jittered exponential
# back-off.  Resubscriptions are replayed automatically.
# 6. **Cancellation** — the main loop honours ``asyncio.CancelledError`` so
# that the caller's ``task.cancel()`` results in a clean shutdown.

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

import aiohttp

from .bus import Bus
from .config import KalshiConfig
from .kalshi_auth import build_headers, load_private_key
from .kalshi_subpenny import parse_count_centicx, parse_count_whole, parse_price_cents
from .logging_utils import ComponentLogger
from .models import (
    FillEvent,
    KalshiOrderDeltaEvent,
    KalshiRtt,
    KalshiTradeEvent,
    OrderbookState,
    OrderUpdate,
    RiskEvent,
    TickerUpdate,
    WsConnectionEvent,
)
from .orderbook import OrderBook

log = ComponentLogger("ws")

# Reconnection parameters.
_BASE_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 60.0
_JITTER_MAX_S = 2.0

# Avoid frequent full unsubscribe/resubscribe churn; remove only occasionally.
_REMOVE_RESUBSCRIBE_COOLDOWN_S = 120.0

# Heartbeat timeout: if no Ping arrives in this many seconds, reconnect.
_HEARTBEAT_TIMEOUT_S = 30.0

# Track consecutive auth failures to detect persistent clock drift.
_MAX_AUTH_FAILURES = 3


class KalshiWebSocket:
    # Manages a single authenticated WebSocket connection to Kalshi.
    #
    # WebSocket authentication
    # Auth headers are computed at handshake time using the same RSA-PSS
    # signing scheme as REST.  The signed path is configurable via
    # ``ws_signing_path`` in config — set this to match whatever Kalshi's
    # validator expects for the WS upgrade (may differ from the URL path).
    #
    # If ``ws_signing_path`` is empty, we derive it from the URL path of
    # ``base_url_ws``.

    def __init__(
        self,
        config: KalshiConfig,
        bus: Bus,
        orderbooks: Dict[str, OrderBook],
    ) -> None:
        self._cfg = config
        self._bus = bus
        self._orderbooks = orderbooks

        self._key_id = config.kalshi_key_id
        self._pk = load_private_key(config.kalshi_private_key_path)
        self._offset_ms: int = 0

        # Resolve the path used for signing the WS handshake.
        if config.ws_signing_path:
            self._ws_sign_path = config.ws_signing_path
        else:
            from urllib.parse import urlparse
            self._ws_sign_path = urlparse(config.base_url_ws).path

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._msg_id: int = 0
        self._consecutive_auth_failures: int = 0

        # Track desired subscriptions so we can replay after reconnect.
        self._desired_channels: List[str] = []
        self._desired_tickers: Set[str] = set()
        self._subscription_ids: List[int] = []

        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

        # Resubscription guard: True while a subscribe command is in-flight.
        # Prevents sending hundreds of redundant subscribe commands when
        # deltas keep arriving for already-invalid books (seq-gap flood).
        self._resubscribe_pending: bool = False
        # Monotonic time when the pending resubscribe was sent.  Used to
        # detect stalled subscriptions (no confirmation within timeout) and
        # trigger a hard WS reconnect.
        self._resubscribe_sent_mono: float = 0.0
        # How long to wait for a "subscribed" confirmation before forcing
        # a full reconnect. 60s when subscribing to 50+ tickers (Kalshi can be slow).
        self._subscribe_confirm_timeout_s: float = 60.0
        # Set by seq-gap recovery to force reconnect instead of in-place resubscribe.
        self._force_reconnect: bool = False
        # Monotonic time when we last sent a subscribe (for WS RTT measurement).
        self._last_subscribe_sent_mono: float = 0.0

        # Health metrics (Phase 3).
        self._total_messages: int = 0
        self._seq_gaps: int = 0
        self._reconnects: int = 0
        self._last_message_mono: float = 0.0
        self._connected_since: Optional[float] = None

        # Seq-gap recovery: resubscribe to get fresh snapshots so UI/strategy see valid OB again.
        self._seq_gap_times: deque = deque(maxlen=200)
        self._last_resubscribe_mono: float = 0.0
        # Relaxed from 2→5: 2 gaps in 60s was too aggressive; Kalshi can send brief bursts
        # of out-of-order deltas on subscribe. Avoid reconnect loops that prevent orderbooks
        # from ever populating (OB 0, Ask/Edge ---).
        self._SEQ_GAP_DEBOUNCE_COUNT = 5
        self._SEQ_GAP_DEBOUNCE_WINDOW_S = 90.0
        self._SEQ_GAP_MIN_WINDOW_S = 50.0  # require gaps over ≥50s so burst (e.g. 200 in 42s) doesn't reconnect under load
        self._SEQ_GAP_RESUBSCRIBE_COOLDOWN_S = 45.0  # allow resubscribe again after 45s if needed
        # Publish invalid at most once per ticker until snapshot; avoids flooding UI with invalid.
        self._invalid_ob_published: Set[str] = set()
        # Throttle expensive full unsubscribe/resubscribe operations caused by
        # rapid near-money set churn.
        self._last_remove_resubscribe_mono: float = 0.0

    def get_health(self) -> Dict[str, Any]:
        # Return structured WS health metrics for monitoring.
        now = time.monotonic()
        msg_age = (now - self._last_message_mono) if self._last_message_mono > 0 else None
        return {
            "connected": self._ws is not None and not self._ws.closed if self._ws else False,
            "total_messages": self._total_messages,
            "seq_gaps": self._seq_gaps,
            "reconnects": self._reconnects,
            "last_message_age_s": round(msg_age, 2) if msg_age is not None else None,
            "uptime_s": round(now - self._connected_since, 1) if self._connected_since else None,
            "auth_failures": self._consecutive_auth_failures,
            "desired_tickers": len(self._desired_tickers),
            "active_sids": len(self._subscription_ids),
            "resubscribe_pending": self._resubscribe_pending,
        }

    # -- lifecycle -----------------------------------------------------------

    async def start(self, offset_ms: int = 0) -> None:
        # Start the WebSocket event loop as a background task.
        self._offset_ms = offset_ms
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        # Gracefully shut down.
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self._close()

    async def _close(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()
            self._session = None

    # -- subscribe -----------------------------------------------------------

    async def subscribe(
        self,
        channels: List[str],
        market_tickers: List[str],
    ) -> None:
        # Register desired subscriptions and send if connected.
        self._desired_channels = channels
        self._desired_tickers = set(market_tickers)
        if self._ws and not self._ws.closed:
            await self._send_subscribe(channels, market_tickers)

    async def update_subscription(
        self,
        *,
        add_tickers: Optional[List[str]] = None,
        remove_tickers: Optional[List[str]] = None,
    ) -> None:
        # Incrementally add/remove tickers from existing subscriptions.
        add_set = set(add_tickers or [])
        remove_set = set(remove_tickers or [])

        if add_set:
            self._desired_tickers.update(add_set)
        if remove_set:
            self._desired_tickers -= remove_set

        if not self._ws or self._ws.closed:
            return

        if not self._subscription_ids:
            # No known server-side subscriptions (fresh connect/reconnect path);
            # rely on full subscribe replay for consistency.
            if self._desired_channels and self._desired_tickers:
                await self._send_subscribe(self._desired_channels, list(self._desired_tickers))
            return

        if add_set:
            # Kalshi API requires exactly one subscription ID per update_subscription request (code 12).
            tickers_sorted = sorted(add_set)
            for sid in self._subscription_ids:
                msg_id = self._next_id()
                msg = {
                    "id": msg_id,
                    "cmd": "update_subscription",
                    "params": {
                        "sids": [sid],
                        "market_tickers": tickers_sorted,
                        "action": "add_markets",
                    },
                }
                await self._ws.send_json(msg)
            log.info(
                "Sent update_subscription add_markets",
                data={"sids": list(self._subscription_ids), "tickers": tickers_sorted},
            )

        if remove_set:
            # Kalshi does not support remove_markets incrementally, but tearing down
            # all subscriptions on every small churn causes repeated snapshots,
            # sequence resets, and prolonged invalid orderbooks. So we only do
            # a full unsubscribe/resubscribe occasionally; otherwise we keep a
            # superset subscribed and rely on strategy/probability filtering.
            now = time.monotonic()
            allow_full_refresh = (
                (now - self._last_remove_resubscribe_mono) >= _REMOVE_RESUBSCRIBE_COOLDOWN_S
            )
            if not allow_full_refresh:
                log.debug(
                    "Skipping remove_markets refresh to avoid WS churn",
                    data={
                        "remove_count": len(remove_set),
                        "cooldown_s": _REMOVE_RESUBSCRIBE_COOLDOWN_S,
                    },
                )
                return

            msg = {
                "id": self._next_id(),
                "cmd": "unsubscribe",
                "params": {"sids": list(self._subscription_ids)},
            }
            await self._ws.send_json(msg)
            self._last_remove_resubscribe_mono = now
            self._subscription_ids.clear()
            if self._desired_channels and self._desired_tickers:
                await self._send_subscribe(self._desired_channels, list(self._desired_tickers))

    # -- internal: connection loop -------------------------------------------

    async def _run_loop(self) -> None:
        # Outer loop: connect, read messages, reconnect on failure.
        backoff = _BASE_BACKOFF_S

        while self._running:
            try:
                await self._connect()
                if self._reconnects > 0:
                    log.info("WebSocket reconnected — resuming trading")
                else:
                    log.info("WebSocket connected")
                await self._bus.publish(
                    "kalshi.ws.status",
                    WsConnectionEvent(status="connected", timestamp=time.time()),
                )
                self._connected_since = time.monotonic()
                backoff = _BASE_BACKOFF_S  # reset on success

                # Replay subscriptions after connect.
                if self._desired_channels and self._desired_tickers:
                    await self._send_subscribe(
                        self._desired_channels, list(self._desired_tickers)
                    )

                await self._read_loop()

                # If we left read_loop due to seq-gap force reconnect, notify before reconnecting.
                if self._force_reconnect:
                    ts = time.time()
                    log.warning(
                        "Seq-gap recovery — reconnecting for fresh orderbook snapshots"
                    )
                    await self._bus.publish(
                        "kalshi.ws.status",
                        WsConnectionEvent(
                            status="disconnected",
                            detail="seq_gap_recovery",
                            timestamp=ts,
                        ),
                    )
                    await self._bus.publish(
                        "kalshi.risk",
                        RiskEvent(
                            event_type="disconnect_halt",
                            detail="WebSocket disconnected (seq-gap recovery)",
                            timestamp=ts,
                        ),
                    )

            except asyncio.CancelledError:
                log.info("WebSocket task cancelled — shutting down")
                break
            except Exception as exc:
                log.error(f"WebSocket error: {exc}")
                log.warning("WebSocket disconnected — halting trading until reconnected")
                ts = time.time()

                # Invalidate ALL orderbooks on disconnect — they cannot be
                # trusted until a fresh snapshot arrives after reconnect.
                for book in self._orderbooks.values():
                    book.invalidate()
                self._invalid_ob_published.clear()
                await self._bus.publish(
                    "kalshi.risk",
                    RiskEvent(
                        event_type="orderbook_invalid",
                        detail="All orderbooks invalidated on WS disconnect",
                        timestamp=ts,
                    ),
                )
                await self._bus.publish(
                    "kalshi.risk",
                    RiskEvent(
                        event_type="disconnect_halt",
                        detail="WebSocket disconnected",
                        timestamp=ts,
                    ),
                )
                await self._bus.publish(
                    "kalshi.ws.status",
                    WsConnectionEvent(
                        status="disconnected",
                        detail=str(exc),
                        timestamp=ts,
                    ),
                )

            # Reconnect back-off with jitter.
            if self._running:
                self._reconnects += 1
                self._connected_since = None
                jitter = random.uniform(0, _JITTER_MAX_S)
                wait = min(backoff + jitter, _MAX_BACKOFF_S)
                log.info(f"Reconnecting in {wait:.1f}s")
                await self._bus.publish(
                    "kalshi.ws.status",
                    WsConnectionEvent(
                        status="reconnecting",
                        detail=f"wait={wait:.1f}s",
                        timestamp=time.time(),
                    ),
                )
                try:
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    break
                backoff = min(backoff * 2, _MAX_BACKOFF_S)

        await self._close()

    async def _connect(self) -> None:
        # Open an authenticated WebSocket connection.
        #
        # The signed path for the handshake is taken from
        # ``config.ws_signing_path`` (or derived from the WS URL if empty).
        # Method is always GET.  The same three KALSHI-ACCESS-* headers
        # used for REST are attached to the upgrade request.
        await self._close()

        self._session = aiohttp.ClientSession()
        url = self._cfg.base_url_ws

        # Auth headers for the WS handshake — uses the configurable
        # signing path, NOT hard-coded.
        headers = build_headers(
            self._key_id, self._pk, "GET", self._ws_sign_path, self._offset_ms
        )

        try:
            self._ws = await self._session.ws_connect(
                url,
                headers=headers,
                heartbeat=10.0,       # aiohttp will auto-pong transport pings
                autoping=True,
                timeout=15.0,
            )
        except aiohttp.WSServerHandshakeError as exc:
            # 401/403 during handshake → likely bad signature / wrong path.
            self._consecutive_auth_failures += 1
            log.error(
                f"WS handshake failed (attempt {self._consecutive_auth_failures}): {exc}",
                data={"sign_path": self._ws_sign_path, "status": getattr(exc, "status", None)},
            )
            if self._consecutive_auth_failures >= _MAX_AUTH_FAILURES:
                await self._bus.publish(
                    "kalshi.risk",
                    RiskEvent(
                        event_type="auth_failure",
                        detail=f"{self._consecutive_auth_failures} consecutive WS auth failures; "
                               f"likely clock drift or wrong ws_signing_path",
                        timestamp=time.time(),
                    ),
                )
            raise

        self._consecutive_auth_failures = 0
        self._msg_id = 0
        self._subscription_ids = []
        self._resubscribe_pending = False
        self._resubscribe_sent_mono = 0.0
        self._force_reconnect = False
        log.info(
            "WebSocket connected",
            data={"url": url, "sign_path": self._ws_sign_path},
        )

    async def _read_loop(self) -> None:
        # Read frames until disconnect or cancellation.
        assert self._ws is not None
        last_heartbeat = time.monotonic()

        async for raw_msg in self._ws:
            if raw_msg.type == aiohttp.WSMsgType.TEXT:
                last_heartbeat = time.monotonic()
                self._last_message_mono = last_heartbeat
                self._total_messages += 1
                try:
                    data = json.loads(raw_msg.data)
                except json.JSONDecodeError:
                    log.warning(f"Non-JSON WS frame: {raw_msg.data[:200]}")
                    continue
                await self._dispatch(data)
                if self._force_reconnect:
                    break

            elif raw_msg.type == aiohttp.WSMsgType.PING:
                last_heartbeat = time.monotonic()
                # aiohttp auto-pongs, but we log it.

            elif raw_msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                log.info("WebSocket closed by server")
                break

            elif raw_msg.type == aiohttp.WSMsgType.ERROR:
                log.error(f"WebSocket error frame: {self._ws.exception()}")
                break

            # Check heartbeat timeout.
            if time.monotonic() - last_heartbeat > _HEARTBEAT_TIMEOUT_S:
                log.warning("Heartbeat timeout — forcing reconnect")
                break

            # Check subscription-confirmation timeout: if a resubscribe was
            # sent but never confirmed within the deadline, the server is not
            # going to send a snapshot on this connection.  Force a full
            # reconnect so the new connection triggers a fresh snapshot.
            if (
                self._resubscribe_pending
                and self._resubscribe_sent_mono > 0
                and (time.monotonic() - self._resubscribe_sent_mono)
                    > self._subscribe_confirm_timeout_s
            ):
                log.warning(
                    f"Subscribe confirmation timeout after "
                    f"{self._subscribe_confirm_timeout_s:.0f}s — forcing reconnect "
                    f"to obtain fresh orderbook snapshots"
                )
                break

    # -- internal: subscribe -------------------------------------------------

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _send_subscribe(
        self, channels: List[str], market_tickers: List[str]
    ) -> None:
        assert self._ws is not None
        msg_id = self._next_id()
        msg = {
            "id": msg_id,
            "cmd": "subscribe",
            "params": {
                "channels": channels,
                "market_tickers": market_tickers,
            },
        }
        await self._ws.send_json(msg)
        self._last_subscribe_sent_mono = time.monotonic()
        log.info(
            "Sent subscribe",
            data={"id": msg_id, "channels": channels, "tickers": market_tickers},
        )

    # -- internal: message dispatch ------------------------------------------

    async def _dispatch(self, data: Dict[str, Any]) -> None:
        # Route an incoming JSON message to the appropriate bus topic.
        msg_type = data.get("type", "")
        sid = data.get("sid")

        # Subscription confirmations. Kalshi may put sid in msg (e.g. msg.sid).
        if msg_type == "subscribed":
            sid = sid or (data.get("msg") or {}).get("sid")
            log.info("Subscription confirmed", data={"sid": sid})
            if sid is not None and sid not in self._subscription_ids:
                self._subscription_ids.append(sid)
            # Clear the resubscription guard — subscription is live again.
            self._resubscribe_pending = False
            # Publish WebSocket RTT (subscribe → subscribed round-trip).
            if self._last_subscribe_sent_mono > 0:
                rtt_ms = (time.monotonic() - self._last_subscribe_sent_mono) * 1000
                await self._bus.publish(
                    "kalshi.rtt",
                    KalshiRtt(rtt_ms=rtt_ms, timestamp=time.time(), source="ws"),
                )
                log.info("Kalshi RTT: %.0fms (ws)", rtt_ms, data={"rtt_ms": round(rtt_ms, 1), "source": "ws"})
                self._last_subscribe_sent_mono = 0
            return

        if msg_type == "error":
            log.error("WS error message", data=data)
            err = (data.get("msg") or {}) if isinstance(data.get("msg"), dict) else {}
            if str(err.get("msg", "")).lower().find("unknown subscription id") >= 0:
                # Server subscription state diverged from our local sid list.
                # Force clean reconnect so we stop sending invalid updates.
                self._force_reconnect = True
            return

        channel = data.get("msg", {}).get("channel") if "msg" in data else None
        if not channel:
            # Some messages have channel at root.
            channel = data.get("channel")

        payload = data.get("msg", data)

        # Kalshi sends orderbook with type at root ("orderbook_snapshot" / "orderbook_delta")
        # but may not include "channel" in the message, so route by msg_type too.
        if msg_type in ("orderbook_snapshot", "orderbook_delta") or channel == "orderbook_delta":
            seq = data.get("seq", payload.get("seq", 0))
            await self._handle_orderbook(payload, msg_type, seq)
        elif channel == "ticker":
            await self._handle_ticker(payload)
        elif channel == "fill":
            await self._handle_fill(payload)
        elif channel == "user_orders":
            await self._handle_user_orders(payload)
        elif channel == "trade" or msg_type == "trade":
            await self._handle_trade(payload)
        elif channel == "market_lifecycle_v2":
            await self._bus.publish("kalshi.market_lifecycle", payload)
        elif channel == "market_positions":
            await self._bus.publish("kalshi.market_positions", payload)
        else:
            log.debug(f"Unhandled channel: {channel}", data={"keys": list(data.keys())})

    async def _handle_orderbook(
        self, payload: Dict[str, Any], msg_type: str, seq: int
    ) -> None:
        # Process orderbook snapshot or delta. *seq* is from root or payload.
        ticker = payload.get("market_ticker", "")

        if ticker not in self._orderbooks:
            self._orderbooks[ticker] = OrderBook(market_ticker=ticker)

        book = self._orderbooks[ticker]

        if "snapshot" in msg_type:
            # Debug: log first level structure to diagnose subpenny format changes
            yes_levels = payload.get("yes", [])
            no_levels = payload.get("no", [])
            first_yes = yes_levels[0] if yes_levels else None
            first_no = no_levels[0] if no_levels else None
            log.debug(
                "OB snapshot received",
                data={
                    "ticker": ticker,
                    "seq": seq,
                    "yes_count": len(yes_levels),
                    "no_count": len(no_levels),
                    "first_yes_type": type(first_yes).__name__ if first_yes else None,
                    "first_yes_sample": str(first_yes)[:80] if first_yes else None,
                },
            )
            book.apply_snapshot(payload, seq)
            self._invalid_ob_published.discard(ticker)  # clear so next gap can publish invalid again
            log.debug(f"OB snapshot for {ticker}", data={"seq": seq})
        elif "delta" in msg_type:
            ok = book.apply_delta(payload, seq)
            if not ok:
                if not book.has_snapshot:
                    # Delta arrived before the initial snapshot — this is normal
                    # during the brief window between subscribe and snapshot delivery.
                    # Silently drop: the snapshot is in transit and will arrive shortly.
                    # Do NOT resubscribe here — that would create duplicate subscriptions
                    # which cause duplicate messages → more seq gaps → cascade failure.
                    log.debug(f"Delta for {ticker} before first snapshot — waiting for snapshot")
                    return

                self._seq_gaps += 1

                # Do not publish a global RiskEvent — that halts all execution.
                # Invalidate only this ticker's book so strategy skips it until
                # a fresh snapshot arrives. Other tickers keep trading.
                if self._resubscribe_pending:
                    return

                now_mono = time.monotonic()
                self._seq_gap_times.append(now_mono)

                log.debug(
                    f"Seq gap on {ticker}: expected {book.last_seq + 1}, got {seq}. "
                    f"Book invalid for this ticker only.",
                    data={"last_seq": book.last_seq, "got_seq": seq},
                )
                # Publish invalid at most once per ticker until next snapshot — avoids flooding
                # the bus so the UI and strategy see valid state again once snapshots arrive.
                if ticker not in self._invalid_ob_published:
                    self._invalid_ob_published.add(ticker)
                    _invalid_ob = OrderbookState(
                        market_ticker=ticker,
                        best_yes_bid_cents=0,
                        best_no_bid_cents=0,
                        implied_yes_ask_cents=100,
                        implied_no_ask_cents=100,
                        seq=book.last_seq,
                        valid=False,
                    )
                    await self._bus.publish(f"kalshi.orderbook.{ticker}", _invalid_ob)
                    await self._bus.publish("kalshi.orderbook", _invalid_ob)

                # Force full reconnect to get fresh snapshots. In-place unsubscribe+subscribe
                # often never gets "subscribed" confirmations (Kalshi timeout), so reconnect
                # so the new connection's first subscribe gets confirmations reliably.
                n = len(self._seq_gap_times)
                window = now_mono - self._seq_gap_times[0] if n else 0
                cooldown_elapsed = (now_mono - self._last_resubscribe_mono) >= self._SEQ_GAP_RESUBSCRIBE_COOLDOWN_S
                # Reconnect on sustained seq-gap patterns.
                # Case A: slow persistent drift over the debounce window.
                # Case B: extreme burst in a short window (often duplicate/stale
                # subscription state) — reconnect immediately instead of waiting.
                burst_threshold = max(20, self._SEQ_GAP_DEBOUNCE_COUNT * 4)
                slow_pattern = (
                    n >= self._SEQ_GAP_DEBOUNCE_COUNT
                    and self._SEQ_GAP_MIN_WINDOW_S <= window <= self._SEQ_GAP_DEBOUNCE_WINDOW_S
                )
                burst_pattern = (n >= burst_threshold and window < self._SEQ_GAP_MIN_WINDOW_S)
                if (
                    (slow_pattern or burst_pattern)
                    and cooldown_elapsed
                    and self._desired_channels
                    and self._desired_tickers
                ):
                    reason = "burst" if burst_pattern else "slow"
                    log.warning(
                        "Seq-gap storm detected — forcing reconnect for fresh snapshots",
                        data={"gaps": n, "window_s": round(window, 2), "pattern": reason},
                    )
                    self._seq_gap_times.clear()
                    self._last_resubscribe_mono = now_mono
                    self._force_reconnect = True
                return

            # Publish signed add/remove flow events derived from deltas.
            delta_entries = payload if isinstance(payload, list) else [payload]
            evt_ts = time.time()
            for entry in delta_entries:
                if not isinstance(entry, dict):
                    continue
                side = str(entry.get("side", "")).lower()
                if side not in ("yes", "no"):
                    continue
                delta_qty = parse_count_centicx(entry, "delta", "delta_fp", default=0)
                if delta_qty == 0:
                    continue
                evt = KalshiOrderDeltaEvent(
                    market_ticker=ticker,
                    side=side,
                    is_add=(delta_qty > 0),
                    qty=abs(int(delta_qty)),
                    ts=evt_ts,
                )
                await self._bus.publish("kalshi.orderbook_delta_flow", evt)
                await self._bus.publish(f"kalshi.orderbook_delta_flow.{ticker}", evt)

        # Publish current book state — only if valid (after snapshot) or
        # after a successfully applied delta.
        # Multi-level depth pressure from top-5 levels (deeper levels down-weighted).
        yes_levels = book.yes_bids.levels_snapshot()[:5]
        no_levels = book.no_bids.levels_snapshot()[:5]
        weighted_yes = sum(qty * (1.0 / (1.0 + i)) for i, (_, qty) in enumerate(yes_levels))
        weighted_no = sum(qty * (1.0 / (1.0 + i)) for i, (_, qty) in enumerate(no_levels))
        total_weighted = weighted_yes + weighted_no
        depth_pressure = ((weighted_yes - weighted_no) / total_weighted) if total_weighted > 0 else 0.0
        depth_pressure = max(-1.0, min(1.0, float(depth_pressure)))

        _ob_state = OrderbookState(
            market_ticker=ticker,
            best_yes_bid_cents=book.best_yes_bid_cents,
            best_no_bid_cents=book.best_no_bid_cents,
            implied_yes_ask_cents=book.implied_yes_ask_cents,
            implied_no_ask_cents=book.implied_no_ask_cents,
            seq=book.last_seq,
            valid=book.valid,
            obi=book.order_book_imbalance,
            depth_pressure=depth_pressure,
            micro_price_cents=book.micro_price_cents,
            best_yes_depth=book.best_yes_bid_depth,
            best_no_depth=book.best_no_bid_depth,
        )
        await self._bus.publish(f"kalshi.orderbook.{ticker}", _ob_state)
        await self._bus.publish("kalshi.orderbook", _ob_state)

    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        # Publish executed trade events for trade-flow tracking.
        ticker = data.get("market_ticker", "")
        taker_side = data.get("taker_side", "")  # "yes" or "no"
        count = int(data.get("count", 0))
        ts = float(data.get("ts", 0)) / 1000.0  # ms → seconds
        if not ticker or not taker_side or count <= 0:
            return
        event = KalshiTradeEvent(
            market_ticker=ticker,
            taker_side=taker_side,
            count=count,
            ts=ts,
        )
        await self._bus.publish("kalshi.trade", event)
        await self._bus.publish(f"kalshi.trade.{ticker}", event)

    async def _handle_ticker(self, payload: Dict[str, Any]) -> None:
        ticker = payload.get("market_ticker", "")
        ts = time.time()
        await self._bus.publish(
            f"kalshi.ticker.{ticker}",
            TickerUpdate(
                market_ticker=ticker,
                yes_bid_cents=parse_price_cents(payload, "yes_bid", "yes_bid_dollars"),
                yes_ask_cents=parse_price_cents(payload, "yes_ask", "yes_ask_dollars"),
                last_price_cents=parse_price_cents(payload, "last_price", "last_price_dollars"),
                volume=int(payload.get("volume", 0)),
                timestamp=ts,
            ),
        )

    async def _handle_fill(self, payload: Dict[str, Any]) -> None:
        fill = payload
        await self._bus.publish(
            "kalshi.fills",
            FillEvent(
                market_ticker=fill.get("market_ticker", ""),
                order_id=fill.get("order_id", ""),
                side=fill.get("side", ""),
                price_cents=parse_price_cents(fill, "yes_price", "yes_price_dollars"),
                count=parse_count_centicx(fill, "count", "count_fp", default=0),
                is_taker=fill.get("is_taker", False),
                timestamp=time.time(),
            ),
        )

    async def _handle_user_orders(self, payload: Dict[str, Any]) -> None:
        order = payload
        await self._bus.publish(
            "kalshi.user_orders",
            OrderUpdate(
                market_ticker=order.get("market_ticker", ""),
                order_id=order.get("order_id", ""),
                status=order.get("status", ""),
                side=order.get("side", ""),
                price_cents=parse_price_cents(order, "yes_price", "yes_price_dollars"),
                quantity_contracts=parse_count_whole(order, "count", "count_fp", default=0),
                filled_contracts=parse_count_whole(order, "filled_count", "filled_count_fp", default=0),
                remaining_contracts=parse_count_whole(order, "remaining_count", "remaining_count_fp", default=0),
                timestamp=time.time(),
            ),
        )
