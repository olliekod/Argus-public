"""
Bybit WebSocket Connector
=========================

Public WebSocket client for Bybit perpetual futures data.
No authentication required - uses public endpoints only.
"""

import asyncio
import json
import random
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import websockets
from websockets.exceptions import ConnectionClosed

from ..core.logger import get_connector_logger
from ..core.events import QuoteEvent, MetricEvent, TOPIC_MARKET_QUOTES, TOPIC_MARKET_METRICS

logger = get_connector_logger('bybit')


def _ws_is_open(ws) -> bool:
    """Check if a websocket connection is open across websockets versions.

    Works with:
    - websockets <13 where ``ws.open`` exists
    - websockets >=13 where ``ws.closed`` exists but ``open`` does not
    - Unexpected wrapper types where neither attribute is present
    """
    if ws is None:
        return False
    # Prefer .closed (available in all modern versions)
    if hasattr(ws, 'closed'):
        return not ws.closed
    # Fallback for older versions that expose .open
    if hasattr(ws, 'open'):
        return ws.open
    # Check state attribute (websockets internal)
    if hasattr(ws, 'state'):
        try:
            from websockets.protocol import State
            return ws.state == State.OPEN
        except (ImportError, AttributeError):
            pass
    # Cannot determine - assume disconnected (will trigger reconnect)
    return False


class BybitWebSocket:
    """
    Bybit public WebSocket client for perpetual futures.

    Provides:
    - Real-time price updates
    - Funding rate data
    - Order book depth

    No API key required for public data.
    """

    # Public WebSocket endpoints
    MAINNET_URL = "wss://stream.bybit.com/v5/public/linear"
    TESTNET_URL = "wss://stream-testnet.bybit.com/v5/public/linear"

    def __init__(
        self,
        symbols: List[str],
        testnet: bool = False,
        on_ticker: Optional[Callable] = None,
        on_funding: Optional[Callable] = None,
        on_orderbook: Optional[Callable] = None,
        event_bus=None,
    ):
        self.url = self.TESTNET_URL if testnet else self.MAINNET_URL
        self.symbols = [self._normalize_symbol(s) for s in symbols]

        # Callbacks
        self.on_ticker = on_ticker
        self.on_funding = on_funding
        self.on_orderbook = on_orderbook

        # Event bus (optional — publishes QuoteEvents when set)
        self._event_bus = event_bus

        # Connection state
        self._ws = None
        self._running = False
        self._reconnect_delay = 5
        self._max_reconnect_delay = 300
        self._ping_interval = 20
        self._ping_timeout = 10
        self._close_timeout = 5
        self._recv_timeout = 60

        # Health / observability
        self.last_message_ts: Optional[float] = None
        self.last_heartbeat_ts: Optional[float] = None
        self.reconnect_attempts: int = 0
        self._connected_since: Optional[float] = None
        self._message_count: int = 0
        self._message_count_ts: float = time.time()
        self._message_rate_per_min: float = 0.0
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures: int = 0
        self.request_count: int = 0
        self.error_count: int = 0
        self.last_latency_ms: Optional[float] = None
        self.avg_latency_ms: Optional[float] = None
        self.last_close_code: Optional[int] = None
        self.last_close_reason: Optional[str] = None

        # Latest data cache
        self.tickers: Dict[str, Dict] = {}
        self.funding_rates: Dict[str, Dict] = {}
        self.quotes_invalid_total: int = 0
        self.quotes_invalid_by_symbol: "OrderedDict[str, int]" = OrderedDict()
        self.quotes_invalid_by_reason: "OrderedDict[str, int]" = OrderedDict()
        self._invalid_log_ts: Dict[Tuple[str, str], float] = {}
        self._invalid_symbol_cap = 100
        self._invalid_reason_cap = 50
        self._invalid_log_interval_s = 60

        logger.info(f"Bybit WebSocket initialized for {len(self.symbols)} symbols")

    def _normalize_symbol(self, symbol: str) -> str:
        if ':' in symbol:
            symbol = symbol.split(':')[0]
        return symbol.replace('/', '')

    async def connect(self) -> None:
        """Start WebSocket connection and message loop."""
        self._running = True

        while self._running:
            try:
                logger.info(f"Connecting to Bybit WebSocket: {self.url}")

                async with websockets.connect(
                    self.url,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    close_timeout=self._close_timeout,
                ) as ws:
                    self._ws = ws
                    self.reconnect_attempts = 0
                    self._reconnect_delay = 5
                    self._connected_since = time.time()
                    self.last_message_ts = None
                    self.last_heartbeat_ts = None
                    self._message_count = 0
                    self._message_count_ts = time.time()
                    self._message_rate_per_min = 0.0

                    logger.info("Bybit WebSocket connected")

                    # Subscribe to channels
                    await self._subscribe()

                    # Message loop
                    await self._message_loop()

            except ConnectionClosed as e:
                self.last_close_code = e.code
                self.last_close_reason = e.reason
                self.last_error = f"closed:{e.code}"
                self.error_count += 1
                self.consecutive_failures += 1
                logger.warning(f"Bybit WebSocket closed: {e.code} - {e.reason}")
            except Exception as e:
                self.last_error = str(e)
                self.error_count += 1
                self.consecutive_failures += 1
                logger.error(f"Bybit WebSocket error: {e}")

            self._connected_since = None
            self.last_message_ts = None
            self.last_heartbeat_ts = None

            if self._running:
                self.reconnect_attempts += 1
                delay = min(
                    self._reconnect_delay * (2 ** min(self.reconnect_attempts, 6)),
                    self._max_reconnect_delay,
                )
                # Add jitter to avoid thundering herd
                delay += random.uniform(0, min(delay * 0.1, 5))
                logger.info(f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_attempts})")
                await asyncio.sleep(delay)

        self._ws = None

    async def disconnect(self) -> None:
        self._running = False
        if self._ws and _ws_is_open(self._ws):
            await self._ws.close()
            logger.info("Bybit WebSocket disconnected")
        self._connected_since = None
        self.last_message_ts = None
        self.last_heartbeat_ts = None

    async def _subscribe(self) -> None:
        if not self._ws:
            return
        ticker_topics = [f"tickers.{symbol}" for symbol in self.symbols]
        subscribe_msg = {"op": "subscribe", "args": ticker_topics}
        await self._ws.send(json.dumps(subscribe_msg))
        logger.debug(f"Subscribed to {len(ticker_topics)} ticker channels")

    async def _message_loop(self) -> None:
        if not self._ws:
            return
        while self._running and self._ws and _ws_is_open(self._ws):
            try:
                message = await asyncio.wait_for(self._ws.recv(), timeout=self._recv_timeout)
            except asyncio.TimeoutError:
                try:
                    await self._ws.ping()
                except Exception as e:
                    self.last_error = f"ping_failed:{e}"
                    self.error_count += 1
                    self.consecutive_failures += 1
                    logger.warning(f"Bybit WebSocket ping failed: {e}")
                    break
                continue
            except ConnectionClosed as e:
                self.last_close_code = e.code
                self.last_close_reason = e.reason
                self.last_error = f"closed:{e.code}"
                self.error_count += 1
                self.consecutive_failures += 1
                logger.warning(f"Bybit WebSocket closed: {e.code} - {e.reason}")
                break
            except Exception as e:
                self.last_error = str(e)
                self.error_count += 1
                self.consecutive_failures += 1
                logger.error(f"Bybit WebSocket recv error: {e}")
                break

            self.last_message_ts = time.time()
            self.last_success_ts = self.last_message_ts
            self._message_count += 1
            self.request_count += 1
            self.consecutive_failures = 0
            now = time.time()
            if now - self._message_count_ts >= 60:
                self._message_rate_per_min = self._message_count / max((now - self._message_count_ts) / 60, 1)
                self._message_count = 0
                self._message_count_ts = now
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError:
                self.last_error = "invalid_json"
                self.error_count += 1
                self.consecutive_failures += 1
                logger.warning(f"Invalid JSON received: {message[:100]}")
            except Exception as e:
                self.last_error = str(e)
                self.error_count += 1
                self.consecutive_failures += 1
                logger.error(f"Error handling message: {e}")

    async def _handle_message(self, data: Dict[str, Any]) -> None:
        if data.get('op') == 'subscribe':
            if data.get('success'):
                logger.debug("Subscription confirmed")
            else:
                logger.warning(f"Subscription failed: {data}")
            return
        if data.get('op') == 'pong':
            self.last_heartbeat_ts = time.time()
            return
        topic = data.get('topic', '')
        if topic.startswith('tickers.'):
            await self._handle_ticker(data)

    async def _handle_ticker(self, data: Dict[str, Any]) -> None:
        """Handle ticker message. Only publish a quote when current message has valid bid and ask (no filling from stale data)."""
        try:
            ticker_data = data.get('data', {})
            symbol = ticker_data.get('symbol', '')
            bid_raw = ticker_data.get('bid1Price')
            ask_raw = ticker_data.get('ask1Price')
            last_raw = ticker_data.get('lastPrice')
            bid, bid_err = self._coerce_float(bid_raw)
            ask, ask_err = self._coerce_float(ask_raw)
            last, last_err = self._coerce_float(last_raw)
            # Require both bid and ask in this message for a valid quote (no synthetic/stale fill)
            invalid_reason = None
            if bid_err or ask_err:
                if bid_err == "missing" or ask_err == "missing":
                    invalid_reason = "missing_fields"
                else:
                    invalid_reason = "non_numeric"
            elif last_err and last_err != "missing":
                invalid_reason = "non_numeric"
            elif bid is not None and ask is not None:
                if bid <= 0 or ask <= 0:
                    invalid_reason = "zero_bidask"
                elif bid > ask:
                    invalid_reason = "crossed"
                elif last is not None and last <= 0 and (bid <= 0 or ask <= 0):
                    invalid_reason = "zero_bidask"
            if invalid_reason:
                self._record_invalid_quote(
                    symbol=symbol,
                    reason=invalid_reason,
                    bid_raw=bid_raw,
                    ask_raw=ask_raw,
                    last_raw=last_raw,
                )
            bid = bid if bid is not None else 0.0
            ask = ask if ask is not None else 0.0
            if bid == 0.0 and ask == 0.0 and not invalid_reason:
                invalid_reason = "both_bidask_missing"
                self._record_invalid_quote(symbol=symbol, reason=invalid_reason, bid_raw=bid_raw, ask_raw=ask_raw, last_raw=last_raw)
            # Use mid for last only when we have both bid and ask in this message (same-message derivation, not stale)
            last = last if last is not None else ((bid + ask) / 2 if (bid and ask) else 0.0)
            parsed = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'last_price': last,
                'mark_price': float(ticker_data.get('markPrice', 0)),
                'index_price': float(ticker_data.get('indexPrice', 0)),
                'bid_price': bid,
                'ask_price': ask,
                'volume_24h': float(ticker_data.get('volume24h', 0)),
                'turnover_24h': float(ticker_data.get('turnover24h', 0)),
                'funding_rate': float(ticker_data.get('fundingRate', 0)),
                'next_funding_time': ticker_data.get('nextFundingTime'),
                'open_interest': float(ticker_data.get('openInterest', 0)),
                'open_interest_value': float(ticker_data.get('openInterestValue', 0)),
                'price_change_24h': float(ticker_data.get('price24hPcnt', 0)) * 100,
            }
            self.tickers[symbol] = parsed
            if parsed['funding_rate'] != 0:
                self.funding_rates[symbol] = {
                    'symbol': symbol,
                    'rate': parsed['funding_rate'],
                    'next_time': parsed['next_funding_time'],
                    'timestamp': parsed['timestamp'],
                }
            if self.on_ticker:
                try:
                    if asyncio.iscoroutinefunction(self.on_ticker):
                        await self.on_ticker(parsed)
                    else:
                        self.on_ticker(parsed)
                except Exception as e:
                    logger.error("Ticker callback error (%s): %s", type(e).__name__, e)
                    logger.debug("Bybit ticker callback error detail", exc_info=True)
            if self.on_funding and parsed['funding_rate'] != 0:
                try:
                    if asyncio.iscoroutinefunction(self.on_funding):
                        await self.on_funding(self.funding_rates[symbol])
                    else:
                        self.on_funding(self.funding_rates[symbol])
                except Exception as e:
                    logger.error("Funding callback error (%s): %s", type(e).__name__, e)
                    logger.debug("Bybit funding callback error detail", exc_info=True)

            # Publish QuoteEvent to the event bus (price-only)
            if self._event_bus is not None:
                try:
                    now = time.time()

                    # Extract upstream timestamp from Bybit WS payload.
                    # Bybit sends 'ts' at message level in milliseconds.
                    raw_ts = data.get('ts')
                    if raw_ts is not None:
                        source_ts = float(raw_ts) / 1000.0  # ms → seconds
                        # Validate timestamp sanity (not 1970, not far future)
                        if abs(now - source_ts) > 86400:  # more than 24h drift
                            logger.warning(
                                "Bybit timestamp drift >24h for %s: source_ts=%.1f, now=%.1f",
                                symbol, source_ts, now,
                            )
                            source_ts = now
                    else:
                        source_ts = now  # fallback: use receive time

                    if not invalid_reason:
                        bid = parsed.get('bid_price', 0.0)
                        ask = parsed.get('ask_price', 0.0)
                        mid = (bid + ask) / 2 if (bid and ask) else parsed['last_price']

                        quote = QuoteEvent(
                            symbol=symbol,
                            bid=bid,
                            ask=ask,
                            mid=mid,
                            last=parsed['last_price'],
                            timestamp=now,
                            source='bybit',
                            volume_24h=parsed.get('volume_24h', 0.0),
                            source_ts=source_ts,
                        )
                        self._event_bus.publish(TOPIC_MARKET_QUOTES, quote)

                    # Publish non-price metrics separately
                    fr = parsed.get('funding_rate', 0.0)
                    if fr:
                        self._event_bus.publish(TOPIC_MARKET_METRICS, MetricEvent(
                            symbol=symbol, metric='funding_rate',
                            value=fr, timestamp=now, source='bybit',
                        ))
                    oi = parsed.get('open_interest', 0.0)
                    if oi:
                        self._event_bus.publish(TOPIC_MARKET_METRICS, MetricEvent(
                            symbol=symbol, metric='open_interest',
                            value=oi, timestamp=now, source='bybit',
                        ))
                except Exception as e:
                    logger.error("QuoteEvent publish error (%s): %s", type(e).__name__, e)
                    logger.debug("Bybit QuoteEvent publish error detail", exc_info=True)
        except Exception as e:
            logger.error("Error parsing ticker (%s): %s", type(e).__name__, e)
            logger.debug("Bybit ticker parse error detail", exc_info=True)

    @staticmethod
    def _coerce_float(value: Any) -> Tuple[Optional[float], Optional[str]]:
        if value is None:
            return None, "missing"
        if isinstance(value, str) and not value.strip():
            return None, "missing"
        try:
            return float(value), None
        except (TypeError, ValueError):
            return None, "non_numeric"

    def _record_invalid_quote(
        self,
        *,
        symbol: str,
        reason: str,
        bid_raw: Any,
        ask_raw: Any,
        last_raw: Any,
    ) -> None:
        self.quotes_invalid_total += 1
        self._bump_bounded_counter(
            self.quotes_invalid_by_symbol, symbol, self._invalid_symbol_cap
        )
        self._bump_bounded_counter(
            self.quotes_invalid_by_reason, reason, self._invalid_reason_cap
        )
        now = time.time()
        log_key = (symbol, reason)
        last_log = self._invalid_log_ts.get(log_key, 0.0)
        if (now - last_log) >= self._invalid_log_interval_s:
            self._invalid_log_ts[log_key] = now
            logger.warning(
                "Rejected Bybit quote for %s (%s): bid=%r ask=%r last=%r",
                symbol,
                reason,
                bid_raw,
                ask_raw,
                last_raw,
            )

    @staticmethod
    def _bump_bounded_counter(
        store: "OrderedDict[str, int]",
        key: str,
        cap: int,
    ) -> None:
        if key in store:
            store[key] += 1
            store.move_to_end(key)
            return
        if len(store) >= cap:
            store.popitem(last=False)
        store[key] = 1

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        normalized = self._normalize_symbol(symbol)
        return self.tickers.get(normalized)

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        normalized = self._normalize_symbol(symbol)
        data = self.funding_rates.get(normalized)
        return data['rate'] if data else None

    def get_price(self, symbol: str) -> Optional[float]:
        ticker = self.get_ticker(symbol)
        return ticker['last_price'] if ticker else None

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected (version-safe)."""
        return _ws_is_open(self._ws)

    def get_health_status(self) -> Dict[str, Any]:
        """Return detailed health info for dashboards and health checks."""
        now = time.time()
        connected = self.is_connected
        since_last_msg = None
        if connected and self.last_message_ts:
            since_last_msg = (now - self.last_message_ts)
        if not connected:
            status = "down" if self.last_error else "unknown"
        elif since_last_msg is not None and since_last_msg > (self._recv_timeout * 2):
            status = "degraded"
        else:
            status = "ok"

        from ..core.status import build_status

        return build_status(
            name="bybit",
            type="ws",
            status=status,
            last_success_ts=self.last_success_ts,
            last_error=self.last_error,
            consecutive_failures=self.consecutive_failures,
            reconnect_attempts=self.reconnect_attempts,
            request_count=self.request_count,
            error_count=self.error_count,
            avg_latency_ms=self.avg_latency_ms,
            last_latency_ms=self.last_latency_ms,
            last_message_ts=self.last_message_ts,
            age_seconds=round(since_last_msg, 1) if since_last_msg is not None else None,
            extras={
                "connected": connected,
                "connected_since": (
                    datetime.fromtimestamp(self._connected_since, tz=timezone.utc).isoformat()
                    if self._connected_since
                    else None
                ),
                "symbols": len(self.symbols),
                "message_rate_per_min": round(self._message_rate_per_min, 2),
                "close_code": self.last_close_code,
                "close_reason": self.last_close_reason,
                "quotes_invalid_total": self.quotes_invalid_total,
                "quotes_invalid_by_symbol": dict(self.quotes_invalid_by_symbol),
                "quotes_invalid_by_reason": dict(self.quotes_invalid_by_reason),
            },
        )
