# Created by Oliver Meihls

# OKX WebSocket Fallback Connector
#
# Public WebSocket client for OKX spot ticker data. Used as a resilient
# fallback when the primary truth feed (e.g. Coinbase WS) is down or stale.
#
# OKX public tickers channel does not require API key or authentication.
# If you have an OKX API key, it is only needed for private channels (e.g. trading).

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional
import websockets
from websockets.exceptions import ConnectionClosed

from ..core.logger import get_connector_logger

logger = get_connector_logger("okx_ws")


class OkxWsFallback:
    # OKX V5 public WebSocket client for spot tickers.
    #
    # Subscribes to the tickers channel (e.g. BTC-USDT) and invokes
    # on_ticker with a normalized payload compatible with the Kalshi
    # truth-feed callback (last_price, symbol, etc.).

    # Public WebSocket endpoint — no auth required for tickers
    DEFAULT_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

    def __init__(
        self,
        inst_ids: List[str],
        on_ticker: Optional[Callable] = None,
        ws_url: Optional[str] = None,
    ):
        # Args:
        # inst_ids: OKX instrument IDs, e.g. ["BTC-USDT"].
        # on_ticker: Async or sync callback receiving dict with last_price, symbol, etc.
        # ws_url: Override WebSocket URL (default: public v5 endpoint).
        self.inst_ids = list(inst_ids)
        self.on_ticker = on_ticker
        self._ws_url = (ws_url or self.DEFAULT_WS_URL).rstrip("/")
        self._ws: Optional[Any] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1.0
        self._last_mid_prices: Dict[str, float] = {}
        logger.info("OKX WebSocket fallback initialized for %s", self.inst_ids)

    async def start(self) -> None:
        # Start the WebSocket connection and message loop (background task).
        self._running = True
        self._task = asyncio.create_task(self._connect_loop())

    async def stop(self) -> None:
        # Stop the WebSocket and cancel the background task.
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

            self._task = None
        logger.info("OKX WebSocket fallback stopped")

    async def _connect_loop(self) -> None:
        # Connect, subscribe, and run message loop; reconnect on failure.
        retry_count = 0
        while self._running:
            try:
                logger.info("Connecting to OKX WebSocket: %s", self._ws_url)
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=25,
                    ping_timeout=10,
                    open_timeout=20,
                ) as ws:
                    self._ws = ws
                    # Subscribe to tickers
                    sub = {
                        "op": "subscribe",
                        "args": [{"channel": "tickers", "instId": inst_id} for inst_id in self.inst_ids],
                    }
                    await ws.send(json.dumps(sub))
                    logger.info("OKX subscribed to tickers: %s", self.inst_ids)
                    retry_count = 0
                    await self._message_loop()
            except ConnectionClosed as e:
                logger.warning("OKX WebSocket closed: %s", e.code)
            except Exception as e:
                logger.error("OKX WebSocket error: %s", e)
            self._ws = None
            if self._running:
                retry_count += 1
                delay = min(self._reconnect_delay * (2 ** min(retry_count, 5)), 60)
                logger.info("OKX reconnect in %.1fs...", delay)
                await asyncio.sleep(delay)

    async def _message_loop(self) -> None:
        # Process incoming messages; handle ticker pushes and event confirmations.
        if not self._ws:
            return
        async for raw in self._ws:
            if not self._running:
                break
            try:
                msg = json.loads(raw)
                # Subscription confirmation: {"event":"subscribe","arg":{...}}
                if msg.get("event") == "subscribe":
                    continue
                if msg.get("event") == "error":
                    logger.error("OKX WS error: %s", msg.get("msg", msg))
                    continue
                # Ticker push: {"arg":{...},"data":[{...}]}
                data_list = msg.get("data")
                if not data_list or not isinstance(data_list, list):
                    continue
                for item in data_list:
                    if isinstance(item, dict):
                        await self._handle_ticker(item)
            except Exception as e:
                logger.error("OKX message parse error: %s", e)

    async def _handle_ticker(self, raw: Dict[str, Any]) -> None:
        # Parse OKX ticker object and invoke callback with normalized payload.
        # OKX: last, lastPx (legacy), bidPx, askPx, ts (ms), instId
        inst_id = raw.get("instId", "")
        last = raw.get("last") or raw.get("lastPx")
        if last is None:
            return
        try:
            price = float(last)
        except (TypeError, ValueError):
            return
        if price <= 0:
            return
        if self._last_mid_prices.get(inst_id) == price:
            return
        self._last_mid_prices[inst_id] = price

        ts_ms = raw.get("ts")
        if ts_ms is not None:
            try:
                ts_sec = int(ts_ms) / 1000.0
            except (TypeError, ValueError):
                ts_sec = time.time()
        else:
            ts_sec = time.time()

        # Normalize to same shape as Luzia/Coinbase callback for runner
        symbol = inst_id.replace("-", "") if inst_id else "BTCUSDT"
        parsed = {
            "symbol": symbol,
            "exchange": "okx",
            "instId": inst_id,
            "timestamp": ts_sec,
            "last_price": price,
            "bid_price": float(raw.get("bidPx") or 0),
            "ask_price": float(raw.get("askPx") or 0),
            "volume_24h": float(raw.get("vol24h") or raw.get("volCcy24h") or 0),
        }

        if self.on_ticker:
            try:
                if asyncio.iscoroutinefunction(self.on_ticker):
                    await self.on_ticker(parsed)
                else:
                    self.on_ticker(parsed)
            except Exception as e:
                logger.error("OKX ticker callback error: %s", e)
