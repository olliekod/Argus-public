"""
Order execution and risk management for Kalshi BTC strike contracts.

Execution policy
----------------
* Upon receiving a ``TradeSignal``, place **one aggressive limit order**
  at the implied ask (crosses the spread).
* If the order is not filled within ``order_timeout_ms``, cancel it.
* **No chasing**: a cancelled order is not retried at a worse price.

Local idempotency
-----------------
Even though the REST client attaches an ``Idempotency-Key`` header,
we enforce local dedup:
* Generate a ``client_order_id`` per signal.
* Track outstanding orders per ``market_ticker``.
* Refuse to place a second order for a market that already has a
  pending order (unless explicitly replacing it).

This protects against duplicate signals arriving from the strategy
engine faster than the REST round-trip.

Risk safeguards
---------------
* Per-market position cap (``max_fraction_per_market``).
* Daily drawdown kill switch (``daily_drawdown_limit``).
* Automatic halt on WebSocket disconnect or invalid orderbook state.
* Halt on repeated auth failures (published as ``auth_failure`` risk).

Emergency cancel-all
--------------------
Uses sequential cancels that respect the write rate limiter, with a
small sleep between requests to avoid cascading 429s.  Kalshi does not
(as of March 2026) expose a batch-cancel endpoint.

Fixed-point quantities
----------------------
Orders are sent with ``count`` as an integer of whole contracts.
The ``count_fp`` field is also populated as a ``"X.00"`` string for
forward-compatibility with Kalshi's *_fp migration (March 5 2026).

Dry-run paper trading
---------------------
When ``dry_run=True``, no REST orders are placed. Instead, each
``TradeSignal`` (including from the mispricing scalper) is turned into
a synthetic ``FillEvent`` on ``kalshi.fills``. The settlement tracker
builds positions from these fills and resolves them at contract
settlement, so PnL and win rate in the UI reflect paper performance.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Set

from .bus import Bus
from .config import KalshiConfig
from .kalshi_rest import KalshiRestClient
from .logging_utils import ComponentLogger
from .models import (
    FillEvent,
    MarketMetadata,
    OrderbookState,
    OrderUpdate,
    RiskEvent,
    TradeSignal,
    TerminalEvent,
)
from .latency_model import EmpiricalLatencyModel
from .decision_context import build_decision_context
from .paper_model import estimate_kalshi_taker_fee_usd
from .paper_log import append_paper_log_sync
from .simulation import SCENARIO_BASE, SCENARIO_PROFILES, ScenarioProfile, assign_family

log = ComponentLogger("execution")

# Stagger between sequential cancels in emergency cancel-all.
_CANCEL_STAGGER_S = 0.1

_GLOBAL_LATENCY_MODEL = EmpiricalLatencyModel()


def _append_paper_log(record: dict) -> asyncio.Task:
    """Non-blocking paper log append. Dispatches to the default thread executor.

    Returns the Task so callers can optionally await it.  In hot paths
    (farm paper trading) callers fire-and-forget; the write happens on a
    background thread so the event loop is never blocked.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, append_paper_log_sync, record)
    except RuntimeError:
        # No running loop (e.g. during tests) — write synchronously.
        append_paper_log_sync(record)
        return None


class _PendingOrder:
    """Tracks an in-flight order and its timeout task."""

    __slots__ = ("order_id", "client_order_id", "signal", "placed_at", "timeout_task", "fill_task")

    def __init__(
        self,
        order_id: str,
        client_order_id: str,
        signal: TradeSignal,
    ) -> None:
        self.order_id = order_id
        self.client_order_id = client_order_id
        self.signal = signal
        self.placed_at = time.monotonic()
        self.timeout_task: Optional[asyncio.Task] = None
        self.fill_task: Optional[asyncio.Task] = None


class ExecutionEngine:
    """Executes TradeSignals via REST and reconciles with WS feeds."""

    def __init__(
        self,
        config: KalshiConfig,
        bus: Bus,
        rest: KalshiRestClient,
        db: Optional[Any] = None,
        shared_orderbooks: Optional[Dict[str, OrderbookState]] = None,
        shared_metadata: Optional[Dict[str, MarketMetadata]] = None,
        shared_truth_prices: Optional[Dict[str, float]] = None,
        shared_fair_probs: Optional[Dict[str, Any]] = None,
        shared_trade_flow: Optional[Dict[str, float]] = None,
    ) -> None:
        self._cfg = config
        self._bus = bus
        self._rest = rest
        self._db = db

        # In-flight orders keyed by order_id.
        self._pending: Dict[str, _PendingOrder] = {}
        # Local dedup: market_ticker → order_id of the pending order.
        # Only one order per market at a time.
        self._pending_by_market: Dict[str, str] = {}
        # Side-aware dedup map so paired strategies (e.g. YES+NO arbitrage)
        # can place one pending order per side on the same ticker.
        self._pending_by_market_side: Dict[tuple[str, str], str] = {}
        # Set of client_order_ids we have placed (idempotency guard).
        self._placed_client_ids: Set[str] = set()
        self._last_buy_signal_ts: Dict[str, float] = {}

        # Position tracking: ticker → net contracts.
        self._positions: Dict[str, int] = {}
        
        # Risk tracking.
        self._current_balance: float = config.bankroll_usd
        self._day_start_balance: float = config.bankroll_usd
        self._daily_pnl: float = 0.0
        self._consecutive_losing_days: int = 0

        # Latest orderbook state per market (for pre-trade depth check).
        self._orderbooks: Dict[str, OrderbookState] = (
            shared_orderbooks if shared_orderbooks is not None else {}
        )
        # Market metadata (for paper_fill log: settlement_time, strike, asset).
        self._metadata: Dict[str, MarketMetadata] = (
            shared_metadata if shared_metadata is not None else {}
        )
        # Latest truth price by asset (used for strike-distance decision context fields).
        self._truth_price_by_asset: Dict[str, float] = (
            shared_truth_prices if shared_truth_prices is not None else {}
        )
        # Fair probs for drift context (optional shared ref).
        self._fair_probs: Dict[str, Any] = (
            shared_fair_probs if shared_fair_probs is not None else {}
        )
        # Trade flow imbalance for flow context (optional shared ref).
        self._trade_flow: Dict[str, float] = (
            shared_trade_flow if shared_trade_flow is not None else {}
        )

        self._halted = False
        self._halt_reason: str = ""
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._managed_router_mode = False
        self._diag_counts: Dict[str, int] = {}
        self._diag_source_counts: Dict[str, int] = {}
        self._diag_family_counts: Dict[str, int] = {}
        self._diag_source_family_counts: Dict[str, int] = {}
        self._diag_gauges: Dict[str, float] = {}
        self._latency_model = _GLOBAL_LATENCY_MODEL
        # Scenario-driven paper execution (Gap 2 fix).
        # Resolved from config.scenario_profile ("best" | "base" | "stress").
        # Overrides individual paper_order_latency_*/paper_slippage_cents at fill time.
        _profile_name = getattr(config, "scenario_profile", "base") or "base"
        self._scenario_profile: ScenarioProfile = SCENARIO_PROFILES.get(
            _profile_name, SCENARIO_BASE
        )

    # -- lifecycle -----------------------------------------------------------

    async def start(self, *, subscribe_bus: bool = True) -> None:
        self._running = True
        self._managed_router_mode = not subscribe_bus

        if not subscribe_bus:
            return

        q_signals = await self._bus.subscribe("kalshi.trade_signal")
        self._tasks.append(asyncio.create_task(self._consume_signals(q_signals)))

        q_orders = await self._bus.subscribe("kalshi.user_orders")
        self._tasks.append(asyncio.create_task(self._consume_order_updates(q_orders)))

        q_fills = await self._bus.subscribe("kalshi.fills")
        self._tasks.append(asyncio.create_task(self._consume_fills(q_fills)))

        q_risk = await self._bus.subscribe("kalshi.risk")
        self._tasks.append(asyncio.create_task(self._consume_risk(q_risk)))

        q_ws = await self._bus.subscribe("kalshi.ws.status")
        self._tasks.append(asyncio.create_task(self._consume_ws_status(q_ws)))

        q_balance = await self._bus.subscribe("kalshi.account_balance")
        self._tasks.append(asyncio.create_task(self._consume_balance(q_balance)))

        q_meta = await self._bus.subscribe("kalshi.market_metadata")
        self._tasks.append(asyncio.create_task(self._consume_metadata(q_meta)))

        # Run startup risk checks in background to avoid blocking start() (esp. in tests).
        self._tasks.append(asyncio.create_task(self._check_consecutive_drawdown_days()))

    async def stop(self) -> None:
        self._running = False
        # Cancel all pending order timeouts.
        for po in self._pending.values():
            if po.timeout_task:
                po.timeout_task.cancel()
            if po.fill_task:
                po.fill_task.cancel()
        tasks = list(self._tasks)
        self._tasks.clear()
        for t in tasks:
            t.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                log.debug("Execution stop: some consumer tasks did not exit in time")

    async def subscribe_orderbook(self, ticker: str) -> None:
        """Lazily subscribe to orderbook updates for *ticker* (for depth checks)."""
        if ticker not in self._orderbooks:
            q_ob = await self._bus.subscribe(f"kalshi.orderbook.{ticker}")
            self._tasks.append(asyncio.create_task(self._consume_orderbook(q_ob, ticker)))

    async def _consume_orderbook(self, q: asyncio.Queue, ticker: str) -> None:
        try:
            while self._running:
                msg: OrderbookState = await q.get()
                self._orderbooks[ticker] = msg
        except asyncio.CancelledError:
            pass

    # -- signal handler ------------------------------------------------------

    async def _consume_signals(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                signal: TradeSignal = await q.get()
                await self.handle_signal(signal)
        except asyncio.CancelledError:
            pass

    async def handle_signal(self, signal: TradeSignal) -> None:
        if getattr(signal, "bot_id", "default") != self._cfg.bot_id:
            return
        source = getattr(signal, "source", "") or "strategy"
        meta = self._metadata.get(signal.market_ticker)
        family = self._ticker_family_from_meta(signal.market_ticker, meta)
        self._diag_counts["signals_received"] = self._diag_counts.get("signals_received", 0) + 1
        self._diag_source_counts[source] = self._diag_source_counts.get(source, 0) + 1
        self._diag_family_counts[family] = self._diag_family_counts.get(family, 0) + 1
        sf_key = f"{source}|{family}"
        self._diag_source_family_counts[sf_key] = self._diag_source_family_counts.get(sf_key, 0) + 1
        if not self._managed_router_mode:
            await self.subscribe_orderbook(signal.market_ticker)
        await self._execute_signal(signal)

    async def _execute_signal(self, signal: TradeSignal) -> None:
        # Per-ticker contract cap: authoritative gate for ALL signal sources.
        # The strategy engine has its own pre-filter, but the scalper does not.
        # Enforcing it here prevents strategy + scalper jointly exceeding the cap.
        if signal.action == "buy":
            ticker_cap = self._cfg.max_contracts_per_ticker
            if ticker_cap > 0:
                current_contracts = abs(self._positions.get(signal.market_ticker, 0)) // 100
                if current_contracts >= ticker_cap:
                    self._diag_counts["drop_ticker_cap"] = self._diag_counts.get("drop_ticker_cap", 0) + 1
                    log.debug(
                        f"Ticker cap ({current_contracts}/{ticker_cap}) for "
                        f"{signal.market_ticker} — dropping "
                        f"{signal.source or 'strategy'} signal"
                    )
                    return

        # Halt gate.
        if self._halted:
            if not self._cfg.dry_run or self._halt_reason in ("drawdown_breach", "consecutive_drawdown", "auth_failure"):
                log.debug(f"Halted ({self._halt_reason}) — dropping signal for {signal.market_ticker}")
                return

        # Daily drawdown check (relative to day start).
        if self._day_start_balance > 0:
            drawdown_pct = -self._daily_pnl / self._day_start_balance
            if drawdown_pct >= self._cfg.daily_drawdown_limit:
                log.warning(
                    f"Daily drawdown limit reached ({drawdown_pct:.1%}) — halting",
                    data={"pnl": self._daily_pnl, "start_balance": self._day_start_balance}
                )
                self._halted = True
                self._halt_reason = "drawdown_breach"
                await self._bus.publish(
                    "kalshi.terminal_event",
                    TerminalEvent("ALERT", f"DAILY DRAWDOWN LIMIT REACHED ({drawdown_pct:.1%}) — HALTING", time.time())
                )
                await self._cancel_all_pending()
                return

        ticker = signal.market_ticker

        # ── Local dedup: refuse if we already have a pending order
        #    for this market.  This prevents duplicate orders from
        #    signals arriving faster than the REST round-trip.
        #    Sell signals (early exit) bypass this guard — we always
        #    want to honour a close even if there is an in-flight buy.
        # Additional debounce in paper/live mode: in ultra-fast loops we can
        # see repeat buy signals for the same ticker before downstream events
        # (fills/order updates) are fully processed.
        now = time.time()
        if signal.action == "buy":
            last_ts = self._last_buy_signal_ts.get(ticker, 0.0)
            if (now - last_ts) < 0.20:
                self._diag_counts["drop_debounce"] = self._diag_counts.get("drop_debounce", 0) + 1
                return
            self._last_buy_signal_ts[ticker] = now
        side_key = (ticker, signal.side)
        if signal.action != "sell" and side_key in self._pending_by_market_side:
            existing_oid = self._pending_by_market_side[side_key]
            self._diag_counts["drop_pending_market"] = self._diag_counts.get("drop_pending_market", 0) + 1
            log.debug(
                f"Already have pending order {existing_oid} for {ticker}/{signal.side} — skipping signal"
            )
            return

        # ── Pre-trade depth check ─────────────────────────────────────
        # Ensure sufficient liquidity at the signal price before placing.
        # This applies to BOTH live and paper trades for simulation fidelity.
        # The depth fields on OrderbookState are in centi-contracts;
        # signal.quantity_contracts is whole contracts (= 100 centi-contracts each).
        ob = self._orderbooks.get(ticker)
        if ob and ob.valid:
            depth_centicx = ob.best_no_depth if signal.side == "yes" else ob.best_yes_depth
            required_centicx = signal.quantity_contracts * 100
            if depth_centicx > 0 and depth_centicx < required_centicx:
                self._diag_counts["drop_insufficient_depth"] = self._diag_counts.get("drop_insufficient_depth", 0) + 1
                log.debug(
                    f"Insufficient depth for {ticker}: need {required_centicx} centicx, have {depth_centicx} — skipping",
                    data={"bot_id": self._cfg.bot_id},
                )
                return

        if self._cfg.dry_run:
            paper_oid = f"paper-{uuid.uuid4().hex[:12]}"
            now = time.time()

            pending = _PendingOrder(paper_oid, paper_oid, signal)
            self._pending[paper_oid] = pending
            if signal.action != "sell":
                self._pending_by_market[ticker] = paper_oid
                self._pending_by_market_side[(ticker, signal.side)] = paper_oid

            await self._bus.publish(
                "kalshi.user_orders",
                OrderUpdate(
                    market_ticker=ticker,
                    order_id=paper_oid,
                    status="placed",
                    side=signal.side,
                    price_cents=signal.limit_price_cents,
                    quantity_contracts=signal.quantity_contracts,
                    filled_contracts=0,
                    remaining_contracts=signal.quantity_contracts,
                    timestamp=now,
                    bot_id=self._cfg.bot_id,
                ),
            )

            if signal.order_style == "passive":
                timeout_s = self._cfg.passive_order_timeout_ms / 1000.0
            else:
                timeout_s = self._cfg.order_timeout_ms / 1000.0
            pending.timeout_task = asyncio.create_task(self._timeout_order(paper_oid, timeout_s))
            pending.fill_task = asyncio.create_task(self._simulate_paper_fill(paper_oid))
            return

        client_oid = str(uuid.uuid4())

        # ── Build order body ──
        # Determine price params.
        if signal.side == "yes":
            yes_price = signal.limit_price_cents
            no_price = None
        else:
            yes_price = None
            no_price = signal.limit_price_cents

        try:
            resp = await self._rest.create_order(
                ticker=ticker,
                action=signal.action,
                side=signal.side,
                order_type="limit",
                count=signal.quantity_contracts,
                yes_price=yes_price,
                no_price=no_price,
                client_order_id=client_oid,
            )
        except Exception as exc:
            log.error(
                f"Order placement failed for {ticker}: {exc}",
                data={"signal": signal.to_json()},
            )
            await self._bus.publish(
                "kalshi.order_update",
                OrderUpdate(
                    market_ticker=ticker,
                    order_id="",
                    status="error",
                    side=signal.side,
                    price_cents=signal.limit_price_cents,
                    quantity_contracts=signal.quantity_contracts,
                    filled_contracts=0,
                    remaining_contracts=signal.quantity_contracts,
                    timestamp=time.time(),
                    error_detail=str(exc),
                    bot_id=self._cfg.bot_id,
                ),
            )
            return

        order_data = resp.get("order", resp)
        order_id = order_data.get("order_id", "")

        pending = _PendingOrder(order_id, client_oid, signal)
        self._pending[order_id] = pending
        self._pending_by_market[ticker] = order_id
        self._pending_by_market_side[(ticker, signal.side)] = order_id
        self._placed_client_ids.add(client_oid)

        argus_price = int(signal.p_yes * 100) if signal.side == "yes" else int((1.0 - signal.p_yes) * 100)
        log.debug(
            f"Order placed: {signal.side} {ticker} | Argus {argus_price}¢ vs Kalshi {signal.limit_price_cents}¢",
            data={"order_id": order_id, "qty": signal.quantity_contracts},
        )

        await self._bus.publish(
            "kalshi.order_update",
            OrderUpdate(
                market_ticker=ticker,
                order_id=order_id,
                status="placed",
                side=signal.side,
                price_cents=signal.limit_price_cents,
                quantity_contracts=signal.quantity_contracts,
                filled_contracts=0,
                remaining_contracts=signal.quantity_contracts,
                timestamp=time.time(),
                bot_id=self._cfg.bot_id,
            ),
        )

        # Start timeout task — passive orders get a longer timeout.
        if signal.order_style == "passive":
            timeout_s = self._cfg.passive_order_timeout_ms / 1000.0
        else:
            timeout_s = self._cfg.order_timeout_ms / 1000.0
        pending.timeout_task = asyncio.create_task(
            self._timeout_order(order_id, timeout_s)
        )

    async def _timeout_order(self, order_id: str, timeout_s: float) -> None:
        """Cancel the order if not filled within timeout."""
        try:
            await asyncio.sleep(timeout_s)
        except asyncio.CancelledError:
            return

        pending = self._pending.get(order_id)
        if not pending:
            return  # Already resolved.

        if self._cfg.dry_run:
            self._diag_counts["paper_timeouts"] = self._diag_counts.get("paper_timeouts", 0) + 1
            pending_latency_ms = max(0.0, (time.monotonic() - pending.placed_at) * 1000.0)
            self._diag_gauges["paper_timeout_latency_ms_total"] = self._diag_gauges.get(
                "paper_timeout_latency_ms_total", 0.0
            ) + pending_latency_ms
            log.debug(
                f"Paper order timeout ? cancelling {order_id}",
                data={"timeout_s": timeout_s},
            )
        else:
            self._diag_counts["live_timeouts"] = self._diag_counts.get("live_timeouts", 0) + 1
            log.info(
                f"Order timeout ? cancelling {order_id}",
                data={"timeout_s": timeout_s},
            )

        if not self._cfg.dry_run:
            try:
                await self._rest.cancel_order(order_id)
            except Exception as exc:
                log.error(f"Cancel failed for {order_id}: {exc}")
                # Still remove from pending — if the cancel failed the order
                # may have already been filled (WS update will reconcile).

        await self._bus.publish(
            "kalshi.user_orders",
            OrderUpdate(
                market_ticker=pending.signal.market_ticker,
                order_id=order_id,
                status="cancelled",
                side=pending.signal.side,
                price_cents=pending.signal.limit_price_cents,
                quantity_contracts=pending.signal.quantity_contracts,
                filled_contracts=0,
                remaining_contracts=pending.signal.quantity_contracts,
                timestamp=time.time(),
                bot_id=self._cfg.bot_id,
            ),
        )

        self._resolve_pending(order_id, "cancelled")

    def _resolve_pending(self, order_id: str, reason: str) -> None:
        """Remove an order from pending tracking."""
        pending = self._pending.pop(order_id, None)
        if pending:
            if pending.timeout_task:
                pending.timeout_task.cancel()
            if pending.fill_task:
                pending.fill_task.cancel()
            ticker = pending.signal.market_ticker
            if self._pending_by_market.get(ticker) == order_id:
                del self._pending_by_market[ticker]
            side_key = (ticker, pending.signal.side)
            if self._pending_by_market_side.get(side_key) == order_id:
                del self._pending_by_market_side[side_key]

    def _paper_book_price_and_depth(self, signal: TradeSignal, ob: Optional[OrderbookState]) -> tuple[Optional[int], int]:
        if ob is None or not ob.valid:
            return None, 0
        if signal.action == "buy":
            if signal.side == "yes":
                return ob.implied_yes_ask_cents, ob.best_no_depth
            return ob.implied_no_ask_cents, ob.best_yes_depth
        if signal.side == "yes":
            return ob.best_yes_bid_cents, ob.best_yes_depth
        return ob.best_no_bid_cents, ob.best_no_depth

    @staticmethod
    def _ticker_family_from_meta(ticker: str, meta: Optional[MarketMetadata]) -> str:
        if meta is None:
            return assign_family(ticker, "", 0, False)
        return assign_family(
            ticker,
            getattr(meta, "asset", "") or "",
            int(getattr(meta, "window_minutes", 0) or 0),
            bool(getattr(meta, "is_range", False)),
        )

    @staticmethod
    def _effective_fillable_contracts(
        signal: TradeSignal,
        order_style: str,
        depth_centicx: int,
        ob: Optional[OrderbookState],
        latency_s: float,
    ) -> int:
        """Estimate fillable contracts from depth/queue/latency, no fixed slippage."""
        available = max(0, depth_centicx // 100)
        if available <= 0:
            return 0
        if ob is None:
            # Deterministic fallback for sparse states/tests where we intentionally
            # bootstrap from signal-limit depth.
            return available

        spread = 0
        if ob is not None:
            try:
                spread = max(0, int(ob.implied_yes_ask_cents) + int(ob.implied_no_ask_cents) - 100)
            except Exception:
                spread = 0

        tau = 0.70 if order_style == "passive" else 0.35
        depth_decay = math.exp(-max(0.0, latency_s) / tau)
        spread_factor = max(0.20, 1.0 - (spread / 12.0))
        fillable = int(available * depth_decay * spread_factor)

        if order_style == "passive":
            # Model queue-ahead uncertainty for maker-style behavior.
            queue_ahead_frac = 0.35 + (random.random() * 0.90)
            fillable = int(fillable * max(0.0, 1.0 - queue_ahead_frac))
            # Passive fills can still miss entirely in quiet books.
            p_fill = max(0.05, min(0.95, 0.15 + depth_decay * 0.70))
            if random.random() > p_fill:
                return 0

        if fillable <= 0:
            # Small chance of micro-fill on thin books for aggressive orders.
            if order_style != "passive" and available > 0 and random.random() < (0.20 * depth_decay):
                fillable = 1
            else:
                return 0
        return min(available, fillable)

    async def _simulate_paper_fill(self, order_id: str) -> None:
        pending = self._pending.get(order_id)
        if pending is None:
            return
        signal = pending.signal
        # Apply scenario profile for realistic paper execution assumptions.
        profile = self._scenario_profile
        meta = self._metadata.get(signal.market_ticker)
        family = self._ticker_family_from_meta(signal.market_ticker, meta)
        latency_s = self._latency_model.sample_latency_s(
            family=family,
            order_style=signal.order_style,
            profile=profile,
        )
        if latency_s > 0:
            try:
                await asyncio.sleep(latency_s)
            except asyncio.CancelledError:
                return

        pending = self._pending.get(order_id)
        if pending is None:
            return
        signal = pending.signal
        ticker = signal.market_ticker
        ob = self._orderbooks.get(ticker)
        current_price, depth_centicx = self._paper_book_price_and_depth(signal, ob)
        if current_price is None:
            current_price = signal.limit_price_cents
            depth_centicx = signal.quantity_contracts * 100
        if depth_centicx <= 0:
            return

        if signal.action == "buy":
            executable = current_price <= signal.limit_price_cents
        else:
            executable = current_price >= signal.limit_price_cents
        if not executable:
            if signal.order_style == "passive":
                # Passive orders may get hit later at their limit.
                cross_prob = max(0.02, min(0.35, 0.05 + latency_s * 0.40))
                if random.random() < cross_prob:
                    current_price = signal.limit_price_cents
                    executable = True
            if not executable:
                return

        available_contracts = self._effective_fillable_contracts(
            signal=signal,
            order_style=signal.order_style,
            depth_centicx=depth_centicx,
            ob=ob,
            latency_s=latency_s,
        )
        if available_contracts <= 0:
            return
        fill_qty = min(signal.quantity_contracts, available_contracts)
        if fill_qty <= 0:
            return

        # No fixed per-fill slippage; fills are driven by future book + queue model.
        if signal.order_style == "passive":
            if signal.action == "buy":
                fill_price = min(current_price, signal.limit_price_cents)
            else:
                fill_price = max(current_price, signal.limit_price_cents)
        else:
            fill_price = current_price

        fill_centicx = fill_qty * 100
        # Apply fee_multiplier from scenario profile (>1.0 simulates wider exchange fees).
        base_fee = estimate_kalshi_taker_fee_usd(fill_price, fill_centicx) if self._cfg.paper_apply_fees else 0.0
        fee_usd = base_fee * profile.fee_multiplier
        # Spread drag: realistic cost of the bid-ask spread on exit.
        spread_drag_usd = profile.spread_drag_per_contract * fill_qty
        now = time.time()
        source = getattr(signal, "source", "") or ""
        observed_latency_ms = max(0.0, (time.monotonic() - pending.placed_at) * 1000.0)
        self._latency_model.observe(
            family=family,
            order_style=signal.order_style,
            scenario=profile.name,
            latency_ms=observed_latency_ms,
        )

        _fp = self._fair_probs.get(ticker)
        _drift = float(getattr(_fp, "drift", 0.0)) if _fp is not None else 0.0
        _flow = self._trade_flow.get(ticker, 0.0)
        decision_context = build_decision_context(
            signal,
            family=family,
            source=source,
            profile_name=profile.name,
            now_ts=now,
            orderbook=ob,
            metadata=meta,
            spot_price=float(self._truth_price_by_asset.get(getattr(meta, "asset", ""), 0.0) if meta else 0.0),
            near_money_pct=float(getattr(self._cfg, "near_money_pct", 0.08)),
            strike_distance_bucket_edges=list(getattr(self._cfg, "strike_distance_bucket_edges", [0.005, 0.01, 0.02, 0.05])),
            drift=_drift,
            flow=_flow,
        )

        await self._bus.publish(
            "kalshi.fills",
            FillEvent(
                market_ticker=ticker,
                order_id=order_id,
                side=signal.side,
                price_cents=fill_price,
                count=fill_centicx,
                is_taker=True,
                timestamp=now,
                fee_usd=fee_usd,
                source=source,
                action=signal.action,
                bot_id=self._cfg.bot_id,
                family=family,
                scenario_profile=profile.name,
                decision_context=decision_context,
            ),
        )

        await self._bus.publish(
            "kalshi.user_orders",
            OrderUpdate(
                market_ticker=ticker,
                order_id=order_id,
                status="filled" if fill_qty == signal.quantity_contracts else "partial_fill",
                side=signal.side,
                price_cents=fill_price,
                quantity_contracts=fill_qty,
                filled_contracts=fill_qty,
                remaining_contracts=max(0, signal.quantity_contracts - fill_qty),
                timestamp=now,
                bot_id=self._cfg.bot_id,
            ),
        )

        record = {
            "type": "paper_fill",
            "order_id": order_id,
            "market_ticker": ticker,
            "side": signal.side,
            "action": signal.action,
            "limit_price_cents": signal.limit_price_cents,
            "fill_price_cents": fill_price,
            "slippage_cents": fill_price - signal.limit_price_cents,
            "quantity_contracts": fill_qty,
            "requested_quantity_contracts": signal.quantity_contracts,
            "edge": signal.edge,
            "p_yes": signal.p_yes,
            "order_style": signal.order_style,
            "source": source,
            "paper_latency_ms": round(latency_s * 1000.0, 1),
            "estimated_taker_fee_usd": fee_usd,
            "spread_drag_usd": round(spread_drag_usd, 6),
            "scenario_profile": profile.name,
            "family": family,
            "timestamp": now,
            "bot_id": self._cfg.bot_id,
            "decision_context": decision_context,
        }
        meta = self._metadata.get(ticker)
        if meta and signal.action == "buy":
            try:
                settle_dt = datetime.fromisoformat(meta.settlement_time_iso.replace("Z", "+00:00"))
                record["settlement_time"] = settle_dt.timestamp()
                record["strike"] = meta.strike_price
                record["asset"] = getattr(meta, "asset", "BTC") or "BTC"
            except Exception:
                pass
        argus_price = int(signal.p_yes * 100) if signal.side == "yes" else int((1.0 - signal.p_yes) * 100)
        log.debug(
            f"PAPER FILL: {ticker} {signal.side.upper()} | "
            f"Argus {argus_price}¢ vs Kalshi {fill_price}¢ | "
            f"x{fill_qty}/{signal.quantity_contracts} edge={signal.edge:+.4f}",
            data={"order_id": order_id, "source": source, "latency_ms": record["paper_latency_ms"], "fee_usd": fee_usd},
        )
        self._diag_counts["paper_fills"] = self._diag_counts.get("paper_fills", 0) + 1
        self._diag_counts["paper_requested_contracts"] = self._diag_counts.get("paper_requested_contracts", 0) + int(signal.quantity_contracts)
        self._diag_counts["paper_filled_contracts"] = self._diag_counts.get("paper_filled_contracts", 0) + int(fill_qty)
        self._diag_gauges["paper_fill_latency_ms_total"] = self._diag_gauges.get("paper_fill_latency_ms_total", 0.0) + observed_latency_ms
        net_edge_cents = (signal.edge * 100.0) - ((fee_usd + spread_drag_usd) * 100.0 / max(1, fill_qty))
        self._diag_gauges["realized_edge_net_cents_sum"] = self._diag_gauges.get("realized_edge_net_cents_sum", 0.0) + net_edge_cents
        if fill_qty < signal.quantity_contracts:
            self._diag_counts["paper_partial_fills"] = self._diag_counts.get("paper_partial_fills", 0) + 1
        _append_paper_log(record)
        self._resolve_pending(order_id, "filled")

    # -- reconciliation from WS ----------------------------------------------

    async def _consume_order_updates(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                update: OrderUpdate = await q.get()
                await self.handle_order_update(update)
        except asyncio.CancelledError:
            pass

    async def handle_order_update(self, update: OrderUpdate) -> None:
        if getattr(update, "bot_id", "default") != self._cfg.bot_id:
            return
        oid = update.order_id
        if oid in self._pending:
            if update.status in ("filled", "cancelled", "canceled"):
                self._resolve_pending(oid, update.status)
            log.debug(
                f"Order update via WS: {update.status} {oid}",
                data={
                    "filled": update.filled_contracts,
                    "remaining": update.remaining_contracts,
                },
            )

    async def _consume_fills(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                fill: FillEvent = await q.get()
                if getattr(fill, "bot_id", "default") != self._cfg.bot_id:
                    continue
                ticker = fill.market_ticker
                # Update position tracking using GROSS (absolute) exposure.
                # NET tracking (old) allowed side-flipping to cancel out positions,
                # resetting the cap to zero — enabling unlimited accumulation.
                # GROSS tracking treats each buy as adding exposure regardless of side,
                # and each sell as reducing it, so the cap holds correctly.
                if fill.action == "buy":
                    self._positions[ticker] = self._positions.get(ticker, 0) + fill.count
                else:
                    self._positions[ticker] = max(0, self._positions.get(ticker, 0) - fill.count)

                log.debug(
                    f"Fill ({fill.action}): {fill.side} {ticker} × {fill.count} @ {fill.price_cents}¢",
                    data={"order_id": fill.order_id, "action": fill.action},
                )
        except asyncio.CancelledError:
            pass

    async def handle_fill(self, fill: FillEvent) -> None:
        if getattr(fill, "bot_id", "default") != self._cfg.bot_id:
            return
        self._diag_counts["fills_seen"] = self._diag_counts.get("fills_seen", 0) + 1
        ticker = fill.market_ticker
        if fill.action == "buy":
            self._positions[ticker] = self._positions.get(ticker, 0) + fill.count
        else:
            self._positions[ticker] = max(0, self._positions.get(ticker, 0) - fill.count)

        log.debug(
            "Fill (%s): %s %s x %s @ %sc",
            fill.action,
            fill.side,
            ticker,
            fill.count,
            fill.price_cents,
            data={"order_id": fill.order_id, "action": fill.action},
        )

    async def _consume_risk(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                event: RiskEvent = await q.get()
                if event.event_type in (
                    "drawdown_breach",
                    "disconnect_halt",
                    "orderbook_invalid",
                    "auth_failure",
                    "truth_feed_stale",
                ):
                    # disconnect_halt/orderbook_invalid/truth_feed_stale: one event per bot, so
                    # with 468 farm bots we get 468× log lines — only log at DEBUG to avoid spam.
                    if not self._halted or self._halt_reason != event.event_type:
                        log.debug(f"Risk halt: {event.event_type} — {event.detail}")
                    self._halted = True
                    self._halt_reason = event.event_type
                    await self._cancel_all_pending()
                elif event.event_type == "truth_feed_resumed":
                    if self._halted and self._halt_reason == "truth_feed_stale":
                        log.debug("ExecutionEngine: truth feed resumed — clearing stale halt")
                        self._halted = False
                        self._halt_reason = ""
        except asyncio.CancelledError:
            pass

    async def handle_risk(self, event: RiskEvent) -> None:
        if event.event_type in (
            "drawdown_breach",
            "disconnect_halt",
            "orderbook_invalid",
            "auth_failure",
            "truth_feed_stale",
        ):
            if not self._halted or self._halt_reason != event.event_type:
                log.debug("Risk halt: %s - %s", event.event_type, event.detail)
            self._halted = True
            self._halt_reason = event.event_type
            await self._cancel_all_pending()
        elif event.event_type == "truth_feed_resumed":
            if self._halted and self._halt_reason == "truth_feed_stale":
                log.debug("ExecutionEngine: truth feed resumed - clearing stale halt")
                self._halted = False
                self._halt_reason = ""

    async def _consume_balance(self, q: asyncio.Queue) -> None:
        try:
            from .models import AccountBalance
            while self._running:
                msg: AccountBalance = await q.get()
                if not self._cfg.use_live_balance and not self._cfg.dry_run:
                    continue
                
                prev_balance = self._current_balance
                self._current_balance = msg.balance_usd
                
                # If this is the first balance of the day, lock it as the start balance.
                # In paper mode, we might want to reset this at midnight; for now
                # we just lock the first one we see in the session.
                if self._day_start_balance == self._cfg.bankroll_usd or self._day_start_balance == 0:
                     self._day_start_balance = self._current_balance
                     log.info(f"Day start balance locked: ${self._day_start_balance:,.2f}")

                log.debug(f"Balance update [{self._cfg.bot_id}]: ${self._current_balance:,.2f} (Δ ${self._current_balance - prev_balance:+.2f})")
        except asyncio.CancelledError:
            pass

    async def handle_balance(self, msg: Any) -> None:
        if not self._cfg.use_live_balance and not self._cfg.dry_run:
            return

        prev_balance = self._current_balance
        self._current_balance = msg.balance_usd
        if self._day_start_balance == self._cfg.bankroll_usd or self._day_start_balance == 0:
            self._day_start_balance = self._current_balance
            log.info(f"Day start balance locked: ${self._day_start_balance:,.2f}")

        log.debug(
            f"Balance update [{self._cfg.bot_id}]: ${self._current_balance:,.2f} "
            f"(delta ${self._current_balance - prev_balance:+.2f})"
        )

    async def _consume_metadata(self, q: asyncio.Queue) -> None:
        """Cache market metadata so paper_fill log can include settlement_time, strike, asset."""
        try:
            while self._running:
                msg = await q.get()
                if isinstance(msg, MarketMetadata):
                    self._metadata[msg.market_ticker] = msg
        except asyncio.CancelledError:
            pass

    def handle_metadata(self, msg: Any) -> None:
        if isinstance(msg, MarketMetadata):
            self._metadata[msg.market_ticker] = msg

    async def _check_consecutive_drawdown_days(self) -> None:
        """Query DB for recent PnL and halt if consecutive losing days limit is hit."""
        if not self._db:
            return

        try:
            # Direct DB call instead of bus request for simplicity in this module.
            history = await self._db.get_kalshi_daily_pnl(days=self._cfg.consecutive_drawdown_limit + 1, bot_id=self._cfg.bot_id)
            if not history:
                return

            losing_streak = 0
            for day in history:
                if day["pnl"] < 0:
                    losing_streak += 1
                else:
                    break  # streak broken

            if losing_streak >= self._cfg.consecutive_drawdown_limit:
                self._halted = True
                self._halt_reason = "consecutive_drawdown"
                log.warning(f"Consecutive drawdown limit hit ({losing_streak} days) — HALTING")
                await self._bus.publish(
                    "kalshi.terminal_event",
                    TerminalEvent("ALERT", f"CONSECUTIVE DRAWDOWN LIMIT HIT ({losing_streak} DAYS) — HALTING", time.time())
                )
        except Exception as e:
            log.error(f"Failed to check consecutive drawdowns: {e}")

    async def _consume_ws_status(self, q: asyncio.Queue) -> None:
        """Auto-resume execution engine when WS reconnects after a disconnect halt."""
        try:
            while self._running:
                event = await q.get()
                if event.status == "connected" and self._halted and self._halt_reason == "disconnect_halt":
                    log.debug("ExecutionEngine: WS reconnected — resuming from disconnect halt")
                    self._halted = False
                    self._halt_reason = ""
        except asyncio.CancelledError:
            pass

    def handle_ws_status(self, event: Any) -> None:
        if event.status == "connected" and self._halted and self._halt_reason == "disconnect_halt":
            log.debug("ExecutionEngine: WS reconnected - resuming from disconnect halt")
            self._halted = False
            self._halt_reason = ""

    async def _cancel_all_pending(self) -> None:
        """Best-effort cancel of all in-flight orders.

        Cancels are sent sequentially with a small stagger to stay
        within the write rate limit.  The rate limiter in the REST
        client provides the primary guard, but the stagger avoids
        bursting all cancels simultaneously.
        """
        items = list(self._pending.items())
        for i, (oid, pending) in enumerate(items):
            if not self._cfg.dry_run:
                try:
                    await self._rest.cancel_order(oid)
                    log.info(f"Emergency cancel: {oid}")
                except Exception as exc:
                    log.error(f"Emergency cancel failed for {oid}: {exc}")
            else:
                log.info(f"Emergency cancel: {oid}")
            if pending.timeout_task:
                pending.timeout_task.cancel()
            if pending.fill_task:
                pending.fill_task.cancel()
            # Stagger between cancels (skip after last).
            if i < len(items) - 1:
                await asyncio.sleep(_CANCEL_STAGGER_S)

        # Clear all tracking.
        self._pending.clear()
        self._pending_by_market.clear()
        self._pending_by_market_side.clear()

    # -- external interface --------------------------------------------------

    def resume(self) -> None:
        """Resume after a risk halt (manual intervention)."""
        self._halted = False
        log.info("Execution resumed after halt")

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def has_pending_for(self, market_ticker: str) -> bool:
        """Return True if there is an outstanding order for this market."""
        return any(t == market_ticker for (t, _s) in self._pending_by_market_side.keys())

    def drain_diagnostics(self) -> Dict[str, Any]:
        data = {
            "counts": dict(self._diag_counts),
            "sources": dict(self._diag_source_counts),
            "families": dict(self._diag_family_counts),
            "source_families": dict(self._diag_source_family_counts),
            "gauges": dict(self._diag_gauges),
            "pending": len(self._pending),
            "pending_markets": len(self._pending_by_market),
            "pending_market_sides": len(self._pending_by_market_side),
            "halted": self._halted,
            "halt_reason": self._halt_reason or None,
        }
        self._diag_counts.clear()
        self._diag_source_counts.clear()
        self._diag_family_counts.clear()
        self._diag_source_family_counts.clear()
        self._diag_gauges.clear()
        return data
