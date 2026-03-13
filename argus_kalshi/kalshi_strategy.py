"""
Trade signal generation for Kalshi BTC strike contracts.

Strategy logic
--------------
1. For each market, compare model fair probability ``p_yes`` to the
   orderbook-implied prices.
2. Compute expected value (EV) for both sides:
       EV_yes = p_yes - yes_ask_prob
       EV_no  = (1 - p_yes) - no_ask_prob
3. If either EV exceeds ``min_edge_threshold`` **and** risk limits allow,
   publish a ``TradeSignal``.

Risk limits / kill switches
---------------------------
* **Per-market exposure**: capped at ``max_fraction_per_market`` of bankroll.
* **Daily drawdown**: halts all trading for the rest of the day.
* **WS disconnect**: halts until reconnected.
* **Orderbook invalid**: halts trading for that market until a fresh
  snapshot re-validates the book.
* **Truth feed stale**: halts if no BTC tick arrives within
  ``truth_feed_stale_timeout_s``.
* **ws_trading_enabled=False**: signals are computed and logged but never
  published (pre-production verification mode).
* **dry_run=True**: same as above — compute and log only.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .bus import Bus
from .config import KalshiConfig
from .logging_utils import ComponentLogger
from .market_selectors import hold_entry_horizon_seconds
from .shared_state import SharedFarmState
from .models import (
    AccountBalance,
    ActiveTicker,
    BtcMidPrice,
    FairProbability,
    OrderbookState,
    RiskEvent,
    StrategyDecision,
    TerminalEvent,
    TradeSignal,
)

log = ComponentLogger("strategy")


def _session_mult(now_wall: float, sleeve: str = "hold") -> float:
    """UTC-hour-based sizing multiplier derived from top-wallet activity patterns.

    0xd0d6 concentrates scalp in 15-21 UTC; 0x1979 concentrates holds in 11-15 UTC.
    Overnight (00-09 UTC) reduces sizing to filter noise from thin books.
    sleeve: "scalp" | "hold"
    """
    utc_hour = datetime.fromtimestamp(now_wall, tz=timezone.utc).hour
    if 15 <= utc_hour < 21:
        return 1.20
    if 11 <= utc_hour < 15:
        return 1.10 if sleeve == "hold" else 1.05
    if 0 <= utc_hour < 9:
        return 0.70
    return 1.00


def _hold_tail_penalty(limit_cents: int, side: str, start_cents: int, per_10c: float) -> float:
    if start_cents <= 0 or per_10c <= 0.0 or limit_cents <= start_cents:
        return 0.0
    penalty = ((limit_cents - start_cents) / 10.0) * per_10c
    if side == "no":
        penalty *= 1.25
    return penalty


def _hold_family_allowed(asset: str, window_minutes: int, is_range: bool) -> bool:
    """Keep hold exploration in the families that still show signal quality.

    Current paper results show SOL is broadly toxic, hourly hold contracts are
    poor across assets, and ETH range holds are particularly weak.  We keep the
    more promising BTC/ETH 15m region and BTC range exploration active.
    """
    if asset == "SOL":
        return False
    if is_range and asset != "BTC":
        return False
    if asset == "BTC":
        return window_minutes in (15, 60) or is_range
    if asset == "ETH":
        return window_minutes in (15, 60) and not is_range
    return True


class StrategyEngine:
    """Subscribes to fair-probability and orderbook, emits TradeSignals.

    Farm mode
    ---------
    Pass ``shared=<SharedFarmState>`` to run in farm mode.  In farm mode
    the engine does NOT subscribe to high-frequency bus topics (orderbook,
    fair_prob, truth prices, market_metadata).  Instead it shares state
    dicts with all other farm bots via ``SharedFarmState``, and the single
    ``FarmDispatcher`` calls ``evaluate_sync()`` on every OB/prob update.
    Only low-frequency topics (fills, ws.status, account_balance) retain
    per-bot subscriptions because they carry bot-specific data.
    """

    def __init__(
        self,
        config: KalshiConfig,
        bus: Bus,
        running_bankroll: float = 0.0,
        shared: Optional[SharedFarmState] = None,
    ) -> None:
        self._cfg = config
        self._bus = bus
        self._farm_mode: bool = shared is not None

        # High-frequency market state.
        # In farm mode these point to shared dicts (same object for every bot).
        if shared is not None:
            self._fair_probs = shared.fair_probs
            self._orderbooks = shared.orderbooks
            self._market_asset = shared.market_asset
            self._market_settlement = shared.market_settlement
            self._market_window_min = shared.market_window_min
            self._market_is_range = shared.market_is_range
            self._last_truth_tick_time_by_asset = shared.last_truth_tick_by_asset
            self._trade_flow: Dict[str, float] = shared.trade_flow_by_ticker
            self._delta_flow_yes: Dict[str, float] = shared.orderbook_delta_flow_yes
            self._delta_flow_no: Dict[str, float] = shared.orderbook_delta_flow_no
        else:
            self._fair_probs: Dict[str, FairProbability] = {}
            self._orderbooks: Dict[str, OrderbookState] = {}
            self._market_asset: Dict[str, str] = {}
            self._market_settlement: Dict[str, float] = {}
            self._market_window_min: Dict[str, int] = {}
            self._market_is_range: Dict[str, bool] = {}
            self._last_truth_tick_time_by_asset: Dict[str, float] = {}
            self._trade_flow: Dict[str, float] = {}
            self._delta_flow_yes: Dict[str, float] = {}
            self._delta_flow_no: Dict[str, float] = {}
        self._trade_deques: Dict[str, deque] = {}
        self._trade_flow_window_s: float = 60.0
        self._delta_yes_deques: Dict[str, deque] = {}
        self._delta_no_deques: Dict[str, deque] = {}
        self._delta_flow_window_s: float = 30.0

        # Risk tracking — always per-bot.
        self._positions: Dict[str, int] = {}
        self._hold_position_qty: Dict[str, int] = {}
        self._hold_position_side: Dict[str, str] = {}
        self._daily_pnl: float = 0.0
        self._halted: bool = False
        self._halt_reason: str = ""
        self._current_balance: float = running_bankroll if running_bankroll > 0 else config.bankroll_usd
        self._start_bankroll: float = self._current_balance

        # Truth feed staleness — in farm mode the dispatcher manages this
        # externally; set to False so the sync evaluator doesn't self-halt.
        self._last_truth_tick_time: float = 0.0
        self._truth_stale: bool = (not self._farm_mode)

        # Persistence filter state: ticker → (side, first_seen_monotonic).
        self._persistence_state: Dict[str, Tuple[str, float]] = {}

        # Signal cooldown: ticker → (side, emit_time).
        self._last_signal_time: Dict[str, Tuple[str, float]] = {}

        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._eval_counts: Dict[str, int] = {}
        self._pass_reasons: Dict[str, int] = {}
        self._last_eval_mono: Dict[str, float] = {}

    # -- lifecycle -----------------------------------------------------------

    async def start(self, market_tickers: list[str]) -> None:
        self._running = True

        if self._farm_mode:
            # Farm mode: shared event routing is handled centrally.
            return

        # Non-farm (single-bot) mode: full per-bot subscriptions.
        q_prob = await self._bus.subscribe("kalshi.fair_prob")
        self._tasks.append(asyncio.create_task(self._consume_probs(q_prob)))

        for ticker in market_tickers:
            q_ob = await self._bus.subscribe(f"kalshi.orderbook.{ticker}")
            self._tasks.append(
                asyncio.create_task(self._consume_orderbook(q_ob, ticker))
            )

        q_fills = await self._bus.subscribe("kalshi.fills")
        self._tasks.append(asyncio.create_task(self._consume_fills(q_fills)))

        q_ws = await self._bus.subscribe("kalshi.ws.status")
        self._tasks.append(asyncio.create_task(self._consume_ws_status(q_ws)))

        q_balance = await self._bus.subscribe("kalshi.account_balance")
        self._tasks.append(asyncio.create_task(self._consume_balance(q_balance)))

        q_truth_btc = await self._bus.subscribe("btc.mid_price")
        q_truth_eth = await self._bus.subscribe("eth.mid_price")
        q_truth_sol = await self._bus.subscribe("sol.mid_price")
        q_meta = await self._bus.subscribe("kalshi.market_metadata")
        q_trade = await self._bus.subscribe("kalshi.trade")
        q_delta = await self._bus.subscribe("kalshi.orderbook_delta_flow")
        self._tasks.append(asyncio.create_task(self._consume_truth(q_truth_btc, "BTC")))
        self._tasks.append(asyncio.create_task(self._consume_truth(q_truth_eth, "ETH")))
        self._tasks.append(asyncio.create_task(self._consume_truth(q_truth_sol, "SOL")))
        self._tasks.append(asyncio.create_task(self._consume_metadata(q_meta)))
        self._tasks.append(asyncio.create_task(self._consume_trades(q_trade)))
        self._tasks.append(asyncio.create_task(self._consume_orderbook_delta_flow(q_delta)))

        # Periodic staleness checker.
        self._tasks.append(asyncio.create_task(self._truth_staleness_watchdog()))
        self._tasks.append(asyncio.create_task(self._strategy_diagnostics()))

    async def stop(self) -> None:
        self._running = False
        self._prune_tasks()
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
                log.debug("Strategy stop: some consumer tasks did not exit in time")

    # -- consumers -----------------------------------------------------------

    async def _consume_probs(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                msg: FairProbability = await q.get()
                self._fair_probs[msg.market_ticker] = msg
                await self._evaluate(msg.market_ticker)
        except asyncio.CancelledError:
            pass

    def handle_fill(self, fill: object) -> None:
        if getattr(fill, "bot_id", "default") != self._cfg.bot_id:
            return
        self._apply_fill_position_update(fill)

    async def _consume_orderbook(self, q: asyncio.Queue, ticker: str) -> None:
        try:
            while self._running:
                msg: OrderbookState = await q.get()
                self._orderbooks[ticker] = msg
                # Orderbook invalid: trading is skipped for this market until a valid snapshot.
                # Log at DEBUG only — with 468 farm bots this would spam at WARNING.
                if not msg.valid:
                    log.debug(
                        "Orderbook invalid for %s — trading halted for this market",
                        ticker,
                    )
                await self._evaluate(ticker)
        except asyncio.CancelledError:
            pass

    def handle_balance(self, msg: AccountBalance) -> None:
        self._current_balance = msg.balance_usd
        log.debug(f"Updated current balance: ${self._current_balance:.2f}")

    async def _consume_fills(self, q: asyncio.Queue) -> None:
        """Track fills for P&L / position tracking."""
        try:
            while self._running:
                fill = await q.get()
                self._apply_fill_position_update(fill)
        except asyncio.CancelledError:
            pass

    def _apply_fill_position_update(self, fill: object) -> None:
        ticker = str(getattr(fill, "market_ticker", "") or "")
        if not ticker:
            return
        fill_count = int(getattr(fill, "count", 0) or 0)
        action = str(getattr(fill, "action", "buy") or "buy").lower()
        source = str(getattr(fill, "source", "") or "")
        side = str(getattr(fill, "side", "") or "").lower()

        if action == "buy":
            self._positions[ticker] = self._positions.get(ticker, 0) + fill_count
        else:
            self._positions[ticker] = max(0, self._positions.get(ticker, 0) - fill_count)

        # Track hold sleeve positions separately for reversal-triggered exits.
        if source == "mispricing_hold":
            if action == "buy":
                self._hold_position_qty[ticker] = self._hold_position_qty.get(ticker, 0) + fill_count
                if side in ("yes", "no"):
                    self._hold_position_side[ticker] = side
            else:
                remaining = max(0, self._hold_position_qty.get(ticker, 0) - fill_count)
                if remaining <= 0:
                    self._hold_position_qty.pop(ticker, None)
                    self._hold_position_side.pop(ticker, None)
                else:
                    self._hold_position_qty[ticker] = remaining

    async def _consume_trades(self, q: asyncio.Queue) -> None:
        """Maintain per-ticker trade-flow imbalance in non-farm mode."""
        try:
            while self._running:
                ev = await q.get()
                ticker = str(getattr(ev, "market_ticker", "") or "")
                taker_side = str(getattr(ev, "taker_side", "") or "").lower()
                count = int(getattr(ev, "count", 0) or 0)
                ts = float(getattr(ev, "ts", 0.0) or 0.0) or time.time()
                if not ticker or taker_side not in ("yes", "no") or count <= 0:
                    continue
                dq = self._trade_deques.get(ticker)
                if dq is None:
                    dq = deque(maxlen=500)
                    self._trade_deques[ticker] = dq
                dq.append((ts, taker_side, count))
                cutoff = time.time() - self._trade_flow_window_s
                while dq and dq[0][0] < cutoff:
                    dq.popleft()
                yes_vol = sum(c for t, s, c in dq if t >= cutoff and s == "yes")
                no_vol = sum(c for t, s, c in dq if t >= cutoff and s == "no")
                total = yes_vol + no_vol
                self._trade_flow[ticker] = ((yes_vol - no_vol) / total) if total > 0 else 0.0
        except asyncio.CancelledError:
            pass

    async def _consume_orderbook_delta_flow(self, q: asyncio.Queue) -> None:
        """Maintain per-ticker YES/NO add-vs-cancel flow in non-farm mode."""
        try:
            while self._running:
                ev = await q.get()
                ticker = str(getattr(ev, "market_ticker", "") or "")
                side = str(getattr(ev, "side", "") or "").lower()
                qty = int(getattr(ev, "qty", 0) or 0)
                ts = float(getattr(ev, "ts", 0.0) or 0.0) or time.time()
                is_add = bool(getattr(ev, "is_add", False))
                if not ticker or side not in ("yes", "no") or qty <= 0:
                    continue
                target = self._delta_yes_deques if side == "yes" else self._delta_no_deques
                dq = target.get(ticker)
                if dq is None:
                    dq = deque(maxlen=2000)
                    target[ticker] = dq
                dq.append((ts, is_add, qty))
                cutoff = time.time() - self._delta_flow_window_s
                while dq and dq[0][0] < cutoff:
                    dq.popleft()
                add_qty = sum(v for t, add, v in dq if t >= cutoff and add)
                cancel_qty = sum(v for t, add, v in dq if t >= cutoff and not add)
                total = add_qty + cancel_qty
                flow = ((add_qty - cancel_qty) / total) if total > 0 else 0.0
                if side == "yes":
                    self._delta_flow_yes[ticker] = flow
                else:
                    self._delta_flow_no[ticker] = flow
        except asyncio.CancelledError:
            pass

    async def _consume_ws_status(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                event = await q.get()
                if event.status == "disconnected":
                    # WS publishes one RiskEvent(disconnect_halt) per disconnect; we just halt.
                    log.debug("WS disconnected — halting strategy")
                    self._halted = True
                    self._halt_reason = "ws_disconnected"
                elif event.status == "connected":
                    # Only resume if the halt was due to WS disconnect.
                    # Other halts (drawdown, etc.) persist.
                    if self._halt_reason == "ws_disconnected":
                        log.debug("WS reconnected — resuming strategy (pending valid orderbooks)")
                        self._halted = False
                        self._halt_reason = ""
        except asyncio.CancelledError:
            pass

    async def _consume_balance(self, q: asyncio.Queue) -> None:
        """Update current balance from account balance messages."""
        try:
            while self._running:
                msg: AccountBalance = await q.get()
                self._current_balance = msg.balance_usd
                log.debug(f"Updated current balance: ${self._current_balance:.2f}")
        except asyncio.CancelledError:
            pass

    async def _consume_metadata(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                msg = await q.get()
                self._prune_tasks()
                ticker = msg.market_ticker
                self._market_asset[ticker] = getattr(msg, "asset", "BTC")
                self._market_window_min[ticker] = getattr(msg, "window_minutes", 15)
                self._market_is_range[ticker] = getattr(msg, "is_range", False)
                # Signal cooldown starts fresh automatically for new tickers because
                # Kalshi uses unique ticker strings per contract window (e.g.
                # KXBTCM-26FEB25-65000). A new ticker never has an entry in
                # _last_signal_time, so no explicit reset is needed here.
                # We do NOT pop on metadata refresh (every ~15s) for known tickers
                # to avoid defeating the configured cooldown window.
                try:
                    settle_dt = datetime.fromisoformat(
                        msg.settlement_time_iso.replace("Z", "+00:00")
                    )
                    self._market_settlement[ticker] = settle_dt.timestamp()
                except Exception:
                    pass
        except asyncio.CancelledError:
            pass

    async def _consume_truth(self, q: asyncio.Queue, asset: str) -> None:
        """Track truth feed liveness."""
        try:
            while self._running:
                msg = await q.get()
                now = time.monotonic()
                self._last_truth_tick_time = now
                self._last_truth_tick_time_by_asset[asset] = now
                # Only clear staleness and publish truth_feed_resumed when ALL watched
                # assets are fresh. Otherwise a BTC tick would clear a halt caused by
                # ETH staleness, then watchdog re-fires — causing spam oscillation.
                watched = set(self._market_asset.values()) or {"BTC"}
                still_stale = [
                    a for a in watched
                    if (now - self._last_truth_tick_time_by_asset.get(a, 0.0)) > self._cfg.truth_feed_stale_timeout_s
                ]
                if still_stale:
                    continue
                if self._truth_stale:
                    self._truth_stale = False
                    log.debug("Truth feed active — staleness cleared")
                    if self._halt_reason == "truth_stale":
                        self._halted = False
                        self._halt_reason = ""
                        await self._bus.publish(
                            "kalshi.risk",
                            RiskEvent(
                                event_type="truth_feed_resumed",
                                detail="Truth feed active again",
                                timestamp=time.time(),
                            ),
                        )
        except asyncio.CancelledError:
            pass

    async def _truth_staleness_watchdog(self) -> None:
        """Periodically check if truth feed has gone stale."""
        try:
            while self._running:
                await asyncio.sleep(5.0)
                if self._last_truth_tick_time == 0.0:
                    continue  # never received a tick yet
                now_mono = time.monotonic()
                watched_assets = set(self._market_asset.values()) or {"BTC"}
                stale_assets = [a for a in watched_assets if (now_mono - self._last_truth_tick_time_by_asset.get(a, 0.0)) > self._cfg.truth_feed_stale_timeout_s]
                if stale_assets and not self._truth_stale:
                    self._truth_stale = True
                    self._halted = True
                    self._halt_reason = "truth_stale"
                    # DEBUG only: with 468 farm bots, each watchdog fires → 468× spam.
                    log.debug(
                        f"Truth feed stale for assets {stale_assets} — halting",
                        data={"timeout": self._cfg.truth_feed_stale_timeout_s},
                    )
                    await self._bus.publish(
                        "kalshi.risk",
                        RiskEvent(
                            event_type="truth_feed_stale",
                            detail=f"No tick within timeout for assets {stale_assets}",
                            timestamp=time.time(),
                        ),
                    )
        except asyncio.CancelledError:
            pass

    # -- evaluation ----------------------------------------------------------

    async def _evaluate(self, ticker: str) -> None:
        """Check if a trade signal should be emitted for *ticker*."""
        if self._halted:
            return

        self._eval_counts[ticker] = self._eval_counts.get(ticker, 0) + 1
        self._last_eval_mono[ticker] = time.monotonic()
        fp = self._fair_probs.get(ticker)
        ob = self._orderbooks.get(ticker)
        p_yes = fp.p_yes if fp else math.nan
            
        async def _log_decision(action: str, reason: str):
            self._pass_reasons[reason] = self._pass_reasons.get(reason, 0) + 1
            if action == "pass":
                return
            decision = StrategyDecision(
                market_ticker=ticker,
                p_yes=p_yes,
                yes_ask=ob.implied_yes_ask_cents if ob else 100,
                no_ask=ob.implied_no_ask_cents if ob else 100,
                action_taken=action,
                reason=reason,
                timestamp=time.time(),
                bot_id=self._cfg.bot_id,
            )
            await self._bus.publish("kalshi.strategy_decision", decision)

        # Refuse to trade if:
        # - no fair probability computed yet
        # - no orderbook state received yet
        # - orderbook is invalid
        if not fp or not ob:
            await _log_decision("pass", "AWAITING_DATA")
            return
        if not ob.valid:
            await _log_decision("pass", "ORDERBOOK_INVALID")
            return
        if math.isnan(p_yes):
            await _log_decision("pass", "NAN_PROBABILITY")
            return

        # Truth feed must be active.
        asset = self._market_asset.get(ticker, "BTC")
        last_asset_tick = self._last_truth_tick_time_by_asset.get(asset, 0.0)
        if self._truth_stale or (last_asset_tick and (time.monotonic() - last_asset_tick) > self._cfg.truth_feed_stale_timeout_s):
            await _log_decision("pass", "truth_stale")
            return

        # OBI-adjusted probability.
        # ob.obi is the Kalshi orderbook imbalance at the best level:
        #   positive → more YES bid depth (market expects price up)
        #   negative → more NO bid depth (market expects price down)
        # Adjusting p_yes by obi * weight makes the edge calculation directionally
        # aware without suppressing trades — it simply shifts the probability
        # estimate toward what the live orderbook is signalling.
        # Weight=0.0 disables the adjustment (default); tune from run data.
        obi_weight = self._cfg.obi_p_yes_bias_weight
        if obi_weight and ob.obi:
            p_yes = max(0.02, min(0.98, p_yes + ob.obi * obi_weight))

        # Momentum-adjusted probability.
        # fp.drift is per-second log-drift from 30s OLS slope of truth-feed prices.
        # Positive drift = uptrend → push p_yes up. Scaled by weight.
        momentum_weight = self._cfg.momentum_p_yes_bias_weight
        if momentum_weight and fp.drift:
            p_yes = max(0.02, min(0.98, p_yes + fp.drift * momentum_weight))

        # Trade flow bias.
        # Positive flow = more YES takers = market leaning YES → push p_yes up.
        flow = float(self._trade_flow.get(ticker, 0.0))
        flow_weight = self._cfg.trade_flow_p_yes_bias_weight
        if flow_weight and flow:
            p_yes = max(0.02, min(0.98, p_yes + flow * flow_weight))

        drift = float(getattr(fp, "drift", 0.0))

        # Early-exit hold when both momentum and flow reverse against position.
        hold_qty_centicx = int(self._hold_position_qty.get(ticker, 0))
        hold_side = self._hold_position_side.get(ticker)
        if hold_qty_centicx > 0 and hold_side in ("yes", "no"):
            min_drift = float(self._cfg.scalp_momentum_min_drift)
            flow_rev = float(self._cfg.hold_flow_reversal_threshold)
            reversal = (
                (hold_side == "yes" and drift < -min_drift and flow < -flow_rev)
                or (hold_side == "no" and drift > min_drift and flow > flow_rev)
            )
            if reversal:
                # Reversal exits are protective sells — cooldown must NOT block them.
                now_mono = time.monotonic()
                limit_cents = ob.best_yes_bid_cents if hold_side == "yes" else ob.best_no_bid_cents
                if limit_cents <= 0:
                    await _log_decision("pass", "hold_reversal_no_bid")
                    return
                self._last_signal_time[ticker] = (hold_side, now_mono)
                signal = TradeSignal(
                    market_ticker=ticker,
                    side=hold_side,
                    action="sell",
                    limit_price_cents=limit_cents,
                    quantity_contracts=max(1, hold_qty_centicx // 100),
                    edge=0.0,
                    p_yes=p_yes,
                    timestamp=time.time(),
                    order_style="aggressive",
                    source="mispricing_hold",
                    bot_id=self._cfg.bot_id,
                )
                await self._bus.publish("kalshi.trade_signal", signal)
                await _log_decision("sell", "hold_signal_reversal")
                return

        # Implied ask probabilities.
        yes_ask_prob = ob.implied_yes_ask_cents / 100.0
        no_ask_prob = ob.implied_no_ask_cents / 100.0

        # Market-implied divergence is the primary hold edge.
        divergence = p_yes - yes_ask_prob
        forced_side: Optional[str] = "yes" if divergence > 0 else "no"
        side_sign = 1.0 if forced_side == "yes" else -1.0
        depth_pressure = float(getattr(ob, "depth_pressure", 0.0))
        yes_delta_flow = float(self._delta_flow_yes.get(ticker, 0.0))
        no_delta_flow = float(self._delta_flow_no.get(ticker, 0.0))
        depth_agree = max(0.0, side_sign * depth_pressure)
        delta_agree_raw = ((yes_delta_flow - no_delta_flow) * side_sign) / 2.0
        delta_agree = max(0.0, delta_agree_raw)
        confirmation = max(0.0, min(1.0, 0.5 * depth_agree + 0.5 * delta_agree))
        divergence_threshold = self._cfg.hold_min_divergence_threshold * (1.0 - 0.2 * confirmation)
        if abs(divergence) < divergence_threshold:
            self._persistence_state.pop(ticker, None)
            await _log_decision("pass", "no_divergence")
            return

        if self._cfg.hold_require_momentum_agreement:
            min_mag = float(self._cfg.hold_momentum_agreement_min_drift)
            if abs(drift) < min_mag:
                # Below noise floor — no clear directional signal, can't confirm agreement.
                self._persistence_state.pop(ticker, None)
                await _log_decision("pass", "hold_momentum_below_min")
                return
            if drift != 0.0:
                if (forced_side == "yes" and drift < 0.0) or (forced_side == "no" and drift > 0.0):
                    self._persistence_state.pop(ticker, None)
                    await _log_decision("pass", "hold_momentum_disagree")
                    return
        if self._cfg.hold_require_flow_agreement:
            min_mag = float(self._cfg.hold_flow_agreement_min_flow)
            if abs(flow) < min_mag:
                self._persistence_state.pop(ticker, None)
                await _log_decision("pass", "hold_flow_below_min")
                return
            if flow != 0.0:
                if (forced_side == "yes" and flow < 0.0) or (forced_side == "no" and flow > 0.0):
                    self._persistence_state.pop(ticker, None)
                    await _log_decision("pass", "hold_flow_disagree")
                    return

        ev_yes = divergence
        ev_no = -divergence

        # Time-to-expiry awareness.
        settle_ts = self._market_settlement.get(ticker, 0.0)
        time_to_settle_s = settle_ts - time.time() if settle_ts > 0 else 9999.0
        near_expiry_s = self._cfg.near_expiry_minutes * 60
        near_expiry = 0 < time_to_settle_s < near_expiry_s

        is_range = self._market_is_range.get(ticker, False)
        window_minutes = self._market_window_min.get(ticker, 15)
        if not _hold_family_allowed(asset, window_minutes, is_range):
            await _log_decision("pass", "hold_family_filtered")
            return
        max_entry_s = hold_entry_horizon_seconds(
            window_minutes=window_minutes,
            is_range=is_range,
            max_entry_minutes_to_expiry=self._cfg.max_entry_minutes_to_expiry,
            range_max_entry_minutes_to_expiry=self._cfg.range_max_entry_minutes_to_expiry,
        )
        if not is_range and max_entry_s > 0 and time_to_settle_s > max_entry_s:
            await _log_decision("pass", "too_early_to_expiry")
            return
        # Range markets use the same helper, but keep a distinct reason for diagnostics.
        range_max_s = max_entry_s
        if is_range and range_max_s > 0 and time_to_settle_s > range_max_s:
            await _log_decision("pass", "range_too_far_to_expiry")
            return
        # Minimum time guard: block entries too close to expiry where the
        # Gaussian σ collapses and p_yes becomes hyper-sensitive to tiny
        # price gaps.  For 15m contracts (3-min max window) a 5-min floor
        # effectively disables 15m mispricing_hold entirely.
        min_entry_s = self._cfg.hold_min_entry_minutes_to_expiry * 60
        if min_entry_s > 0 and time_to_settle_s < min_entry_s:
            await _log_decision("pass", "too_close_to_expiry")
            return

        if near_expiry:
            min_edge = self._cfg.near_expiry_min_edge
            persistence_ms_eff = self._cfg.near_expiry_persistence_ms
        else:
            min_edge = self._cfg.min_edge_threshold
            persistence_ms_eff = self._cfg.persistence_window_ms

        # Fee-aware effective edge.
        fee = self._cfg.effective_edge_fee_pct
        effective_yes = ev_yes - fee
        effective_no = ev_no - fee
        penalized_yes = effective_yes - _hold_tail_penalty(
            ob.implied_yes_ask_cents,
            "yes",
            self._cfg.hold_tail_penalty_start_cents,
            self._cfg.hold_tail_penalty_per_10c,
        )
        penalized_no = effective_no - _hold_tail_penalty(
            ob.implied_no_ask_cents,
            "no",
            self._cfg.hold_tail_penalty_start_cents,
            self._cfg.hold_tail_penalty_per_10c,
        )

        side: Optional[str] = None
        edge: float = 0.0
        limit_cents: int = 0

        if forced_side == "yes":
            if penalized_yes >= min_edge:
                side = "yes"
                edge = penalized_yes
                limit_cents = ob.implied_yes_ask_cents
        elif forced_side == "no":
            if penalized_no >= min_edge:
                side = "no"
                edge = penalized_no
                limit_cents = ob.implied_no_ask_cents
        else:
            if penalized_yes >= min_edge and penalized_yes >= penalized_no:
                side = "yes"
                edge = penalized_yes
                limit_cents = ob.implied_yes_ask_cents
            elif penalized_no >= min_edge:
                side = "no"
                edge = penalized_no
                limit_cents = ob.implied_no_ask_cents

        if side is None:
            self._persistence_state.pop(ticker, None)
            await _log_decision("pass", "no_edge")
            return

        # Price limits.
        mn = self._cfg.min_entry_cents
        mx = self._cfg.max_entry_cents
        if (mn > 0 and limit_cents < mn) or (mx < 100 and limit_cents > mx):
            self._persistence_state.pop(ticker, None)
            await _log_decision("pass", "entry_price_out_of_range")
            return
        if (
            side == "no"
            and self._cfg.no_avoid_above_cents > 0
            and limit_cents >= self._cfg.no_avoid_above_cents
        ):
            self._persistence_state.pop(ticker, None)
            await _log_decision("pass", "no_tail_filter")
            return

        # YES mid-range filter.
        if side == "yes":
            y_min = self._cfg.yes_avoid_min_cents
            y_max = self._cfg.yes_avoid_max_cents
            if y_min > 0 and y_max > 0 and y_min <= limit_cents <= y_max:
                self._persistence_state.pop(ticker, None)
                await _log_decision("pass", "yes_mid_range_filter")
                return

        # Persistence filter.
        now_mono = time.monotonic()
        if persistence_ms_eff > 0:
            prev = self._persistence_state.get(ticker)
            if prev is None or prev[0] != side:
                self._persistence_state[ticker] = (side, now_mono)
                await _log_decision("pass", "persistence")
                return
            elapsed_ms = (now_mono - prev[1]) * 1000
            if elapsed_ms < persistence_ms_eff:
                return

        # Signal cooldown.
        cooldown_s = self._cfg.signal_cooldown_s
        if cooldown_s > 0:
            last = self._last_signal_time.get(ticker)
            if last is not None:
                last_side, last_ts = last
                if last_side == side and (now_mono - last_ts) < cooldown_s:
                    return

        # Latency breaker.
        latency_limit_ms = self._cfg.latency_circuit_breaker_ms
        if latency_limit_ms > 0 and last_asset_tick > 0:
            truth_age_ms = (now_mono - last_asset_tick) * 1000
            if truth_age_ms > latency_limit_ms:
                await _log_decision("pass", "latency_breaker")
                return

        # --- RISK & SIZING ---
        current_position_centicx = abs(self._positions.get(ticker, 0))
        current_position_contracts = current_position_centicx // 100

        # Exposure cap.
        max_usd = self._current_balance * self._cfg.max_fraction_per_market
        max_contracts_limit = int(max_usd / (limit_cents / 100.0)) if limit_cents > 0 else 0

        # Dynamic Sizing: qty = (balance * risk_fraction) / (price / 100)
        risk_usd_per_trade = self._current_balance * self._cfg.risk_fraction_per_trade
        cost_per_contract = max(limit_cents / 100.0, 0.01)

        # Conservative worst-case sizing for binary contracts: max loss is premium paid.
        base_qty = int(risk_usd_per_trade / cost_per_contract) if cost_per_contract > 0 else 0
        base_qty = max(1, base_qty)
        
        # Scale with edge: 2x min_edge = 2x base_qty, etc.
        min_edge = self._cfg.min_edge_threshold
        qty = max(base_qty, int(base_qty * (edge / min_edge)))
        
        # Ticker Cap.
        ticker_cap = self._cfg.max_contracts_per_ticker
        if ticker_cap > 0:
            qty = min(qty, ticker_cap - current_position_contracts)
        
        # Final exposure cap.
        qty = min(qty, max_contracts_limit - current_position_contracts)

        if qty <= 0:
            await _log_decision("pass", "max_position_reached")
            return

        # Daily drawdown check.
        if self._daily_pnl <= -self._start_bankroll * self._cfg.daily_drawdown_limit:
            if not self._halted:
                self._halted = True
                self._halt_reason = "drawdown"
                log.warning("Daily drawdown limit breached — halting")
                await self._bus.publish(
                    "kalshi.terminal_event",
                    TerminalEvent("ALERT", "DAILY DRAWDOWN LIMIT BREACHED — HALTING TRADING", time.time(), bot_id=self._cfg.bot_id)
                )
                await self._bus.publish(
                    "kalshi.risk",
                    RiskEvent(
                        event_type="drawdown_breach",
                        detail=f"pnl={self._daily_pnl:.2f}",
                        timestamp=time.time(),
                    ),
                )
            await _log_decision("pass", "drawdown_breach")
            return

        # Order style.
        order_style = self._cfg.default_order_style
        if order_style == "passive":
            if side == "yes":
                limit_cents = ob.best_yes_bid_cents if ob.best_yes_bid_cents > 0 else limit_cents
            else:
                limit_cents = ob.best_no_bid_cents if ob.best_no_bid_cents > 0 else limit_cents

        signal = TradeSignal(
            market_ticker=ticker,
            side=side,
            action="buy",
            limit_price_cents=limit_cents,
            quantity_contracts=qty,
            edge=round(edge, 4),
            p_yes=round(p_yes, 4),
            timestamp=time.time(),
            order_style=order_style,
            source="mispricing_hold",
            bot_id=self._cfg.bot_id,
        )

        mode_tag = f"GAMMA({time_to_settle_s:.0f}s)" if near_expiry else "NORMAL"
        argus_price = int(p_yes * 100) if side == "yes" else int((1.0 - p_yes) * 100)
        log.debug(
            f"Trade signal [{mode_tag}]: {side.upper()} {ticker} | "
            f"Argus {argus_price}¢ vs Kalshi {limit_cents}¢",
            data={
                "edge": signal.edge,
                "p_yes": signal.p_yes,
                "qty": qty,
                "yes_ask": ob.implied_yes_ask_cents,
                "no_ask": ob.implied_no_ask_cents,
                "tts_s": round(time_to_settle_s, 1),
            },
        )

        if not self._cfg.dry_run and not self._cfg.ws_trading_enabled:
            log.debug(f"ws_trading_enabled=False — signal NOT published for {ticker}")
            await _log_decision(f"buy_{side}", "ws_trading_disabled")
            return

        self._last_signal_time[ticker] = (side, now_mono)
        self._pass_reasons["signal_emitted"] = self._pass_reasons.get("signal_emitted", 0) + 1
        await self._bus.publish("kalshi.trade_signal", signal)
        if self._cfg.dry_run:
            log.debug(f"DRY RUN paper signal published for {ticker}")
            await _log_decision(f"buy_{side}", "paper_trade")
        else:
            await _log_decision(f"buy_{side}", "executed")

    async def _strategy_diagnostics(self, interval_s: float = 30.0) -> None:
        """Periodic strategy health log for stale-signal investigation."""
        try:
            while self._running:
                await asyncio.sleep(interval_s)
                now = time.monotonic()
                eval_total = sum(self._eval_counts.values())
                top_eval = sorted(self._eval_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                stalled = sorted(
                    ((t, now - ts) for t, ts in self._last_eval_mono.items()),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
                truth_age = {
                    asset: round(now - ts, 2)
                    for asset, ts in self._last_truth_tick_time_by_asset.items()
                }
                log.info(
                    "Strategy diagnostics",
                    data={
                        "bot_id": self._cfg.bot_id,
                        "farm_mode": self._farm_mode,
                        "halted": self._halted,
                        "halt_reason": self._halt_reason or None,
                        "truth_stale": self._truth_stale,
                        "truth_age_s": truth_age,
                        "eval_total": eval_total,
                        "eval_tickers": len(self._eval_counts),
                        "top_eval_tickers": [
                            {"ticker": t, "evals": c} for t, c in top_eval
                        ],
                        "stalled_eval_tickers": [
                            {"ticker": t, "age_s": round(age, 2)} for t, age in stalled
                        ],
                        "decision_counts": dict(sorted(self._pass_reasons.items(), key=lambda x: x[1], reverse=True)[:12]),
                    },
                )
                self._eval_counts.clear()
                self._pass_reasons.clear()
        except asyncio.CancelledError:
            pass

    # -- farm mode: synchronous evaluation -----------------------------------

    def evaluate_sync(self, ticker: str, truth_stale: bool = False) -> Optional[TradeSignal]:
        """Farm-mode synchronous evaluation — no bus I/O.

        Called by FarmDispatcher on every OB/prob update.
        Returns a TradeSignal to publish, or None to pass.
        All state reads are from shared dicts (zero fan-out overhead).
        """
        if self._halted:
            return None
        if truth_stale:
            return None

        fp = self._fair_probs.get(ticker)
        ob = self._orderbooks.get(ticker)

        if not fp or not ob:
            return None
        if not ob.valid:
            return None

        p_yes = fp.p_yes
        if math.isnan(p_yes):
            return None

        # Per-asset truth tick age check.
        asset = self._market_asset.get(ticker, "BTC")
        last_asset_tick = self._last_truth_tick_time_by_asset.get(asset, 0.0)
        if last_asset_tick and (time.monotonic() - last_asset_tick) > self._cfg.truth_feed_stale_timeout_s:
            return None

        flow = float(self._trade_flow.get(ticker, 0.0))
        drift = float(getattr(fp, "drift", 0.0))

        # Early-exit hold when both momentum and flow reverse against position.
        hold_qty_centicx = int(self._hold_position_qty.get(ticker, 0))
        hold_side = self._hold_position_side.get(ticker)
        if hold_qty_centicx > 0 and hold_side in ("yes", "no"):
            min_drift = float(self._cfg.scalp_momentum_min_drift)
            flow_rev = float(self._cfg.hold_flow_reversal_threshold)
            reversal = (
                (hold_side == "yes" and drift < -min_drift and flow < -flow_rev)
                or (hold_side == "no" and drift > min_drift and flow > flow_rev)
            )
            if reversal:
                now_mono = time.monotonic()
                cooldown_s = self._cfg.signal_cooldown_s
                last = self._last_signal_time.get(ticker)
                if cooldown_s > 0 and last is not None and last[0] == hold_side and (now_mono - last[1]) < cooldown_s:
                    return None
                limit_cents = ob.best_yes_bid_cents if hold_side == "yes" else ob.best_no_bid_cents
                if limit_cents <= 0:
                    return None
                self._last_signal_time[ticker] = (hold_side, now_mono)
                return TradeSignal(
                    market_ticker=ticker,
                    side=hold_side,
                    action="sell",
                    limit_price_cents=limit_cents,
                    quantity_contracts=max(1, hold_qty_centicx // 100),
                    edge=0.0,
                    p_yes=float(p_yes),
                    timestamp=time.time(),
                    order_style="aggressive",
                    source="mispricing_hold",
                    bot_id=self._cfg.bot_id,
                )

        # Time-to-expiry.
        settle_ts = self._market_settlement.get(ticker, 0.0)
        time_to_settle_s = settle_ts - time.time() if settle_ts > 0 else 9999.0
        near_expiry_s = self._cfg.near_expiry_minutes * 60
        near_expiry = 0 < time_to_settle_s < near_expiry_s

        is_range = self._market_is_range.get(ticker, False)
        window_minutes = self._market_window_min.get(ticker, 15)
        if not _hold_family_allowed(asset, window_minutes, is_range):
            return None
        max_entry_s = hold_entry_horizon_seconds(
            window_minutes=window_minutes,
            is_range=is_range,
            max_entry_minutes_to_expiry=self._cfg.max_entry_minutes_to_expiry,
            range_max_entry_minutes_to_expiry=self._cfg.range_max_entry_minutes_to_expiry,
        )
        if not is_range and max_entry_s > 0 and time_to_settle_s > max_entry_s:
            return None
        range_max_s = max_entry_s
        if is_range and range_max_s > 0 and time_to_settle_s > range_max_s:
            return None
        min_entry_s = self._cfg.hold_min_entry_minutes_to_expiry * 60
        if min_entry_s > 0 and time_to_settle_s < min_entry_s:
            return None

        if near_expiry:
            min_edge = self._cfg.near_expiry_min_edge
            persistence_ms_eff = self._cfg.near_expiry_persistence_ms
        else:
            min_edge = self._cfg.min_edge_threshold
            persistence_ms_eff = self._cfg.persistence_window_ms

        # OBI-adjusted probability (same logic as async path — see comment there).
        obi_weight = self._cfg.obi_p_yes_bias_weight
        if obi_weight and ob.obi:
            p_yes = max(0.02, min(0.98, p_yes + ob.obi * obi_weight))

        # Momentum-adjusted probability (same logic as async path).
        momentum_weight = self._cfg.momentum_p_yes_bias_weight
        if momentum_weight and fp.drift:
            p_yes = max(0.02, min(0.98, p_yes + fp.drift * momentum_weight))

        # Trade flow bias (same logic as async path).
        flow_weight = self._cfg.trade_flow_p_yes_bias_weight
        if flow_weight and flow:
            p_yes = max(0.02, min(0.98, p_yes + flow * flow_weight))

        fee = self._cfg.effective_edge_fee_pct
        yes_ask_prob = ob.implied_yes_ask_cents / 100.0
        no_ask_prob = ob.implied_no_ask_cents / 100.0
        divergence = p_yes - yes_ask_prob
        forced_side: Optional[str] = "yes" if divergence > 0 else "no"
        side_sign = 1.0 if forced_side == "yes" else -1.0
        depth_pressure = float(getattr(ob, "depth_pressure", 0.0))
        yes_delta_flow = float(self._delta_flow_yes.get(ticker, 0.0))
        no_delta_flow = float(self._delta_flow_no.get(ticker, 0.0))
        depth_agree = max(0.0, side_sign * depth_pressure)
        delta_agree_raw = ((yes_delta_flow - no_delta_flow) * side_sign) / 2.0
        delta_agree = max(0.0, delta_agree_raw)
        confirmation = max(0.0, min(1.0, 0.5 * depth_agree + 0.5 * delta_agree))
        divergence_threshold = self._cfg.hold_min_divergence_threshold * (1.0 - 0.2 * confirmation)
        if abs(divergence) < divergence_threshold:
            self._persistence_state.pop(ticker, None)
            return None
        if self._cfg.hold_require_momentum_agreement:
            min_mag = float(self._cfg.hold_momentum_agreement_min_drift)
            if abs(drift) < min_mag:
                self._persistence_state.pop(ticker, None)
                return None
            if drift != 0.0:
                if (forced_side == "yes" and drift < 0.0) or (forced_side == "no" and drift > 0.0):
                    self._persistence_state.pop(ticker, None)
                    return None
        if self._cfg.hold_require_flow_agreement:
            min_mag = float(self._cfg.hold_flow_agreement_min_flow)
            if abs(flow) < min_mag:
                self._persistence_state.pop(ticker, None)
                return None
            if flow != 0.0:
                if (forced_side == "yes" and flow < 0.0) or (forced_side == "no" and flow > 0.0):
                    self._persistence_state.pop(ticker, None)
                    return None
        ev_yes = divergence
        ev_no = -divergence
        effective_yes = ev_yes - fee
        effective_no = ev_no - fee
        penalized_yes = effective_yes - _hold_tail_penalty(
            ob.implied_yes_ask_cents,
            "yes",
            self._cfg.hold_tail_penalty_start_cents,
            self._cfg.hold_tail_penalty_per_10c,
        )
        penalized_no = effective_no - _hold_tail_penalty(
            ob.implied_no_ask_cents,
            "no",
            self._cfg.hold_tail_penalty_start_cents,
            self._cfg.hold_tail_penalty_per_10c,
        )

        side: Optional[str] = None
        edge: float = 0.0
        limit_cents: int = 0

        if forced_side == "yes":
            if penalized_yes >= min_edge:
                side = "yes"
                edge = penalized_yes
                limit_cents = ob.implied_yes_ask_cents
        elif forced_side == "no":
            if penalized_no >= min_edge:
                side = "no"
                edge = penalized_no
                limit_cents = ob.implied_no_ask_cents
        else:
            if penalized_yes >= min_edge and penalized_yes >= penalized_no:
                side = "yes"
                edge = penalized_yes
                limit_cents = ob.implied_yes_ask_cents
            elif penalized_no >= min_edge:
                side = "no"
                edge = penalized_no
                limit_cents = ob.implied_no_ask_cents

        if side is None:
            self._persistence_state.pop(ticker, None)
            return None

        mn = self._cfg.min_entry_cents
        mx = self._cfg.max_entry_cents
        if (mn > 0 and limit_cents < mn) or (mx < 100 and limit_cents > mx):
            self._persistence_state.pop(ticker, None)
            return None
        if (
            side == "no"
            and self._cfg.no_avoid_above_cents > 0
            and limit_cents >= self._cfg.no_avoid_above_cents
        ):
            self._persistence_state.pop(ticker, None)
            return None

        if side == "yes":
            y_min = self._cfg.yes_avoid_min_cents
            y_max = self._cfg.yes_avoid_max_cents
            if y_min > 0 and y_max > 0 and y_min <= limit_cents <= y_max:
                self._persistence_state.pop(ticker, None)
                return None

        now_mono = time.monotonic()
        if persistence_ms_eff > 0:
            prev = self._persistence_state.get(ticker)
            if prev is None or prev[0] != side:
                self._persistence_state[ticker] = (side, now_mono)
                return None
            if (now_mono - prev[1]) * 1000 < persistence_ms_eff:
                return None

        cooldown_s = self._cfg.signal_cooldown_s
        if cooldown_s > 0:
            last = self._last_signal_time.get(ticker)
            if last is not None and last[0] == side and (now_mono - last[1]) < cooldown_s:
                return None

        latency_limit_ms = self._cfg.latency_circuit_breaker_ms
        if latency_limit_ms > 0 and last_asset_tick > 0:
            if (now_mono - last_asset_tick) * 1000 > latency_limit_ms:
                return None

        current_position_centicx = abs(self._positions.get(ticker, 0))
        current_position_contracts = current_position_centicx // 100
        max_usd = self._current_balance * self._cfg.max_fraction_per_market
        max_contracts_limit = int(max_usd / (limit_cents / 100.0)) if limit_cents > 0 else 0
        risk_usd_per_trade = self._current_balance * self._cfg.risk_fraction_per_trade
        cost_per_contract = max(limit_cents / 100.0, 0.01)
        base_qty = max(1, int(risk_usd_per_trade / cost_per_contract))
        qty = max(base_qty, int(base_qty * (edge / min_edge)))
        qty = max(1, int(qty * _session_mult(time.time(), sleeve="hold")))

        ticker_cap = self._cfg.max_contracts_per_ticker
        if ticker_cap > 0:
            qty = min(qty, ticker_cap - current_position_contracts)
        qty = min(qty, max_contracts_limit - current_position_contracts)

        if qty <= 0:
            return None

        if self._daily_pnl <= -self._start_bankroll * self._cfg.daily_drawdown_limit:
            if not self._halted:
                self._halted = True
                self._halt_reason = "drawdown"
                log.warning("Daily drawdown limit breached — halting [%s]", self._cfg.bot_id)
            return None

        order_style = self._cfg.default_order_style
        if order_style == "passive":
            if side == "yes":
                limit_cents = ob.best_yes_bid_cents if ob.best_yes_bid_cents > 0 else limit_cents
            else:
                limit_cents = ob.best_no_bid_cents if ob.best_no_bid_cents > 0 else limit_cents

        self._last_signal_time[ticker] = (side, now_mono)
        return TradeSignal(
            market_ticker=ticker,
            side=side,
            action="buy",
            limit_price_cents=limit_cents,
            quantity_contracts=qty,
            edge=round(edge, 4),
            p_yes=round(p_yes, 4),
            timestamp=time.time(),
            order_style=order_style,
            source="mispricing_hold",
            bot_id=self._cfg.bot_id,
        )

    # -- external updates ----------------------------------------------------

    def update_pnl(self, delta: float) -> None:
        """Called by execution layer to update cumulative daily P&L."""
        self._daily_pnl += delta

    async def update_tickers(self, added: List[str], removed: List[str]) -> None:
        """Dynamically update the set of markets this strategy watches."""
        if not self._running:
            return

        # Prune completed tasks to prevent unbounded list growth.
        self._prune_tasks()

        if not self._farm_mode:
            # Non-farm: subscribe to each new OB topic individually.
            for ticker in added:
                if ticker not in self._orderbooks:
                    q_ob = await self._bus.subscribe(f"kalshi.orderbook.{ticker}")
                    self._tasks.append(
                        asyncio.create_task(self._consume_orderbook(q_ob, ticker))
                    )
                    log.debug(f"Strategy subscribed to {ticker}")
        # Farm mode: FarmDispatcher owns OB subscriptions; no per-bot subscribe needed.

        # Announce active tickers for the UI (one per asset) — single-bot only.
        if not self._farm_mode:
            current_active = set(self._orderbooks.keys()) | set(added)
            active_by_asset: Dict[str, List[str]] = {}
            for t in current_active:
                if t in removed:
                    continue
                asset = self._market_asset.get(t, "BTC")
                active_by_asset.setdefault(asset, []).append(t)
            for asset, tickers in active_by_asset.items():
                if not tickers:
                    continue
                primary = sorted(tickers)[-1]
                await self._bus.publish(
                    "kalshi.strategy.active_ticker",
                    ActiveTicker(ticker=primary, timestamp=time.time()),
                )

    def _prune_tasks(self) -> None:
        """Drop completed tasks so the task registry remains bounded."""
        self._tasks = [t for t in self._tasks if not t.done()]
