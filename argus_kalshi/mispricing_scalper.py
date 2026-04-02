# Created by Oliver Meihls

# Directional scalper for Kalshi contracts.
#
# Entry model
# Uses a directional composite score from:
# - drift (FairProbability.drift)
# - orderbook imbalance (obi)
# - depth pressure (multi-level imbalance)
# - trade flow imbalance
# - order-delta flow (YES adds vs cancels and NO adds vs cancels)
#
# No GBM fair-vs-ask edge is used for scalp entry direction.

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from .bus import Bus
from .config import KalshiConfig
from .logging_utils import ComponentLogger
from .paper_model import estimate_kalshi_taker_fee_usd
from .shared_state import SharedFarmState
from .models import (
    FairProbability,
    FillEvent,
    MarketMetadata,
    OrderbookState,
    SelectedMarkets,
    TradeSignal,
)

log = ComponentLogger("mispricing_scalper")


def _session_mult(now_wall: float, sleeve: str = "scalp") -> float:
    # UTC-hour-based sizing multiplier derived from top-wallet activity patterns.
    utc_hour = datetime.fromtimestamp(now_wall, tz=timezone.utc).hour
    if 15 <= utc_hour < 21:
        return 1.20
    if 11 <= utc_hour < 15:
        return 1.10 if sleeve == "hold" else 1.05
    if 0 <= utc_hour < 9:
        return 0.70
    return 1.00


def _scalp_family_allowed(asset: str, window_minutes: int, is_range: bool) -> bool:
    # Restrict scalping to families with plausible short-horizon edge.
    if asset == "SOL":
        return False
    if is_range:
        return False
    if asset == "BTC":
        return window_minutes in (15, 60)
    if asset == "ETH":
        return window_minutes in (15, 60)
    return False


@dataclass(slots=True)
class _ScalpPosition:
    ticker: str
    side: str
    entry_price_cents: int
    quantity_centicx: int
    opened_at: float
    entry_source: str = "mispricing_scalp"


class MispricingScalper:
    # Directional scalp entry + early-exit manager.
    #
    # Farm mode
    # Pass ``shared=<SharedFarmState>`` to run in farm mode. In farm mode,
    # high-frequency state is read from shared dicts and dispatcher calls
    # ``scalp_sync`` / ``exit_sync`` directly.

    def __init__(self, config: KalshiConfig, bus: Bus, shared: Optional[SharedFarmState] = None) -> None:
        self._cfg = config
        self._bus = bus
        self._farm_mode: bool = shared is not None

        if shared is not None:
            self._fair_probs = shared.fair_probs
            self._orderbooks = shared.orderbooks
            self._scalp_eligible = shared.scalp_eligible
            self._settlement_time_epoch = shared.scalp_settlement_epoch
            self._prev_fair_probs = shared.prev_fair_prob_by_ticker
            self._prev_prob_ts = shared.prev_fair_prob_ts_by_ticker
            self._last_prob_ts = shared.last_fair_prob_ts_by_ticker
            self._trade_flow = shared.trade_flow_by_ticker
            self._delta_flow_yes = shared.orderbook_delta_flow_yes
            self._delta_flow_no = shared.orderbook_delta_flow_no
        else:
            self._fair_probs: Dict[str, object] = {}
            self._orderbooks: Dict[str, OrderbookState] = {}
            self._scalp_eligible: Dict[str, bool] = {}
            self._settlement_time_epoch: Dict[str, float] = {}
            self._prev_fair_probs: Dict[str, float] = {}
            self._prev_prob_ts: Dict[str, float] = {}
            self._last_prob_ts: Dict[str, float] = {}
            self._trade_flow: Dict[str, float] = {}
            self._delta_flow_yes: Dict[str, float] = {}
            self._delta_flow_no: Dict[str, float] = {}
            self._trade_deques: Dict[str, Deque[Tuple[float, str, int]]] = {}
            self._delta_yes_deques: Dict[str, Deque[Tuple[float, bool, int]]] = {}
            self._delta_no_deques: Dict[str, Deque[Tuple[float, bool, int]]] = {}
            self._trade_flow_window_s: float = 60.0
            self._delta_flow_window_s: float = 30.0

        self._last_signal_ts: Dict[str, float] = {}
        self._arb_last_entry_ts: Dict[str, float] = {}
        self._open_positions: Dict[str, _ScalpPosition] = {}
        self._current_balance: float = config.bankroll_usd
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        self._running = True
        if self._farm_mode:
            return

        q_prob = await self._bus.subscribe("kalshi.fair_prob")
        q_markets = await self._bus.subscribe("kalshi.ranked_markets")
        q_fills = await self._bus.subscribe("kalshi.fills")
        q_meta = await self._bus.subscribe("kalshi.market_metadata")
        q_balance = await self._bus.subscribe("kalshi.account_balance")
        q_trade = await self._bus.subscribe("kalshi.trade")
        q_delta = await self._bus.subscribe("kalshi.orderbook_delta_flow")
        self._tasks.append(asyncio.create_task(self._consume_prob(q_prob)))
        self._tasks.append(asyncio.create_task(self._consume_markets(q_markets)))
        self._tasks.append(asyncio.create_task(self._consume_fills(q_fills)))
        self._tasks.append(asyncio.create_task(self._consume_metadata(q_meta)))
        self._tasks.append(asyncio.create_task(self._consume_balance(q_balance)))
        self._tasks.append(asyncio.create_task(self._consume_trades(q_trade)))
        self._tasks.append(asyncio.create_task(self._consume_orderbook_delta_flow(q_delta)))

    async def stop(self) -> None:
        self._running = False
        tasks = list(self._tasks)
        self._tasks.clear()
        for t in tasks:
            t.cancel()
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=15.0)
            except asyncio.TimeoutError:
                log.debug("Scalper stop: some consumer tasks did not exit in time")

    async def _consume_markets(self, q: asyncio.Queue) -> None:
        subscribed: set[str] = set()
        try:
            while self._running:
                msg: SelectedMarkets = await q.get()
                for ticker in msg.tickers:
                    if ticker in subscribed:
                        continue
                    q_ob = await self._bus.subscribe(f"kalshi.orderbook.{ticker}")
                    self._tasks = [t for t in self._tasks if not t.done()]
                    self._tasks.append(asyncio.create_task(self._consume_orderbook(q_ob, ticker)))
                    subscribed.add(ticker)
        except asyncio.CancelledError:
            pass

    def handle_balance(self, msg: object) -> None:
        self._current_balance = float(getattr(msg, "balance_usd", self._current_balance))

    async def _consume_prob(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                msg: FairProbability = await q.get()
                prev = self._fair_probs.get(msg.market_ticker)
                if prev is not None:
                    prev_val = prev.p_yes if hasattr(prev, "p_yes") else float(prev)
                    self._prev_fair_probs[msg.market_ticker] = float(prev_val)
                    self._prev_prob_ts[msg.market_ticker] = self._last_prob_ts.get(
                        msg.market_ticker,
                        time.monotonic(),
                    )
                self._fair_probs[msg.market_ticker] = msg
                self._last_prob_ts[msg.market_ticker] = time.monotonic()
                await self._maybe_emit(msg.market_ticker)
        except asyncio.CancelledError:
            pass

    async def _consume_trades(self, q: asyncio.Queue) -> None:
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
                    dq = deque(maxlen=1000)
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

    def handle_fill(self, fill: FillEvent) -> None:
        if fill.action != "buy":
            return
        if fill.source not in {"mispricing_scalp", "momentum_scalp"}:
            return
        if getattr(fill, "bot_id", "default") != self._cfg.bot_id:
            return
        self._record_fill(fill)

    async def _consume_orderbook(self, q: asyncio.Queue, ticker: str) -> None:
        try:
            while self._running:
                msg: OrderbookState = await q.get()
                if not isinstance(msg, OrderbookState):
                    continue
                self._orderbooks[ticker] = msg
                await self._maybe_exit(ticker)
                await self._maybe_emit(ticker)
        except asyncio.CancelledError:
            pass

    async def _consume_balance(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                msg = await q.get()
                self._current_balance = float(getattr(msg, "balance_usd", self._current_balance))
        except asyncio.CancelledError:
            pass

    async def _consume_metadata(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                meta: MarketMetadata = await q.get()
                settlement_epoch = None
                try:
                    dt = datetime.fromisoformat(meta.settlement_time_iso.replace("Z", "+00:00"))
                    settlement_epoch = dt.timestamp()
                    self._settlement_time_epoch[meta.market_ticker] = settlement_epoch
                except (ValueError, TypeError):
                    pass

                time_to_settle_s = (settlement_epoch - time.time()) if settlement_epoch else None
                eligible = (
                    meta.window_minutes is not None
                    and _scalp_family_allowed(meta.asset, meta.window_minutes, meta.is_range)
                    and (
                        not meta.is_range
                        or (time_to_settle_s is not None and 0 < time_to_settle_s <= 3600.0)
                    )
                )
                self._scalp_eligible[meta.market_ticker] = eligible
        except asyncio.CancelledError:
            pass

    async def _consume_fills(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                fill: FillEvent = await q.get()
                if fill.action != "buy":
                    continue
                if fill.source not in {"mispricing_scalp", "momentum_scalp"}:
                    continue
                if getattr(fill, "bot_id", "default") != self._cfg.bot_id:
                    continue
                self._record_fill(fill)
        except asyncio.CancelledError:
            pass

    def _record_fill(self, fill: FillEvent) -> None:
        ticker = fill.market_ticker
        if ticker in self._open_positions:
            pos = self._open_positions[ticker]
            total = pos.quantity_centicx + fill.count
            avg_entry = int(
                (pos.entry_price_cents * pos.quantity_centicx + fill.price_cents * fill.count)
                / max(1, total)
            )
            self._open_positions[ticker] = _ScalpPosition(
                ticker=ticker,
                side=pos.side,
                entry_price_cents=avg_entry,
                quantity_centicx=total,
                opened_at=pos.opened_at,
                entry_source=pos.entry_source,
            )
        else:
            self._open_positions[ticker] = _ScalpPosition(
                ticker=ticker,
                side=fill.side,
                entry_price_cents=fill.price_cents,
                quantity_centicx=fill.count,
                opened_at=fill.timestamp,
                entry_source=fill.source or "mispricing_scalp",
            )

    async def _maybe_exit(self, ticker: str) -> None:
        signal = self.exit_sync(ticker)
        if signal is None:
            return
        self._open_positions.pop(ticker, None)
        await self._bus.publish("kalshi.trade_signal", signal)

    def _directional_score(self, ticker: str, fp: object, ob: OrderbookState) -> float:
        drift = float(getattr(fp, "drift", 0.0)) if fp is not None else 0.0
        drift_scale = max(1e-9, float(self._cfg.scalp_directional_drift_scale))
        drift_component = max(-1.0, min(1.0, drift / drift_scale))
        obi = float(getattr(ob, "obi", 0.0))
        depth_pressure = float(getattr(ob, "depth_pressure", 0.0))
        flow = float(self._trade_flow.get(ticker, 0.0))
        yes_delta_flow = float(self._delta_flow_yes.get(ticker, 0.0))
        no_delta_flow = float(self._delta_flow_no.get(ticker, 0.0))
        score = (
            float(self._cfg.scalp_directional_drift_weight) * drift_component
            + float(self._cfg.scalp_directional_obi_weight) * obi
            + float(self._cfg.scalp_directional_flow_weight) * flow
            + float(self._cfg.scalp_directional_depth_weight) * depth_pressure
            + float(self._cfg.scalp_directional_delta_yes_weight) * yes_delta_flow
            + float(self._cfg.scalp_directional_delta_no_weight) * (-no_delta_flow)
        )
        return max(-1.0, min(1.0, score))

    def _projected_scalp_net_profit(self, entry_cents: int, spread_cents: int) -> int:
        target_move_cents = int(self._cfg.scalp_min_profit_cents)
        target_exit_cents = min(99, int(entry_cents) + target_move_cents)
        round_trip_fee_cents = int(
            round(
                (
                    estimate_kalshi_taker_fee_usd(int(entry_cents), 100)
                    + estimate_kalshi_taker_fee_usd(int(target_exit_cents), 100)
                )
                * 100.0
            )
        )
        slippage_buffer = max(
            2 * max(0, int(getattr(self._cfg, "paper_slippage_cents", 0))),
            max(1, int(round(max(0, int(spread_cents)) * 0.5))),
        )
        return target_move_cents - slippage_buffer - round_trip_fee_cents

    async def _maybe_emit(self, ticker: str) -> None:
        signal = self.scalp_sync(ticker)
        if signal is None:
            return
        await self._bus.publish("kalshi.trade_signal", signal)

    def exit_sync(self, ticker: str) -> Optional[TradeSignal]:
        pos = self._open_positions.get(ticker)
        if pos is None:
            return None
        ob = self._orderbooks.get(ticker)
        if ob is None or not ob.valid:
            return None

        current_bid = ob.best_yes_bid_cents if pos.side == "yes" else ob.best_no_bid_cents
        now = time.time()
        age_minutes = (now - pos.opened_at) / 60.0
        min_profit = self._cfg.scalp_min_profit_cents
        max_hold = self._cfg.scalp_max_hold_minutes
        p_yes_val = self._current_p_yes(ticker)
        held_fair_cents = round(p_yes_val * 100) if pos.side == "yes" else round((1.0 - p_yes_val) * 100)
        held_ask_cents = ob.implied_yes_ask_cents if pos.side == "yes" else ob.implied_no_ask_cents
        held_edge_cents = held_fair_cents - held_ask_cents
        grace_elapsed = (now - pos.opened_at) >= self._cfg.scalp_exit_grace_s
        slippage_buffer = max(1, int(getattr(self._cfg, "paper_slippage_cents", 0))) * 2

        settlement_ts = self._settlement_time_epoch.get(ticker)
        if settlement_ts is not None and settlement_ts > now:
            minutes_to_settlement = (settlement_ts - now) / 60.0
            effective_max_hold = min(max_hold, minutes_to_settlement - 0.5)
            should_exit = minutes_to_settlement <= 2.0
        elif settlement_ts is not None and settlement_ts <= now:
            # Contract already expired — position is a ghost, clear it immediately.
            self._open_positions.pop(ticker, None)
            return None
        else:
            effective_max_hold = max_hold
            should_exit = False

        stop_loss_cents = int(getattr(self._cfg, "scalp_stop_loss_cents", 0))
        if not should_exit:
            if current_bid >= pos.entry_price_cents + min_profit:
                should_exit = True
            elif (
                stop_loss_cents > 0
                and grace_elapsed
                and current_bid <= pos.entry_price_cents - stop_loss_cents
            ):
                # Symmetric stop-loss: cut the trade when it moves against us by stop_loss_cents
                should_exit = True
            elif (
                grace_elapsed
                and current_bid >= pos.entry_price_cents + max(1, slippage_buffer // 2)
                and held_edge_cents <= self._cfg.scalp_exit_edge_threshold_cents
            ):
                should_exit = True
            elif age_minutes >= effective_max_hold and current_bid > pos.entry_price_cents:
                should_exit = True
            elif age_minutes >= effective_max_hold * 2 and held_edge_cents <= 0:
                should_exit = True

        if not should_exit:
            return None

        return TradeSignal(
            market_ticker=ticker,
            side=pos.side,
            action="sell",
            limit_price_cents=current_bid,
            quantity_contracts=max(1, pos.quantity_centicx // 100),
            edge=float(current_bid - pos.entry_price_cents),
            p_yes=p_yes_val,
            timestamp=now,
            order_style="aggressive",
            source=getattr(pos, "entry_source", "mispricing_scalp"),
            bot_id=self._cfg.bot_id,
        )

    def scalp_sync(self, ticker: str) -> Optional[TradeSignal]:
        # Prune ghost positions from expired contracts (OB updates stop after expiry,
        # so _maybe_exit never fires and positions get permanently stuck).
        # Two conditions: known-expired (settlement_ts <= now) OR very old (>45min
        # fallback for tickers whose metadata never arrived, settlement_ts is None).
        now = time.time()
        _MAX_AGE_S = 2700.0  # 45 minutes
        expired = [
            t for t, pos in self._open_positions.items()
            if (self._settlement_time_epoch.get(t, float("inf")) <= now
                or (now - pos.opened_at) > _MAX_AGE_S)
        ]
        for t in expired:
            self._open_positions.pop(t, None)

        if ticker in self._open_positions:
            return None
        if not self._scalp_eligible.get(ticker, True):
            return None
        cooldown = self._cfg.scalp_cooldown_s
        if cooldown > 0 and (now - self._last_signal_ts.get(ticker, 0.0)) < cooldown:
            return None

        fp_obj = self._fair_probs.get(ticker)
        ob = self._orderbooks.get(ticker)
        if fp_obj is None or ob is None or not ob.valid:
            return None

        spread = max(
            0,
            (ob.implied_yes_ask_cents - ob.best_yes_bid_cents)
            + (ob.implied_no_ask_cents - ob.best_no_bid_cents),
        )
        if spread > self._cfg.scalp_max_spread_cents:
            return None

        score = self._directional_score(ticker, fp_obj, ob)
        threshold = float(self._cfg.scalp_directional_score_threshold)
        if abs(score) < threshold:
            return None

        side = "yes" if score > 0 else "no"
        limit_cents = ob.implied_yes_ask_cents if side == "yes" else ob.implied_no_ask_cents
        if self._cfg.scalp_min_entry_cents > 0 and limit_cents < self._cfg.scalp_min_entry_cents:
            return None
        if self._cfg.scalp_max_entry_cents < 100 and limit_cents > self._cfg.scalp_max_entry_cents:
            return None

        projected_net_profit = self._projected_scalp_net_profit(limit_cents, spread)
        if projected_net_profit < int(self._cfg.scalp_entry_cost_buffer_cents):
            return None

        risk_usd_per_trade = self._current_balance * self._cfg.risk_fraction_per_trade
        cost_per_contract = max(limit_cents / 100.0, 0.01)
        base_qty = max(1, int(risk_usd_per_trade / cost_per_contract))
        confidence_mult = max(1.0, abs(score) / max(1e-9, threshold))
        quantity = max(base_qty, int(base_qty * confidence_mult))
        quantity = max(1, int(quantity * _session_mult(now, sleeve="scalp")))
        quantity = min(self._cfg.scalp_max_quantity, quantity)

        self._last_signal_ts[ticker] = now
        return TradeSignal(
            market_ticker=ticker,
            side=side,
            action="buy",
            limit_price_cents=limit_cents,
            quantity_contracts=quantity,
            edge=round(abs(score), 4),
            p_yes=self._current_p_yes(ticker),
            timestamp=now,
            order_style="aggressive",
            source="mispricing_scalp",
            bot_id=self._cfg.bot_id,
        )

    def _current_p_yes(self, ticker: str) -> float:
        fp_obj = self._fair_probs.get(ticker)
        if hasattr(fp_obj, "p_yes"):
            return float(fp_obj.p_yes)
        return float(fp_obj or 0.5)
