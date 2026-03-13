"""
Settlement Tracking for Argus Kalshi
====================================

Listens for fills and final BTC window averages to calculate and log
the outcome of completed trades.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .bus import Bus
from .models import BtcWindowState, FairProbability, FillEvent, MarketMetadata, SettlementOutcome
from .paper_log import append_paper_log_sync
from .paper_model import estimate_kalshi_taker_fee_usd

log = logging.getLogger("argus.kalshi.settlement")
_PAPER_LOG_PATH = "logs/paper_trades.jsonl"

def _append_paper_log(record: dict) -> None:
    append_paper_log_sync(record)


@dataclass
class Position:
    market_ticker: str
    side: str           # "yes" or "no"
    avg_price_cents: float
    total_qty: int      # centi-contracts
    settlement_time: float   # epoch seconds
    strike: float
    source: str = ""    # from first fill, e.g. "mispricing_scalp" for scalp attribution
    asset: str = "BTC"
    is_range: bool = False
    strike_floor: Optional[float] = None
    strike_cap: Optional[float] = None
    entry_fees_usd: float = 0.0
    family: str = ""
    scenario_profile: str = "base"
    decision_context: Dict[str, Any] = field(default_factory=dict)


class SettlementTracker:
    """
    Watches the bus for fills and BTC window completions to report outcomes.
    """

    def __init__(self, bus: Bus, db: Optional[Any] = None):
        self._bus = bus
        self._db = db
        self._markets: Dict[str, MarketMetadata] = {}
        self._paper_positions: Dict[str, Dict[str, Position]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Buffer for buy fills that arrived before metadata was received.
        # Processed when the corresponding metadata arrives (within ~15s).
        self._pending_fills: List[FillEvent] = []

        # Calibration / Brier score tracking (Phase 3).
        # Cache the latest p_yes per ticker so we can record it at settlement.
        self._last_p_yes: Dict[str, float] = {}
        # Running Brier score accumulator: sum of (p - outcome)^2 and count.
        self._brier_sum: float = 0.0
        self._brier_count: int = 0

    def _positions_for_bot(self, bot_id: str) -> Dict[str, Position]:
        return self._paper_positions.setdefault(bot_id, {})

    @property
    def brier_score(self) -> Optional[float]:
        """Running Brier score (lower is better). None if no settlements yet."""
        if self._brier_count == 0:
            return None
        return self._brier_sum / self._brier_count

    async def start(self):
        self._running = True
        
        # Phase 0: Reconstruct open positions from JSONL logs (trade continuity)
        try:
            self._load_open_fills()
        except Exception as e:
            log.error(f"Failed to load open fills from JSONL: {e}")

        self._tasks = [
            asyncio.create_task(self._loop_metadata()),
            asyncio.create_task(self._loop_fills()),
            asyncio.create_task(self._loop_windows()),
            asyncio.create_task(self._loop_fair_probs()),
        ]
        log.info("Settlement tracker started")

    def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()

    def _load_open_fills(self):
        """Scan the JSONL paper log to reconstruct open positions from previous runs."""
        if not os.path.exists(_PAPER_LOG_PATH):
            return

        log.info(f"Scanning {_PAPER_LOG_PATH} for open paper positions...")
        
        # temporary storage: bot_id -> ticker -> Position
        active_positions: Dict[str, Dict[str, Position]] = {}

        try:
            with open(_PAPER_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        rtype = record.get("type")
                        ticker = record.get("market_ticker")
                        bot_id = record.get("bot_id", "default")
                        
                        if not ticker: continue
                        
                        bot_map = active_positions.setdefault(bot_id, {})

                        if rtype == "paper_fill":
                            # New fill: create or update position
                            #
                            # IMPORTANT: Only BUY-side fills open/increase positions.
                            # SELL fills are exits (incl. mispricing scalper early-exits)
                            # and are already reflected in separate "settlement" records.
                            # If we treated SELL fills as new buys here, then every
                            # restart would reconstruct synthetic positions from those
                            # exit fills and settle them again, compounding scalp PnL
                            # across restarts and blowing up the leaderboard.
                            action = record.get("action", "buy")
                            if action != "buy":
                                continue

                            settlement_time = record.get("settlement_time", 0)
                            if settlement_time <= 0 and record.get("settlement_time_iso"):
                                try:
                                    settle_dt = datetime.fromisoformat(
                                        record["settlement_time_iso"].replace("Z", "+00:00")
                                    )
                                    settlement_time = settle_dt.timestamp()
                                except (KeyError, TypeError, ValueError):
                                    pass
                            strike = record.get("strike", 0)
                            asset = record.get("asset", "BTC") or "BTC"
                            entry_fee_usd = float(record.get("estimated_taker_fee_usd", 0.0) or 0.0)

                            if ticker not in bot_map:
                                bot_map[ticker] = Position(
                                    market_ticker=ticker,
                                    side=record["side"],
                                    avg_price_cents=record["fill_price_cents"],
                                    total_qty=record["quantity_contracts"] * 100,  # convert to centicx
                                    settlement_time=settlement_time,
                                    strike=strike,
                                    source=record.get("source", ""),
                                    asset=asset,
                                    entry_fees_usd=entry_fee_usd,
                                    family=record.get("family", "") or "",
                                    scenario_profile=record.get("scenario_profile", "base") or "base",
                                    decision_context=record.get("decision_context", {}) or {},
                                )
                            else:
                                pos = bot_map[ticker]
                                new_total = pos.total_qty + (record["quantity_contracts"] * 100)
                                pos.avg_price_cents = (
                                    (pos.avg_price_cents * pos.total_qty) + 
                                    (record["fill_price_cents"] * record["quantity_contracts"] * 100)
                                ) / new_total
                                pos.total_qty = int(new_total)
                                if settlement_time > 0:
                                    pos.settlement_time = settlement_time
                                if strike:
                                    pos.strike = strike
                                if asset:
                                    pos.asset = asset
                                pos.entry_fees_usd += entry_fee_usd
                                if not pos.family:
                                    pos.family = record.get("family", "") or pos.family
                                rec_scenario = record.get("scenario_profile")
                                if rec_scenario:
                                    pos.scenario_profile = rec_scenario
                                if not pos.decision_context:
                                    pos.decision_context = record.get("decision_context", {}) or {}
                        
                        elif rtype == "settlement":
                            # Settled: remove from active tracking
                            bot_map.pop(ticker, None)

                    except (json.JSONDecodeError, KeyError):
                        continue
            
            # Filter and apply to self._paper_positions
            # Do NOT recover positions with settlement_time <= 0: they would be re-settled
            # on the first window with wrong asset/expiry and inflate PnL (e.g. $28 → $200+).
            count = 0
            for bot_id, bots_positions in active_positions.items():
                target_map = self._positions_for_bot(bot_id)
                for ticker, pos in bots_positions.items():
                    if pos.settlement_time <= 0:
                        log.debug(
                            "Skipping recovery for %s/%s: no settlement_time in log (would re-settle on wrong window)",
                            bot_id, ticker,
                        )
                        continue
                    target_map[ticker] = pos
                    count += 1
            
            if count > 0:
                log.info(f"Recovered {count} open paper positions across {len(active_positions)} bots")
            else:
                log.info("No open paper positions to recover")

        except Exception as e:
            log.warning(f"Error during paper fill recovery: {e}")

    async def _loop_metadata(self):
        """Track market metadata so we know strikes and settlement times.

        Also drains any buy fills that arrived before this metadata was
        received — they were buffered in _pending_fills rather than dropped.
        """
        q = await self._bus.subscribe("kalshi.market_metadata")
        try:
            while self._running:
                msg: MarketMetadata = await q.get()
                self._markets[msg.market_ticker] = msg

                # Process buffered fills that were waiting for this metadata.
                pending_for_ticker = [f for f in self._pending_fills if f.market_ticker == msg.market_ticker]
                if pending_for_ticker:
                    log.info(f"Draining {len(pending_for_ticker)} buffered fill(s) for {msg.market_ticker}")
                    for fill in pending_for_ticker:
                        self._pending_fills.remove(fill)
                        self._apply_buy_fill(fill, msg)
        except asyncio.CancelledError:
            pass

    def _apply_buy_fill(self, fill: FillEvent, meta: MarketMetadata) -> None:
        """Update position tracking for a single buy fill (metadata must be available)."""
        ticker = fill.market_ticker
        bot_id = getattr(fill, "bot_id", "default")
        positions = self._positions_for_bot(bot_id)
        settle_dt = datetime.fromisoformat(meta.settlement_time_iso.replace("Z", "+00:00"))
        settle_ts = settle_dt.timestamp()
        fill_source = getattr(fill, "source", "") or ""
        fill_fee_usd = float(getattr(fill, "fee_usd", 0.0) or estimate_kalshi_taker_fee_usd(fill.price_cents, fill.count))

        if ticker not in positions:
            positions[ticker] = Position(
                market_ticker=ticker,
                side=fill.side,
                avg_price_cents=fill.price_cents,
                total_qty=fill.count,
                settlement_time=settle_ts,
                strike=meta.strike_price,
                source=fill_source,
                asset=meta.asset,
                is_range=meta.is_range,
                strike_floor=meta.strike_floor,
                strike_cap=meta.strike_cap,
                entry_fees_usd=fill_fee_usd,
                family=getattr(fill, "family", "") or "",
                scenario_profile=getattr(fill, "scenario_profile", "base") or "base",
                decision_context=getattr(fill, "decision_context", {}) or {},
            )
        else:
            pos = positions[ticker]
            if not pos.source and fill_source:
                pos.source = fill_source
            new_total = pos.total_qty + fill.count
            pos.avg_price_cents = (
                (pos.avg_price_cents * pos.total_qty) + (fill.price_cents * fill.count)
            ) / new_total
            pos.total_qty = new_total
            pos.entry_fees_usd += fill_fee_usd
            if not pos.family:
                pos.family = getattr(fill, "family", "") or pos.family
            fill_scenario = getattr(fill, "scenario_profile", "")
            if fill_scenario:
                pos.scenario_profile = fill_scenario
            if not pos.decision_context:
                pos.decision_context = getattr(fill, "decision_context", {}) or {}

        log.info(
            f"Position updated: {ticker} | Side: {fill.side} "
            f"| Bot: {bot_id} | Net Qty: {positions[ticker].total_qty / 100:.2f} "
            f"| Avg: {positions[ticker].avg_price_cents:.1f}¢"
        )

    async def _loop_fills(self):
        """Track fills to build positions. Sell fills trigger immediate early-exit resolution."""
        q = await self._bus.subscribe("kalshi.fills")
        try:
            while self._running:
                fill: FillEvent = await q.get()
                ticker = fill.market_ticker
                fill_action = getattr(fill, "action", "buy")

                # Early-exit sell: resolve PnL immediately without waiting for settlement.
                if fill_action == "sell":
                    await self._resolve_early_exit(fill)
                    continue

                meta = self._markets.get(ticker)
                if not meta:
                    log.debug(f"Buffering fill for {ticker} (metadata not yet received)")
                    self._pending_fills.append(fill)
                    continue

                self._apply_buy_fill(fill, meta)
        except asyncio.CancelledError:
            pass

    async def _resolve_early_exit(self, fill: FillEvent) -> None:
        """Resolve an early-exit sell fill: compute PnL and publish SettlementOutcome immediately."""
        ticker = fill.market_ticker
        bot_id = getattr(fill, "bot_id", "default")
        positions = self._positions_for_bot(bot_id)
        pos = positions.pop(ticker, None)
        if pos is None:
            # No tracked position — stale fill or position already resolved.
            log.debug(f"Early-exit fill for {ticker} but no open position tracked; ignoring.")
            return

        sell_price = fill.price_cents
        entry_price = pos.avg_price_cents
        # Use whichever qty is smaller (fill may close a partial position).
        qty_centicx = min(fill.count, pos.total_qty)
        gross_pnl = (sell_price - entry_price) * qty_centicx / 10000.0
        entry_fee_alloc = pos.entry_fees_usd * (qty_centicx / pos.total_qty) if pos.total_qty > 0 else 0.0
        exit_fee_usd = float(getattr(fill, "fee_usd", 0.0) or estimate_kalshi_taker_fee_usd(sell_price, qty_centicx))
        total_fees = entry_fee_alloc + exit_fee_usd
        total_pnl = gross_pnl - total_fees
        won = total_pnl > 0

        status_str = "WIN" if won else "LOSS"
        log.info(
            f"EARLY EXIT [{status_str}]: {pos.market_ticker} {pos.side.upper()} "
            f"entry={entry_price:.1f}¢ exit={sell_price}¢ "
            f"qty={qty_centicx} centicx  PnL=${total_pnl:+.4f}"
        )

        settlement_record = {
            "type": "settlement",
            "settlement_method": "early_exit",
            "market_ticker": ticker,
            "asset": pos.asset,
            "side": pos.side,
            "won": won,
            "pnl_usd": total_pnl,
            "gross_pnl_usd": gross_pnl,
            "fees_usd": total_fees,
            "quantity_centicx": qty_centicx,
            "quantity_contracts": qty_centicx / 100.0,
            "entry_price_cents": entry_price,
            "exit_price_cents": sell_price,
            "strike": pos.strike,
            "source": pos.source or "",
            "family": pos.family,
            "scenario_profile": pos.scenario_profile,
            "timestamp": time.time(),
            "bot_id": bot_id,
        }
        if pos.decision_context:
            settlement_record["decision_context"] = pos.decision_context
        _append_paper_log(settlement_record)

        outcome = SettlementOutcome(
            market_ticker=ticker,
            side=pos.side,
            won=won,
            pnl=total_pnl,
            quantity_centicx=qty_centicx,
            entry_price_cents=entry_price,
            final_avg=float(sell_price),
            strike=pos.strike,
            timestamp=time.time(),
            gross_pnl=gross_pnl,
            fees_usd=total_fees,
            bot_id=bot_id,
            source=pos.source or "",
            family=pos.family,
            scenario_profile=pos.scenario_profile,
            decision_context=pos.decision_context or {},
        )
        await self._bus.publish("kalshi.settlement_outcome", outcome)

        if self._db:
            try:
                await self._db.insert_kalshi_outcome(
                    market_ticker=ticker,
                    strike=pos.strike,
                    direction=pos.side,
                    entry_price=entry_price / 100.0,
                    exit_price=sell_price / 100.0,
                    quantity=qty_centicx,
                    pnl=total_pnl,
                    outcome="WON" if won else "LOST",
                    final_avg=float(sell_price),
                    settled_at_ms=int(time.time() * 1000),
                    is_paper=True,
                    bot_id=bot_id,
                )
            except Exception as e:
                log.error(f"Failed to persist early-exit outcome: {e}")

    async def _loop_fair_probs(self):
        """Cache latest p_yes per ticker for Brier calibration at settlement."""
        q = await self._bus.subscribe("kalshi.fair_prob")
        try:
            while self._running:
                msg: FairProbability = await q.get()
                self._last_p_yes[msg.market_ticker] = msg.p_yes
        except asyncio.CancelledError:
            pass

    async def _loop_windows(self):
        """Watch for window completions to settle positions."""
        topics = ["btc.window_state", "eth.window_state", "sol.window_state"]
        queues = [await self._bus.subscribe(t) for t in topics]
        tasks = [asyncio.create_task(q.get()) for q in queues]
        try:
            while self._running:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for i, task in enumerate(tasks):
                    if task in done:
                        window: BtcWindowState = task.result()
                        tasks[i] = asyncio.create_task(queues[i].get())
                        window_asset = getattr(window, "asset", topics[i].split(".")[0].upper()).upper()

                        for bot_id, positions in list(self._paper_positions.items()):
                            settled_tickers = []
                            for ticker, pos in positions.items():
                                if pos.asset != window_asset:
                                    continue
                                if window.timestamp >= pos.settlement_time:
                                    await self._resolve_settlement(pos, window.last_60_avg, bot_id)
                                    settled_tickers.append(ticker)
                            for t in settled_tickers:
                                del positions[t]
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()

    async def _resolve_settlement(self, pos: Position, final_avg: float, bot_id: str):
        """Calculate and log the final outcome of a position."""
        # For KXBTC 15m: "Yes" wins if Avg > Strike?
        # Actually it depends on the specific contract ticker's definition.
        # KXBTC usually has markets for "Above X" (Yes) or "Below X" (Yes).
        # We'll assume the standard Kalshi "Yes" = Above Strike for mapping purposes,
        # but Kalshi tickers often look like "KXBTC-..." with Strike in the name.
        
        # Note: In the probability engine we used (Strike - last_avg).
        # We need to be careful with the side.
        
        # In kalshi_strategy.py:
        # if p_yes > (implied_yes_ask / 100) + edge: buy yes
        # if (1 - p_yes) > (implied_no_ask / 100) + edge: buy no
        
        # For most KXBTC 15m contracts:
        # YES resolves to $1.00 if Avg > Strike? No, Kalshi is specific.
        # Example: KXBTC-25SEP24-65000: "Will BTC be > 65000?"
        
        # We'll use a simple heuristic: if side is 'yes', payout is $1.00 if Avg > Strike.
        # Side is 'no', payout is $1.00 if Avg <= Strike.
        
        if pos.is_range and pos.strike_floor is not None and pos.strike_cap is not None:
            yes_wins = pos.strike_floor <= final_avg <= pos.strike_cap
        else:
            # Up/down resolves YES when final average is at or above strike.
            yes_wins = final_avg >= pos.strike
        won = (pos.side == "yes" and yes_wins) or (pos.side == "no" and not yes_wins)
        
        payout_cents = 100 if won else 0
        profit_per_contract = payout_cents - pos.avg_price_cents
        # pos.total_qty is in centi-contracts (100 per whole contract).
        # profit_per_contract is in cents per whole contract.
        # → divide by 10000 = (100 cents→$) × (100 centicx→contract) to get dollars.
        total_payout = (payout_cents * pos.total_qty) / 10000.0
        gross_profit = (profit_per_contract * pos.total_qty) / 10000.0
        total_profit = gross_profit - pos.entry_fees_usd
        won = total_profit > 0
        
        status_str = "WON 🏆" if won else "LOST 💀"
        log.info(
            f"💰 SETTLEMENT: {pos.market_ticker} | Final Avg: {final_avg:.2f} | Strike: {pos.strike} | "
            f"Result: {status_str} | Payout: ${total_payout:.2f} | Net PnL: ${total_profit:+.2f}"
        )

        # Brier calibration: record (p_yes, outcome) for this settlement.
        p_yes_at_trade = self._last_p_yes.pop(pos.market_ticker, None)
        if p_yes_at_trade is not None:
            outcome_binary = 1.0 if yes_wins else 0.0
            brier_contrib = (p_yes_at_trade - outcome_binary) ** 2
            self._brier_sum += brier_contrib
            self._brier_count += 1
            log.info(
                f"Brier: p_yes={p_yes_at_trade:.4f} outcome={outcome_binary} "
                f"sq_err={brier_contrib:.4f} running_brier={self.brier_score:.4f} "
                f"(n={self._brier_count})"
            )

        # Persist every settlement to the paper trades log for offline analysis.
        settlement_record = {
            "type": "settlement",
            "market_ticker": pos.market_ticker,
            "asset": pos.asset,
            "side": pos.side,
            "won": won,
            "pnl_usd": total_profit,
            "gross_pnl_usd": gross_profit,
            "fees_usd": pos.entry_fees_usd,
            "payout_usd": total_payout,
            "quantity_centicx": pos.total_qty,
            "quantity_contracts": pos.total_qty / 100.0,
            "entry_price_cents": pos.avg_price_cents,
            "payout_cents": payout_cents,
            "profit_per_contract_cents": profit_per_contract,
            "strike": pos.strike,
            "strike_floor": pos.strike_floor,
            "strike_cap": pos.strike_cap,
            "is_range": pos.is_range,
            "final_avg": final_avg,
            "settlement_time": pos.settlement_time,
            "source": pos.source or "",
            "family": pos.family,
            "scenario_profile": pos.scenario_profile,
            "timestamp": time.time(),
            "bot_id": bot_id,
        }
        if pos.decision_context:
            settlement_record["decision_context"] = pos.decision_context
        _append_paper_log(settlement_record)

        # Publish for UI (PnL, win rate, event log)
        outcome = SettlementOutcome(
            market_ticker=pos.market_ticker,
            side=pos.side,
            won=won,
            pnl=total_profit,
            quantity_centicx=pos.total_qty,
            entry_price_cents=pos.avg_price_cents,
            final_avg=final_avg,
            strike=pos.strike,
            timestamp=time.time(),
            gross_pnl=gross_profit,
            fees_usd=pos.entry_fees_usd,
            bot_id=bot_id,
            source=pos.source or "",
            family=pos.family,
            scenario_profile=pos.scenario_profile,
            decision_context=pos.decision_context or {},
        )
        await self._bus.publish("kalshi.settlement_outcome", outcome)

        # Persist to database if available
        if self._db:
            try:
                await self._db.insert_kalshi_outcome(
                    market_ticker=pos.market_ticker,
                    strike=pos.strike,
                    direction=pos.side,
                    entry_price=pos.avg_price_cents / 100.0,
                    exit_price=payout_cents / 100.0,
                    quantity=pos.total_qty,
                    pnl=total_profit,
                    outcome="WON" if won else "LOST",
                    final_avg=final_avg,
                    settled_at_ms=int(time.time() * 1000),
                    is_paper=True,  # TODO: Wire this from config if live trading is active
                    bot_id=bot_id,
                )
            except Exception as e:
                log.error(f"Failed to persist Kalshi outcome: {e}")
