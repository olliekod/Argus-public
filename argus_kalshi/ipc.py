"""
IPC for Kalshi terminal UI in a separate process.

When visualizer_process == "separate", the trading process runs an IPC server
that aggregates bus state and sends JSON snapshots to a connected UI process.
The UI process connects, receives snapshots, and renders without sharing the
trading process event loop — avoiding ping spikes from UI/calculations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .bus import Bus
from .models import (
    AccountBalance,
    FairProbability,
    FillEvent,
    KalshiRtt,
    MarketMetadata,
    OrderbookState,
    OrderUpdate,
    SettlementOutcome,
    WsConnectionEvent,
)

log = logging.getLogger("argus_kalshi.ipc")

# Snapshot send interval (seconds) when a client is connected.
IPC_SNAPSHOT_INTERVAL = 0.2  # 5 Hz for dashboard
# Trim IPC snapshots so the UI does not OOM or hit stream limit (7488 bots + 863 states).
IPC_MAX_BOT_STATS = 1000
IPC_MAX_STATES = 1500
IPC_DRAIN_MAX_META = 3000
IPC_DRAIN_MAX_OB = 6000
IPC_DRAIN_MAX_PROB = 6000

SCALP_BEST_RELIEF_USD_PER_CONTRACT = 0.005
HOLD_BEST_RELIEF_USD_PER_CONTRACT = 0.001
SCALP_STRESS_DRAG_USD_PER_CONTRACT = 0.02
HOLD_STRESS_DRAG_USD_PER_CONTRACT = 0.005


def _source_bucket(source: str) -> str:
    src = (source or "").strip()
    if src == "mispricing_scalp":
        return "scalp"
    if src == "pair_arb":
        return "arb"
    return "expiry"


def _event_bot_id(event: Any) -> Optional[str]:
    raw = getattr(event, "bot_id", None)
    if not isinstance(raw, str):
        return None
    val = raw.strip()
    return val or None


class StateAggregator:
    """Subscribes to bus topics and maintains a JSON-serializable UI snapshot."""

    def __init__(self, bus: Bus, primary_bot_id: Optional[str] = None) -> None:
        self._bus = bus
        _pid = (primary_bot_id or "").strip()
        self._primary_bot_id = _pid if _pid and _pid.lower() != "default" else None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._subs: Optional[Dict[str, asyncio.Queue]] = None  # set in start() for drain_for_snapshot

        # State mirroring TerminalVisualizer internals
        self._prices: Dict[str, float] = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}
        self._price_hist: Dict[str, deque] = {
            "BTC": deque(maxlen=24),
            "ETH": deque(maxlen=24),
            "SOL": deque(maxlen=24),
        }
        self._states: Dict[str, Dict[str, Any]] = {}  # ticker -> state dict
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._balance_usd: Optional[float] = None
        self._initial_balance_usd: Optional[float] = None
        self._kalshi_rtt_ms: Optional[float] = None
        self._kalshi_rtt_source: Optional[str] = None  # "rest" | "ws"
        self._ws_connected = False
        self._selected_tickers: set[str] = set()
        self._session_pnl = 0.0
        self._alltime_pnl = 0.0
        self._primary_session_peak_pnl = 0.0
        self._wins = 0
        self._losses = 0
        self._win_pnl_total = 0.0
        self._loss_pnl_total = 0.0
        self._win_streak = 0
        self._loss_streak = 0
        self._best_win = 0.0
        self._worst_loss = 0.0
        self._total_contracts = 0
        self._open_orders = 0
        self._recent_fills: deque = deque(maxlen=500)
        self._history: deque = deque(maxlen=500)
        self._bot_stats: Dict[str, Dict[str, Any]] = {}
        # Per-market-bucket ACTIVE order counts (kept under legacy snapshot key
        # `market_fill_counts` for wire compatibility with UI clients).
        self._market_fill_counts: Dict[str, int] = {}
        self._open_order_bucket_by_oid: Dict[str, str] = {}
        self._empty_bot_stats = {
            "pnl": 0.0,
            "pnl_e": 0.0,  # expiry / hold-to-settlement PnL
            "pnl_s": 0.0,  # scalp / mispricing_scalp PnL
            "pnl_a": 0.0,  # pair-arb PnL
            "gross_pnl": 0.0,
            "fees_usd": 0.0,
            "qty_e_contracts": 0.0,
            "qty_s_contracts": 0.0,
            "qty_a_contracts": 0.0,
            "wins": 0,
            "losses": 0,
            "fills": 0,
            "fills_e": 0,
            "fills_s": 0,
            "fills_a": 0,
            "buy_contracts": 0.0,  # cumulative contracts bought across all markets
            "orders": 0,
            "trade_count": 0, "gross_profit": 0.0, "gross_loss": 0.0,
            "peak_pnl": 0.0, "max_drawdown": 0.0, "last_active": 0.0,
            "tail_loss_10pct": 0.0,
            "generation": 0, "run_id": "", "parent_run_id": "",
            "family_pnl": {},
            "start_equity": 5000.0, "scenario": "base",
        }
        self._bot_stats_overlay_provider: Optional[Any] = None
        self._population_overlay_provider: Optional[Any] = None

    async def start(self, run_aggregate_loop: bool = True) -> None:
        """Start aggregator. When run_aggregate_loop=False (separate UI), only subscriptions
        are set; the IPC loop is the sole consumer and drains via drain_for_snapshot().
        This avoids two tasks competing for the same queues and keeps UI data path clear.
        """
        self._running = True
        subs = {
            "btc": await self._bus.subscribe("btc.mid_price"),
            "eth": await self._bus.subscribe("eth.mid_price"),
            "sol": await self._bus.subscribe("sol.mid_price"),
            "prob": await self._bus.subscribe("kalshi.fair_prob"),
            "meta": await self._bus.subscribe("kalshi.market_metadata"),
            "selected": await self._bus.subscribe("kalshi.selected_markets"),
            "outcome": await self._bus.subscribe("kalshi.settlement_outcome"),
            "ob": await self._bus.subscribe("kalshi.orderbook"),
            "fills": await self._bus.subscribe("kalshi.fills"),
            "orders": await self._bus.subscribe("kalshi.user_orders"),
            "balance": await self._bus.subscribe("kalshi.account_balance"),
            "ws": await self._bus.subscribe("kalshi.ws.status"),
            "rtt": await self._bus.subscribe("kalshi.rtt"),
        }
        self._subs = subs
        if run_aggregate_loop:
            self._task = asyncio.create_task(self._aggregate_loop(subs))
        else:
            self._task = None  # IPC loop is sole consumer; drain_for_snapshot() only

    def set_bot_stats_overlay_provider(self, provider: Optional[Any]) -> None:
        """Set callable returning per-bot overlay dict merged into bot_stats."""
        self._bot_stats_overlay_provider = provider

    def set_population_overlay_provider(self, provider: Optional[Any]) -> None:
        """Set callable returning global population diagnostics for snapshot root."""
        self._population_overlay_provider = provider

    def _merge_bot_stats_overlay(self) -> None:
        provider = self._bot_stats_overlay_provider
        if provider is None:
            return
        try:
            overlay = provider() or {}
        except Exception:
            return
        if not isinstance(overlay, dict):
            return
        for bot_id, extra in overlay.items():
            if not isinstance(extra, dict):
                continue
            base = self._bot_stats.setdefault(bot_id, dict(self._empty_bot_stats))
            merged = dict(extra)
            # Overlay providers may expose session-scoped counters (e.g., trade_count)
            # while base stats can include persisted all-time counters from JSONL.
            # Never let overlay reduce cumulative counters, or WR can exceed 100%.
            for k in ("trade_count", "wins", "losses", "fills", "fills_e", "fills_s", "fills_a"):
                if k in merged:
                    try:
                        merged[k] = max(int(base.get(k, 0)), int(merged[k]))
                    except Exception:
                        merged[k] = base.get(k, merged[k])
            base.update(merged)

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _aggregate_loop(self, subs: Dict[str, asyncio.Queue]) -> None:
        try:
            while self._running:
                # Drain queues with limits to avoid blocking too long
                for _ in range(50):
                    # Control and price first so header prices stay fresh when ob/prob are flooded.
                    if not subs["ws"].empty():
                        msg = subs["ws"].get_nowait()
                        if isinstance(msg, WsConnectionEvent):
                            self._ws_connected = msg.status in ("connected", "reconnecting")
                    if not subs["rtt"].empty():
                        msg = subs["rtt"].get_nowait()
                        if hasattr(msg, "rtt_ms"):
                            self._kalshi_rtt_ms = msg.rtt_ms
                            self._kalshi_rtt_source = getattr(msg, "source", None)
                    if not subs["balance"].empty():
                        msg = subs["balance"].get_nowait()
                        if isinstance(msg, AccountBalance):
                            self._balance_usd = msg.balance_usd
                            if self._initial_balance_usd is None:
                                self._initial_balance_usd = msg.balance_usd
                    if not subs["selected"].empty():
                        msg = subs["selected"].get_nowait()
                        tickers = getattr(msg, "tickers", None)
                        if isinstance(tickers, list):
                            self._apply_selected_tickers(tickers)
                    for key in ["btc", "eth", "sol"]:
                        if not subs[key].empty():
                            msg = subs[key].get_nowait()
                            if hasattr(msg, "price"):
                                self._prices[key.upper()] = msg.price
                    # Then ob / prob / meta (can be flooded in production).
                    if not subs["ob"].empty():
                        msg = subs["ob"].get_nowait()
                        if isinstance(msg, OrderbookState):
                            t = msg.market_ticker
                            if t not in self._states:
                                self._states[t] = self._state_dict_for_ticker(t)
                            s = self._states[t]
                            s["ob_valid"] = msg.valid
                            if msg.valid:
                                s["ob_had_valid"] = True
                                s["yes_ask"] = msg.implied_yes_ask_cents
                                s["no_ask"] = msg.implied_no_ask_cents
                                s["yes_bid"] = msg.best_yes_bid_cents
                                s["no_bid"] = msg.best_no_bid_cents
                    if not subs["prob"].empty():
                        msg = subs["prob"].get_nowait()
                        if isinstance(msg, FairProbability):
                            t = msg.market_ticker
                            if t not in self._states:
                                self._states[t] = self._state_dict_for_ticker(t)
                            self._states[t]["p_yes"] = msg.p_yes
                            hist = self._states[t].setdefault("p_yes_hist", [])
                            hist.append(msg.p_yes)
                            if len(hist) > 20:
                                self._states[t]["p_yes_hist"] = hist[-20:]
                    if not subs["meta"].empty():
                        msg = subs["meta"].get_nowait()
                        if isinstance(msg, MarketMetadata):
                            self._metadata[msg.market_ticker] = asdict(msg)
                            if msg.market_ticker not in self._states:
                                self._states[msg.market_ticker] = self._state_dict_from_meta(msg)
                    # Outcomes
                    for _ in range(100):
                        if subs["outcome"].empty():
                            break
                        out = subs["outcome"].get_nowait()
                        if not isinstance(out, SettlementOutcome):
                            continue
                        bot_id = _event_bot_id(out) or "default"
                        b = self._bot_stats.setdefault(bot_id, dict(self._empty_bot_stats))
                        b["pnl"] = b.get("pnl", 0) + out.pnl
                        b["gross_pnl"] = b.get("gross_pnl", 0.0) + float(getattr(out, "gross_pnl", out.pnl))
                        b["fees_usd"] = b.get("fees_usd", 0.0) + float(getattr(out, "fees_usd", 0.0))
                        src = getattr(out, "source", "") or ""
                        family = getattr(out, "family", "") or ""
                        scenario_profile = getattr(out, "scenario_profile", "") or ""
                        qty_contracts = float(getattr(out, "quantity_centicx", 0)) / 100.0
                        bucket = _source_bucket(src)
                        if bucket == "scalp":
                            b["pnl_s"] = b.get("pnl_s", 0) + out.pnl
                            b["qty_s_contracts"] = b.get("qty_s_contracts", 0.0) + qty_contracts
                        elif bucket == "arb":
                            b["pnl_a"] = b.get("pnl_a", 0) + out.pnl
                            b["qty_a_contracts"] = b.get("qty_a_contracts", 0.0) + qty_contracts
                        else:
                            b["pnl_e"] = b.get("pnl_e", 0) + out.pnl
                            b["qty_e_contracts"] = b.get("qty_e_contracts", 0.0) + qty_contracts
                        b["trade_count"] = b.get("trade_count", 0) + 1
                        if family:
                            fp = b.get("family_pnl", {}) or {}
                            fp[family] = float(fp.get(family, 0.0)) + float(out.pnl)
                            b["family_pnl"] = fp
                        if scenario_profile:
                            b["scenario"] = scenario_profile
                        if out.pnl >= 0:
                            b["gross_profit"] = b.get("gross_profit", 0) + out.pnl
                        else:
                            b["gross_loss"] = b.get("gross_loss", 0) + abs(out.pnl)
                        b["peak_pnl"] = max(b.get("peak_pnl", 0), b.get("pnl", 0))
                        b["max_drawdown"] = max(b.get("max_drawdown", 0), b.get("peak_pnl", 0) - b.get("pnl", 0))
                        b["last_active"] = time.time()
                        if out.won:
                            b["wins"] = b.get("wins", 0) + 1
                        else:
                            b["losses"] = b.get("losses", 0) + 1
                        if self._primary_bot_id and bot_id == self._primary_bot_id:
                            self._session_pnl += out.pnl
                            self._primary_session_peak_pnl = max(self._primary_session_peak_pnl, self._session_pnl)
                            self._alltime_pnl += out.pnl
                            if out.won:
                                self._wins += 1
                                self._win_pnl_total += out.pnl
                                self._best_win = max(self._best_win, out.pnl)
                            else:
                                self._losses += 1
                                self._loss_pnl_total += out.pnl
                                self._worst_loss = min(self._worst_loss, out.pnl)
                            self._history.appendleft({
                                "won": out.won, "ticker": out.market_ticker,
                                "pnl": out.pnl, "side": out.side, "source": getattr(out, "source", "") or ""
                            })
                    # Fills
                    for _ in range(200):
                        if subs["fills"].empty():
                            break
                        fill = subs["fills"].get_nowait()
                        if not isinstance(fill, FillEvent):
                            continue
                        bot_id = _event_bot_id(fill) or "default"
                        b = self._bot_stats.setdefault(bot_id, dict(self._empty_bot_stats))
                        b["fills"] = b.get("fills", 0) + 1
                        bucket = _source_bucket(getattr(fill, "source", "") or "")
                        if bucket == "scalp":
                            b["fills_s"] = b.get("fills_s", 0) + 1
                        elif bucket == "arb":
                            b["fills_a"] = b.get("fills_a", 0) + 1
                        else:
                            b["fills_e"] = b.get("fills_e", 0) + 1
                        if (getattr(fill, "action", "buy") or "buy") == "buy":
                            b["buy_contracts"] = b.get("buy_contracts", 0.0) + (float(fill.count) / 100.0)
                        b["last_active"] = time.time()
                        if self._primary_bot_id and bot_id == self._primary_bot_id:
                            self._recent_fills.appendleft({
                                "ticker": fill.market_ticker, "side": fill.side,
                                "price": fill.price_cents, "count": fill.count // 100,
                                "ts": fill.timestamp, "source": getattr(fill, "source", "")
                            })
                    # Orders
                    for _ in range(100):
                        if subs["orders"].empty():
                            break
                        ou = subs["orders"].get_nowait()
                        if not isinstance(ou, OrderUpdate):
                            continue
                        bot_id = _event_bot_id(ou) or "default"
                        b = self._bot_stats.setdefault(bot_id, dict(self._empty_bot_stats))
                        b["orders"] = b.get("orders", 0) + 1
                        b["last_active"] = time.time()
                        self._apply_order_bucket_update(ou)
                        if self._primary_bot_id and bot_id == self._primary_bot_id:
                            if ou.status == "placed":
                                self._open_orders += 1
                            elif ou.status in ("filled", "partial_fill", "cancelled", "canceled", "error"):
                                self._open_orders = max(0, self._open_orders - 1)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    def _state_dict_for_ticker(self, ticker: str) -> Dict[str, Any]:
        meta = self._metadata.get(ticker, {})
        return self._state_dict_from_meta_dict(ticker, meta)

    def _state_dict_from_meta(self, meta: MarketMetadata) -> Dict[str, Any]:
        return self._state_dict_from_meta_dict(meta.market_ticker, asdict(meta))

    def _state_dict_from_meta_dict(self, ticker: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        exp_ts = 0.0
        try:
            from datetime import datetime, timezone
            iso = meta.get("settlement_time_iso", "").replace("Z", "+00:00")
            exp_ts = datetime.fromisoformat(iso).timestamp()
        except Exception:
            pass
        return {
            "ticker": ticker,
            "asset": meta.get("asset", "BTC"),
            "p_yes": 0.5,
            "p_yes_hist": list(),
            "yes_ask": 0, "no_ask": 0, "yes_bid": 0, "no_bid": 0,
            "ob_valid": False, "ob_had_valid": False,
            "exp_ts": exp_ts,
            "window_min": meta.get("window_minutes", 15),
            "is_range": meta.get("is_range", False),
            "strike": meta.get("strike_price", 0.0),
            "strike_floor": meta.get("strike_floor"),
            "strike_cap": meta.get("strike_cap"),
        }

    def _apply_selected_tickers(self, tickers: List[str]) -> None:
        selected = set(tickers)
        self._selected_tickers = selected
        if not selected:
            return
        self._states = {t: s for t, s in self._states.items() if t in selected}
        self._metadata = {t: m for t, m in self._metadata.items() if t in selected}

    def _market_bucket_key(self, ticker: str) -> Optional[str]:
        # Per-ticker open-order attribution so UI "Ord" matches the exact row.
        return ticker or None

    def _apply_order_bucket_update(self, ou: OrderUpdate) -> None:
        """Track active open order counts by market bucket.

        We increment on `placed` and decrement when an order reaches a terminal
        status (`filled`, `partial_fill`, `cancelled`/`canceled`, `error`).
        """
        order_id = (ou.order_id or ou.client_order_id or "").strip()
        status = (ou.status or "").strip().lower()
        if not order_id or not status:
            return

        if status == "placed":
            bucket = self._market_bucket_key(ou.market_ticker)
            if not bucket:
                return
            if order_id in self._open_order_bucket_by_oid:
                return
            self._open_order_bucket_by_oid[order_id] = bucket
            self._market_fill_counts[bucket] = self._market_fill_counts.get(bucket, 0) + 1
            return

        if status in {"filled", "partial_fill", "cancelled", "canceled", "error"}:
            bucket = self._open_order_bucket_by_oid.pop(order_id, None)
            if not bucket:
                return
            cur = self._market_fill_counts.get(bucket, 0)
            if cur <= 1:
                self._market_fill_counts.pop(bucket, None)
            else:
                self._market_fill_counts[bucket] = cur - 1

    def seed_from_jsonl(
        self,
        bot_stats: Dict[str, Dict[str, Any]],
        primary_pnl: float = 0.0,
        primary_wins: int = 0,
        primary_losses: int = 0,
        primary_win_pnl: float = 0.0,
        primary_loss_pnl: float = 0.0,
        primary_best_win: float = 0.0,
        primary_worst_loss: float = 0.0,
    ) -> None:
        """Seed aggregator from paper_trades.jsonl so PnL is correct after restart (no 0 → inflated jump)."""
        self._bot_stats = {k: dict(v) for k, v in bot_stats.items()}
        self._alltime_pnl = primary_pnl
        self._session_pnl = 0.0
        self._primary_session_peak_pnl = 0.0
        self._wins = primary_wins
        self._losses = primary_losses
        self._win_pnl_total = primary_win_pnl
        self._loss_pnl_total = primary_loss_pnl
        self._best_win = primary_best_win
        self._worst_loss = primary_worst_loss

    def ensure_bot_stats_entries(self, bot_ids: List[str]) -> None:
        """Pre-seed empty stats for all known bot_ids so leaderboard shows dwarf names before they trade."""
        for bid in bot_ids:
            if bid and bid not in self._bot_stats:
                self._bot_stats[bid] = dict(self._empty_bot_stats)

    @staticmethod
    def _scenario_projection(stats: Dict[str, Any]) -> Dict[str, float]:
        pnl = float(stats.get("pnl", 0.0))
        qty_s = float(stats.get("qty_s_contracts", 0.0))
        qty_a = float(stats.get("qty_a_contracts", 0.0))
        qty_e = float(stats.get("qty_e_contracts", 0.0))
        scalp_like_qty = qty_s + qty_a
        best = pnl + (scalp_like_qty * SCALP_BEST_RELIEF_USD_PER_CONTRACT) + (qty_e * HOLD_BEST_RELIEF_USD_PER_CONTRACT)
        stress = pnl - (scalp_like_qty * SCALP_STRESS_DRAG_USD_PER_CONTRACT) - (qty_e * HOLD_STRESS_DRAG_USD_PER_CONTRACT)
        return {"best": best, "base": pnl, "stress": stress}

    @classmethod
    def _robust_score(cls, stats: Dict[str, Any]) -> float:
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total = max(0, stats.get("trade_count", wins + losses))
        scenarios = cls._scenario_projection(stats)
        worst_case = min(scenarios["best"], scenarios["base"], scenarios["stress"])
        base_case = scenarios["base"]
        fragility = max(0.0, scenarios["best"] - scenarios["stress"])
        dd = max(0.0, float(stats.get("max_drawdown", 0.0)))
        wins_rate = (wins / total) if total > 0 else 0.0
        significance = max(0.0, min(1.0, total / 25.0))
        worst_score = max(-1.0, min(1.0, worst_case / 400.0))
        base_score = max(-1.0, min(1.0, base_case / 400.0))
        dd_score = 1.0 - max(0.0, min(1.0, dd / 300.0))
        win_score = max(0.0, min(1.0, (wins_rate - 0.35) / 0.40))
        fragility_penalty = max(0.0, min(1.0, fragility / 250.0))
        composite = 100.0 * (
            0.45 * worst_score +
            0.20 * base_score +
            0.15 * dd_score +
            0.10 * win_score +
            0.10 * significance
        )
        composite *= (1.0 - 0.35 * fragility_penalty)
        if total < 5:
            composite *= 0.15 + (0.85 * (total / 5.0))
        return round(composite, 2)

    def get_snapshot(
        self,
        max_bot_stats: Optional[int] = None,
        max_states: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot for the UI process.

        When max_bot_stats / max_states are set (e.g. for IPC), trim to avoid
        huge payloads that can crash the UI (OOM or stream limit).
        """
        self._merge_bot_stats_overlay()
        states_items = list(self._states.items())
        total_states = len(states_items)
        total_metadata = len(self._metadata)
        if self._selected_tickers:
            states_items = [(t, s) for t, s in states_items if t in self._selected_tickers]
        # Prefer recently-updated states when we must trim so UI shows active markets.
        states_items.sort(
            key=lambda kv: (
                kv[1].get("_last_update_ts", 0.0),
                kv[1].get("ob_valid", False),
                kv[1].get("ob_had_valid", False),
            ),
            reverse=True,
        )
        if max_states is not None and len(states_items) > max_states:
            states_items = states_items[: max_states]
        states_ser: Dict[str, Dict[str, Any]] = {}
        for t, s in states_items:
            s2 = dict(s)
            s2.pop("_last_update_ts", None)
            hist = s2.get("p_yes_hist") or []
            s2["p_yes_hist"] = hist[-20:] if len(hist) > 20 else list(hist)
            states_ser[t] = s2

        bot_stats_items = list(self._bot_stats.items())
        bot_stats_items.sort(
            key=lambda kv: (
                self._robust_score(kv[1]),
                float(kv[1].get("pnl", 0.0)),
                float(kv[1].get("last_active", 0.0)),
            ),
            reverse=True,
        )
        if max_bot_stats is not None and len(bot_stats_items) > max_bot_stats:
            primary_item = None
            if self._primary_bot_id:
                for item in bot_stats_items:
                    if item[0] == self._primary_bot_id:
                        primary_item = item
                        break
            # Keep cap unchanged, but reserve last 5 slots for global worst bots
            # so UI can render true bottom-5 across the full farm universe.
            if max_bot_stats > 10:
                worst_n = 5
                top_n = max(0, max_bot_stats - worst_n)
                top_slice = bot_stats_items[:top_n]
                bottom_slice = bot_stats_items[-worst_n:]
                merged: List[tuple[str, Dict[str, Any]]] = []
                seen: set[str] = set()
                for bid, stats in top_slice + bottom_slice:
                    if bid in seen:
                        continue
                    merged.append((bid, stats))
                    seen.add(bid)
                bot_stats_items = merged[:max_bot_stats]
            else:
                bot_stats_items = bot_stats_items[: max_bot_stats]
            if primary_item is not None and all(bid != self._primary_bot_id for bid, _ in bot_stats_items):
                bot_stats_items[-1] = primary_item
        bot_stats_ser = {}
        for k, v in bot_stats_items:
            item = dict(v)
            item["robust_score"] = self._robust_score(v)
            item["scenario_pnl"] = self._scenario_projection(v)
            bot_stats_ser[k] = item

        # Metadata only for states we send
        meta_ser = {t: self._metadata[t] for t in states_ser if t in self._metadata}
        ob_valid_total = sum(1 for _, s in self._states.items() if s.get("ob_valid"))
        prob_set_total = sum(1 for _, s in self._states.items() if s.get("p_yes", 0.5) != 0.5)

        population_overlay: Dict[str, Any] = {}
        if self._population_overlay_provider is not None:
            try:
                raw_population = self._population_overlay_provider() or {}
                if isinstance(raw_population, dict):
                    population_overlay = raw_population
            except Exception:
                population_overlay = {}

        return {
            "primary_bot_id": self._primary_bot_id,
            "prices": dict(self._prices),
            "_spark": {k: list(v) for k, v in self._price_hist.items()},
            "states": states_ser,
            "metadata": meta_ser,
            "states_total": total_states,
            "states_shown": len(states_ser),
            "metadata_total": total_metadata,
            "ob_valid_total": ob_valid_total,
            "prob_set_total": prob_set_total,
            "balance_usd": self._balance_usd,
            "initial_balance_usd": self._initial_balance_usd,
            "kalshi_rtt_ms": self._kalshi_rtt_ms,
            "kalshi_rtt_source": self._kalshi_rtt_source,
            "ws_connected": self._ws_connected,
            "session_pnl": self._session_pnl,
            "alltime_pnl": self._alltime_pnl,
            "primary_session_peak_pnl": self._primary_session_peak_pnl,
            "wins": self._wins,
            "losses": self._losses,
            "win_pnl_total": self._win_pnl_total,
            "loss_pnl_total": self._loss_pnl_total,
            "win_streak": self._win_streak,
            "loss_streak": self._loss_streak,
            "best_win": self._best_win,
            "worst_loss": self._worst_loss,
            "total_contracts": self._total_contracts,
            "open_orders": self._open_orders,
            "recent_fills": list(self._recent_fills),
            "history": list(self._history),
            "bot_stats": bot_stats_ser,
            "market_fill_counts": dict(self._market_fill_counts),
            "population": population_overlay,
            "ts": time.time(),
        }

    def drain_for_snapshot(
        self,
        max_meta: int = IPC_DRAIN_MAX_META,
        max_ob: int = IPC_DRAIN_MAX_OB,
        max_prob: int = IPC_DRAIN_MAX_PROB,
    ) -> None:
        """Drain control, prices, metadata, and ob/prob so the next get_snapshot()
        is fresh. Call from the IPC send loop every interval so the separate UI gets updates
        even when the aggregator task is starved by the farm.
        In separate-UI mode we do NOT run _aggregate_loop, so we must drain metadata here
        or states have no asset/window (BTC/ETH/SOL rows never appear). Drain limits must
        be high enough to cover all subscribed tickers (~60+) and catch up when the bus is busy.
        """
        subs = self._subs
        if not subs:
            return

        def _budget(q: asyncio.Queue, floor: int, ceiling: int) -> int:
            # Queue sizes are a useful proxy for burst pressure in separate-UI mode.
            # Drain at least *floor*, and opportunistically more when behind.
            return max(floor, min(ceiling, q.qsize()))

        meta_budget = _budget(subs["meta"], max_meta, max_meta * 4)
        ob_budget = _budget(subs["ob"], max_ob, max_ob * 4)
        prob_budget = _budget(subs["prob"], max_prob, max_prob * 4)

        # Control (header: ws, rtt, balance)
        while not subs["ws"].empty():
            msg = subs["ws"].get_nowait()
            if isinstance(msg, WsConnectionEvent):
                self._ws_connected = msg.status in ("connected", "reconnecting")
        while not subs["rtt"].empty():
            msg = subs["rtt"].get_nowait()
            if hasattr(msg, "rtt_ms"):
                self._kalshi_rtt_ms = msg.rtt_ms
                self._kalshi_rtt_source = getattr(msg, "source", None)
        while not subs["balance"].empty():
            msg = subs["balance"].get_nowait()
            if isinstance(msg, AccountBalance):
                self._balance_usd = msg.balance_usd
                if self._initial_balance_usd is None:
                    self._initial_balance_usd = msg.balance_usd
        while not subs["selected"].empty():
            try:
                msg = subs["selected"].get_nowait()
            except asyncio.QueueEmpty:
                break
            tickers = getattr(msg, "tickers", None)
            if isinstance(tickers, list):
                self._apply_selected_tickers(tickers)
        # Prices (header BTC/ETH/SOL)
        for key in ("btc", "eth", "sol"):
            q = subs[key]
            while not q.empty():
                try:
                    msg = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if hasattr(msg, "price"):
                    asset = key.upper()
                    self._prices[asset] = msg.price
                    if msg.price > 0:
                        self._price_hist[asset].append(msg.price)
        # Metadata: must drain so states have asset/window_min (BTC vs ETH vs SOL, 15m vs 60m).
        # Without this, separate-UI mode never gets metadata and all states default to BTC/15m.
        for _ in range(meta_budget):
            if subs["meta"].empty():
                break
            try:
                msg = subs["meta"].get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(msg, MarketMetadata):
                self._metadata[msg.market_ticker] = asdict(msg)
                if msg.market_ticker not in self._states:
                    self._states[msg.market_ticker] = self._state_dict_from_meta(msg)
                else:
                    s = self._states[msg.market_ticker]
                    s["asset"] = msg.asset
                    s["window_min"] = msg.window_minutes
                    s["is_range"] = msg.is_range
                    s["strike"] = msg.strike_price
                    s["strike_floor"] = msg.strike_floor
                    s["strike_cap"] = msg.strike_cap
                    if msg.settlement_time_iso:
                        try:
                            from datetime import datetime

                            iso = msg.settlement_time_iso.replace("Z", "+00:00")
                            s["exp_ts"] = datetime.fromisoformat(iso).timestamp()
                        except Exception:
                            pass
                    s["_last_update_ts"] = time.time()
        # Ob/prob: drain enough to cover all subscribed tickers and keep ob_valid/ask/edge fresh.
        for _ in range(ob_budget):
            if subs["ob"].empty():
                break
            try:
                msg = subs["ob"].get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(msg, OrderbookState):
                t = msg.market_ticker
                if t not in self._states:
                    self._states[t] = self._state_dict_for_ticker(t)
                s = self._states[t]
                s["ob_valid"] = msg.valid
                s["_last_update_ts"] = time.time()
                if msg.valid:
                    s["ob_had_valid"] = True
                    s["yes_ask"] = msg.implied_yes_ask_cents
                    s["no_ask"] = msg.implied_no_ask_cents
                    s["yes_bid"] = msg.best_yes_bid_cents
                    s["no_bid"] = msg.best_no_bid_cents
        for _ in range(prob_budget):
            if subs["prob"].empty():
                break
            try:
                msg = subs["prob"].get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(msg, FairProbability):
                t = msg.market_ticker
                if t not in self._states:
                    self._states[t] = self._state_dict_for_ticker(t)
                self._states[t]["p_yes"] = msg.p_yes
                self._states[t]["_last_update_ts"] = time.time()
                hist = self._states[t].setdefault("p_yes_hist", [])
                hist.append(msg.p_yes)
                if len(hist) > 20:
                    self._states[t]["p_yes_hist"] = hist[-20:]

        # Bot activity: required for dashboard leaderboard, fills, trades, and promoted stats.
        for _ in range(400):
            if subs["outcome"].empty():
                break
            try:
                out = subs["outcome"].get_nowait()
            except asyncio.QueueEmpty:
                break
            if not isinstance(out, SettlementOutcome):
                continue
            bot_id = _event_bot_id(out) or "default"
            b = self._bot_stats.setdefault(bot_id, dict(self._empty_bot_stats))
            b["pnl"] = b.get("pnl", 0.0) + out.pnl
            b["gross_pnl"] = b.get("gross_pnl", 0.0) + float(getattr(out, "gross_pnl", out.pnl))
            b["fees_usd"] = b.get("fees_usd", 0.0) + float(getattr(out, "fees_usd", 0.0))
            src = getattr(out, "source", "") or ""
            family = getattr(out, "family", "") or ""
            scenario_profile = getattr(out, "scenario_profile", "") or ""
            qty_contracts = float(getattr(out, "quantity_centicx", 0)) / 100.0
            bucket = _source_bucket(src)
            if bucket == "scalp":
                b["pnl_s"] = b.get("pnl_s", 0.0) + out.pnl
                b["qty_s_contracts"] = b.get("qty_s_contracts", 0.0) + qty_contracts
            elif bucket == "arb":
                b["pnl_a"] = b.get("pnl_a", 0.0) + out.pnl
                b["qty_a_contracts"] = b.get("qty_a_contracts", 0.0) + qty_contracts
            else:
                b["pnl_e"] = b.get("pnl_e", 0.0) + out.pnl
                b["qty_e_contracts"] = b.get("qty_e_contracts", 0.0) + qty_contracts
            b["trade_count"] = b.get("trade_count", 0) + 1
            if family:
                fp = b.get("family_pnl", {}) or {}
                fp[family] = float(fp.get(family, 0.0)) + float(out.pnl)
                b["family_pnl"] = fp
            if scenario_profile:
                b["scenario"] = scenario_profile
            if out.pnl >= 0:
                b["gross_profit"] = b.get("gross_profit", 0.0) + out.pnl
            else:
                b["gross_loss"] = b.get("gross_loss", 0.0) + abs(out.pnl)
            b["peak_pnl"] = max(b.get("peak_pnl", 0.0), b.get("pnl", 0.0))
            b["max_drawdown"] = max(b.get("max_drawdown", 0.0), b.get("peak_pnl", 0.0) - b.get("pnl", 0.0))
            b["last_active"] = time.time()
            if out.won:
                b["wins"] = b.get("wins", 0) + 1
            else:
                b["losses"] = b.get("losses", 0) + 1
            if self._primary_bot_id and bot_id == self._primary_bot_id:
                self._session_pnl += out.pnl
                self._primary_session_peak_pnl = max(self._primary_session_peak_pnl, self._session_pnl)
                self._alltime_pnl += out.pnl
                if out.won:
                    self._wins += 1
                    self._win_pnl_total += out.pnl
                    self._best_win = max(self._best_win, out.pnl)
                else:
                    self._losses += 1
                    self._loss_pnl_total += out.pnl
                    self._worst_loss = min(self._worst_loss, out.pnl)
                self._history.appendleft({
                    "won": out.won,
                    "ticker": out.market_ticker,
                    "pnl": out.pnl,
                    "side": out.side,
                    "source": src,
                    "bot_id": bot_id,
                })

        for _ in range(800):
            if subs["fills"].empty():
                break
            try:
                fill = subs["fills"].get_nowait()
            except asyncio.QueueEmpty:
                break
            if not isinstance(fill, FillEvent):
                continue
            bot_id = _event_bot_id(fill) or "default"
            b = self._bot_stats.setdefault(bot_id, dict(self._empty_bot_stats))
            b["fills"] = b.get("fills", 0) + 1
            bucket = _source_bucket(getattr(fill, "source", "") or "")
            if bucket == "scalp":
                b["fills_s"] = b.get("fills_s", 0) + 1
            elif bucket == "arb":
                b["fills_a"] = b.get("fills_a", 0) + 1
            else:
                b["fills_e"] = b.get("fills_e", 0) + 1
            if (getattr(fill, "action", "buy") or "buy") == "buy":
                b["buy_contracts"] = b.get("buy_contracts", 0.0) + (float(fill.count) / 100.0)
            b["last_active"] = time.time()
            if self._primary_bot_id and bot_id == self._primary_bot_id:
                self._recent_fills.appendleft({
                    "ticker": fill.market_ticker,
                    "side": fill.side,
                    "price": fill.price_cents,
                    "count": fill.count // 100,
                    "ts": fill.timestamp,
                    "source": getattr(fill, "source", "") or "",
                    "bot_id": bot_id,
                })
                self._total_contracts += fill.count // 100

        for _ in range(400):
            if subs["orders"].empty():
                break
            try:
                ou = subs["orders"].get_nowait()
            except asyncio.QueueEmpty:
                break
            if not isinstance(ou, OrderUpdate):
                continue
            bot_id = _event_bot_id(ou) or "default"
            b = self._bot_stats.setdefault(bot_id, dict(self._empty_bot_stats))
            b["orders"] = b.get("orders", 0) + 1
            b["last_active"] = time.time()
            self._apply_order_bucket_update(ou)
            if self._primary_bot_id and bot_id == self._primary_bot_id:
                if ou.status == "placed":
                    self._open_orders += 1
                elif ou.status in ("filled", "partial_fill", "cancelled", "canceled", "error"):
                    self._open_orders = max(0, self._open_orders - 1)

        if self._selected_tickers:
            self._states = {t: s for t, s in self._states.items() if t in self._selected_tickers}
            self._metadata = {t: m for t, m in self._metadata.items() if t in self._selected_tickers}


# Diagnostic: log IPC send rate and snapshot summary every N seconds.
_IPC_DIAG_INTERVAL_S = 10.0


async def _broadcast_loop(
    aggregator: StateAggregator,
    client_writer_ref: List[Optional[asyncio.StreamWriter]],
    interval: float,
    on_snapshot: Optional[Any],
) -> None:
    """Single producer: drain once per interval, get snapshot, send to IPC client(s) and optional callback."""
    log.info("IPC broadcast loop started (snapshots every %.1fs)", interval)
    send_count = 0
    last_diag_ts = time.time()
    first_sent_logged = False
    try:
        while True:
            await asyncio.sleep(interval)
            await asyncio.sleep(0)
            aggregator.drain_for_snapshot()
            snapshot = aggregator.get_snapshot(
                max_bot_stats=IPC_MAX_BOT_STATS,
                max_states=IPC_MAX_STATES,
            )
            if on_snapshot is not None:
                try:
                    on_snapshot(snapshot)
                except Exception:
                    pass
            writer = client_writer_ref[0] if client_writer_ref else None
            if writer is not None:
                line = json.dumps(snapshot, separators=(",", ":")) + "\n"
                try:
                    writer.write(line.encode("utf-8"))
                    await writer.drain()
                except (ConnectionResetError, BrokenPipeError, OSError):
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass
                    client_writer_ref[0] = None
            send_count += 1
            if not first_sent_logged:
                first_sent_logged = True
                log.info("IPC first snapshot sent")
            now = time.time()
            if now - last_diag_ts >= _IPC_DIAG_INTERVAL_S:
                prices = snapshot.get("prices") or {}
                states = snapshot.get("states") or {}
                ob_ok = sum(1 for s in states.values() if s.get("ob_valid"))
                prob_ok = sum(1 for s in states.values() if s.get("p_yes", 0.5) != 0.5)
                log.info(
                    "IPC: sent %s snapshots in %.0fs | last snapshot BTC=%s ETH=%s SOL=%s states=%s ob_valid=%s p_yes_set=%s",
                    send_count,
                    now - last_diag_ts,
                    prices.get("BTC"),
                    prices.get("ETH"),
                    prices.get("SOL"),
                    len(states),
                    ob_ok,
                    prob_ok,
                )
                send_count = 0
                last_diag_ts = now
    except asyncio.CancelledError:
        pass


async def ipc_server_start(
    aggregator: StateAggregator,
    host: str,
    port: int,
    on_snapshot: Optional[Any] = None,
) -> asyncio.Server:
    """Start TCP server and single broadcast loop. One snapshot producer feeds IPC client(s) and on_snapshot callback."""
    client_writer_ref: List[Optional[asyncio.StreamWriter]] = [None]

    def accept_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername", "unknown")
        log.info("IPC UI client connected: %s", peer)
        if client_writer_ref[0] is not None:
            try:
                client_writer_ref[0].close()
            except Exception:
                pass
        client_writer_ref[0] = writer

    server = await asyncio.start_server(accept_handler, host, port)
    asyncio.create_task(
        _broadcast_loop(
            aggregator,
            client_writer_ref,
            IPC_SNAPSHOT_INTERVAL,
            on_snapshot,
        )
    )
    log.info("IPC server listening on %s:%s", host, port)
    return server


async def ipc_client_recv_line_async(reader: asyncio.StreamReader) -> Optional[Dict[str, Any]]:
    """Read one newline-delimited JSON line; return parsed dict or None on EOF."""
    try:
        data = await reader.readuntil(b"\n")
    except asyncio.IncompleteReadError:
        return None
    except asyncio.LimitOverrunError:
        log.warning("IPC snapshot line exceeded stream limit; reconnect with larger limit")
        return None
    line = data.decode("utf-8").strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None
