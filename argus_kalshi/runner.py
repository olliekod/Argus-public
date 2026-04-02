# Created by Oliver Meihls

# Entrypoint runner for the Argus Kalshi trading system.
#
# Wires together all modules — market discovery, BTC window engine,
# probability engine, strategy engine, execution engine, and optionally
# the WebSocket feed — then runs the asyncio event loop with structured
# shutdown handling.
#
# Usage (programmatic)::
#
# import asyncio
# from argus_kalshi.runner import run
# asyncio.run(run("config/settings.yaml", "config/secrets.yaml"))
#
# See ``__main__.py`` for the CLI wrapper.

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from .btc_window_engine import BtcWindowEngine
from .bus import Bus
from .config import KalshiConfig, TruthFeedConfig, load_config
from .kalshi_execution import ExecutionEngine
from .kalshi_markets import MarketDiscovery
from .kalshi_probability import ProbabilityEngine
from .kalshi_rest import KalshiRestClient
from .kalshi_strategy import StrategyEngine
from .kalshi_ws import KalshiWebSocket
from .logging_utils import ComponentLogger, StructuredFormatter, setup_logging
from .market_selectors import time_to_settlement_seconds
from .orderbook import OrderBook
from .models import (
    AccountBalance,
    ActiveTicker,
    BtcMidPrice,
    KalshiRtt,
    RiskEvent,
    SelectedMarkets,
    StrategyDecision,
    TerminalEvent,
)
from .settlement_tracker import SettlementTracker
from .terminal_ui import TerminalVisualizer
from .mispricing_scalper import MispricingScalper
from . import ipc as ipc_module

# Import core components from Argus core
try:
    from src.core.database import Database
    from src.core.bus import EventBus
    from src.core.bar_builder import BarBuilder
    from src.core.persistence import PersistenceManager
    from src.connectors.coinbase_ws import CoinbaseWebSocket
    from src.connectors.luzia_rest_fallback import LuziaPollingFeed
    from src.connectors.okx_ws_fallback import OkxWsFallback
except ImportError:
    # Fallback if src is not in path (e.g. running from inside argus_kalshi)
    Database = None
    EventBus = None
    BarBuilder = None
    PersistenceManager = None
    CoinbaseWebSocket = None
    LuziaPollingFeed = None
    OkxWsFallback = None

log = ComponentLogger("runner")

# WS channels to subscribe to for full functionality.
_WS_CHANNELS = [
    "orderbook_delta",
    "trade",
    "ticker",
    "market_lifecycle_v2",
    "fill",
    "user_orders",
    "market_positions",
]

_WS_SELECTION_MAX_TIME_TO_SETTLE_S = 60 * 60

# ── ANSI (Mirroring terminal_ui.py for splash screen) ─────────────────
CYAN    = "\033[38;5;51m"
ORANGE  = "\033[38;5;208m"
AMBER   = "\033[38;5;214m"
WHITE   = "\033[38;5;253m"
GREEN   = "\033[38;5;42m"
RED     = "\033[38;5;196m"
DKGRAY  = "\033[38;5;236m"
BOLD    = "\033[1m"
RESET   = "\033[0m"


def _launch_terminal_ui_client(connect_arg: str) -> subprocess.Popen[Any]:
    # Launch the separate terminal UI in a new console window.
    cwd = os.getcwd()
    if os.name == "nt":
        ui_cols = 112
        ui_rows = 80
        python_exe = str(Path(sys.executable).resolve())
        ps_cwd = cwd.replace("'", "''")
        ps_python = python_exe.replace("'", "''")
        ps_connect = connect_arg.replace("'", "''")
        ps_cmd = (
            "$Host.UI.RawUI.WindowTitle = 'Argus Kalshi UI'; "
            f"$size = New-Object Management.Automation.Host.Size({ui_cols}, {ui_rows}); "
            "$Host.UI.RawUI.BufferSize = $size; "
            "$Host.UI.RawUI.WindowSize = $size; "
            f"Set-Location '{ps_cwd}'; "
            f"& '{ps_python}' -m argus_kalshi --ui-only --connect '{ps_connect}'"
        )
        return subprocess.Popen(
            ["powershell", "-NoExit", "-Command", ps_cmd],
            cwd=cwd,
            close_fds=False,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
    return subprocess.Popen(
        [sys.executable, "-m", "argus_kalshi", "--ui-only", "--connect", connect_arg],
        cwd=cwd,
        close_fds=False,
    )


def _asset_topics_for_cfg(cfg: KalshiConfig) -> List[tuple[str, str]]:
    mapping = {
        "BTC": "btc.mid_price",
        "ETH": "eth.mid_price",
        "SOL": "sol.mid_price",
    }
    assets = [a.upper() for a in cfg.assets]
    return [(mapping[a], a) for a in assets if a in mapping]


def _infer_asset_from_ticker(
    ticker: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    # Best-effort asset extraction for Kalshi market tickers.
    if metadata:
        meta = metadata.get(ticker)
        asset = getattr(meta, "asset", "") if meta is not None else ""
        if asset:
            return str(asset).upper()
    t = (ticker or "").upper()
    if t.startswith("KXBTC"):
        return "BTC"
    if t.startswith("KXETH"):
        return "ETH"
    if t.startswith("KXSOL"):
        return "SOL"
    return ""


def _extract_mid_price(evt: Any) -> Optional[float]:
    if isinstance(evt, (int, float)):
        p = float(evt)
        return p if p > 0 else None
    if isinstance(evt, dict):
        for key in ("mid_price", "price", "value", "last", "close"):
            val = evt.get(key)
            if isinstance(val, (int, float)) and float(val) > 0:
                return float(val)
        return None
    for attr in ("mid_price", "price", "value", "last", "close"):
        val = getattr(evt, attr, None)
        if isinstance(val, (int, float)) and float(val) > 0:
            return float(val)
    return None


def _standalone_classify_vol(
    abs_ret_ema: float,
    vol_baseline_ema: float,
    last_jump_pct: float,
) -> str:
    # Volatility regime classifier tuned to avoid blind over-filtering.
    base = max(vol_baseline_ema, 1e-6)
    rel = abs_ret_ema / base
    if last_jump_pct >= 0.0035 or (abs_ret_ema >= 0.0018 and rel >= 2.8):
        return "VOL_SPIKE"
    if last_jump_pct >= 0.0018 or (abs_ret_ema >= 0.0009 and rel >= 1.8):
        return "VOL_HIGH"
    if abs_ret_ema <= 0.00025 and rel <= 0.8:
        return "VOL_LOW"
    return "VOL_NORMAL"


def _standalone_classify_liq(
    spread_ema: float,
    depth_ema: float,
    ob_idle_s: float,
) -> str:
    # Liquidity regime classifier from top-of-book quality.
    if ob_idle_s >= 12.0 or (spread_ema >= 7.0 and depth_ema < 120.0):
        return "LIQ_DRIED"
    if ob_idle_s >= 6.0 or spread_ema >= 4.0 or depth_ema < 260.0:
        return "LIQ_LOW"
    if spread_ema <= 2.0 and depth_ema >= 700.0:
        return "LIQ_HIGH"
    return "LIQ_NORMAL"


def _standalone_risk_regime(vol_by_asset: Dict[str, str], liq_by_asset: Dict[str, str]) -> str:
    # Market-wide risk regime for gating size caps.
    spike_assets = [a for a, v in vol_by_asset.items() if v == "VOL_SPIKE"]
    low_liq_assets = [a for a, l in liq_by_asset.items() if l in ("LIQ_LOW", "LIQ_DRIED")]
    if spike_assets and low_liq_assets:
        return "RISK_OFF"
    if len(spike_assets) >= 2:
        return "RISK_OFF"
    return "NEUTRAL"


#  YAML / config helpers

def _load_yaml(path: str) -> Dict[str, Any]:
    # Load a YAML file, returning an empty dict if the file is empty.
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


from .farm_runner import load_farm_configs, KalshiPaperFarm

def _build_configs(
    settings: Dict[str, Any],
    secrets: Dict[str, Any],
    settings_path: Optional[str] = None,
) -> List[KalshiConfig]:
    # Merge Kalshi secrets into all loaded configurations.
    configs = load_farm_configs(settings, settings_path=settings_path)
    
    kalshi_secrets = secrets.get("kalshi", {})
    luzia_secrets = secrets.get("luzia", {})
    
    for cfg in configs:
        if "key_id" in kalshi_secrets and not cfg.kalshi_key_id:
            object.__setattr__(cfg, "kalshi_key_id", kalshi_secrets["key_id"])
        if "private_key_path" in kalshi_secrets and not cfg.kalshi_private_key_path:
            object.__setattr__(cfg, "kalshi_private_key_path", kalshi_secrets["private_key_path"])
        if "api_key" in luzia_secrets and not cfg.luzia_api_key:
            object.__setattr__(cfg, "luzia_api_key", luzia_secrets["api_key"])
            
    return configs


def _resolve_truth_feeds(cfg: KalshiConfig) -> List[TruthFeedConfig]:
    if cfg.truth_feeds:
        return list(cfg.truth_feeds)
    # Backward-compatible BTC-only config path.
    return [TruthFeedConfig(asset="BTC", topic=cfg.truth_feed_topic, coinbase_symbol="BTC/USDT", publish_to_core_bus=True)]


#  Mock BTC price feed (dry-run / test only)

class MockBtcFeed:
    # Emits constant ``BtcMidPrice`` messages at 1 Hz.
    #
    # **Only** used when ``dry_run=True`` and no external truth feed is
    # configured.  This lets the window / probability / strategy engines
    # exercise their code paths without a real exchange connection.

    def __init__(
        self,
        bus: Bus,
        *,
        topic: str = "btc.mid_price",
        price: float = 65000.0,
        interval: float = 1.0,
    ) -> None:
        self._bus = bus
        self._topic = topic
        self._price = price
        self._interval = interval
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._emit_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _emit_loop(self) -> None:
        try:
            while True:
                msg = BtcMidPrice(price=self._price, timestamp=time.time(), source="mock", asset="BTC")
                await self._bus.publish(self._topic, msg)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass


async def _standalone_regime_publisher(
    bus: Bus,
    cfg: KalshiConfig,
    discovery: Optional[MarketDiscovery],
) -> None:
    # Publish synthetic regime events in standalone Kalshi mode.
    #
    # Uses truth-feed returns (volatility) + top-of-book quality (liquidity) and
    # emits ``kalshi.regime`` events with lightweight hysteresis so regime gates
    # influence sizing/entries without excessive chattering.
    topics = _asset_topics_for_cfg(cfg)
    if not topics:
        return
    truth_queues: Dict[str, asyncio.Queue] = {}
    for topic, asset in topics:
        truth_queues[asset] = await bus.subscribe(topic)
    q_ob = await bus.subscribe("kalshi.orderbook")

    assets = [asset for _, asset in topics]
    now = time.monotonic()
    state: Dict[str, Dict[str, Any]] = {
        asset: {
            "last_price": 0.0,
            "last_truth_mono": now,
            "abs_ret_ema": 0.0,
            "vol_baseline_ema": 0.0005,
            "last_jump_pct": 0.0,
            "spread_ema": 2.0,
            "depth_ema": 500.0,
            "last_ob_mono": now,
            "vol_regime": "VOL_NORMAL",
            "liq_regime": "LIQ_NORMAL",
            "vol_candidate": "VOL_NORMAL",
            "liq_candidate": "LIQ_NORMAL",
            "vol_votes": 0,
            "liq_votes": 0,
        }
        for asset in assets
    }

    last_log = 0.0
    publish_interval_s = 2.0
    next_emit = time.monotonic() + publish_interval_s
    try:
        while True:
            # Drain truth queues (BTC/ETH/SOL) without blocking.
            for asset, q in truth_queues.items():
                while True:
                    try:
                        evt = q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    price = _extract_mid_price(evt)
                    if price is None or price <= 0:
                        continue
                    s = state[asset]
                    prev = float(s["last_price"])
                    s["last_truth_mono"] = time.monotonic()
                    if prev > 0:
                        ret = abs((price - prev) / prev)
                        s["last_jump_pct"] = ret
                        s["abs_ret_ema"] = (0.22 * ret) + (0.78 * float(s["abs_ret_ema"]))
                        s["vol_baseline_ema"] = (0.04 * ret) + (0.96 * float(s["vol_baseline_ema"]))
                    s["last_price"] = price

            # Drain orderbook queue for liquidity estimates.
            while True:
                try:
                    ob_evt = q_ob.get_nowait()
                except asyncio.QueueEmpty:
                    break
                ticker = getattr(ob_evt, "market_ticker", "") if not isinstance(ob_evt, dict) else str(ob_evt.get("market_ticker", ""))
                asset = _infer_asset_from_ticker(
                    ticker,
                    metadata=getattr(discovery, "metadata", None) if discovery is not None else None,
                )
                if asset not in state:
                    continue
                valid = bool(getattr(ob_evt, "valid", True)) if not isinstance(ob_evt, dict) else bool(ob_evt.get("valid", True))
                if not valid:
                    continue
                yes_bid = int(getattr(ob_evt, "best_yes_bid_cents", 0)) if not isinstance(ob_evt, dict) else int(ob_evt.get("best_yes_bid_cents", 0) or 0)
                no_bid = int(getattr(ob_evt, "best_no_bid_cents", 0)) if not isinstance(ob_evt, dict) else int(ob_evt.get("best_no_bid_cents", 0) or 0)
                yes_ask = int(getattr(ob_evt, "implied_yes_ask_cents", 100)) if not isinstance(ob_evt, dict) else int(ob_evt.get("implied_yes_ask_cents", 100) or 100)
                no_ask = int(getattr(ob_evt, "implied_no_ask_cents", 100)) if not isinstance(ob_evt, dict) else int(ob_evt.get("implied_no_ask_cents", 100) or 100)
                y_depth = int(getattr(ob_evt, "best_yes_depth", 0)) if not isinstance(ob_evt, dict) else int(ob_evt.get("best_yes_depth", 0) or 0)
                n_depth = int(getattr(ob_evt, "best_no_depth", 0)) if not isinstance(ob_evt, dict) else int(ob_evt.get("best_no_depth", 0) or 0)
                spread_yes = max(0, yes_ask - yes_bid)
                spread_no = max(0, no_ask - no_bid)
                spread = (spread_yes + spread_no) / 2.0
                depth = float(min(y_depth, n_depth) if (y_depth > 0 and n_depth > 0) else max(y_depth, n_depth))
                s = state[asset]
                s["last_ob_mono"] = time.monotonic()
                s["spread_ema"] = (0.20 * spread) + (0.80 * float(s["spread_ema"]))
                s["depth_ema"] = (0.20 * depth) + (0.80 * float(s["depth_ema"]))

            now = time.monotonic()
            if now >= next_emit:
                vol_by_asset: Dict[str, str] = {}
                liq_by_asset: Dict[str, str] = {}
                for asset in assets:
                    s = state[asset]
                    vol_candidate = _standalone_classify_vol(
                        abs_ret_ema=float(s["abs_ret_ema"]),
                        vol_baseline_ema=float(s["vol_baseline_ema"]),
                        last_jump_pct=float(s["last_jump_pct"]),
                    )
                    liq_candidate = _standalone_classify_liq(
                        spread_ema=float(s["spread_ema"]),
                        depth_ema=float(s["depth_ema"]),
                        ob_idle_s=max(0.0, now - float(s["last_ob_mono"])),
                    )

                    # Two-cycle hysteresis to reduce regime flapping.
                    if vol_candidate == s["vol_candidate"]:
                        s["vol_votes"] = int(s["vol_votes"]) + 1
                    else:
                        s["vol_candidate"] = vol_candidate
                        s["vol_votes"] = 1
                    if liq_candidate == s["liq_candidate"]:
                        s["liq_votes"] = int(s["liq_votes"]) + 1
                    else:
                        s["liq_candidate"] = liq_candidate
                        s["liq_votes"] = 1
                    if int(s["vol_votes"]) >= 2:
                        s["vol_regime"] = vol_candidate
                    if int(s["liq_votes"]) >= 2:
                        s["liq_regime"] = liq_candidate

                    vol_by_asset[asset] = str(s["vol_regime"])
                    liq_by_asset[asset] = str(s["liq_regime"])

                risk = _standalone_risk_regime(vol_by_asset, liq_by_asset)
                # Use UTC (gmtime) — never local time. DST does not affect UTC.
                # "US" session = roughly 13:00–23:00 UTC (US market hours in UTC).
                session = "US" if time.gmtime().tm_hour in range(13, 23) else "OFF_HOURS"
                for asset in assets:
                    await bus.publish(
                        "kalshi.regime",
                        {
                            "asset": asset,
                            "vol_regime": vol_by_asset.get(asset, "VOL_NORMAL"),
                            "liq_regime": liq_by_asset.get(asset, "LIQ_NORMAL"),
                            "risk_regime": risk,
                            "session_regime": session,
                            "market": "CRYPTO",
                            "source": "standalone_local",
                        },
                    )

                if (now - last_log) >= 30.0:
                    last_log = now
                    log.info(
                        "Standalone regime publisher",
                        data={
                            "assets": assets,
                            "vol": vol_by_asset,
                            "liq": liq_by_asset,
                            "risk": risk,
                            "interval_s": publish_interval_s,
                        },
                    )
                next_emit = now + publish_interval_s

            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        pass


#  WS ticker-change monitor

def _rank_near_money(
    tickers: Set[str],
    discovery: MarketDiscovery,
    probability: ProbabilityEngine,
    max_count: int,
) -> List[str]:
    # Rank tickers by proximity to current price, return top N.
    now_ts = time.time()
    scored: List[tuple[tuple[float, float], str, tuple[str, int, bool]]] = []
    bucket_best: Dict[tuple[str, int, bool], tuple[tuple[float, float], str]] = {}
    for t in tickers:
        meta = discovery.metadata.get(t)
        if not meta:
            continue
        price = probability._last_price_by_asset.get(meta.asset, 0.0)
        if price <= 0:
            scored.append((t, 0.5))  # no price yet — neutral priority
            continue
        if meta.is_range and meta.strike_floor is not None and meta.strike_cap is not None:
            mid = (meta.strike_floor + meta.strike_cap) / 2
            dist = abs(price - mid) / price
        else:
            dist = abs(price - meta.strike_price) / price if meta.strike_price > 0 else 999
        scored.append((t, dist))
    scored.sort(key=lambda x: x[1])
    return [t for t, _ in scored[:max_count]]


def _rank_active_near_money(
    tickers: Set[str],
    discovery: MarketDiscovery,
    probability: ProbabilityEngine,
    max_count: int,
) -> List[str]:
    # Rank tickers by current expiry horizon first, then near-money distance.
    now_ts = time.time()
    scored: List[tuple[tuple[float, float], str, tuple[str, int, bool]]] = []
    bucket_best: Dict[tuple[str, int, bool], tuple[tuple[float, float], str]] = {}
    for ticker in tickers:
        meta = discovery.metadata.get(ticker)
        if not meta:
            continue
        time_to_settle_s = time_to_settlement_seconds(
            meta.settlement_time_iso,
            now_ts=now_ts,
        )
        if (
            time_to_settle_s is None
            or time_to_settle_s <= 0
            or time_to_settle_s > _WS_SELECTION_MAX_TIME_TO_SETTLE_S
        ):
            continue

        price = probability._last_price_by_asset.get(meta.asset, 0.0)
        if price <= 0:
            dist = 0.5
        elif meta.is_range and meta.strike_floor is not None and meta.strike_cap is not None:
            mid = (meta.strike_floor + meta.strike_cap) / 2
            dist = abs(price - mid) / price
        else:
            dist = abs(price - meta.strike_price) / price if meta.strike_price > 0 else 999.0

        score = (dist, time_to_settle_s)
        bucket = (meta.asset, meta.window_minutes, meta.is_range)
        scored.append((score, ticker, bucket))
        current_best = bucket_best.get(bucket)
        if current_best is None or score < current_best[0]:
            bucket_best[bucket] = (score, ticker)

    if not scored:
        return []

    selected: List[str] = []
    seen: Set[str] = set()
    for _, ticker in sorted(bucket_best.values(), key=lambda item: item[0]):
        if ticker in seen or len(selected) >= max_count:
            continue
        selected.append(ticker)
        seen.add(ticker)

    for _, ticker, _ in sorted(scored, key=lambda item: item[0]):
        if ticker in seen or len(selected) >= max_count:
            continue
        selected.append(ticker)
        seen.add(ticker)
    return selected


async def _monitor_ticker_changes(
    bus: Bus,
    ws: Optional[KalshiWebSocket],
    strategy: StrategyEngine,
    probability: ProbabilityEngine,
    discovery: MarketDiscovery,
    vision: Optional["TerminalVisualizer"],
    current_tickers: Set[str],
    max_ws_markets: int = 120,
) -> None:
    # Watch ``kalshi.selected_markets`` for changes and update components.
    #
    # Only subscribes the top *max_ws_markets* near-money tickers on the WS
    # and strategy to prevent event loop starvation from 1,700+ concurrent tasks.
    q = await bus.subscribe("kalshi.selected_markets")
    try:
        while True:
            msg: SelectedMarkets = await q.get()
            all_discovered = set(msg.tickers)

            # Filter to top N near-money tickers for WS/strategy subscription.
            # We ensure coverage of diverse market types by forcing at least one per asset/window.
            ranked = _rank_active_near_money(all_discovered, discovery, probability, max_ws_markets)
            
            # Force-include any UI "best" markets that are still in the live
            # discovered set. Cap at max_ws_markets to avoid subscription bloat.
            # IMPORTANT: only include tickers in all_discovered — never re-add
            # an expired contract whose metadata has already been cleared.
            if vision:
                best_per_type = vision._best_per_type()
                for asset_bucket in best_per_type.values():
                    for st in asset_bucket.values():
                        if (st and st.ticker not in ranked
                                and st.ticker in all_discovered
                                and len(ranked) < max_ws_markets):
                            ranked.append(st.ticker)
            
            new_tickers = set(ranked)

            added = new_tickers - current_tickers
            removed = current_tickers - new_tickers

            if added or removed:
                log.debug(
                    "Ticker set changed — updating subscriptions",
                    data={
                        "added": len(added), "removed": len(removed),
                        "ws_subscribed": len(new_tickers),
                        "total_discovered": len(all_discovered),
                    },
                )
                await strategy.update_tickers(list(added), list(removed))

                # Dynamically update probability tracked markets (only near-money)
                added_meta = {t: discovery.metadata[t] for t in added if t in discovery.metadata}
                removed_from_prob = [t for t in removed if t not in new_tickers]
                await probability.update_markets(added_meta, removed_from_prob)

                if vision:
                    await vision.update_markets(added_meta, list(removed))

                for t in list(added)[:3]:  # only announce first few to avoid log spam
                    await bus.publish("kalshi.strategy.active_ticker", ActiveTicker(ticker=t, timestamp=time.time()))

                # Publish the full ranked set for the mispricing scalper.
                # Unlike 'selected_markets' (which is unfiltered), this only contains
                # the top N near-money tickers currently being watched by WS/strategy.
                await bus.publish(
                    "kalshi.ranked_markets",
                    SelectedMarkets(tickers=list(new_tickers), timestamp=time.time()),
                )

                # Update WS subscriptions
                if ws:
                    ws._desired_channels = list(_WS_CHANNELS)
                    # Keep WS market set sticky to avoid frequent full unsubscribe
                    # churn (Kalshi has no incremental remove). Strategy/probability
                    # still apply the exact near-money set for decisions.
                    await ws.update_subscription(
                        add_tickers=sorted(added) if added else None,
                        remove_tickers=None,
                    )
                current_tickers.clear()
                current_tickers.update(new_tickers)
    except asyncio.CancelledError:
        pass


#  Database Logging Monitors

async def _monitor_database_events(
    bus: Bus,
    db: Optional[Database],
) -> None:
    # Watch for TerminalEvents and StrategyDecisions and persist them.
    if not db:
        return

    q_term = await bus.subscribe("kalshi.terminal_event")
    q_dec = await bus.subscribe("kalshi.strategy_decision")

    try:
        while True:
            # Check terminal events
            while not q_term.empty():
                try:
                    msg: TerminalEvent = q_term.get_nowait()
                    await db.execute(
                        "INSERT INTO kalshi_terminal_events (timestamp, level, message, bot_id) VALUES (?, ?, ?, ?)",
                        (msg.timestamp, msg.level, msg.message, getattr(msg, "bot_id", "default"))
                    )
                except Exception as e:
                    log.error(f"Error persisting TerminalEvent: {e}")

            # Check decisions
            while not q_dec.empty():
                try:
                    dec: StrategyDecision = q_dec.get_nowait()
                    await db.execute(
                        """INSERT INTO kalshi_decisions
                           (timestamp, market_ticker, p_yes, yes_ask, no_ask, action_taken, reason, bot_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (dec.timestamp, dec.market_ticker, dec.p_yes, dec.yes_ask, dec.no_ask, dec.action_taken, dec.reason, getattr(dec, "bot_id", "default"))
                    )
                except Exception as e:
                    log.error(f"Error persisting StrategyDecision: {e}")

            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass

#  Persistent running bankroll

def _load_running_bankroll(
    initial_usd: float,
    log_path: str = "logs/paper_trades.jsonl",
    bot_id: str = "default",
) -> float:
    # Return initial_usd + all historical settled PnL from the JSONL log.
    #
    # This makes position sizing and the drawdown kill-switch aware of
    # accumulated paper gains/losses across restarts.  Falls back to
    # ``initial_usd`` if the log doesn't exist yet.
    p = Path(log_path)
    if not p.exists():
        return initial_usd
    total_pnl = 0.0
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "settlement":
                continue
            if rec.get("bot_id", "default") != bot_id:
                continue
            total_pnl += rec.get("pnl_usd", 0.0)
    except Exception as exc:
        log.warning(f"Could not load running bankroll from {log_path}: {exc}")
        return initial_usd
    running = initial_usd + total_pnl
    # Floor at 10 % of initial so the system never size to zero after a drawdown.
    floor = initial_usd * 0.10
    running = max(running, floor)
    log.info(
        f"Running bankroll loaded [{bot_id}]: ${running:.2f} "
        f"(initial=${initial_usd:.2f}, historical_pnl={total_pnl:+.2f})"
    )
    return running


def _load_ui_stats_from_jsonl(
    log_path: str = "logs/paper_trades.jsonl",
    bot_id: Optional[str] = None,
) -> tuple[float, int, int, float]:
    # Read all-time PnL / win-rate from the JSONL log for the UI.
    #
    # Returns (total_pnl, wins, total_settlements, first_entry_ts).
    # first_entry_ts is the Unix timestamp of the very first JSONL record —
    # used as the bot's "origin time" so uptime and rate stats reflect total
    # historical runtime across all sessions, not just the current one.
    # The JSONL is the canonical ground truth; the DB can miss records on
    # write errors so it must not be used as the authoritative stats source.
    p = Path(log_path)
    if not p.exists() or p.stat().st_size == 0:
        return 0.0, 0, 0, time.time()
    total_pnl = 0.0
    wins = 0
    total = 0
    first_ts: float = 0.0
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if bot_id is not None and rec.get("bot_id", "default") != bot_id:
                continue
            # Capture the very first timestamp for this bot in the file.
            if first_ts == 0.0 and rec.get("timestamp"):
                first_ts = float(rec["timestamp"])
            if rec.get("type") == "settlement":
                total += 1
                total_pnl += rec.get("pnl_usd", 0.0)
                if rec.get("won"):
                    wins += 1
    except Exception as exc:
        log.warning(f"Could not load UI stats from {log_path}: {exc}")
        return 0.0, 0, 0, time.time()
    return total_pnl, wins, total, first_ts if first_ts > 0 else time.time()


def _load_bot_stats_from_jsonl(
    log_path: str = "logs/paper_trades.jsonl",
    primary_bot_id: Optional[str] = None,
) -> tuple[Dict[str, Dict[str, Any]], float, int, int, float, float, float, float, float]:
    # Build per-bot stats from settlement and paper_fill records so UI shows correct PnL and fill counts after restart.
    #
    # Returns (bot_stats, primary_pnl, primary_wins, primary_losses, win_pnl_total, loss_pnl_total,
    # best_win, worst_loss, first_entry_ts) for the given promoted bot id.
    # first_entry_ts is the first settlement timestamp for that bot (for uptime).
    empty = {
        "pnl": 0.0, "pnl_e": 0.0, "pnl_s": 0.0,
        "wins": 0, "losses": 0, "fills": 0, "orders": 0,
        "trade_count": 0, "gross_profit": 0.0, "gross_loss": 0.0,
        "peak_pnl": 0.0, "max_drawdown": 0.0, "last_active": 0.0,
    }
    bot_stats: Dict[str, Dict[str, Any]] = {}
    primary_pnl = 0.0
    primary_wins = 0
    primary_losses = 0
    primary_win_pnl = 0.0
    primary_loss_pnl = 0.0
    primary_best_win = 0.0
    primary_worst_loss = 0.0
    first_entry_ts: float = time.time()
    _pid = (primary_bot_id or "").strip()
    primary_id: Optional[str] = _pid if _pid and _pid.lower() != "default" else None

    p = Path(log_path)
    if not p.exists() or p.stat().st_size == 0:
        return bot_stats, primary_pnl, primary_wins, primary_losses, primary_win_pnl, primary_loss_pnl, primary_best_win, primary_worst_loss, first_entry_ts

    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = rec.get("type")
            if rtype == "paper_fill":
                bot_id = rec.get("bot_id", "default") or "default"
                b = bot_stats.setdefault(bot_id, dict(empty))
                b["fills"] = b["fills"] + 1
                continue
            if rtype != "settlement":
                continue
            bot_id = rec.get("bot_id", "default") or "default"
            pnl = float(rec.get("pnl_usd", 0.0))
            won = bool(rec.get("won", False))
            src = (rec.get("source") or "").strip()
            is_scalp = src == "mispricing_scalp"
            ts = float(rec.get("timestamp", 0)) or 0.0

            b = bot_stats.setdefault(bot_id, dict(empty))
            b["pnl"] = b["pnl"] + pnl
            if is_scalp:
                b["pnl_s"] = b["pnl_s"] + pnl
            else:
                b["pnl_e"] = b["pnl_e"] + pnl
            b["trade_count"] = b["trade_count"] + 1
            if pnl >= 0:
                b["gross_profit"] = b["gross_profit"] + pnl
            else:
                b["gross_loss"] = b["gross_loss"] + abs(pnl)
            if won:
                b["wins"] = b["wins"] + 1
            else:
                b["losses"] = b["losses"] + 1
            b["peak_pnl"] = max(b["peak_pnl"], b["pnl"])
            b["max_drawdown"] = max(b["max_drawdown"], b["peak_pnl"] - b["pnl"])

            if primary_id is not None and bot_id == primary_id:
                if ts > 0:
                    first_entry_ts = min(first_entry_ts, ts)
                primary_pnl += pnl
                if won:
                    primary_wins += 1
                    primary_win_pnl += pnl
                    primary_best_win = max(primary_best_win, pnl)
                else:
                    primary_losses += 1
                    primary_loss_pnl += pnl
                    primary_worst_loss = min(primary_worst_loss, pnl)
    except Exception as exc:
        log.warning(f"Could not load bot stats from {log_path}: {exc}")
    return bot_stats, primary_pnl, primary_wins, primary_losses, primary_win_pnl, primary_loss_pnl, primary_best_win, primary_worst_loss, first_entry_ts


#  Account balance poller

async def _poll_balance(
    bus: Bus,
    rest: "KalshiRestClient",
    interval_s: float = 60.0,
) -> None:
    # Fetch Kalshi portfolio balance every *interval_s* seconds and publish it.
    try:
        while True:
            try:
                cents = await rest.get_balance()
                await bus.publish(
                    "kalshi.account_balance",
                    AccountBalance(balance_cents=cents, timestamp=time.time()),
                )
            except Exception as exc:
                log.warning(f"Balance poll failed: {exc}")
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        pass


def _measure_rtt_sync(base_url: str, samples: int = 5) -> float:
    # Measure REST RTT from a thread (sync) so event-loop load does not inflate the value.
    import urllib.request
    url = f"{base_url.rstrip('/')}/exchange/status"
    times: List[float] = []
    for _ in range(samples):
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as r:
                r.read()
            times.append((time.monotonic() - t0) * 1000)
        except Exception:
            pass

    if not times:
        return 200.0
    times.sort()
    return times[len(times) // 2]


async def _poll_kalshi_rtt(
    bus: Bus,
    rest: "KalshiRestClient",
    interval_s: float = 30.0,
    run_in_thread: bool = True,
) -> None:
    # Re-measure Kalshi REST RTT every *interval_s* seconds and publish for the UI.
    #
    # When run_in_thread is True, the HTTP GETs run in a thread so the reported
    # RTT reflects actual network latency, not event-loop delay.
    try:
        while True:
            await asyncio.sleep(interval_s)
            try:
                if run_in_thread:
                    loop = asyncio.get_running_loop()
                    rtt_ms = await loop.run_in_executor(
                        None,
                        _measure_rtt_sync,
                        rest._base,
                        5,
                    )
                else:
                    rtt_ms = await rest.measure_rtt_ms(samples=5)
                await bus.publish(
                    "kalshi.rtt",
                    KalshiRtt(rtt_ms=rtt_ms, timestamp=time.time(), source="rest"),
                )
                log.info("Kalshi RTT: %.0fms (rest)", rtt_ms, data={"rtt_ms": round(rtt_ms, 1), "source": "rest"})
            except Exception as exc:
                log.debug("Kalshi RTT poll failed: %s", exc)
    except asyncio.CancelledError:
        pass


#  Main entrypoint

async def run(settings_path: str, secrets_path: str) -> None:
    # Load config, build all components, and run until shutdown.
    #
    # Parameters
    # settings_path :
    # Path to the settings YAML (must contain an ``argus_kalshi`` key).
    # secrets_path :
    # Path to the secrets YAML (must contain a ``kalshi`` key with
    # ``key_id`` and ``private_key_path``).
    setup_logging()
    runner_start_mono = time.monotonic()
    log.info(
        "Argus Kalshi runner starting",
        data={"settings": settings_path, "secrets": secrets_path},
    )

    # ── Load configuration ──────────────────────────────────────────
    settings = _load_yaml(settings_path)
    secrets = _load_yaml(secrets_path)
    configs = _build_configs(settings, secrets, settings_path=settings_path)
    cfg = configs[0]  # Primary config controls shared connections
    # Promoted slot: which bot (by name, e.g. dwarf name like Thorin) gets STATS/ORDERS/HISTORY.
    # Set promoted_bot_id in config to that bot's name to see its stats; leave unset or "default"
    # to keep the slot empty until you choose a bot to promote.
    _promoted = (settings.get("promoted_bot_id") or "").strip()
    primary_bot_id: Optional[str] = _promoted if _promoted and _promoted.lower() != "default" else None

    log.info(
        f"Configuration(s) loaded: {len(configs)} isolated bots.",
        data={
            "dry_run": cfg.dry_run,
            "ws_trading_enabled": cfg.ws_trading_enabled,
            "series_filter": cfg.series_filter,
            "target_tickers": cfg.target_market_tickers,
        },
    )
    # Count of bots with distinct strategy params (for UI "Unique N" in leaderboard).
    unique_config_count: Optional[int] = None
    if len(configs) > 1:
        fingerprints = {
            (c.min_edge_threshold, c.persistence_window_ms, c.min_entry_cents, c.max_entry_cents)
            for c in configs
        }
        unique_config_count = len(fingerprints)

    # Log a sample of strategy params so we can verify the grid (468 bots with unique params).
    if len(configs) > 1:
        for idx in [0, 1, len(configs) // 2, len(configs) - 1]:
            c = configs[idx]
            log.info(
                f"Farm config sample [{idx}] {c.bot_id}: "
                f"min_edge={c.min_edge_threshold} persistence_ms={c.persistence_window_ms} "
                f"entry_cents=[{c.min_entry_cents},{c.max_entry_cents}]",
                data={
                    "bot_id": c.bot_id,
                    "min_edge_threshold": c.min_edge_threshold,
                    "persistence_window_ms": c.persistence_window_ms,
                    "min_entry_cents": c.min_entry_cents,
                    "max_entry_cents": c.max_entry_cents,
                },
            )

    # ── Initialise shared state ─────────────────────────────────────
    # Keep subscriber queues finite and freshness-biased. Extremely deep queues
    # turn a live system into a stale replay under farm overload.
    bus = Bus(subscriber_queue_maxsize=5_000)
    orderbooks: Dict[str, OrderBook] = {}

    # Initialize core persistence if available
    db: Optional[Database] = None
    core_bus: Optional[EventBus] = None
    bar_builder: Optional[BarBuilder] = None
    persistence: Optional[PersistenceManager] = None

    if Database and EventBus:
        try:
            # Use the argus_kalshi-specific db_path to avoid write-lock contention
            # with main.py which also writes to data/argus.db.
            # Resolve database path from settings (handles single dict or list/farm mode)
            argus_block = settings.get("argus_kalshi", {})
            if isinstance(argus_block, list) and len(argus_block) > 0:
                argus_block = argus_block[0]
            argus_block = argus_block if isinstance(argus_block, dict) else {}

            db_path = argus_block.get("database_path") or "data/kalshi.db"
            enable_core_persistence = argus_block.get("enable_core_persistence", True)

            db = Database(db_path)
            await db.connect()
            log.info(f"Core database connected: {db_path}")

            if enable_core_persistence:
                core_bus = EventBus()
                bar_builder = BarBuilder(core_bus)
                loop = asyncio.get_running_loop()
                persistence = PersistenceManager(core_bus, db, loop)
                persistence.start()
                core_bus.start()
                log.info("Core persistence layer started")
            else:
                core_bus = None
                log.info("Core persistence disabled (enable_core_persistence=false)")
        except Exception as e:
            log.error(f"Failed to initialize core persistence: {e}")
            db = None
            core_bus = None


    #  Regime bridge: core EventBus  Kalshi Bus ─
    # Forwards SymbolRegimeEvent/MarketRegimeEvent from core regime
    # detector to `kalshi.regime` topic on the Kalshi bus so FarmDispatcher
    # can update SharedFarmState.regime_* fields without direct import.
    _regime_bridge_active = False
    if core_bus is not None:
        try:
            from src.core.regimes import get_market_for_symbol

            # Capture the running event loop so core-bus thread callbacks
            # can safely schedule async bus.publish via call_soon_threadsafe.
            _bridge_loop = asyncio.get_running_loop()

            def _bridge_publish(payload: dict) -> None:
                _bridge_loop.call_soon_threadsafe(
                    asyncio.ensure_future,
                    bus.publish('kalshi.regime', payload),
                )

            def _bridge_symbol_regime(event) -> None:
                symbol = getattr(event, 'symbol', '')
                asset = symbol.split('/')[0].split('-')[0].upper() if symbol else ''
                if not asset:
                    return
                _bridge_publish({
                    'asset': asset,
                    'vol_regime': getattr(event, 'vol_regime', ''),
                    'liq_regime': getattr(event, 'liquidity_regime', ''),
                    'risk_regime': '',
                    'session_regime': '',
                    'market': get_market_for_symbol(symbol),
                })

            def _bridge_market_regime(event) -> None:
                market = getattr(event, 'market', '')
                _bridge_publish({
                    'asset': 'BTC' if market == 'CRYPTO' else '',
                    'vol_regime': '',
                    'liq_regime': '',
                    'risk_regime': getattr(event, 'risk_regime', ''),
                    'session_regime': getattr(event, 'session_regime', ''),
                    'market': market,
                })

            core_bus.subscribe('regimes.symbol', _bridge_symbol_regime)
            core_bus.subscribe('regimes.market', _bridge_market_regime)
            _regime_bridge_active = True
            log.info('Regime bridge active: core EventBus -> kalshi.regime (thread-safe)')
        except Exception as e:
            log.warning(f'Regime bridge setup failed (standalone mode): {e}')
    else:
        log.info('No core EventBus - regime bridge not available (standalone Kalshi)')

    # Components — initialised to None so the finally block knows
    # what has been started and needs cleanup.
    rest: Optional[KalshiRestClient] = None
    discovery: Optional[MarketDiscovery] = None
    farm: Optional[KalshiPaperFarm] = None
    strategy: Optional[StrategyEngine] = None
    execution: Optional[ExecutionEngine] = None
    ws: Optional[KalshiWebSocket] = None
    luzia_feed: Optional[LuziaPollingFeed] = None
    okx_feed: Optional[Any] = None
    mock_feed: Optional[MockBtcFeed] = None
    settlement: Optional[SettlementTracker] = None
    vision: Optional[TerminalVisualizer] = None
    scalper: Optional[MispricingScalper] = None
    ticker_monitor_task: Optional[asyncio.Task[None]] = None
    discovery_task: Optional[asyncio.Task[None]] = None
    standalone_regime_task: Optional[asyncio.Task[None]] = None
    db_logging_task: Optional[asyncio.Task[None]] = None
    balance_task: Optional[asyncio.Task[None]] = None
    rtt_task: Optional[asyncio.Task[None]] = None
    compute_executor: Optional[ThreadPoolExecutor] = None
    ipc_server: Optional[asyncio.Server] = None
    state_aggregator: Optional[ipc_module.StateAggregator] = None
    tickers: List[str] = []
    coinbase_streams: List[CoinbaseWebSocket] = []
    coinbase_tasks: List[asyncio.Task] = []
    window_engines: List[BtcWindowEngine] = []

    try:
        # ── REST client ─────────────────────────────────────────────
        rest = KalshiRestClient(cfg)
        await rest.start()
        log.info("REST client started")

        # ── Measure Kalshi RTT for UI and paper slippage ─────────────
        # Paper fills use this to simulate the price movement that occurs
        # during the REST round-trip.  Slippage = max(1, rtt_ms ÷ 100)¢.
        kalshi_rtt_ms: Optional[float] = None
        if cfg.visualizer_enabled or cfg.dry_run:
            rtt_ms = await rest.measure_rtt_ms(samples=5)
            kalshi_rtt_ms = rtt_ms
            await bus.publish("kalshi.rtt", KalshiRtt(rtt_ms=rtt_ms, timestamp=time.time(), source="rest"))
            log.info("Kalshi RTT: %.0fms (rest, startup)", rtt_ms, data={"rtt_ms": round(rtt_ms, 1), "source": "rest"})
            if cfg.dry_run:
                if (
                    getattr(cfg, "paper_order_latency_min_ms", 0) <= 0
                    and getattr(cfg, "paper_order_latency_max_ms", 0) <= 0
                ):
                    lat_min = max(50, round(rtt_ms * 0.75))
                    lat_max = max(lat_min + 30, round(rtt_ms * 1.5))
                    object.__setattr__(cfg, "paper_order_latency_min_ms", lat_min)
                    object.__setattr__(cfg, "paper_order_latency_max_ms", lat_max)
                log.info(
                    "Paper execution realism",
                    data={
                        "latency_min_ms": getattr(cfg, "paper_order_latency_min_ms", 0),
                        "latency_max_ms": getattr(cfg, "paper_order_latency_max_ms", 0),
                        "slippage_cents": getattr(cfg, "paper_slippage_cents", 0),
                        "apply_fees": bool(getattr(cfg, "paper_apply_fees", False)),
                    },
                )

        # ── Initialise components ──────────────────────────────────
        # Discovery stays idle until start() is called.
        discovery = MarketDiscovery(cfg, rest, bus)

        if cfg.enable_regime_gating and not _regime_bridge_active:
            standalone_regime_task = asyncio.create_task(
                _standalone_regime_publisher(bus, cfg, discovery)
            )
            log.info("Standalone regime publisher started (truth+orderbook -> kalshi.regime)")
        
        # ── Start UI / Vision or IPC for separate UI process ─────────
        if cfg.visualizer_enabled:
            if getattr(cfg, "visualizer_process", "inline") == "separate":
                # UI runs in another process; trading process runs IPC server.
                state_aggregator = ipc_module.StateAggregator(
                    bus, primary_bot_id=primary_bot_id,
                )
                # Separate UI: only IPC loop drains; no competing _aggregate_loop task.
                await state_aggregator.start(run_aggregate_loop=False)
                # Seed from paper_trades.jsonl so PnL and fill counts are correct after restart.
                (
                    bot_stats, primary_pnl, primary_wins, primary_losses,
                    primary_win_pnl, primary_loss_pnl, primary_best_win, primary_worst_loss, _,
                ) = _load_bot_stats_from_jsonl(primary_bot_id=primary_bot_id)
                state_aggregator.seed_from_jsonl(
                    bot_stats,
                    primary_pnl=primary_pnl, primary_wins=primary_wins, primary_losses=primary_losses,
                    primary_win_pnl=primary_win_pnl, primary_loss_pnl=primary_loss_pnl,
                    primary_best_win=primary_best_win, primary_worst_loss=primary_worst_loss,
                )
                all_bot_ids = [c.bot_id for c in configs if c.bot_id]
                state_aggregator.ensure_bot_stats_entries(all_bot_ids)
                dashboard_enabled = bool(getattr(cfg, "dashboard_enabled", True))
                on_snapshot = None
                dashboard_host = getattr(cfg, "ipc_bind", "127.0.0.1")
                dashboard_port = getattr(cfg, "dashboard_port", 9998)
                if dashboard_enabled:
                    # One snapshot producer can feed both IPC (terminal) and dashboard (browser).
                    from . import dashboard as dashboard_module
                    dashboard_module.start_dashboard_thread(dashboard_host, dashboard_port)
                    on_snapshot = dashboard_module.set_snapshot

                ipc_server = await ipc_module.ipc_server_start(
                    state_aggregator,
                    getattr(cfg, "ipc_bind", "127.0.0.1"),
                    getattr(cfg, "ipc_port", 9999),
                    on_snapshot=on_snapshot,
                )
                host = getattr(cfg, "ipc_bind", "127.0.0.1")
                port = getattr(cfg, "ipc_port", 9999)
                connect_arg = f"{host}:{port}"
                await asyncio.sleep(0.5)
                if dashboard_enabled:
                    dashboard_url = f"http://{dashboard_host}:{dashboard_port}"
                    log.info(
                        "Dashboard (decoupled): %s — backend runs at full speed.",
                        dashboard_url,
                    )
                    try:
                        webbrowser.open(dashboard_url)
                    except Exception as e:
                        log.debug("Could not open dashboard in browser: %s", e)
                    log.info(
                        "IPC server ready. Optional terminal UI in another terminal: "
                        "python -m argus_kalshi --ui-only --connect %s",
                        connect_arg,
                    )
                else:
                    auto_launch_terminal_ui = bool(getattr(cfg, "auto_launch_terminal_ui", False))
                    log.info(
                        "Terminal UI (decoupled IPC only): run in another terminal with "
                        "python -m argus_kalshi --ui-only --connect %s",
                        connect_arg,
                    )
                    if auto_launch_terminal_ui:
                        try:
                            ui_proc = _launch_terminal_ui_client(connect_arg)
                            log.info(
                                "Terminal UI client launched",
                                data={"pid": ui_proc.pid, "connect": connect_arg},
                            )
                        except Exception as e:
                            log.warning(
                                "Failed to auto-launch terminal UI client",
                                data={"connect": connect_arg, "error": str(e)},
                            )
            else:
                # Inline UI: redirect logs and start TerminalVisualizer in this process.
                from logging.handlers import RotatingFileHandler
                os.makedirs("logs", exist_ok=True)
                fh = RotatingFileHandler("logs/argus.log", maxBytes=500_000_000, backupCount=5)
                fh.setFormatter(StructuredFormatter())
                root_logger = logging.getLogger()
                ak_logger = logging.getLogger("argus_kalshi")
                for l in [root_logger, ak_logger]:
                    for h in l.handlers[:]:
                        if isinstance(h, logging.StreamHandler):
                            l.removeHandler(h)
                ak_logger.propagate = True
                root_logger.addHandler(fh)
                for name in logging.root.manager.loggerDict:
                    l = logging.getLogger(name)
                    if l.handlers:
                        for h in l.handlers[:]:
                            if isinstance(h, logging.StreamHandler):
                                l.removeHandler(h)
                vision = TerminalVisualizer(
                    bus,
                    metadata=discovery.metadata,
                    dry_run=cfg.dry_run,
                    primary_bot_id=primary_bot_id,
                    leaderboard_only=False,
                    unique_config_count=unique_config_count,
                    initial_kalshi_rtt_ms=kalshi_rtt_ms,
                )
                # Seed dwarf ranking and fill counts from paper_trades.jsonl so PnL/fills correct after restart.
                (
                    bot_stats, primary_pnl, primary_wins, primary_losses,
                    _, _, _, _, jsonl_first_ts,
                ) = _load_bot_stats_from_jsonl(primary_bot_id=primary_bot_id)
                vision.seed_bot_stats(bot_stats)
                # Pre-seed all known bot_ids so leaderboard shows dwarf names before they trade.
                all_bot_ids = [c.bot_id for c in configs if c.bot_id]
                vision.ensure_bot_stats_entries(all_bot_ids)
                primary_total = primary_wins + primary_losses
                paper_balance_usd = cfg.bankroll_usd + primary_pnl
                vision.set_initial_stats(
                    primary_pnl, primary_wins, primary_total, jsonl_first_ts,
                    seed_balance_usd=paper_balance_usd,
                )
                if primary_bot_id and primary_total > 0:
                    log.info(
                        f"UI stats from JSONL [{primary_bot_id}]: pnl={primary_pnl:+.4f} "
                        f"wins={primary_wins}/{primary_total}"
                    )
                else:
                    log.info(
                        "Promoted slot is not set (STATS/ORDERS/HISTORY stay neutral until promoted_bot_id is configured)"
                    )
                await vision.start()
                log.info("Argus Vision visualizer started")
                print(f"\n{ORANGE}{BOLD}ARGUS VISION NGE v5.1{RESET}")
                print(f"{CYAN}Fair %{RESET} = Probability model based on spot drift & volatility.")
                print(f"{GREEN}Edge{RESET}   = Difference between Fair % and market Ask.")
                print(f"{AMBER}Spot{RESET}   = Argus truth feed; {DKGRAY}(+123){RESET} = Distance from strike.")
                print(f"{DKGRAY}Starting UI in 1s...{RESET}\n")
                await asyncio.sleep(1.0)

        # ── Start discovery early; we await it before the farm so prob/WS get markets quickly ─
        discovery_task: Optional[asyncio.Task[None]] = None
        if discovery is not None:
            discovery_task = asyncio.create_task(discovery.start())
            log.info("Market discovery task launched (running in parallel)")

        # ── Start Truth Feeds (Coinbase / Luzia) early ─────────────
        coinbase_streams: List[CoinbaseWebSocket] = []
        coinbase_tasks = []
        last_coinbase_tick_ts_by_asset: Dict[str, float] = {}
        truth_feeds = _resolve_truth_feeds(cfg)

        if cfg.use_coinbase_ws and CoinbaseWebSocket:
            for tf in truth_feeds:
                async def _on_coinbase_ticker(data: Dict[str, Any], tf_cfg: TruthFeedConfig = tf):
                    ts = time.time()
                    asset = tf_cfg.asset.upper()
                    last_coinbase_tick_ts_by_asset[asset] = ts
                    msg = BtcMidPrice(
                        price=data['last_price'],
                        timestamp=ts,
                        source="coinbase_ws",
                        asset=asset,
                    )
                    await bus.publish(tf_cfg.topic, msg)

                ws_client = CoinbaseWebSocket(
                    symbols=[tf.coinbase_symbol],
                    on_ticker=_on_coinbase_ticker,
                    event_bus=core_bus if tf.publish_to_core_bus else None,
                )
                coinbase_streams.append(ws_client)
                coinbase_tasks.append(asyncio.create_task(ws_client.connect()))
            log.info(f"Coinbase WebSocket truth feeds started for {len(truth_feeds)} assets")

        # Fallback feed: OKX WS (preferred) or Luzia REST
        luzia_feed: Optional[LuziaPollingFeed] = None
        okx_feed: Optional[Any] = None
        if cfg.use_okx_fallback and OkxWsFallback:
            async def _on_okx_ticker(data: Dict[str, Any]):
                # Single timestamp check: use OKX when Coinbase silent >= fallback_activation_s (5s). Instant switch.
                btc_last = last_coinbase_tick_ts_by_asset.get("BTC", 0.0)
                if btc_last and (time.time() - btc_last) < cfg.fallback_activation_s:
                    return
                msg = BtcMidPrice(
                    price=data["last_price"],
                    timestamp=data.get("timestamp", time.time()),
                    source="okx_fallback",
                    asset="BTC",
                )
                await bus.publish("btc.mid_price", msg)

            okx_feed = OkxWsFallback(
                inst_ids=cfg.okx_ticker_inst_ids,
                on_ticker=_on_okx_ticker,
                ws_url=cfg.okx_ws_url or None,
            )
            await okx_feed.start()
            log.info("OKX WebSocket fallback feed started")
        elif cfg.use_luzia_fallback and LuziaPollingFeed and cfg.luzia_api_key:
            async def _on_luzia_ticker(data: Dict[str, Any]):
                # Same instant switch: use Luzia when Coinbase silent >= fallback_activation_s.
                btc_last = last_coinbase_tick_ts_by_asset.get("BTC", 0.0)
                if btc_last and (time.time() - btc_last) < cfg.fallback_activation_s:
                    return
                msg = BtcMidPrice(
                    price=data['last_price'],
                    timestamp=time.time(),
                    source="luzia_fallback",
                    asset="BTC",
                )
                await bus.publish("btc.mid_price", msg)

            luzia_feed = LuziaPollingFeed(
                api_key=cfg.luzia_api_key,
                endpoints=cfg.luzia_endpoints,
                on_ticker=_on_luzia_ticker,
                interval_seconds=15.0
            )
            await luzia_feed.start()
            log.info("Luzia.dev Fallback feed started")

        # One-time log when each truth asset first ticks (diagnostic for slow UI).
        async def _first_tick_logger(bus: Bus, start_mono: float) -> None:
            topics = _asset_topics_for_cfg(cfg)
            queues = [(await bus.subscribe(t), name) for t, name in topics]
            seen: Set[str] = set()
            while len(seen) < len(topics):
                pending = {name: asyncio.create_task(q.get()) for q, name in queues if name not in seen}
                if not pending:
                    break
                done, still_pending = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    for (q, name) in queues:
                        if name in seen:
                            continue
                        if task == pending.get(name):
                            elapsed = time.monotonic() - start_mono
                            log.info(
                                "First %s tick at +%.1fs",
                                name,
                                elapsed,
                                data={"asset": name, "elapsed_s": round(elapsed, 1)},
                            )
                            seen.add(name)
                            break
                for t in still_pending:
                    t.cancel()
            if seen:
                log.info(
                    "All truth feed assets have ticked",
                    data={"elapsed_s": round(time.monotonic() - start_mono, 1), "assets": list(seen)},
                )

        asyncio.create_task(_first_tick_logger(bus, runner_start_mono))

        # Periodic truth-feed tick counts (every 60s) for diagnosing slow price updates.
        async def _truth_feed_diagnostic(bus: Bus) -> None:
            topics = _asset_topics_for_cfg(cfg)
            queues = [(await bus.subscribe(t), name) for t, name in topics]
            counts: Dict[str, int] = {name: 0 for _, name in topics}
            window_s = 60.0
            while True:
                try:
                    deadline = time.monotonic() + window_s
                    while time.monotonic() < deadline:
                        for (q, name) in queues:
                            while True:
                                try:
                                    q.get_nowait()
                                    counts[name] = counts.get(name, 0) + 1
                                except asyncio.QueueEmpty:
                                    break
                        await asyncio.sleep(0.25)
                    log.info(
                        "Truth feed (60s): %s",
                        ", ".join(f"{name} {counts[name]}" for _, name in topics),
                        data={**{name.lower(): counts[name] for _, name in topics}, "window_s": window_s},
                    )
                    counts = {name: 0 for _, name in topics}
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.debug("Truth feed diagnostic error: %s", e)

        asyncio.create_task(_truth_feed_diagnostic(bus))

        # ── Per-asset window engines & probability ─────────────────────
        window_engines = []
        for tf in truth_feeds:
            we = BtcWindowEngine(bus, truth_topic=tf.topic, asset=tf.asset)
            await we.start()
            window_engines.append(we)

        # Thread pool for CPU-bound work (probability recompute). Use multiple
        # workers so the event loop stays responsive and recompute runs in parallel.
        _cpu = os.cpu_count() or 4
        _n_threads = cfg.compute_threads if cfg.compute_threads > 0 else min(32, _cpu * 2)
        compute_executor = ThreadPoolExecutor(
            max_workers=_n_threads,
            thread_name_prefix="kalshi_compute",
        )
        log.info("Compute thread pool started", data={"workers": _n_threads})
        prob_engine = ProbabilityEngine(
            bus, discovery.metadata,
            urgent_move_pct=cfg.urgent_move_pct,
            executor=compute_executor,
        )
        await prob_engine.start()

        # ── WebSocket & Monitor (start before farm so OB/prices flow within ~8s) ─
        # Farm startup takes ~40s for 7k+ bots; WS + ticker_monitor must run early
        # so discovery's selected_markets triggers WS subscribe within ~8s.
        ws = KalshiWebSocket(cfg, bus, orderbooks)
        await ws.start()
        log.info("Kalshi Market Data WebSocket started")

        async def _ws_orderbook_diagnostic(bus: Bus, ws_client: KalshiWebSocket, interval_s: float = 30.0) -> None:
            # Periodic WS/orderbook health log to diagnose stale books.
            try:
                while True:
                    await asyncio.sleep(interval_s)
                    health = ws_client.get_health()
                    ob_valid = sum(1 for ob in orderbooks.values() if getattr(ob, "valid", False))
                    ob_total = len(orderbooks)
                    log.info(
                        "WS/orderbook diagnostics",
                        data={
                            "window_s": interval_s,
                            "ws": health,
                            "orderbooks": {
                                "total": ob_total,
                                "valid": ob_valid,
                                "invalid": max(0, ob_total - ob_valid),
                            },
                            "subscribers": {
                                "kalshi.orderbook": bus.subscriber_count("kalshi.orderbook"),
                                "kalshi.fair_prob": bus.subscriber_count("kalshi.fair_prob"),
                                "kalshi.trade_signal": bus.subscriber_count("kalshi.trade_signal"),
                            },
                        },
                    )
            except asyncio.CancelledError:
                pass

        asyncio.create_task(_ws_orderbook_diagnostic(bus, ws))

        # ── Wait for discovery so probability engine and WS get markets before farm starves the loop ─
        if discovery_task is not None:
            try:
                await asyncio.wait_for(discovery_task, timeout=90.0)
                log.info("Market discovery completed — seeding probability and WS")
            except asyncio.TimeoutError:
                log.warning("Market discovery did not complete within 90s — proceeding with current metadata")
            except asyncio.CancelledError:
                pass

        # ── Strategy & Execution ───────────────────────────────────
        farm_task: Optional[asyncio.Task] = None
        if len(configs) > 1:
            log.info(f"Starting Kalshi Paper Farm ({len(configs)} isolated bots)")
            farm = KalshiPaperFarm(configs, bus, rest, db=db)
            strategy = farm
        else:
            log.info(f"Starting single Kalshi bot ({cfg.bot_id})")
            running_bankroll = _load_running_bankroll(cfg.bankroll_usd, bot_id=cfg.bot_id)
            strategy = StrategyEngine(cfg, bus, running_bankroll=running_bankroll)
            await strategy.start([])

            execution = ExecutionEngine(cfg, bus, rest, db=db)
            await execution.start()

            if cfg.scalper_enabled:
                scalper = MispricingScalper(cfg, bus)
                await scalper.start()
                log.info("Mispricing scalper started")
            else:
                log.info("Mispricing scalper disabled (scalper_enabled=False)")

        await bus.publish("kalshi.terminal_event", TerminalEvent(
            timestamp=time.time(),
            level="INFO",
            message="SYS: WebSocket uplink established"
        ))

        # Start ticker monitor and seed selected_markets so probability + WS get markets
        # before the farm (if any) starves the event loop. Give monitor a moment to process.
        effective_max_ws_markets = cfg.max_ws_markets
        if len(configs) >= 4000:
            effective_max_ws_markets = min(effective_max_ws_markets, 144)
        if len(configs) >= 7000:
            effective_max_ws_markets = min(effective_max_ws_markets, 120)
        log.info(
            "WS market cap selected",
            data={
                "configured": cfg.max_ws_markets,
                "effective": effective_max_ws_markets,
                "farm_bots": len(configs),
            },
        )
        ticker_monitor_task = asyncio.create_task(
            _monitor_ticker_changes(
                bus,
                ws,
                strategy,
                prob_engine,
                discovery,
                vision,
                set(),
                effective_max_ws_markets,
            )
        )
        if discovery is not None and discovery.metadata:
            await bus.publish(
                "kalshi.selected_markets",
                SelectedMarkets(tickers=list(discovery.metadata.keys()), timestamp=time.time()),
            )
            await asyncio.sleep(0.5)

        # Now start the farm so it does not delay the initial prob/WS subscription.
        if len(configs) > 1:
            farm_task = asyncio.create_task(farm.start([]))
            await farm_task
            if state_aggregator is not None:
                state_aggregator.set_bot_stats_overlay_provider(farm.get_bot_stats_overlay)
                state_aggregator.set_population_overlay_provider(farm.get_population_diagnostics)
            # Re-publish selected_markets so the farm (now with all strategies) gets the ticker set.
            if discovery is not None and discovery.metadata:
                await bus.publish(
                    "kalshi.selected_markets",
                    SelectedMarkets(tickers=list(discovery.metadata.keys()), timestamp=time.time()),
                )

        db_logging_task = asyncio.create_task(
            _monitor_database_events(bus, db)
        )

        # ── Settlement tracker ──────────────────────────────────────
        settlement: Optional[SettlementTracker] = None
        if cfg.enable_settlement_tracker:
            settlement = SettlementTracker(bus, db=db)
            await settlement.start()
            log.info("Settlement tracker started")
        elif cfg.dry_run and not cfg.use_proxy_truth_feed and not (cfg.use_coinbase_ws and coinbase_streams):
            mock_feed = MockBtcFeed(bus, topic=cfg.truth_feed_topic)
            await mock_feed.start()
            log.info("Mock BTC feed started (dry_run mode)")

        # ── Balance poller (fetch once immediately, then every 60s) ─
        if not cfg.dry_run:
            balance_task = asyncio.create_task(_poll_balance(bus, rest, interval_s=60.0))
        else:
            balance_task = None

        # ── Kalshi RTT poll (update terminal UI latency every N seconds) ─
        if cfg.visualizer_enabled and cfg.kalshi_rtt_poll_interval_s > 0:
            interval = max(1.0, cfg.kalshi_rtt_poll_interval_s)
            rtt_task = asyncio.create_task(_poll_kalshi_rtt(bus, rest, interval_s=interval))

        # ── Wait for shutdown signal ───────────────────────────────
        log.info("All components running — awaiting shutdown signal")
        shutdown_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown_event.set)
            except NotImplementedError:
                pass  # Windows / nested loops

        try:
            await shutdown_event.wait()
        except asyncio.CancelledError:
            pass

        log.info("Shutdown signal received")

    except (asyncio.CancelledError, KeyboardInterrupt):
        log.info("Interrupted — shutting down")

    finally:
        # ── Clean shutdown (reverse startup order) ──────────────────
        if rtt_task is not None:
            rtt_task.cancel()
            try:
                await rtt_task
            except (asyncio.CancelledError, Exception):
                pass

        if balance_task is not None:
            balance_task.cancel()
            try:
                await balance_task
            except (asyncio.CancelledError, Exception):
                pass

        if ticker_monitor_task is not None:
            ticker_monitor_task.cancel()
            try:
                await ticker_monitor_task
            except (asyncio.CancelledError, Exception):
                pass

        if standalone_regime_task is not None:
            standalone_regime_task.cancel()
            try:
                await standalone_regime_task
            except (asyncio.CancelledError, Exception):
                pass

        if db_logging_task is not None:
            db_logging_task.cancel()
            try:
                await db_logging_task
            except (asyncio.CancelledError, Exception):
                pass

        if mock_feed is not None:
            await mock_feed.stop()
            log.info("Mock BTC feed stopped")

        for ws in coinbase_streams:
            await ws.disconnect()
        if coinbase_streams:
            log.info("Coinbase WebSocket feeds stopped")
        for task in coinbase_tasks:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        for we in window_engines:
            await we.stop()

        if okx_feed is not None:
            await okx_feed.stop()
            log.info("OKX WebSocket fallback stopped")
        if luzia_feed is not None:
            await luzia_feed.stop()
            log.info("Luzia Polling Feed stopped")

        if state_aggregator is not None:
            state_aggregator.stop()
            log.info("IPC state aggregator stopped")
        if ipc_server is not None:
            ipc_server.close()
            await ipc_server.wait_closed()
            log.info("IPC server closed")
        if vision is not None:
            await vision.stop()
            log.info("Argus Vision stopped")

        if settlement:
            settlement.stop()
            log.info("Settlement tracker stopped")

        if persistence:
            persistence.shutdown()
            log.info("Core persistence stopped")

        if core_bus:
            core_bus.stop()
            log.info("Core event bus stopped")

        if db:
            await db.close()
            log.info("Core database closed")

        if scalper is not None:
            await scalper.stop()
            log.info("Mispricing scalper stopped")

        if farm is not None:
            await farm.stop()
            log.info("Kalshi Paper Farm completely shut down")

        # Cancel pending orders on shutdown (safe mode).
        if execution is not None:
            if cfg.cancel_on_shutdown and execution.pending_count > 0:
                log.info(
                    f"Cancelling {execution.pending_count} pending order(s) "
                    "on shutdown"
                )
                await bus.publish(
                    "kalshi.risk",
                    RiskEvent(
                        event_type="disconnect_halt",
                        detail="Runner shutdown",
                        timestamp=time.time(),
                    ),
                )
                await asyncio.sleep(0.5)  # let execution process the event
            await execution.stop()
            log.info("Execution engine stopped")

        if strategy is not None and strategy != farm:
            await strategy.stop()
            log.info("Strategy engine stopped")

        try:
            prob_engine.stop()
            log.info("Probability engine stopped")
        except (NameError, AttributeError):
            pass

        if compute_executor is not None:
            compute_executor.shutdown(wait=True)
            log.info("Compute executor shut down")

        # ProbabilityEngine tasks are now explicitly cancelled via stop()

        # Window engines already stopped in the loop above (lines 568-569)

        if ws is not None:
            await ws.stop()
            log.info("WebSocket stopped")

        if discovery is not None:
            await discovery.stop()
            log.info("Market discovery stopped")

        if rest is not None:
            await rest.close()
            log.info("REST client closed")

        log.info("Argus Kalshi runner shutdown complete")
