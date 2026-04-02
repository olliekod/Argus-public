# Created by Oliver Meihls

# Shared market state for Kalshi farm mode.
#
# All 7,488 farm bots read from ONE copy of these dicts instead of each
# maintaining their own asyncio subscriber queues.  This eliminates the
# O(N_bots) bus fan-out that blocked the event loop.
#
# Architecture
# * FarmDispatcher (in farm_runner.py) subscribes ONCE per high-frequency
# topic and writes into SharedFarmState.
# * StrategyEngine and MispricingScalper in farm mode point their internal
# state dicts at the shared objects — dict identity, not copy.
# * Low-frequency topics (fills, ws.status, account_balance) are still
# fanned out per-bot because they fire rarely and carry bot-specific data.
from __future__ import annotations

from collections import deque
from typing import Dict

from .models import FairProbability, OrderbookState


class SharedFarmState:
    # Single source of truth for all market data in farm mode.

    __slots__ = (
        "fair_probs",
        "orderbooks",
        "market_asset",
        "market_settlement",
        "market_window_min",
        "market_is_range",
        "scalp_eligible",
        "scalp_settlement_epoch",
        "prev_fair_prob_by_ticker",
        "prev_fair_prob_ts_by_ticker",
        "last_fair_prob_ts_by_ticker",
        "last_truth_tick_by_asset",
        "truth_stale",
        "ws_halted",
        "data_quality_halted",
        "data_quality_halt_reason",
        # ── Regime cache (updated via bus bridge → FarmDispatcher) ──────
        "regime_vol",           # asset -> "VOL_LOW"/"VOL_NORMAL"/"VOL_HIGH"/"VOL_SPIKE"
        "regime_liq",           # asset -> "LIQ_HIGH"/"LIQ_NORMAL"/"LIQ_LOW"/"LIQ_DRIED"
        "regime_risk",          # global risk state: "RISK_ON"/"RISK_OFF"/"NEUTRAL"/"UNKNOWN"
        "regime_session",       # market -> session regime string
        "regime_last_update",   # asset -> monotonic timestamp of last regime update
        # Per-asset truth prices for cross-asset context features
        "truth_price_by_asset",         # asset -> latest mid price
        "prev_truth_price_by_asset",    # asset -> previous mid price
        # Dynamic family allocation weights (family -> multiplier)
        "family_weights",
        # Side-level guard blocks for persistent underperformance
        "side_guard_block_until",  # "ticker|side" -> unix ts when block expires
        # Phase 0 diagnostics / Phase 2-4 controls
        "projected_cap_util_by_side",     # "ticker|side" -> max projected utilization ratio
        "param_region_penalty_hits",      # counter (windowed by diagnostics reset in dispatcher)
        "param_region_block_hits",        # counter (windowed by diagnostics reset in dispatcher)
        "crowding_fill_times_by_side",    # "bot_id|ticker|side" -> deque[timestamps]
        "crowding_fills_per_sec_by_side", # "bot_id|ticker|side" -> current fills/sec
        "crowding_candidate_events",      # counter
        "crowding_active_events",         # counter (actual throttles when enabled)
        "crowding_pause_until",           # "bot_id|ticker|side" -> unix ts
        # ── Context policy engine state ──────────────────────────────────
        "context_policy_version",         # str: loaded policy version (empty = none)
        "context_policy_core_count",      # int: number of core (promoted) context keys
        "context_policy_challenger_count",  # int: number of challenger context keys
        "context_policy_explore_count",   # int: number of explore (probation) context keys
        "context_policy_weight_applied",  # counter: times policy weight was applied
        # ── Trade tape flow imbalance ────────────────────────────────────
        "trade_flow_by_ticker",       # ticker -> float in [-1, +1]; +1=all YES takers
        "orderbook_delta_flow_yes",   # ticker -> float in [-1, +1]
        "orderbook_delta_flow_no",    # ticker -> float in [-1, +1]
        # ── Drift guard state ────────────────────────────────────────────
        "drift_guard_alerts",             # counter: drift alerts triggered
        "drift_guard_auto_demotions",     # counter: auto-demotions executed
        # ── Adaptive cap state ───────────────────────────────────────────
        "adaptive_cap_events",            # counter: cap tightening events
        "adaptive_cap_cooldown_events",   # counter: cooldown activations
        "adaptive_cap_cooldown_until",    # "context_key" -> unix ts
        # ── Edge tracking state ──────────────────────────────────────────
        "edge_tracking_poor_retention_keys",  # int: count of keys with poor retention
        "edge_tracking_weight_applied",       # counter: times edge weight was applied
        # ── Population scaler state ──────────────────────────────────────
        "population_current_stage",       # int: current scaling stage index
        "population_current_multiplier",  # float: current multiplier
        "population_scale_events",        # counter: total scale events
    )

    def __init__(self) -> None:
        # High-frequency: updated on every market tick.
        self.fair_probs: Dict[str, FairProbability] = {}
        self.orderbooks: Dict[str, OrderbookState] = {}

        # Market metadata: updated once per discovery cycle (~30 s).
        self.market_asset: Dict[str, str] = {}
        self.market_settlement: Dict[str, float] = {}
        self.market_window_min: Dict[str, int] = {}
        self.market_is_range: Dict[str, bool] = {}

        # Scalper metadata (shared across all scalpers).
        self.scalp_eligible: Dict[str, bool] = {}
        self.scalp_settlement_epoch: Dict[str, float] = {}
        self.prev_fair_prob_by_ticker: Dict[str, float] = {}
        self.prev_fair_prob_ts_by_ticker: Dict[str, float] = {}
        self.last_fair_prob_ts_by_ticker: Dict[str, float] = {}

        # Truth feed liveness (monotonic timestamp per asset).
        self.last_truth_tick_by_asset: Dict[str, float] = {}

        # Global halt flags — set by FarmDispatcher.
        self.truth_stale: bool = True
        self.ws_halted: bool = False
        # Market data quality guard — set when configured live families have
        # no valid markets for longer than the halt threshold (live mode only).
        self.data_quality_halted: bool = False
        self.data_quality_halt_reason: str = ""

        # Regime cache: updated by FarmDispatcher from bus bridge events.
        # Source: core RegimeDetector → core EventBus → bridge → Kalshi Bus → FarmDispatcher.
        self.regime_vol: Dict[str, str] = {}           # asset -> vol regime string
        self.regime_liq: Dict[str, str] = {}           # asset -> liquidity regime string
        self.regime_risk: str = "UNKNOWN"               # global risk state
        self.regime_session: Dict[str, str] = {}        # market -> session regime
        self.regime_last_update: Dict[str, float] = {}  # asset -> monotonic ts
        self.truth_price_by_asset: Dict[str, float] = {}
        self.prev_truth_price_by_asset: Dict[str, float] = {}
        self.family_weights: Dict[str, float] = {}
        self.side_guard_block_until: Dict[str, float] = {}
        # Trade tape flow imbalance per ticker (updated by FarmDispatcher).
        # Value in [-1, +1]: +1 = all YES takers, -1 = all NO takers, 0 = balanced.
        self.trade_flow_by_ticker: Dict[str, float] = {}
        self.orderbook_delta_flow_yes: Dict[str, float] = {}
        self.orderbook_delta_flow_no: Dict[str, float] = {}
        self.projected_cap_util_by_side: Dict[str, float] = {}
        self.param_region_penalty_hits: int = 0
        self.param_region_block_hits: int = 0
        self.crowding_fill_times_by_side: Dict[str, deque] = {}
        self.crowding_fills_per_sec_by_side: Dict[str, float] = {}
        self.crowding_candidate_events: int = 0
        self.crowding_active_events: int = 0
        self.crowding_pause_until: Dict[str, float] = {}

        # Context policy engine
        self.context_policy_version: str = ""
        self.context_policy_core_count: int = 0
        self.context_policy_challenger_count: int = 0
        self.context_policy_explore_count: int = 0
        self.context_policy_weight_applied: int = 0
        # Drift guard
        self.drift_guard_alerts: int = 0
        self.drift_guard_auto_demotions: int = 0
        # Adaptive caps
        self.adaptive_cap_events: int = 0
        self.adaptive_cap_cooldown_events: int = 0
        self.adaptive_cap_cooldown_until: Dict[str, float] = {}
        # Edge tracking
        self.edge_tracking_poor_retention_keys: int = 0
        self.edge_tracking_weight_applied: int = 0
        # Population scaler
        self.population_current_stage: int = 0
        self.population_current_multiplier: float = 1.0
        self.population_scale_events: int = 0
