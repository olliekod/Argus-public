# Created by Oliver Meihls

# Kalshi Paper Farm runner.
#
# Runs multiple isolated bot instances (strategy/scalper/execution) on a shared
# Kalshi bus/connectors. Each bot routes events using bot_id.
#
# Note: All bots receive the same market feed (orderbook, fair prob) and the same
# ticker set. If strategy config (min_edge_threshold, signal_cooldown_s, sizing,
# etc.) is identical or very similar across configs, they will fire on the same
# signals and take the same trades, so leaderboard PnL and win rate will look
# nearly identical. To get meaningfully different results, vary those params
# per bot (e.g. different min_edge_threshold, persistence_window_ms, bankroll).
#
# Farm config can be a compact "farm" block so params are generated once and
# never drift (see farm_grid.py and load_farm_configs).

from __future__ import annotations

import asyncio
import os
import statistics
import random
import time
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .bus import Bus
from .config import KalshiConfig, load_config
from .farm_grid import MAX_BOT_COUNT, generate_farm_configs, load_dwarf_names
from .kalshi_execution import ExecutionEngine
from .kalshi_rest import KalshiRestClient
from .kalshi_strategy import StrategyEngine, _hold_family_allowed, _hold_tail_penalty
from .logging_utils import ComponentLogger
from .market_selectors import hold_entry_horizon_seconds
from .mispricing_scalper import MispricingScalper, _scalp_family_allowed
from .paper_model import estimate_kalshi_taker_fee_usd
from .decision_context import _strike_distance_bucket, _strike_distance_pct
from .decision_tape import DecisionTapeWriter, build_tape_record
from .models import (
    FairProbability,
    KalshiOrderDeltaEvent,
    MarketMetadata,
    OrderUpdate,
    OrderbookState,
    RiskEvent,
    TradeSignal,
    WsConnectionEvent,
)
from .context_policy import AdaptiveCapEngine, ContextPolicyEngine, DriftGuard, build_context_key, momentum_bucket
from .edge_tracker import EdgeTracker
from .population_scaler import PopulationScaler
from .regime_gate import RegimeGate
from .shared_state import SharedFarmState
from .simulation import (
    BotEquityLedger,
    BotRunRecord,
    FamilyWeightManager,
    FAMILIES,
    PopulationManager,
    assign_family,
    calculate_robustness_score as sim_robustness_score,
    novelty_distance,
    perturb_config_tight,
)

log = ComponentLogger("farm_runner")


@dataclass(slots=True)
class _PreparedTickerState:
    ticker: str
    now_wall: float
    now_mono: float
    p_yes: float
    yes_ask_cents: int
    no_ask_cents: int
    yes_bid_cents: int
    no_bid_cents: int
    yes_bid_depth_centicx: int
    no_bid_depth_centicx: int
    ev_yes: float
    ev_no: float
    asset: str
    last_asset_tick: float
    time_to_settle_s: float
    window_minutes: int
    is_range: bool
    scalp_eligible: bool
    prev_p_yes: Optional[float]
    prev_prob_ts: float
    last_prob_ts: float
    # Directional signals (from FairProbability.drift and trade_flow_by_ticker)
    momentum_drift: float = 0.0
    trade_flow: float = 0.0
    obi: float = 0.0
    depth_pressure: float = 0.0
    delta_flow_yes: float = 0.0
    delta_flow_no: float = 0.0
    # Regime state (from SharedFarmState cache; empty string = unknown)
    vol_regime: str = ""
    liq_regime: str = ""
    risk_regime: str = ""
    # Cross-family context (Phase 3): other asset's recent state.
    # Used for confidence/size adjustment only — never forces trade side.
    cross_asset_vol_regime: str = ""
    cross_asset_price_change_pct: float = 0.0
    own_asset_price_change_pct: float = 0.0
    family_weight: float = 1.0
    strike_distance_pct: Optional[float] = None
    settlement_epoch: float = 0.0


@dataclass(slots=True)
class _StrategyCohort:
    strategies: List[StrategyEngine]
    effective_edge_fee_pct: float
    min_edge_threshold: float
    near_expiry_min_edge: float
    near_expiry_minutes: int
    persistence_window_ms: int
    near_expiry_persistence_ms: int
    min_entry_cents: int
    max_entry_cents: int
    no_avoid_above_cents: int
    hold_tail_penalty_start_cents: int
    hold_tail_penalty_per_10c: float
    hold_min_net_edge_cents: int
    hold_entry_cost_buffer_cents: int
    yes_avoid_min_cents: int
    yes_avoid_max_cents: int
    signal_cooldown_s: float
    latency_circuit_breaker_ms: int
    max_fraction_per_market: float
    risk_fraction_per_trade: float
    max_contracts_per_ticker: int
    daily_drawdown_limit: float
    default_order_style: str
    max_entry_minutes_to_expiry: int
    range_max_entry_minutes_to_expiry: int
    live_families: Tuple[str, ...]
    shadow_families: Tuple[str, ...]
    trade_enabled_assets: Tuple[str, ...]
    shadow_assets: Tuple[str, ...]
    enable_market_side_caps: bool
    market_side_cap_contracts: int
    market_side_cap_usd: float
    family_side_cap_usd: float
    market_side_cap_enforcement_mode: str
    enable_crowding_throttle: bool
    crowding_window_s: float
    crowding_fills_per_sec_threshold: float
    crowding_qty_multiplier: float
    crowding_pause_s: float
    crowding_mode: str
    hold_min_divergence_threshold: float
    hold_require_momentum_agreement: bool
    hold_require_flow_agreement: bool
    hold_momentum_agreement_min_drift: float
    hold_flow_agreement_min_flow: float
    hold_flow_reversal_threshold: float
    scalp_momentum_min_drift: float
    obi_p_yes_bias_weight: float
    momentum_p_yes_bias_weight: float
    trade_flow_p_yes_bias_weight: float


@dataclass(slots=True)
class _ScalperEntryCohort:
    scalpers: List[MispricingScalper]
    scalp_cooldown_s: float
    scalp_max_spread_cents: int
    scalp_min_edge_cents: int
    scalp_min_profit_cents: int
    scalp_min_entry_cents: int
    scalp_max_entry_cents: int
    scalp_max_quantity: int
    risk_fraction_per_trade: float
    scalp_min_reprice_move_cents: int
    scalp_reprice_window_s: float
    scalp_entry_cost_buffer_cents: int
    scalp_directional_score_threshold: float
    scalp_directional_drift_weight: float
    scalp_directional_drift_scale: float
    scalp_directional_obi_weight: float
    scalp_directional_flow_weight: float
    scalp_directional_depth_weight: float
    scalp_directional_delta_yes_weight: float
    scalp_directional_delta_no_weight: float
    momentum_scalp_enabled: bool
    momentum_min_reprice_move_cents: int
    momentum_reprice_window_s: float
    momentum_min_orderbook_imbalance: float
    momentum_max_spread_cents: int
    momentum_min_edge_cents: int
    momentum_min_profit_cents: int
    momentum_entry_cost_buffer_cents: int
    momentum_max_quantity: int
    live_families: Tuple[str, ...]
    shadow_families: Tuple[str, ...]
    trade_enabled_assets: Tuple[str, ...]
    shadow_assets: Tuple[str, ...]
    enable_market_side_caps: bool
    market_side_cap_contracts: int
    market_side_cap_usd: float
    family_side_cap_usd: float
    market_side_cap_enforcement_mode: str
    enable_crowding_throttle: bool
    crowding_window_s: float
    crowding_fills_per_sec_threshold: float
    crowding_qty_multiplier: float
    crowding_pause_s: float
    crowding_mode: str


@dataclass(slots=True)
class _ArbEntryCohort:
    scalpers: List[MispricingScalper]
    arb_min_sum_ask_cents: int
    arb_min_net_edge_cents: int
    arb_min_entry_cents: int
    arb_max_entry_cents: int
    arb_max_quantity: int
    arb_cooldown_s: float
    risk_fraction_per_trade: float
    live_families: Tuple[str, ...]
    shadow_families: Tuple[str, ...]
    trade_enabled_assets: Tuple[str, ...]
    shadow_assets: Tuple[str, ...]
    enable_market_side_caps: bool
    market_side_cap_contracts: int
    market_side_cap_usd: float
    family_side_cap_usd: float
    market_side_cap_enforcement_mode: str
    enable_crowding_throttle: bool
    crowding_window_s: float
    crowding_fills_per_sec_threshold: float
    crowding_qty_multiplier: float
    crowding_pause_s: float
    crowding_mode: str


def _family_label(asset: str, window_minutes: int, is_range: bool) -> str:
    a = (asset or "BTC").upper()
    if is_range:
        return f"{a} Range"
    return f"{a} {int(window_minutes)}m"


def _side_guard_key(ticker: str, side: str) -> str:
    return f"{ticker}|{side}"


def _crowding_key(bot_id: str, ticker: str, side: str) -> str:
    bot = str(bot_id or "").strip()
    market = str(ticker or "").strip()
    dirn = str(side or "").strip()
    return f"{bot}|{market}|{dirn}"


def _edge_bucket(edge: float) -> str:
    if edge < 0.05:
        return "lt_0.05"
    if edge < 0.10:
        return "0.05_0.10"
    if edge < 0.20:
        return "0.10_0.20"
    return "ge_0.20"


def _price_bucket(price_cents: int) -> str:
    if price_cents < 40:
        return "lt_40"
    if price_cents < 55:
        return "40_55"
    if price_cents < 70:
        return "55_70"
    if price_cents < 78:
        return "70_78"
    return "ge_78"


def _tts_bucket(tts_s: float) -> str:
    if tts_s < 60:
        return "lt_1m"
    if tts_s < 180:
        return "1_3m"
    if tts_s < 600:
        return "3_10m"
    if tts_s < 1800:
        return "10_30m"
    return "ge_30m"


def _session_mult(now_wall: float, sleeve: str = "scalp") -> float:
    # UTC-hour-based sizing multiplier derived from top-wallet activity patterns.
    #
    # 0xd0d6 concentrates scalp in 15-21 UTC; 0x1979 concentrates holds in 11-15 UTC.
    # Overnight (00-09 UTC) reduces sizing to filter noise from thin books.
    # sleeve: "scalp" | "hold"
    from datetime import timezone as _tz
    utc_hour = datetime.fromtimestamp(now_wall, tz=_tz.utc).hour
    if 15 <= utc_hour < 21:
        return 1.20  # US afternoon: highest BTC momentum + Kalshi activity
    if 11 <= utc_hour < 15:
        return 1.10 if sleeve == "hold" else 1.05  # US morning: good for holds
    if 0 <= utc_hour < 9:
        return 0.70  # Overnight: thin books, low momentum, high noise ratio
    return 1.00


@dataclass(slots=True)
class _RegionAdjustment:
    scale_mult: float = 1.0
    blocked: bool = False
    candidate: bool = False
    rewarded: bool = False
    keys: Tuple[str, ...] = ()


class _ParamRegionPenaltyEngine:
    # Rolling scorecard for repeatedly losing parameter regions.

    def __init__(self, cfg0: KalshiConfig) -> None:
        self._enabled = bool(getattr(cfg0, "enable_param_region_penalties", False))
        self._window = int(getattr(cfg0, "param_region_window_settles", 300))
        self._min_samples = int(getattr(cfg0, "param_region_min_samples", 80))
        self._loss_threshold = float(getattr(cfg0, "param_region_loss_threshold_usd", -100.0))
        self._penalty_factor = float(getattr(cfg0, "param_region_penalty_factor", 0.5))
        self._gain_factor = float(getattr(cfg0, "param_region_gain_factor", 1.10))
        self._context_min_samples = int(getattr(cfg0, "param_region_context_min_samples", 300))
        self._loss_avg_pnl = float(getattr(cfg0, "param_region_loss_avg_pnl_usd", -2.0))
        self._gain_avg_pnl = float(getattr(cfg0, "param_region_gain_avg_pnl_usd", 2.0))
        self._block_minutes = float(getattr(cfg0, "param_region_block_minutes", 30.0))
        self._mode = str(getattr(cfg0, "param_region_mode", "downweight"))
        self._low_edge_max = float(getattr(cfg0, "param_region_low_edge_max", 0.07))
        self._low_entry_max = int(getattr(cfg0, "param_region_low_entry_max_cents", 55))
        self._low_entry_floor = int(getattr(cfg0, "param_region_low_entry_floor_cents", 40))
        self._enable_family_side = bool(getattr(cfg0, "param_region_enable_family_side", True))
        allow = getattr(cfg0, "param_region_allow_high_entry_families", []) or []
        self._allow_high_entry_families = {str(x).strip() for x in allow if str(x).strip()}
        self._context_enabled = True

        self._lock = threading.Lock()
        self._pnl_by_key: Dict[str, deque[float]] = {}
        self._block_until: Dict[str, float] = {}
        self._diag_hits: int = 0
        self._diag_blocks: int = 0
        self._diag_rewards: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _keys_for(
        self,
        cfg: KalshiConfig,
        family: str,
        side: Optional[str] = None,
        decision_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ...]:
        keys: List[str] = []
        if float(getattr(cfg, "min_edge_threshold", 0.0)) <= self._low_edge_max:
            keys.append("min_edge_low")
        if int(getattr(cfg, "max_entry_cents", 0)) <= self._low_entry_max:
            keys.append("entry_max_low")
        if int(getattr(cfg, "min_entry_cents", 0)) <= self._low_entry_floor:
            keys.append("entry_floor_low")
        if self._enable_family_side and side in ("yes", "no"):
            keys.append(f"family_side:{family}|{side}")
        if self._context_enabled and side in ("yes", "no") and decision_context:
            eb = str(decision_context.get("eb", "na"))
            pb = str(decision_context.get("pb", "na"))
            tb = str(decision_context.get("tb", "na"))
            keys.append(f"ctx:{family}|{side}|{eb}|{pb}|{tb}")
            keys.append(f"ctx4:{family}|{side}|{eb}|{pb}")
        return tuple(keys)

    def record_settlement(
        self,
        cfg: KalshiConfig,
        family: str,
        net_pnl: float,
        side: Optional[str] = None,
        decision_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        keys = self._keys_for(cfg, family, side=side, decision_context=decision_context)
        if not keys:
            return
        with self._lock:
            for key in keys:
                dq = self._pnl_by_key.get(key)
                if dq is None:
                    dq = deque(maxlen=self._window)
                    self._pnl_by_key[key] = dq
                dq.append(float(net_pnl))

    def evaluate_candidate(
        self,
        cfg: KalshiConfig,
        family: str,
        now_ts: float,
        side: Optional[str] = None,
        decision_context: Optional[Dict[str, Any]] = None,
    ) -> _RegionAdjustment:
        keys = self._keys_for(cfg, family, side=side, decision_context=decision_context)
        if not keys:
            return _RegionAdjustment()
        candidate = False
        blocked = False
        rewarded = False
        scale_mult = 1.0
        with self._lock:
            for key in keys:
                dq = self._pnl_by_key.get(key)
                if dq is None or len(dq) < self._min_samples:
                    continue
                total = float(sum(dq))
                avg = total / max(1, len(dq))
                if total <= self._loss_threshold:
                    candidate = True
                    if self._enabled and self._mode == "cooldown_block":
                        until = self._block_until.get(key, 0.0)
                        if until <= now_ts:
                            self._block_until[key] = now_ts + (self._block_minutes * 60.0)
                            until = self._block_until[key]
                        if until > now_ts:
                            blocked = True
                    elif self._enabled:
                        scale_mult = min(scale_mult, self._penalty_factor)
                # Context-level soft sizing: downweight persistent low expectancy,
                # upweight persistent high expectancy. Never hard-block here.
                if key.startswith("ctx:") or key.startswith("ctx4:"):
                    if len(dq) >= self._context_min_samples:
                        if avg <= self._loss_avg_pnl and self._enabled:
                            candidate = True
                            scale_mult = min(scale_mult, self._penalty_factor)
                        elif avg >= self._gain_avg_pnl and self._enabled:
                            rewarded = True
                            scale_mult = max(scale_mult, self._gain_factor)
            if candidate:
                self._diag_hits += 1
            if blocked:
                self._diag_blocks += 1
            if rewarded:
                self._diag_rewards += 1
        return _RegionAdjustment(
            scale_mult=max(0.0, min(2.0, scale_mult)),
            blocked=blocked,
            candidate=candidate,
            rewarded=rewarded,
            keys=keys,
        )

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            by_key: Dict[str, Dict[str, float]] = {}
            now_ts = time.time()
            for key, dq in self._pnl_by_key.items():
                samples = len(dq)
                total = float(sum(dq))
                by_key[key] = {
                    "samples": samples,
                    "total_pnl_usd": round(total, 4),
                    "blocked": bool(self._block_until.get(key, 0.0) > now_ts),
                    "block_until": float(self._block_until.get(key, 0.0)),
                }
            return {
                "enabled": self._enabled,
                "mode": self._mode,
                "window_settles": self._window,
                "min_samples": self._min_samples,
                "loss_threshold_usd": self._loss_threshold,
                "penalty_factor": self._penalty_factor,
                "gain_factor": self._gain_factor,
                "context_min_samples": self._context_min_samples,
                "loss_avg_pnl_usd": self._loss_avg_pnl,
                "gain_avg_pnl_usd": self._gain_avg_pnl,
                "low_edge_max": self._low_edge_max,
                "low_entry_max_cents": self._low_entry_max,
                "low_entry_floor_cents": self._low_entry_floor,
                "family_side_enabled": self._enable_family_side,
                "context_enabled": self._context_enabled,
                "hits": self._diag_hits,
                "blocks": self._diag_blocks,
                "rewards": self._diag_rewards,
                "regions": by_key,
            }


class _StrategyBatchEvaluator:
    # Evaluate hold-to-expiry strategies using shared per-ticker features.

    def __init__(
        self,
        strategies: List[StrategyEngine],
        regime_gate: object = None,
        shared: Optional[SharedFarmState] = None,
        param_region_engine: Optional[_ParamRegionPenaltyEngine] = None,
        context_policy: Optional[ContextPolicyEngine] = None,
        edge_tracker: Optional[EdgeTracker] = None,
        tape_writer: Optional[DecisionTapeWriter] = None,
    ) -> None:
        self._cohorts = self._build_cohorts(strategies)
        self._strategy_count = len(strategies)
        self._regime_gate = regime_gate  # Optional[RegimeGate]
        self._shared = shared
        self._param_region_engine = param_region_engine
        self._context_policy = context_policy
        self._edge_tracker = edge_tracker
        self._tape_writer = tape_writer
        self._diag_lock = threading.Lock()
        self._diag_reasons: Dict[str, int] = {}
        self._last_hold_prune_ts: float = 0.0

    def _count(self, reason: str, n: int = 1) -> None:
        if n <= 0:
            return
        with self._diag_lock:
            self._diag_reasons[reason] = self._diag_reasons.get(reason, 0) + n

    def drain_diagnostics(self) -> Dict[str, int]:
        with self._diag_lock:
            snap = dict(self._diag_reasons)
            self._diag_reasons.clear()
        return snap

    def _write_rejection(self, prepared: _PreparedTickerState, reason: str, side: Optional[str] = None) -> None:
        if self._tape_writer is None:
            return
        self._tape_writer.write_rejection(
            build_tape_record(
                prepared=prepared,
                source="mispricing_hold",
                decision="rejected",
                side=side,
                edge=0.0,
                reject_reason=reason,
                params=None,
                ctx_key=None,
                sampled_rejection=True,
            )
        )

    @staticmethod
    def _market_side_cap_max_contracts(
        cap_contracts: int,
        cap_usd: float,
        limit_cents: int,
    ) -> int:
        max_allowed = 0
        if cap_contracts > 0:
            max_allowed = cap_contracts
        if cap_usd > 0 and limit_cents > 0:
            usd_cap_contracts = int(cap_usd / max(limit_cents / 100.0, 0.01))
            if usd_cap_contracts > 0:
                max_allowed = usd_cap_contracts if max_allowed <= 0 else min(max_allowed, usd_cap_contracts)
        return max_allowed

    @staticmethod
    def _apply_market_side_cap(
        *,
        cap_enabled: bool,
        cap_contracts: int,
        cap_usd: float,
        cap_mode: str,
        limit_cents: int,
        requested_qty: int,
        planned_contracts_for_side: int,
    ) -> Tuple[int, str]:
        # Return (final_qty, reason) where reason in {"none","scaled","blocked"}.
        if requested_qty <= 0 or not cap_enabled:
            return requested_qty, "none"
        max_allowed = _StrategyBatchEvaluator._market_side_cap_max_contracts(
            cap_contracts=cap_contracts,
            cap_usd=cap_usd,
            limit_cents=limit_cents,
        )
        if max_allowed <= 0:
            return requested_qty, "none"
        remaining = max_allowed - max(0, planned_contracts_for_side)
        if remaining <= 0:
            return 0, "blocked"
        if requested_qty <= remaining:
            return requested_qty, "none"
        if cap_mode == "scale":
            return remaining, "scaled"
        return 0, "blocked"

    @staticmethod
    def _build_cohorts(strategies: List[StrategyEngine]) -> List[_StrategyCohort]:
        grouped: Dict[Tuple[Any, ...], List[StrategyEngine]] = {}
        for strategy in strategies:
            cfg = strategy._cfg
            key = (
                cfg.effective_edge_fee_pct,
                cfg.min_edge_threshold,
                cfg.near_expiry_min_edge,
                cfg.near_expiry_minutes,
                cfg.persistence_window_ms,
                cfg.near_expiry_persistence_ms,
                cfg.min_entry_cents,
                cfg.max_entry_cents,
                cfg.no_avoid_above_cents,
                cfg.hold_tail_penalty_start_cents,
                cfg.hold_tail_penalty_per_10c,
                cfg.hold_min_net_edge_cents,
                cfg.hold_entry_cost_buffer_cents,
                cfg.yes_avoid_min_cents,
                cfg.yes_avoid_max_cents,
                cfg.signal_cooldown_s,
                cfg.latency_circuit_breaker_ms,
                cfg.max_fraction_per_market,
                cfg.risk_fraction_per_trade,
                cfg.max_contracts_per_ticker,
                cfg.daily_drawdown_limit,
                cfg.default_order_style,
                cfg.max_entry_minutes_to_expiry,
                cfg.range_max_entry_minutes_to_expiry,
                tuple(cfg.live_families),
                tuple(cfg.shadow_families),
                tuple(cfg.trade_enabled_assets),
                tuple(cfg.shadow_assets),
                cfg.enable_market_side_caps,
                cfg.market_side_cap_contracts,
                cfg.market_side_cap_usd,
                cfg.family_side_cap_usd,
                cfg.market_side_cap_enforcement_mode,
                cfg.enable_crowding_throttle,
                cfg.crowding_window_s,
                cfg.crowding_fills_per_sec_threshold,
                cfg.crowding_qty_multiplier,
                cfg.crowding_pause_s,
                cfg.crowding_mode,
                cfg.hold_min_divergence_threshold,
                cfg.hold_require_momentum_agreement,
                cfg.hold_require_flow_agreement,
                cfg.hold_momentum_agreement_min_drift,
                cfg.hold_flow_agreement_min_flow,
                cfg.hold_flow_reversal_threshold,
                cfg.scalp_momentum_min_drift,
                cfg.obi_p_yes_bias_weight,
                cfg.momentum_p_yes_bias_weight,
                cfg.trade_flow_p_yes_bias_weight,
            )
            grouped.setdefault(key, []).append(strategy)

        cohorts: List[_StrategyCohort] = []
        for key, members in grouped.items():
            cohorts.append(
                _StrategyCohort(
                    strategies=members,
                    effective_edge_fee_pct=key[0],
                    min_edge_threshold=key[1],
                    near_expiry_min_edge=key[2],
                    near_expiry_minutes=key[3],
                    persistence_window_ms=key[4],
                    near_expiry_persistence_ms=key[5],
                    min_entry_cents=key[6],
                    max_entry_cents=key[7],
                    no_avoid_above_cents=key[8],
                    hold_tail_penalty_start_cents=key[9],
                    hold_tail_penalty_per_10c=key[10],
                    hold_min_net_edge_cents=key[11],
                    hold_entry_cost_buffer_cents=key[12],
                    yes_avoid_min_cents=key[13],
                    yes_avoid_max_cents=key[14],
                    signal_cooldown_s=key[15],
                    latency_circuit_breaker_ms=key[16],
                    max_fraction_per_market=key[17],
                    risk_fraction_per_trade=key[18],
                    max_contracts_per_ticker=key[19],
                    daily_drawdown_limit=key[20],
                    default_order_style=key[21],
                    max_entry_minutes_to_expiry=key[22],
                    range_max_entry_minutes_to_expiry=key[23],
                    live_families=key[24],
                    shadow_families=key[25],
                    trade_enabled_assets=key[26],
                    shadow_assets=key[27],
                    enable_market_side_caps=key[28],
                    market_side_cap_contracts=key[29],
                    market_side_cap_usd=key[30],
                    family_side_cap_usd=key[31],
                    market_side_cap_enforcement_mode=key[32],
                    enable_crowding_throttle=key[33],
                    crowding_window_s=key[34],
                    crowding_fills_per_sec_threshold=key[35],
                    crowding_qty_multiplier=key[36],
                    crowding_pause_s=key[37],
                    crowding_mode=key[38],
                    hold_min_divergence_threshold=key[39],
                    hold_require_momentum_agreement=key[40],
                    hold_require_flow_agreement=key[41],
                    hold_momentum_agreement_min_drift=key[42],
                    hold_flow_agreement_min_flow=key[43],
                    hold_flow_reversal_threshold=key[44],
                    scalp_momentum_min_drift=key[45],
                    obi_p_yes_bias_weight=key[46],
                    momentum_p_yes_bias_weight=key[47],
                    trade_flow_p_yes_bias_weight=key[48],
                )
            )
        return cohorts

    def _prune_ghost_hold_positions(self, now_wall: float) -> None:
        # Clear _hold_position_qty/_hold_position_side for expired contracts.
        #
        # Hold exits depend on reversal signals which only fire on active OB updates.
        # Once a contract expires and is unsubscribed, no more dispatches arrive,
        # so hold positions get permanently stuck without this sweep.
        # Rate-limited to once per 30s. Uses shared.market_settlement to detect expiry.
        if now_wall - self._last_hold_prune_ts < 30.0:
            return
        self._last_hold_prune_ts = now_wall
        if self._shared is None:
            return
        settlement_dict = self._shared.market_settlement
        pruned = 0
        for cohort in self._cohorts:
            for strategy in cohort.strategies:
                if not strategy._hold_position_qty:
                    continue
                expired = [
                    t for t in strategy._hold_position_qty
                    if settlement_dict.get(t, float("inf")) <= now_wall
                ]
                for t in expired:
                    strategy._hold_position_qty.pop(t, None)
                    strategy._hold_position_side.pop(t, None)
                    pruned += 1
        if pruned:
            self._count("hold_ghost_position_pruned", pruned)

    def evaluate(self, prepared: _PreparedTickerState, truth_stale: bool) -> List[TradeSignal]:
        self._prune_ghost_hold_positions(prepared.now_wall)

        if truth_stale:
            self._count("truth_or_ws_stale", self._strategy_count)
            return []

        signals: List[TradeSignal] = []
        planned_contracts_by_side: Dict[str, int] = {"yes": 0, "no": 0}
        planned_contracts_by_family_side: Dict[str, int] = {}
        family = _family_label(prepared.asset, prepared.window_minutes, prepared.is_range)
        asset = (prepared.asset or "").upper()
        for cohort in self._cohorts:
            if cohort.trade_enabled_assets and asset not in cohort.trade_enabled_assets:
                self._count("asset_not_enabled", len(cohort.strategies))
                continue
            if cohort.shadow_assets and asset in cohort.shadow_assets:
                self._count("asset_shadow_mode", len(cohort.strategies))
                continue
            if cohort.live_families and family not in cohort.live_families:
                self._count("family_not_enabled", len(cohort.strategies))
                continue
            if cohort.shadow_families and family in cohort.shadow_families:
                self._count("family_shadow_mode", len(cohort.strategies))
                continue

            if not _hold_family_allowed(
                prepared.asset,
                prepared.window_minutes,
                prepared.is_range,
            ):
                self._count("hold_family_filtered", len(cohort.strategies))
                self._write_rejection(prepared, "hold_family_filtered")
                for strategy in cohort.strategies:
                    strategy._persistence_state.pop(prepared.ticker, None)
                continue

            # Regime gate for hold entries
            if self._regime_gate is not None:
                gate_result = self._regime_gate.gate_hold(
                    asset=prepared.asset,
                    time_to_settle_s=prepared.time_to_settle_s,
                    edge=max(prepared.ev_yes, prepared.ev_no),  # raw edge before fee
                )
                if not gate_result.allowed:
                    self._count("regime_gate_hold_blocked", len(cohort.strategies))
                    self._write_rejection(prepared, "regime_gate_hold_blocked")
                    continue

            near_expiry = 0 < prepared.time_to_settle_s < (cohort.near_expiry_minutes * 60)
            max_entry_s = hold_entry_horizon_seconds(
                window_minutes=prepared.window_minutes,
                is_range=prepared.is_range,
                max_entry_minutes_to_expiry=cohort.max_entry_minutes_to_expiry,
                range_max_entry_minutes_to_expiry=cohort.range_max_entry_minutes_to_expiry,
            )
            if (
                not prepared.is_range
                and max_entry_s > 0
                and prepared.time_to_settle_s > max_entry_s
            ):
                self._count("too_early_to_expiry", len(cohort.strategies))
                self._write_rejection(prepared, "too_early_to_expiry")
                continue

            range_max_s = max_entry_s
            if (
                prepared.is_range
                and range_max_s > 0
                and prepared.time_to_settle_s > range_max_s
            ):
                self._count("range_too_far_to_expiry", len(cohort.strategies))
                self._write_rejection(prepared, "range_too_far_to_expiry")
                continue

            min_edge = (
                cohort.near_expiry_min_edge
                if near_expiry
                else cohort.min_edge_threshold
            )
            persistence_ms_eff = (
                cohort.near_expiry_persistence_ms
                if near_expiry
                else cohort.persistence_window_ms
            )
            reversed_bot_ids: set[str] = set()
            for strategy in cohort.strategies:
                if strategy._halted:
                    continue
                hold_qty_centicx = int(strategy._hold_position_qty.get(prepared.ticker, 0))
                hold_side = strategy._hold_position_side.get(prepared.ticker)
                if hold_qty_centicx <= 0 or hold_side not in ("yes", "no"):
                    continue
                min_drift = float(cohort.scalp_momentum_min_drift)
                flow_rev = float(cohort.hold_flow_reversal_threshold)
                reversal = (
                    (hold_side == "yes" and prepared.momentum_drift < -min_drift and prepared.trade_flow < -flow_rev)
                    or (hold_side == "no" and prepared.momentum_drift > min_drift and prepared.trade_flow > flow_rev)
                )
                if not reversal:
                    continue
                # Reversal exits are protective sells — cooldown must NOT block them.
                limit_cents_exit = prepared.yes_bid_cents if hold_side == "yes" else prepared.no_bid_cents
                if limit_cents_exit <= 0:
                    continue
                strategy._last_signal_time[prepared.ticker] = (hold_side, prepared.now_mono)
                self._count("hold_reversal_exit_signal")
                reversed_bot_ids.add(strategy._cfg.bot_id)
                signals.append(
                    TradeSignal(
                        market_ticker=prepared.ticker,
                        side=hold_side,
                        action="sell",
                        limit_price_cents=limit_cents_exit,
                        quantity_contracts=max(1, hold_qty_centicx // 100),
                        edge=0.0,
                        p_yes=float(prepared.p_yes),
                        timestamp=prepared.now_wall,
                        order_style="aggressive",
                        source="mispricing_hold",
                        bot_id=strategy._cfg.bot_id,
                    )
                )

            p_yes = float(prepared.p_yes)
            if cohort.obi_p_yes_bias_weight and prepared.obi:
                p_yes = max(0.02, min(0.98, p_yes + (prepared.obi * cohort.obi_p_yes_bias_weight)))
            if cohort.momentum_p_yes_bias_weight and prepared.momentum_drift:
                p_yes = max(0.02, min(0.98, p_yes + (prepared.momentum_drift * cohort.momentum_p_yes_bias_weight)))
            if cohort.trade_flow_p_yes_bias_weight and prepared.trade_flow:
                p_yes = max(0.02, min(0.98, p_yes + (prepared.trade_flow * cohort.trade_flow_p_yes_bias_weight)))

            yes_ask_prob = prepared.yes_ask_cents / 100.0
            divergence = p_yes - yes_ask_prob
            forced_side: Optional[str] = "yes" if divergence > 0 else "no"
            side_sign = 1.0 if forced_side == "yes" else -1.0
            depth_agree = max(0.0, side_sign * float(prepared.depth_pressure))
            delta_agree_raw = ((float(prepared.delta_flow_yes) - float(prepared.delta_flow_no)) * side_sign) / 2.0
            delta_agree = max(0.0, delta_agree_raw)
            confirmation = max(0.0, min(1.0, 0.5 * depth_agree + 0.5 * delta_agree))
            divergence_threshold = float(cohort.hold_min_divergence_threshold) * (1.0 - 0.2 * confirmation)
            if abs(divergence) < divergence_threshold:
                self._count("no_divergence", len(cohort.strategies))
                self._write_rejection(prepared, "no_divergence", forced_side)
                for strategy in cohort.strategies:
                    if strategy._cfg.bot_id in reversed_bot_ids:
                        continue
                    strategy._persistence_state.pop(prepared.ticker, None)
                continue
            if cohort.hold_require_momentum_agreement:
                min_drift_mag = float(cohort.hold_momentum_agreement_min_drift)
                drift_val = float(prepared.momentum_drift)
                if abs(drift_val) < min_drift_mag:
                    self._count("hold_momentum_below_min", len(cohort.strategies))
                    for strategy in cohort.strategies:
                        if strategy._cfg.bot_id in reversed_bot_ids:
                            continue
                        strategy._persistence_state.pop(prepared.ticker, None)
                    continue
                if drift_val != 0.0 and (
                    (forced_side == "yes" and drift_val < 0.0)
                    or (forced_side == "no" and drift_val > 0.0)
                ):
                    self._count("hold_momentum_disagree", len(cohort.strategies))
                    for strategy in cohort.strategies:
                        if strategy._cfg.bot_id in reversed_bot_ids:
                            continue
                        strategy._persistence_state.pop(prepared.ticker, None)
                    continue
            if cohort.hold_require_flow_agreement:
                min_flow_mag = float(cohort.hold_flow_agreement_min_flow)
                flow_val = float(prepared.trade_flow)
                if abs(flow_val) < min_flow_mag:
                    self._count("hold_flow_below_min", len(cohort.strategies))
                    for strategy in cohort.strategies:
                        if strategy._cfg.bot_id in reversed_bot_ids:
                            continue
                        strategy._persistence_state.pop(prepared.ticker, None)
                    continue
                if flow_val != 0.0 and (
                    (forced_side == "yes" and flow_val < 0.0)
                    or (forced_side == "no" and flow_val > 0.0)
                ):
                    self._count("hold_flow_disagree", len(cohort.strategies))
                    for strategy in cohort.strategies:
                        if strategy._cfg.bot_id in reversed_bot_ids:
                            continue
                        strategy._persistence_state.pop(prepared.ticker, None)
                    continue

            ev_yes = divergence
            ev_no = -divergence
            effective_yes = ev_yes - cohort.effective_edge_fee_pct
            effective_no = ev_no - cohort.effective_edge_fee_pct
            penalized_yes = effective_yes - _hold_tail_penalty(
                prepared.yes_ask_cents,
                "yes",
                cohort.hold_tail_penalty_start_cents,
                cohort.hold_tail_penalty_per_10c,
            )
            penalized_no = effective_no - _hold_tail_penalty(
                prepared.no_ask_cents,
                "no",
                cohort.hold_tail_penalty_start_cents,
                cohort.hold_tail_penalty_per_10c,
            )

            side: Optional[str] = None
            edge = 0.0
            limit_cents = 0
            if forced_side == "yes":
                if penalized_yes >= min_edge:
                    side = "yes"
                    edge = penalized_yes
                    limit_cents = prepared.yes_ask_cents
            elif forced_side == "no":
                if penalized_no >= min_edge:
                    side = "no"
                    edge = penalized_no
                    limit_cents = prepared.no_ask_cents

            if side is None:
                self._count("no_edge", len(cohort.strategies))
                self._write_rejection(prepared, "no_edge", forced_side)
                for strategy in cohort.strategies:
                    if strategy._cfg.bot_id in reversed_bot_ids:
                        continue
                    strategy._persistence_state.pop(prepared.ticker, None)
                continue

            if self._shared is not None:
                until = self._shared.side_guard_block_until.get(_side_guard_key(prepared.ticker, side), 0.0)
                if until > prepared.now_wall:
                    self._count("side_guard_blocked", len(cohort.strategies))
                    self._write_rejection(prepared, "side_guard_blocked", side)
                    continue

            if (
                (cohort.min_entry_cents > 0 and limit_cents < cohort.min_entry_cents)
                or (cohort.max_entry_cents < 100 and limit_cents > cohort.max_entry_cents)
            ):
                self._count("entry_price_out_of_range", len(cohort.strategies))
                self._write_rejection(prepared, "entry_price_out_of_range", side)
                for strategy in cohort.strategies:
                    strategy._persistence_state.pop(prepared.ticker, None)
                continue
            if (
                side == "no"
                and cohort.no_avoid_above_cents > 0
                and limit_cents >= cohort.no_avoid_above_cents
            ):
                self._count("no_tail_filter", len(cohort.strategies))
                self._write_rejection(prepared, "no_tail_filter", side)
                for strategy in cohort.strategies:
                    strategy._persistence_state.pop(prepared.ticker, None)
                continue

            if (
                side == "yes"
                and cohort.yes_avoid_min_cents > 0
                and cohort.yes_avoid_max_cents > 0
                and cohort.yes_avoid_min_cents <= limit_cents <= cohort.yes_avoid_max_cents
            ):
                self._count("yes_mid_range_filter", len(cohort.strategies))
                self._write_rejection(prepared, "yes_mid_range_filter", side)
                for strategy in cohort.strategies:
                    strategy._persistence_state.pop(prepared.ticker, None)
                continue

            for strategy in cohort.strategies:
                if strategy._cfg.bot_id in reversed_bot_ids:
                    continue
                if strategy._halted:
                    self._count("halted")
                    continue

                if prepared.last_asset_tick and (
                    prepared.now_mono - prepared.last_asset_tick
                ) > strategy._cfg.truth_feed_stale_timeout_s:
                    self._count("truth_age_stale")
                    continue

                if strategy._daily_pnl <= -strategy._start_bankroll * cohort.daily_drawdown_limit:
                    if not strategy._halted:
                        strategy._halted = True
                        strategy._halt_reason = "drawdown"
                        log.warning("Daily drawdown limit breached - halting [%s]", strategy._cfg.bot_id)
                    self._count("drawdown_breach")
                    continue

                # Hold sleeve strict net-edge gate after one-way entry friction.
                # (Hold entries are assumed to settle, so we model entry fee + entry slippage.)
                hold_fee_cents = int(round(estimate_kalshi_taker_fee_usd(limit_cents, 100) * 100.0))
                # Dynamic entry slippage estimate from current side spread, with config floor.
                side_spread_cents = (
                    max(0, int(prepared.yes_ask_cents) - int(prepared.yes_bid_cents))
                    if side == "yes"
                    else max(0, int(prepared.no_ask_cents) - int(prepared.no_bid_cents))
                )
                spread_slippage_cents = max(0, int(round(side_spread_cents * 0.5)))
                cfg_slippage_floor = max(0, int(getattr(strategy._cfg, "paper_slippage_cents", 0)))
                entry_slippage_cents = max(cfg_slippage_floor, spread_slippage_cents)
                net_edge_cents = int(round(float(edge) * 100.0)) - hold_fee_cents - entry_slippage_cents
                required_net_edge_cents = (
                    int(cohort.hold_min_net_edge_cents) + int(cohort.hold_entry_cost_buffer_cents)
                )
                if net_edge_cents < required_net_edge_cents:
                    self._count("hold_net_edge_too_low")
                    continue

                if persistence_ms_eff > 0:
                    prev = strategy._persistence_state.get(prepared.ticker)
                    if prev is None or prev[0] != side:
                        strategy._persistence_state[prepared.ticker] = (side, prepared.now_mono)
                        self._count("persistence")
                        continue
                    if (prepared.now_mono - prev[1]) * 1000 < persistence_ms_eff:
                        self._count("persistence")
                        continue

                cooldown_s = cohort.signal_cooldown_s
                if cooldown_s > 0:
                    last = strategy._last_signal_time.get(prepared.ticker)
                    if last is not None and last[0] == side and (prepared.now_mono - last[1]) < cooldown_s:
                        self._count("signal_cooldown")
                        continue

                latency_limit_ms = cohort.latency_circuit_breaker_ms
                if latency_limit_ms > 0 and prepared.last_asset_tick > 0:
                    if (prepared.now_mono - prepared.last_asset_tick) * 1000 > latency_limit_ms:
                        self._count("latency_breaker")
                        continue

                current_position_centicx = abs(strategy._positions.get(prepared.ticker, 0))
                current_position_contracts = current_position_centicx // 100
                max_usd = strategy._current_balance * cohort.max_fraction_per_market
                max_contracts_limit = int(max_usd / (limit_cents / 100.0)) if limit_cents > 0 else 0
                risk_usd_per_trade = strategy._current_balance * cohort.risk_fraction_per_trade
                cost_per_contract = max(limit_cents / 100.0, 0.01)
                base_qty = max(1, int(risk_usd_per_trade / cost_per_contract))
                qty = max(base_qty, int(base_qty * (edge / min_edge))) if min_edge > 0 else base_qty
                # Apply family allocation weight + cross-asset context multiplier.
                context_mult = 1.0
                if prepared.cross_asset_vol_regime == "VOL_SPIKE":
                    context_mult *= 0.90
                move_abs = abs(prepared.cross_asset_price_change_pct)
                if move_abs >= 0.006:
                    context_mult *= 1.15
                elif move_abs >= 0.002:
                    context_mult *= 1.05
                if prepared.risk_regime == "RISK_OFF":
                    context_mult *= 0.90
                # Own-asset momentum gate (wallet-derived: 0x1979 only holds with momentum).
                # Counter-trend entries get 40% size reduction.
                OWN_MOMENTUM_GATE_PCT = 0.002
                own_pct = prepared.own_asset_price_change_pct
                if side == "yes" and own_pct < -OWN_MOMENTUM_GATE_PCT:
                    context_mult *= 0.60
                elif side == "no" and own_pct > OWN_MOMENTUM_GATE_PCT:
                    context_mult *= 0.60
                context_mult *= _session_mult(prepared.now_wall, sleeve="hold")
                qty = max(1, int(qty * context_mult))

                # Apply regime gate qty_multiplier if active
                if self._regime_gate is not None:
                    gate_result = self._regime_gate.gate_hold(
                        asset=prepared.asset,
                        time_to_settle_s=prepared.time_to_settle_s,
                        edge=edge,
                    )
                    if gate_result.allowed and gate_result.qty_multiplier < 1.0:
                        qty = max(1, int(qty * gate_result.qty_multiplier))

                ticker_cap = cohort.max_contracts_per_ticker
                if ticker_cap > 0:
                    qty = min(qty, ticker_cap - current_position_contracts)
                qty = min(qty, max_contracts_limit - current_position_contracts)
                if qty <= 0:
                    self._count("max_position_reached")
                    continue

                if self._shared is not None:
                    crowd_key = _crowding_key(strategy._cfg.bot_id, prepared.ticker, side)
                    fills_per_sec = float(self._shared.crowding_fills_per_sec_by_side.get(crowd_key, 0.0))
                    is_crowded = fills_per_sec >= float(cohort.crowding_fills_per_sec_threshold)
                    if is_crowded:
                        self._shared.crowding_candidate_events += 1
                    if cohort.enable_crowding_throttle:
                        if cohort.crowding_mode == "pause":
                            until = float(self._shared.crowding_pause_until.get(crowd_key, 0.0))
                            if until > prepared.now_wall:
                                self._count("crowding_pause_active")
                                continue
                            if is_crowded and cohort.crowding_pause_s > 0:
                                self._shared.crowding_pause_until[crowd_key] = (
                                    prepared.now_wall + float(cohort.crowding_pause_s)
                                )
                                self._shared.crowding_active_events += 1
                                self._count("crowding_paused")
                                continue
                        else:
                            if is_crowded and 0.0 < cohort.crowding_qty_multiplier < 1.0:
                                scaled_qty = max(1, int(qty * cohort.crowding_qty_multiplier))
                                if scaled_qty < qty:
                                    qty = scaled_qty
                                    self._shared.crowding_active_events += 1
                                    self._count("crowding_scaled")

                if self._param_region_engine is not None:
                    candidate_ctx = {
                        "eb": _edge_bucket(float(edge)),
                        "pb": _price_bucket(int(limit_cents)),
                        "tb": _tts_bucket(float(prepared.time_to_settle_s)),
                        "drift": round(prepared.momentum_drift, 6) if prepared.momentum_drift else None,
                        "flow": round(prepared.trade_flow, 4) if prepared.trade_flow else None,
                    }
                    adj = self._param_region_engine.evaluate_candidate(
                        strategy._cfg,
                        family=family,
                        now_ts=prepared.now_wall,
                        side=side,
                        decision_context=candidate_ctx,
                    )
                    if adj.candidate:
                        self._count("param_region_penalty_candidate")
                    if adj.blocked:
                        self._count("param_region_blocked")
                        if self._shared is not None:
                            self._shared.param_region_block_hits += 1
                        continue
                    if abs(adj.scale_mult - 1.0) > 1e-6:
                        scaled_qty = max(1, int(qty * adj.scale_mult))
                        if scaled_qty < qty:
                            qty = scaled_qty
                            self._count("param_region_downweighted")
                            if self._shared is not None:
                                self._shared.param_region_penalty_hits += 1
                        elif scaled_qty > qty:
                            qty = scaled_qty
                            self._count("param_region_upweighted")

                sd_pct = prepared.strike_distance_pct
                sdb = _strike_distance_bucket(sd_pct, strategy._cfg.strike_distance_bucket_edges)
                near_money = bool(sd_pct is not None and sd_pct <= float(strategy._cfg.near_money_pct))
                entry_drift = float(getattr(prepared, "momentum_drift", 0.0))
                ctx_key = build_context_key(
                    family=family,
                    side=side,
                    edge_bucket=_edge_bucket(float(edge)),
                    price_bucket=_price_bucket(int(limit_cents)),
                    strike_distance_bucket=sdb,
                    near_money=near_money,
                    momentum=momentum_bucket(entry_drift),
                )

                nm_mode = str(getattr(strategy._cfg, "near_money_penalty_mode", "off"))
                if near_money and nm_mode == "hard":
                    self._count("near_money_hard_block")
                    continue
                if near_money and nm_mode == "soft":
                    nm_mult = float(getattr(strategy._cfg, "near_money_penalty_multiplier", 1.0))
                    if 0.0 < nm_mult < 1.0:
                        qty = max(1, int(qty * nm_mult))
                        self._count("near_money_soft_scaled")

                # Context policy weight (soft allocation engine)
                if self._context_policy is not None:
                    cp_weight = self._context_policy.get_weight(ctx_key)
                    if cp_weight == 0.0:
                        self._count("context_policy_hard_blocked")
                        continue
                    elif abs(cp_weight - 1.0) > 1e-6:
                        qty = max(1, int(qty * cp_weight))
                        self._count("context_policy_applied")
                        if self._shared is not None:
                            self._shared.context_policy_weight_applied += 1

                # Edge retention weight
                if self._edge_tracker is not None:
                    et_mult = self._edge_tracker.get_weight_multiplier(ctx_key)
                    if et_mult < 1.0:
                        qty = max(1, int(qty * et_mult))
                        self._count("edge_retention_decay")
                        if self._shared is not None:
                            self._shared.edge_tracking_weight_applied += 1

                cap_max_allowed = self._market_side_cap_max_contracts(
                    cap_contracts=cohort.market_side_cap_contracts,
                    cap_usd=cohort.market_side_cap_usd,
                    limit_cents=limit_cents,
                )
                if cap_max_allowed > 0 and self._shared is not None:
                    cap_key = _side_guard_key(prepared.ticker, side)
                    projected = planned_contracts_by_side.get(side, 0) + max(0, qty)
                    util = min(10.0, projected / max(1, cap_max_allowed))
                    prev_util = self._shared.projected_cap_util_by_side.get(cap_key, 0.0)
                    if util > prev_util:
                        self._shared.projected_cap_util_by_side[cap_key] = util

                qty, cap_reason = self._apply_market_side_cap(
                    cap_enabled=cohort.enable_market_side_caps,
                    cap_contracts=cohort.market_side_cap_contracts,
                    cap_usd=cohort.market_side_cap_usd,
                    cap_mode=cohort.market_side_cap_enforcement_mode,
                    limit_cents=limit_cents,
                    requested_qty=qty,
                    planned_contracts_for_side=planned_contracts_by_side.get(side, 0),
                )
                if cap_reason == "blocked" or qty <= 0:
                    self._count("market_side_cap_blocked")
                    continue
                if cap_reason == "scaled":
                    self._count("market_side_cap_scaled")

                family_side_key = f"{family}|{side}"
                family_cap_allowed = self._market_side_cap_max_contracts(
                    cap_contracts=0,
                    cap_usd=cohort.family_side_cap_usd,
                    limit_cents=limit_cents,
                )
                if family_cap_allowed > 0 and self._shared is not None:
                    fam_cap_key = f"family:{family_side_key}"
                    fam_projected = planned_contracts_by_family_side.get(family_side_key, 0) + max(0, qty)
                    fam_util = min(10.0, fam_projected / max(1, family_cap_allowed))
                    prev_fam_util = self._shared.projected_cap_util_by_side.get(fam_cap_key, 0.0)
                    if fam_util > prev_fam_util:
                        self._shared.projected_cap_util_by_side[fam_cap_key] = fam_util
                qty, family_cap_reason = self._apply_market_side_cap(
                    cap_enabled=cohort.enable_market_side_caps and cohort.family_side_cap_usd > 0,
                    cap_contracts=0,
                    cap_usd=cohort.family_side_cap_usd,
                    cap_mode=cohort.market_side_cap_enforcement_mode,
                    limit_cents=limit_cents,
                    requested_qty=qty,
                    planned_contracts_for_side=planned_contracts_by_family_side.get(family_side_key, 0),
                )
                if family_cap_reason == "blocked" or qty <= 0:
                    self._count("family_side_cap_blocked")
                    continue
                if family_cap_reason == "scaled":
                    self._count("family_side_cap_scaled")

                order_style = cohort.default_order_style
                order_limit_cents = limit_cents
                if order_style == "passive":
                    if side == "yes":
                        order_limit_cents = prepared.yes_bid_cents if prepared.yes_bid_cents > 0 else limit_cents
                    else:
                        order_limit_cents = prepared.no_bid_cents if prepared.no_bid_cents > 0 else limit_cents

                strategy._last_signal_time[prepared.ticker] = (side, prepared.now_mono)
                if self._edge_tracker is not None:
                    self._edge_tracker.record_entry(ctx_key, float(edge), int(order_limit_cents))
                self._count("signal_emitted")
                if self._tape_writer is not None:
                    self._tape_writer.write_signal(
                        build_tape_record(
                            prepared=prepared,
                            source="mispricing_hold",
                            decision="signal",
                            side=side,
                            edge=edge,
                            reject_reason=None,
                            params={
                                "min_entry_cents": strategy._cfg.min_entry_cents,
                                "max_entry_cents": strategy._cfg.max_entry_cents,
                                "min_edge_threshold": cohort.min_edge_threshold,
                                "persistence_window_ms": cohort.persistence_window_ms,
                                "scalp_min_edge_cents": 0,
                                "scalp_min_profit_cents": 0,
                            },
                            ctx_key=ctx_key,
                            sampled_rejection=False,
                            quantity_contracts=qty,
                        )
                    )
                signals.append(
                    TradeSignal(
                        market_ticker=prepared.ticker,
                        side=side,
                        action="buy",
                        limit_price_cents=order_limit_cents,
                        quantity_contracts=qty,
                        edge=round(edge, 4),
                        p_yes=round(prepared.p_yes, 4),
                        timestamp=prepared.now_wall,
                        order_style=order_style,
                        source="mispricing_hold",
                        bot_id=strategy._cfg.bot_id,
                    )
                )
                planned_contracts_by_side[side] = planned_contracts_by_side.get(side, 0) + max(0, qty)
                planned_contracts_by_family_side[family_side_key] = (
                    planned_contracts_by_family_side.get(family_side_key, 0) + max(0, qty)
                )
        return signals


class _ScalperBatchEvaluator:
    # Evaluate scalper entry/exit with shared per-ticker features.

    def __init__(
        self,
        scalpers: List[MispricingScalper],
        regime_gate: object = None,
        shared: Optional[SharedFarmState] = None,
        param_region_engine: Optional[_ParamRegionPenaltyEngine] = None,
        context_policy: Optional[ContextPolicyEngine] = None,
        edge_tracker: Optional[EdgeTracker] = None,
        tape_writer: Optional[DecisionTapeWriter] = None,
    ) -> None:
        self._scalpers = scalpers
        self._entry_cohorts = self._build_entry_cohorts(scalpers)
        self._scalper_count = len(scalpers)
        self._regime_gate = regime_gate  # Optional[RegimeGate]
        self._shared = shared
        self._param_region_engine = param_region_engine
        self._context_policy = context_policy
        self._edge_tracker = edge_tracker
        self._tape_writer = tape_writer
        self._diag_lock = threading.Lock()
        self._diag_reasons: Dict[str, int] = {}
        self._last_prune_ts: float = 0.0

    def _count(self, reason: str, n: int = 1) -> None:
        if n <= 0:
            return
        with self._diag_lock:
            self._diag_reasons[reason] = self._diag_reasons.get(reason, 0) + n

    def drain_diagnostics(self) -> Dict[str, int]:
        with self._diag_lock:
            snap = dict(self._diag_reasons)
            self._diag_reasons.clear()
        return snap

    def _write_rejection(self, prepared: _PreparedTickerState, reason: str, side: Optional[str] = None) -> None:
        if self._tape_writer is None:
            return
        self._tape_writer.write_rejection(
            build_tape_record(
                prepared=prepared,
                source="mispricing_scalp",
                decision="rejected",
                side=side,
                edge=0.0,
                reject_reason=reason,
                params=None,
                ctx_key=None,
                sampled_rejection=True,
            )
        )

    @staticmethod
    def _market_side_cap_max_contracts(
        cap_contracts: int,
        cap_usd: float,
        limit_cents: int,
    ) -> int:
        max_allowed = 0
        if cap_contracts > 0:
            max_allowed = cap_contracts
        if cap_usd > 0 and limit_cents > 0:
            usd_cap_contracts = int(cap_usd / max(limit_cents / 100.0, 0.01))
            if usd_cap_contracts > 0:
                max_allowed = usd_cap_contracts if max_allowed <= 0 else min(max_allowed, usd_cap_contracts)
        return max_allowed

    @staticmethod
    def _apply_market_side_cap(
        *,
        cap_enabled: bool,
        cap_contracts: int,
        cap_usd: float,
        cap_mode: str,
        limit_cents: int,
        requested_qty: int,
        planned_contracts_for_side: int,
    ) -> Tuple[int, str]:
        # Return (final_qty, reason) where reason in {"none","scaled","blocked"}.
        if requested_qty <= 0 or not cap_enabled:
            return requested_qty, "none"
        max_allowed = _ScalperBatchEvaluator._market_side_cap_max_contracts(
            cap_contracts=cap_contracts,
            cap_usd=cap_usd,
            limit_cents=limit_cents,
        )
        if max_allowed <= 0:
            return requested_qty, "none"
        remaining = max_allowed - max(0, planned_contracts_for_side)
        if remaining <= 0:
            return 0, "blocked"
        if requested_qty <= remaining:
            return requested_qty, "none"
        if cap_mode == "scale":
            return remaining, "scaled"
        return 0, "blocked"

    @staticmethod
    def _build_entry_cohorts(scalpers: List[MispricingScalper]) -> List[_ScalperEntryCohort]:
        grouped: Dict[Tuple[Any, ...], List[MispricingScalper]] = {}
        for scalper in scalpers:
            cfg = scalper._cfg
            if not bool(getattr(cfg, "scalper_enabled", True)):
                continue
            key = (
                cfg.scalp_cooldown_s,
                cfg.scalp_max_spread_cents,
                cfg.scalp_min_edge_cents,
                cfg.scalp_min_profit_cents,
                cfg.scalp_min_entry_cents,
                cfg.scalp_max_entry_cents,
                cfg.scalp_max_quantity,
                cfg.risk_fraction_per_trade,
                cfg.scalp_min_reprice_move_cents,
                cfg.scalp_reprice_window_s,
                cfg.scalp_entry_cost_buffer_cents,
                cfg.scalp_directional_score_threshold,
                cfg.scalp_directional_drift_weight,
                cfg.scalp_directional_drift_scale,
                cfg.scalp_directional_obi_weight,
                cfg.scalp_directional_flow_weight,
                cfg.scalp_directional_depth_weight,
                cfg.scalp_directional_delta_yes_weight,
                cfg.scalp_directional_delta_no_weight,
                cfg.momentum_scalp_enabled,
                cfg.momentum_min_reprice_move_cents,
                cfg.momentum_reprice_window_s,
                cfg.momentum_min_orderbook_imbalance,
                cfg.momentum_max_spread_cents,
                cfg.momentum_min_edge_cents,
                cfg.momentum_min_profit_cents,
                cfg.momentum_entry_cost_buffer_cents,
                cfg.momentum_max_quantity,
                tuple(cfg.live_families),
                tuple(cfg.shadow_families),
                tuple(cfg.trade_enabled_assets),
                tuple(cfg.shadow_assets),
                cfg.enable_market_side_caps,
                cfg.market_side_cap_contracts,
                cfg.market_side_cap_usd,
                cfg.family_side_cap_usd,
                cfg.market_side_cap_enforcement_mode,
                cfg.enable_crowding_throttle,
                cfg.crowding_window_s,
                cfg.crowding_fills_per_sec_threshold,
                cfg.crowding_qty_multiplier,
                cfg.crowding_pause_s,
                cfg.crowding_mode,
            )
            grouped.setdefault(key, []).append(scalper)

        cohorts: List[_ScalperEntryCohort] = []
        for key, members in grouped.items():
            cohorts.append(
                _ScalperEntryCohort(
                    scalpers=members,
                    scalp_cooldown_s=key[0],
                    scalp_max_spread_cents=key[1],
                    scalp_min_edge_cents=key[2],
                    scalp_min_profit_cents=key[3],
                    scalp_min_entry_cents=key[4],
                    scalp_max_entry_cents=key[5],
                    scalp_max_quantity=key[6],
                    risk_fraction_per_trade=key[7],
                    scalp_min_reprice_move_cents=key[8],
                    scalp_reprice_window_s=key[9],
                    scalp_entry_cost_buffer_cents=key[10],
                    scalp_directional_score_threshold=key[11],
                    scalp_directional_drift_weight=key[12],
                    scalp_directional_drift_scale=key[13],
                    scalp_directional_obi_weight=key[14],
                    scalp_directional_flow_weight=key[15],
                    scalp_directional_depth_weight=key[16],
                    scalp_directional_delta_yes_weight=key[17],
                    scalp_directional_delta_no_weight=key[18],
                    momentum_scalp_enabled=key[19],
                    momentum_min_reprice_move_cents=key[20],
                    momentum_reprice_window_s=key[21],
                    momentum_min_orderbook_imbalance=key[22],
                    momentum_max_spread_cents=key[23],
                    momentum_min_edge_cents=key[24],
                    momentum_min_profit_cents=key[25],
                    momentum_entry_cost_buffer_cents=key[26],
                    momentum_max_quantity=key[27],
                    live_families=key[28],
                    shadow_families=key[29],
                    trade_enabled_assets=key[30],
                    shadow_assets=key[31],
                    enable_market_side_caps=key[32],
                    market_side_cap_contracts=key[33],
                    market_side_cap_usd=key[34],
                    family_side_cap_usd=key[35],
                    market_side_cap_enforcement_mode=key[36],
                    enable_crowding_throttle=key[37],
                    crowding_window_s=key[38],
                    crowding_fills_per_sec_threshold=key[39],
                    crowding_qty_multiplier=key[40],
                    crowding_pause_s=key[41],
                    crowding_mode=key[42],
                )
            )
        return cohorts

    def _prune_ghost_positions(self, now_wall: float) -> None:
        # Clear open positions for expired contracts (no more OB updates → exit never fires).
        #
        # Rate-limited to once per 30s since it scans all bots.
        #
        # Two expiry conditions are checked:
        # 1. settlement_time_epoch[ticker] <= now  — normal case, metadata known.
        # 2. position age > 45 minutes             — fallback for tickers whose metadata
        # never arrived (settlement_ts is None), which would otherwise never expire.
        # 45 min covers the longest live contract window (60m) plus a margin.
        #
        # NOTE: scalp_settlement_epoch entries are intentionally NOT cleaned in
        # _prune_stale_tickers when a ticker goes stale. Cleaning them would break
        # this method — the old settlement time is exactly what lets us detect that
        # a ghost position is on an expired contract.
        if now_wall - self._last_prune_ts < 30.0:
            return
        self._last_prune_ts = now_wall
        if not self._scalpers:
            return
        epoch_dict = self._scalpers[0]._settlement_time_epoch
        _MAX_POSITION_AGE_S = 2700.0  # 45 minutes: fallback for None-settlement tickers
        pruned = 0
        for scalper in self._scalpers:
            if not scalper._open_positions:
                continue
            expired = [
                t for t, pos in scalper._open_positions.items()
                if (epoch_dict.get(t, float("inf")) <= now_wall
                    or (now_wall - pos.opened_at) > _MAX_POSITION_AGE_S)
            ]
            for t in expired:
                del scalper._open_positions[t]
                pruned += 1
        if pruned:
            self._count("scalp_ghost_position_pruned", pruned)

    def evaluate(self, prepared: _PreparedTickerState) -> List[TradeSignal]:
        # Periodically sweep all bots for ghost positions (expired contracts whose
        # WS was unsubscribed, so exit_sync never fires for them).
        self._prune_ghost_positions(prepared.now_wall)

        signals: List[TradeSignal] = []
        planned_contracts_by_side: Dict[str, int] = {"yes": 0, "no": 0}
        planned_contracts_by_family_side: Dict[str, int] = {}
        cfg_slippage_floor = (
            max(0, int(getattr(self._scalpers[0]._cfg, "paper_slippage_cents", 0)))
            if self._scalpers
            else 0
        )
        exited_scalpers: set[int] = set()
        family = _family_label(prepared.asset, prepared.window_minutes, prepared.is_range)
        asset = (prepared.asset or "").upper()

        for scalper in self._scalpers:
            pos = scalper._open_positions.get(prepared.ticker)
            if pos is None:
                continue

            current_bid = prepared.yes_bid_cents if pos.side == "yes" else prepared.no_bid_cents
            side_spread_cents = (
                max(0, int(prepared.yes_ask_cents) - int(prepared.yes_bid_cents))
                if pos.side == "yes"
                else max(0, int(prepared.no_ask_cents) - int(prepared.no_bid_cents))
            )
            slippage_buffer = max(
                cfg_slippage_floor * 2,
                max(1, int(round(side_spread_cents * 0.5))),
            )
            age_minutes = (prepared.now_wall - pos.opened_at) / 60.0
            min_profit = scalper._cfg.scalp_min_profit_cents
            max_hold = scalper._cfg.scalp_max_hold_minutes
            held_fair_cents = round(prepared.p_yes * 100) if pos.side == "yes" else round((1.0 - prepared.p_yes) * 100)
            held_ask_cents = prepared.yes_ask_cents if pos.side == "yes" else prepared.no_ask_cents
            held_edge_cents = held_fair_cents - held_ask_cents
            grace_elapsed = (prepared.now_wall - pos.opened_at) >= scalper._cfg.scalp_exit_grace_s

            settlement_ts = scalper._settlement_time_epoch.get(prepared.ticker)
            if settlement_ts is not None and settlement_ts > prepared.now_wall:
                minutes_to_settlement = (settlement_ts - prepared.now_wall) / 60.0
                effective_max_hold = min(max_hold, minutes_to_settlement - 0.5)
                should_exit = minutes_to_settlement <= 2.0
            elif settlement_ts is not None and settlement_ts <= prepared.now_wall:
                # Contract already expired — position is a ghost, clear immediately.
                del scalper._open_positions[prepared.ticker]
                exited_scalpers.add(id(scalper))
                self._count("scalp_ghost_position_pruned")
                continue
            else:
                effective_max_hold = max_hold
                should_exit = False

            stop_loss_cents = int(getattr(scalper._cfg, "scalp_stop_loss_cents", 0))
            if not should_exit:
                if current_bid >= pos.entry_price_cents + min_profit:
                    should_exit = True
                elif (
                    stop_loss_cents > 0
                    and grace_elapsed
                    and current_bid <= pos.entry_price_cents - stop_loss_cents
                ):
                    should_exit = True
                elif (
                    grace_elapsed
                    and current_bid >= pos.entry_price_cents + max(1, slippage_buffer // 2)
                    and held_edge_cents <= scalper._cfg.scalp_exit_edge_threshold_cents
                ):
                    should_exit = True
                elif age_minutes >= effective_max_hold and current_bid > pos.entry_price_cents:
                    should_exit = True
                elif age_minutes >= effective_max_hold * 2 and held_edge_cents <= 0:
                    should_exit = True

            if not should_exit:
                continue

            del scalper._open_positions[prepared.ticker]
            exited_scalpers.add(id(scalper))
            self._count("scalp_exit_signal")
            fp_obj = scalper._fair_probs.get(prepared.ticker)
            p_yes_val = fp_obj.p_yes if hasattr(fp_obj, "p_yes") else float(fp_obj or 0.5)
            signals.append(
                TradeSignal(
                    market_ticker=prepared.ticker,
                    side=pos.side,
                    action="sell",
                    limit_price_cents=current_bid,
                    quantity_contracts=max(1, pos.quantity_centicx // 100),
                    edge=float(current_bid - pos.entry_price_cents),
                    p_yes=p_yes_val,
                    timestamp=prepared.now_wall,
                    order_style="aggressive",
                    source=getattr(pos, "entry_source", "mispricing_scalp"),
                    bot_id=scalper._cfg.bot_id,
                )
            )

        if not prepared.scalp_eligible:
            self._count("scalp_family_filtered", 1)
            self._write_rejection(prepared, "scalp_family_filtered")
            return signals

        spread = max(
            0,
            (prepared.yes_ask_cents - prepared.yes_bid_cents)
            + (prepared.no_ask_cents - prepared.no_bid_cents),
        )

        for cohort in self._entry_cohorts:
            if cohort.trade_enabled_assets and asset not in cohort.trade_enabled_assets:
                self._count("scalp_asset_not_enabled", len(cohort.scalpers))
                continue
            if cohort.shadow_assets and asset in cohort.shadow_assets:
                self._count("scalp_asset_shadow_mode", len(cohort.scalpers))
                continue
            if cohort.live_families and family not in cohort.live_families:
                self._count("scalp_family_not_enabled", len(cohort.scalpers))
                continue
            if cohort.shadow_families and family in cohort.shadow_families:
                self._count("scalp_family_shadow_mode", len(cohort.scalpers))
                continue

            if spread > cohort.scalp_max_spread_cents:
                self._count("scalp_spread_too_wide", len(cohort.scalpers))
                self._write_rejection(prepared, "scalp_spread_too_wide")
                continue

            drift_scale = max(1e-9, float(cohort.scalp_directional_drift_scale))
            drift_component = max(-1.0, min(1.0, float(prepared.momentum_drift) / drift_scale))
            directional_score = (
                float(cohort.scalp_directional_drift_weight) * drift_component
                + float(cohort.scalp_directional_obi_weight) * float(prepared.obi)
                + float(cohort.scalp_directional_flow_weight) * float(prepared.trade_flow)
                + float(cohort.scalp_directional_depth_weight) * float(prepared.depth_pressure)
                + float(cohort.scalp_directional_delta_yes_weight) * float(prepared.delta_flow_yes)
                + float(cohort.scalp_directional_delta_no_weight) * (-float(prepared.delta_flow_no))
            )
            directional_score = max(-1.0, min(1.0, directional_score))
            threshold = float(cohort.scalp_directional_score_threshold)
            if abs(directional_score) < threshold:
                self._count("scalp_directional_below_threshold", len(cohort.scalpers))
                self._write_rejection(prepared, "scalp_directional_below_threshold")
                continue

            best_side = "yes" if directional_score > 0 else "no"
            best_limit_cents = prepared.yes_ask_cents if best_side == "yes" else prepared.no_ask_cents
            if cohort.scalp_min_entry_cents > 0 and best_limit_cents < cohort.scalp_min_entry_cents:
                self._count("scalp_entry_price_out_of_range", len(cohort.scalpers))
                self._write_rejection(prepared, "scalp_entry_price_out_of_range", best_side)
                continue
            if cohort.scalp_max_entry_cents < 100 and best_limit_cents > cohort.scalp_max_entry_cents:
                self._count("scalp_entry_price_out_of_range", len(cohort.scalpers))
                self._write_rejection(prepared, "scalp_entry_price_out_of_range", best_side)
                continue

            target_move_cents = int(cohort.scalp_min_profit_cents)
            target_exit_cents = min(99, int(best_limit_cents) + target_move_cents)
            slippage_buffer = max(
                cfg_slippage_floor * 2,
                max(1, int(round(spread * 0.5))),
            )
            round_trip_fee_cents = int(
                round(
                    (
                        estimate_kalshi_taker_fee_usd(best_limit_cents, 100)
                        + estimate_kalshi_taker_fee_usd(target_exit_cents, 100)
                    )
                    * 100.0
                )
            )
            projected_net_profit = target_move_cents - slippage_buffer - round_trip_fee_cents
            if projected_net_profit < int(cohort.scalp_entry_cost_buffer_cents):
                self._count("scalp_projected_profit_too_low", len(cohort.scalpers))
                self._write_rejection(prepared, "scalp_projected_profit_too_low", best_side)
                continue

            if self._regime_gate is not None:
                gate_result = self._regime_gate.gate_scalp(
                    asset=prepared.asset,
                    spread_cents=spread,
                    reprice_move_cents=max(1, int(round(abs(directional_score) * 100))),
                    projected_net_edge_cents=projected_net_profit,
                )
                if not gate_result.allowed:
                    self._count("regime_gate_scalp_blocked", len(cohort.scalpers))
                    self._write_rejection(prepared, "regime_gate_scalp_blocked", best_side)
                    continue

            if self._shared is not None:
                until = self._shared.side_guard_block_until.get(_side_guard_key(prepared.ticker, best_side), 0.0)
                if until > prepared.now_wall:
                    self._count("side_guard_blocked", len(cohort.scalpers))
                    self._write_rejection(prepared, "side_guard_blocked", best_side)
                    continue

            best_edge = float(abs(directional_score))
            for scalper in cohort.scalpers:
                if id(scalper) in exited_scalpers:
                    self._count("scalp_recent_exit_skip")
                    continue
                if prepared.ticker in scalper._open_positions:
                    self._count("scalp_already_in_position")
                    continue
                if not scalper._scalp_eligible.get(prepared.ticker, True):
                    self._count("scalp_not_eligible")
                    continue
                if cohort.scalp_cooldown_s > 0 and (
                    prepared.now_wall - scalper._last_signal_ts.get(prepared.ticker, 0.0)
                ) < cohort.scalp_cooldown_s:
                    self._count("scalp_cooldown")
                    continue
                _ARB_GUARD_S = 120.0
                arb_ts = scalper._arb_last_entry_ts.get(prepared.ticker, 0.0)
                if arb_ts > 0 and (prepared.now_wall - arb_ts) < _ARB_GUARD_S:
                    self._count("scalp_blocked_by_arb_open")
                    continue

                risk_usd_per_trade = scalper._current_balance * cohort.risk_fraction_per_trade
                cost_per_contract = max(best_limit_cents / 100.0, 0.01)
                base_qty = max(1, int(risk_usd_per_trade / cost_per_contract))
                confidence_mult = max(1.0, abs(directional_score) / max(1e-9, threshold))
                quantity = max(base_qty, int(base_qty * confidence_mult))
                context_mult = 1.0
                if prepared.cross_asset_vol_regime == "VOL_SPIKE":
                    context_mult *= 0.90
                move_abs = abs(prepared.cross_asset_price_change_pct)
                if move_abs >= 0.006:
                    context_mult *= 1.10
                elif move_abs >= 0.002:
                    context_mult *= 1.04
                if prepared.risk_regime == "RISK_OFF":
                    context_mult *= 0.90
                context_mult *= _session_mult(prepared.now_wall, sleeve="scalp")
                quantity = max(1, int(quantity * context_mult))
                quantity = min(cohort.scalp_max_quantity, quantity)

                if self._shared is not None:
                    crowd_key = _crowding_key(scalper._cfg.bot_id, prepared.ticker, best_side)
                    fills_per_sec = float(self._shared.crowding_fills_per_sec_by_side.get(crowd_key, 0.0))
                    is_crowded = fills_per_sec >= float(cohort.crowding_fills_per_sec_threshold)
                    if is_crowded:
                        self._shared.crowding_candidate_events += 1
                    if cohort.enable_crowding_throttle:
                        if cohort.crowding_mode == "pause":
                            until = float(self._shared.crowding_pause_until.get(crowd_key, 0.0))
                            if until > prepared.now_wall:
                                self._count("crowding_pause_active")
                                continue
                            if is_crowded and cohort.crowding_pause_s > 0:
                                self._shared.crowding_pause_until[crowd_key] = (
                                    prepared.now_wall + float(cohort.crowding_pause_s)
                                )
                                self._shared.crowding_active_events += 1
                                self._count("crowding_paused")
                                continue
                        else:
                            if is_crowded and 0.0 < cohort.crowding_qty_multiplier < 1.0:
                                scaled_qty = max(1, int(quantity * cohort.crowding_qty_multiplier))
                                if scaled_qty < quantity:
                                    quantity = scaled_qty
                                    self._shared.crowding_active_events += 1
                                    self._count("crowding_scaled")

                if self._param_region_engine is not None:
                    candidate_ctx = {
                        "eb": _edge_bucket(float(best_edge)),
                        "pb": _price_bucket(int(best_limit_cents)),
                        "tb": _tts_bucket(float(prepared.time_to_settle_s)),
                    }
                    adj = self._param_region_engine.evaluate_candidate(
                        scalper._cfg,
                        family=family,
                        now_ts=prepared.now_wall,
                        side=best_side,
                        decision_context=candidate_ctx,
                    )
                    if adj.candidate:
                        self._count("param_region_penalty_candidate")
                    if adj.blocked:
                        self._count("param_region_blocked")
                        if self._shared is not None:
                            self._shared.param_region_block_hits += 1
                        continue
                    if abs(adj.scale_mult - 1.0) > 1e-6:
                        scaled_qty = max(1, int(quantity * adj.scale_mult))
                        if scaled_qty < quantity:
                            quantity = scaled_qty
                            self._count("param_region_downweighted")
                            if self._shared is not None:
                                self._shared.param_region_penalty_hits += 1
                        elif scaled_qty > quantity:
                            quantity = scaled_qty
                            self._count("param_region_upweighted")

                cap_max_allowed = self._market_side_cap_max_contracts(
                    cap_contracts=cohort.market_side_cap_contracts,
                    cap_usd=cohort.market_side_cap_usd,
                    limit_cents=best_limit_cents,
                )
                if cap_max_allowed > 0 and self._shared is not None:
                    cap_key = _side_guard_key(prepared.ticker, best_side)
                    projected = planned_contracts_by_side.get(best_side, 0) + max(0, quantity)
                    util = min(10.0, projected / max(1, cap_max_allowed))
                    prev_util = self._shared.projected_cap_util_by_side.get(cap_key, 0.0)
                    if util > prev_util:
                        self._shared.projected_cap_util_by_side[cap_key] = util

                quantity, cap_reason = self._apply_market_side_cap(
                    cap_enabled=cohort.enable_market_side_caps,
                    cap_contracts=cohort.market_side_cap_contracts,
                    cap_usd=cohort.market_side_cap_usd,
                    cap_mode=cohort.market_side_cap_enforcement_mode,
                    limit_cents=best_limit_cents,
                    requested_qty=quantity,
                    planned_contracts_for_side=planned_contracts_by_side.get(best_side, 0),
                )
                if cap_reason == "blocked" or quantity <= 0:
                    self._count("market_side_cap_blocked")
                    continue
                if cap_reason == "scaled":
                    self._count("market_side_cap_scaled")

                family_side_key = f"{family}|{best_side}"
                family_cap_allowed = self._market_side_cap_max_contracts(
                    cap_contracts=0,
                    cap_usd=cohort.family_side_cap_usd,
                    limit_cents=best_limit_cents,
                )
                if family_cap_allowed > 0 and self._shared is not None:
                    fam_cap_key = f"family:{family_side_key}"
                    fam_projected = planned_contracts_by_family_side.get(family_side_key, 0) + max(0, quantity)
                    fam_util = min(10.0, fam_projected / max(1, family_cap_allowed))
                    prev_fam_util = self._shared.projected_cap_util_by_side.get(fam_cap_key, 0.0)
                    if fam_util > prev_fam_util:
                        self._shared.projected_cap_util_by_side[fam_cap_key] = fam_util
                quantity, family_cap_reason = self._apply_market_side_cap(
                    cap_enabled=cohort.enable_market_side_caps and cohort.family_side_cap_usd > 0,
                    cap_contracts=0,
                    cap_usd=cohort.family_side_cap_usd,
                    cap_mode=cohort.market_side_cap_enforcement_mode,
                    limit_cents=best_limit_cents,
                    requested_qty=quantity,
                    planned_contracts_for_side=planned_contracts_by_family_side.get(family_side_key, 0),
                )
                if family_cap_reason == "blocked" or quantity <= 0:
                    self._count("family_side_cap_blocked")
                    continue
                if family_cap_reason == "scaled":
                    self._count("family_side_cap_scaled")

                scalper._last_signal_ts[prepared.ticker] = prepared.now_wall
                self._count("scalp_entry_signal")
                if self._tape_writer is not None:
                    self._tape_writer.write_signal(
                        build_tape_record(
                            prepared=prepared,
                            source="mispricing_scalp",
                            decision="signal",
                            side=best_side,
                            edge=best_edge,
                            reject_reason=None,
                            params={
                                "min_entry_cents": cohort.scalp_min_entry_cents,
                                "max_entry_cents": cohort.scalp_max_entry_cents,
                                "min_edge_threshold": 0.0,
                                "persistence_window_ms": 0,
                                "scalp_min_edge_cents": cohort.scalp_min_edge_cents,
                                "scalp_min_profit_cents": cohort.scalp_min_profit_cents,
                            },
                            ctx_key=None,
                            sampled_rejection=False,
                            quantity_contracts=quantity,
                        )
                    )
                signals.append(
                    TradeSignal(
                        market_ticker=prepared.ticker,
                        side=best_side,
                        action="buy",
                        limit_price_cents=best_limit_cents,
                        quantity_contracts=quantity,
                        edge=best_edge,
                        p_yes=float(prepared.p_yes),
                        timestamp=prepared.now_wall,
                        order_style="aggressive",
                        source="mispricing_scalp",
                        bot_id=scalper._cfg.bot_id,
                    )
                )
                planned_contracts_by_side[best_side] = planned_contracts_by_side.get(best_side, 0) + max(0, quantity)
                planned_contracts_by_family_side[family_side_key] = (
                    planned_contracts_by_family_side.get(family_side_key, 0) + max(0, quantity)
                )
        return signals


class _ArbBatchEvaluator:
    # Evaluate paired YES+NO entry opportunities (sum-ask arbitrage).

    def __init__(self, scalpers: List[MispricingScalper]) -> None:
        self._scalpers = scalpers
        self._arb_count = len(scalpers)
        self._entry_cohorts = self._build_entry_cohorts(scalpers)
        self._diag_lock = threading.Lock()
        self._diag_reasons: Dict[str, int] = {}

    def _count(self, reason: str, n: int = 1) -> None:
        if n <= 0:
            return
        with self._diag_lock:
            self._diag_reasons[reason] = self._diag_reasons.get(reason, 0) + n

    def drain_diagnostics(self) -> Dict[str, int]:
        with self._diag_lock:
            snap = dict(self._diag_reasons)
            self._diag_reasons.clear()
        return snap

    @staticmethod
    def _build_entry_cohorts(scalpers: List[MispricingScalper]) -> List[_ArbEntryCohort]:
        grouped: Dict[Tuple[Any, ...], List[MispricingScalper]] = {}
        for scalper in scalpers:
            cfg = scalper._cfg
            if not bool(getattr(cfg, "arb_enabled", False)):
                continue
            key = (
                cfg.arb_min_sum_ask_cents,
                cfg.arb_min_net_edge_cents,
                cfg.arb_min_entry_cents,
                cfg.arb_max_entry_cents,
                cfg.arb_max_quantity,
                cfg.arb_cooldown_s,
                cfg.risk_fraction_per_trade,
                tuple(cfg.live_families),
                tuple(cfg.shadow_families),
                tuple(cfg.trade_enabled_assets),
                tuple(cfg.shadow_assets),
                cfg.enable_market_side_caps,
                cfg.market_side_cap_contracts,
                cfg.market_side_cap_usd,
                cfg.family_side_cap_usd,
                cfg.market_side_cap_enforcement_mode,
                cfg.enable_crowding_throttle,
                cfg.crowding_window_s,
                cfg.crowding_fills_per_sec_threshold,
                cfg.crowding_qty_multiplier,
                cfg.crowding_pause_s,
                cfg.crowding_mode,
            )
            grouped.setdefault(key, []).append(scalper)

        cohorts: List[_ArbEntryCohort] = []
        for key, members in grouped.items():
            cohorts.append(
                _ArbEntryCohort(
                    scalpers=members,
                    arb_min_sum_ask_cents=key[0],
                    arb_min_net_edge_cents=key[1],
                    arb_min_entry_cents=key[2],
                    arb_max_entry_cents=key[3],
                    arb_max_quantity=key[4],
                    arb_cooldown_s=key[5],
                    risk_fraction_per_trade=key[6],
                    live_families=key[7],
                    shadow_families=key[8],
                    trade_enabled_assets=key[9],
                    shadow_assets=key[10],
                    enable_market_side_caps=key[11],
                    market_side_cap_contracts=key[12],
                    market_side_cap_usd=key[13],
                    family_side_cap_usd=key[14],
                    market_side_cap_enforcement_mode=key[15],
                    enable_crowding_throttle=key[16],
                    crowding_window_s=key[17],
                    crowding_fills_per_sec_threshold=key[18],
                    crowding_qty_multiplier=key[19],
                    crowding_pause_s=key[20],
                    crowding_mode=key[21],
                )
            )
        return cohorts

    def evaluate(self, prepared: _PreparedTickerState) -> List[TradeSignal]:
        if prepared.is_range:
            self._count("arb_range_filtered", self._arb_count)
            return []

        signals: List[TradeSignal] = []
        family = _family_label(prepared.asset, prepared.window_minutes, prepared.is_range)
        asset = (prepared.asset or "").upper()
        sum_ask = int(prepared.yes_ask_cents) + int(prepared.no_ask_cents)
        cfg_slippage_floor = (
            max(0, int(getattr(self._scalpers[0]._cfg, "paper_slippage_cents", 0)))
            if self._scalpers
            else 0
        )
        # Dynamic pair-entry friction proxy from current side spreads.
        pair_spread = max(0, int(prepared.yes_ask_cents) - int(prepared.yes_bid_cents)) + max(
            0, int(prepared.no_ask_cents) - int(prepared.no_bid_cents)
        )
        slippage_buffer = max(cfg_slippage_floor * 2, max(1, int(round(pair_spread * 0.5))))
        # One-way pair entry fees (buy YES + buy NO), held to settlement.
        fee_cents = int(round(
            (
                estimate_kalshi_taker_fee_usd(prepared.yes_ask_cents, 100)
                + estimate_kalshi_taker_fee_usd(prepared.no_ask_cents, 100)
            ) * 100.0
        ))
        net_edge_cents = 100 - sum_ask - fee_cents - slippage_buffer

        for cohort in self._entry_cohorts:
            if cohort.trade_enabled_assets and asset not in cohort.trade_enabled_assets:
                self._count("arb_asset_not_enabled", len(cohort.scalpers))
                continue
            if cohort.shadow_assets and asset in cohort.shadow_assets:
                self._count("arb_asset_shadow_mode", len(cohort.scalpers))
                continue
            if cohort.live_families and family not in cohort.live_families:
                self._count("arb_family_not_enabled", len(cohort.scalpers))
                continue
            if cohort.shadow_families and family in cohort.shadow_families:
                self._count("arb_family_shadow_mode", len(cohort.scalpers))
                continue
            if prepared.yes_ask_cents < cohort.arb_min_entry_cents or prepared.yes_ask_cents > cohort.arb_max_entry_cents:
                self._count("arb_yes_entry_price_out_of_range", len(cohort.scalpers))
                continue
            if prepared.no_ask_cents < cohort.arb_min_entry_cents or prepared.no_ask_cents > cohort.arb_max_entry_cents:
                self._count("arb_no_entry_price_out_of_range", len(cohort.scalpers))
                continue
            if sum_ask > cohort.arb_min_sum_ask_cents:
                self._count("arb_no_cross", len(cohort.scalpers))
                continue
            if net_edge_cents < cohort.arb_min_net_edge_cents:
                self._count("arb_net_edge_too_low", len(cohort.scalpers))
                continue

            # Pair size capped by balance risk, config cap, and both-side depth.
            # Buy YES consumes NO-bid depth; buy NO consumes YES-bid depth.
            pair_depth_contracts = max(
                0,
                min(prepared.no_bid_depth_centicx, prepared.yes_bid_depth_centicx) // 100,
            )
            # Track planned arb contracts per side for concentration control.
            planned_contracts_by_side: Dict[str, int] = {"yes": 0, "no": 0}

            for scalper in cohort.scalpers:
                # Reuse per-bot cooldown storage for arb sleeve.
                last = scalper._last_signal_ts.get(prepared.ticker, 0.0)
                if cohort.arb_cooldown_s > 0 and (prepared.now_wall - last) < cohort.arb_cooldown_s:
                    self._count("arb_cooldown")
                    continue
                if prepared.ticker in scalper._open_positions:
                    self._count("arb_scalper_open_position_skip")
                    continue

                risk_usd_per_trade = scalper._current_balance * cohort.risk_fraction_per_trade
                pair_cost_usd = max(sum_ask / 100.0, 0.01)
                qty = max(1, int(risk_usd_per_trade / pair_cost_usd))
                qty = min(cohort.arb_max_quantity, qty)
                if pair_depth_contracts > 0:
                    qty = min(qty, pair_depth_contracts)
                if qty <= 0:
                    self._count("arb_no_depth")
                    continue

                # Market-side cap: YES leg
                qty_yes, cap_reason_yes = _StrategyBatchEvaluator._apply_market_side_cap(
                    cap_enabled=cohort.enable_market_side_caps,
                    cap_contracts=cohort.market_side_cap_contracts,
                    cap_usd=cohort.market_side_cap_usd,
                    cap_mode=cohort.market_side_cap_enforcement_mode,
                    limit_cents=prepared.yes_ask_cents,
                    requested_qty=qty,
                    planned_contracts_for_side=planned_contracts_by_side.get("yes", 0),
                )
                if cap_reason_yes == "blocked" or qty_yes <= 0:
                    self._count("arb_market_side_cap_blocked")
                    continue
                # Market-side cap: NO leg
                qty_no, cap_reason_no = _StrategyBatchEvaluator._apply_market_side_cap(
                    cap_enabled=cohort.enable_market_side_caps,
                    cap_contracts=cohort.market_side_cap_contracts,
                    cap_usd=cohort.market_side_cap_usd,
                    cap_mode=cohort.market_side_cap_enforcement_mode,
                    limit_cents=prepared.no_ask_cents,
                    requested_qty=qty,
                    planned_contracts_for_side=planned_contracts_by_side.get("no", 0),
                )
                if cap_reason_no == "blocked" or qty_no <= 0:
                    self._count("arb_market_side_cap_blocked")
                    continue
                qty = min(qty_yes, qty_no)

                scalper._last_signal_ts[prepared.ticker] = prepared.now_wall
                scalper._arb_last_entry_ts[prepared.ticker] = prepared.now_wall
                self._count("arb_entry_signal")
                planned_contracts_by_side["yes"] = planned_contracts_by_side.get("yes", 0) + qty
                planned_contracts_by_side["no"] = planned_contracts_by_side.get("no", 0) + qty

                # Emit paired buys with same bot_id/timestamp.
                signals.append(
                    TradeSignal(
                        market_ticker=prepared.ticker,
                        side="yes",
                        action="buy",
                        limit_price_cents=prepared.yes_ask_cents,
                        quantity_contracts=qty,
                        edge=net_edge_cents / 100.0,
                        p_yes=float(prepared.p_yes),
                        timestamp=prepared.now_wall,
                        order_style="aggressive",
                        source="pair_arb",
                        bot_id=scalper._cfg.bot_id,
                    )
                )
                signals.append(
                    TradeSignal(
                        market_ticker=prepared.ticker,
                        side="no",
                        action="buy",
                        limit_price_cents=prepared.no_ask_cents,
                        quantity_contracts=qty,
                        edge=net_edge_cents / 100.0,
                        p_yes=float(prepared.p_yes),
                        timestamp=prepared.now_wall,
                        order_style="aggressive",
                        source="pair_arb",
                        bot_id=scalper._cfg.bot_id,
                    )
                )
        return signals


#  FarmDispatcher — single bus consumer for all high-frequency topics

class FarmDispatcher:
    # Subscribes ONCE per high-frequency topic and evaluates all bots.
    #
    # Replaces the O(N_bots) fan-out model where each bot had its own asyncio
    # queue per topic.  Instead:
    # - One queue per topic, managed here.
    # - Shared state dicts injected into all bots at construction time.
    # - ``evaluate_sync`` / ``scalp_sync`` / ``exit_sync`` are called in a
    # tight loop with periodic ``asyncio.sleep(0)`` yields so the event
    # loop can still drain WS I/O between batches of bots. When using
    # separate UI, heavy work runs in a thread pool so the loop stays free
    # for IPC and StateAggregator.
    #
    # Truth-feed staleness and WS disconnect are managed centrally here;
    # per-bot halt flags (drawdown) are still set by the bot itself inside
    # ``evaluate_sync``.

    # Yield to the event loop after processing this many bots per tick.
    # Lower = more responsive I/O but more scheduling overhead.
    def __init__(
        self,
        bus: Bus,
        shared: SharedFarmState,
        strategies: List[StrategyEngine],
        scalpers: List[MispricingScalper],
        shared_metadata: Optional[Dict[str, MarketMetadata]] = None,
        truth_stale_timeout_s: float = 30.0,
        executor_workers: Optional[int] = None,
        regime_gate: Optional[object] = None,
        param_region_engine: Optional[_ParamRegionPenaltyEngine] = None,
        family_context_features_enabled: bool = False,
        context_policy: Optional[ContextPolicyEngine] = None,
        drift_guard: Optional[DriftGuard] = None,
        adaptive_cap: Optional[AdaptiveCapEngine] = None,
        edge_tracker: Optional[EdgeTracker] = None,
        base_cfg0: Optional[KalshiConfig] = None,
        tape_writer: Optional[DecisionTapeWriter] = None,
    ) -> None:
        self._bus = bus
        self._shared = shared
        self._strategies = strategies
        self._scalpers = scalpers
        self._truth_stale_timeout_s = truth_stale_timeout_s
        cpu_count = os.cpu_count() or 4
        default_workers = max(4, min(12, cpu_count // 2 if cpu_count > 1 else 1))
        self._executor_workers = max(1, executor_workers or default_workers)
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._ob_updates: int = 0
        self._prob_updates: int = 0
        self._dispatch_count: int = 0
        self._dispatch_signal_count: int = 0
        self._dispatch_latency_ms: deque[float] = deque(maxlen=5000)
        self._ob_truth_gap_ms: deque[float] = deque(maxlen=5000)
        self._prob_truth_gap_ms: deque[float] = deque(maxlen=5000)
        self._last_ob_mono_by_ticker: Dict[str, float] = {}
        self._last_prob_mono_by_ticker: Dict[str, float] = {}
        self._dispatch_queue: asyncio.Queue[str] = asyncio.Queue()
        self._queued_tickers: set[str] = set()
        self._dispatching_tickers: set[str] = set()
        self._redispatch_tickers: set[str] = set()
        self._regime_gate = regime_gate  # Optional[RegimeGate]
        self._param_region_engine = param_region_engine
        self._family_context_features_enabled = family_context_features_enabled
        self._context_policy = context_policy
        self._drift_guard = drift_guard
        self._adaptive_cap = adaptive_cap
        self._edge_tracker = edge_tracker
        self._tape_writer = tape_writer
        self._shared_metadata = shared_metadata if shared_metadata is not None else {}
        self._base_cfg0 = base_cfg0
        self._policy_last_reload_check_ts: float = 0.0
        self._policy_last_mtime: float = 0.0
        self._decision_tape_writer: Optional[DecisionTapeWriter] = None
        self._active_tickers: set[str] = set()
        # Trade tape flow imbalance: per-ticker deque of (ts, side, count).
        self._trade_deques: Dict[str, deque] = {}
        self._trade_flow_window_s: float = 60.0
        # Orderbook delta-flow imbalance: per-ticker/per-side deque of (ts, is_add, qty).
        self._delta_yes_deques: Dict[str, deque] = {}
        self._delta_no_deques: Dict[str, deque] = {}
        self._delta_flow_window_s: float = 30.0
        self._strategy_evaluator = _StrategyBatchEvaluator(
            strategies,
            regime_gate=regime_gate,
            shared=shared,
            param_region_engine=param_region_engine,
            context_policy=context_policy,
            edge_tracker=edge_tracker,
            tape_writer=tape_writer,
        )
        self._scalper_evaluator = _ScalperBatchEvaluator(
            scalpers,
            regime_gate=regime_gate,
            shared=shared,
            param_region_engine=param_region_engine,
            context_policy=context_policy,
            edge_tracker=edge_tracker,
            tape_writer=tape_writer,
        )
        self._arb_evaluator = _ArbBatchEvaluator(scalpers)

    async def start(self) -> None:
        self._running = True
        self._executor = ThreadPoolExecutor(
            max_workers=self._executor_workers,
            thread_name_prefix="farm_dispatch",
        )
        # One subscription per high-frequency topic — not one per bot.
        q_ob = await self._bus.subscribe("kalshi.orderbook")
        q_prob = await self._bus.subscribe("kalshi.fair_prob")
        q_meta = await self._bus.subscribe("kalshi.market_metadata")
        q_btc = await self._bus.subscribe("btc.mid_price")
        q_eth = await self._bus.subscribe("eth.mid_price")
        q_sol = await self._bus.subscribe("sol.mid_price")
        q_ws = await self._bus.subscribe("kalshi.ws.status")
        q_selected = await self._bus.subscribe("kalshi.selected_markets")
        q_trade = await self._bus.subscribe("kalshi.trade")
        q_delta_flow = await self._bus.subscribe("kalshi.orderbook_delta_flow")

        self._tasks = [
            asyncio.create_task(self._consume_ob(q_ob)),
            asyncio.create_task(self._consume_prob(q_prob)),
            asyncio.create_task(self._consume_meta(q_meta)),
            asyncio.create_task(self._consume_truth(q_btc, "BTC")),
            asyncio.create_task(self._consume_truth(q_eth, "ETH")),
            asyncio.create_task(self._consume_truth(q_sol, "SOL")),
            asyncio.create_task(self._consume_ws_status(q_ws)),
            asyncio.create_task(self._consume_selected_markets(q_selected)),
            asyncio.create_task(self._consume_trades(q_trade)),
            asyncio.create_task(self._consume_orderbook_delta_flow(q_delta_flow)),
            asyncio.create_task(self._truth_watchdog()),
            asyncio.create_task(self._pipeline_diagnostics()),
            asyncio.create_task(self._data_quality_monitor()),
            *(asyncio.create_task(self._dispatch_worker()) for _ in range(self._executor_workers)),
        ]

        # Subscribe to regime bridge topic (optional — fails safe if no bridge)
        try:
            q_regime = await self._bus.subscribe("kalshi.regime")
            self._tasks.append(asyncio.create_task(self._consume_regime(q_regime)))
            log.info("FarmDispatcher subscribed to kalshi.regime (regime bridge active)")
        except Exception:
            log.info("FarmDispatcher: kalshi.regime topic not available (regime bridge inactive)")
        log.info(
            "FarmDispatcher started",
            data={
                "strategies": len(self._strategies),
                "scalpers": len(self._scalpers),
                "strategy_cohorts": len(self._strategy_evaluator._cohorts),
                "scalper_entry_cohorts": len(self._scalper_evaluator._entry_cohorts),
                "arb_entry_cohorts": len(self._arb_evaluator._entry_cohorts),
                "executor_workers": self._executor_workers,
            },
        )

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def register_bot(self, strategy: StrategyEngine, scalper: MispricingScalper) -> None:
        # Hot-add a new bot to the running dispatcher (thread-safe rebuild of cohorts).
        self._strategies.append(strategy)
        self._scalpers.append(scalper)
        # Rebuild the evaluator cohort lists atomically, preserving all engines.
        self._strategy_evaluator = _StrategyBatchEvaluator(
            self._strategies,
            regime_gate=self._regime_gate,
            shared=self._shared,
            param_region_engine=self._param_region_engine,
            context_policy=self._context_policy,
            edge_tracker=self._edge_tracker,
            tape_writer=self._tape_writer,
        )
        self._scalper_evaluator = _ScalperBatchEvaluator(
            self._scalpers,
            regime_gate=self._regime_gate,
            shared=self._shared,
            param_region_engine=self._param_region_engine,
            context_policy=self._context_policy,
            edge_tracker=self._edge_tracker,
            tape_writer=self._tape_writer,
        )
        self._arb_evaluator = _ArbBatchEvaluator(self._scalpers)

    def deregister_bot(self, bot_id: str) -> None:
        # Hot-remove a bot from the running dispatcher by bot_id.
        self._strategies = [s for s in self._strategies if s._cfg.bot_id != bot_id]
        self._scalpers = [s for s in self._scalpers if s._cfg.bot_id != bot_id]
        self._strategy_evaluator = _StrategyBatchEvaluator(
            self._strategies,
            regime_gate=self._regime_gate,
            shared=self._shared,
            param_region_engine=self._param_region_engine,
            context_policy=self._context_policy,
            edge_tracker=self._edge_tracker,
            tape_writer=self._tape_writer,
        )
        self._scalper_evaluator = _ScalperBatchEvaluator(
            self._scalpers,
            regime_gate=self._regime_gate,
            shared=self._shared,
            param_region_engine=self._param_region_engine,
            context_policy=self._context_policy,
            edge_tracker=self._edge_tracker,
            tape_writer=self._tape_writer,
        )
        self._arb_evaluator = _ArbBatchEvaluator(self._scalpers)


    # ── consumers ────────────────────────────────────────────────────────────

    async def _consume_ob(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                ob: OrderbookState = await q.get()
                now = time.monotonic()
                self._ob_updates += 1
                self._last_ob_mono_by_ticker[ob.market_ticker] = now
                self._shared.orderbooks[ob.market_ticker] = ob
                asset = self._shared.market_asset.get(ob.market_ticker, "BTC")
                last_truth = self._shared.last_truth_tick_by_asset.get(asset, 0.0)
                if last_truth > 0:
                    self._ob_truth_gap_ms.append(max(0.0, (now - last_truth) * 1000.0))
                self._mark_ticker_dirty(ob.market_ticker)
        except asyncio.CancelledError:
            pass

    async def _consume_prob(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                fp: FairProbability = await q.get()
                now = time.monotonic()
                self._prob_updates += 1
                self._last_prob_mono_by_ticker[fp.market_ticker] = now
                prev_fp = self._shared.fair_probs.get(fp.market_ticker)
                if prev_fp is not None:
                    self._shared.prev_fair_prob_by_ticker[fp.market_ticker] = prev_fp.p_yes
                    self._shared.prev_fair_prob_ts_by_ticker[fp.market_ticker] = (
                        self._shared.last_fair_prob_ts_by_ticker.get(fp.market_ticker, now)
                    )
                self._shared.fair_probs[fp.market_ticker] = fp
                self._shared.last_fair_prob_ts_by_ticker[fp.market_ticker] = now
                asset = self._shared.market_asset.get(fp.market_ticker, "BTC")
                last_truth = self._shared.last_truth_tick_by_asset.get(asset, 0.0)
                if last_truth > 0:
                    self._prob_truth_gap_ms.append(max(0.0, (now - last_truth) * 1000.0))
                self._mark_ticker_dirty(fp.market_ticker)
        except asyncio.CancelledError:
            pass

    async def _consume_meta(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                msg = await q.get()
                ticker = msg.market_ticker
                self._shared.market_asset[ticker] = getattr(msg, "asset", "BTC")
                self._shared.market_window_min[ticker] = getattr(msg, "window_minutes", 15)
                self._shared.market_is_range[ticker] = getattr(msg, "is_range", False)
                try:
                    settle_dt = datetime.fromisoformat(
                        msg.settlement_time_iso.replace("Z", "+00:00")
                    )
                    self._shared.market_settlement[ticker] = settle_dt.timestamp()
                    self._shared.scalp_settlement_epoch[ticker] = settle_dt.timestamp()
                except Exception:
                    pass

                time_to_settle_s = self._shared.market_settlement.get(ticker, 0.0) - time.time()
                eligible = (
                    _scalp_family_allowed(
                        self._shared.market_asset.get(ticker, "BTC"),
                        self._shared.market_window_min.get(ticker, 0),
                        self._shared.market_is_range.get(ticker, False),
                    )
                    and (
                        not self._shared.market_is_range.get(ticker, False)
                        or (0 < time_to_settle_s <= 3600.0)
                    )
                )
                self._shared.scalp_eligible[ticker] = eligible
        except asyncio.CancelledError:
            pass

    async def _consume_selected_markets(self, q: asyncio.Queue) -> None:
        # Track active ticker set and prune stale diagnostic/tracking maps.
        try:
            while self._running:
                msg = await q.get()
                tickers = getattr(msg, "tickers", None)
                if not isinstance(tickers, list):
                    continue
                active = {str(t) for t in tickers if t}
                self._active_tickers = active
                self._prune_stale_tickers(active)
        except asyncio.CancelledError:
            pass

    def _prune_stale_tickers(self, active: set[str]) -> None:
        if not active:
            return
        stale = set(self._last_ob_mono_by_ticker.keys()) - active
        stale |= set(self._last_prob_mono_by_ticker.keys()) - active
        if not stale:
            return
        for t in stale:
            self._last_ob_mono_by_ticker.pop(t, None)
            self._last_prob_mono_by_ticker.pop(t, None)
            self._queued_tickers.discard(t)
            self._dispatching_tickers.discard(t)
            self._redispatch_tickers.discard(t)
        # NOTE: scalp_settlement_epoch is intentionally NOT cleared for stale tickers.
        # _ScalperBatchEvaluator._prune_ghost_positions relies on those old settlement
        # timestamps persisting after the ticker is unsubscribed so it can detect that
        # lingering _open_positions entries are on expired contracts. Cleaning them here
        # would cause the prune loop to default to float("inf") and never clear ghosts.
        if self._shared is not None and self._shared.side_guard_block_until:
            stale_prefixes = {f"{t}|" for t in stale}
            for key in list(self._shared.side_guard_block_until.keys()):
                if any(key.startswith(pref) for pref in stale_prefixes):
                    self._shared.side_guard_block_until.pop(key, None)

    async def _consume_truth(self, q: asyncio.Queue, asset: str) -> None:
        try:
            while self._running:
                evt = await q.get()
                self._shared.last_truth_tick_by_asset[asset] = time.monotonic()
                price: Optional[float] = None
                if isinstance(evt, (int, float)):
                    price = float(evt)
                elif isinstance(evt, dict):
                    for key in ("mid_price", "price", "value", "last", "close"):
                        val = evt.get(key)
                        if isinstance(val, (int, float)):
                            price = float(val)
                            break
                else:
                    for attr in ("mid_price", "price", "value", "last", "close"):
                        val = getattr(evt, attr, None)
                        if isinstance(val, (int, float)):
                            price = float(val)
                            break
                if price is not None and price > 0:
                    prev = self._shared.truth_price_by_asset.get(asset)
                    if prev is not None:
                        self._shared.prev_truth_price_by_asset[asset] = prev
                    self._shared.truth_price_by_asset[asset] = price
        except asyncio.CancelledError:
            pass

    async def _consume_ws_status(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                event = await q.get()
                if event.status == "disconnected":
                    self._shared.ws_halted = True
                    # Propagate halt to all bots so per-bot drawdown logic sees it.
                    for s in self._strategies:
                        if s._halt_reason in ("", "ws_disconnected"):
                            s._halted = True
                            s._halt_reason = "ws_disconnected"
                elif event.status == "connected":
                    self._shared.ws_halted = False
                    for s in self._strategies:
                        if s._halt_reason == "ws_disconnected":
                            s._halted = False
                            s._halt_reason = ""
        except asyncio.CancelledError:
            pass

    async def _consume_trades(self, q: asyncio.Queue) -> None:
        # Consume KalshiTradeEvent messages and maintain rolling flow imbalance.
        import time as _time
        try:
            while self._running:
                ev = await q.get()  # KalshiTradeEvent
                ticker = ev.market_ticker
                if ticker not in self._trade_deques:
                    self._trade_deques[ticker] = deque(maxlen=500)
                self._trade_deques[ticker].append((ev.ts, ev.taker_side, ev.count))
                # Recompute flow imbalance for this ticker.
                cutoff = _time.time() - self._trade_flow_window_s
                dq = self._trade_deques[ticker]
                yes_vol = sum(c for ts, s, c in dq if ts >= cutoff and s == "yes")
                no_vol  = sum(c for ts, s, c in dq if ts >= cutoff and s == "no")
                total = yes_vol + no_vol
                flow = (yes_vol - no_vol) / total if total > 0 else 0.0
                self._shared.trade_flow_by_ticker[ticker] = flow
        except asyncio.CancelledError:
            pass

    async def _consume_orderbook_delta_flow(self, q: asyncio.Queue) -> None:
        # Consume KalshiOrderDeltaEvent and maintain rolling add/cancel imbalance.
        import time as _time
        try:
            while self._running:
                ev: KalshiOrderDeltaEvent = await q.get()
                ticker = str(getattr(ev, "market_ticker", "") or "")
                side = str(getattr(ev, "side", "") or "").lower()
                qty = int(getattr(ev, "qty", 0) or 0)
                if not ticker or side not in ("yes", "no") or qty <= 0:
                    continue
                ts = float(getattr(ev, "ts", 0.0) or 0.0) or _time.time()
                is_add = bool(getattr(ev, "is_add", False))
                target = self._delta_yes_deques if side == "yes" else self._delta_no_deques
                dq = target.get(ticker)
                if dq is None:
                    dq = deque(maxlen=2000)
                    target[ticker] = dq
                dq.append((ts, is_add, qty))
                cutoff = _time.time() - self._delta_flow_window_s
                while dq and dq[0][0] < cutoff:
                    dq.popleft()
                add_qty = sum(v for t, add, v in dq if t >= cutoff and add)
                cancel_qty = sum(v for t, add, v in dq if t >= cutoff and not add)
                total = add_qty + cancel_qty
                flow = (add_qty - cancel_qty) / total if total > 0 else 0.0
                if side == "yes":
                    self._shared.orderbook_delta_flow_yes[ticker] = flow
                else:
                    self._shared.orderbook_delta_flow_no[ticker] = flow
        except asyncio.CancelledError:
            pass

    async def _consume_regime(self, q: asyncio.Queue) -> None:
        # Consume regime events bridged from core EventBus into SharedFarmState cache.
        #
        # Expected payload shape (from bus bridge):
        # {"asset": "BTC", "vol_regime": "VOL_NORMAL", "liq_regime": "LIQ_NORMAL",
        # "risk_regime": "NEUTRAL", "session_regime": "US", "market": "CRYPTO"}
        try:
            while self._running:
                event = await q.get()
                if not isinstance(event, dict):
                    continue
                asset = event.get("asset", "")
                if not asset:
                    continue
                now_mono = time.monotonic()
                vol = event.get("vol_regime", "")
                liq = event.get("liq_regime", "")
                risk = event.get("risk_regime", "")
                session = event.get("session_regime", "")
                market = event.get("market", "")

                if vol:
                    self._shared.regime_vol[asset] = vol
                if liq:
                    self._shared.regime_liq[asset] = liq
                if risk:
                    self._shared.regime_risk = risk
                if session and market:
                    self._shared.regime_session[market] = session
                self._shared.regime_last_update[asset] = now_mono
        except asyncio.CancelledError:
            pass

    async def _truth_watchdog(self) -> None:
        # Periodically check truth feed staleness and update the shared flag.
        try:
            while self._running:
                await asyncio.sleep(5.0)
                now = time.monotonic()
                ticks = self._shared.last_truth_tick_by_asset
                if not ticks:
                    continue  # nothing received yet — stay stale
                stale = any(
                    (now - ts) > self._truth_stale_timeout_s
                    for ts in ticks.values()
                )
                was_stale = self._shared.truth_stale
                self._shared.truth_stale = stale
                if stale and not was_stale:
                    log.warning("Farm truth feed went stale")
                elif not stale and was_stale:
                    log.info("Farm truth feed recovered")
        except asyncio.CancelledError:
            pass

    # ── dispatch ─────────────────────────────────────────────────────────────

    def _dispatch_ticker_sync(self, ticker: str) -> List[TradeSignal]:
        # Evaluate all farm bots for *ticker* in a thread. Returns signals to publish.
        truth_stale = self._shared.truth_stale or self._shared.ws_halted

        # Live-mode data quality circuit breaker: drop all signals when market
        # metadata is known-bad (e.g. Kalshi API returning corrupt strike prices).
        # In dry-run / paper mode we allow signals to continue so the paper run
        # can still record what *would* have been traded.
        cfg = self._base_cfg0
        if (self._shared.data_quality_halted
                and cfg is not None
                and not cfg.dry_run
                and cfg.ws_trading_enabled):
            return []

        prepared = self._prepare_ticker_state(ticker)
        if prepared is None:
            return []
        strategy_signals = self._strategy_evaluator.evaluate(prepared, truth_stale)
        scalp_signals = self._scalper_evaluator.evaluate(prepared)
        arb_signals = self._arb_evaluator.evaluate(prepared)

        sleeve_mode = str(getattr(self._base_cfg0, "sleeve_mode", "parallel") or "parallel")
        if sleeve_mode == "flow_first":
            # Prevent mixed fast entries on the same bot+ticker in the same cycle.
            # For closer wallet mimicry, directional scalp wins; pair-arb is additive/fallback.
            scalp_entry_keys = {
                (str(sig.bot_id), str(sig.market_ticker))
                for sig in scalp_signals
                if sig.action == "buy"
            }
            if scalp_entry_keys:
                before = len(arb_signals)
                arb_signals = [
                    sig
                    for sig in arb_signals
                    if not (
                        sig.action == "buy"
                        and (str(sig.bot_id), str(sig.market_ticker)) in scalp_entry_keys
                    )
                ]
                deferred = before - len(arb_signals)
                if deferred > 0:
                    self._arb_evaluator._count("arb_deferred_to_scalp", deferred)

            fast_signals = scalp_signals + arb_signals
            has_fast_entry = any(sig.action == "buy" for sig in fast_signals)
            if has_fast_entry:
                dropped_hold_entries = sum(1 for sig in strategy_signals if sig.action == "buy")
                if dropped_hold_entries > 0:
                    # Keep diagnostics explicit so gate-dominance panels reveal fallback behavior.
                    self._strategy_evaluator._count("hold_deferred_to_flow", dropped_hold_entries)
                strategy_signals = [sig for sig in strategy_signals if sig.action != "buy"]
            return fast_signals + strategy_signals

        return strategy_signals + scalp_signals + arb_signals

    def _prepare_ticker_state(self, ticker: str) -> Optional[_PreparedTickerState]:
        fp = self._shared.fair_probs.get(ticker)
        ob = self._shared.orderbooks.get(ticker)
        if fp is None or ob is None or not ob.valid:
            return None

        p_yes = fp.p_yes
        if p_yes != p_yes:
            return None

        now_wall = time.time()
        now_mono = time.monotonic()
        asset = self._shared.market_asset.get(ticker, "BTC")
        last_asset_tick = self._shared.last_truth_tick_by_asset.get(asset, 0.0)
        settle_ts = self._shared.market_settlement.get(ticker, 0.0)
        time_to_settle_s = settle_ts - now_wall if settle_ts > 0 else 9999.0
        yes_ask_prob = ob.implied_yes_ask_cents / 100.0
        no_ask_prob = ob.implied_no_ask_cents / 100.0
        ev_yes = p_yes - yes_ask_prob
        ev_no = (1.0 - p_yes) - no_ask_prob
        family = _family_label(
            asset,
            self._shared.market_window_min.get(ticker, 15),
            self._shared.market_is_range.get(ticker, False),
        )
        family_weight = self._shared.family_weights.get(family, 1.0)

        cross_asset = ""
        if self._family_context_features_enabled:
            if asset == "BTC":
                cross_asset = "ETH"
            elif asset == "ETH":
                cross_asset = "BTC"
        cross_asset_vol_regime = self._shared.regime_vol.get(cross_asset, "") if cross_asset else ""
        cross_asset_price_change_pct = 0.0
        if cross_asset:
            px = self._shared.truth_price_by_asset.get(cross_asset)
            prev_px = self._shared.prev_truth_price_by_asset.get(cross_asset)
            if isinstance(px, (int, float)) and isinstance(prev_px, (int, float)) and prev_px > 0:
                cross_asset_price_change_pct = (float(px) - float(prev_px)) / float(prev_px)
        own_asset_price_change_pct = 0.0
        own_px = self._shared.truth_price_by_asset.get(asset)
        own_prev_px = self._shared.prev_truth_price_by_asset.get(asset)
        if isinstance(own_px, (int, float)) and isinstance(own_prev_px, (int, float)) and own_prev_px > 0:
            own_asset_price_change_pct = (float(own_px) - float(own_prev_px)) / float(own_prev_px)
        spot_price = float(own_px or 0.0)
        meta = self._shared_metadata.get(ticker)
        sd_pct = _strike_distance_pct(spot_price, meta)

        momentum_drift = float(getattr(fp, "drift", 0.0))
        trade_flow = self._shared.trade_flow_by_ticker.get(ticker, 0.0)
        obi = float(getattr(ob, "obi", 0.0))
        depth_pressure = float(getattr(ob, "depth_pressure", 0.0))
        delta_flow_yes = float(self._shared.orderbook_delta_flow_yes.get(ticker, 0.0))
        delta_flow_no = float(self._shared.orderbook_delta_flow_no.get(ticker, 0.0))

        return _PreparedTickerState(
            ticker=ticker,
            now_wall=now_wall,
            now_mono=now_mono,
            p_yes=p_yes,
            yes_ask_cents=ob.implied_yes_ask_cents,
            no_ask_cents=ob.implied_no_ask_cents,
            yes_bid_cents=ob.best_yes_bid_cents,
            no_bid_cents=ob.best_no_bid_cents,
            yes_bid_depth_centicx=ob.best_yes_depth,
            no_bid_depth_centicx=ob.best_no_depth,
            ev_yes=ev_yes,
            ev_no=ev_no,
            asset=asset,
            last_asset_tick=last_asset_tick,
            time_to_settle_s=time_to_settle_s,
            window_minutes=self._shared.market_window_min.get(ticker, 15),
            is_range=self._shared.market_is_range.get(ticker, False),
            scalp_eligible=self._shared.scalp_eligible.get(ticker, False),
            prev_p_yes=self._shared.prev_fair_prob_by_ticker.get(ticker),
            prev_prob_ts=self._shared.prev_fair_prob_ts_by_ticker.get(ticker, 0.0),
            last_prob_ts=self._shared.last_fair_prob_ts_by_ticker.get(ticker, 0.0),
            vol_regime=self._shared.regime_vol.get(asset, ""),
            liq_regime=self._shared.regime_liq.get(asset, ""),
            risk_regime=self._shared.regime_risk,
            cross_asset_vol_regime=cross_asset_vol_regime,
            cross_asset_price_change_pct=cross_asset_price_change_pct,
            own_asset_price_change_pct=own_asset_price_change_pct,
            family_weight=max(0.05, min(2.0, family_weight)),
            strike_distance_pct=sd_pct,
            settlement_epoch=settle_ts if settle_ts > 0 else 0.0,
            momentum_drift=momentum_drift,
            trade_flow=trade_flow,
            obi=obi,
            depth_pressure=depth_pressure,
            delta_flow_yes=delta_flow_yes,
            delta_flow_no=delta_flow_no,
        )

    def _mark_ticker_dirty(self, ticker: str) -> None:
        # Queue ticker for dispatch once; collapse bursts into one latest-state evaluation.
        if ticker in self._dispatching_tickers:
            self._redispatch_tickers.add(ticker)
            return
        if ticker in self._queued_tickers:
            return
        self._queued_tickers.add(ticker)
        self._dispatch_queue.put_nowait(ticker)

    async def _dispatch_worker(self) -> None:
        try:
            while self._running:
                ticker = await self._dispatch_queue.get()
                self._queued_tickers.discard(ticker)
                self._dispatching_tickers.add(ticker)
                try:
                    try:
                        await self._dispatch_ticker(ticker)
                    except Exception as exc:
                        log.error(
                            "Dispatch worker ticker failed",
                            data={"ticker": ticker, "error": str(exc)},
                        )
                finally:
                    self._dispatching_tickers.discard(ticker)
                    if ticker in self._redispatch_tickers:
                        self._redispatch_tickers.discard(ticker)
                        self._mark_ticker_dirty(ticker)
        except asyncio.CancelledError:
            pass

        except Exception as exc:
            log.error("Dispatch worker crashed", data={"error": str(exc)})

    async def _dispatch_ticker(self, ticker: str) -> None:
        # Run heavy dispatch in executor so event loop stays free for IPC/aggregator.
        t0 = time.monotonic()
        if self._executor is None:
            signals = self._dispatch_ticker_sync(ticker)
        else:
            loop = asyncio.get_running_loop()
            signals = await loop.run_in_executor(
                self._executor,
                self._dispatch_ticker_sync,
                ticker,
            )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._dispatch_count += 1
        self._dispatch_signal_count += len(signals)
        self._dispatch_latency_ms.append(elapsed_ms)
        # Yield every few signals so the event loop can run IPC (each publish fans out to 7k+ queues).
        for i, sig in enumerate(signals):
            await self._bus.publish("kalshi.trade_signal", sig)
            if (i + 1) % 4 == 0:
                await asyncio.sleep(0)

    def _maybe_reload_context_policy(self, now_ts: float) -> None:
        if self._context_policy is None or self._base_cfg0 is None:
            return
        if not bool(getattr(self._base_cfg0, "context_policy_auto_reload", False)):
            return
        interval = float(getattr(self._base_cfg0, "context_policy_reload_interval_s", 300.0))
        if interval <= 0:
            return
        if (now_ts - self._policy_last_reload_check_ts) < interval:
            return
        self._policy_last_reload_check_ts = now_ts
        policy_path = str(getattr(self._base_cfg0, "context_policy_file", "") or "")
        if not policy_path:
            return
        p = Path(policy_path)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return
        if mtime <= self._policy_last_mtime:
            return
        if self._context_policy.load_policy(policy_path):
            self._policy_last_mtime = mtime
            diag = self._context_policy.diagnostics()
            if self._shared is not None:
                self._shared.context_policy_version = policy_path
                self._shared.context_policy_core_count = int(diag.get("core_keys", 0))
                self._shared.context_policy_challenger_count = int(diag.get("challenger_keys", 0))
                self._shared.context_policy_explore_count = int(diag.get("explore_keys", 0))
            log.info(
                "Context policy reloaded",
                data={
                    "path": policy_path,
                    "core": int(diag.get("core_keys", 0)),
                    "challenger": int(diag.get("challenger_keys", 0)),
                    "explore": int(diag.get("explore_keys", 0)),
                },
            )

    async def _data_quality_monitor(self, check_interval_s: float = 60.0, halt_threshold_minutes: float = 10.0) -> None:
        # Live-mode circuit breaker: halt execution when configured families have no valid markets.
        #
        # In dry-run / paper mode this only logs a warning — no real money is at risk.
        # In live mode (ws_trading_enabled=True, dry_run=False) it sets
        # SharedFarmState.data_quality_halted, which blocks all signal generation
        # until markets recover.
        #
        # Why: Kalshi occasionally returns bad market metadata (e.g. strike_price = time
        # fragments after DST transitions). When that happens, our parser correctly skips
        # the market and returns None, but the UI shows EMPTY and no signals are generated.
        # In live mode, silently having zero valid markets for >10 minutes is dangerous —
        # this guard makes the outage explicit and stops order placement.
        cfg = self._base_cfg0
        if cfg is None:
            return

        is_live = not cfg.dry_run and cfg.ws_trading_enabled
        halt_threshold_s = halt_threshold_minutes * 60.0

        # Parse live_families: "BTC 15M" -> ("BTC", 15), "ETH 60M" -> ("ETH", 60)
        family_specs: Dict[str, tuple] = {}
        for family in (cfg.live_families or []):
            parts = family.upper().split()
            if len(parts) == 2:
                try:
                    win = int(parts[1].replace("M", ""))
                    family_specs[family] = (parts[0], win)
                except ValueError:
                    pass

        if not family_specs:
            return

        family_empty_since: Dict[str, float] = {}  # family -> monotonic ts when first went to 0

        try:
            while self._running:
                await asyncio.sleep(check_interval_s)
                now = time.monotonic()

                # Count valid markets per family from shared market metadata.
                family_counts: Dict[str, int] = {f: 0 for f in family_specs}
                for ticker, asset in list(self._shared.market_asset.items()):
                    win_min = self._shared.market_window_min.get(ticker, 0)
                    for family_name, (f_asset, f_win) in family_specs.items():
                        if asset == f_asset and win_min == f_win:
                            family_counts[family_name] += 1

                halted_families = []
                for family_name in family_specs:
                    if family_counts.get(family_name, 0) == 0:
                        if family_name not in family_empty_since:
                            family_empty_since[family_name] = now
                        empty_s = now - family_empty_since[family_name]
                        if empty_s >= halt_threshold_s:
                            halted_families.append(f"{family_name} ({empty_s / 60:.0f}min)")
                    else:
                        family_empty_since.pop(family_name, None)

                if halted_families:
                    reason = "No valid markets for: " + ", ".join(halted_families)
                    if is_live and not self._shared.data_quality_halted:
                        self._shared.data_quality_halted = True
                        self._shared.data_quality_halt_reason = reason
                        log.critical(
                            "LIVE TRADING HALTED — market data quality degraded. "
                            "Restart the program once Kalshi market metadata is valid again.",
                            data={"reason": reason, "halt_threshold_minutes": halt_threshold_minutes},
                        )
                    elif not is_live:
                        log.warning(
                            "Market data quality degraded (paper mode — no halt)",
                            data={"reason": reason},
                        )
                elif self._shared.data_quality_halted:
                    self._shared.data_quality_halted = False
                    self._shared.data_quality_halt_reason = ""
                    log.info("Live trading RESUMED — market data quality restored for all families")

        except asyncio.CancelledError:
            pass

    async def _pipeline_diagnostics(self, interval_s: float = 30.0) -> None:
        # Periodic pipeline health logs for stale-feed debugging.
        try:
            while self._running:
                await asyncio.sleep(interval_s)
                now = time.monotonic()
                self._maybe_reload_context_policy(now_ts=time.time())
                cp_diag = self._context_policy.diagnostics() if self._context_policy is not None else {}
                dg_diag = self._drift_guard.diagnostics() if self._drift_guard is not None else {}
                ad_diag = self._adaptive_cap.diagnostics() if self._adaptive_cap is not None else {}
                et_diag = self._edge_tracker.diagnostics() if self._edge_tracker is not None else {}

                truth_ages = {
                    asset: round(now - ts, 2)
                    for asset, ts in self._shared.last_truth_tick_by_asset.items()
                }
                stale_truth_assets = [
                    asset for asset, age in truth_ages.items()
                    if age > self._truth_stale_timeout_s
                ]

                ob_ages = sorted(
                    (
                        (ticker, now - ts)
                        for ticker, ts in self._last_ob_mono_by_ticker.items()
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
                prob_ages = sorted(
                    (
                        (ticker, now - ts)
                        for ticker, ts in self._last_prob_mono_by_ticker.items()
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
                worst_ob = [{"ticker": t, "age_s": round(age, 2)} for t, age in ob_ages[:3]]
                worst_prob = [{"ticker": t, "age_s": round(age, 2)} for t, age in prob_ages[:3]]

                dispatch_samples = list(self._dispatch_latency_ms)
                dispatch_p50 = round(statistics.median(dispatch_samples), 2) if dispatch_samples else 0.0
                dispatch_p95 = round(
                    statistics.quantiles(dispatch_samples, n=100)[94],
                    2,
                ) if len(dispatch_samples) >= 100 else dispatch_p50
                ob_gap_samples = list(self._ob_truth_gap_ms)
                prob_gap_samples = list(self._prob_truth_gap_ms)

                def _quantiles(samples: List[float]) -> Dict[str, float]:
                    if not samples:
                        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
                    p50 = round(statistics.median(samples), 2)
                    p95 = round(statistics.quantiles(samples, n=100)[94], 2) if len(samples) >= 100 else p50
                    return {"p50": p50, "p95": p95, "max": round(max(samples), 2)}

                cap_top = []
                crowding_top = []
                if self._shared is not None:
                    cap_top = [
                        {"key": k, "util_pct": round(v * 100.0, 2)}
                        for k, v in sorted(
                            self._shared.projected_cap_util_by_side.items(),
                            key=lambda kv: kv[1],
                            reverse=True,
                        )[:8]
                    ]
                    crowding_top = [
                        {"key": k, "fills_per_sec": round(v, 3)}
                        for k, v in sorted(
                            self._shared.crowding_fills_per_sec_by_side.items(),
                            key=lambda kv: kv[1],
                            reverse=True,
                        )[:8]
                    ]

                log.info(
                    "Farm pipeline diagnostics",
                    data={
                        "window_s": interval_s,
                        "truth_stale": self._shared.truth_stale,
                        "ws_halted": self._shared.ws_halted,
                        "truth_ages_s": truth_ages,
                        "stale_truth_assets": stale_truth_assets,
                        "ob_updates": self._ob_updates,
                        "prob_updates": self._prob_updates,
                        "tracked_ob_tickers": len(self._last_ob_mono_by_ticker),
                        "tracked_prob_tickers": len(self._last_prob_mono_by_ticker),
                        "worst_ob_age": worst_ob,
                        "worst_prob_age": worst_prob,
                        "dispatch_calls": self._dispatch_count,
                        "dispatch_signals": self._dispatch_signal_count,
                        "dispatch_gate_reasons": {
                            "strategy": dict(
                                sorted(
                                    self._strategy_evaluator.drain_diagnostics().items(),
                                    key=lambda kv: kv[1],
                                    reverse=True,
                                )[:15]
                            ),
                            "scalper": dict(
                                sorted(
                                    self._scalper_evaluator.drain_diagnostics().items(),
                                    key=lambda kv: kv[1],
                                    reverse=True,
                                )[:15]
                            ),
                            "arb": dict(
                                sorted(
                                    self._arb_evaluator.drain_diagnostics().items(),
                                    key=lambda kv: kv[1],
                                    reverse=True,
                                )[:15]
                            ),
                        },
                        "dispatch_latency_ms": {
                            "p50": dispatch_p50,
                            "p95": dispatch_p95,
                            "max": round(max(dispatch_samples), 2) if dispatch_samples else 0.0,
                        },
                        "truth_gap_ms": {
                            "orderbook": _quantiles(ob_gap_samples),
                            "probability": _quantiles(prob_gap_samples),
                        },
                        "dispatch_queue_depth": self._dispatch_queue.qsize(),
                        "dispatching_tickers": len(self._dispatching_tickers),
                        "queued_unique_tickers": len(self._queued_tickers),
                        "redispatch_tickers": len(self._redispatch_tickers),
                        "side_guard_active_blocks": len(
                            [
                                k for k, ts in self._shared.side_guard_block_until.items()
                                if ts > time.time()
                            ]
                        ) if self._shared is not None else 0,
                        "cap_projected_util_top": cap_top,
                        "param_region_penalty_hits": (
                            int(self._shared.param_region_penalty_hits) if self._shared is not None else 0
                        ),
                        "param_region_block_hits": (
                            int(self._shared.param_region_block_hits) if self._shared is not None else 0
                        ),
                        "crowding_top_fills_per_sec": crowding_top,
                        "crowding_candidate_events": (
                            int(self._shared.crowding_candidate_events) if self._shared is not None else 0
                        ),
                        "crowding_active_events": (
                            int(self._shared.crowding_active_events) if self._shared is not None else 0
                        ),
                        "context_policy_weight_applied": (
                            int(self._shared.context_policy_weight_applied) if self._shared is not None else 0
                        ),
                        "context_policy_core_count": (
                            int(cp_diag.get("core_keys", 0))
                        ),
                        "context_policy_challenger_count": (
                            int(cp_diag.get("challenger_keys", 0))
                        ),
                        "context_policy_dropped_new_keys": int(cp_diag.get("dropped_new_keys", 0)),
                        "drift_guard_alerts": (
                            int(self._shared.drift_guard_alerts) if self._shared is not None else 0
                        ),
                        "drift_guard_auto_demotions": (
                            int(self._shared.drift_guard_auto_demotions) if self._shared is not None else 0
                        ),
                        "drift_guard_dropped_new_keys": int(dg_diag.get("dropped_new_keys", 0)),
                        "adaptive_cap_events": (
                            int(self._shared.adaptive_cap_events) if self._shared is not None else 0
                        ),
                        "adaptive_cap_dropped_new_keys": int(ad_diag.get("dropped_new_keys", 0)),
                        "edge_tracking_weight_applied": (
                            int(self._shared.edge_tracking_weight_applied) if self._shared is not None else 0
                        ),
                        "edge_tracking_poor_retention_keys": int(et_diag.get("keys_poor_retention", 0)),
                        "edge_tracking_dropped_new_keys": int(et_diag.get("dropped_new_keys", 0)),
                        "population_scale_stage": (
                            int(self._shared.population_current_stage) if self._shared is not None else 0
                        ),
                        "population_scale_multiplier": (
                            float(self._shared.population_current_multiplier) if self._shared is not None else 1.0
                        ),
                        "population_scale_events": (
                            int(self._shared.population_scale_events) if self._shared is not None else 0
                        ),
                    },
                )

                self._ob_updates = 0
                self._prob_updates = 0
                self._dispatch_count = 0
                self._dispatch_signal_count = 0
                self._dispatch_latency_ms.clear()
                self._ob_truth_gap_ms.clear()
                self._prob_truth_gap_ms.clear()
                if self._shared is not None:
                    self._shared.projected_cap_util_by_side.clear()
                    self._shared.param_region_penalty_hits = 0
                    self._shared.param_region_block_hits = 0
                    self._shared.crowding_candidate_events = 0
                    self._shared.crowding_active_events = 0
                    self._shared.context_policy_weight_applied = 0
                    self._shared.edge_tracking_weight_applied = 0
                    # Update policy summary counts
                    if self._context_policy is not None:
                        cp_diag = self._context_policy.diagnostics()
                        self._shared.context_policy_core_count = cp_diag.get("core_keys", 0)
                        self._shared.context_policy_challenger_count = cp_diag.get("challenger_keys", 0)
                        self._shared.context_policy_explore_count = cp_diag.get("explore_keys", 0)
                    if self._edge_tracker is not None:
                        et_diag = self._edge_tracker.diagnostics()
                        self._shared.edge_tracking_poor_retention_keys = int(
                            et_diag.get("keys_poor_retention", 0)
                        )
        except asyncio.CancelledError:
            pass

        except Exception as exc:
            log.error("Farm pipeline diagnostics loop crashed", data={"error": str(exc)})


#  FarmExecutionCoordinator

class FarmExecutionCoordinator:
    # Single bus router for all farm execution engines.

    def __init__(
        self,
        bus: Bus,
        strategies: List[StrategyEngine],
        scalpers: List[MispricingScalper],
        executions: List[ExecutionEngine],
        shared_orderbooks: Dict[str, OrderbookState],
        shared_metadata: Dict[str, MarketMetadata],
        shared_state: Optional[SharedFarmState] = None,
    ) -> None:
        self._bus = bus
        self._strategies = strategies
        self._scalpers = scalpers
        self._executions = executions
        self._shared_orderbooks = shared_orderbooks
        self._shared_metadata = shared_metadata
        self._by_bot_id: Dict[str, ExecutionEngine] = {
            e._cfg.bot_id: e for e in executions if e._cfg.bot_id
        }
        self._strategy_by_bot_id: Dict[str, StrategyEngine] = {
            s._cfg.bot_id: s for s in strategies if s._cfg.bot_id
        }
        self._scalper_by_bot_id: Dict[str, MispricingScalper] = {
            s._cfg.bot_id: s for s in scalpers if s._cfg.bot_id
        }
        self._shared_state = shared_state
        self._tasks: List[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        self._running = True
        q_signals = await self._bus.subscribe("kalshi.trade_signal")
        q_orders = await self._bus.subscribe("kalshi.user_orders")
        q_fills = await self._bus.subscribe("kalshi.fills")
        q_risk = await self._bus.subscribe("kalshi.risk")
        q_ws = await self._bus.subscribe("kalshi.ws.status")
        q_balance = await self._bus.subscribe("kalshi.account_balance")
        q_meta = await self._bus.subscribe("kalshi.market_metadata")
        self._tasks = [
            asyncio.create_task(self._consume_signals(q_signals)),
            asyncio.create_task(self._consume_order_updates(q_orders)),
            asyncio.create_task(self._consume_fills(q_fills)),
            asyncio.create_task(self._consume_risk(q_risk)),
            asyncio.create_task(self._consume_ws_status(q_ws)),
            asyncio.create_task(self._consume_balance(q_balance)),
            asyncio.create_task(self._consume_metadata(q_meta)),
            asyncio.create_task(self._execution_diagnostics()),
        ]
        log.info(
            "FarmExecutionCoordinator started",
            data={"executions": len(self._executions)},
        )

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def register_execution(
        self,
        execution: ExecutionEngine,
        strategy: StrategyEngine,
        scalper: MispricingScalper,
    ) -> None:
        # Hot-add an execution engine and its associated strategy/scalper.
        self._executions.append(execution)
        self._strategies.append(strategy)
        self._scalpers.append(scalper)
        bot_id = execution._cfg.bot_id
        if bot_id:
            self._by_bot_id[bot_id] = execution
            self._strategy_by_bot_id[bot_id] = strategy
            self._scalper_by_bot_id[bot_id] = scalper

    def deregister_execution(self, bot_id: str) -> None:
        # Hot-remove an execution engine and its associated strategy/scalper by bot_id.
        self._executions = [e for e in self._executions if e._cfg.bot_id != bot_id]
        self._strategies = [s for s in self._strategies if s._cfg.bot_id != bot_id]
        self._scalpers = [s for s in self._scalpers if s._cfg.bot_id != bot_id]
        self._by_bot_id.pop(bot_id, None)
        self._strategy_by_bot_id.pop(bot_id, None)
        self._scalper_by_bot_id.pop(bot_id, None)


    async def _consume_signals(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                signal = await q.get()
                engine = self._by_bot_id.get(getattr(signal, "bot_id", "default"))
                if engine is not None:
                    await engine.handle_signal(signal)
        except asyncio.CancelledError:
            pass

    async def _consume_order_updates(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                update: OrderUpdate = await q.get()
                engine = self._by_bot_id.get(getattr(update, "bot_id", "default"))
                if engine is not None:
                    await engine.handle_order_update(update)
        except asyncio.CancelledError:
            pass

    async def _consume_fills(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                fill = await q.get()
                bot_id = getattr(fill, "bot_id", "default")
                engine = self._by_bot_id.get(bot_id)
                if engine is not None:
                    await engine.handle_fill(fill)
                strategy = self._strategy_by_bot_id.get(bot_id)
                if strategy is not None:
                    strategy.handle_fill(fill)
                scalper = self._scalper_by_bot_id.get(bot_id)
                if scalper is not None:
                    scalper.handle_fill(fill)
                if self._shared_state is not None:
                    key = _crowding_key(
                        str(bot_id or ""),
                        str(getattr(fill, "market_ticker", "") or ""),
                        str(getattr(fill, "side", "") or ""),
                    )
                    if key != "||":
                        ts = float(getattr(fill, "timestamp", 0.0) or 0.0) or time.time()
                        cfg = engine._cfg if engine is not None else (self._executions[0]._cfg if self._executions else None)
                        if cfg is not None:
                            window_s = max(1.0, float(getattr(cfg, "crowding_window_s", 10.0)))
                            threshold = float(getattr(cfg, "crowding_fills_per_sec_threshold", 4.0))
                        else:
                            window_s = 10.0
                            threshold = 4.0
                        dq = self._shared_state.crowding_fill_times_by_side.get(key)
                        if dq is None:
                            dq = deque()
                            self._shared_state.crowding_fill_times_by_side[key] = dq
                        dq.append(ts)
                        cutoff = ts - window_s
                        while dq and dq[0] < cutoff:
                            dq.popleft()
                        fills_per_sec = len(dq) / window_s
                        self._shared_state.crowding_fills_per_sec_by_side[key] = fills_per_sec
                        if fills_per_sec >= threshold:
                            self._shared_state.crowding_candidate_events += 1
        except asyncio.CancelledError:
            pass

    async def _consume_risk(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                event: RiskEvent = await q.get()
                await asyncio.gather(
                    *(engine.handle_risk(event) for engine in self._executions),
                    return_exceptions=True,
                )
        except asyncio.CancelledError:
            pass

    async def _consume_ws_status(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                event: WsConnectionEvent = await q.get()
                for engine in self._executions:
                    engine.handle_ws_status(event)
        except asyncio.CancelledError:
            pass

    async def _consume_balance(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                msg = await q.get()
                await asyncio.gather(
                    *(engine.handle_balance(msg) for engine in self._executions),
                    return_exceptions=True,
                )
                for strategy in self._strategies:
                    strategy.handle_balance(msg)
                for scalper in self._scalpers:
                    scalper.handle_balance(msg)
        except asyncio.CancelledError:
            pass

    async def _consume_metadata(self, q: asyncio.Queue) -> None:
        try:
            while self._running:
                msg = await q.get()
                if isinstance(msg, MarketMetadata):
                    self._shared_metadata[msg.market_ticker] = msg
        except asyncio.CancelledError:
            pass

    async def _execution_diagnostics(self, interval_s: float = 30.0) -> None:
        try:
            while self._running:
                await asyncio.sleep(interval_s)
                totals: Dict[str, int] = {}
                sources: Dict[str, int] = {}
                families: Dict[str, int] = {}
                source_families: Dict[str, int] = {}
                gauges: Dict[str, float] = {}
                pending = 0
                pending_markets = 0
                halted = 0
                for engine in self._executions:
                    diag = engine.drain_diagnostics()
                    for key, value in diag.get("counts", {}).items():
                        totals[key] = totals.get(key, 0) + int(value)
                    for key, value in diag.get("sources", {}).items():
                        sources[key] = sources.get(key, 0) + int(value)
                    for key, value in diag.get("families", {}).items():
                        families[key] = families.get(key, 0) + int(value)
                    for key, value in diag.get("source_families", {}).items():
                        source_families[key] = source_families.get(key, 0) + int(value)
                    for key, value in diag.get("gauges", {}).items():
                        gauges[key] = gauges.get(key, 0.0) + float(value)
                    pending += int(diag.get("pending", 0))
                    pending_markets += int(diag.get("pending_markets", 0))
                    if diag.get("halted"):
                        halted += 1
                signals = max(1, int(totals.get("signals_received", 0)))
                paper_fills = int(totals.get("paper_fills", 0))
                partial = int(totals.get("paper_partial_fills", 0))
                timeouts = int(totals.get("paper_timeouts", 0))
                requested = max(1, int(totals.get("paper_requested_contracts", 0)))
                filled = int(totals.get("paper_filled_contracts", 0))
                fill_rate = paper_fills / signals
                partial_rate = partial / max(1, paper_fills)
                timeout_rate = timeouts / signals
                fill_fraction = filled / requested
                avg_fill_latency_ms = gauges.get("paper_fill_latency_ms_total", 0.0) / max(1, paper_fills)
                avg_timeout_latency_ms = gauges.get("paper_timeout_latency_ms_total", 0.0) / max(1, timeouts)
                avg_realized_edge_net_cents = gauges.get("realized_edge_net_cents_sum", 0.0) / max(1, paper_fills)
                log.info(
                    "Farm execution diagnostics",
                    data={
                        "window_s": interval_s,
                        "counts": totals,
                        "signal_sources": dict(sorted(sources.items(), key=lambda kv: kv[1], reverse=True)),
                        "signal_families": dict(sorted(families.items(), key=lambda kv: kv[1], reverse=True)),
                        "signal_source_families": dict(sorted(source_families.items(), key=lambda kv: kv[1], reverse=True)[:20]),
                        "execution_quality": {
                            "fill_rate": round(fill_rate, 4),
                            "partial_rate": round(partial_rate, 4),
                            "timeout_rate": round(timeout_rate, 4),
                            "fill_fraction": round(fill_fraction, 4),
                            "avg_fill_latency_ms": round(avg_fill_latency_ms, 2),
                            "avg_timeout_latency_ms": round(avg_timeout_latency_ms, 2),
                            "avg_realized_edge_net_cents": round(avg_realized_edge_net_cents, 3),
                        },
                        "pending_orders": pending,
                        "pending_markets": pending_markets,
                        "halted_executions": halted,
                        "population": self.get_population_diagnostics(),
                    },
                )
        except asyncio.CancelledError:
            pass


#  KalshiPaperFarm

class KalshiPaperFarm:
    def __init__(
        self,
        configs: Iterable[KalshiConfig],
        bus: Bus,
        rest: KalshiRestClient,
        db: Optional[Any] = None,
    ) -> None:
        self._configs = list(configs)
        self._bus = bus
        self._rest = rest
        self._db = db
        self._strategies: List[StrategyEngine] = []
        self._scalpers: List[MispricingScalper] = []
        self._executions: List[ExecutionEngine] = []
        self._dispatcher: Optional[FarmDispatcher] = None
        self._shared: Optional[SharedFarmState] = None
        self._execution_coordinator: Optional[FarmExecutionCoordinator] = None
        self._shared_metadata: Dict[str, MarketMetadata] = {}
        # Regime gate (Phase 1).
        self._regime_gate: Optional[RegimeGate] = None
        # Family weight manager (Phase 2-3).
        self._family_weight_manager: Optional[FamilyWeightManager] = None
        # Parameter-region penalty engine (Phase 2).
        self._param_region_engine: Optional[_ParamRegionPenaltyEngine] = None
        # Context policy engine (soft allocation).
        self._context_policy: Optional[ContextPolicyEngine] = None
        self._drift_guard: Optional[DriftGuard] = None
        self._adaptive_cap: Optional[AdaptiveCapEngine] = None
        self._edge_tracker: Optional[EdgeTracker] = None
        self._population_scaler: Optional[PopulationScaler] = None
        # Bot -> primary family mapping (source of truth for per-family epoch).
        self._bot_family_map: Dict[str, str] = {}
        # Population management (Phase 3-4).
        self._equity_ledgers: Dict[str, BotEquityLedger] = {}
        self._run_records: Dict[str, BotRunRecord] = {}
        self._population_manager: Optional[PopulationManager] = None
        self._population_tasks: List[asyncio.Task] = []
        self._generation: int = 0
        # Stored at start() for use in reseed
        self._active_market_tickers: List[str] = []
        self._configs_by_bot_id: Dict[str, KalshiConfig] = {}
        self._base_cfg0: Optional[KalshiConfig] = None
        self._scenario_target_weights: Dict[str, float] = {
            "best": 0.20,
            "base": 0.60,
            "stress": 0.20,
        }
        self._population_epoch_count: int = 0
        self._reseed_total: int = 0
        self._reseed_epoch: int = 0
        self._reseed_drawdown: int = 0
        self._last_reseed_ts: float = 0.0
        self._last_epoch_ts: float = 0.0
        self._population_min_trades_for_eval: int = 0
        self._population_min_trades_for_drawdown: int = 0
        self._population_epoch_minutes: float = 60.0
        self._family_exploit_fraction: float = 0.80
        self._family_explore_fraction: float = 0.20
        self._side_perf_stats: Dict[str, Dict[str, float]] = {}
        self._policy_last_reload_check_ts: float = 0.0
        self._policy_last_mtime: float = 0.0
        self._decision_tape_writer: Optional[DecisionTapeWriter] = None

    # Batch size for parallel bot startup (avoids sequential awaits)
    _START_BATCH_SIZE = 128

    async def start(self, market_tickers: List[str]) -> None:
        # Create ONE shared state object for all bots.
        shared = SharedFarmState()
        default_families = list(dict.fromkeys(FAMILIES))
        shared.family_weights = {family: 1.0 for family in default_families}
        self._shared = shared

        seen_ids: set[str] = set()
        configs_with_ids: List[KalshiConfig] = []
        for idx, raw_cfg in enumerate(self._configs, start=1):
            bot_id = (raw_cfg.bot_id or "").strip()
            if not bot_id or bot_id == "default" or bot_id in seen_ids:
                bot_id = f"farm_{idx:03d}"
            while bot_id in seen_ids:
                bot_id = f"{bot_id}_x"
            seen_ids.add(bot_id)
            configs_with_ids.append(replace(raw_cfg, bot_id=bot_id))

        # Apply staged population scaling at startup (deterministic).
        if configs_with_ids:
            scale_probe = PopulationScaler(configs_with_ids[0])
            target_count = scale_probe.effective_bot_count(len(configs_with_ids))
            if target_count != len(configs_with_ids):
                base = list(sorted(configs_with_ids, key=lambda c: c.bot_id))
                if target_count < len(base):
                    configs_with_ids = base[:target_count]
                else:
                    expanded: List[KalshiConfig] = []
                    for i in range(target_count):
                        src = base[i % len(base)]
                        expanded.append(replace(src, bot_id=f"{src.bot_id}_s{i+1:05d}"))
                    configs_with_ids = expanded
                log.info(
                    "Population scaling applied at startup",
                    data={
                        "base_count": len(self._configs),
                        "scaled_count": len(configs_with_ids),
                        "stage": scale_probe.current_stage,
                        "multiplier": scale_probe.current_multiplier,
                    },
                )

        # If population manager is enabled and all bots are on the same scenario,
        # pre-seed a balanced scenario mix so execution assumptions are actually diversified.
        if configs_with_ids:
            cfg0_pre = configs_with_ids[0]
            if getattr(cfg0_pre, "enable_population_manager", False):
                scenario_set = {getattr(c, "scenario_profile", "base") for c in configs_with_ids}
                if len(scenario_set) == 1:
                    n = len(configs_with_ids)
                    n_best = max(1, int(round(n * self._scenario_target_weights["best"])))
                    n_stress = max(1, int(round(n * self._scenario_target_weights["stress"])))
                    if n_best + n_stress >= n:
                        n_best = max(1, n // 5)
                        n_stress = max(1, n // 5)
                    n_base = max(0, n - n_best - n_stress)
                    profile_slots = (
                        ["best"] * n_best +
                        ["base"] * n_base +
                        ["stress"] * n_stress
                    )
                    profile_slots = profile_slots[:n]
                    while len(profile_slots) < n:
                        profile_slots.append("base")
                    # Deterministic shuffle by bot_id ordering (stable across restarts).
                    for i, cfg in enumerate(sorted(configs_with_ids, key=lambda x: x.bot_id)):
                        assigned = profile_slots[i]
                        # Update in-place list entry by bot_id match.
                        for j, original in enumerate(configs_with_ids):
                            if original.bot_id == cfg.bot_id:
                                configs_with_ids[j] = replace(original, scenario_profile=assigned)
                                break
                    log.info(
                        "PopulationManager scenario mix seeded",
                        data={
                            "best": n_best,
                            "base": n_base,
                            "stress": n_stress,
                        },
                    )
        self._active_market_tickers = list(market_tickers)
        self._configs_by_bot_id = {cfg.bot_id: cfg for cfg in configs_with_ids}

        for i in range(0, len(configs_with_ids), self._START_BATCH_SIZE):
            batch = configs_with_ids[i : i + self._START_BATCH_SIZE]
            strategies_batch: List[StrategyEngine] = []
            scalpers_batch: List[MispricingScalper] = []
            executions_batch: List[ExecutionEngine] = []

            for cfg in batch:
                # Inject shared state — eliminates per-bot high-freq subscriptions.
                strategy = StrategyEngine(cfg, self._bus, shared=shared)
                scalper = MispricingScalper(cfg, self._bus, shared=shared)
                execution = ExecutionEngine(
                    cfg,
                    self._bus,
                    self._rest,
                    db=self._db,
                    shared_orderbooks=shared.orderbooks,
                    shared_metadata=self._shared_metadata,
                    shared_truth_prices=shared.truth_price_by_asset,
                )
                strategies_batch.append(strategy)
                scalpers_batch.append(scalper)
                executions_batch.append(execution)

            # start() with shared state only creates 3 tasks per bot (fills/ws/balance).
            await asyncio.gather(
                *(s.start(market_tickers) for s in strategies_batch),
                return_exceptions=True,
            )
            await asyncio.gather(
                *(s.start() for s in scalpers_batch),
                return_exceptions=True,
            )
            await asyncio.gather(
                *(e.start(subscribe_bus=False) for e in executions_batch),
                return_exceptions=True,
            )
            self._strategies.extend(strategies_batch)
            self._scalpers.extend(scalpers_batch)
            self._executions.extend(executions_batch)

            # Yield after each batch so the event loop can service I/O.
            await asyncio.sleep(0)

        log.info(
            f"All {len(configs_with_ids)} farm bots started (farm mode — shared state)",
            data={"tasks_per_bot": 2, "total_bots": len(configs_with_ids)},
        )

        self._execution_coordinator = FarmExecutionCoordinator(
            self._bus,
            self._strategies,
            self._scalpers,
            self._executions,
            shared.orderbooks,
            self._shared_metadata,
            shared_state=shared,
        )
        await self._execution_coordinator.start()
        log.info("FarmExecutionCoordinator started - single router for execution events")

        # Start the single FarmDispatcher that evaluates all bots on every tick.
        cfg0 = configs_with_ids[0] if configs_with_ids else self._configs[0]
        self._param_region_engine = _ParamRegionPenaltyEngine(cfg0)
        if getattr(cfg0, "decision_tape_enabled", False):
            run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            tape_path = str(getattr(cfg0, "decision_tape_path", "logs/decision_tape/tape_{run_id}.jsonl"))
            tape_path = tape_path.format(run_id=run_id)
            self._decision_tape_writer = DecisionTapeWriter(
                path=tape_path,
                signal_sample_rate=float(getattr(cfg0, "decision_tape_signal_sample_rate", 1.0)),
                rejection_sample_rate=float(getattr(cfg0, "decision_tape_rejection_sample_rate", 0.10)),
            )
            log.info("Decision tape enabled", data={"path": tape_path})

        # ── Context policy engines (opt-in) ────────────────────────────────
        self._context_policy = ContextPolicyEngine(cfg0)
        self._drift_guard = DriftGuard(cfg0)
        self._adaptive_cap = AdaptiveCapEngine(cfg0)
        self._edge_tracker = EdgeTracker(cfg0)
        self._population_scaler = PopulationScaler(cfg0)

        # Load persisted policy if available
        policy_path = getattr(cfg0, "context_policy_file", "")
        if policy_path and self._context_policy.load_policy(policy_path):
            diag = self._context_policy.diagnostics()
            shared.context_policy_version = policy_path
            shared.context_policy_core_count = diag.get("core_keys", 0)
            shared.context_policy_challenger_count = diag.get("challenger_keys", 0)
            shared.context_policy_explore_count = diag.get("explore_keys", 0)
            try:
                self._policy_last_mtime = Path(policy_path).stat().st_mtime
            except OSError:
                self._policy_last_mtime = 0.0
            log.info(
                "Context policy loaded",
                data={
                    "path": policy_path,
                    "core": diag.get("core_keys", 0),
                    "challenger": diag.get("challenger_keys", 0),
                    "explore": diag.get("explore_keys", 0),
                },
            )
        else:
            log.info("Context policy: no persisted policy loaded (starting fresh)")

        if self._population_scaler is not None:
            shared.population_current_stage = self._population_scaler.current_stage
            shared.population_current_multiplier = self._population_scaler.current_multiplier

        # ── Phase 1: Instantiate RegimeGate ──────────────────────────────
        regime_gate: Optional[RegimeGate] = None
        if getattr(cfg0, "enable_regime_gating", False):
            regime_gate = RegimeGate(cfg0, shared)
            self._regime_gate = regime_gate
            log.info(
                "\n"
                "┌─────────────────────────────────────────────────────┐\n"
                "│        REGIME GATE: ON                              │\n"
                f"│  fallback_mode : {cfg0.regime_fallback_mode:<34s}│\n"
                f"│  risk_off_mult : {str(cfg0.risk_off_qty_multiplier):<34s}│\n"
                "│  source        : bus bridge → SharedFarmState       │\n"
                "│  standalone    : fallback applies until bridge live │\n"
                "└─────────────────────────────────────────────────────┘",
            )
        else:
            log.info(
                "\n"
                "┌─────────────────────────────────────────────────────┐\n"
                "│        REGIME GATE: OFF                             │\n"
                "│  enable_regime_gating=false in config               │\n"
                "│  All entries pass without regime checks.            │\n"
                "└─────────────────────────────────────────────────────┘",
            )

        self._dispatcher = FarmDispatcher(
            self._bus,
            shared,
            self._strategies,
            self._scalpers,
            shared_metadata=self._shared_metadata,
            truth_stale_timeout_s=getattr(cfg0, "truth_feed_stale_timeout_s", 30.0),
            executor_workers=(cfg0.compute_threads if getattr(cfg0, "compute_threads", 0) > 0 else None),
            regime_gate=regime_gate,
            param_region_engine=self._param_region_engine,
            family_context_features_enabled=bool(getattr(cfg0, "family_context_features_enabled", False)),
            context_policy=self._context_policy,
            drift_guard=self._drift_guard,
            adaptive_cap=self._adaptive_cap,
            edge_tracker=self._edge_tracker,
            base_cfg0=cfg0,
            tape_writer=self._decision_tape_writer,
        )
        await self._dispatcher.start()
        log.info("FarmDispatcher started — single consumer for all high-frequency topics")

        # --- Population manager (opt-in) ---
        cfg0 = configs_with_ids[0] if configs_with_ids else self._configs[0]
        self._base_cfg0 = cfg0
        if getattr(cfg0, "enable_population_manager", False):
            self._population_manager = PopulationManager(
                exploit_fraction=cfg0.population_exploit_fraction,
                retire_bottom_pct=cfg0.population_retire_bottom_pct,
                drawdown_retire_pct=cfg0.drawdown_retire_pct,
            )
            self._population_min_trades_for_eval = int(
                getattr(cfg0, "population_min_trades_for_eval", 20)
            )
            self._population_min_trades_for_drawdown = int(
                getattr(cfg0, "population_min_trades_for_drawdown", 8)
            )
            self._population_epoch_minutes = float(
                getattr(cfg0, "population_epoch_minutes", 60.0)
            )
            self._family_exploit_fraction = float(
                getattr(cfg0, "family_exploit_fraction", cfg0.population_exploit_fraction)
            )
            self._family_explore_fraction = float(
                getattr(cfg0, "family_explore_fraction", 1.0 - self._family_exploit_fraction)
            )
            # Initialize equity ledgers and run records for all bots.
            for cfg in configs_with_ids:
                run_id = BotRunRecord.new_run_id()
                self._equity_ledgers[cfg.bot_id] = BotEquityLedger(
                    start_equity=cfg.bankroll_usd
                )
                self._run_records[cfg.bot_id] = BotRunRecord(
                    run_id=run_id,
                    bot_id=cfg.bot_id,
                    generation=0,
                    scenario=cfg.scenario_profile,
                    start_equity=cfg.bankroll_usd,
                    started_at=time.time(),
                )
            epoch_min = getattr(cfg0, "population_epoch_minutes", 60.0)
            q_outcomes = await self._bus.subscribe("kalshi.settlement_outcome")
            self._population_tasks = [
                asyncio.create_task(self._consume_settlement_outcomes(q_outcomes)),
                asyncio.create_task(self._drawdown_monitor_loop()),
                asyncio.create_task(self._epoch_loop(epoch_min)),
            ]
            log.info(
                "PopulationManager enabled",
                data={
                    "exploit_fraction": cfg0.population_exploit_fraction,
                    "retire_bottom_pct": cfg0.population_retire_bottom_pct,
                    "drawdown_retire_pct": cfg0.drawdown_retire_pct,
                    "epoch_minutes": epoch_min,
                    "min_trades_eval": self._population_min_trades_for_eval,
                    "min_trades_drawdown": self._population_min_trades_for_drawdown,
                },
            )

        # ── Phase 2: FamilyWeightManager (opt-in) ──────────────────────
        if getattr(cfg0, "family_population_enabled", False):
            configured_families = list(getattr(cfg0, "live_families", []) or [])
            if not configured_families:
                configured_families = default_families
            if not bool(getattr(cfg0, "include_range_markets", True)):
                configured_families = [f for f in configured_families if "Range" not in str(f)]
            active_families = list(dict.fromkeys(configured_families))
            if not active_families:
                active_families = default_families
            self._family_weight_manager = FamilyWeightManager(
                families=active_families,
                min_weight=cfg0.family_min_weight,
                max_weight=cfg0.family_max_weight,
                rebalance_interval_s=cfg0.family_rebalance_interval_minutes * 60.0,
            )
            shared.family_weights = dict(self._family_weight_manager.weights)
            self._population_tasks.append(
                asyncio.create_task(self._family_rebalance_loop())
            )
            log.info(
                "FamilyWeightManager enabled",
                data={
                    "families": active_families,
                    "min_weight": cfg0.family_min_weight,
                    "max_weight": cfg0.family_max_weight,
                    "rebalance_interval_min": cfg0.family_rebalance_interval_minutes,
                    "per_family_epoch": True,
                    "context_features": cfg0.family_context_features_enabled,
                },
            )

    async def stop(self) -> None:
        for t in self._population_tasks:
            t.cancel()
        if self._population_tasks:
            await asyncio.gather(*self._population_tasks, return_exceptions=True)
            self._population_tasks.clear()
        if self._execution_coordinator is not None:
            await self._execution_coordinator.stop()
        if self._dispatcher is not None:
            await self._dispatcher.stop()
        if self._decision_tape_writer is not None:
            self._decision_tape_writer.flush_and_close()
            self._decision_tape_writer = None
        await asyncio.gather(*(s.stop() for s in self._strategies), return_exceptions=True)
        await asyncio.gather(*(s.stop() for s in self._scalpers), return_exceptions=True)
        await asyncio.gather(*(e.stop() for e in self._executions), return_exceptions=True)
        # Persist context policy at shutdown so learned context evidence carries over.
        if self._base_cfg0 is not None and self._context_policy is not None:
            policy_path = str(getattr(self._base_cfg0, "context_policy_file", "") or "")
            if policy_path:
                try:
                    self._context_policy.save_policy(policy_path)
                    log.info("Context policy saved on shutdown", data={"path": policy_path})
                except Exception as exc:
                    log.warning(
                        "Failed to save context policy on shutdown",
                        data={"path": policy_path, "error": str(exc)},
                    )

    async def update_tickers(self, added: List[str], removed: List[str]) -> None:
        # Pass ticker updates down to all strategy instances
        await asyncio.gather(
            *(s.update_tickers(added, removed) for s in self._strategies),
            return_exceptions=True
        )

    # --- Population management background tasks ---

    async def _drawdown_monitor_loop(self, interval_s: float = 30.0) -> None:
        # Periodic check: retire and reseed any bot whose drawdown breaches the hard-stop.
        try:
            while True:
                await asyncio.sleep(interval_s)
                if self._population_manager is None:
                    continue
                for bot_id, ledger in list(self._equity_ledgers.items()):
                    # Skip bots whose run is already marked ended
                    record = self._run_records.get(bot_id)
                    if record and record.reason_ended:
                        continue
                    if ledger.trade_count < self._population_min_trades_for_drawdown:
                        continue
                    reason = self._population_manager.check_drawdown(
                        bot_id,
                        ledger.equity,
                        ledger.peak_equity,
                        ledger.start_equity,
                    )
                    if reason:
                        log.info(
                            f"Drawdown retire+reseed: {bot_id} "
                            f"(dd={ledger.drawdown_pct()*100:.1f}%)"
                        )
                        self._reseed_total += 1
                        self._reseed_drawdown += 1
                        self._last_reseed_ts = time.time()
                        await self._retire_and_reseed_bot(bot_id, reason, is_exploit=True)
        except asyncio.CancelledError:
            pass

    async def _consume_settlement_outcomes(self, q: asyncio.Queue) -> None:
        # Update per-bot equity ledgers and family pnl from realized outcomes.
        try:
            while True:
                out = await q.get()
                bot_id = getattr(out, "bot_id", "default")
                if not bot_id:
                    continue
                ledger = self._equity_ledgers.get(bot_id)
                record = self._run_records.get(bot_id)
                if ledger is None or record is None or record.reason_ended:
                    continue
                gross = float(getattr(out, "gross_pnl", getattr(out, "pnl", 0.0)) or 0.0)
                fees = float(getattr(out, "fees_usd", 0.0) or 0.0)
                net = float(getattr(out, "pnl", gross - fees) or 0.0)
                # Keep ledger cost accounting explicit when gross is present.
                if abs((gross - fees) - net) < 1e-9:
                    ledger.record_trade(gross_pnl=gross, fee_usd=fees)
                else:
                    ledger.record_trade(gross_pnl=net)

                meta = self._shared_metadata.get(getattr(out, "market_ticker", ""))
                family = assign_family(
                    getattr(out, "market_ticker", ""),
                    getattr(meta, "asset", "") if meta else "",
                    int(getattr(meta, "window_minutes", 0) or 0) if meta else 0,
                    bool(getattr(meta, "is_range", False)) if meta else False,
                )
                record.family_pnl[family] = record.family_pnl.get(family, 0.0) + net

                # Phase 2: update family weight manager and bot-family map.
                self._bot_family_map[bot_id] = family
                if self._family_weight_manager is not None:
                    self._family_weight_manager.record_family_trade(family, net)
                cfg = self._configs_by_bot_id.get(bot_id)
                if cfg is not None and self._param_region_engine is not None:
                    self._param_region_engine.record_settlement(
                        cfg,
                        family,
                        net,
                        side=getattr(out, "side", None),
                        decision_context=getattr(out, "decision_context", None),
                    )

                # Context policy, drift guard, adaptive caps, edge tracking
                dc = getattr(out, "decision_context", None) or {}
                side = getattr(out, "side", "")
                settle_drift = float(dc.get("drift", 0.0))
                ctx_key = build_context_key(
                    family=family,
                    side=side,
                    edge_bucket=str(dc.get("eb", "na")),
                    price_bucket=str(dc.get("pb", "na")),
                    strike_distance_bucket=str(dc.get("sdb", "na")),
                    near_money=bool(dc.get("nm", False)),
                    momentum=str(dc.get("mb", momentum_bucket(settle_drift))),
                )
                if self._context_policy is not None:
                    self._context_policy.record_settlement(ctx_key, net)
                if self._drift_guard is not None:
                    drift_result = self._drift_guard.check_drift(ctx_key)
                    if drift_result == "demote" and self._shared is not None:
                        self._shared.drift_guard_auto_demotions += 1
                    if drift_result is not None and self._shared is not None:
                        self._shared.drift_guard_alerts += 1
                if self._adaptive_cap is not None:
                    # Use projected cap utilization as concentration proxy
                    cap_key = _side_guard_key(getattr(out, "market_ticker", ""), side)
                    conc_share = 0.0
                    if self._shared is not None:
                        conc_share = self._shared.projected_cap_util_by_side.get(cap_key, 0.0)
                    self._adaptive_cap.record_settlement(ctx_key, net, conc_share)
                    if self._shared is not None:
                        ad_diag = self._adaptive_cap.diagnostics()
                        self._shared.adaptive_cap_events = ad_diag.get("adaptive_cap_events", 0)
                        self._shared.adaptive_cap_cooldown_events = ad_diag.get("key_cooldown_events", 0)
                if self._edge_tracker is not None:
                    entry_px = int(dc.get("px", 0))
                    entry_qty = int(dc.get("qty", 1))
                    self._edge_tracker.record_settlement(ctx_key, net, entry_px, entry_qty)

                # Side guard: temporarily block persistent losing (ticker, side) combos.
                cfg0 = self._base_cfg0
                if cfg0 is not None and getattr(cfg0, "side_guard_enabled", False) and self._shared is not None:
                    ticker = getattr(out, "market_ticker", "")
                    side = getattr(out, "side", "")
                    if ticker and side in ("yes", "no"):
                        key = _side_guard_key(ticker, side)
                        stats = self._side_perf_stats.setdefault(key, {"count": 0.0, "wins": 0.0, "pnl": 0.0})
                        stats["count"] += 1.0
                        stats["wins"] += 1.0 if bool(getattr(out, "won", False)) else 0.0
                        stats["pnl"] += net
                        count = int(stats["count"])
                        if count >= int(cfg0.side_guard_min_settles):
                            wr = float(stats["wins"]) / max(1.0, float(stats["count"]))
                            avg_pnl = float(stats["pnl"]) / max(1.0, float(stats["count"]))
                            if wr <= float(cfg0.side_guard_max_win_rate) and avg_pnl <= float(cfg0.side_guard_min_avg_pnl_usd):
                                now_ts = time.time()
                                block_until = now_ts + (float(cfg0.side_guard_block_minutes) * 60.0)
                                prev_until = self._shared.side_guard_block_until.get(key, 0.0)
                                self._shared.side_guard_block_until[key] = max(prev_until, block_until)
                                # Warn only on transition into blocked state to avoid log spam.
                                if prev_until <= now_ts:
                                    log.warning(
                                        "Side guard blocked %s",
                                        key,
                                        data={
                                            "win_rate": round(wr, 4),
                                            "avg_pnl_usd": round(avg_pnl, 4),
                                            "samples": count,
                                            "block_minutes": cfg0.side_guard_block_minutes,
                                        },
                                    )
        except asyncio.CancelledError:
            pass

    async def _epoch_loop(self, epoch_minutes: float) -> None:
        # Periodic epoch evaluation: rank bots, retire+reseed bottom cohort.
        #
        # When family_population_enabled, ranking is done per-family bucket.
        # Otherwise falls back to global ranking.
        try:
            while True:
                await asyncio.sleep(epoch_minutes * 60)
                if self._population_manager is None:
                    continue
                bot_scores = []
                eligible = 0
                for bot_id, ledger in list(self._equity_ledgers.items()):
                    record = self._run_records.get(bot_id)
                    if record and record.reason_ended:
                        continue  # skip already-retired
                    if ledger.trade_count < self._population_min_trades_for_eval:
                        continue
                    eligible += 1
                    stats = {
                        "pnl": ledger.equity - ledger.start_equity,
                        "wins": sum(1 for p in ledger.trade_pnls if p >= 0),
                        "losses": sum(1 for p in ledger.trade_pnls if p < 0),
                        "trade_count": ledger.trade_count,
                        "max_drawdown": ledger.max_drawdown,
                        "tail_loss_10pct": ledger.tail_loss(),
                    }
                    score = sim_robustness_score(stats)
                    bot_scores.append((bot_id, score, stats))

                if self._population_scaler is not None and self._shared is not None:
                    total_pnl = 0.0
                    total_trades = 0
                    max_dd_pct = 0.0
                    for ledger in self._equity_ledgers.values():
                        total_pnl += (ledger.equity - ledger.start_equity)
                        total_trades += int(ledger.trade_count)
                        max_dd_pct = max(max_dd_pct, float(ledger.max_drawdown_pct))
                    expectancy = (total_pnl / total_trades) if total_trades > 0 else 0.0
                    edge_diag = self._edge_tracker.diagnostics() if self._edge_tracker is not None else {"avg_retention": 1.0}
                    edge_retention = float(edge_diag.get("avg_retention", 1.0))
                    top_concentration = 0.0
                    if self._shared.projected_cap_util_by_side:
                        top_concentration = max(float(v) for v in self._shared.projected_cap_util_by_side.values())
                    crowding_stable = bool(self._shared.crowding_active_events == 0)
                    gate = self._population_scaler.evaluate_gate(
                        concentration_share=top_concentration,
                        edge_retention=edge_retention,
                        expectancy_usd=float(expectancy),
                        drawdown_pct=max_dd_pct,
                        crowding_stable=crowding_stable,
                        dispatch_p95_ms=0.0,
                        now_ts=time.time(),
                    )
                    evt = self._population_scaler.attempt_scale(gate, now_ts=time.time())
                    self._shared.population_current_stage = self._population_scaler.current_stage
                    self._shared.population_current_multiplier = self._population_scaler.current_multiplier
                    if evt is not None:
                        self._shared.population_scale_events += 1
                        log.info(
                            "Population scale stage updated (applies on restart)",
                            data={
                                "action": evt.action,
                                "from_stage": evt.from_stage,
                                "to_stage": evt.to_stage,
                                "from_multiplier": evt.from_multiplier,
                                "to_multiplier": evt.to_multiplier,
                                "reason": evt.reason,
                            },
                        )

                # Phase 2: per-family epoch when family_population_enabled.
                use_per_family = (
                    self._family_weight_manager is not None
                    and self._bot_family_map
                )

                if use_per_family:
                    retire_ids, family_split = (
                        self._population_manager.evaluate_epoch_by_family(
                            self._bot_family_map, bot_scores
                        )
                    )
                    # Override exploit/explore split with family-specific fractions.
                    retire_counts_by_family: Dict[str, int] = {}
                    for rid in retire_ids:
                        fam = self._bot_family_map.get(rid, "Other")
                        retire_counts_by_family[fam] = retire_counts_by_family.get(fam, 0) + 1
                    family_split = {}
                    for fam, n_retire in retire_counts_by_family.items():
                        n_exploit = int(round(n_retire * self._family_exploit_fraction))
                        n_exploit = max(0, min(n_retire, n_exploit))
                        n_explore = max(0, n_retire - n_exploit)
                        family_split[fam] = (n_exploit, n_explore)
                    self._population_epoch_count += 1
                    self._last_epoch_ts = time.time()

                    if retire_ids:
                        self._generation += 1
                        self._reseed_total += len(retire_ids)
                        self._reseed_epoch += len(retire_ids)
                        self._last_reseed_ts = time.time()
                        log.info(
                            f"Per-family epoch retire+reseed: {len(retire_ids)} bots "
                            f"(gen {self._generation})",
                            data={
                                "retired": retire_ids[:10],
                                "eligible_bots": eligible,
                                "scored_bots": len(bot_scores),
                                "family_split": {
                                    f: {"exploit": e, "explore": x}
                                    for f, (e, x) in family_split.items()
                                },
                            },
                        )
                        family_budget = dict(family_split)
                        for bot_id in retire_ids:
                            family = self._bot_family_map.get(bot_id, "Other")
                            f_exploit, f_explore = family_budget.get(family, (1, 0))
                            is_exploit = f_exploit > 0
                            if is_exploit:
                                family_budget[family] = (max(0, f_exploit - 1), f_explore)
                            else:
                                family_budget[family] = (f_exploit, max(0, f_explore - 1))
                            await self._retire_and_reseed_bot(
                                bot_id, "retired_epoch",
                                is_exploit=is_exploit,
                                family_hint=family,
                            )
                            if self._family_weight_manager is not None:
                                self._family_weight_manager.record_family_reseed(family)
                            await asyncio.sleep(0)
                    else:
                        log.info(
                            "Per-family epoch complete (no reseeds)",
                            data={"eligible_bots": eligible, "scored_bots": len(bot_scores), "generation": self._generation},
                        )
                else:
                    # Fallback: global ranking.
                    retire_ids, n_exploit, n_explore = (
                        self._population_manager.evaluate_epoch(bot_scores)
                    )
                    self._population_epoch_count += 1
                    self._last_epoch_ts = time.time()
                    if retire_ids:
                        self._generation += 1
                        self._reseed_total += len(retire_ids)
                        self._reseed_epoch += len(retire_ids)
                        self._last_reseed_ts = time.time()
                        log.info(
                            f"Epoch retire+reseed: {len(retire_ids)} bots "
                            f"(gen {self._generation}, exploit={n_exploit}, explore={n_explore})",
                            data={"retired": retire_ids[:10], "eligible_bots": eligible, "scored_bots": len(bot_scores)},
                        )
                        for idx, bot_id in enumerate(retire_ids):
                            is_exploit = idx < n_exploit
                            await self._retire_and_reseed_bot(
                                bot_id, "retired_epoch", is_exploit=is_exploit
                            )
                            await asyncio.sleep(0)
                    else:
                        log.info(
                            "Epoch evaluation complete (no reseeds)",
                            data={"eligible_bots": eligible, "scored_bots": len(bot_scores), "generation": self._generation},
                        )
        except asyncio.CancelledError:
            pass

    async def _family_rebalance_loop(self, interval_s: float = 60.0) -> None:
        # Periodically rebalance family weights based on performance EMA.
        try:
            while True:
                await asyncio.sleep(interval_s)
                if self._family_weight_manager is None:
                    continue
                rebalanced = self._family_weight_manager.rebalance(time.time())
                if rebalanced:
                    diag = self._family_weight_manager.get_diagnostics()
                    if self._shared is not None:
                        self._shared.family_weights = dict(self._family_weight_manager.weights)
                    log.info(
                        "Family weights rebalanced",
                        data=diag,
                    )
        except asyncio.CancelledError:
            pass

    async def _retire_and_reseed_bot(
        self,
        bot_id: str,
        reason: str,
        is_exploit: bool = True,
        family_hint: str = "",
    ) -> None:
        # Full retire-and-reseed: stop old bot, start fresh bot with new params/run_id.
        #
        # Args:
        # family_hint: When non-empty, scope survivor pool and param domains
        # to this family (Phase 2 per-family reseed).
        #
        # Guarantees:
        # - Old bot's params are never mutated.
        # - New bot gets a new run_id, incrementing generation, and lineage link.
        # - New strategy/scalper/execution instances are hot-registered with the
        # running FarmDispatcher and FarmExecutionCoordinator.
        # 1. Finalize the existing run record.
        record = self._run_records.get(bot_id)
        if record is None:
            return
        if record.reason_ended:  # already retired
            return

        old_ledger = self._equity_ledgers.get(bot_id)
        record.reason_ended = reason
        record.ended_at = time.time()
        if old_ledger:
            record.end_equity = old_ledger.equity
            record.max_drawdown = old_ledger.max_drawdown
        log.info(
            f"Retiring bot {bot_id}: reason={reason}, "
            f"pnl={record.end_equity - record.start_equity:+.2f}, "
            f"gen={record.generation}"
        )

        # 2. Stop the old bot's instances and deregister from dispatcher/coordinator.
        old_strategy = next(
            (s for s in self._strategies if s._cfg.bot_id == bot_id), None
        )
        old_scalper = next(
            (s for s in self._scalpers if s._cfg.bot_id == bot_id), None
        )
        old_execution = next(
            (e for e in self._executions if e._cfg.bot_id == bot_id), None
        )
        if self._dispatcher:
            self._dispatcher.deregister_bot(bot_id)
        if self._execution_coordinator:
            self._execution_coordinator.deregister_execution(bot_id)
        # Remove from farm-level lists
        self._strategies = [s for s in self._strategies if s._cfg.bot_id != bot_id]
        self._scalpers = [s for s in self._scalpers if s._cfg.bot_id != bot_id]
        self._executions = [e for e in self._executions if e._cfg.bot_id != bot_id]
        self._configs_by_bot_id.pop(bot_id, None)
        # Stop in background (don't lose positions — execution drains pending orders)
        try:
            if old_strategy:
                await old_strategy.stop()
            if old_scalper:
                await old_scalper.stop()
            if old_execution:
                await old_execution.stop()
        except Exception as exc:
            log.warning(f"Error stopping retired bot {bot_id}: {exc}")

        # 3. Generate new config via population manager.
        pm = self._population_manager
        base_cfg = self._base_cfg0
        if base_cfg is None:
            return
        next_generation = max(self._generation, record.generation + 1)
        self._generation = next_generation

        new_params: Dict[str, Any] = {}
        parent_scenario = getattr(self._configs_by_bot_id.get(bot_id, base_cfg), "scenario_profile", "base")
        if is_exploit and pm is not None:
            # Exploit should perturb actual top survivors by robustness score.
            top_configs, parent_run_ids = self._get_top_survivor_pool(
                limit=24,
                exclude_bot_id=bot_id,
                family_hint=family_hint,
            )
            if parent_run_ids:
                # Use first parent scenario as exploitation anchor.
                parent_id = parent_run_ids[0]
                for rid, rec in self._run_records.items():
                    if rec.run_id == parent_id:
                        parent_scenario = rec.scenario
                        break
            parent_cfg = dict(top_configs[0]) if top_configs else {
                k: getattr(base_cfg, k) for k in base_cfg.__dataclass_fields__
            }
            # Tight perturbation for exploit lane, family-scoped bounds.
            new_params = perturb_config_tight(parent_cfg, pm._rng, family=family_hint)
            if top_configs:
                # Reject near-clones by escalating perturbation once.
                if novelty_distance(new_params, top_configs) < 0.03:
                    new_params = perturb_config_tight(new_params, pm._rng, family=family_hint)
            new_params["_run_id"] = BotRunRecord.new_run_id()
            new_params["_parent_run_id"] = (parent_run_ids[0] if parent_run_ids else record.run_id)
            new_params["_generation"] = next_generation
            new_params["_role"] = "exploit"
        elif pm is not None:
            from .simulation import random_explore_config as _rexplore
            new_params = _rexplore(pm._rng, family=family_hint)
            new_params["_role"] = "explore"
            new_params["_parent_run_id"] = ""
            new_params["_generation"] = next_generation
            new_params["_run_id"] = BotRunRecord.new_run_id()

        new_run_id = new_params.pop("_run_id", BotRunRecord.new_run_id())
        parent_run_id = new_params.pop("_parent_run_id", record.run_id)
        _generation = new_params.pop("_generation", self._generation)
        new_params.pop("_role", None)

        # Build a new config from the base config + perturbed overrides.
        new_cfg_dict = {k: getattr(base_cfg, k) for k in base_cfg.__dataclass_fields__}
        new_cfg_dict.update(new_params)
        new_cfg_dict["bot_id"] = bot_id  # reuse same bot_id slot
        new_cfg_dict["bankroll_usd"] = base_cfg.bankroll_usd  # always reset to fresh $5k
        new_cfg_dict["scenario_profile"] = self._choose_replacement_scenario(
            is_exploit=is_exploit,
            parent_scenario=parent_scenario,
        )
        from .config import load_config as _load_cfg
        try:
            new_cfg = _load_cfg(new_cfg_dict)
        except Exception as exc:
            log.warning(f"Failed to build replacement config for {bot_id}: {exc}; reusing base")
            new_cfg = replace(base_cfg, bot_id=bot_id)

        # 4. Start fresh bot instances.
        shared = self._shared
        if shared is None:
            return
        new_strategy = StrategyEngine(new_cfg, self._bus, shared=shared)
        new_scalper = MispricingScalper(new_cfg, self._bus, shared=shared)
        new_execution = ExecutionEngine(
            new_cfg,
            self._bus,
            self._rest,
            db=self._db,
            shared_orderbooks=shared.orderbooks,
            shared_metadata=self._shared_metadata,
        )
        try:
            await new_strategy.start(self._active_market_tickers)
            await new_scalper.start()
            await new_execution.start(subscribe_bus=False)
        except Exception as exc:
            log.warning(f"Error starting replacement bot {bot_id}: {exc}")
            return

        # 5. Hot-register the new instances.
        self._strategies.append(new_strategy)
        self._scalpers.append(new_scalper)
        self._executions.append(new_execution)
        self._configs_by_bot_id[bot_id] = new_cfg
        if self._dispatcher:
            self._dispatcher.register_bot(new_strategy, new_scalper)
        if self._execution_coordinator:
            self._execution_coordinator.register_execution(new_execution, new_strategy, new_scalper)

        # 6. Fresh equity ledger + new run record.
        self._equity_ledgers[bot_id] = BotEquityLedger(start_equity=new_cfg.bankroll_usd)
        self._run_records[bot_id] = BotRunRecord(
            run_id=new_run_id,
            bot_id=bot_id,
            parent_run_id=parent_run_id,
            generation=_generation,
            scenario=getattr(new_cfg, "scenario_profile", "base"),
            start_equity=new_cfg.bankroll_usd,
            started_at=time.time(),
            params_snapshot={k: v for k, v in new_params.items()},
        )
        log.info(
            f"Reseeded bot {bot_id}: run_id={new_run_id}, gen={_generation}, "
            f"role={'exploit' if is_exploit else 'explore'}, "
            f"scenario={getattr(new_cfg, 'scenario_profile', 'base')}"
        )

    def _get_top_survivor_pool(
        self,
        limit: int = 24,
        exclude_bot_id: Optional[str] = None,
        family_hint: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        # Return top surviving configs and their run_ids ranked by robustness.
        ranked: List[Tuple[float, str]] = []
        for bot_id, cfg in self._configs_by_bot_id.items():
            if exclude_bot_id and bot_id == exclude_bot_id:
                continue
            if family_hint and self._bot_family_map.get(bot_id, "") != family_hint:
                continue
            rec = self._run_records.get(bot_id)
            if rec and rec.reason_ended:
                continue
            ledger = self._equity_ledgers.get(bot_id)
            if ledger is None:
                continue
            stats = {
                "pnl": ledger.equity - ledger.start_equity,
                "wins": sum(1 for p in ledger.trade_pnls if p >= 0),
                "losses": sum(1 for p in ledger.trade_pnls if p < 0),
                "trade_count": ledger.trade_count,
                "max_drawdown": ledger.max_drawdown,
                "tail_loss_10pct": ledger.tail_loss(),
            }
            score = sim_robustness_score(stats)
            ranked.append((score, bot_id))

        ranked.sort(reverse=True, key=lambda x: x[0])
        selected = ranked[:max(1, limit)]
        top_configs: List[Dict[str, Any]] = []
        parent_run_ids: List[str] = []
        for _, bot_id in selected:
            cfg = self._configs_by_bot_id.get(bot_id)
            if cfg is None:
                continue
            top_configs.append({k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
            rec = self._run_records.get(bot_id)
            if rec is not None:
                parent_run_ids.append(rec.run_id)

        if not top_configs and self._base_cfg0 is not None:
            top_configs = [{k: getattr(self._base_cfg0, k) for k in self._base_cfg0.__dataclass_fields__}]
        return top_configs, parent_run_ids

    def _choose_replacement_scenario(
        self,
        *,
        is_exploit: bool,
        parent_scenario: str = "base",
    ) -> str:
        # Choose scenario profile for a replacement bot.
        #
        # Exploit bots bias toward their parent's scenario while keeping global
        # population close to target mix. Explore bots are assigned to the most
        # underrepresented scenario.
        valid = ("best", "base", "stress")
        current_counts = {k: 0 for k in valid}
        for cfg in self._configs_by_bot_id.values():
            s = getattr(cfg, "scenario_profile", "base")
            if s in current_counts:
                current_counts[s] += 1
        total = max(1, sum(current_counts.values()))
        target = {
            k: self._scenario_target_weights[k] * total
            for k in valid
        }
        deficits = {k: target[k] - current_counts[k] for k in valid}
        underrepresented = max(valid, key=lambda k: deficits[k])

        if is_exploit and parent_scenario in valid:
            # Keep exploit lineage stable unless parent scenario is already overrepresented.
            if deficits[parent_scenario] >= -1.0:
                return parent_scenario
        return underrepresented


    def get_bot_stats_overlay(self) -> Dict[str, Dict[str, Any]]:
        # Return population manager data for IPC snapshot overlay.
        #
        # The returned dict maps bot_id -> {generation, run_id, parent_run_id, ...}
        # which the IPC aggregator can merge into its bot_stats.
        overlay: Dict[str, Dict[str, Any]] = {}
        for bot_id, record in self._run_records.items():
            ledger = self._equity_ledgers.get(bot_id)
            overlay[bot_id] = {
                "generation": record.generation,
                "run_id": record.run_id,
                "parent_run_id": record.parent_run_id,
                "start_equity": record.start_equity,
                "scenario": record.scenario,
            }
            if ledger:
                overlay[bot_id]["tail_loss_10pct"] = ledger.tail_loss()
                overlay[bot_id]["equity"] = ledger.equity
                overlay[bot_id]["trade_count"] = ledger.trade_count
                overlay[bot_id]["max_drawdown"] = ledger.max_drawdown
                overlay[bot_id]["max_drawdown_pct"] = ledger.max_drawdown_pct
            if record.family_pnl:
                overlay[bot_id]["family_pnl"] = dict(record.family_pnl)
        return overlay

    def get_population_diagnostics(self) -> Dict[str, Any]:
        # Runtime population-manager stats for IPC/dashboard visibility.
        active_runs = sum(1 for r in self._run_records.values() if not r.reason_ended)
        ended_runs = sum(1 for r in self._run_records.values() if r.reason_ended)
        result: Dict[str, Any] = {
            "enabled": self._population_manager is not None,
            "generation": self._generation,
            "epoch_count": self._population_epoch_count,
            "reseed_total": self._reseed_total,
            "reseed_epoch": self._reseed_epoch,
            "reseed_drawdown": self._reseed_drawdown,
            "last_reseed_ts": self._last_reseed_ts,
            "last_epoch_ts": self._last_epoch_ts,
            "min_trades_eval": self._population_min_trades_for_eval,
            "min_trades_drawdown": self._population_min_trades_for_drawdown,
            "epoch_minutes": self._population_epoch_minutes,
            "active_runs": active_runs,
            "ended_runs": ended_runs,
        }
        # Phase 1: regime gate diagnostics.
        if self._regime_gate is not None:
            result["regime_gate"] = self._regime_gate.get_diagnostics()
        else:
            result["regime_gate"] = {"enabled": False}
        # Phase 2: family weight diagnostics.
        if self._family_weight_manager is not None:
            result["family_weights"] = self._family_weight_manager.get_diagnostics()
        else:
            result["family_weights"] = {"enabled": False}
        if self._param_region_engine is not None:
            result["param_region"] = self._param_region_engine.diagnostics()
        else:
            result["param_region"] = {"enabled": False}
        if self._shared is not None:
            result["crowding"] = {
                "keys": len(self._shared.crowding_fills_per_sec_by_side),
                "candidate_events": int(self._shared.crowding_candidate_events),
                "active_events": int(self._shared.crowding_active_events),
                "top_fills_per_sec": [
                    {"key": k, "fills_per_sec": round(v, 3)}
                    for k, v in sorted(
                        self._shared.crowding_fills_per_sec_by_side.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:8]
                ],
            }
        else:
            result["crowding"] = {"keys": 0, "candidate_events": 0, "active_events": 0, "top_fills_per_sec": []}
        return result



def _load_farm_base(farm: Dict[str, Any], settings_dir: Optional[Path] = None) -> Dict[str, Any]:
    # Load base config for farm: from base_path (YAML) merged with farm.base.
    base: Dict[str, Any] = {}
    base_path = farm.get("base_path")
    if base_path:
        path = Path(base_path)
        if settings_dir and not path.is_absolute():
            path = settings_dir / path
        if path.exists():
            with path.open() as f:
                data = yaml.safe_load(f) or {}
            block = data.get("argus_kalshi", {})
            if isinstance(block, list) and block:
                base = block[0] if isinstance(block[0], dict) else {}
            elif isinstance(block, dict):
                base = block
    inline = farm.get("base") or {}
    if isinstance(inline, dict):
        base = {**base, **inline}
    return base


def load_farm_configs(raw: Dict[str, Any], settings_path: Optional[str] = None) -> List[KalshiConfig]:
    # Load one or many configs for farm mode.
    #
    # Accepts:
    # - {"argus_kalshi": {...single config...}}  → one bot
    # - {"argus_kalshi": [{...}, {...}]}         → list of full configs (legacy, e.g. 21k line file)
    # - {"argus_kalshi": {"farm": {...}}}         → compact farm: params generated once, no drift.
    # farm.base_path: path to YAML for shared defaults (argus_kalshi block).
    # farm.base: optional inline overrides.
    # farm.dwarf_names_file: path to file with one bot_id per line (e.g. 468 names).
    # farm.bot_count: optional; if set and no dwarf_names_file, use farm_001, farm_002, ...
    # farm.seed: optional; for future range-based sampling.
    # farm.farm_cycle_offset: optional; rotates candidate/explore pool starting
    # positions between runs. If unset or 0, defaults to the current UTC hour.
    # Same (farm block + dwarf list) → same configs every run. Params stick unless you reset.
    block = raw.get("argus_kalshi", {})
    if isinstance(block, list):
        return [load_config(item) for item in block if isinstance(item, dict)]
    if not isinstance(block, dict):
        return [KalshiConfig()]

    farm = block.get("farm")
    if farm and isinstance(farm, dict):
        # Compact farm: generate configs deterministically.
        settings_dir = Path(settings_path).resolve().parent if settings_path else None
        base = _load_farm_base(farm, settings_dir)
        dwarf_file = farm.get("dwarf_names_file") or "argus_kalshi/dwarf_names.txt"
        # Resolve path: try project root (parent of config dir) then cwd
        candidates = []
        if settings_dir:
            candidates.append(settings_dir.parent / dwarf_file)
        candidates.append(Path(dwarf_file))
        bot_ids = []
        for p in candidates:
            bot_ids = load_dwarf_names(str(p))
            if bot_ids:
                break
        if not bot_ids and farm.get("bot_count"):
            n = min(int(farm["bot_count"]), MAX_BOT_COUNT)
            bot_ids = [f"farm_{i:04d}" for i in range(1, n + 1)]
        if not bot_ids:
            log.warning("Farm block has no dwarf_names_file or bot_count; using single bot farm_001")
            bot_ids = ["farm_001"]
        seed = int(farm.get("seed", 0))
        cycle_offset = int(farm.get("farm_cycle_offset", 0) or 0)
        if cycle_offset == 0:
            cycle_offset = round(time.time() / 3600)
        # If bot_count is set and we have fewer names than needed, expand by reusing names (seeded).
        requested = farm.get("bot_count")
        if requested is not None and bot_ids:
            target_count = min(int(requested), MAX_BOT_COUNT)
            if len(bot_ids) < target_count:
                rng = random.Random(seed)
                bot_ids = [rng.choice(bot_ids) for _ in range(target_count)]
        # Cap total bots so we don't overload the event loop / memory.
        if len(bot_ids) > MAX_BOT_COUNT:
            log.info(f"Capping farm at {MAX_BOT_COUNT} bots (had {len(bot_ids)} names)")
            bot_ids = bot_ids[:MAX_BOT_COUNT]
        winner_zone_path = farm.get("winner_zone_path") or "config/kalshi_bot_performance.json"
        config_dicts = generate_farm_configs(
            base, bot_ids, grid_overrides=None, seed=seed,
            winner_zone_path=winner_zone_path,
            cycle_offset=cycle_offset,
        )
        log.info(
            "Farm config generated from grid (params fixed per bot, no drift)",
            data={"bot_count": len(config_dicts), "seed": seed,
                  "winner_zone_path": winner_zone_path, "cycle_offset": cycle_offset},
        )
        return [load_config(c) for c in config_dicts]

    return [load_config(block)]
