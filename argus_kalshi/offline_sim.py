from __future__ import annotations

import concurrent.futures
import csv
import itertools
import json
import math
import multiprocessing as _mp
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .context_policy import build_context_key, momentum_bucket
from .decision_context import _strike_distance_bucket
from .kalshi_strategy import _hold_family_allowed, _hold_tail_penalty
from .market_selectors import hold_entry_horizon_seconds
from .mispricing_scalper import _scalp_family_allowed
from .paper_model import estimate_kalshi_taker_fee_usd
from .settlement_index import SettlementIndex, SettlementRecord
from .simulation import BotEquityLedger, calculate_robustness_score, get_scenario_profile

TapeRecord = Dict[str, Any]

# Scalp-path sweep dimensions (these actually gate the scalp path on the tape):
#   score_threshold: tape contains signals that passed >= 0.10 live; sweeping higher
#                    filters to only the strongest directional signals.
#   max_spread: tape contains signals that passed spread <= 8; sweeping lower
#               filters to tighter-spread (more liquid) markets only.
#   cost_buffer: shifts the projected-profit acceptance threshold.
GRID_SCALP_SCORE_THRESHOLD = [0.10, 0.12, 0.15, 0.20]
GRID_SCALP_MAX_SPREAD = [4, 6, 8]
GRID_SCALP_COST_BUFFER = [-5, 0, 3, 5]

# Hold-path sweep dimensions:
GRID_MIN_EDGE = [0.08, 0.09, 0.10]
GRID_HOLD_DIVERGENCE = [0.02, 0.04, 0.06, 0.08]

# Fixed (not swept — see notes):
#   persistence_window_ms: signals in tape already passed live persistence filter → no-op
#   scalp_min_edge_cents: tape signals all have score>=0.10 → edge_cents>=10, above any [5-8] value
#   scalp_min_profit_cents: with cost_buffer=-5 and typical spread, projected never blocks values [4-7]
#   min_entry/max_entry_cents: only affect hold path; hold signals rare (no_divergence dominant)
GRID_PERSISTENCE_MS = [180]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _norm_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    return side if side in {"yes", "no"} else ""


@dataclass(frozen=True, slots=True)
class SimParams:
    # --- swept dimensions (varied by build_param_grid) ---
    min_entry_cents: int = 30
    max_entry_cents: int = 72
    min_edge_threshold: float = 0.09
    persistence_window_ms: int = 180
    scalp_min_edge_cents: int = 6
    scalp_min_profit_cents: int = 5
    # --- base values: must match kalshi_family_adaptive.yaml ---
    effective_edge_fee_pct: float = 0.02
    yes_avoid_min_cents: int = 80   # live: 80 (only block extreme tail YES)
    yes_avoid_max_cents: int = 100  # live: 100
    no_avoid_above_cents: int = 0   # live: disabled
    max_entry_minutes_to_expiry: int = 20
    range_max_entry_minutes_to_expiry: int = 8
    hold_min_divergence_threshold: float = 0.04  # live: hold_min_divergence_threshold
    hold_tail_penalty_start_cents: int = 55
    hold_tail_penalty_per_10c: float = 0.015
    hold_min_net_edge_cents: int = 6   # live: hold_min_net_edge_cents
    hold_entry_cost_buffer_cents: int = 3  # live: hold_entry_cost_buffer_cents
    scalp_max_spread_cents: int = 8    # live: 8
    scalp_entry_cost_buffer_cents: int = -3  # live: -3
    scalp_directional_score_threshold: float = 0.12  # live: 0.12
    scalp_min_entry_cents: int = 20    # live: 20
    scalp_max_entry_cents: int = 80    # live: 80
    bankroll_usd: float = 5000.0
    risk_fraction_per_trade: float = 0.005  # live: sizing_risk_fraction
    max_contracts_per_ticker: int = 50  # live: 50


@dataclass(slots=True)
class SimResult:
    params: SimParams
    total_pnl_usd: float = 0.0
    total_fees_usd: float = 0.0
    fills_count: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_pnl_per_fill: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    per_context_pnl: Dict[str, float] = field(default_factory=dict)
    per_family_pnl: Dict[str, float] = field(default_factory=dict)
    per_source_pnl: Dict[str, float] = field(default_factory=dict)
    rejected_count: int = 0
    no_outcome_count: int = 0
    tape_records_evaluated: int = 0
    run_duration_s: float = 0.0
    robustness_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["params"] = asdict(self.params)
        return payload


@dataclass(slots=True)
class CrossValFold:
    fold_id: int
    train_run_ids: List[str]
    test_run_id: str
    best_params: SimParams
    train_result: SimResult
    test_result: SimResult
    overfit_pnl_ratio: float
    overfit_sharpe_ratio: float


@dataclass(slots=True)
class StrategyDiffResult:
    tape_path: str
    params_old: SimParams
    params_new: SimParams
    result_old: SimResult
    result_new: SimResult
    delta_pnl_usd: float
    delta_win_rate: float
    delta_sharpe: float
    delta_max_drawdown_usd: float
    delta_fills_count: int
    delta_avg_pnl_per_fill: float
    context_deltas: Dict[str, Tuple[float, float, float]]
    improved_contexts: List[str]
    degraded_contexts: List[str]
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tape_path": self.tape_path,
            "params_old": asdict(self.params_old),
            "params_new": asdict(self.params_new),
            "result_old": self.result_old.to_dict(),
            "result_new": self.result_new.to_dict(),
            "delta_pnl_usd": self.delta_pnl_usd,
            "delta_win_rate": self.delta_win_rate,
            "delta_sharpe": self.delta_sharpe,
            "delta_max_drawdown_usd": self.delta_max_drawdown_usd,
            "delta_fills_count": self.delta_fills_count,
            "delta_avg_pnl_per_fill": self.delta_avg_pnl_per_fill,
            "context_deltas": {
                key: [values[0], values[1], values[2]] for key, values in self.context_deltas.items()
            },
            "improved_contexts": self.improved_contexts,
            "degraded_contexts": self.degraded_contexts,
            "verdict": self.verdict,
        }


def params_from_kalshi_config(cfg: Any) -> SimParams:
    getter = cfg.get if isinstance(cfg, dict) else lambda key, default=None: getattr(cfg, key, default)
    return SimParams(
        min_entry_cents=_as_int(getter("min_entry_cents", 30), 30),
        max_entry_cents=_as_int(getter("max_entry_cents", 75), 75),
        min_edge_threshold=_as_float(getter("min_edge_threshold", 0.09), 0.09),
        effective_edge_fee_pct=_as_float(getter("effective_edge_fee_pct", 0.02), 0.02),
        persistence_window_ms=_as_int(getter("persistence_window_ms", 120), 120),
        yes_avoid_min_cents=_as_int(getter("yes_avoid_min_cents", 0), 0),
        yes_avoid_max_cents=_as_int(getter("yes_avoid_max_cents", 0), 0),
        no_avoid_above_cents=_as_int(getter("no_avoid_above_cents", 0), 0),
        max_entry_minutes_to_expiry=_as_int(getter("max_entry_minutes_to_expiry", 20), 20),
        range_max_entry_minutes_to_expiry=_as_int(getter("range_max_entry_minutes_to_expiry", 60), 60),
        hold_min_divergence_threshold=_as_float(getter("hold_min_divergence_threshold", 0.0), 0.0),
        hold_tail_penalty_start_cents=_as_int(getter("hold_tail_penalty_start_cents", 55), 55),
        hold_tail_penalty_per_10c=_as_float(getter("hold_tail_penalty_per_10c", 0.015), 0.015),
        hold_min_net_edge_cents=_as_int(getter("hold_min_net_edge_cents", 1), 1),
        hold_entry_cost_buffer_cents=_as_int(getter("hold_entry_cost_buffer_cents", 1), 1),
        scalp_min_edge_cents=_as_int(getter("scalp_min_edge_cents", 6), 6),
        scalp_min_profit_cents=_as_int(getter("scalp_min_profit_cents", 5), 5),
        scalp_max_spread_cents=_as_int(getter("scalp_max_spread_cents", 2), 2),
        scalp_entry_cost_buffer_cents=_as_int(getter("scalp_entry_cost_buffer_cents", 4), 4),
        scalp_directional_score_threshold=_as_float(getter("scalp_directional_score_threshold", 0.15), 0.15),
        scalp_min_entry_cents=_as_int(getter("scalp_min_entry_cents", 30), 30),
        scalp_max_entry_cents=_as_int(getter("scalp_max_entry_cents", 65), 65),
        bankroll_usd=_as_float(getter("bankroll_usd", 5000.0), 5000.0),
        risk_fraction_per_trade=_as_float(
            getter("risk_fraction_per_trade", getter("sizing_risk_fraction", 0.001)),
            0.001,
        ),
        max_contracts_per_ticker=_as_int(getter("max_contracts_per_ticker", 3), 3),
    )


def load_tape(path: str, include_rejections: bool = False) -> List[TapeRecord]:
    records: List[TapeRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            decision = str(obj.get("decision") or "")
            if decision == "signal" or (include_rejections and decision == "rejected"):
                records.append(obj)
    records.sort(key=lambda row: (_as_float(row.get("ts")), str(row.get("ticker") or "")))
    return records


def load_tapes(paths: Sequence[str], include_rejections: bool = False) -> List[TapeRecord]:
    combined: List[TapeRecord] = []
    for path in paths:
        combined.extend(load_tape(path, include_rejections=include_rejections))
    combined.sort(key=lambda row: (_as_float(row.get("ts")), str(row.get("ticker") or "")))
    return combined


def _record_ctx_key(record: TapeRecord, side: str, edge: float, limit_cents: int) -> str:
    existing = str(record.get("ctx_key") or "").strip()
    if existing:
        return existing
    family = str(record.get("family") or "Other")
    strike_distance_pct = record.get("strike_distance_pct")
    strike_bucket = _strike_distance_bucket(strike_distance_pct, [0.005, 0.01, 0.02, 0.04])
    near_money = strike_distance_pct is not None and _as_float(strike_distance_pct) <= 0.08
    price_bucket = "lt_40"
    if limit_cents >= 78:
        price_bucket = "ge_78"
    elif limit_cents >= 70:
        price_bucket = "70_78"
    elif limit_cents >= 55:
        price_bucket = "55_70"
    elif limit_cents >= 40:
        price_bucket = "40_55"
    edge_bucket = "lt_0.05"
    if edge >= 0.20:
        edge_bucket = "ge_0.20"
    elif edge >= 0.10:
        edge_bucket = "0.10_0.20"
    elif edge >= 0.05:
        edge_bucket = "0.05_0.10"
    return build_context_key(
        family=family,
        side=side,
        edge_bucket=edge_bucket,
        price_bucket=price_bucket,
        strike_distance_bucket=strike_bucket,
        near_money=bool(near_money),
        momentum=momentum_bucket(_as_float(record.get("momentum_drift"))),
    )


def _hold_gate_check(params: SimParams, record: TapeRecord) -> Tuple[bool, str, str, int, float]:
    asset = str(record.get("asset") or "")
    window_minutes = _as_int(record.get("window_minutes"), 0)
    is_range = bool(record.get("is_range", False))
    if not _hold_family_allowed(asset, window_minutes, is_range):
        return False, "hold_family_filtered", "", 0, 0.0
    time_to_settle_s = _as_float(record.get("time_to_settle_s"))
    max_entry_s = hold_entry_horizon_seconds(
        window_minutes=window_minutes,
        is_range=is_range,
        max_entry_minutes_to_expiry=params.max_entry_minutes_to_expiry,
        range_max_entry_minutes_to_expiry=params.range_max_entry_minutes_to_expiry,
    )
    if not is_range and max_entry_s > 0 and time_to_settle_s > max_entry_s:
        return False, "too_early_to_expiry", "", 0, 0.0
    if is_range and max_entry_s > 0 and time_to_settle_s > max_entry_s:
        return False, "range_too_far_to_expiry", "", 0, 0.0
    p_yes = _as_float(record.get("p_yes"))
    yes_ask = _as_int(record.get("yes_ask_cents"))
    no_ask = _as_int(record.get("no_ask_cents"))
    divergence = p_yes - (yes_ask / 100.0)
    forced_side = "yes" if divergence > 0 else "no"
    if abs(divergence) < params.hold_min_divergence_threshold:
        return False, "no_divergence", forced_side, 0, 0.0
    ev_yes = divergence
    ev_no = -divergence
    effective_yes = ev_yes - params.effective_edge_fee_pct
    effective_no = ev_no - params.effective_edge_fee_pct
    penalized_yes = effective_yes - _hold_tail_penalty(
        yes_ask,
        "yes",
        params.hold_tail_penalty_start_cents,
        params.hold_tail_penalty_per_10c,
    )
    penalized_no = effective_no - _hold_tail_penalty(
        no_ask,
        "no",
        params.hold_tail_penalty_start_cents,
        params.hold_tail_penalty_per_10c,
    )
    side = ""
    edge = 0.0
    limit_cents = 0
    if forced_side == "yes" and penalized_yes >= params.min_edge_threshold:
        side = "yes"
        edge = penalized_yes
        limit_cents = yes_ask
    elif forced_side == "no" and penalized_no >= params.min_edge_threshold:
        side = "no"
        edge = penalized_no
        limit_cents = no_ask
    if not side:
        return False, "no_edge", forced_side, 0, 0.0
    if (params.min_entry_cents > 0 and limit_cents < params.min_entry_cents) or (
        params.max_entry_cents < 100 and limit_cents > params.max_entry_cents
    ):
        return False, "entry_price_out_of_range", side, limit_cents, edge
    if side == "no" and params.no_avoid_above_cents > 0 and limit_cents >= params.no_avoid_above_cents:
        return False, "no_tail_filter", side, limit_cents, edge
    if (
        side == "yes"
        and params.yes_avoid_min_cents > 0
        and params.yes_avoid_max_cents > 0
        and params.yes_avoid_min_cents <= limit_cents <= params.yes_avoid_max_cents
    ):
        return False, "yes_mid_range_filter", side, limit_cents, edge
    fee_cents = int(round(estimate_kalshi_taker_fee_usd(limit_cents, 100) * 100.0))
    bid_cents = _as_int(record.get("yes_bid_cents")) if side == "yes" else _as_int(record.get("no_bid_cents"))
    side_spread_cents = max(0, limit_cents - bid_cents)
    entry_slippage_cents = max(0, int(round(side_spread_cents * 0.5)))
    net_edge_cents = int(round(edge * 100.0)) - fee_cents - entry_slippage_cents
    required = params.hold_min_net_edge_cents + params.hold_entry_cost_buffer_cents
    if net_edge_cents < required:
        return False, "hold_net_edge_too_low", side, limit_cents, edge
    return True, "", side, limit_cents, edge


def _scalp_directional_score(record: TapeRecord) -> float:
    drift_scale = 0.0002
    drift_component = max(-1.0, min(1.0, _as_float(record.get("momentum_drift")) / drift_scale))
    score = (
        0.30 * drift_component
        + 0.15 * _as_float(record.get("obi"))
        + 0.10 * _as_float(record.get("trade_flow"))
        + 0.25 * _as_float(record.get("depth_pressure"))
        + 0.10 * _as_float(record.get("delta_flow_yes"))
        - 0.10 * _as_float(record.get("delta_flow_no"))
    )
    return max(-1.0, min(1.0, score))


def _scalp_gate_check(params: SimParams, record: TapeRecord) -> Tuple[bool, str, str, int, float]:
    asset = str(record.get("asset") or "")
    window_minutes = _as_int(record.get("window_minutes"), 0)
    is_range = bool(record.get("is_range", False))
    if not _scalp_family_allowed(asset, window_minutes, is_range):
        return False, "scalp_family_filtered", "", 0, 0.0
    spread = max(
        0,
        (_as_int(record.get("yes_ask_cents")) - _as_int(record.get("yes_bid_cents")))
        + (_as_int(record.get("no_ask_cents")) - _as_int(record.get("no_bid_cents"))),
    )
    if spread > params.scalp_max_spread_cents:
        return False, "scalp_spread_too_wide", "", 0, 0.0
    directional_score = _scalp_directional_score(record)
    if abs(directional_score) < params.scalp_directional_score_threshold:
        return False, "scalp_directional_below_threshold", "", 0, 0.0
    if int(round(abs(directional_score) * 100.0)) < params.scalp_min_edge_cents:
        return False, "scalp_edge_too_low", "", 0, 0.0
    side = "yes" if directional_score > 0 else "no"
    limit_cents = _as_int(record.get("yes_ask_cents")) if side == "yes" else _as_int(record.get("no_ask_cents"))
    if params.scalp_min_entry_cents > 0 and limit_cents < params.scalp_min_entry_cents:
        return False, "scalp_entry_price_out_of_range", side, limit_cents, abs(directional_score)
    if params.scalp_max_entry_cents < 100 and limit_cents > params.scalp_max_entry_cents:
        return False, "scalp_entry_price_out_of_range", side, limit_cents, abs(directional_score)
    target_exit_cents = min(99, limit_cents + params.scalp_min_profit_cents)
    slippage_buffer = max(1, int(round(spread * 0.5)))
    round_trip_fee_cents = int(
        round(
            (
                estimate_kalshi_taker_fee_usd(limit_cents, 100)
                + estimate_kalshi_taker_fee_usd(target_exit_cents, 100)
            )
            * 100.0
        )
    )
    projected_net_profit = params.scalp_min_profit_cents - slippage_buffer - round_trip_fee_cents
    if projected_net_profit < params.scalp_entry_cost_buffer_cents:
        return False, "scalp_projected_profit_too_low", side, limit_cents, abs(directional_score)
    return True, "", side, limit_cents, abs(directional_score)


def _trade_quantity(record: TapeRecord, params: SimParams, limit_cents: int) -> int:
    explicit = _as_int(record.get("quantity_contracts"), 0)
    if explicit > 0:
        return explicit
    if limit_cents <= 0:
        return 1
    risk_usd = max(0.0, params.bankroll_usd * params.risk_fraction_per_trade)
    qty = max(1, int(risk_usd / max(limit_cents / 100.0, 0.01)))
    if params.max_contracts_per_ticker > 0:
        qty = min(qty, params.max_contracts_per_ticker)
    return max(1, qty)


def compute_sharpe(pnls: List[float], periods_per_year: float = 365 * 24) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = statistics.fmean(pnls)
    stdev = statistics.pstdev(pnls)
    if stdev <= 1e-12:
        return 0.0
    return (mean / stdev) * math.sqrt(periods_per_year)


def simulate(
    tape_records: List[TapeRecord],
    settlement_index: SettlementIndex,
    params: SimParams,
    include_rejections: bool = False,
    scenario: str = "base",
    random_seed: Optional[int] = 42,
) -> SimResult:
    del random_seed
    started = time.perf_counter()
    profile = get_scenario_profile(scenario)
    result = SimResult(params=params)
    ledger = BotEquityLedger(start_equity=params.bankroll_usd)
    trade_pnls: List[float] = []

    for record in tape_records:
        decision = str(record.get("decision") or "")
        if decision == "rejected" and not include_rejections:
            continue
        result.tape_records_evaluated += 1
        source = str(record.get("source") or "")
        if source in {"mispricing_hold", "pair_arb"}:
            would_take, _reason, side, limit_cents, edge = _hold_gate_check(params, record)
        elif "scalp" in source:
            would_take, _reason, side, limit_cents, edge = _scalp_gate_check(params, record)
        else:
            result.rejected_count += 1
            continue
        if not would_take:
            if decision == "signal" or include_rejections:
                result.rejected_count += 1
            continue
        qty = _trade_quantity(record, params, limit_cents)
        if "scalp" in source:
            rows = settlement_index.get_records(str(record.get("ticker") or ""), side)
            if not rows:
                result.no_outcome_count += 1
                continue
            ref = rows[0]
            gross_pnl, fee_usd = settlement_index.get_scalp_pnl(limit_cents, ref.exit_price_cents, qty, source=source)
        else:
            pnl_tuple = settlement_index.get_hold_pnl(str(record.get("ticker") or ""), side, limit_cents, qty)
            if pnl_tuple is None:
                result.no_outcome_count += 1
                continue
            gross_pnl, fee_usd = pnl_tuple
        fee_usd *= profile.fee_multiplier
        slippage_usd = qty * profile.slippage_cents * 0.01
        spread_drag_usd = qty * profile.spread_drag_per_contract
        net = ledger.record_trade(
            gross_pnl,
            fee_usd=fee_usd,
            slippage_usd=slippage_usd,
            spread_drag_usd=spread_drag_usd,
        )
        trade_pnls.append(net)
        ctx_key = _record_ctx_key(record, side, edge, limit_cents)
        family = str(record.get("family") or "Other")
        result.per_context_pnl[ctx_key] = result.per_context_pnl.get(ctx_key, 0.0) + net
        result.per_family_pnl[family] = result.per_family_pnl.get(family, 0.0) + net
        result.per_source_pnl[source] = result.per_source_pnl.get(source, 0.0) + net
        result.fills_count += 1
        result.total_pnl_usd += net
        result.total_fees_usd += fee_usd
        if net >= 0:
            result.wins += 1
        else:
            result.losses += 1

    result.win_rate = result.wins / result.fills_count if result.fills_count else 0.0
    result.avg_pnl_per_fill = result.total_pnl_usd / result.fills_count if result.fills_count else 0.0
    result.sharpe_ratio = compute_sharpe(trade_pnls)
    result.max_drawdown_usd = ledger.max_drawdown
    result.max_drawdown_pct = ledger.max_drawdown_pct
    result.robustness_score = calculate_robustness_score(
        {
            "pnl": result.total_pnl_usd,
            "wins": result.wins,
            "losses": result.losses,
            "trade_count": result.fills_count,
            "max_drawdown": result.max_drawdown_usd,
            "tail_loss_10pct": ledger.tail_loss(),
        }
    )
    result.run_duration_s = time.perf_counter() - started
    return result


def build_param_grid(
    scalp_score_threshold: List[float] = GRID_SCALP_SCORE_THRESHOLD,
    scalp_max_spread: List[int] = GRID_SCALP_MAX_SPREAD,
    scalp_cost_buffer: List[int] = GRID_SCALP_COST_BUFFER,
    min_edge: List[float] = GRID_MIN_EDGE,
    hold_divergence: List[float] = GRID_HOLD_DIVERGENCE,
) -> List[SimParams]:
    grid: List[SimParams] = []
    for values in itertools.product(scalp_score_threshold, scalp_max_spread, scalp_cost_buffer, min_edge, hold_divergence):
        grid.append(SimParams(
            scalp_directional_score_threshold=float(values[0]),
            scalp_max_spread_cents=int(values[1]),
            scalp_entry_cost_buffer_cents=int(values[2]),
            min_edge_threshold=float(values[3]),
            hold_min_divergence_threshold=float(values[4]),
        ))
    unique: Dict[Tuple[Any, ...], SimParams] = {}
    for params in grid:
        unique[tuple(asdict(params).items())] = params
    return list(unique.values())


# ---------------------------------------------------------------------------
# ProcessPoolExecutor worker state (module-level globals set by initializer)
# ---------------------------------------------------------------------------
_WORKER_TAPE: List[TapeRecord] = []
_WORKER_SETTLEMENT: Optional[SettlementIndex] = None


def _serialize_settlement(idx: SettlementIndex) -> Dict[str, Any]:
    """Minimal JSON-serializable form of SettlementIndex for pickling to workers."""
    binary: Dict[str, Optional[bool]] = dict(idx._binary_outcomes)
    records: Dict[str, List[int]] = {}
    for (ticker, side), rows in idx._records.items():
        key = f"{ticker}\x00{side}"
        records[key] = [r.exit_price_cents for r in rows]
    return {"binary": binary, "records": records}


def _deserialize_settlement(data: Dict[str, Any]) -> SettlementIndex:
    rebuilt = SettlementIndex()
    rebuilt._binary_outcomes = data["binary"]
    for key_str, exit_prices in data["records"].items():
        ticker, side = key_str.split("\x00", 1)
        won = (side == "yes")  # exit_price determines win at settlement scoring time
        for ep in exit_prices:
            rebuilt._records[(ticker, side)].append(
                SettlementRecord(
                    ticker=ticker,
                    side=side,
                    won=(ep > 50),
                    entry_price_cents=0,
                    exit_price_cents=ep,
                    pnl_usd=0.0,
                    fees_usd=0.0,
                    settlement_method="inferred",
                    timestamp=0.0,
                    source="index",
                )
            )
    return rebuilt


def _process_worker_init(tape_paths: List[str], settlement_data: Dict[str, Any], include_rejections: bool) -> None:
    """Runs once per worker process to load the tape into a module-level global."""
    global _WORKER_TAPE, _WORKER_SETTLEMENT
    _WORKER_TAPE = load_tapes(tape_paths, include_rejections=include_rejections)
    _WORKER_SETTLEMENT = _deserialize_settlement(settlement_data)


def _simulate_process_worker(args: Tuple) -> SimResult:
    params, include_rejections, scenario = args
    try:
        return simulate(_WORKER_TAPE, _WORKER_SETTLEMENT, params, include_rejections, scenario)
    except Exception:
        return SimResult(params=params, total_pnl_usd=float("nan"))


def evaluate_grid(
    tape_paths: List[str],
    settlement_index: SettlementIndex,
    param_grid: List[SimParams],
    n_workers: Optional[int] = None,
    scenario: str = "base",
    include_rejections: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[SimResult]:
    total = len(param_grid)
    # Default: 4 workers (each loads its own copy of the tape; RAM = 4 × tape size).
    # Pass n_workers=1 to disable multiprocessing for tests or low-RAM environments.
    workers = n_workers if n_workers is not None else min(4, max(1, os.cpu_count() or 1))

    if workers <= 1:
        tape_records = load_tapes(tape_paths, include_rejections=include_rejections)
        results: List[SimResult] = []
        for i, p in enumerate(param_grid):
            results.append(simulate(tape_records, settlement_index, p, include_rejections, scenario))
            if progress_callback is not None:
                progress_callback(i + 1, total)
        return results

    settlement_data = _serialize_settlement(settlement_index)
    work = [(p, include_rejections, scenario) for p in param_grid]
    ordered: List[SimResult] = [None] * total  # type: ignore[list-item]
    ctx = _mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_process_worker_init,
        initargs=(tape_paths, settlement_data, include_rejections),
    ) as pool:
        futures = {pool.submit(_simulate_process_worker, item): idx for idx, item in enumerate(work)}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            ordered[idx] = future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)
    return ordered


def rank_results(results: List[SimResult], by: str = "robustness_score", min_fills: int = 10) -> List[Dict[str, Any]]:
    metric_name = {
        "total_pnl": "total_pnl_usd",
        "sharpe": "sharpe_ratio",
        "win_rate": "win_rate",
        "robustness_score": "robustness_score",
    }.get(by, by)
    filtered = [result for result in results if result.fills_count >= min_fills and not math.isnan(result.total_pnl_usd)]
    filtered.sort(key=lambda item: getattr(item, metric_name, float("-inf")), reverse=True)
    total = max(1, len(filtered))
    ranked: List[Dict[str, Any]] = []
    for idx, result in enumerate(filtered, start=1):
        row = result.to_dict()
        row["rank"] = idx
        row["percentile"] = 1.0 - ((idx - 1) / total)
        ranked.append(row)
    return ranked


def save_ranked_results(ranked: List[Dict[str, Any]], output_dir: str, tag: str = "") -> Tuple[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    suffix = f"_{tag}" if tag else ""
    json_path = out_dir / f"grid_ranked_{stamp}{suffix}.json"
    csv_path = out_dir / f"grid_ranked_{stamp}{suffix}.csv"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(ranked, fh, indent=2)
    fieldnames = [
        "rank",
        "total_pnl_usd",
        "robustness_score",
        "fills_count",
        "wins",
        "losses",
        "win_rate",
        "avg_pnl_per_fill",
        "sharpe_ratio",
        "max_drawdown_usd",
        "max_drawdown_pct",
        "min_entry_cents",
        "max_entry_cents",
        "min_edge_threshold",
        "persistence_window_ms",
        "scalp_min_edge_cents",
        "scalp_min_profit_cents",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            params = row.get("params", {})
            writer.writerow(
                {
                    "rank": row.get("rank"),
                    "total_pnl_usd": row.get("total_pnl_usd"),
                    "robustness_score": row.get("robustness_score"),
                    "fills_count": row.get("fills_count"),
                    "wins": row.get("wins"),
                    "losses": row.get("losses"),
                    "win_rate": row.get("win_rate"),
                    "avg_pnl_per_fill": row.get("avg_pnl_per_fill"),
                    "sharpe_ratio": row.get("sharpe_ratio"),
                    "max_drawdown_usd": row.get("max_drawdown_usd"),
                    "max_drawdown_pct": row.get("max_drawdown_pct"),
                    "min_entry_cents": params.get("min_entry_cents"),
                    "max_entry_cents": params.get("max_entry_cents"),
                    "min_edge_threshold": params.get("min_edge_threshold"),
                    "persistence_window_ms": params.get("persistence_window_ms"),
                    "scalp_min_edge_cents": params.get("scalp_min_edge_cents"),
                    "scalp_min_profit_cents": params.get("scalp_min_profit_cents"),
                }
            )
    return str(json_path), str(csv_path)


def cross_validate(
    tape_paths: List[str],
    settlement_index: SettlementIndex,
    param_grid: List[SimParams],
    n_workers: Optional[int] = None,
    scenario: str = "base",
    min_fills_train: int = 10,
    ranking_metric: str = "robustness_score",
) -> List[CrossValFold]:
    folds: List[CrossValFold] = []
    if not tape_paths:
        return folds
    if len(tape_paths) == 1:
        all_records = load_tape(tape_paths[0], include_rejections=False)
        split = max(1, len(all_records) // 2)
        train_records = all_records[:split]
        test_records = all_records[split:]
        ranked = rank_results(
            [simulate(train_records, settlement_index, params, scenario=scenario) for params in param_grid],
            by=ranking_metric,
            min_fills=min_fills_train,
        )
        if not ranked:
            return folds
        best_params = SimParams(**ranked[0]["params"])
        train_result = simulate(train_records, settlement_index, best_params, scenario=scenario)
        test_result = simulate(test_records, settlement_index, best_params, scenario=scenario)
        folds.append(
            CrossValFold(
                fold_id=0,
                train_run_ids=["train_half"],
                test_run_id="test_half",
                best_params=best_params,
                train_result=train_result,
                test_result=test_result,
                overfit_pnl_ratio=(test_result.total_pnl_usd / train_result.total_pnl_usd) if train_result.total_pnl_usd else 0.0,
                overfit_sharpe_ratio=(test_result.sharpe_ratio / train_result.sharpe_ratio) if train_result.sharpe_ratio else 0.0,
            )
        )
        return folds
    for idx, test_path in enumerate(tape_paths):
        train_paths = [path for j, path in enumerate(tape_paths) if j != idx]
        train_results = evaluate_grid(
            tape_paths=train_paths,
            settlement_index=settlement_index,
            param_grid=param_grid,
            n_workers=n_workers,
            scenario=scenario,
        )
        ranked = rank_results(train_results, by=ranking_metric, min_fills=min_fills_train)
        if not ranked:
            continue
        best_params = SimParams(**ranked[0]["params"])
        train_result = simulate(load_tapes(train_paths), settlement_index, best_params, scenario=scenario)
        test_result = simulate(load_tape(test_path), settlement_index, best_params, scenario=scenario)
        folds.append(
            CrossValFold(
                fold_id=idx,
                train_run_ids=[Path(path).stem for path in train_paths],
                test_run_id=Path(test_path).stem,
                best_params=best_params,
                train_result=train_result,
                test_result=test_result,
                overfit_pnl_ratio=(test_result.total_pnl_usd / train_result.total_pnl_usd) if train_result.total_pnl_usd else 0.0,
                overfit_sharpe_ratio=(test_result.sharpe_ratio / train_result.sharpe_ratio) if train_result.sharpe_ratio else 0.0,
            )
        )
    return folds


def summarize_cross_validation(folds: List[CrossValFold]) -> Dict[str, Any]:
    if not folds:
        return {
            "n_folds": 0,
            "avg_train_pnl": 0.0,
            "avg_test_pnl": 0.0,
            "avg_overfit_ratio": 0.0,
            "overfit_verdict": "inconclusive",
            "best_params_across_folds": None,
            "per_fold": [],
        }
    avg_train = statistics.fmean(fold.train_result.total_pnl_usd for fold in folds)
    avg_test = statistics.fmean(fold.test_result.total_pnl_usd for fold in folds)
    avg_ratio = statistics.fmean(fold.overfit_pnl_ratio for fold in folds)
    if avg_ratio >= 0.70:
        verdict = "clean"
    elif avg_ratio >= 0.40:
        verdict = "mild"
    else:
        verdict = "severe"
    counter: Dict[Tuple[Any, ...], int] = {}
    params_by_key: Dict[Tuple[Any, ...], SimParams] = {}
    for fold in folds:
        key = tuple(asdict(fold.best_params).items())
        counter[key] = counter.get(key, 0) + 1
        params_by_key[key] = fold.best_params
    best_key = max(counter, key=counter.get)
    return {
        "n_folds": len(folds),
        "avg_train_pnl": avg_train,
        "avg_test_pnl": avg_test,
        "avg_overfit_ratio": avg_ratio,
        "overfit_verdict": verdict,
        "best_params_across_folds": asdict(params_by_key[best_key]),
        "per_fold": [
            {
                "fold_id": fold.fold_id,
                "train_run_ids": fold.train_run_ids,
                "test_run_id": fold.test_run_id,
                "best_params": asdict(fold.best_params),
                "train_pnl": fold.train_result.total_pnl_usd,
                "test_pnl": fold.test_result.total_pnl_usd,
                "overfit_pnl_ratio": fold.overfit_pnl_ratio,
                "overfit_sharpe_ratio": fold.overfit_sharpe_ratio,
            }
            for fold in folds
        ],
    }


def diff_strategies(
    tape_paths: List[str],
    settlement_index: SettlementIndex,
    params_old: SimParams,
    params_new: SimParams,
    scenario: str = "base",
    min_fills_for_verdict: int = 20,
) -> StrategyDiffResult:
    records = load_tapes(tape_paths)
    result_old = simulate(records, settlement_index, params_old, scenario=scenario)
    result_new = simulate(records, settlement_index, params_new, scenario=scenario)
    context_keys = sorted(set(result_old.per_context_pnl) | set(result_new.per_context_pnl))
    context_deltas: Dict[str, Tuple[float, float, float]] = {}
    improved: List[str] = []
    degraded: List[str] = []
    for key in context_keys:
        old_pnl = result_old.per_context_pnl.get(key, 0.0)
        new_pnl = result_new.per_context_pnl.get(key, 0.0)
        delta = new_pnl - old_pnl
        context_deltas[key] = (old_pnl, new_pnl, delta)
        if delta > 0:
            improved.append(key)
        elif delta < 0:
            degraded.append(key)
    if min(result_old.fills_count, result_new.fills_count) < min_fills_for_verdict:
        verdict = "inconclusive"
    elif result_new.total_pnl_usd > result_old.total_pnl_usd + 1e-9:
        verdict = "improved"
    elif result_new.total_pnl_usd < result_old.total_pnl_usd - 1e-9:
        verdict = "degraded"
    else:
        verdict = "neutral"
    return StrategyDiffResult(
        tape_path=",".join(tape_paths),
        params_old=params_old,
        params_new=params_new,
        result_old=result_old,
        result_new=result_new,
        delta_pnl_usd=result_new.total_pnl_usd - result_old.total_pnl_usd,
        delta_win_rate=result_new.win_rate - result_old.win_rate,
        delta_sharpe=result_new.sharpe_ratio - result_old.sharpe_ratio,
        delta_max_drawdown_usd=result_new.max_drawdown_usd - result_old.max_drawdown_usd,
        delta_fills_count=result_new.fills_count - result_old.fills_count,
        delta_avg_pnl_per_fill=result_new.avg_pnl_per_fill - result_old.avg_pnl_per_fill,
        context_deltas=context_deltas,
        improved_contexts=improved,
        degraded_contexts=degraded,
        verdict=verdict,
    )
