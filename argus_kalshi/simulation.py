# Created by Oliver Meihls

# Simulation fidelity, robustness scoring, and population management.
#
# This module provides:
# - ScenarioProfile: execution scenario presets (best/base/stress)
# - BotEquityLedger: isolated $5,000 bankroll tracking per bot
# - BotRunRecord: lineage metadata for retire/reseed lifecycle
# - PopulationManager: exploit/explore epoch-based parameter search
# - calculate_robustness_score / project_execution_scenarios: scoring helpers

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


#  Scenario profiles — configurable execution assumptions

@dataclass(frozen=True, slots=True)
class ScenarioProfile:
    # Execution assumption profile applied to paper fills.
    name: str
    latency_min_ms: int
    latency_max_ms: int
    slippage_cents: int
    fee_multiplier: float       # 1.0 = standard Kalshi fees
    spread_drag_per_contract: float  # USD drag per contract for spread cost


# Pre-built profiles

SCENARIO_BEST = ScenarioProfile(
    name="best",
    latency_min_ms=10,
    latency_max_ms=30,
    slippage_cents=0,
    fee_multiplier=1.0,
    spread_drag_per_contract=0.001,
)

SCENARIO_BASE = ScenarioProfile(
    name="base",
    latency_min_ms=30,
    latency_max_ms=80,
    slippage_cents=1,
    fee_multiplier=1.0,
    spread_drag_per_contract=0.005,
)

SCENARIO_STRESS = ScenarioProfile(
    name="stress",
    latency_min_ms=80,
    latency_max_ms=200,
    slippage_cents=2,
    fee_multiplier=1.2,
    spread_drag_per_contract=0.02,
)

SCENARIO_PROFILES: Dict[str, ScenarioProfile] = {
    "best": SCENARIO_BEST,
    "base": SCENARIO_BASE,
    "stress": SCENARIO_STRESS,
}


def get_scenario_profile(name: str) -> ScenarioProfile:
    return SCENARIO_PROFILES.get(name, SCENARIO_BASE)


#  Equity ledger — per-bot isolated bankroll tracking

_DEFAULT_START_EQUITY = 5000.0


class BotEquityLedger:
    # Isolated equity tracking for one bot run.
    #
    # Tracks running equity, peak, drawdown, and per-trade cost breakdown.
    # Start equity is always exactly $5,000 unless overridden.

    __slots__ = (
        "start_equity", "equity", "peak_equity",
        "max_drawdown", "max_drawdown_pct",
        "total_fees", "total_slippage", "total_spread_drag",
        "trade_pnls", "trade_count",
    )

    def __init__(self, start_equity: float = _DEFAULT_START_EQUITY) -> None:
        self.start_equity = start_equity
        self.equity = start_equity
        self.peak_equity = start_equity
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_spread_drag = 0.0
        self.trade_pnls: List[float] = []
        self.trade_count = 0

    def record_trade(
        self,
        gross_pnl: float,
        fee_usd: float = 0.0,
        slippage_usd: float = 0.0,
        spread_drag_usd: float = 0.0,
    ) -> float:
        # Record a completed trade. Returns net PnL after all costs.
        net = gross_pnl - fee_usd - slippage_usd - spread_drag_usd
        self.equity += net
        self.total_fees += fee_usd
        self.total_slippage += slippage_usd
        self.total_spread_drag += spread_drag_usd
        self.trade_pnls.append(net)
        self.trade_count += 1

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        dd = self.peak_equity - self.equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        dd_pct = dd / self.start_equity if self.start_equity > 0 else 0.0
        if dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = dd_pct

        return net

    def drawdown_pct(self) -> float:
        # Current drawdown as % of start equity.
        if self.start_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.start_equity)

    def tail_loss(self, percentile: float = 0.10) -> float:
        # Average of worst N% of trade PnLs (CVaR-like).
        if not self.trade_pnls:
            return 0.0
        sorted_pnls = sorted(self.trade_pnls)
        n = max(1, int(math.ceil(len(sorted_pnls) * percentile)))
        return sum(sorted_pnls[:n]) / n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_equity": self.start_equity,
            "equity": round(self.equity, 4),
            "peak_equity": round(self.peak_equity, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            "total_fees": round(self.total_fees, 4),
            "total_slippage": round(self.total_slippage, 4),
            "total_spread_drag": round(self.total_spread_drag, 4),
            "trade_count": self.trade_count,
            "tail_loss_10pct": round(self.tail_loss(), 4),
        }


#  Bot run record — lineage tracking for retire/reseed

@dataclass
class BotRunRecord:
    # Immutable record of one bot run for lineage tracking.
    run_id: str
    bot_id: str
    parent_run_id: str = ""
    generation: int = 0
    scenario: str = "base"
    start_equity: float = _DEFAULT_START_EQUITY
    end_equity: float = _DEFAULT_START_EQUITY
    max_drawdown: float = 0.0
    reason_ended: str = ""       # "retired_drawdown", "retired_epoch", "active", ""
    started_at: float = 0.0
    ended_at: float = 0.0
    params_snapshot: Dict[str, Any] = field(default_factory=dict)
    family_pnl: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex[:12]


#  Execution scenario projection

# Per-contract drag/relief constants (match terminal_ui.py originals)
SCALP_BEST_RELIEF_USD = 0.005
HOLD_BEST_RELIEF_USD = 0.001
SCALP_STRESS_DRAG_USD = 0.02
HOLD_STRESS_DRAG_USD = 0.005

MIN_SETTLED_TRADES_FOR_SCORE = 5


def project_execution_scenarios(stats: Dict[str, Any]) -> Dict[str, float]:
    # Project PnL under best/base/stress execution scenarios.
    pnl = float(stats.get("pnl", 0.0))
    qty_s = float(stats.get("qty_s_contracts", 0.0))
    qty_e = float(stats.get("qty_e_contracts", 0.0))
    return {
        "best": pnl + (qty_s * SCALP_BEST_RELIEF_USD) + (qty_e * HOLD_BEST_RELIEF_USD),
        "base": pnl,
        "stress": pnl - (qty_s * SCALP_STRESS_DRAG_USD) - (qty_e * HOLD_STRESS_DRAG_USD),
    }


def calculate_robustness_score(stats: Dict[str, Any]) -> float:
    # Rank bots by worst-case execution resilience, not optimistic PnL.
    #
    # Incorporates:
    # - Worst-scenario expectancy (45%)
    # - Base-case expectancy (20%)
    # - Max drawdown penalty (15%)
    # - Win rate (10%)
    # - Trade significance (10%)
    # - Fragility penalty (how much PnL varies across scenarios)
    # - Tail-loss penalty (worst 10% of trades)
    # - Small-sample penalty
    wins = int(stats.get("wins", 0))
    losses = int(stats.get("losses", 0))
    trade_count = int(stats.get("trade_count", wins + losses))
    max_drawdown = max(0.0, float(stats.get("max_drawdown", 0.0)))
    scenarios = project_execution_scenarios(stats)
    worst_case = min(scenarios["best"], scenarios["base"], scenarios["stress"])
    base_case = scenarios["base"]
    fragility = max(0.0, scenarios["best"] - scenarios["stress"])
    win_rate = wins / trade_count if trade_count > 0 else 0.0

    worst_score = max(-1.0, min(1.0, worst_case / 400.0))
    base_score = max(-1.0, min(1.0, base_case / 400.0))
    dd_score = 1.0 - max(0.0, min(1.0, max_drawdown / 300.0))
    win_score = max(0.0, min(1.0, (win_rate - 0.35) / 0.40))
    significance = max(0.0, min(1.0, trade_count / 25.0))
    fragility_penalty = max(0.0, min(1.0, fragility / 250.0))

    # Tail-loss penalty: average of worst 10% of trades
    tail_loss = float(stats.get("tail_loss_10pct", 0.0))
    tail_penalty = max(0.0, min(1.0, abs(tail_loss) / 100.0))

    composite = 100.0 * (
        0.45 * worst_score +
        0.20 * base_score +
        0.15 * dd_score +
        0.10 * win_score +
        0.10 * significance
    )
    composite *= (1.0 - 0.35 * fragility_penalty)
    composite *= (1.0 - 0.15 * tail_penalty)

    if trade_count < MIN_SETTLED_TRADES_FOR_SCORE:
        sample_penalty = 0.15 + (0.85 * (trade_count / MIN_SETTLED_TRADES_FOR_SCORE))
        composite *= sample_penalty

    # Execution quality penalties/rewards (optional fields).
    fill_rate = float(stats.get("fill_rate", 1.0))
    partial_rate = float(stats.get("partial_rate", 0.0))
    timeout_rate = float(stats.get("timeout_rate", 0.0))
    realized_edge_net_cents = float(stats.get("realized_edge_net_cents", 0.0))

    # Penalize fragile execution conversion and timeout-heavy behavior.
    if fill_rate < 0.6:
        composite *= max(0.25, 0.5 + fill_rate * 0.5)
    if timeout_rate > 0.2:
        composite *= max(0.30, 1.0 - min(0.6, timeout_rate))
    if partial_rate > 0.5:
        composite *= max(0.50, 1.0 - (partial_rate - 0.5) * 0.6)

    # Reward genuinely positive net realized edge, penalize negative edge.
    edge_factor = max(0.70, min(1.30, 1.0 + (realized_edge_net_cents / 50.0)))
    composite *= edge_factor

    return round(composite, 2)


# Backward-compat alias
calculate_alpha_score = calculate_robustness_score


#  Family classification

FAMILIES = ("BTC 15m", "BTC 60m", "BTC Range", "ETH 15m", "ETH 60m", "ETH Range")


def assign_family(ticker: str, asset: str = "", window_minutes: int = 0, is_range: bool = False) -> str:
    # Extract family key from ticker string or explicit metadata.
    #
    # Returns one of the 6 core families or "Other".
    if asset and window_minutes:
        a = asset.upper()
        if is_range:
            return f"{a} Range"
        if window_minutes == 15:
            return f"{a} 15m"
        if window_minutes == 60:
            return f"{a} 60m"
        return "Other"

    # Parse from ticker string
    series = ticker.upper().split("-")[0] if "-" in ticker else ticker.upper()
    if "BTC" in series:
        a = "BTC"
    elif "ETH" in series:
        a = "ETH"
    else:
        return "Other"

    # Range check must come before the "H" suffix check, otherwise "KXETH"
    # is misclassified as 60m because it ends with "H".
    if series in ("KXBTC", "KXETH"):
        return f"{a} Range"
    elif series.endswith("15M"):
        return f"{a} 15m"
    elif series.endswith("H"):
        return f"{a} 60m"
    else:
        return "Other"


#  Population manager — exploration + exploitation with epoch lifecycle

class PopulationManager:
    # Manages exploit/explore population split and epoch-based retire/reseed.
    #
    # Does NOT mutate bot params in-place. Instead returns instructions for
    # the farm runner to stop old bots and create new ones with new run_ids.

    def __init__(
        self,
        exploit_fraction: float = 0.80,
        retire_bottom_pct: float = 0.20,
        drawdown_retire_pct: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.exploit_fraction = max(0.0, min(1.0, exploit_fraction))
        self.retire_bottom_pct = max(0.0, min(1.0, retire_bottom_pct))
        self.drawdown_retire_pct = max(0.0, min(1.0, drawdown_retire_pct))
        self._rng = random.Random(seed)
        self._epoch_count = 0

    def check_drawdown(
        self,
        bot_id: str,
        current_equity: float,
        peak_equity: float,
        start_equity: float = _DEFAULT_START_EQUITY,
    ) -> Optional[str]:
        # Check if a bot's drawdown exceeds the hard-stop threshold.
        #
        # Returns "retired_drawdown" if breached, None otherwise.
        if start_equity <= 0:
            return None
        dd_pct = max(0.0, (peak_equity - current_equity) / start_equity)
        if dd_pct >= self.drawdown_retire_pct:
            return "retired_drawdown"
        return None

    def evaluate_epoch(
        self,
        bot_scores: List[Tuple[str, float, Dict[str, Any]]],
    ) -> Tuple[List[str], int, int]:
        # Evaluate an epoch: identify bots to retire (global ranking).
        #
        # Args:
        # bot_scores: list of (bot_id, robustness_score, stats_dict)
        #
        # Returns:
        # (retire_ids, exploit_count, explore_count) where:
        # - retire_ids: bot_ids to retire at epoch end
        # - exploit_count: how many replacements should be exploit (perturbation)
        # - explore_count: how many should be explore (random)
        self._epoch_count += 1
        if not bot_scores:
            return [], 0, 0

        # Sort ascending by score — bottom N% are retired
        sorted_bots = sorted(bot_scores, key=lambda x: x[1])
        n_retire = max(1, int(len(sorted_bots) * self.retire_bottom_pct))
        retire_ids = [b[0] for b in sorted_bots[:n_retire]]

        n_exploit = max(0, int(n_retire * self.exploit_fraction))
        n_explore = n_retire - n_exploit

        return retire_ids, n_exploit, n_explore

    def evaluate_epoch_by_family(
        self,
        bot_family_map: Dict[str, str],
        bot_scores: List[Tuple[str, float, Dict[str, Any]]],
    ) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
        # Evaluate epoch within each family bucket independently.
        #
        # Args:
        # bot_family_map: bot_id -> primary family (from config/metadata).
        # bot_scores: list of (bot_id, robustness_score, stats_dict).
        #
        # Returns:
        # (retire_ids, family_split) where family_split maps
        # family -> (n_exploit, n_explore) for replacement generation.
        self._epoch_count += 1
        if not bot_scores:
            return [], {}

        # Bucket scores by family
        family_buckets: Dict[str, List[Tuple[str, float, Dict[str, Any]]]] = {}
        for bot_id, score, stats in bot_scores:
            family = bot_family_map.get(bot_id, "Other")
            family_buckets.setdefault(family, []).append((bot_id, score, stats))

        all_retire_ids: List[str] = []
        family_split: Dict[str, Tuple[int, int]] = {}

        for family, bucket in family_buckets.items():
            if len(bucket) < 2:
                # Never retire in a family with only 1 bot
                family_split[family] = (0, 0)
                continue

            sorted_bots = sorted(bucket, key=lambda x: x[1])
            n_retire = max(1, int(len(sorted_bots) * self.retire_bottom_pct))
            retire_ids = [b[0] for b in sorted_bots[:n_retire]]
            all_retire_ids.extend(retire_ids)

            n_exploit = max(0, int(n_retire * self.exploit_fraction))
            n_explore = n_retire - n_exploit
            family_split[family] = (n_exploit, n_explore)

        return all_retire_ids, family_split

    def generate_replacement_params(
        self,
        top_configs: List[Dict[str, Any]],
        n_exploit: int,
        n_explore: int,
        generation: int,
        parent_run_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        # Generate replacement bot configs.
        #
        # Args:
        # top_configs: configs of top-performing bots to perturb
        # n_exploit: number of exploitation (perturbation) configs
        # n_explore: number of exploration (random) configs
        # generation: current generation number
        # parent_run_ids: run_ids of parent bots (for lineage)
        #
        # Returns:
        # list of new config dicts with run_id, parent_run_id, generation set.
        results: List[Dict[str, Any]] = []

        # Exploit: perturb top performers
        for i in range(n_exploit):
            if not top_configs:
                break
            parent = self._rng.choice(top_configs)
            new_cfg = perturb_config(dict(parent), self._rng)
            new_cfg["_run_id"] = BotRunRecord.new_run_id()
            new_cfg["_parent_run_id"] = (parent_run_ids[i % len(parent_run_ids)]
                                         if parent_run_ids else "")
            new_cfg["_generation"] = generation
            new_cfg["_role"] = "exploit"
            results.append(new_cfg)

        # Explore: random from full parameter space
        for _ in range(n_explore):
            new_cfg = random_explore_config(self._rng)
            new_cfg["_run_id"] = BotRunRecord.new_run_id()
            new_cfg["_parent_run_id"] = ""
            new_cfg["_generation"] = generation
            new_cfg["_role"] = "explore"
            results.append(new_cfg)

        return results


#  FamilyWeightManager — dynamic per-family allocation

class FamilyWeightManager:
    # Dynamic per-family execution weight allocation.
    #
    # Replaces binary drop-loser behavior with continuous weight adaptation.
    # Families never go to zero — min_weight floor guarantees continued
    # exploration.  Weights are applied as a qty multiplier at order sizing
    # time (not just diagnostics).

    def __init__(
        self,
        families: Tuple[str, ...] = FAMILIES,
        min_weight: float = 0.05,
        max_weight: float = 0.40,
        rebalance_interval_s: float = 1800.0,
        decay_alpha: float = 0.3,
    ) -> None:
        self.families = families
        self.min_weight = max(0.01, min(0.5, min_weight))
        self.max_weight = max(self.min_weight, min(1.0, max_weight))
        self.rebalance_interval_s = max(60.0, rebalance_interval_s)
        self.decay_alpha = max(0.01, min(1.0, decay_alpha))
        n = len(families)
        self.weights: Dict[str, float] = {f: 1.0 / n for f in families}
        self._family_pnl_ema: Dict[str, float] = {f: 0.0 for f in families}
        self._family_trade_count: Dict[str, int] = {f: 0 for f in families}
        self._family_reseed_count: Dict[str, int] = {f: 0 for f in families}
        self._last_rebalance_ts: float = 0.0

    def record_family_trade(self, family: str, pnl: float) -> None:
        # Update per-family EMA performance tracker.
        if family not in self._family_pnl_ema:
            self._family_pnl_ema[family] = 0.0
            self._family_trade_count[family] = 0
        alpha = self.decay_alpha
        self._family_pnl_ema[family] = (
            alpha * pnl + (1.0 - alpha) * self._family_pnl_ema[family]
        )
        self._family_trade_count[family] = self._family_trade_count.get(family, 0) + 1

    def record_family_reseed(self, family: str) -> None:
        # Increment reseed counter for a family.
        self._family_reseed_count[family] = self._family_reseed_count.get(family, 0) + 1

    def rebalance(self, now_ts: float) -> bool:
        # Recompute weights from EMA performance. Returns True if rebalanced.
        #
        # Uses softmax-like allocation: higher EMA → more weight.
        # Clamps to [min_weight, max_weight], then normalizes to sum=1.
        # Always allow the first rebalance call; interval applies afterward.
        if (
            self._last_rebalance_ts > 0.0
            and now_ts - self._last_rebalance_ts < self.rebalance_interval_s
        ):
            return False

        emas = self._family_pnl_ema
        families_with_trades = [
            f for f in self.families
            if self._family_trade_count.get(f, 0) > 0
        ]
        if not families_with_trades:
            return False

        # Shift using min across all families to avoid negative values for
        # untraded families when traded-family min_ema > 0.
        min_ema = min(emas.get(f, 0.0) for f in self.families)
        shifted = {f: emas.get(f, 0.0) - min_ema + 1e-6 for f in self.families}

        # Apply softmax-like proportional allocation
        import math as _math
        # Use temperature scaling to prevent extreme concentration
        total_shifted = sum(shifted.values())
        if total_shifted <= 0:
            return False

        raw_weights = {f: shifted[f] / total_shifted for f in self.families}

        # Clamp to [min_weight, max_weight]
        clamped = {
            f: max(self.min_weight, min(self.max_weight, w))
            for f, w in raw_weights.items()
        }

        # Normalize to sum=1
        total = sum(clamped.values())
        if total > 0:
            self.weights = {f: w / total for f, w in clamped.items()}
            self._last_rebalance_ts = now_ts

        return True

    def get_weight(self, family: str) -> float:
        # Return current weight for a family (used as qty multiplier).
        return self.weights.get(family, 1.0 / max(1, len(self.families)))

    def get_diagnostics(self) -> Dict[str, Any]:
        # Return current weights, EMAs, trade counts, reseed counts.
        return {
            "weights": dict(self.weights),
            "pnl_ema": {k: round(v, 4) for k, v in self._family_pnl_ema.items()},
            "trade_count": dict(self._family_trade_count),
            "reseed_count": dict(self._family_reseed_count),
        }


#  Config perturbation / exploration helpers

# Numeric params that can be perturbed
_PERTURBABLE_PARAMS = {
    "min_edge_threshold": (0.03, 0.20),
    "persistence_window_ms": (30, 600),
    "min_entry_cents": (25, 55),
    "max_entry_cents": (55, 85),
    "scalp_min_edge_cents": (8, 30),
    "scalp_min_profit_cents": (12, 30),
    "scalp_cooldown_s": (2.0, 15.0),
    "scalp_min_reprice_move_cents": (2, 10),
    "scalp_reprice_window_s": (1.0, 8.0),
    "scalp_entry_cost_buffer_cents": (2, 16),
    "sizing_risk_fraction": (0.001, 0.01),
}

# Per-family domain overrides (bounds tuning per market type).
# Keys that appear here override _PERTURBABLE_PARAMS for that family.
FAMILY_PARAM_DOMAINS: Dict[str, Dict[str, Tuple[Any, Any]]] = {
    "BTC 15m":   {"persistence_window_ms": (30, 300), "min_edge_threshold": (0.03, 0.15)},
    "BTC 60m":   {"persistence_window_ms": (60, 600), "min_edge_threshold": (0.04, 0.20)},
    "BTC Range": {"min_entry_cents": (30, 60), "max_entry_cents": (55, 80)},
    "ETH 15m":   {"persistence_window_ms": (30, 300), "min_edge_threshold": (0.03, 0.15)},
    "ETH 60m":   {"persistence_window_ms": (60, 600), "min_edge_threshold": (0.04, 0.20)},
    "ETH Range": {"min_entry_cents": (30, 60), "max_entry_cents": (55, 80)},
}


def _resolve_param_bounds(
    family: str = "",
) -> Dict[str, Tuple[Any, Any]]:
    # Return perturbable param bounds, applying family overrides if any.
    bounds = dict(_PERTURBABLE_PARAMS)
    if family and family in FAMILY_PARAM_DOMAINS:
        bounds.update(FAMILY_PARAM_DOMAINS[family])
    return bounds


def perturb_config(
    config: Dict[str, Any],
    rng: random.Random,
    magnitude: float = 0.15,
    family: str = "",
) -> Dict[str, Any]:
    # Generate a neighbor config by perturbing numeric params ±magnitude.
    result = dict(config)
    bounds = _resolve_param_bounds(family)
    for key, (lo, hi) in bounds.items():
        if key not in result:
            continue
        val = result[key]
        if isinstance(val, int):
            delta = max(1, int(abs(val) * magnitude))
            new_val = val + rng.randint(-delta, delta)
            result[key] = max(lo, min(hi, new_val))
        elif isinstance(val, float):
            delta = abs(val) * magnitude
            new_val = val + rng.uniform(-delta, delta)
            result[key] = max(lo, min(hi, round(new_val, 6)))

    # Ensure min_entry <= max_entry
    if result.get("min_entry_cents", 0) > result.get("max_entry_cents", 100):
        result["min_entry_cents"], result["max_entry_cents"] = (
            result["max_entry_cents"], result["min_entry_cents"]
        )
    return result


def perturb_config_tight(
    config: Dict[str, Any],
    rng: random.Random,
    family: str = "",
) -> Dict[str, Any]:
    # Exploit lane: perturb top survivors with 50% tighter step sizes.
    return perturb_config(config, rng, magnitude=0.075, family=family)


def random_explore_config(
    rng: random.Random,
    family: str = "",
) -> Dict[str, Any]:
    # Sample a fully random config from the param space for exploration.
    config: Dict[str, Any] = {}
    bounds = _resolve_param_bounds(family)
    for key, (lo, hi) in bounds.items():
        if isinstance(lo, int) and isinstance(hi, int):
            config[key] = rng.randint(lo, hi)
        else:
            config[key] = round(rng.uniform(float(lo), float(hi)), 6)

    # Defaults for non-perturbable params
    config.setdefault("bankroll_usd", 5000.0)
    config.setdefault("dry_run", True)
    config.setdefault("ws_trading_enabled", False)
    config.setdefault("scalper_enabled", True)
    config.setdefault("no_avoid_above_cents", 75)
    config.setdefault("hold_tail_penalty_start_cents", 55)
    config.setdefault("hold_tail_penalty_per_10c", 0.015)
    config.setdefault("scalp_max_spread_cents", 2)
    config.setdefault("scalp_stop_loss_cents", 0)
    config.setdefault("scalp_exit_grace_s", 2.0)
    config.setdefault("scalp_exit_edge_threshold_cents", 0)
    config.setdefault("scalp_max_entry_cents", 65)
    config.setdefault("scalp_min_entry_cents", 30)
    config.setdefault("use_okx_fallback", True)
    config.setdefault("use_luzia_fallback", False)

    # Fix constraint
    if config.get("min_entry_cents", 0) > config.get("max_entry_cents", 100):
        config["min_entry_cents"], config["max_entry_cents"] = (
            config["max_entry_cents"], config["min_entry_cents"]
        )

    return config


def novelty_distance(
    candidate: Dict[str, Any],
    population: List[Dict[str, Any]],
) -> float:
    # Compute minimum normalized Euclidean distance from candidate to population.
    #
    # Returns 0.0 if population is empty, otherwise a distance in [0, 1] range
    # where 0 = exact clone and 1 = maximally different.
    # Used as a novelty penalty to prevent near-clone configs from being spawned.
    if not population:
        return 1.0

    min_dist = float("inf")
    for existing in population:
        sq_sum = 0.0
        n_dims = 0
        for key, (lo, hi) in _PERTURBABLE_PARAMS.items():
            if key not in candidate or key not in existing:
                continue
            span = float(hi - lo) if hi != lo else 1.0
            diff = (float(candidate[key]) - float(existing[key])) / span
            sq_sum += diff * diff
            n_dims += 1
        if n_dims > 0:
            dist = (sq_sum / n_dims) ** 0.5
            min_dist = min(min_dist, dist)

    return min(1.0, min_dist) if min_dist != float("inf") else 1.0
