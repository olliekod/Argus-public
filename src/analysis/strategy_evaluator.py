# Created by Oliver Meihls

# Strategy Evaluator
#
# Loads experiment JSON artifacts produced by :class:`ExperimentRunner`,
# computes standardized metrics, applies composite scoring with penalty
# factors, and outputs a deterministic ranking.
#
# Metrics
# - PnL, Sharpe proxy, max drawdown, expectancy, profit factor
# - Trade counts, reject ratios
# - Regime-conditioned performance
#
# Composite scoring
# Weighted sum of normalized metrics with penalties for:
# - Drawdown severity
# - High reject rate
# - Low robustness (parameter fragility across sweeps)
# - Regime dependency (concentration in a single regime)
# - Walk-forward instability
#
# Edge-case handling
# All metric computations are NaN-safe and handle:
# - No trades, no losses, zero variance
# - Missing or null fields in experiment JSON
#
# Output
# ------
# ``logs/strategy_rankings_<date>.json`` — ranked list with composite
# scores, per-metric breakdowns, and manifest references.

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .deflated_sharpe import (
    compute_deflated_sharpe_ratio,
    deflated_sharpe_ratio,
    threshold_sharpe_ratio,
)
from .reality_check import reality_check

logger = logging.getLogger("argus.strategy_evaluator")


# Metric extraction

def _safe_float(val: Any, default: float = 0.0) -> float:
    # Convert *val* to float, returning *default* on failure or NaN.
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def extract_metrics(experiment: Dict[str, Any]) -> Dict[str, Any]:
    # Extract standardized metrics from a single experiment artifact.
    #
    # Parameters
    # experiment : dict
    # A JSON-loaded experiment artifact with ``manifest`` and ``result`` keys.
    #
    # Returns
    # dict
    # Flat dict of named metrics, all numeric or None.
    result = experiment.get("result", {})
    portfolio = result.get("portfolio", {})
    execution = result.get("execution", {})
    manifest = experiment.get("manifest", {})
    mc_bootstrap = manifest.get("mc_bootstrap") or {}
    mc_metrics = mc_bootstrap.get("metrics") if isinstance(mc_bootstrap, dict) else {}

    total_pnl = _safe_float(portfolio.get("total_realized_pnl"))
    starting_cash = _safe_float(portfolio.get("starting_cash"), 10_000.0)
    total_return_pct = _safe_float(portfolio.get("total_return_pct"))
    sharpe = _safe_float(portfolio.get("sharpe_annualized_proxy"))
    max_dd = _safe_float(portfolio.get("max_drawdown"))
    max_dd_pct = _safe_float(portfolio.get("max_drawdown_pct"))
    expectancy = _safe_float(portfolio.get("expectancy"))
    profit_factor = _safe_float(portfolio.get("profit_factor"))
    win_rate = _safe_float(portfolio.get("win_rate"))

    total_trades = int(_safe_float(portfolio.get("total_trades")))
    winners = int(_safe_float(portfolio.get("winners")))
    losers = int(_safe_float(portfolio.get("losers")))

    fills = int(_safe_float(execution.get("fills")))
    rejects = int(_safe_float(execution.get("rejects")))
    fill_rate = _safe_float(execution.get("fill_rate"), 1.0)

    bars_replayed = int(_safe_float(result.get("bars_replayed")))

    # Regime breakdown
    regime_breakdown = portfolio.get("regime_breakdown", {})

    return {
        "strategy_id": result.get("strategy_id", "UNKNOWN"),
        "run_id": manifest.get("run_id", ""),
        "strategy_class": manifest.get("strategy_class", ""),
        "strategy_params": manifest.get("strategy_params", {}),
        "total_pnl": total_pnl,
        "starting_cash": starting_cash,
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "winners": winners,
        "losers": losers,
        "fills": fills,
        "rejects": rejects,
        "fill_rate": fill_rate,
        "bars_replayed": bars_replayed,
        "regime_breakdown": regime_breakdown,
        "mc_bootstrap": mc_bootstrap if isinstance(mc_bootstrap, dict) else {},
        "mc_median_return": _safe_float((mc_metrics or {}).get("median_return"), default=0.0),
        "mc_p5_max_drawdown": _safe_float((mc_metrics or {}).get("p5_max_drawdown"), default=0.0),
        "mc_p95_max_drawdown": _safe_float((mc_metrics or {}).get("p95_max_drawdown"), default=0.0),
        "mc_ruin_probability": _safe_float((mc_metrics or {}).get("ruin_probability"), default=0.0),
        "mc_fraction_positive": _safe_float((mc_metrics or {}).get("fraction_positive"), default=0.0),
    }


# Penalty computations

def compute_reject_penalty(metrics: Dict[str, Any]) -> float:
    # Penalty for high rejection rate. Range [0, 1].
    #
    # - fill_rate >= 0.8 → 0 penalty
    # - fill_rate == 0   → 1.0 penalty
    fill_rate = _safe_float(metrics.get("fill_rate"), 1.0)
    if fill_rate >= 0.8:
        return 0.0
    return round(min(1.0, (0.8 - fill_rate) / 0.8), 4)


def compute_drawdown_penalty(metrics: Dict[str, Any]) -> float:
    # Penalty for excessive drawdown relative to starting capital.
    #
    # - max_dd_pct <= 5%  → 0 penalty
    # - max_dd_pct >= 50% → 1.0 penalty
    # - Linear between
    dd_pct = _safe_float(metrics.get("max_drawdown_pct"))
    if dd_pct <= 5.0:
        return 0.0
    if dd_pct >= 50.0:
        return 1.0
    return round((dd_pct - 5.0) / 45.0, 4)


def compute_regime_dependency_penalty(metrics: Dict[str, Any]) -> float:
    # Penalty for regime-concentrated performance.
    #
    # If >80% of PnL comes from a single regime bucket, penalty = 0.5.
    # If >90%, penalty = 0.8. Otherwise 0.
    breakdown = metrics.get("regime_breakdown", {})
    if not breakdown:
        return 0.0

    pnls = []
    for key, stats in breakdown.items():
        if isinstance(stats, dict):
            pnls.append(abs(_safe_float(stats.get("pnl"))))
        else:
            pnls.append(0.0)

    total_abs = sum(pnls)
    if total_abs == 0:
        return 0.0

    max_concentration = max(pnls) / total_abs
    if max_concentration > 0.9:
        return 0.8
    if max_concentration > 0.8:
        return 0.5
    return 0.0


def compute_regime_sensitivity_score(metrics: Dict[str, Any]) -> float:
    # Score regime balance based on PnL-per-bar dispersion. Range [0, 1].
    #
    # 1.0 means stable across regimes; 0.0 means highly regime-sensitive.
    breakdown = metrics.get("regime_breakdown", {})
    if not breakdown:
        return 0.5

    pnl_per_bar_values: List[float] = []
    for stats in breakdown.values():
        if not isinstance(stats, dict):
            continue
        bars = int(_safe_float(stats.get("bars")))
        if bars <= 0:
            continue
        pnl = _safe_float(stats.get("pnl"))
        pnl_per_bar_values.append(pnl / bars)

    if len(pnl_per_bar_values) == 0:
        return 0.5
    if len(pnl_per_bar_values) == 1:
        return 0.0

    mean = sum(pnl_per_bar_values) / len(pnl_per_bar_values)
    variance = sum((v - mean) ** 2 for v in pnl_per_bar_values) / len(pnl_per_bar_values)
    std = math.sqrt(variance)
    if abs(mean) < 1e-9:
        cv = 1.0 if std > 0 else 0.0
    else:
        cv = abs(std / mean)

    sensitivity = max(0.0, 1.0 - min(cv, 1.0))
    return round(sensitivity, 4)


def compute_robustness_penalty(
    all_metrics: Sequence[Dict[str, Any]],
    target_run_id: str,
) -> float:
    # Penalize parameter fragility across sweep results.
    #
    # Groups experiments by strategy_class.  If the same strategy class
    # has multiple runs (parameter sweep), we measure the coefficient of
    # variation of PnL across runs.  High CV → fragile → penalty.
    #
    # Returns penalty in [0, 1] for the target run.
    # Find the strategy class of the target
    target = None
    for m in all_metrics:
        if m.get("run_id") == target_run_id:
            target = m
            break
    if target is None:
        return 0.0

    strategy_class = target.get("strategy_class", "")
    if not strategy_class:
        return 0.0

    # Collect PnL values for the same strategy class
    pnls = [
        m["total_pnl"]
        for m in all_metrics
        if m.get("strategy_class") == strategy_class
    ]

    if len(pnls) < 2:
        return 0.0  # Can't assess robustness with a single run

    mean_pnl = sum(pnls) / len(pnls)
    if mean_pnl == 0:
        return 0.5  # Zero mean with variance → fragile

    variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
    std_pnl = math.sqrt(variance)
    cv = abs(std_pnl / mean_pnl) if mean_pnl != 0 else 0.0

    # CV thresholds
    if cv <= 0.3:
        return 0.0
    if cv >= 2.0:
        return 1.0
    return round((cv - 0.3) / 1.7, 4)


def compute_walk_forward_penalty(
    all_metrics: Sequence[Dict[str, Any]],
    target_run_id: str,
) -> float:
    # Penalize walk-forward instability.
    #
    # If the strategy's performance varies wildly across different
    # time windows (detected by strategy_params containing window indices),
    # apply a stability penalty.
    #
    # Returns penalty in [0, 1].
    target = None
    for m in all_metrics:
        if m.get("run_id") == target_run_id:
            target = m
            break
    if target is None:
        return 0.0

    strategy_class = target.get("strategy_class", "")
    if not strategy_class:
        return 0.0

    # Find all runs with same strategy class and same params
    # (walk-forward produces multiple windows for same config)
    same_config = [
        m for m in all_metrics
        if m.get("strategy_class") == strategy_class
        and m.get("strategy_params") == target.get("strategy_params")
    ]

    if len(same_config) < 2:
        return 0.0

    pnls = [m["total_pnl"] for m in same_config]
    # Count sign changes
    positive = sum(1 for p in pnls if p > 0)
    negative = sum(1 for p in pnls if p < 0)
    total = len(pnls)

    # If majority is positive, low penalty
    if total == 0:
        return 0.0

    consistency = max(positive, negative) / total
    if consistency >= 0.8:
        return 0.0
    if consistency <= 0.5:
        return 0.8
    # Linear interpolation
    return round((0.8 - consistency) / 0.3 * 0.8, 4)


# Regime-conditioned performance

def compute_regime_scores(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    # Compute per-regime PnL and bar counts.
    #
    # Returns a dict mapping regime keys to ``{"pnl": float, "bars": int, "pnl_per_bar": float}``.
    breakdown = metrics.get("regime_breakdown", {})
    scores: Dict[str, Dict[str, float]] = {}

    for key, stats in breakdown.items():
        if not isinstance(stats, dict):
            continue
        pnl = _safe_float(stats.get("pnl"))
        bars = int(_safe_float(stats.get("bars")))
        pnl_per_bar = round(pnl / bars, 4) if bars > 0 else 0.0
        scores[key] = {
            "pnl": round(pnl, 2),
            "bars": bars,
            "pnl_per_bar": pnl_per_bar,
        }
    return scores


# Composite scoring

DEFAULT_WEIGHTS = {
    "return": 0.10,            # Profitability (Requested: 10%)
    "sharpe": 0.35,            # Sharpe takes precedence
    "win_rate": 0.20,
    "mc_stability": 0.20,
    "drawdown_penalty": -0.10,
    "reject_penalty": -0.05,
}


def compute_composite_score(
    metrics: Dict[str, Any],
    all_metrics: Sequence[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    # Compute the composite score for a single experiment.
    #
    # Returns a dict with the composite score (0-100), component scores, and penalties.
    w = weights or DEFAULT_WEIGHTS

    # 1. Normalized return component [0, 1]
    all_returns = [_safe_float(m.get("total_return_pct")) for m in all_metrics]
    min_ret = min(all_returns) if all_returns else 0.0
    max_ret = max(all_returns) if all_returns else 0.0
    ret_range = max_ret - min_ret
    norm_return = (_safe_float(metrics.get("total_return_pct")) - min_ret) / ret_range if ret_range > 0 else 0.5

    # 2. Normalized Sharpe [0, 1]
    all_sharpes = [_safe_float(m.get("sharpe")) for m in all_metrics]
    min_sharpe = min(all_sharpes) if all_sharpes else 0.0
    max_sharpe = max(all_sharpes) if all_sharpes else 0.0
    sharpe_range = max_sharpe - min_sharpe
    norm_sharpe = (_safe_float(metrics.get("sharpe")) - min_sharpe) / sharpe_range if sharpe_range > 0 else 0.5

    # 3. Normalized Win Rate [0, 1]
    all_win_rates = [_safe_float(m.get("win_rate")) for m in all_metrics]
    min_wr = min(all_win_rates) if all_win_rates else 0.0
    max_wr = max(all_win_rates) if all_win_rates else 0.0
    wr_range = max_wr - min_wr
    norm_win_rate = (metrics.get("win_rate", 0.0) - min_wr) / wr_range if wr_range > 0 else 0.5

    # 4. MC Stability Score [0, 1]
    mc_frac = _safe_float(metrics.get("mc_fraction_positive"), 0.0)
    mc_ruin = _safe_float(metrics.get("mc_ruin_probability"), 1.0)
    mc_stability = (mc_frac * 0.7) + ((1.0 - mc_ruin) * 0.3)

    # 5. Penalties
    dd_penalty = compute_drawdown_penalty(metrics)
    rej_penalty = compute_reject_penalty(metrics)
    rob_penalty = compute_robustness_penalty(all_metrics, metrics.get("run_id", ""))
    
    # Calculate raw score
    composite_raw = (
        w.get("return", 0.30) * norm_return
        + w.get("sharpe", 0.25) * norm_sharpe
        + w.get("win_rate", 0.15) * norm_win_rate
        + w.get("mc_stability", 0.15) * mc_stability
        + w.get("drawdown_penalty", -0.10) * dd_penalty
        + w.get("reject_penalty", -0.05) * rej_penalty
    )

    # Scale to 0-100 for human readability
    composite_100 = round(max(0.0, min(1.0, composite_raw)) * 100.0, 2)

    return {
        "composite_score": composite_100,
        "raw_score": round(composite_raw, 6),
        "components": {
            "return_norm": round(norm_return, 4),
            "sharpe_norm": round(norm_sharpe, 4),
            "win_rate_norm": round(norm_win_rate, 4),
            "mc_stability": round(mc_stability, 4),
            "drawdown_penalty": round(dd_penalty, 4),
            "reject_penalty": round(rej_penalty, 4),
            "robustness_penalty": round(rob_penalty, 4),
        },
        "weights": w,
    }


# Evaluator

class StrategyEvaluator:
    # Load experiment results, compute metrics, and rank strategies.

    def __init__(
        self,
        input_dir: str = "logs/experiments",
        output_dir: str = "logs",
        weights: Optional[Dict[str, float]] = None,
        kill_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.weights = weights or DEFAULT_WEIGHTS
        self.kill_thresholds = kill_thresholds or {
            "robustness_penalty": 0.6,
            "walk_forward_penalty": 0.6,
            "regime_dependency_penalty": 0.7,
            "composite_score_min": 0.1,
            "mc_median_return_min": -1.0,
            "mc_p5_drawdown_max": 2.0,
            "mc_ruin_prob_max": 2.0,
            "mc_fraction_positive_min": -1.0,
            # Deploy gate thresholds (Phase 4C)
            "dsr_min": 0.95,
            "slippage_sensitivity_max_cost_multiplier": 1.50,
            "reality_check_p_max": 0.05,
        }
        self._experiments: List[Dict[str, Any]] = []
        self._metrics: List[Dict[str, Any]] = []
        self._rankings: List[Dict[str, Any]] = []
        self._killed: List[Dict[str, Any]] = []

    def load_experiments(self, paths: Optional[List[str]] = None) -> int:
        # Load experiment JSON files.
        #
        # Parameters
        # paths : list of str, optional
        # Explicit list of JSON file paths. If None, scans ``input_dir``.
        #
        # Returns
        # int
        # Number of experiments loaded.
        if paths:
            files = [Path(p) for p in paths]
        else:
            if not self.input_dir.exists():
                logger.warning("Input directory does not exist: %s", self.input_dir)
                return 0
            files = sorted(self.input_dir.glob("*.json"))

        self._experiments = []
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                # Validate minimal structure
                if "result" in data:
                    data["_source_file"] = str(f)
                    self._experiments.append(data)
                else:
                    logger.warning("Skipping %s: missing 'result' key", f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Skipping %s: %s", f, e)

        logger.info("Loaded %d experiments", len(self._experiments))
        return len(self._experiments)

    def evaluate(self) -> List[Dict[str, Any]]:
        # Compute metrics and composite scores for all loaded experiments.
        #
        # Returns
        # list of dict
        # Ranked list (best first) of evaluation records.
        # 1. Extract metrics
        self._metrics = [extract_metrics(exp) for exp in self._experiments]

        # 2. Compute DSR for all experiments (if enough data)
        dsr_results = self._compute_dsr_for_all()

        # 3. Compute composite scores
        scored: List[Dict[str, Any]] = []
        for i, metrics in enumerate(self._metrics):
            score_info = compute_composite_score(metrics, self._metrics, self.weights)
            regime_scores = compute_regime_scores(metrics)
            run_id = metrics["run_id"]

            record = {
                "rank": 0,  # filled after sorting
                "strategy_id": metrics["strategy_id"],
                "run_id": run_id,
                "strategy_class": metrics["strategy_class"],
                "strategy_params": metrics["strategy_params"],
                "composite_score": score_info["composite_score"],
                "regime_sensitivity_score": score_info["components"].get("regime_sensitivity", 0.5),
                "scoring": score_info,
                "dsr": dsr_results.get(run_id, {}).get("dsr", 0.0),
                "dsr_details": dsr_results.get(run_id, {}),
                "metrics": {
                    "total_pnl": metrics["total_pnl"],
                    "total_return_pct": metrics["total_return_pct"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "expectancy": metrics["expectancy"],
                    "profit_factor": metrics["profit_factor"],
                    "win_rate": metrics["win_rate"],
                    "total_trades": metrics["total_trades"],
                    "fill_rate": metrics["fill_rate"],
                    "fills": metrics["fills"],
                    "rejects": metrics["rejects"],
                    "mc_median_return": metrics["mc_median_return"],
                    "mc_p95_max_drawdown": metrics["mc_p95_max_drawdown"],
                    "mc_ruin_probability": metrics["mc_ruin_probability"],
                    "mc_fraction_positive": metrics["mc_fraction_positive"],
                    "dsr": dsr_results.get(run_id, {}).get("dsr", 0.0),
                },
                "regime_scores": regime_scores,
                "manifest_ref": self._experiments[i].get("manifest", {}),
                "source_file": self._experiments[i].get("_source_file", ""),
            }

            kill_reasons = self._compute_kill_reasons(record)
            record["killed"] = len(kill_reasons) > 0
            record["kill_reasons"] = kill_reasons
            scored.append(record)

        # 3. Sort by composite score descending
        scored.sort(key=lambda r: r["composite_score"], reverse=True)

        # 4. Assign ranks
        for i, rec in enumerate(scored):
            rec["rank"] = i + 1

        self._rankings = scored
        self._killed = []
        for rec in scored:
            for reason in rec.get("kill_reasons", []):
                self._killed.append(
                    {
                        "run_id": rec.get("run_id", ""),
                        "strategy_id": rec.get("strategy_id", ""),
                        "strategy_class": rec.get("strategy_class", ""),
                        "strategy_params": rec.get("strategy_params", {}),
                        "reason": reason["reason"],
                        "value": reason["value"],
                        "threshold": reason["threshold"],
                    }
                )
        return scored

    def _compute_dsr_for_all(self) -> Dict[str, Dict[str, Any]]:
        # Compute Deflated Sharpe Ratio for all experiments.
        #
        # Uses cross-sectional Sharpe ratios across experiments and the
        # total experiment count as the number of trials (conservative).
        #
        # Returns a dict mapping run_id -> DSR result dict.
        n_trials = len(self._metrics)
        if n_trials < 2:
            return {}

        # Collect all Sharpe ratios
        all_sharpes = [
            _safe_float(m.get("sharpe")) for m in self._metrics
        ]

        results: Dict[str, Dict[str, Any]] = {}
        for i, metrics in enumerate(self._metrics):
            run_id = metrics.get("run_id", "")
            sharpe = _safe_float(metrics.get("sharpe"))
            n_obs = metrics.get("total_trades", 0)

            if n_obs < 10:
                results[run_id] = {"dsr": 0.0, "reason": "insufficient_observations"}
                continue

            # If trade_pnls are available in the metrics, compute skew and kurtosis
            trade_pnls = metrics.get("trade_pnls", [])
            skew_val = 0.0
            kurt_val = 0.0
            if len(trade_pnls) >= 3:
                mean_pnl = sum(trade_pnls) / len(trade_pnls)
                var_pnl = sum((p - mean_pnl) ** 2 for p in trade_pnls) / len(trade_pnls)
                if var_pnl > 0:
                    std_pnl = var_pnl ** 0.5
                    skew_val = sum((p - mean_pnl) ** 3 for p in trade_pnls) / (len(trade_pnls) * std_pnl ** 3)
                    kurt_val = sum((p - mean_pnl) ** 4 for p in trade_pnls) / (len(trade_pnls) * std_pnl ** 4) - 3.0

            try:
                if all_sharpes and len(all_sharpes) >= 2:
                    mean_sr = sum(all_sharpes) / len(all_sharpes)
                    sr_var = sum((s - mean_sr) ** 2 for s in all_sharpes) / (
                        len(all_sharpes) - 1
                    )
                else:
                    sr_var = 1.0
                sr_0 = threshold_sharpe_ratio(sr_var, n_trials)
                dsr_val = compute_deflated_sharpe_ratio(
                    observed_sharpe=sharpe,
                    threshold_sr=sr_0,
                    n_obs=n_obs,
                    skewness=skew_val,
                    kurtosis=kurt_val,
                )
                results[run_id] = {
                    "dsr": round(dsr_val, 6),
                    "observed_sharpe": round(sharpe, 6),
                    "threshold_sr": round(sr_0, 6),
                    "n_obs": n_obs,
                    "n_trials": n_trials,
                    "skew": round(skew_val, 6),
                    "kurtosis": round(kurt_val, 6),
                    "sharpe_variance": round(sr_var, 6),
                }
            except Exception as e:
                logger.warning("DSR computation failed for %s: %s", run_id, e)
                results[run_id] = {"dsr": 0.0, "reason": str(e)}

        return results

    def _compute_kill_reasons(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        reasons: List[Dict[str, Any]] = []
        thresholds = self.kill_thresholds
        components = record.get("scoring", {}).get("components", {})

        if components.get("robustness_penalty", 0.0) >= thresholds.get("robustness_penalty", 2.0):
            reasons.append(
                {
                    "reason": "robustness_penalty",
                    "value": round(components.get("robustness_penalty", 0.0), 4),
                    "threshold": thresholds.get("robustness_penalty"),
                }
            )

        if components.get("walk_forward_penalty", 0.0) >= thresholds.get("walk_forward_penalty", 2.0):
            reasons.append(
                {
                    "reason": "walk_forward_penalty",
                    "value": round(components.get("walk_forward_penalty", 0.0), 4),
                    "threshold": thresholds.get("walk_forward_penalty"),
                }
            )

        if components.get("regime_dependency_penalty", 0.0) >= thresholds.get("regime_dependency_penalty", 2.0):
            reasons.append(
                {
                    "reason": "regime_dependency_penalty",
                    "value": round(components.get("regime_dependency_penalty", 0.0), 4),
                    "threshold": thresholds.get("regime_dependency_penalty"),
                }
            )

        if record.get("composite_score", 0.0) < thresholds.get("composite_score_min", -1.0):
            reasons.append(
                {
                    "reason": "composite_score_min",
                    "value": round(record.get("composite_score", 0.0), 6),
                    "threshold": thresholds.get("composite_score_min"),
                }
            )


        mc_block = record.get("manifest_ref", {}).get("mc_bootstrap")
        if isinstance(mc_block, dict):
            for reason in mc_block.get("reasons", []):
                if isinstance(reason, dict) and {"reason", "value", "threshold"}.issubset(reason.keys()):
                    reasons.append(
                        {
                            "reason": str(reason.get("reason")),
                            "value": round(_safe_float(reason.get("value")), 6),
                            "threshold": _safe_float(reason.get("threshold")),
                        }
                    )

        metrics = record.get("metrics", {})
        if metrics.get("mc_median_return", 0.0) < thresholds.get("mc_median_return_min", -1.0):
            reasons.append(
                {
                    "reason": "mc_median_return",
                    "value": round(metrics.get("mc_median_return", 0.0), 6),
                    "threshold": thresholds.get("mc_median_return_min"),
                }
            )

        mc_p95_dd_threshold = thresholds.get("mc_p95_drawdown_max", thresholds.get("mc_p5_drawdown_max", 2.0))
        if metrics.get("mc_p95_max_drawdown", 0.0) > mc_p95_dd_threshold:
            reasons.append(
                {
                    "reason": "mc_p95_drawdown",
                    "value": round(metrics.get("mc_p95_max_drawdown", 0.0), 6),
                    "threshold": mc_p95_dd_threshold,
                }
            )

        if metrics.get("mc_ruin_probability", 0.0) > thresholds.get("mc_ruin_prob_max", 2.0):
            reasons.append(
                {
                    "reason": "mc_ruin_prob",
                    "value": round(metrics.get("mc_ruin_probability", 0.0), 6),
                    "threshold": thresholds.get("mc_ruin_prob_max"),
                }
            )

        if metrics.get("mc_fraction_positive", 0.0) < thresholds.get("mc_fraction_positive_min", -1.0):
            reasons.append(
                {
                    "reason": "mc_fraction_positive",
                    "value": round(metrics.get("mc_fraction_positive", 0.0), 6),
                    "threshold": thresholds.get("mc_fraction_positive_min"),
                }
            )

        # ── Deploy gate: DSR below threshold ───────────────────────
        dsr_min = thresholds.get("dsr_min", 0.95)
        dsr_val = record.get("dsr", 0.0)
        if dsr_min > 0 and dsr_val < dsr_min:
            reasons.append(
                {
                    "reason": "dsr_below_threshold",
                    "value": round(dsr_val, 6),
                    "threshold": dsr_min,
                }
            )

        # ── Deploy gate: slippage sensitivity ──────────────────────
        slippage_block = record.get("manifest_ref", {}).get("slippage_sensitivity")
        if isinstance(slippage_block, dict) and slippage_block.get("killed", False):
            reasons.append(
                {
                    "reason": "slippage_sensitivity",
                    "value": slippage_block.get("sharpe_at_150pct", 0.0),
                    "threshold": 0.0,
                }
            )

        # ── Deploy gate: Reality Check p-value ─────────────────────
        rc_block = record.get("manifest_ref", {}).get("reality_check")
        if isinstance(rc_block, dict):
            rc_p = rc_block.get("p_value", 1.0)
            rc_max = thresholds.get("reality_check_p_max", 0.05)
            if rc_p >= rc_max:
                reasons.append(
                    {
                        "reason": "reality_check_failed",
                        "value": round(rc_p, 6),
                        "threshold": rc_max,
                    }
                )

        return reasons

    def save_rankings(self, output_path: Optional[str] = None) -> str:
        # Write the rankings to JSON.
        #
        # Returns the output file path.
        if not self._rankings:
            self.evaluate()

        if output_path is None:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            output_path = str(self.output_dir / f"strategy_rankings_{date_str}.json")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment_count": len(self._rankings),
            "weights": self.weights,
            "kill_thresholds": self.kill_thresholds,
            "rankings": self._rankings,
            "killed": self._killed,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info("Rankings saved to %s", output_path)
        return output_path

    def print_summary(self) -> str:
        # Print a console-friendly summary table.
        #
        # Returns the formatted string.
        if not self._rankings:
            self.evaluate()

        lines = []
        lines.append("=" * 100)
        lines.append("STRATEGY RANKINGS")
        lines.append("=" * 100)
        lines.append(
            f"{'Rank':<5} {'Strategy':<25} {'Score':>8} {'PnL':>10} "
            f"{'Sharpe':>8} {'DD%':>7} {'Trades':>7} {'FillR':>7} {'WinR':>7}"
        )
        lines.append("-" * 100)

        for r in self._rankings:
            m = r["metrics"]
            lines.append(
                f"{r['rank']:<5} {r['strategy_id']:<25} "
                f"{r['composite_score']:>8.2f} "
                f"{m['total_pnl']:>10.2f} "
                f"{m['sharpe']:>8.2f} "
                f"{m['max_drawdown_pct']:>7.1f} "
                f"{m['total_trades']:>7} "
                f"{m['fill_rate']:>7.2f} "
                f"{m['win_rate']:>7.1f}"
            )

        lines.append("=" * 100)
        summary = "\n".join(lines)
        print(summary)
        return summary

    @property
    def rankings(self) -> List[Dict[str, Any]]:
        return list(self._rankings)

    @property
    def metrics(self) -> List[Dict[str, Any]]:
        return list(self._metrics)

    @property
    def killed(self) -> List[Dict[str, Any]]:
        return list(self._killed)
