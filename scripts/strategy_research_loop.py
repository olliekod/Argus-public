#!/usr/bin/env python3
"""
Strategy Research Loop
=======================

Single entry point that runs the full strategy research cycle:

1. Resolve date range from config.
2. Optionally backfill outcomes for the date range.
3. Build replay packs (single-symbol or universe).
4. Run experiments for each strategy (single or parameter sweep)
   with optional MC/bootstrap and regime-stress.
5. Evaluate experiments, persist rankings, killed list, and candidate set.

Usage::

    # One-shot cycle
    python scripts/strategy_research_loop.py --config config/research_loop.yaml --once

    # Daemon mode (runs every loop.interval_hours)
    python scripts/strategy_research_loop.py --config config/research_loop.yaml

    # Dry-run (validate config, log steps, no execution)
    python scripts/strategy_research_loop.py --config config/research_loop.yaml --dry-run

Authority: MASTER_PLAN.md §8.4, COMMANDS_AND_NEXT_STEPS.md §5-6.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml

from src.analysis.research_loop_config import (
    ConfigValidationError,
    ResearchLoopConfig,
    load_research_loop_config,
)
from src.analysis.sweep_grid import expand_sweep_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("argus.research_loop")


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy class loader (same logic as run_experiment.py)
# ═══════════════════════════════════════════════════════════════════════════

_STRATEGY_MODULES = [
    "src.strategies.vrp_credit_spread",
    "src.strategies.dow_regime_timing",
    "src.strategies.overnight_session",
]


def load_strategy_class(name: str):
    """Dynamically load a ReplayStrategy class by name."""
    for mod_name in _STRATEGY_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, name):
                return getattr(mod, name)
        except ImportError:
            continue
    raise ImportError(f"Could not find strategy class '{name}' in known modules.")


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1: Outcomes backfill
# ═══════════════════════════════════════════════════════════════════════════


def run_outcomes_backfill(config: ResearchLoopConfig) -> None:
    """Backfill outcomes for the configured date range.

    Calls ``python -m src.outcomes backfill`` (single) or
    ``python -m src.outcomes backfill-all`` (universe) as a subprocess.
    """
    if not config.outcomes.ensure_before_pack:
        logger.info("Outcomes backfill skipped (ensure_before_pack=false).")
        return

    start = config.pack.start_date
    end = config.pack.end_date

    if config.pack.mode == "universe":
        cmd = [
            sys.executable, "-m", "src.outcomes",
            "--db", config.pack.db_path,
            "backfill-all",
            "--start", start,
            "--end", end,
        ]
        logger.info("Running outcomes backfill-all: %s to %s", start, end)
        _run_subprocess(cmd)
    else:
        # Determine bars provider
        bars_provider = config.pack.bars_provider
        if bars_provider is None:
            bars_provider = _read_bars_primary()

        for symbol in config.pack.symbols:
            cmd = [
                sys.executable, "-m", "src.outcomes",
                "--db", config.pack.db_path,
                "backfill",
                "--provider", bars_provider,
                "--symbol", symbol,
                "--bar", str(config.outcomes.bar_duration),
                "--start", start,
                "--end", end,
            ]
            logger.info(
                "Running outcomes backfill: %s/%s %s to %s",
                bars_provider, symbol, start, end,
            )
            _run_subprocess(cmd)


def _read_bars_primary() -> str:
    """Read data_sources.bars_primary from config/config.yaml."""
    cfg_path = _REPO / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("data_sources", {}).get("bars_primary", "alpaca")
    return "alpaca"


def _run_subprocess(cmd: List[str]) -> None:
    """Run a subprocess command and raise on non-zero exit."""
    logger.debug("Subprocess: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (exit {result.returncode}): {' '.join(cmd)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Recent-bars freshness check (loop.require_recent_bars_hours)
# ═══════════════════════════════════════════════════════════════════════════


def _check_recent_bars_freshness(db_path: str, max_age_hours: float) -> bool:
    """Return True if the most recent bar in the DB is within max_age_hours of now.

    Queries market_bars via get_bar_inventory and parses max_ts (ISO format).
    Returns False if the DB has no bars or the newest bar is older than max_age_hours.
    """
    async def _check() -> bool:
        from src.core.database import Database
        db = Database(db_path)
        await db.connect()
        try:
            inv = await db.get_bar_inventory()
            if not inv:
                return False
            latest_ts: Optional[datetime] = None
            for row in inv:
                max_ts_str = row.get("max_ts")
                if not max_ts_str:
                    continue
                try:
                    dt = datetime.fromisoformat(
                        str(max_ts_str).replace("Z", "+00:00")
                    )
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if latest_ts is None or dt > latest_ts:
                        latest_ts = dt
                except (ValueError, TypeError):
                    continue
            if latest_ts is None:
                return False
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            return latest_ts >= cutoff
        finally:
            await db.close()

    return asyncio.run(_check())


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2: Build replay packs
# ═══════════════════════════════════════════════════════════════════════════


def build_packs(config: ResearchLoopConfig) -> List[str]:
    """Build replay packs and return a list of pack file paths.

    Uses ``src.tools.replay_pack`` (async API) wrapped in ``asyncio.run()``.
    """
    from src.tools.replay_pack import create_replay_pack, create_universe_packs

    start = config.pack.start_date
    end = config.pack.end_date
    output_dir = config.pack.packs_output_dir
    db_path = config.pack.db_path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if config.pack.mode == "universe":
        logger.info("Building universe packs: %s to %s", start, end)
        paths = asyncio.run(create_universe_packs(
            start_date=start,
            end_date=end,
            output_dir=output_dir,
            provider=config.pack.bars_provider,
            db_path=db_path,
            snapshot_provider=config.pack.options_snapshot_provider,
        ))
        logger.info("Built %d universe packs.", len(paths))
        return [str(p) for p in paths]
    else:
        paths: List[str] = []
        for symbol in config.pack.symbols:
            out_path = str(
                Path(output_dir) / f"{symbol}_{start}_{end}.json"
            )
            logger.info("Building pack: %s -> %s", symbol, out_path)
            asyncio.run(create_replay_pack(
                symbol=symbol,
                start_date=start,
                end_date=end,
                output_path=out_path,
                provider=config.pack.bars_provider,
                db_path=db_path,
                snapshot_provider=config.pack.options_snapshot_provider,
            ))
            paths.append(out_path)
        logger.info("Built %d packs.", len(paths))
        return paths


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3: Run experiments
# ═══════════════════════════════════════════════════════════════════════════


def run_experiments(
    config: ResearchLoopConfig,
    pack_paths: List[str],
    case_id: str = "",
) -> List[Dict[str, Any]]:
    """Run experiments for each strategy in the config.

    Returns
    -------
    list
        A list of result summary dicts, each containing strategy_id and
        the portfolio summary (pnl, sharpe, win_rate, etc.).
    """
    from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig

    runner = ExperimentRunner(output_dir=config.experiment.output_dir)

    # Load MC kill thresholds from file if specified
    mc_kill_thresholds: Dict[str, float] = {}
    if config.experiment.mc_kill_thresholds:
        mc_kill_path = Path(config.experiment.mc_kill_thresholds)
        if mc_kill_path.exists():
            with open(mc_kill_path) as f:
                mc_kill_thresholds = dict(yaml.safe_load(f) or {})

    # Case Override Mode
    target_specs = config.strategies
    if case_id:
        from src.agent.pantheon.factory import FactoryPipe
        pipe = FactoryPipe()
        case_data = pipe.get_case(case_id)
        if case_data and case_data["strategy"]["final_manifest"]:
            manifest = json.loads(case_data["strategy"]["final_manifest"])
            logger.info("  [Case Override] Re-running strategy from case_id=%s: %s", case_id, manifest.get("name"))

            # Create a synthetic StrategySpec from the manifest
            # Prometheus manifests never include a "class" key, so we infer from name/objective
            strat_class = manifest.get("class")
            if not strat_class:
                name_lower = (manifest.get("name", "") + " " + manifest.get("objective", "")).lower()
                if any(kw in name_lower for kw in ("vrp", "credit spread", "put spread", "volatility risk")):
                    strat_class = "VRPCreditSpreadStrategy"
                elif any(kw in name_lower for kw in ("dow", "regime timing", "gate")):
                    strat_class = "DowRegimeTimingGateReplayStrategy"
                elif any(kw in name_lower for kw in ("overnight", "session")):
                    strat_class = "OvernightSessionReplayStrategy"
                else:
                    # Last resort: fall back to the first configured strategy class
                    strat_class = (config.strategies[0].strategy_class if config.strategies else "VRPCreditSpreadStrategy")
                    logger.warning("  [Case Override] Could not infer strategy class from manifest name/objective — using '%s'", strat_class)
            
            logger.info("  [Case Override] Resolved strategy class: %s", strat_class)
            from src.analysis.research_loop_config import StrategySpec
            target_specs = [
                StrategySpec(
                    strategy_class=strat_class,
                    params=manifest.get("parameters", {}),
                    sweep=None, # TBD if we want to extract sweep from manifest
                )
            ]
        else:
            logger.warning("  [Case Override] case_id=%s not found or no manifest - falling back to config.", case_id)

    all_results: List[Dict[str, Any]] = []

    for spec in target_specs:
        logger.info("Running experiments for strategy: %s", spec.strategy_class)

        strat_cls = load_strategy_class(spec.strategy_class)

        base_config = ExperimentConfig(
            strategy_class=strat_cls,
            strategy_params=dict(spec.params),
            replay_pack_paths=list(pack_paths),
            starting_cash=config.experiment.starting_cash,
            output_dir=config.experiment.output_dir,
            mc_bootstrap_enabled=config.experiment.mc_bootstrap,
            mc_paths=config.experiment.mc_paths,
            mc_method=config.experiment.mc_method,
            mc_block_size=config.experiment.mc_block_size,
            mc_random_seed=config.experiment.mc_seed,
            mc_ruin_level=config.experiment.mc_ruin_level,
            mc_kill_thresholds=mc_kill_thresholds,
            case_id=case_id,
        )

        if spec.sweep:
            sweep_path = Path(spec.sweep)
            if not sweep_path.exists():
                logger.warning(
                    "Sweep file not found: %s — running single experiment.",
                    spec.sweep,
                )
                result = runner.run(base_config)
                ps = result.portfolio_summary
                logger.info(
                    "Single run complete: %s PnL=%.2f Sharpe=%.2f Win%%=%.1f",
                    result.strategy_id,
                    ps["total_realized_pnl"],
                    ps["sharpe_annualized_proxy"],
                    ps["win_rate"],
                )
                all_results.append({
                    "strategy_id": result.strategy_id,
                    "strategy_class": spec.strategy_class,
                    "portfolio": ps,
                })
            else:
                with open(sweep_path) as f:
                    grid = expand_sweep_grid(yaml.safe_load(f))
                logger.info(
                    "Running parameter sweep from %s", spec.sweep,
                )
                results = runner.run_parameter_grid(strat_cls, base_config, grid)
                logger.info(
                    "Sweep complete: %d runs for %s",
                    len(results), spec.strategy_class,
                )
                for r in results:
                    all_results.append({
                        "strategy_id": r.strategy_id,
                        "strategy_class": spec.strategy_class,
                        "portfolio": r.portfolio_summary,
                    })
        else:
            result = runner.run(base_config)
            ps = result.portfolio_summary
            logger.info(
                "Single run complete: %s PnL=%.2f Sharpe=%.2f Win%%=%.1f",
                result.strategy_id,
                ps["total_realized_pnl"],
                ps["sharpe_annualized_proxy"],
                ps["win_rate"],
            )
            all_results.append({
                "strategy_id": result.strategy_id,
                "strategy_class": spec.strategy_class,
                "portfolio": ps,
            })

        # Regime-stress (once per strategy after all runs)
        if config.experiment.regime_stress:
            _run_regime_stress(runner, pack_paths, strat_cls, spec.params, config)

    return all_results


def _run_regime_stress(
    runner,
    pack_paths: List[str],
    strat_cls,
    strategy_params: Dict[str, Any],
    config: ResearchLoopConfig,
) -> None:
    """Run regime subset stress test for a strategy."""
    try:
        from src.analysis.regime_stress import run_regime_subset_stress
        from src.core.outcome_engine import BarData

        bars: List[BarData] = []
        outcomes: List[Dict[str, Any]] = []
        regimes: List[Dict[str, Any]] = []
        snapshots: List[Dict[str, Any]] = []

        for pack_path in pack_paths:
            pack = runner.load_pack(pack_path)
            for b in pack.get("bars", []):
                bar = BarData(
                    timestamp_ms=b["timestamp_ms"],
                    open=b["open"],
                    high=b["high"],
                    low=b["low"],
                    close=b["close"],
                    volume=b.get("volume", 0),
                )
                if "symbol" in b:
                    object.__setattr__(bar, "symbol", b["symbol"])
                bars.append(bar)
            outcomes.extend(pack.get("outcomes", []))
            regimes.extend(pack.get("regimes", []))
            snapshots.extend(pack.get("snapshots", []))

        if not bars:
            logger.warning("No bars loaded — skipping regime stress.")
            return

        stress = run_regime_subset_stress(
            bars=bars,
            outcomes=outcomes,
            regimes=regimes,
            snapshots=snapshots,
            strategy_class=strat_cls,
            strategy_params=dict(strategy_params),
            starting_cash=config.experiment.starting_cash,
        )

        stress_path = (
            Path(config.experiment.output_dir)
            / f"{strat_cls.__name__}_regime_stress.json"
        )
        stress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stress_path, "w") as f:
            json.dump(stress, f, indent=2)

        logger.info(
            "Regime stress for %s: score=%.3f saved to %s",
            strat_cls.__name__,
            stress.get("stress_score", 0.0),
            stress_path,
        )
    except Exception as exc:
        logger.warning("Regime stress failed for %s: %s", strat_cls.__name__, exc)


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4: Evaluate and persist
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_and_persist(config: ResearchLoopConfig) -> str:
    """Run strategy evaluation and persist rankings, killed list, candidates.

    Returns the rankings output path.
    """
    from src.analysis.strategy_evaluator import StrategyEvaluator

    input_dir = config.evaluation.input_dir

    # Load kill thresholds from file if specified
    kill_thresholds = None
    if config.evaluation.kill_thresholds:
        kt_path = Path(config.evaluation.kill_thresholds)
        if kt_path.exists():
            with open(kt_path) as f:
                kill_thresholds = yaml.safe_load(f)

    evaluator = StrategyEvaluator(
        input_dir=input_dir,
        output_dir=config.evaluation.rankings_output_dir,
        kill_thresholds=kill_thresholds,
    )

    count = evaluator.load_experiments()
    if count == 0:
        logger.warning("No experiment files found in %s", input_dir)
        return ""

    evaluator.evaluate()

    # Save rankings
    rankings_path = evaluator.save_rankings()
    logger.info("Rankings saved to %s (%d experiments)", rankings_path, count)

    # Print summary
    evaluator.print_summary()

    # Save killed list
    if config.evaluation.killed_output_path:
        killed_out = {
            "killed_count": len(evaluator.killed),
            "killed": evaluator.killed,
        }
        killed_path = Path(config.evaluation.killed_output_path)
        killed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(killed_path, "w") as f:
            json.dump(killed_out, f, indent=2)
        logger.info("Killed list written to %s", killed_path)

    # Build and save candidate set
    if config.evaluation.candidate_set_output_path:
        candidates = [
            {
                "run_id": rec.get("run_id", ""),
                "strategy_id": rec.get("strategy_id", ""),
                "strategy_class": rec.get("strategy_class", ""),
                "strategy_params": rec.get("strategy_params", {}),
                "composite_score": rec.get("composite_score", 0.0),
                "metrics": rec.get("metrics", {}),
            }
            for rec in evaluator.rankings
            if not rec.get("killed", False)
        ]
        candidate_set = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        cand_path = Path(config.evaluation.candidate_set_output_path)
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cand_path, "w") as f:
            json.dump(candidate_set, f, indent=2)
        logger.info(
            "Candidate set written to %s (%d candidates)",
            cand_path, len(candidates),
        )

    return rankings_path


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5: Allocate (optional — closes the research → allocation loop)
# ═══════════════════════════════════════════════════════════════════════════


def run_allocation(
    config: ResearchLoopConfig,
    evaluator_rankings: List[Dict[str, Any]],
) -> Optional[str]:
    """Build Forecasts from evaluator rankings, run AllocationEngine, persist.

    Returns the allocations output path, or None if allocation is skipped.
    """
    alloc_opts = config.evaluation.allocation
    output_path = config.evaluation.allocations_output_path

    if alloc_opts is None or output_path is None:
        logger.info("Allocation step skipped (allocation config or output path is null).")
        return None

    from src.analysis.allocation_engine import AllocationEngine, AllocationConfig
    from src.analysis.strategy_registry import StrategyRegistry
    from src.analysis.sizing import Forecast

    # 1. Load registry from rankings (filters by min_dsr / min_composite_score)
    registry = StrategyRegistry()
    registered = registry.load_from_rankings(
        evaluator_rankings,
        min_dsr=config.evaluation.min_dsr,
        min_composite_score=config.evaluation.min_composite_score,
    )

    if registered == 0:
        logger.warning("No candidates passed registry filters — skipping allocation.")
        return None

    # 2. Build Forecasts from registry entries + original ranking data
    rankings_by_id: Dict[str, Dict[str, Any]] = {}
    for rec in evaluator_rankings:
        sid = rec.get("strategy_id", "")
        if sid:
            rankings_by_id[sid] = rec

    forecasts: List[Any] = []
    max_loss_per_contract: Dict[str, float] = {}

    alloc_max_loss_cfg = alloc_opts.max_loss_per_contract
    alloc_default_max_loss: Optional[float] = None
    alloc_by_strategy: Dict[str, float] = {}
    if isinstance(alloc_max_loss_cfg, dict):
        alloc_by_strategy = {str(k): float(v) for k, v in alloc_max_loss_cfg.items()}
    elif alloc_max_loss_cfg is not None:
        alloc_default_max_loss = float(alloc_max_loss_cfg)

    for entry in registry.candidates:
        rec = rankings_by_id.get(entry.strategy_id, {})
        metrics = rec.get("metrics", {})

        # mu: annualized return proxy from total_return_pct
        total_return_pct = metrics.get("total_return_pct", 0.0)
        mu = total_return_pct / 100.0 if total_return_pct else 0.0

        # sigma: use max_drawdown_pct / 2 as proxy if no rolling vol available
        max_dd_pct = metrics.get("max_drawdown_pct", 0.0)
        sigma = max(max_dd_pct / 200.0, 0.01)  # floor to avoid zero

        # edge_score from composite_score
        edge_score = entry.composite_score

        # confidence from regime_sensitivity_score or 0.5 default
        confidence = rec.get("regime_sensitivity_score", 0.5)

        # instrument: derive from strategy_params or default
        params = entry.params or {}
        instrument = params.get("symbol", params.get("underlying", "UNKNOWN"))

        strategy_max_loss = rec.get("max_loss_per_contract")
        if strategy_max_loss is None:
            strategy_max_loss = params.get("max_loss_per_contract")
        if strategy_max_loss is None:
            strategy_max_loss = alloc_by_strategy.get(entry.strategy_id, alloc_default_max_loss)
        if strategy_max_loss is not None:
            try:
                max_loss_per_contract[entry.strategy_id] = float(strategy_max_loss)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid max_loss_per_contract for %s: %r",
                    entry.strategy_id,
                    strategy_max_loss,
                )

        forecast = Forecast(
            strategy_id=entry.strategy_id,
            instrument=instrument,
            mu=mu,
            sigma=sigma,
            edge_score=edge_score,
            cost=0.0,
            confidence=confidence,
            meta={
                "run_id": entry.run_id,
                "dsr": entry.dsr,
                "sharpe": entry.sharpe,
            },
        )
        forecasts.append(forecast)

    logger.info("Built %d forecasts for allocation.", len(forecasts))

    # 3. Run AllocationEngine (with or without RiskEngine)
    alloc_config = AllocationConfig(
        kelly_fraction=alloc_opts.kelly_fraction,
        per_play_cap=alloc_opts.per_play_cap,
        vol_target_annual=alloc_opts.vol_target_annual,
        min_edge_over_cost=alloc_opts.min_edge_over_cost,
    )
    engine = AllocationEngine(config=alloc_config, equity=config.evaluation.equity)

    clamp_reasons_list = []
    risk_attribution_dict = {}
    risk_engine_opts = config.evaluation.risk_engine

    if risk_engine_opts and risk_engine_opts.enabled:
        # Build RiskEngineConfig from parsed opts
        from src.analysis.risk_engine import RiskEngineConfig
        from src.analysis.drawdown_containment import DrawdownConfig
        from src.analysis.correlation_exposure import CorrelationConfig
        from src.analysis.greek_limits import GreekLimitsConfig
        from src.analysis.tail_risk_scenario import TailRiskConfig
        from src.analysis.portfolio_state import PortfolioState, build_portfolio_state_from_context
        from src.analysis.risk_attribution import persist_risk_attribution

        risk_config = RiskEngineConfig(
            enabled=True,
            aggregate_exposure_cap=risk_engine_opts.aggregate_exposure_cap,
            drawdown=DrawdownConfig(
                threshold_pct=risk_engine_opts.drawdown.threshold_pct,
                throttle_mode=risk_engine_opts.drawdown.throttle_mode,
                throttle_scale=risk_engine_opts.drawdown.throttle_scale,
                k=risk_engine_opts.drawdown.k,
                recovery_threshold_pct=risk_engine_opts.drawdown.recovery_threshold_pct,
                min_throttle=risk_engine_opts.drawdown.min_throttle,
            ),
            correlation=CorrelationConfig(
                rolling_days=risk_engine_opts.correlation.rolling_days,
                min_obs=risk_engine_opts.correlation.min_obs,
                estimator=risk_engine_opts.correlation.estimator,
                nan_policy=risk_engine_opts.correlation.nan_policy,
                max_exposure_per_underlying_usd=risk_engine_opts.correlation.max_exposure_per_underlying_usd,
                max_exposure_per_cluster_usd=risk_engine_opts.correlation.max_exposure_per_cluster_usd,
                max_correlated_pair_exposure_usd=risk_engine_opts.correlation.max_correlated_pair_exposure_usd,
                cluster_method=risk_engine_opts.correlation.cluster_method,
                corr_threshold_for_cluster=risk_engine_opts.correlation.corr_threshold_for_cluster,
            ),
            greek_limits=GreekLimitsConfig(
                per_underlying=dict(risk_engine_opts.greek_limits.per_underlying),
                portfolio_max_delta_shares=risk_engine_opts.greek_limits.portfolio_max_delta_shares,
                portfolio_max_vega=risk_engine_opts.greek_limits.portfolio_max_vega,
                portfolio_max_gamma=risk_engine_opts.greek_limits.portfolio_max_gamma,
                enforce_existing_positions=risk_engine_opts.greek_limits.enforce_existing_positions,
            ),
            tail_risk=TailRiskConfig(
                enabled_for_options=risk_engine_opts.tail_risk.enabled_for_options,
                max_prob_touch=risk_engine_opts.tail_risk.max_prob_touch,
                stress_iv_bump=risk_engine_opts.tail_risk.stress_iv_bump,
                max_stress_loss_pct=risk_engine_opts.tail_risk.max_stress_loss_pct,
                mc_simulations=risk_engine_opts.tail_risk.mc_simulations,
                mc_steps_per_year=risk_engine_opts.tail_risk.mc_steps_per_year,
                seed=risk_engine_opts.tail_risk.seed,
                default_heston_kappa=risk_engine_opts.tail_risk.default_heston_kappa,
                default_heston_sigma_v=risk_engine_opts.tail_risk.default_heston_sigma_v,
                default_heston_rho=risk_engine_opts.tail_risk.default_heston_rho,
            ),
            risk_attribution_output_path=risk_engine_opts.risk_attribution_output_path,
            enforce_idempotence_check=risk_engine_opts.enforce_idempotence_check,
            enforce_monotone_check=risk_engine_opts.enforce_monotone_check,
        )

        # Build PortfolioState for the evaluation as_of_ts_ms
        as_of_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        portfolio_state = build_portfolio_state_from_context(
            as_of_ts_ms=as_of_ts_ms,
            equity_usd=config.evaluation.equity,
        )

        allocations, clamp_reasons, risk_attribution = engine.allocate_with_risk_engine(
            forecasts=forecasts,
            portfolio_state=portfolio_state,
            risk_config=risk_config,
            max_loss_per_contract=max_loss_per_contract or None,
        )

        clamp_reasons_list = [cr.to_dict() for cr in clamp_reasons]
        risk_attribution_dict = risk_attribution.to_dict()

        # Persist risk attribution artifact
        if risk_engine_opts.risk_attribution_output_path:
            persist_risk_attribution(
                risk_attribution,
                risk_engine_opts.risk_attribution_output_path,
            )

        logger.info(
            "Risk engine applied: %d clamp reasons, config_hash=%s",
            len(clamp_reasons), risk_config.config_hash(),
        )
    else:
        allocations = engine.allocate(
            forecasts,
            max_loss_per_contract=max_loss_per_contract or None,
        )

    # 4. Persist allocations as JSON
    alloc_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "equity": config.evaluation.equity,
        "config": {
            "kelly_fraction": alloc_opts.kelly_fraction,
            "per_play_cap": alloc_opts.per_play_cap,
            "vol_target_annual": alloc_opts.vol_target_annual,
            "min_edge_over_cost": alloc_opts.min_edge_over_cost,
            "max_loss_per_contract": alloc_opts.max_loss_per_contract,
            "min_dsr": config.evaluation.min_dsr,
            "min_composite_score": config.evaluation.min_composite_score,
        },
        "allocations": [
            {
                "strategy_id": a.strategy_id,
                "instrument": a.instrument,
                "weight": a.weight,
                "dollar_risk": a.dollar_risk,
                "kelly_raw": a.kelly_raw,
                "vol_adjusted": a.vol_adjusted,
                "contracts": a.contracts,
            }
            for a in allocations
        ],
    }

    # Include risk engine artifacts if enabled
    if risk_engine_opts and risk_engine_opts.enabled:
        alloc_output["risk_engine"] = {
            "enabled": True,
            "config_hash": risk_config.config_hash(),
            "clamp_reasons_count": len(clamp_reasons_list),
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(alloc_output, f, indent=2, sort_keys=False)

    # Persist clamp reasons alongside allocations
    if clamp_reasons_list:
        clamp_path = out.parent / "clamp_reasons.json"
        with open(clamp_path, "w") as f:
            json.dump(clamp_reasons_list, f, indent=2)
        logger.info("Clamp reasons written to %s", clamp_path)

    active = sum(1 for a in allocations if a.weight != 0)
    logger.info(
        "Allocations written to %s (%d active / %d total)",
        output_path, active, len(allocations),
    )
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════
#  Cycle orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def run_cycle(
    config: ResearchLoopConfig,
    dry_run: bool = False,
    case_id: str = "",
) -> Optional[List[Dict[str, Any]]]:
    """Execute one full research cycle.

    Steps: outcomes -> packs -> experiments -> evaluate -> allocate.

    Returns
    -------
    list or None
        List of experiment result dicts (strategy_id + portfolio summary)
        for every run in this cycle, or None when the cycle is aborted.
    """
    has_alloc = (
        config.evaluation.allocation is not None
        and config.evaluation.allocations_output_path is not None
    )
    n_steps = 5 if has_alloc else 4

    logger.info(
        "=== Research cycle starting: %s to %s, %d strategies ===",
        config.pack.start_date,
        config.pack.end_date,
        len(config.strategies),
    )

    # Enforce require_recent_bars_hours guard before running cycle
    if config.loop.require_recent_bars_hours is not None and not dry_run:
        if not _check_recent_bars_freshness(
            config.pack.db_path, config.loop.require_recent_bars_hours
        ):
            logger.warning(
                "Skipping cycle: most recent bars in %s are older than "
                "require_recent_bars_hours=%.1f — data may be stale.",
                config.pack.db_path,
                config.loop.require_recent_bars_hours,
            )
            return None

    if dry_run:
        logger.info("[DRY RUN] Config validated. Steps that would run:")
        if config.outcomes.ensure_before_pack:
            logger.info("  1. Outcomes backfill (%s, %s to %s)",
                        config.pack.mode, config.pack.start_date,
                        config.pack.end_date)
        else:
            logger.info("  1. Outcomes backfill: SKIPPED")
        logger.info("  2. Build packs: mode=%s symbols=%s",
                     config.pack.mode, config.pack.symbols)
        for s in config.strategies:
            logger.info("  3. Experiment: %s sweep=%s",
                         s.strategy_class, s.sweep or "none")
        logger.info("  4. Evaluate: input=%s", config.evaluation.input_dir)
        if has_alloc:
            logger.info("  5. Allocate: output=%s", config.evaluation.allocations_output_path)
        logger.info("[DRY RUN] No actions taken.")
        return []

    # Step 1: Outcomes backfill
    logger.info("--- Step 1/%d: Outcomes backfill ---", n_steps)
    run_outcomes_backfill(config)

    # Step 2: Build packs
    logger.info("--- Step 2/%d: Build replay packs ---", n_steps)
    pack_paths = build_packs(config)
    if not pack_paths:
        logger.error("No packs built — aborting cycle.")
        return None

    # Step 3: Run experiments
    logger.info("--- Step 3/%d: Run experiments ---", n_steps)
    experiment_results = run_experiments(config, pack_paths, case_id=case_id)

    # Step 4: Evaluate and persist
    logger.info("--- Step 4/%d: Evaluate and persist ---", n_steps)
    rankings_path = evaluate_and_persist(config)

    # Step 5: Allocate (optional — only if allocation config + output path set)
    alloc_path = None
    if has_alloc and rankings_path:
        logger.info("--- Step 5/%d: Allocate ---", n_steps)
        # Load the rankings we just saved to get the evaluator output
        with open(rankings_path) as f:
            rankings_data = json.load(f)
        alloc_path = run_allocation(config, rankings_data.get("rankings", []))

    logger.info(
        "=== Research cycle complete. Rankings: %s | Allocations: %s ===",
        rankings_path, alloc_path or "skipped",
    )

    return experiment_results



# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Argus Strategy Research Loop — run the full "
                    "research cycle: outcomes -> packs -> experiments -> evaluation.",
    )
    parser.add_argument(
        "--config",
        default="config/research_loop.yaml",
        help="Path to research loop YAML config "
             "(default: config/research_loop.yaml)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run a single cycle and exit (default: daemon mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and log steps without executing anything",
    )

    args = parser.parse_args()

    # Load and validate config
    try:
        config = load_research_loop_config(args.config)
    except (FileNotFoundError, ConfigValidationError) as exc:
        logger.error("Config error: %s", exc)
        sys.exit(1)

    if args.once or args.dry_run:
        try:
            run_cycle(config, dry_run=args.dry_run)
        except Exception:
            logger.exception("Research cycle failed.")
            sys.exit(1)
    else:
        # Daemon mode
        logger.info(
            "Starting daemon mode (interval=%.1fh).",
            config.loop.interval_hours,
        )
        while True:
            try:
                # Re-load config each cycle to pick up changes
                config = load_research_loop_config(args.config)
                run_cycle(config)
            except (FileNotFoundError, ConfigValidationError) as exc:
                logger.error("Config error (will retry next cycle): %s", exc)
            except Exception:
                logger.exception("Research cycle failed (will retry next cycle).")

            sleep_seconds = config.loop.interval_hours * 3600
            logger.info(
                "Sleeping %.1f hours until next cycle...",
                config.loop.interval_hours,
            )
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
