"""
Research Loop Configuration
============================

Loads, validates, and resolves the YAML config for the Strategy Research Loop.

The config schema is documented in ``config/research_loop.example.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger("argus.research_loop_config")


class ConfigValidationError(Exception):
    """Raised when the research loop config is invalid."""


@dataclass
class PackConfig:
    mode: str  # "single" or "universe"
    symbols: List[str]
    start_date: str  # resolved YYYY-MM-DD
    end_date: str  # resolved YYYY-MM-DD
    bars_provider: Optional[str]
    options_snapshot_provider: Optional[str]
    packs_output_dir: str
    db_path: str


@dataclass
class OutcomesConfig:
    ensure_before_pack: bool
    bar_duration: int


@dataclass
class StrategySpec:
    strategy_class: str
    params: Dict[str, Any]
    sweep: Optional[str]  # path to YAML param grid


@dataclass
class ExperimentOpts:
    output_dir: str
    starting_cash: float
    regime_stress: bool
    mc_bootstrap: bool
    mc_paths: int
    mc_method: str
    mc_block_size: Optional[int]
    mc_seed: Optional[int]
    mc_ruin_level: float
    mc_kill_thresholds: Optional[str]  # path to YAML or None


@dataclass
class DeployGatesOpts:
    """Configuration for Phase 4C deploy gates."""
    dsr_min: float = 0.95
    dsr_trials_mode: str = "count"  # "count" or "clustered"
    slippage_sweep: bool = True
    slippage_sweep_kill_if_edge_disappears: bool = True
    reality_check_benchmark: Optional[str] = "buy_and_hold"  # or None to skip
    reality_check_p_max: float = 0.05


@dataclass
class AllocationOpts:
    """Configuration for Phase 5 allocation engine."""
    kelly_fraction: float = 0.25
    per_play_cap: float = 0.07
    vol_target_annual: Optional[float] = 0.10
    min_edge_over_cost: float = 0.0
    max_loss_per_contract: Optional[Union[float, Dict[str, float]]] = None


@dataclass
class RiskEngineDrawdownOpts:
    """Drawdown containment config within risk engine."""
    threshold_pct: float = 0.10
    throttle_mode: str = "linear"
    throttle_scale: float = 0.5
    k: float = 5.0
    recovery_threshold_pct: float = 0.05
    min_throttle: float = 0.1


@dataclass
class RiskEngineCorrelationOpts:
    """Correlation exposure config within risk engine."""
    rolling_days: int = 60
    min_obs: int = 45
    estimator: str = "pearson"
    nan_policy: str = "skip"
    max_exposure_per_underlying_usd: float = 0.0
    max_exposure_per_cluster_usd: float = 0.0
    max_correlated_pair_exposure_usd: float = 0.0
    cluster_method: str = "threshold"
    corr_threshold_for_cluster: float = 0.8


@dataclass
class RiskEngineGreekLimitsOpts:
    """Greek limits config within risk engine."""
    per_underlying: Dict[str, Dict[str, float]] = field(default_factory=dict)
    portfolio_max_delta_shares: float = float("inf")
    portfolio_max_vega: float = float("inf")
    portfolio_max_gamma: float = float("inf")
    enforce_existing_positions: bool = True


@dataclass
class RiskEngineTailRiskOpts:
    """Tail-risk scenario config within risk engine."""
    enabled_for_options: bool = True
    max_prob_touch: float = 0.35
    stress_iv_bump: float = 0.20
    max_stress_loss_pct: float = 0.05
    mc_simulations: int = 100_000
    mc_steps_per_year: int = 252
    seed: int = 42
    default_heston_kappa: float = 2.0
    default_heston_sigma_v: float = 0.5
    default_heston_rho: float = -0.7


@dataclass
class RiskEngineOpts:
    """Configuration for Phase 5 Portfolio Risk Engine.

    Nested under ``evaluation.allocation.risk_engine`` in YAML.
    """
    enabled: bool = False
    aggregate_exposure_cap: float = 1.0
    drawdown: RiskEngineDrawdownOpts = field(default_factory=RiskEngineDrawdownOpts)
    correlation: RiskEngineCorrelationOpts = field(default_factory=RiskEngineCorrelationOpts)
    greek_limits: RiskEngineGreekLimitsOpts = field(default_factory=RiskEngineGreekLimitsOpts)
    tail_risk: RiskEngineTailRiskOpts = field(default_factory=RiskEngineTailRiskOpts)
    risk_attribution_output_path: str = "logs/risk_attribution.json"
    enforce_idempotence_check: bool = True
    enforce_monotone_check: bool = True


@dataclass
class EvaluationOpts:
    input_dir: str
    kill_thresholds: Optional[str]  # path to YAML or None
    rankings_output_dir: str
    killed_output_path: Optional[str]
    candidate_set_output_path: Optional[str]
    deploy_gates: Optional[DeployGatesOpts] = None
    allocation: Optional[AllocationOpts] = None
    allocations_output_path: Optional[str] = None
    risk_engine: Optional[RiskEngineOpts] = None
    equity: float = 10_000.0
    min_dsr: float = 0.0
    min_composite_score: float = 0.0


@dataclass
class LoopOpts:
    interval_hours: float
    require_recent_bars_hours: Optional[float]


@dataclass
class ResearchLoopConfig:
    pack: PackConfig
    outcomes: OutcomesConfig
    strategies: List[StrategySpec]
    experiment: ExperimentOpts
    evaluation: EvaluationOpts
    loop: LoopOpts


def _resolve_path(raw: Optional[str], project_root: Path) -> Optional[str]:
    """Resolve a path relative to project root, or return None."""
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str(project_root / p)


def load_research_loop_config(
    config_path: str,
    project_root: Optional[Path] = None,
) -> ResearchLoopConfig:
    """Load and validate a research loop YAML config.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    project_root : Path, optional
        Project root for resolving relative paths.  Defaults to
        two levels up from this file (``src/analysis/`` -> repo root).

    Returns
    -------
    ResearchLoopConfig

    Raises
    ------
    ConfigValidationError
        If the config is missing required fields or has invalid values.
    FileNotFoundError
        If the config file does not exist.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # ── Pack ───────────────────────────────────────────────────────────
    pack_raw = raw.get("pack", {})
    mode = pack_raw.get("mode", "single")
    if mode not in ("single", "universe"):
        raise ConfigValidationError(
            f"pack.mode must be 'single' or 'universe', got '{mode}'"
        )

    symbols = pack_raw.get("symbols") or []
    if mode == "single" and not symbols:
        raise ConfigValidationError(
            "pack.symbols must be a non-empty list when pack.mode is 'single'"
        )

    # Resolve dates
    last_n_days = pack_raw.get("last_n_days")
    if last_n_days is not None:
        today = datetime.now(timezone.utc).date()
        end_date = today.strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=int(last_n_days))).strftime("%Y-%m-%d")
    else:
        start_date = pack_raw.get("start_date")
        end_date = pack_raw.get("end_date")
        if not start_date or not end_date:
            raise ConfigValidationError(
                "Either pack.last_n_days or both pack.start_date and "
                "pack.end_date must be set"
            )

    # Validate date format
    for label, val in [("start_date", start_date), ("end_date", end_date)]:
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            raise ConfigValidationError(
                f"pack.{label} must be YYYY-MM-DD, got '{val}'"
            )

    pack = PackConfig(
        mode=mode,
        symbols=list(symbols),
        start_date=start_date,
        end_date=end_date,
        bars_provider=pack_raw.get("bars_provider"),
        options_snapshot_provider=pack_raw.get("options_snapshot_provider"),
        packs_output_dir=_resolve_path(
            pack_raw.get("packs_output_dir", "data/packs"), project_root
        ),
        db_path=_resolve_path(
            pack_raw.get("db_path", "data/argus.db"), project_root
        ),
    )

    # ── Outcomes ───────────────────────────────────────────────────────
    outcomes_raw = raw.get("outcomes", {})
    outcomes = OutcomesConfig(
        ensure_before_pack=outcomes_raw.get("ensure_before_pack", True),
        bar_duration=int(outcomes_raw.get("bar_duration", 60)),
    )

    # ── Strategies ─────────────────────────────────────────────────────
    strats_raw = raw.get("strategies", [])
    if not strats_raw:
        raise ConfigValidationError(
            "strategies must have at least one entry"
        )

    strategies: List[StrategySpec] = []
    for i, s in enumerate(strats_raw):
        cls_name = s.get("strategy_class")
        if not cls_name:
            raise ConfigValidationError(
                f"strategies[{i}].strategy_class is required"
            )
        sweep = s.get("sweep")
        if sweep:
            sweep = _resolve_path(sweep, project_root)
        strategies.append(StrategySpec(
            strategy_class=cls_name,
            params=s.get("params") or {},
            sweep=sweep,
        ))

    # ── Experiment ─────────────────────────────────────────────────────
    exp_raw = raw.get("experiment", {})
    mc_kill_path = exp_raw.get("mc_kill_thresholds")
    if mc_kill_path:
        mc_kill_path = _resolve_path(mc_kill_path, project_root)

    experiment = ExperimentOpts(
        output_dir=_resolve_path(
            exp_raw.get("output_dir", "logs/experiments"), project_root
        ),
        starting_cash=float(exp_raw.get("starting_cash", 10000.0)),
        regime_stress=bool(exp_raw.get("regime_stress", True)),
        mc_bootstrap=bool(exp_raw.get("mc_bootstrap", True)),
        mc_paths=int(exp_raw.get("mc_paths", 1000)),
        mc_method=exp_raw.get("mc_method", "bootstrap"),
        mc_block_size=exp_raw.get("mc_block_size"),
        mc_seed=exp_raw.get("mc_seed", 42),
        mc_ruin_level=float(exp_raw.get("mc_ruin_level", 0.2)),
        mc_kill_thresholds=mc_kill_path,
    )

    # ── Evaluation ─────────────────────────────────────────────────────
    eval_raw = raw.get("evaluation", {})
    eval_input = eval_raw.get("input_dir")
    if eval_input:
        eval_input = _resolve_path(eval_input, project_root)
    else:
        eval_input = experiment.output_dir

    kill_thresh_path = eval_raw.get("kill_thresholds")
    if kill_thresh_path:
        kill_thresh_path = _resolve_path(kill_thresh_path, project_root)

    # ── Deploy gates (Phase 4C) ──────────────────────────────────
    gates_raw = eval_raw.get("deploy_gates", {})
    deploy_gates = None
    if gates_raw:
        deploy_gates = DeployGatesOpts(
            dsr_min=float(gates_raw.get("dsr_min", 0.95)),
            dsr_trials_mode=gates_raw.get("dsr_trials_mode", "count"),
            slippage_sweep=bool(gates_raw.get("slippage_sweep", True)),
            slippage_sweep_kill_if_edge_disappears=bool(
                gates_raw.get("slippage_sweep_kill_if_edge_disappears", True)
            ),
            reality_check_benchmark=gates_raw.get("reality_check_benchmark", "buy_and_hold"),
            reality_check_p_max=float(gates_raw.get("reality_check_p_max", 0.05)),
        )

    # ── Allocation (Phase 5 prelude) ──────────────────────────────
    alloc_raw = eval_raw.get("allocation") or raw.get("allocation", {})
    allocation = None
    if alloc_raw:
        # vol_target_annual: missing -> 0.10; explicit null -> None (no overlay)
        vol_raw = alloc_raw.get("vol_target_annual")
        if vol_raw is None and "vol_target_annual" in alloc_raw:
            vol_target_annual = None
        else:
            vol_target_annual = float(vol_raw if vol_raw is not None else 0.10)
        max_loss_cfg = alloc_raw.get("max_loss_per_contract")
        if isinstance(max_loss_cfg, dict):
            max_loss_cfg = {str(k): float(v) for k, v in max_loss_cfg.items()}
        elif max_loss_cfg is not None:
            max_loss_cfg = float(max_loss_cfg)

        allocation = AllocationOpts(
            kelly_fraction=float(alloc_raw.get("kelly_fraction", 0.25)),
            per_play_cap=float(alloc_raw.get("per_play_cap", 0.07)),
            vol_target_annual=vol_target_annual,
            min_edge_over_cost=float(alloc_raw.get("min_edge_over_cost", 0.0)),
            max_loss_per_contract=max_loss_cfg,
        )

    # ── Risk Engine (Phase 5 full) ──────────────────────────────
    risk_engine = None
    re_raw = eval_raw.get("risk_engine") or (alloc_raw or {}).get("risk_engine", {})
    if re_raw:
        dd_raw = re_raw.get("drawdown", {})
        corr_raw = re_raw.get("correlation", {})
        gl_raw = re_raw.get("greek_limits", {})
        tr_raw = re_raw.get("tail_risk", {})
        policy_raw = re_raw.get("policy", {})

        drawdown_opts = RiskEngineDrawdownOpts(
            threshold_pct=float(dd_raw.get("threshold_pct", 0.10)),
            throttle_mode=dd_raw.get("throttle_mode", "linear"),
            throttle_scale=float(dd_raw.get("throttle_scale", 0.5)),
            k=float(dd_raw.get("k", 5.0)),
            recovery_threshold_pct=float(dd_raw.get("recovery_threshold_pct", 0.05)),
            min_throttle=float(dd_raw.get("min_throttle", 0.1)),
        )

        corr_opts = RiskEngineCorrelationOpts(
            rolling_days=int(corr_raw.get("rolling_days", 60)),
            min_obs=int(corr_raw.get("min_obs", 45)),
            estimator=corr_raw.get("estimator", "pearson"),
            nan_policy=corr_raw.get("nan_policy", "skip"),
            max_exposure_per_underlying_usd=float(
                corr_raw.get("max_exposure_per_underlying_usd", 0.0)
            ),
            max_exposure_per_cluster_usd=float(
                corr_raw.get("max_exposure_per_cluster_usd", 0.0)
            ),
            max_correlated_pair_exposure_usd=float(
                corr_raw.get("max_correlated_pair_exposure_usd", 0.0)
            ),
            cluster_method=corr_raw.get("cluster_method", "threshold"),
            corr_threshold_for_cluster=float(
                corr_raw.get("corr_threshold_for_cluster", 0.8)
            ),
        )

        # Parse per_underlying greek limits
        per_und_raw = gl_raw.get("per_underlying", {})
        per_underlying = {}
        for und, limits in per_und_raw.items():
            per_underlying[str(und)] = {
                str(k): float(v) for k, v in (limits or {}).items()
            }

        gl_opts = RiskEngineGreekLimitsOpts(
            per_underlying=per_underlying,
            portfolio_max_delta_shares=float(
                gl_raw.get("portfolio_max_delta_shares", float("inf"))
            ),
            portfolio_max_vega=float(
                gl_raw.get("portfolio_max_vega", float("inf"))
            ),
            portfolio_max_gamma=float(
                gl_raw.get("portfolio_max_gamma", float("inf"))
            ),
            enforce_existing_positions=bool(
                gl_raw.get("enforce_existing_positions", True)
            ),
        )

        tr_opts = RiskEngineTailRiskOpts(
            enabled_for_options=bool(tr_raw.get("enabled_for_options", True)),
            max_prob_touch=float(tr_raw.get("max_prob_touch", 0.35)),
            stress_iv_bump=float(tr_raw.get("stress_iv_bump", 0.20)),
            max_stress_loss_pct=float(tr_raw.get("max_stress_loss_pct", 0.05)),
            mc_simulations=int(tr_raw.get("mc_simulations", 100_000)),
            mc_steps_per_year=int(tr_raw.get("mc_steps_per_year", 252)),
            seed=int(tr_raw.get("seed", 42)),
            default_heston_kappa=float(tr_raw.get("default_heston_kappa", 2.0)),
            default_heston_sigma_v=float(tr_raw.get("default_heston_sigma_v", 0.5)),
            default_heston_rho=float(tr_raw.get("default_heston_rho", -0.7)),
        )

        risk_engine = RiskEngineOpts(
            enabled=bool(re_raw.get("enabled", False)),
            aggregate_exposure_cap=float(re_raw.get("aggregate_exposure_cap", 1.0)),
            drawdown=drawdown_opts,
            correlation=corr_opts,
            greek_limits=gl_opts,
            tail_risk=tr_opts,
            risk_attribution_output_path=_resolve_path(
                re_raw.get("risk_attribution_output_path", "logs/risk_attribution.json"),
                project_root,
            ),
            enforce_idempotence_check=bool(
                policy_raw.get("enforce_idempotence_check", True)
            ),
            enforce_monotone_check=bool(
                policy_raw.get("enforce_monotone_check", True)
            ),
        )

    evaluation = EvaluationOpts(
        input_dir=eval_input,
        kill_thresholds=kill_thresh_path,
        rankings_output_dir=_resolve_path(
            eval_raw.get("rankings_output_dir", "logs"), project_root
        ),
        killed_output_path=_resolve_path(
            eval_raw.get("killed_output_path"), project_root
        ),
        candidate_set_output_path=_resolve_path(
            eval_raw.get("candidate_set_output_path"), project_root
        ),
        deploy_gates=deploy_gates,
        allocation=allocation,
        allocations_output_path=_resolve_path(
            eval_raw.get("allocations_output_path"), project_root
        ),
        risk_engine=risk_engine,
        equity=float(eval_raw.get("equity", 10_000.0)),
        min_dsr=float(eval_raw.get("min_dsr", 0.0)),
        min_composite_score=float(eval_raw.get("min_composite_score", 0.0)),
    )

    # ── Loop ───────────────────────────────────────────────────────────
    loop_raw = raw.get("loop", {})
    loop = LoopOpts(
        interval_hours=float(loop_raw.get("interval_hours", 24)),
        require_recent_bars_hours=loop_raw.get("require_recent_bars_hours"),
    )

    config = ResearchLoopConfig(
        pack=pack,
        outcomes=outcomes,
        strategies=strategies,
        experiment=experiment,
        evaluation=evaluation,
        loop=loop,
    )

    logger.info(
        "Research loop config loaded: mode=%s, symbols=%s, "
        "date_range=%s to %s, strategies=%d",
        pack.mode,
        pack.symbols,
        pack.start_date,
        pack.end_date,
        len(strategies),
    )

    return config
