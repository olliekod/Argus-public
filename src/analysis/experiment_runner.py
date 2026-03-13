"""
Experiment Runner
=================

Orchestrates strategy evaluation over Replay Packs.
Provides deterministic walk-forward splitting and parameter sweeps.
"""

import json
import logging
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Type, Optional, Generator

from src.analysis.replay_harness import (
    MarketDataSnapshot,
    ReplayHarness,
    ReplayConfig,
    ReplayStrategy,
    ReplayResult,
)
from src.strategies.vrp_credit_spread import _derive_iv_from_quotes
from src.tools.replay_pack import _atm_iv_from_quotes_json
from src.analysis.execution_model import ExecutionModel, ExecutionConfig
from src.core.outcome_engine import BarData
from src.core.data_sources import get_data_source_policy
from src.analysis.mc_bootstrap import run_mc_paths, evaluate_mc_kill
from src.analysis.reality_check import reality_check as run_reality_check

logger = logging.getLogger("argus.experiment_runner")

@dataclass
class ExperimentConfig:
    """Settings for a single experiment run."""
    strategy_class: Type[ReplayStrategy]
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    replay_pack_paths: List[str] = field(default_factory=list)
    starting_cash: float = 10000.0
    execution_config: Optional[ExecutionConfig] = None
    output_dir: str = "logs/experiments"
    tag: str = "default"
    mc_bootstrap_enabled: bool = False
    mc_paths: int = 1000
    mc_method: str = "bootstrap"
    mc_block_size: Optional[int] = None
    mc_random_seed: Optional[int] = 42
    mc_ruin_level: float = 0.2
    mc_kill_thresholds: Dict[str, float] = field(default_factory=dict)
    case_id: str = ""  # Pantheon research case ID for attribution

class ExperimentRunner:
    """Standardized handler for strategy research experiments."""

    def __init__(self, output_dir: str = "logs/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_pack(self, path: str) -> Dict[str, Any]:
        """Load a Replay Pack JSON file."""
        with open(path, "r") as f:
            pack = json.load(f)
        return pack

    @staticmethod
    def _float_or_none(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pack_snapshots_to_objects(snapshot_dicts: List[Dict[str, Any]]) -> List[MarketDataSnapshot]:
        """Convert replay pack snapshot dicts to MarketDataSnapshot for ReplayHarness.

        Passes quotes_json so strategies can derive IV from bid/ask when atm_iv is absent.
        When pack atm_iv is None/0, attempts derivation so VRP and similar strategies get IV.
        """
        out: List[MarketDataSnapshot] = []
        for s in snapshot_dicts or []:
            recv_ts = s.get("recv_ts_ms")
            if recv_ts is None:
                recv_ts = s.get("timestamp_ms", 0)
            try:
                recv_ts = int(recv_ts)
            except (TypeError, ValueError):
                recv_ts = 0
            qj = s.get("quotes_json")
            if qj is not None and not isinstance(qj, str):
                qj = json.dumps(qj)
            atm_iv = ExperimentRunner._float_or_none(s.get("atm_iv"))
            underlying = float(s.get("underlying_price", 0.0) or 0.0)
            if underlying <= 0 and qj:
                try:
                    data = json.loads(qj) if isinstance(qj, str) else qj
                    underlying = float(data.get("underlying_price") or 0.0)
                except (TypeError, ValueError, Exception):
                    pass
            if (atm_iv is None or atm_iv <= 0) and qj:
                derived = _derive_iv_from_quotes(s)
                if derived is not None and derived > 0:
                    atm_iv = derived
                if (atm_iv is None or atm_iv <= 0) and underlying > 0:
                    fallback_iv = _atm_iv_from_quotes_json(qj, underlying)
                    if fallback_iv is not None and fallback_iv > 0:
                        atm_iv = fallback_iv
            out.append(
                MarketDataSnapshot(
                    symbol=s.get("symbol", ""),
                    recv_ts_ms=recv_ts,
                    underlying_price=underlying,
                    atm_iv=atm_iv,
                    source=s.get("provider", ""),
                    quotes_json=qj,
                )
            )
        return out

    def run(self, config: ExperimentConfig) -> ReplayResult:
        """Run a single experiment configuration."""
        # 1. Prepare data from packs
        all_bars: List[BarData] = []
        all_outcomes: List[Dict[str, Any]] = []
        all_regimes: List[Dict[str, Any]] = []
        all_snapshots: List[Any] = []
        
        for pack_path in config.replay_pack_paths:
            pack = self.load_pack(pack_path)
            # Reconstruct BarData objects
            for b in pack.get("bars", []):
                bar = BarData(
                    timestamp_ms=b["timestamp_ms"],
                    open=b["open"], high=b["high"], low=b["low"], 
                    close=b["close"], volume=b.get("volume", 0)
                )
                # Apply symbol if present
                if "symbol" in b:
                    object.__setattr__(bar, 'symbol', b["symbol"])
                all_bars.append(bar)
            
            all_outcomes.extend(pack.get("outcomes", []))
            all_regimes.extend(pack.get("regimes", []))
            all_snapshots.extend(pack.get("snapshots", []))

        snapshots_objs = self._pack_snapshots_to_objects(all_snapshots)
        n_with_iv = sum(1 for s in snapshots_objs if getattr(s, "atm_iv", None) is not None and s.atm_iv > 0)
        if snapshots_objs and n_with_iv == 0:
            print("WARNING: Pack has %d snapshots but 0 have atm_iv on MarketDataSnapshot (check pack→object conversion)." % len(snapshots_objs))

        # Align bars with snapshot window: only keep bars whose sim_time (bar close) can
        # release at least one snapshot. Packs often have bars from midnight while
        # snapshots start at market open, so without this we'd process hundreds of bars
        # with 0 visible snapshots and never release the IV-heavy ones.
        bar_duration_ms = 60 * 1000  # ReplayConfig default
        if snapshots_objs:
            min_recv = min(s.recv_ts_ms for s in snapshots_objs)
            n_bars_before = len(all_bars)
            all_bars = [b for b in all_bars if (b.timestamp_ms + bar_duration_ms) >= min_recv]
            if not all_bars:
                print("WARNING: No bars after aligning to snapshot window (min recv_ts_ms=%s). Check pack bar/snapshot times." % min_recv)
            elif n_bars_before != len(all_bars):
                print("Replay: using %d bars (aligned to snapshot window; dropped %d bars before first snapshot)." % (len(all_bars), n_bars_before - len(all_bars)))

        # Warn if pack is missing data many strategies need (e.g. VRP needs outcomes + snapshots)
        if not all_outcomes:
            print("WARNING: Pack has 0 outcomes. Strategies that need realized_vol (e.g. VRPCreditSpreadStrategy) will never signal. Run: python -m src.outcomes backfill --provider <source> --symbol <sym> --bar 60 --start YYYY-MM-DD --end YYYY-MM-DD then re-pack.")
        if not snapshots_objs:
            print("WARNING: Pack has 0 option snapshots. Strategies that need IV/options will never signal.")

        # 2. Instantiate Strategy
        strategy = config.strategy_class(config.strategy_params)
        
        # 3. Setup Replay
        exec_model = ExecutionModel(config.execution_config)
        replay_cfg = ReplayConfig(starting_cash=config.starting_cash)
        
        harness = ReplayHarness(
            bars=all_bars,
            outcomes=all_outcomes,
            strategy=strategy,
            execution_model=exec_model,
            regimes=all_regimes,
            snapshots=snapshots_objs,
            config=replay_cfg
        )
        
        # 4. Execute
        result = harness.run()

        mc_bootstrap = None
        if config.mc_bootstrap_enabled:
            mc_bootstrap = self._compute_mc_bootstrap(config, result.trade_pnls)

        # Reality Check (White's SPA test)
        rc_result = None
        if result.trade_pnls and len(result.trade_pnls) >= 2:
            try:
                rc_result = run_reality_check(
                    strategy_returns={result.strategy_id: result.trade_pnls},
                    n_bootstrap=1000,
                    seed=config.mc_random_seed,
                )
            except Exception as e:
                logger.warning("Reality check computation failed: %s", e)

        # 5. Save Artifact
        manifest_overrides = {}
        if mc_bootstrap is not None:
            manifest_overrides["mc_bootstrap"] = mc_bootstrap
        if rc_result is not None:
            manifest_overrides["reality_check"] = rc_result
        manifest_overrides = manifest_overrides or None
        self._save_result(config, result, manifest_overrides=manifest_overrides)
        
        return result

    def run_parameter_grid(self, strategy_cls: Type[ReplayStrategy], 
                           base_config: ExperimentConfig, 
                           param_grid: Dict[str, List[Any]]) -> List[ReplayResult]:
        """Run a parameter sweep/grid search."""
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        results = []
        print(f"Starting sweep: {len(combinations)} combinations.")
        for i, params in enumerate(combinations):
            # Merge base params with sweep entry so sweeps that vary only a subset
            # of parameters inherit the rest from spec.params.
            merged_params = {**base_config.strategy_params, **params}
            print(f"Sweep {i+1}/{len(combinations)}: {merged_params}")
            config = ExperimentConfig(
                strategy_class=strategy_cls,
                strategy_params=merged_params,
                replay_pack_paths=base_config.replay_pack_paths,
                starting_cash=base_config.starting_cash,
                execution_config=base_config.execution_config,
                tag=f"sweep_{i}",
                mc_bootstrap_enabled=base_config.mc_bootstrap_enabled,
                mc_paths=base_config.mc_paths,
                mc_method=base_config.mc_method,
                mc_block_size=base_config.mc_block_size,
                mc_random_seed=base_config.mc_random_seed,
                mc_ruin_level=base_config.mc_ruin_level,
                mc_kill_thresholds=dict(base_config.mc_kill_thresholds),
            )
            results.append(self.run(config))
        return results

    def run_cost_sensitivity_sweep(
        self,
        config: ExperimentConfig,
        cost_multipliers: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run experiments at multiple cost multipliers.

        Tests whether a strategy's edge survives reasonable cost
        inflation.  Kills if edge disappears at +50% costs.

        Parameters
        ----------
        config : ExperimentConfig
            Base experiment configuration.
        cost_multipliers : list of float, optional
            Cost multipliers to test (default [1.0, 1.25, 1.50]).

        Returns
        -------
        dict
            ``killed``: bool, ``results``: list of per-multiplier dicts,
            ``sharpe_at_100pct``, ``sharpe_at_125pct``, ``sharpe_at_150pct``,
            ``mean_return_at_150pct``.
        """
        if cost_multipliers is None:
            cost_multipliers = [1.0, 1.25, 1.50]

        base_exec_config = config.execution_config or ExecutionConfig()
        results = []
        baseline_strategy_id = None

        for cm in cost_multipliers:
            # Create a new ExecutionConfig with scaled costs
            scaled_config = ExecutionConfig(
                slippage_per_contract=base_exec_config.slippage_per_contract,
                min_bid_size=base_exec_config.min_bid_size,
                min_ask_size=base_exec_config.min_ask_size,
                max_stale_ms=base_exec_config.max_stale_ms,
                max_spread_pct=base_exec_config.max_spread_pct,
                commission_per_contract=base_exec_config.commission_per_contract,
                allow_partial_fills=base_exec_config.allow_partial_fills,
                cost_multiplier=cm,
            )

            sweep_config = ExperimentConfig(
                strategy_class=config.strategy_class,
                strategy_params=dict(config.strategy_params),
                replay_pack_paths=list(config.replay_pack_paths),
                starting_cash=config.starting_cash,
                execution_config=scaled_config,
                output_dir=config.output_dir,
                tag=f"cost_{int(cm * 100)}pct",
                mc_bootstrap_enabled=False,
            )

            result = self.run(sweep_config)
            if baseline_strategy_id is None:
                baseline_strategy_id = result.strategy_id
            summary = result.summary()
            portfolio = summary.get("portfolio", {})

            results.append({
                "cost_multiplier": cm,
                "sharpe": portfolio.get("sharpe_annualized_proxy", 0.0),
                "total_pnl": portfolio.get("total_realized_pnl", 0.0),
                "total_return_pct": portfolio.get("total_return_pct", 0.0),
                "mean_return_per_trade": (
                    portfolio.get("total_realized_pnl", 0.0)
                    / max(portfolio.get("total_trades", 1), 1)
                ),
                "total_trades": portfolio.get("total_trades", 0),
            })

        # Evaluate: kill if edge disappears at +50%
        sharpe_at_150 = 0.0
        mean_ret_at_150 = 0.0
        sharpe_at_100 = 0.0
        sharpe_at_125 = 0.0

        for r in results:
            if r["cost_multiplier"] == 1.0:
                sharpe_at_100 = r["sharpe"]
            elif r["cost_multiplier"] == 1.25:
                sharpe_at_125 = r["sharpe"]
            elif r["cost_multiplier"] == 1.50:
                sharpe_at_150 = r["sharpe"]
                mean_ret_at_150 = r["mean_return_per_trade"]

        killed = sharpe_at_150 < 0 or mean_ret_at_150 <= 0

        sweep_summary = {
            "killed": killed,
            "results": results,
            "sharpe_at_100pct": sharpe_at_100,
            "sharpe_at_125pct": sharpe_at_125,
            "sharpe_at_150pct": sharpe_at_150,
            "mean_return_at_150pct": mean_ret_at_150,
        }

        # Merge slippage_sensitivity into the baseline (1x) artifact so the
        # evaluator can apply the deploy gate when loading experiments.
        # Remove the 125% and 150% artifacts so the evaluator sees one experiment
        # per strategy (the 1x run with the gate result in the manifest).
        output_path = Path(config.output_dir)
        if results and baseline_strategy_id:
            baseline_pattern = f"{baseline_strategy_id}_cost_100pct_*.json"
            candidates = list(output_path.glob(baseline_pattern))
            if len(candidates) == 1:
                path = candidates[0]
                try:
                    with open(path, "r") as f:
                        artifact = json.load(f)
                    artifact.setdefault("manifest", {})["slippage_sensitivity"] = sweep_summary
                    with open(path, "w") as f:
                        json.dump(artifact, f, indent=2)
                    logger.info("Updated %s with slippage_sensitivity", path)
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning("Could not update baseline artifact with slippage_sensitivity: %s", e)
            else:
                logger.warning(
                    "Expected one baseline artifact matching %s, found %d",
                    baseline_pattern, len(candidates),
                )
            # Remove 125% and 150% artifacts so evaluator gets one record per strategy
            for tag in ("cost_125pct", "cost_150pct"):
                for p in output_path.glob(f"{baseline_strategy_id}_{tag}_*.json"):
                    try:
                        p.unlink()
                        logger.debug("Removed sweep-only artifact %s", p)
                    except OSError as e:
                        logger.warning("Could not remove %s: %s", p, e)

        return sweep_summary
    def _extract_pack_data_sources(self, pack_paths: List[str]) -> Dict[str, Any]:
        """Extract data-source metadata from loaded packs."""
        bars_providers = set()
        options_snapshot_providers = set()
        secondary_included = False

        for p in pack_paths:
            try:
                with open(p, "r") as f:
                    meta = json.load(f).get("metadata", {})
                bars_providers.add(
                    meta.get("bars_provider", meta.get("provider", "unknown"))
                )
                options_snapshot_providers.add(
                    meta.get("options_snapshot_provider", "unknown")
                )
                if meta.get("secondary_options_included", False):
                    secondary_included = True
            except Exception:
                pass

        return {
            "bars_providers": sorted(bars_providers),
            "options_snapshot_providers": sorted(options_snapshot_providers),
            "secondary_options_included": secondary_included,
        }


    def _compute_mc_bootstrap(self, config: ExperimentConfig, trade_pnls: List[float]) -> Dict[str, Any]:
        mc_summary = run_mc_paths(
            trade_pnls=list(trade_pnls),
            starting_cash=float(config.starting_cash),
            n_paths=int(config.mc_paths),
            method=config.mc_method,
            block_size=config.mc_block_size,
            random_seed=config.mc_random_seed,
            ruin_level=config.mc_ruin_level,
        )
        kill_info = evaluate_mc_kill(mc_summary, config.mc_kill_thresholds)
        return {
            "enabled": True,
            "metrics": mc_summary,
            "killed": kill_info.get("killed", False),
            "reasons": kill_info.get("reasons", []),
            "thresholds": dict(config.mc_kill_thresholds),
        }

    def _save_result(
        self,
        config: ExperimentConfig,
        result: ReplayResult,
        run_id_salt: str = "",
        manifest_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist result to JSON artifact with rich manifest."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Environment Metadata
        git_commit = "UNKNOWN"
        try:
            import subprocess
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except: pass

        # 2. Input Integrity (Pack Hashes)
        pack_info = []
        for p in config.replay_pack_paths:
            try:
                content = open(p, "rb").read()
                pack_info.append({
                    "path": str(p),
                    "hash": hashlib.sha256(content).hexdigest()[:12]
                })
            except:
                pack_info.append({"path": str(p), "hash": "ERROR"})

        # 3. Data Source Policy (from packs + live config)
        pack_data_sources = self._extract_pack_data_sources(config.replay_pack_paths)
        try:
            policy = get_data_source_policy()
            policy_snapshot = {
                "bars_primary": policy.bars_primary,
                "options_snapshots_primary": policy.options_snapshots_primary,
                "options_stream_primary": policy.options_stream_primary,
            }
        except Exception:
            policy_snapshot = {}

        # 4. Deterministic Run ID
        # Strategy + Sorted Params + Sorted Pack Paths
        input_data = f"{result.strategy_id}_{json.dumps(config.strategy_params, sort_keys=True)}_{sorted(config.replay_pack_paths)}_{run_id_salt}"
        run_id = hashlib.md5(input_data.encode()).hexdigest()[:8]
        filename = f"{result.strategy_id}_{config.tag}_{run_id}.json"

        output_file = self.output_dir / filename

        manifest = {
                "run_id": run_id,
                "strategy_class": config.strategy_class.__name__,
                "strategy_params": config.strategy_params,
                "execution_config": config.execution_config.__dict__ if config.execution_config else "DEFAULT",
                "replay_packs": pack_info,
                "data_sources": {
                    "from_packs": pack_data_sources,
                    "config_policy": policy_snapshot,
                },
                "environment": {
                    "git_commit": git_commit,
                    "python_version": __import__("sys").version,
                    "timestamp": timestamp
                },
                "case_id": config.case_id,
            }
        if manifest_overrides:
            manifest.update(manifest_overrides)

        artifact = {
            "manifest": manifest,
            "result": result.summary()
        }

        with open(output_file, "w") as f:
            json.dump(artifact, f, indent=2)

        logger.info(f"Experiment result saved to {output_file}")

    def split_walk_forward(self, bars: List[BarData], 
                          train_days: int, test_days: int, 
                          step_days: Optional[int] = None) -> Generator[tuple[List[BarData], List[BarData]], None, None]:
        """Generator that yields (train_bars, test_bars) windows based on unique trading days."""
        if not bars: return
        
        # 1. Group bars by date (YYYY-MM-DD)
        sorted_bars = sorted(bars, key=lambda b: b.timestamp_ms)
        date_groups: Dict[str, List[BarData]] = {}
        from datetime import timezone
        for b in sorted_bars:
            date_str = datetime.fromtimestamp(b.timestamp_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
            if date_str not in date_groups:
                date_groups[date_str] = []
            date_groups[date_str].append(b)
        
        unique_dates = sorted(date_groups.keys())
        step_days = step_days or test_days
        if step_days < 1:
            step_days = 1  # avoid infinite loop when test_days=0

        # 2. Slide window across dates
        current_idx = 0
        while current_idx + train_days + test_days <= len(unique_dates):
            train_dates = unique_dates[current_idx : current_idx + train_days]
            test_dates = unique_dates[current_idx + train_days : current_idx + train_days + test_days]
            
            train_bars = []
            for d in train_dates: train_bars.extend(date_groups[d])
            
            test_bars = []
            for d in test_dates: test_bars.extend(date_groups[d])
            
            if train_bars and test_bars:
                yield (train_bars, test_bars)
            
            current_idx += step_days

    def _bar_range(self, bars: List[BarData]) -> tuple[int, int]:
        if not bars:
            return (0, 0)
        timestamps = [b.timestamp_ms for b in bars]
        return (min(timestamps), max(timestamps))

    def _filter_dicts_by_time(self, records: List[Dict[str, Any]], start_ms: int, end_ms: int, ts_key: str = "timestamp_ms") -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in records:
            ts = rec.get(ts_key)
            if ts is None:
                continue
            if start_ms <= int(ts) <= end_ms:
                out.append(rec)
        return out

    def run_walk_forward(self, config: ExperimentConfig, 
                         train_days: int, test_days: int, persist_windows: bool = True) -> List[Dict[str, Any]]:
        """Run rolling walk-forward evaluation."""
        # 1. Load all data
        all_bars: List[BarData] = []
        all_outcomes: List[Dict[str, Any]] = []
        all_regimes: List[Dict[str, Any]] = []
        all_snapshots: List[Dict[str, Any]] = []

        for pack_path in config.replay_pack_paths:
            pack = self.load_pack(pack_path)
            for b in pack.get("bars", []):
                bar = BarData(timestamp_ms=b["timestamp_ms"], open=b["open"], high=b["high"], low=b["low"], close=b["close"], volume=b.get("volume", 0))
                if "symbol" in b:
                    object.__setattr__(bar, 'symbol', b["symbol"])
                all_bars.append(bar)
            all_outcomes.extend(pack.get("outcomes", []))
            all_regimes.extend(pack.get("regimes", []))
            all_snapshots.extend(pack.get("snapshots", []))

        snapshots_objs = self._pack_snapshots_to_objects(all_snapshots)

        results = []
        for i, (train_bars, test_bars) in enumerate(self.split_walk_forward(all_bars, train_days, test_days)):
            print(f"Window {i+1}: Train {len(train_bars)} bars, Test {len(test_bars)} bars")

            start_ms, end_ms = self._bar_range(test_bars)
            window_outcomes = self._filter_dicts_by_time(all_outcomes, start_ms, end_ms, ts_key="timestamp_ms")
            window_regimes = self._filter_dicts_by_time(all_regimes, start_ms, end_ms, ts_key="timestamp_ms")
            window_snapshots_dicts = self._filter_dicts_by_time(all_snapshots, start_ms, end_ms, ts_key="recv_ts_ms")
            window_snapshots = self._pack_snapshots_to_objects(window_snapshots_dicts)

            strategy = config.strategy_class(config.strategy_params)
            exec_model = ExecutionModel(config.execution_config)
            harness = ReplayHarness(
                bars=test_bars,
                outcomes=window_outcomes,
                strategy=strategy,
                execution_model=exec_model,
                regimes=window_regimes,
                snapshots=window_snapshots,
                config=ReplayConfig(starting_cash=config.starting_cash)
            )

            res = harness.run()

            mc_bootstrap = None
            if config.mc_bootstrap_enabled:
                mc_seed = None if config.mc_random_seed is None else int(config.mc_random_seed) + i
                mc_cfg = ExperimentConfig(
                    strategy_class=config.strategy_class,
                    strategy_params=config.strategy_params,
                    replay_pack_paths=config.replay_pack_paths,
                    starting_cash=config.starting_cash,
                    execution_config=config.execution_config,
                    output_dir=config.output_dir,
                    tag=config.tag,
                    mc_bootstrap_enabled=True,
                    mc_paths=config.mc_paths,
                    mc_method=config.mc_method,
                    mc_block_size=config.mc_block_size,
                    mc_random_seed=mc_seed,
                    mc_ruin_level=config.mc_ruin_level,
                    mc_kill_thresholds=dict(config.mc_kill_thresholds),
                )
                mc_bootstrap = self._compute_mc_bootstrap(mc_cfg, res.trade_pnls)

            if persist_windows:
                manifest_overrides = {
                    "walk_forward": {
                        "enabled": True,
                        "window_index": i,
                        "train_days": train_days,
                        "test_days": test_days,
                        "train_bars": len(train_bars),
                        "test_bars": len(test_bars),
                        "window_start_ms": start_ms,
                        "window_end_ms": end_ms,
                    },
                    "mc_bootstrap": mc_bootstrap,
                }
                self._save_result(
                    config=config,
                    result=res,
                    run_id_salt=f"wf_window_{i}",
                    manifest_overrides=manifest_overrides,
                )

            results.append({
                "window": i,
                "pnl": res.portfolio_summary["total_realized_pnl"],
                "sharpe": res.portfolio_summary["sharpe_annualized_proxy"],
                "win_rate": res.portfolio_summary["win_rate"],
                "run_summary": res.summary(),
                "mc_bootstrap": mc_bootstrap,
            })

        return results
