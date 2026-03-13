"""
Run Experiment CLI
==================

Entry point for running standardized strategy experiments.

Usage:
  python scripts/run_experiment.py --strategy VRPCreditSpreadStrategy --pack data/spy_pack.json --params '{"min_vrp": 0.06}'
"""

import argparse
import json
import logging
import yaml
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Type

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# So strategy diagnostics (e.g. why 0 trades) are visible
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig
from src.analysis.regime_stress import run_regime_subset_stress
from src.analysis.replay_harness import ReplayStrategy

def load_strategy_class(name: str) -> Type[ReplayStrategy]:
    """Dynamically load strategy class from src.strategies."""
    # Common locations
    modules = [
        "src.strategies.vrp_credit_spread",
        "src.strategies.dow_regime_timing",
    ]
    
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, name):
                return getattr(mod, name)
        except ImportError:
            continue
            
    raise ImportError(f"Could not find strategy class {name} in known modules.")

def main():
    parser = argparse.ArgumentParser(description="Argus Experiment Runner CLI")
    parser.add_argument("--strategy", required=True, help="Strategy class name (e.g. VRPCreditSpreadStrategy)")
    parser.add_argument("--pack", required=True, action="append", help="Path to one or more Replay Packs (JSON)")
    parser.add_argument("--params", help="JSON string of strategy parameters", default="{}")
    parser.add_argument("--config", help="Path to YAML config file (overrides CLI params)")
    parser.add_argument("--sweep", help="Path to YAML param grid for parameter sweep")
    parser.add_argument("--output", default="logs/experiments", help="Output directory for results")
    parser.add_argument("--cash", type=float, default=10000.0, help="Starting cash")
    parser.add_argument("--regime-stress", action="store_true", default=False, help="Run regime subset stress test after the main run")
    parser.add_argument("--mc-bootstrap", action="store_true", default=False, help="Enable MC/bootstrap path stress over realized trades")
    parser.add_argument("--mc-paths", type=int, default=1000, help="Number of MC paths when --mc-bootstrap is enabled")
    parser.add_argument("--mc-method", default="bootstrap", choices=["bootstrap", "iid"], help="MC resampling method")
    parser.add_argument("--mc-block-size", type=int, default=None, help="Optional fixed block size for bootstrap method")
    parser.add_argument("--mc-seed", type=int, default=42, help="Random seed for MC bootstrap reproducibility")
    parser.add_argument("--mc-ruin-level", type=float, default=0.2, help="Ruin threshold as fraction of starting cash")
    parser.add_argument("--mc-kill-thresholds", default=None, help="JSON string or YAML file path with MC kill thresholds")

    args = parser.parse_args()

    # 1. Load params
    params = json.loads(args.params)
    if args.config:
        with open(args.config, "r") as f:
            params.update(yaml.safe_load(f))

    mc_kill_thresholds: Dict[str, float] = {}
    if args.mc_kill_thresholds:
        candidate = Path(args.mc_kill_thresholds)
        if candidate.exists():
            with open(candidate, "r") as f:
                loaded = yaml.safe_load(f) or {}
                mc_kill_thresholds = dict(loaded)
        else:
            mc_kill_thresholds = dict(json.loads(args.mc_kill_thresholds))

    # 2. Resolve Strategy
    try:
        strat_cls = load_strategy_class(args.strategy)
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 3. Initialize Runner
    runner = ExperimentRunner(output_dir=args.output)

    # 4. Handle Sweep vs Single Run
    if args.sweep:
        with open(args.sweep, "r") as f:
            grid = yaml.safe_load(f)
        
        base_config = ExperimentConfig(
            strategy_class=strat_cls,
            strategy_params=params,
            replay_pack_paths=args.pack,
            starting_cash=args.cash,
            mc_bootstrap_enabled=args.mc_bootstrap,
            mc_paths=args.mc_paths,
            mc_method=args.mc_method,
            mc_block_size=args.mc_block_size,
            mc_random_seed=args.mc_seed,
            mc_ruin_level=args.mc_ruin_level,
            mc_kill_thresholds=mc_kill_thresholds,
        )
        
        results = runner.run_parameter_grid(strat_cls, base_config, grid)
        print(f"\n--- Sweep Results ({len(results)}) ---")
        for r in sorted(results, key=lambda x: x.portfolio_summary['total_realized_pnl'], reverse=True):
            p = r.portfolio_summary
            print(f"PnL: {p['total_realized_pnl']:>8} | Sharpe: {p['sharpe_annualized_proxy']:>5} | Params: {r.strategy_state.get('params', 'N/A')}")
            
    else:
        config = ExperimentConfig(
            strategy_class=strat_cls,
            strategy_params=params,
            replay_pack_paths=args.pack,
            starting_cash=args.cash,
            mc_bootstrap_enabled=args.mc_bootstrap,
            mc_paths=args.mc_paths,
            mc_method=args.mc_method,
            mc_block_size=args.mc_block_size,
            mc_random_seed=args.mc_seed,
            mc_ruin_level=args.mc_ruin_level,
            mc_kill_thresholds=mc_kill_thresholds,
        )
        
        result = runner.run(config)
        
        print("\n" + "="*40)
        print(f" EXPERIMENT COMPLETE: {result.strategy_id}")
        print("="*40)
        p = result.portfolio_summary
        print(f"Total PnL:     {p['total_realized_pnl']:>10}")
        print(f"Return %:      {p['total_return_pct']:>10}%")
        print(f"Sharpe Proxy:  {p['sharpe_annualized_proxy']:>10}")
        print(f"Win Rate:      {p['win_rate']:>10}%")
        print(f"Trades:        {p['total_trades']:>10}")
        print(f"Profit Factor: {p['profit_factor']:>10}")
        print(f"Max DD:        {p['max_drawdown']:>10}")
        print("-" * 40)
        print(f"Output saved to: {args.output}")
        if args.mc_bootstrap:
            print("MC/bootstrap summary persisted in experiment artifact manifest.mc_bootstrap")
        if args.regime_stress:
            packs = [runner.load_pack(p) for p in args.pack]
            bars = []
            outcomes = []
            regimes = []
            snapshots = []
            from src.core.outcome_engine import BarData
            for pack in packs:
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

            stress = run_regime_subset_stress(
                bars=bars,
                outcomes=outcomes,
                regimes=regimes,
                snapshots=snapshots,
                strategy_class=strat_cls,
                strategy_params=params,
                starting_cash=args.cash,
            )
            stress_path = Path(args.output) / f"{result.strategy_id}_regime_stress.json"
            stress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stress_path, "w") as f:
                json.dump(stress, f, indent=2)
            print(f"Regime stress saved to: {stress_path}")
            print(f"Regime stress score (fraction profitable): {stress['stress_score']}")

if __name__ == "__main__":
    main()
