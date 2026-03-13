"""
Trader Config Generator
=======================

Generates 35,000 unique trader configurations across 4 strategies.
Covers all 33,600 unique parameter combinations with small buffer.
Includes market regime awareness.
"""

import itertools
import logging
import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import random

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.paper_trader import TraderConfig, StrategyType

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime for strategy filtering."""
    BULL = "bull"      # BTC up >5% in 2 weeks
    BEAR = "bear"      # BTC down >5% in 2 weeks  
    NEUTRAL = "neutral"  # Within ±5%
    ANY = "any"        # Trade in all regimes


# Parameter ranges - expanded for ~400,000 total traders
# Fits within ~15GB/year storage budget with 150K-400K traders
PARAM_RANGES = {
    'warmth_min': [3, 4, 5, 6, 7, 8],                # 6 values
    'pop_min': [55, 60, 65, 70, 75, 80],            # 6 values
    'dte_target': [14, 21, 30, 45, 60, 90],         # 6 values
    'gap_tolerance_pct': [8, 10, 12, 14, 16],       # 5 values
    'profit_target_pct': [30, 50, 75],              # 3 values
    'stop_loss_pct': [150, 200, 300],               # 3 values
    'session_filter': ['any', 'morning'],             # 2 values
    'budget_tier': [5, 14],                           # 2 values (risk %)
}

# Strategy-specific IV ranges (finer steps, ~8-10 values each)
STRATEGY_IV_RANGES = {
    StrategyType.BULL_PUT: [40, 44, 48, 52, 56, 60, 64, 68, 72, 76],
    StrategyType.BEAR_CALL: [40, 44, 48, 52, 56, 60, 64, 68, 72, 76],
    StrategyType.IRON_CONDOR: [45, 48, 51, 54, 57, 60, 63, 66, 69, 72],
    StrategyType.STRADDLE_SELL: [55, 59, 63, 67, 71, 75, 79, 83, 87, 91],
}

# Calculated combos per strategy: 10 (iv) * 6 (warmth) * 6 (pop) * 6 (dte) * 5 (gap) * 3 (profit) * 3 (stop) * 3 (session)
# = 10 * 6 * 6 * 6 * 5 * 3 * 3 * 3 = 291,600
# With regimes: ~400K-500K total

# Strategy-regime compatibility (2 regimes for directional, 1 for neutral)
STRATEGY_REGIME_COMPAT = {
    StrategyType.BULL_PUT: [MarketRegime.BULL, MarketRegime.NEUTRAL, MarketRegime.ANY],
    StrategyType.BEAR_CALL: [MarketRegime.BEAR, MarketRegime.NEUTRAL, MarketRegime.ANY],
    StrategyType.IRON_CONDOR: [MarketRegime.NEUTRAL, MarketRegime.ANY],
    StrategyType.STRADDLE_SELL: [MarketRegime.NEUTRAL, MarketRegime.ANY],
}


def generate_all_combinations_for_strategy(strategy: StrategyType) -> List[tuple]:
    """Generate all valid parameter combinations for a strategy."""
    iv_range = STRATEGY_IV_RANGES.get(strategy, [55, 60, 65, 70, 75])
    valid_regimes = STRATEGY_REGIME_COMPAT.get(strategy, [MarketRegime.ANY])
    
    combos = list(itertools.product(
        iv_range,
        PARAM_RANGES['warmth_min'],
        PARAM_RANGES['pop_min'],
        PARAM_RANGES['dte_target'],
        PARAM_RANGES['gap_tolerance_pct'],
        PARAM_RANGES['profit_target_pct'],
        PARAM_RANGES['stop_loss_pct'],
        valid_regimes,
        PARAM_RANGES['session_filter'],
        PARAM_RANGES['budget_tier'],
    ))
    
    return combos


def generate_configs_for_strategy(
    strategy: StrategyType,
    max_count: Optional[int] = None,
    start_id: int = 0,
) -> List[TraderConfig]:
    """
    Generate trader configs for a single strategy.
    
    Args:
        strategy: Strategy type
        max_count: Optional max number of configs (None = all combinations)
        start_id: Starting ID number
        
    Returns:
        List of TraderConfig objects
    """
    all_combos = generate_all_combinations_for_strategy(strategy)
    
    # Sample if max_count specified and less than total
    if max_count and len(all_combos) > max_count:
        random.seed(42)  # Reproducible
        sampled = random.sample(all_combos, max_count)
    else:
        sampled = all_combos
    
    configs = []
    for i, combo in enumerate(sampled):
        iv_min, warmth, pop, dte, gap, profit, stop, regime, session, budget = combo
        
        config = TraderConfig(
            trader_id=f"PT-{start_id + i:06d}",  # 6 digits for 400K+
            strategy_type=strategy,
            iv_min=iv_min,
            iv_max=100.0,
            warmth_min=warmth,
            pop_min=pop,
            dte_target=dte,
            dte_min=max(7, dte - 14),
            dte_max=dte + 21,
            gap_tolerance_pct=gap,
            profit_target_pct=profit,
            stop_loss_pct=stop,
            regime=regime.value,
            session_filter=session,
            position_size_pct=budget,
            max_risk_dollars=5000 * (budget / 100), # Heuristic for default cap
        )
        configs.append(config)
    
    return configs


def generate_all_configs(
    total_traders: int = 35000,
    strategies: List[StrategyType] = None,
    full_coverage: bool = True,
    max_traders: int = 2_000_000,
) -> List[TraderConfig]:
    """
    Generate all trader configurations.
    
    Args:
        total_traders: Target number of traders
        strategies: List of strategies to use (default: all 4)
        full_coverage: If True, generate ALL unique combinations
        
    Returns:
        List of all TraderConfig objects
    """
    if strategies is None:
        strategies = list(StrategyType)
    
    all_configs = []
    current_id = 0
    
    if full_coverage:
        # Generate ALL unique combinations for each strategy — no artificial cap.
        # The number of traders = the number of unique parameter combos.
        for strategy in strategies:
            configs = generate_configs_for_strategy(
                strategy=strategy,
                max_count=None,  # All combos
                start_id=current_id,
            )
            all_configs.extend(configs)
            current_id += len(configs)
        if len(all_configs) > max_traders:
            import random
            random.seed(42)  # Reproducible
            all_configs = random.sample(all_configs, max_traders)
            logger.info(f"Sampled {max_traders:,} from {current_id:,} total combinations")
    else:
        # Distribute total_traders evenly across strategies
        traders_per_strategy = total_traders // len(strategies)
        remainder = total_traders % len(strategies)
        
        for i, strategy in enumerate(strategies):
            count = traders_per_strategy + (1 if i < remainder else 0)
            configs = generate_configs_for_strategy(
                strategy=strategy,
                max_count=count,
                start_id=current_id,
            )
            all_configs.extend(configs)
            current_id += len(configs)
    
    return all_configs


def get_config_summary(configs: List[TraderConfig]) -> Dict:
    """Get summary statistics for a list of configs."""
    from collections import Counter
    
    strategies = Counter(c.strategy_type.value for c in configs)
    iv_mins = [c.iv_min for c in configs]
    warmths = [c.warmth_min for c in configs]
    pops = [c.pop_min for c in configs]
    dtes = [c.dte_target for c in configs]
    
    # Count regimes and sessions
    regimes = Counter(c.regime for c in configs)
    sessions = Counter(c.session_filter for c in configs)
    
    summary = {
        'total': len(configs),
        'by_strategy': dict(strategies),
        'by_regime': dict(regimes),
        'by_session': dict(sessions),
        'iv_min': {
            'min': min(iv_mins),
            'max': max(iv_mins),
            'mean': sum(iv_mins) / len(iv_mins),
        },
        'warmth_min': {
            'min': min(warmths),
            'max': max(warmths),
            'mean': sum(warmths) / len(warmths),
        },
        'pop_min': {
            'min': min(pops),
            'max': max(pops),
            'mean': sum(pops) / len(pops),
        },
        'dte_target': {
            'min': min(dtes),
            'max': max(dtes),
            'mean': sum(dtes) / len(dtes),
        },
    }
    
    return summary


def get_total_combinations() -> Dict[str, int]:
    """Calculate total unique combinations per strategy."""
    totals = {}
    for strategy in StrategyType:
        combos = generate_all_combinations_for_strategy(strategy)
        totals[strategy.value] = len(combos)
    totals['total'] = sum(totals.values())
    return totals


def save_configs_to_file(configs: List[TraderConfig], filepath: str) -> None:
    """Save configs to JSON file for persistence."""
    import json
    from datetime import datetime, timezone
    
    data = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'total_traders': len(configs),
        'configs': [c.to_dict() for c in configs],
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_configs_from_file(filepath: str) -> List[TraderConfig]:
    """Load configs from JSON file."""
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [TraderConfig.from_dict(c) for c in data['configs']]


# Test function
def test_generator():
    """Test configuration generator."""
    print("Trader Config Generator Test (Full Coverage)")
    print("=" * 60)
    
    # Show total possible combinations
    totals = get_total_combinations()
    print(f"\nTotal unique combinations by strategy:")
    for strat, count in totals.items():
        print(f"  {strat}: {count:,}")
    
    # Generate all configs with full coverage
    print(f"\nGenerating full coverage configs...")
    configs = generate_all_configs(full_coverage=True)
    
    summary = get_config_summary(configs)
    
    print(f"\nGenerated {summary['total']:,} traders")
    print(f"\nBy strategy:")
    for strat, count in summary['by_strategy'].items():
        print(f"  {strat}: {count:,}")
    
    if 'by_regime' in summary:
        print(f"\nBy regime:")
        for regime, count in summary['by_regime'].items():
            print(f"  {regime}: {count:,}")
    
    print(f"\nParameter ranges covered:")
    print(f"  IV min: {summary['iv_min']['min']}-{summary['iv_min']['max']} (avg: {summary['iv_min']['mean']:.1f})")
    print(f"  Warmth: {summary['warmth_min']['min']}-{summary['warmth_min']['max']} (avg: {summary['warmth_min']['mean']:.1f})")
    print(f"  PoP: {summary['pop_min']['min']}-{summary['pop_min']['max']} (avg: {summary['pop_min']['mean']:.1f})")
    print(f"  DTE: {summary['dte_target']['min']}-{summary['dte_target']['max']} (avg: {summary['dte_target']['mean']:.1f})")
    
    # Sample configs
    print(f"\nSample configs:")
    for config in configs[:5]:
        print(f"  {config.trader_id}: {config.strategy_type.value}, IV>{config.iv_min}, warmth>{config.warmth_min}, regime={config.regime}, budget={config.position_size_pct}%")
    
    # Memory usage
    import sys
    mem_bytes = sys.getsizeof(configs) + sum(sys.getsizeof(c) for c in configs)
    print(f"\nMemory usage: ~{mem_bytes / 1024 / 1024:.1f} MB")
    
    print(f"\n✅ Generated {len(configs):,} unique trader configs with full coverage!")


if __name__ == "__main__":
    test_generator()
