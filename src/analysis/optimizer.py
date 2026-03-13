"""
Parameter Optimizer v2
======================

Optimizes strategy parameters for MAXIMUM PROFIT.
Uses $5,000 account size, shows % returns.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple
import itertools

from .backtester import StrategyBacktester, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: Dict
    best_metric: float
    best_result: BacktestResult
    metric_name: str
    all_results: List[Tuple[Dict, BacktestResult]]


class ParameterOptimizer:
    """
    Optimizes strategy parameters for PROFIT.
    
    Focuses on:
    - Total Return % (not dollars, not win rate)
    - More trades = more compounding opportunity
    """
    
    # Parameter ranges
    PARAM_GRID = {
        'iv_threshold': [0.35, 0.40, 0.45, 0.50],  # Lower = more trades
        'price_drop_trigger': [-0.01, -0.02, -0.03],  # Looser = more trades
        'target_delta': [0.15, 0.18, 0.20, 0.22],
        'profit_target': [0.40, 0.50, 0.60],
        'spread_width_pct': [0.04, 0.05, 0.06],
    }
    
    def __init__(self, symbol: str = "BITO", account_size: float = 5000.0):
        """Initialize optimizer."""
        self.symbol = symbol
        self.account_size = account_size
        logger.info(f"Optimizer: {symbol}, ${account_size:,.0f} account")
    
    def _generate_combinations(self, grid: Dict = None) -> List[Dict]:
        """Generate all parameter combinations."""
        grid = grid or self.PARAM_GRID
        keys = list(grid.keys())
        values = list(grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    def optimize(
        self,
        start_date: str,
        end_date: str,
        metric: str = 'total_return_pct',
        min_trades: int = 15,
    ) -> OptimizationResult:
        """
        Run optimization for PROFIT.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            metric: 'total_return_pct' (default) or 'avg_return_pct'
            min_trades: Minimum trades required for valid result
            
        Returns:
            OptimizationResult with best parameters
        """
        combinations = self._generate_combinations()
        logger.info(f"Testing {len(combinations)} parameter combinations...")
        
        all_results = []
        best_params = None
        best_metric = float('-inf')
        best_result = None
        
        for i, params in enumerate(combinations):
            try:
                backtester = StrategyBacktester(
                    symbol=self.symbol, 
                    params=params,
                    account_size=self.account_size,
                )
                result = backtester.run_backtest(start_date, end_date)
                
                # Skip if not enough trades
                if result.total_trades < min_trades:
                    continue
                
                all_results.append((params, result))
                
                # Get metric value
                if metric == 'total_return_pct':
                    metric_value = result.total_return_pct
                elif metric == 'avg_return_pct':
                    metric_value = result.avg_return_pct
                else:
                    metric_value = result.total_return_pct
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_result = result
                    
            except Exception as e:
                logger.warning(f"Error: {e}")
                continue
        
        if not best_params:
            # Return defaults if no valid result
            backtester = StrategyBacktester(symbol=self.symbol, account_size=self.account_size)
            best_result = backtester.run_backtest(start_date, end_date)
            best_params = backtester.params
            best_metric = best_result.total_return_pct
        
        logger.info(f"Best {metric}: {best_metric:+.1f}%")
        
        return OptimizationResult(
            best_params=best_params,
            best_metric=best_metric,
            best_result=best_result,
            metric_name=metric,
            all_results=all_results,
        )
    
    def format_report(self, opt_result: OptimizationResult) -> str:
        """Format optimization results."""
        lines = [
            "=" * 60,
            "🔧 PARAMETER OPTIMIZATION RESULTS",
            "=" * 60,
            "",
            f"Optimized for: {opt_result.metric_name.replace('_', ' ').upper()}",
            f"Account Size: ${self.account_size:,.0f}",
            f"Combinations tested: {len(opt_result.all_results)}",
            "",
            "BEST PARAMETERS",
            "-" * 40,
            f"  IV Threshold: {opt_result.best_params.get('iv_threshold', 0.4)*100:.0f}%",
            f"  Drop Trigger: {opt_result.best_params.get('price_drop_trigger', -0.02)*100:.1f}%",
            f"  Target Delta: {opt_result.best_params.get('target_delta', 0.18)}",
            f"  Profit Target: {opt_result.best_params.get('profit_target', 0.5)*100:.0f}%",
            f"  Spread Width: {opt_result.best_params.get('spread_width_pct', 0.05)*100:.0f}%",
            "",
            "BEST RESULT",
            "-" * 40,
            f"  Total Return: {opt_result.best_result.total_return_pct:+.1f}%",
            f"  Total P&L: ${opt_result.best_result.total_pnl:,.2f}",
            f"  Trades: {opt_result.best_result.total_trades}",
            f"  Win Rate: {opt_result.best_result.win_rate:.0f}%",
            f"  Max Drawdown: {opt_result.best_result.max_drawdown_pct:.1f}%",
            "",
        ]
        
        # Top 5 by metric
        if opt_result.all_results:
            lines.extend([
                "TOP 5 COMBINATIONS",
                "-" * 40,
            ])
            
            sorted_results = sorted(
                opt_result.all_results,
                key=lambda x: x[1].total_return_pct,
                reverse=True
            )[:5]
            
            for params, result in sorted_results:
                lines.append(
                    f"  IV:{params['iv_threshold']*100:.0f}% "
                    f"Δ:{params['target_delta']} "
                    f"TP:{params['profit_target']*100:.0f}% "
                    f"→ {result.total_return_pct:+.1f}% ({result.total_trades} trades)"
                )
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)


def run_optimization_cli():
    """Run optimization from command line."""
    print("=" * 60)
    print("🔧 STRATEGY PARAMETER OPTIMIZATION")
    print("    Optimizing for PROFIT")
    print("=" * 60)
    
    optimizer = ParameterOptimizer(symbol="BITO", account_size=5000.0)
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=540)).strftime("%Y-%m-%d")
    
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Account: $5,000")
    
    # Count combinations
    total = 1
    for values in optimizer.PARAM_GRID.values():
        total *= len(values)
    print(f"Testing: {total} combinations")
    print("\n⏳ This may take a few minutes...\n")
    
    result = optimizer.optimize(
        start_date, 
        end_date, 
        metric='total_return_pct',
        min_trades=15,
    )
    
    print(optimizer.format_report(result))
    
    # Clear recommendation
    print("\n📊 RECOMMENDATION")
    print("-" * 40)
    print(f"Use these parameters for maximum profit:")
    print(f"  IV Threshold: {result.best_params.get('iv_threshold', 0.4)*100:.0f}%")
    print(f"  Drop Trigger: {result.best_params.get('price_drop_trigger', -0.02)*100:.1f}%")
    print(f"  Target Delta: {result.best_params.get('target_delta', 0.18)}")
    print(f"  Profit Target: {result.best_params.get('profit_target', 0.5)*100:.0f}%")
    print(f"  Spread Width: {result.best_params.get('spread_width_pct', 0.05)*100:.0f}%")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_optimization_cli()
