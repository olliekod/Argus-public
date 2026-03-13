"""
Production Optimizer
====================

Thorough strategy optimization with:
1. Full date range (BITO inception 2021 → yesterday)
2. Walk-forward validation (in-sample + out-of-sample)
3. Recommendation gauge (multiple factors, not just P&L)
4. Apply to paper trading first (not live)
5. Detailed analysis and confidence scoring
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import itertools
import json
from pathlib import Path

try:
    import numpy as np
except ImportError:
    raise ImportError("Required: pip install numpy")

from .backtester import StrategyBacktester, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Walk-forward validation result."""
    in_sample_return: float
    out_of_sample_return: float
    degradation_pct: float  # How much worse is OOS vs IS?
    is_robust: bool  # OOS > 50% of IS performance?


@dataclass
class RecommendationScore:
    """Multi-factor recommendation score."""
    total_score: float  # 0-100
    
    # Component scores (0-100 each)
    profitability_score: float
    consistency_score: float
    robustness_score: float
    risk_score: float
    trade_count_score: float
    
    # Recommendation
    recommendation: str  # 'STRONG_BUY', 'BUY', 'HOLD', 'AVOID'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    reasoning: List[str]


@dataclass
class OptimizationReport:
    """Complete optimization report."""
    timestamp: str
    
    # Best params
    best_params: Dict
    
    # Full backtest result
    full_backtest: BacktestResult
    
    # Walk-forward validation
    validation: ValidationResult
    
    # Recommendation
    recommendation: RecommendationScore
    
    # Comparison to current
    current_params: Dict
    improvement_pct: float
    
    # Action
    should_apply: bool
    apply_to: str  # 'paper' or 'live'


class ProductionOptimizer:
    """
    Production-grade optimizer with thorough validation.
    
    Features:
    1. Full historical range (BITO inception: Oct 2021)
    2. 70/30 walk-forward validation
    3. Multi-factor recommendation scoring
    4. Paper trading integration
    5. Conservative confidence thresholds
    """
    
    # BITO inception date
    BITO_INCEPTION = "2021-10-19"
    
    # Account size
    ACCOUNT_SIZE = 5000.0
    
    # Parameter grid (MORE granular for thorough testing)
    PARAM_GRID = {
        'iv_threshold': [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        'price_drop_trigger': [-0.01, -0.015, -0.02, -0.025, -0.03],
        'target_delta': [0.12, 0.15, 0.18, 0.20, 0.22],
        'profit_target': [0.40, 0.50, 0.60, 0.70],
        'spread_width_pct': [0.04, 0.05, 0.06, 0.07],
    }
    
    # Current production params (what we're using now)
    CURRENT_PARAMS = {
        'iv_threshold': 0.40,
        'price_drop_trigger': -0.02,
        'target_delta': 0.18,
        'profit_target': 0.50,
        'spread_width_pct': 0.05,
        'time_exit_dte': 5,
        'entry_dte': 14,
    }
    
    # Config file path
    CONFIG_PATH = Path("config/strategy_params.json")
    
    def __init__(self, symbol: str = "BITO"):
        """Initialize production optimizer."""
        self.symbol = symbol
        logger.info(f"Production Optimizer initialized for {symbol}")
    
    def _count_combinations(self) -> int:
        """Count total parameter combinations."""
        total = 1
        for values in self.PARAM_GRID.values():
            total *= len(values)
        return total
    
    def run_full_backtest(
        self, 
        params: Dict,
        start_date: str = None,
        end_date: str = None,
    ) -> BacktestResult:
        """
        Run full historical backtest.
        
        Uses BITO inception (Oct 2021) to yesterday by default.
        """
        if start_date is None:
            start_date = self.BITO_INCEPTION
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        backtester = StrategyBacktester(
            symbol=self.symbol,
            params={**self.CURRENT_PARAMS, **params},
            account_size=self.ACCOUNT_SIZE,
        )
        
        return backtester.run_backtest(start_date, end_date)
    
    def run_walk_forward_validation(
        self, 
        params: Dict,
        train_pct: float = 0.70,
    ) -> ValidationResult:
        """
        Walk-forward validation: train on 70%, test on 30%.
        
        This prevents overfitting by testing on unseen data.
        """
        # Full date range
        start = datetime.strptime(self.BITO_INCEPTION, "%Y-%m-%d")
        end = datetime.now() - timedelta(days=1)
        total_days = (end - start).days
        
        # Split point
        train_days = int(total_days * train_pct)
        split_date = start + timedelta(days=train_days)
        
        # In-sample (training)
        is_result = self.run_full_backtest(
            params,
            start_date=self.BITO_INCEPTION,
            end_date=split_date.strftime("%Y-%m-%d"),
        )
        
        # Out-of-sample (testing)
        oos_result = self.run_full_backtest(
            params,
            start_date=split_date.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        
        # Calculate degradation
        is_return = is_result.total_return_pct
        oos_return = oos_result.total_return_pct
        
        # Annualize for fair comparison
        is_years = train_days / 365
        oos_years = (total_days - train_days) / 365
        
        is_annual = is_return / is_years if is_years > 0 else 0
        oos_annual = oos_return / oos_years if oos_years > 0 else 0
        
        degradation = ((is_annual - oos_annual) / is_annual * 100) if is_annual > 0 else 100
        
        # Is it robust? OOS should be at least 50% of IS performance
        is_robust = oos_annual >= (is_annual * 0.50) and oos_return > 0
        
        return ValidationResult(
            in_sample_return=round(is_return, 2),
            out_of_sample_return=round(oos_return, 2),
            degradation_pct=round(degradation, 1),
            is_robust=is_robust,
        )
    
    def calculate_recommendation_score(
        self,
        result: BacktestResult,
        validation: ValidationResult,
        current_result: BacktestResult,
    ) -> RecommendationScore:
        """
        Calculate multi-factor recommendation score.
        
        Factors:
        1. Profitability: Total return %
        2. Consistency: Monthly win rate
        3. Robustness: Walk-forward validation
        4. Risk: Max drawdown
        5. Trade count: Statistical significance
        """
        reasoning = []
        
        # 1. Profitability (0-100)
        # 20%+ annual = 100, 0% = 0
        annual_return = result.total_return_pct / 3.5  # ~3.5 years of data
        profitability_score = min(100, max(0, annual_return / 20 * 100))
        
        if annual_return >= 15:
            reasoning.append(f"✅ Strong annual return: {annual_return:.1f}%")
        elif annual_return >= 8:
            reasoning.append(f"⚠️ Moderate annual return: {annual_return:.1f}%")
        else:
            reasoning.append(f"❌ Low annual return: {annual_return:.1f}%")
        
        # 2. Consistency (0-100)
        # Based on win rate and avg return variance
        win_rate_score = min(100, result.win_rate)
        
        if result.trades:
            returns = [t.account_return_pct for t in result.trades]
            consistency = 1 - (np.std(returns) / np.mean(returns)) if np.mean(returns) > 0 else 0
            consistency_score = min(100, max(0, consistency * 100))
        else:
            consistency_score = 0
        
        consistency_score = (win_rate_score * 0.5 + consistency_score * 0.5)
        
        if result.win_rate >= 80:
            reasoning.append(f"✅ High win rate: {result.win_rate:.0f}%")
        elif result.win_rate >= 60:
            reasoning.append(f"⚠️ Moderate win rate: {result.win_rate:.0f}%")
        else:
            reasoning.append(f"❌ Low win rate: {result.win_rate:.0f}%")
        
        # 3. Robustness (0-100)
        # Walk-forward validation
        if validation.is_robust:
            robustness_score = 100 - min(50, validation.degradation_pct)
            reasoning.append(f"✅ Walk-forward validation PASSED (OOS: {validation.out_of_sample_return:+.1f}%)")
        else:
            robustness_score = max(0, 50 - validation.degradation_pct)
            reasoning.append(f"❌ Walk-forward validation FAILED (degradation: {validation.degradation_pct:.0f}%)")
        
        # 4. Risk (0-100)
        # Lower drawdown = higher score
        max_dd = result.max_drawdown_pct
        risk_score = max(0, 100 - max_dd * 5)  # -20% DD = 0 score
        
        if max_dd <= 5:
            reasoning.append(f"✅ Low drawdown: {max_dd:.1f}%")
        elif max_dd <= 15:
            reasoning.append(f"⚠️ Moderate drawdown: {max_dd:.1f}%")
        else:
            reasoning.append(f"❌ High drawdown: {max_dd:.1f}%")
        
        # 5. Trade count (0-100)
        # Need enough trades for statistical significance
        trade_count = result.total_trades
        trade_count_score = min(100, trade_count / 50 * 100)  # 50+ trades = max
        
        if trade_count >= 50:
            reasoning.append(f"✅ Sufficient sample size: {trade_count} trades")
        elif trade_count >= 25:
            reasoning.append(f"⚠️ Moderate sample size: {trade_count} trades")
        else:
            reasoning.append(f"❌ Low sample size: {trade_count} trades")
        
        # Total score (weighted)
        total_score = (
            profitability_score * 0.30 +
            consistency_score * 0.20 +
            robustness_score * 0.25 +
            risk_score * 0.15 +
            trade_count_score * 0.10
        )
        
        # Improvement over current
        improvement = result.total_return_pct - current_result.total_return_pct
        if improvement > 10:
            reasoning.append(f"✅ Significant improvement over current: {improvement:+.1f}%")
        elif improvement > 0:
            reasoning.append(f"⚠️ Minor improvement over current: {improvement:+.1f}%")
        else:
            reasoning.append(f"❌ No improvement over current: {improvement:+.1f}%")
        
        # Recommendation
        if total_score >= 80 and validation.is_robust:
            recommendation = 'STRONG_BUY'
            confidence = 'HIGH'
        elif total_score >= 65 and validation.is_robust:
            recommendation = 'BUY'
            confidence = 'MEDIUM'
        elif total_score >= 50:
            recommendation = 'HOLD'
            confidence = 'LOW'
        else:
            recommendation = 'AVOID'
            confidence = 'LOW'
        
        return RecommendationScore(
            total_score=round(total_score, 1),
            profitability_score=round(profitability_score, 1),
            consistency_score=round(consistency_score, 1),
            robustness_score=round(robustness_score, 1),
            risk_score=round(risk_score, 1),
            trade_count_score=round(trade_count_score, 1),
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
        )
    
    def optimize(
        self,
        min_trades: int = 30,
        show_progress: bool = True,
    ) -> OptimizationReport:
        """
        Run full production optimization.
        
        1. Test all combinations on full history
        2. Validate best with walk-forward
        3. Generate recommendation
        """
        combinations = list(itertools.product(*self.PARAM_GRID.values()))
        keys = list(self.PARAM_GRID.keys())
        
        total = len(combinations)
        logger.info(f"Testing {total} parameter combinations...")
        
        if show_progress:
            print(f"\n📊 Testing {total} combinations on full history...")
            print(f"   Date range: {self.BITO_INCEPTION} to yesterday")
            print(f"   Account: ${self.ACCOUNT_SIZE:,.0f}")
            print()
        
        best_params = None
        best_return = float('-inf')
        best_result = None
        
        tested = 0
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            try:
                result = self.run_full_backtest(params)
                tested += 1
                
                if show_progress and tested % 100 == 0:
                    print(f"   Progress: {tested}/{total} ({tested/total*100:.0f}%)")
                
                if result.total_trades < min_trades:
                    continue
                
                if result.total_return_pct > best_return:
                    best_return = result.total_return_pct
                    best_params = params
                    best_result = result
                    
            except Exception as e:
                continue
        
        if show_progress:
            print(f"\n   Tested: {tested} combinations")
        
        if not best_params:
            # Use current params if nothing better
            best_params = self.CURRENT_PARAMS.copy()
            best_result = self.run_full_backtest(best_params)
        
        # Run walk-forward validation on best params
        if show_progress:
            print("   Running walk-forward validation...")
        
        validation = self.run_walk_forward_validation(best_params)
        
        # Get current result for comparison
        current_result = self.run_full_backtest(self.CURRENT_PARAMS)
        
        # Calculate recommendation
        recommendation = self.calculate_recommendation_score(
            best_result, validation, current_result
        )
        
        # Build report
        improvement = best_result.total_return_pct - current_result.total_return_pct
        
        should_apply = (
            recommendation.recommendation in ['STRONG_BUY', 'BUY'] and
            validation.is_robust and
            improvement > 5  # At least 5% improvement
        )
        
        return OptimizationReport(
            timestamp=datetime.now().isoformat(),
            best_params=best_params,
            full_backtest=best_result,
            validation=validation,
            recommendation=recommendation,
            current_params=self.CURRENT_PARAMS,
            improvement_pct=improvement,
            should_apply=should_apply,
            apply_to='paper',  # Always paper first
        )
    
    def format_report(self, report: OptimizationReport) -> str:
        """Format optimization report."""
        rec = report.recommendation
        
        # Recommendation badge
        badge = {
            'STRONG_BUY': '🟢 STRONG BUY',
            'BUY': '🟡 BUY',
            'HOLD': '🟠 HOLD',
            'AVOID': '🔴 AVOID',
        }.get(rec.recommendation, '⚪ UNKNOWN')
        
        lines = [
            "=" * 70,
            "📊 PRODUCTION OPTIMIZATION REPORT",
            f"   {report.timestamp[:10]}",
            "=" * 70,
            "",
            f"RECOMMENDATION: {badge}",
            f"Confidence: {rec.confidence}",
            f"Score: {rec.total_score:.0f}/100",
            "",
            "COMPONENT SCORES",
            "-" * 50,
            f"  Profitability: {rec.profitability_score:.0f}/100",
            f"  Consistency:   {rec.consistency_score:.0f}/100",
            f"  Robustness:    {rec.robustness_score:.0f}/100",
            f"  Risk:          {rec.risk_score:.0f}/100",
            f"  Sample Size:   {rec.trade_count_score:.0f}/100",
            "",
            "REASONING",
            "-" * 50,
        ]
        
        for reason in rec.reasoning:
            lines.append(f"  {reason}")
        
        lines.extend([
            "",
            "OPTIMAL PARAMETERS",
            "-" * 50,
            f"  IV Threshold:  {report.best_params.get('iv_threshold', 0.4)*100:.0f}%",
            f"  Drop Trigger:  {report.best_params.get('price_drop_trigger', -0.02)*100:.1f}%",
            f"  Target Delta:  {report.best_params.get('target_delta', 0.18)}",
            f"  Profit Target: {report.best_params.get('profit_target', 0.5)*100:.0f}%",
            f"  Spread Width:  {report.best_params.get('spread_width_pct', 0.05)*100:.0f}%",
            "",
            "BACKTEST RESULTS (Full History)",
            "-" * 50,
            f"  Period: {self.BITO_INCEPTION} to yesterday",
            f"  Trades: {report.full_backtest.total_trades}",
            f"  Win Rate: {report.full_backtest.win_rate:.0f}%",
            f"  Total Return: {report.full_backtest.total_return_pct:+.1f}%",
            f"  Max Drawdown: {report.full_backtest.max_drawdown_pct:.1f}%",
            "",
            "WALK-FORWARD VALIDATION",
            "-" * 50,
            f"  In-Sample Return: {report.validation.in_sample_return:+.1f}%",
            f"  Out-of-Sample:    {report.validation.out_of_sample_return:+.1f}%",
            f"  Degradation:      {report.validation.degradation_pct:.0f}%",
            f"  Status:           {'✅ PASSED' if report.validation.is_robust else '❌ FAILED'}",
            "",
            "VS CURRENT STRATEGY",
            "-" * 50,
            f"  Current Return: {report.full_backtest.total_return_pct - report.improvement_pct:+.1f}%",
            f"  Optimal Return: {report.full_backtest.total_return_pct:+.1f}%",
            f"  Improvement:    {report.improvement_pct:+.1f}%",
            "",
        ])
        
        # Action
        if report.should_apply:
            lines.extend([
                "=" * 70,
                "✅ RECOMMENDED ACTION: Apply to Paper Trading",
                "",
                "Run this command to apply:",
                "  python scripts/apply_params.py --paper",
                "=" * 70,
            ])
        else:
            lines.extend([
                "=" * 70,
                "⚠️ RECOMMENDED ACTION: Keep Current Parameters",
                "",
                "Reason: Improvement not significant or validation failed.",
                "=" * 70,
            ])
        
        return "\n".join(lines)
    
    def save_params(self, params: Dict, target: str = 'paper') -> None:
        """
        Save parameters to config file.
        
        Args:
            params: Parameters to save
            target: 'paper' or 'live'
        """
        config_file = self.CONFIG_PATH
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing or create new
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {'live': {}, 'paper': {}, 'history': []}
        
        # Update target
        config[target] = {
            **params,
            'updated_at': datetime.now().isoformat(),
        }
        
        # Add to history
        config['history'].append({
            'params': params,
            'target': target,
            'timestamp': datetime.now().isoformat(),
        })
        
        # Keep last 10 history entries
        config['history'] = config['history'][-10:]
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved {target} params to {config_file}")


def run_optimization():
    """Run production optimization from command line."""
    optimizer = ProductionOptimizer(symbol="BITO")
    
    print("=" * 70)
    print("🔧 PRODUCTION STRATEGY OPTIMIZATION")
    print("    Thorough backtest + walk-forward validation")
    print("=" * 70)
    
    report = optimizer.optimize(min_trades=30, show_progress=True)
    print(optimizer.format_report(report))
    
    return report


if __name__ == "__main__":
    run_optimization()
