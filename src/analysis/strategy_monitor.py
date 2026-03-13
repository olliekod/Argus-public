"""
Strategy Monitor
================

Intelligent strategy monitoring with:
1. Risk-adjusted optimization (Sharpe ratio, not just P&L)
2. Regime detection (high vol vs low vol periods)
3. Quarterly health checks
4. Automatic parameter recommendations
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

try:
    import numpy as np
except ImportError:
    raise ImportError("Required: pip install numpy")

from .backtester import StrategyBacktester, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ParameterRecommendation:
    """Recommendation for strategy parameters."""
    iv_threshold: float
    target_delta: float
    profit_target: float
    spread_width: float
    
    sharpe_ratio: float
    win_rate: float
    avg_pnl: float
    
    confidence: str  # 'high', 'medium', 'low'
    reasoning: str


@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime: str  # 'low_vol', 'normal', 'high_vol', 'crisis'
    avg_iv: float
    iv_percentile: float
    recommendation: str


class StrategyMonitor:
    """
    Monitors strategy performance and adapts parameters intelligently.
    
    Key principles:
    1. Optimize for RISK-ADJUSTED returns (Sharpe), not just P&L
    2. Prefer FEWER high-edge trades over MANY low-edge trades
    3. Adapt to market regimes (be more aggressive in high IV)
    4. Run quarterly health checks
    """
    
    # Conservative parameter bounds (don't over-optimize)
    CONSERVATIVE_BOUNDS = {
        'iv_threshold': (0.45, 0.70),   # 45-70%, not lower
        'target_delta': (0.12, 0.22),   # 12-22 delta
        'profit_target': (0.40, 0.60),  # 40-60% profit
        'spread_width': (0.04, 0.08),   # 4-8% width
    }
    
    # Regime thresholds
    REGIME_THRESHOLDS = {
        'low_vol': 0.40,    # IV < 40%
        'normal': 0.60,     # IV 40-60%
        'high_vol': 0.80,   # IV 60-80%
        'crisis': 1.00,     # IV 80%+
    }
    
    def __init__(self, symbol: str = "BITO"):
        """Initialize strategy monitor."""
        self.symbol = symbol
        self.last_check: Optional[datetime] = None
        self.current_params: Dict = {}
        
        logger.info(f"Strategy Monitor initialized for {symbol}")
    
    def calculate_sharpe(self, trades: List, risk_free_rate: float = 0.04) -> float:
        """
        Calculate Sharpe ratio from trade results.
        
        Args:
            trades: List of BacktestTrade objects
            risk_free_rate: Annual risk-free rate (default 4%)
            
        Returns:
            Annualized Sharpe ratio
        """
        if not trades or len(trades) < 2:
            return 0.0
        
        pnls = [t.pnl_dollars for t in trades]
        
        avg_return = np.mean(pnls)
        std_return = np.std(pnls)
        
        if std_return == 0:
            return 10.0 if avg_return > 0 else 0.0  # Perfect Sharpe if no variance
        
        # Annualize: assume ~20 trades/year
        trades_per_year = 20
        annual_return = avg_return * trades_per_year
        annual_vol = std_return * np.sqrt(trades_per_year)
        
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        return round(sharpe, 2)
    
    def calculate_edge_per_trade(self, result: BacktestResult) -> float:
        """Calculate average edge (profit / risk) per trade."""
        if not result.trades:
            return 0.0
        
        edges = []
        for trade in result.trades:
            max_risk = trade.spread_width * 100  # Per contract
            edge = trade.pnl_dollars / max_risk if max_risk > 0 else 0
            edges.append(edge)
        
        return np.mean(edges) if edges else 0.0
    
    def run_smart_optimization(
        self,
        start_date: str,
        end_date: str,
    ) -> ParameterRecommendation:
        """
        Run optimization focused on RISK-ADJUSTED returns.
        
        Unlike simple optimizer, this:
        1. Uses Sharpe ratio as primary metric
        2. Penalizes low trade counts (need statistical significance)
        3. Stays within conservative bounds
        4. Explains the recommendation
        """
        logger.info("Running smart optimization...")
        
        # Define parameter grid within conservative bounds
        param_grid = {
            'iv_threshold': [0.45, 0.50, 0.55, 0.60, 0.65],
            'target_delta': [0.14, 0.16, 0.18, 0.20],
            'profit_target': [0.40, 0.50, 0.60],
            'spread_width_pct': [0.04, 0.05, 0.06, 0.07],
        }
        
        import itertools
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        best_score = float('-inf')
        best_params = None
        best_result = None
        
        for params in combinations:
            try:
                backtester = StrategyBacktester(symbol=self.symbol, params=params)
                result = backtester.run_backtest(start_date, end_date)
                
                if not result.trades or len(result.trades) < 5:
                    continue  # Need minimum sample size
                
                # Calculate composite score:
                # 50% Sharpe, 25% Win Rate, 25% Avg Edge
                sharpe = self.calculate_sharpe(result.trades)
                edge = self.calculate_edge_per_trade(result)
                
                # Normalize components
                sharpe_score = min(sharpe / 2.0, 1.0)  # Cap at Sharpe 2
                wr_score = result.win_rate / 100.0
                edge_score = min(edge / 0.30, 1.0)    # Cap at 30% edge
                
                # Composite score with trade count bonus
                trade_bonus = min(len(result.trades) / 30, 1.0)  # Bonus for more trades
                
                composite = (
                    0.40 * sharpe_score +
                    0.25 * wr_score +
                    0.25 * edge_score +
                    0.10 * trade_bonus
                )
                
                if composite > best_score:
                    best_score = composite
                    best_params = params
                    best_result = result
                    
            except Exception as e:
                continue
        
        if not best_params or not best_result:
            # Return current defaults
            return ParameterRecommendation(
                iv_threshold=0.50,
                target_delta=0.18,
                profit_target=0.50,
                spread_width=0.05,
                sharpe_ratio=0,
                win_rate=0,
                avg_pnl=0,
                confidence='low',
                reasoning="Insufficient data for optimization. Using conservative defaults."
            )
        
        sharpe = self.calculate_sharpe(best_result.trades)
        
        # Determine confidence
        if len(best_result.trades) >= 20 and sharpe > 1.0:
            confidence = 'high'
        elif len(best_result.trades) >= 10 and sharpe > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Build reasoning
        reasoning_parts = []
        
        if best_params['iv_threshold'] >= 0.55:
            reasoning_parts.append("Higher IV threshold = better edge per trade")
        if best_params['profit_target'] <= 0.50:
            reasoning_parts.append("Tighter profit target = higher win rate")
        if best_params['target_delta'] <= 0.16:
            reasoning_parts.append("Lower delta = more OTM = higher PoP")
        
        reasoning_parts.append(f"Based on {len(best_result.trades)} trades")
        reasoning_parts.append(f"Sharpe: {sharpe:.2f}")
        
        return ParameterRecommendation(
            iv_threshold=best_params['iv_threshold'],
            target_delta=best_params['target_delta'],
            profit_target=best_params['profit_target'],
            spread_width=best_params['spread_width_pct'],
            sharpe_ratio=sharpe,
            win_rate=best_result.win_rate,
            avg_pnl=best_result.avg_pnl,
            confidence=confidence,
            reasoning="; ".join(reasoning_parts),
        )
    
    def detect_regime(self, df=None) -> MarketRegime:
        """
        Detect current market regime based on IV levels.
        
        Returns:
            MarketRegime with recommendations
        """
        # Fetch recent data if not provided
        if df is None:
            backtester = StrategyBacktester(symbol=self.symbol)
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
            df = backtester.fetch_historical_data(start, end)
        
        # Get recent IV
        recent_iv = df['IV_Proxy'].dropna().tail(5).mean()
        
        # Calculate percentile
        all_iv = df['IV_Proxy'].dropna()
        iv_percentile = (all_iv < recent_iv).mean() * 100
        
        # Classify regime
        if recent_iv < 0.40:
            regime = 'low_vol'
            recommendation = "Low IV environment. Wait for better entry. Current premiums are cheap."
        elif recent_iv < 0.55:
            regime = 'normal'
            recommendation = "Normal conditions. Standard parameters apply. Look for 3%+ dips."
        elif recent_iv < 0.75:
            regime = 'high_vol'
            recommendation = "Elevated IV. Good premium selling environment. Be more aggressive."
        else:
            regime = 'crisis'
            recommendation = "Crisis-level IV. Maximum premium but maximum risk. Size down."
        
        return MarketRegime(
            regime=regime,
            avg_iv=round(recent_iv, 3),
            iv_percentile=round(iv_percentile, 1),
            recommendation=recommendation,
        )
    
    def generate_health_report(
        self,
        lookback_days: int = 180,
    ) -> str:
        """
        Generate quarterly health check report.
        
        Returns:
            Formatted report string
        """
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # Get recommendation
        rec = self.run_smart_optimization(start, end)
        
        # Detect regime
        regime = self.detect_regime()
        
        lines = [
            "=" * 60,
            "📊 STRATEGY HEALTH CHECK",
            f"   {datetime.now().strftime('%Y-%m-%d')}",
            "=" * 60,
            "",
            "MARKET REGIME",
            "-" * 40,
            f"  Current: {regime.regime.upper()}",
            f"  IV Level: {regime.avg_iv*100:.0f}% ({regime.iv_percentile:.0f}th percentile)",
            f"  → {regime.recommendation}",
            "",
            "PARAMETER RECOMMENDATION",
            "-" * 40,
            f"  IV Threshold: {rec.iv_threshold*100:.0f}%",
            f"  Target Delta: {rec.target_delta}",
            f"  Profit Target: {rec.profit_target*100:.0f}%",
            f"  Spread Width: {rec.spread_width*100:.0f}%",
            "",
            f"  Confidence: {rec.confidence.upper()}",
            f"  Reasoning: {rec.reasoning}",
            "",
            "EXPECTED PERFORMANCE",
            "-" * 40,
            f"  Sharpe Ratio: {rec.sharpe_ratio:.2f}",
            f"  Win Rate: {rec.win_rate:.0f}%",
            f"  Avg P&L: ${rec.avg_pnl:.2f}",
            "",
        ]
        
        # Action items
        lines.extend([
            "ACTION ITEMS",
            "-" * 40,
        ])
        
        if rec.confidence == 'high':
            lines.append("  ✅ Parameters validated. Continue current strategy.")
        elif rec.confidence == 'medium':
            lines.append("  ⚠️ Consider adjusting parameters based on recommendations.")
        else:
            lines.append("  ⚠️ Insufficient data. Stick with conservative defaults.")
        
        if regime.regime == 'crisis':
            lines.append("  🔴 REDUCE position sizes during crisis regime.")
        elif regime.regime == 'low_vol':
            lines.append("  🟡 Low IV. Be patient, wait for better setups.")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)
    
    def should_run_check(self, days_between: int = 90) -> bool:
        """Check if it's time for a quarterly health check."""
        if self.last_check is None:
            return True
        
        elapsed = (datetime.now() - self.last_check).days
        return elapsed >= days_between


def run_health_check():
    """Run strategy health check from command line."""
    monitor = StrategyMonitor(symbol="BITO")
    print(monitor.generate_health_report())


if __name__ == "__main__":
    run_health_check()
