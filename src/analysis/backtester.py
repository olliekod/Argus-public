"""
Strategy Backtester v2
======================

Backtests the IBIT put spread strategy using historical data.
Uses BITO (ProShares Bitcoin Strategy ETF) as proxy since IBIT options are newer.

v2 Changes:
- $5,000 default account size
- Shows % returns on account (not just dollars)
- Looser entry criteria for more trades
- Optimizes for PROFIT, not win rate
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    raise ImportError("Required: pip install yfinance pandas numpy")

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A single backtested trade."""
    entry_date: str
    exit_date: str
    underlying_price: float
    short_strike: float
    long_strike: float
    spread_width: float
    
    # Per contract
    entry_credit: float
    exit_debit: float
    pnl_per_contract: float
    
    # With position sizing
    num_contracts: int
    total_pnl: float
    account_return_pct: float  # % return on account
    
    iv_at_entry: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Complete backtest results."""
    symbol: str
    start_date: str
    end_date: str
    account_size: float
    
    trades: List[BacktestTrade] = field(default_factory=list)
    
    # Summary stats
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    
    # P&L
    total_pnl: float = 0.0
    total_return_pct: float = 0.0  # Total % return on account
    avg_return_pct: float = 0.0    # Avg % return per trade
    
    # Best/worst
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # By IV bucket
    pnl_by_iv: Dict[str, Dict] = field(default_factory=dict)


class StrategyBacktester:
    """
    Backtests the put spread strategy on historical data.
    
    v2: Focused on PROFIT with percentage returns.
    """
    
    # Default account size
    DEFAULT_ACCOUNT_SIZE = 5000.0
    
    # Position sizing: risk 5% per trade
    MAX_RISK_PER_TRADE = 0.05
    
    # Default parameters
    DEFAULT_PARAMS = {
        'iv_threshold': 0.40,         # LOWERED: Enter when IV > 40%
        'price_drop_trigger': -0.02,  # LOWERED: Enter when price drops 2%+
        'target_delta': 0.18,         # Short strike delta
        'spread_width_pct': 0.05,     # 5% spread width
        'profit_target': 0.50,        # Close at 50% profit
        'time_exit_dte': 5,           # Close at 5 DTE
        'entry_dte': 14,              # Enter with ~14 DTE
        'use_gpu': True,              # Use GPU for PoP/Greeks if available
    }
    
    def __init__(
        self, 
        symbol: str = "BITO", 
        params: Dict = None,
        account_size: float = None,
    ):
        """
        Initialize backtester.
        
        Args:
            symbol: Ticker to backtest
            params: Strategy parameters
            account_size: Account size in dollars (default $5,000)
        """
        self.symbol = symbol
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.account_size = account_size or self.DEFAULT_ACCOUNT_SIZE
        
        logger.info(f"Backtester: {symbol}, ${self.account_size:,.0f} account")
    
    def fetch_historical_data(
        self, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Fetch historical price and IV data."""
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['Return'] = df['Close'].pct_change()
        
        # Rolling volatility as IV proxy
        df['RealizedVol'] = df['Return'].rolling(20).std() * np.sqrt(252)
        df['IV_Proxy'] = df['RealizedVol']
        
        # Boost IV during drops
        df['5d_Return'] = df['Close'].pct_change(5)
        df.loc[df['5d_Return'] < -0.05, 'IV_Proxy'] *= 1.3
        df.loc[df['5d_Return'] < -0.10, 'IV_Proxy'] *= 1.5
        
        logger.info(f"Fetched {len(df)} days of data")
        return df
    
    def find_entry_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Find entry signals - LOOSER criteria for more trades."""
        signals = []
        
        iv_threshold = self.params['iv_threshold']
        drop_trigger = self.params['price_drop_trigger']
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            
            # Check IV threshold
            if pd.isna(row['IV_Proxy']) or row['IV_Proxy'] < iv_threshold:
                continue
            
            # Check price drop (can be 1-day OR 5-day)
            daily_drop = row['Return'] if not pd.isna(row['Return']) else 0
            weekly_drop = row['5d_Return'] if not pd.isna(row['5d_Return']) else 0
            
            # Entry if EITHER daily drop OR weekly drop meets threshold
            if daily_drop > drop_trigger and weekly_drop > drop_trigger * 2:
                continue  # Neither condition met
            
            signals.append({
                'date': row['Date'],
                'price': row['Close'],
                'iv': row['IV_Proxy'],
                'return': daily_drop,
            })
        
        logger.info(f"Found {len(signals)} entry signals")
        return signals
    
    def calculate_position_size(self, spread_width: float, price: float) -> int:
        """
        Calculate number of contracts based on 5% max risk.
        
        Args:
            spread_width: Width of spread in dollars
            price: Current underlying price
            
        Returns:
            Number of contracts
        """
        max_risk = spread_width * 100  # Max risk per contract
        capital_at_risk = self.account_size * self.MAX_RISK_PER_TRADE
        
        num_contracts = int(capital_at_risk / max_risk)
        return max(1, num_contracts)  # Minimum 1 contract
    
    def simulate_trade(
        self, 
        entry: Dict, 
        df: pd.DataFrame
    ) -> Optional[BacktestTrade]:
        """Simulate a single trade from entry to exit."""
        entry_dte = self.params['entry_dte']
        entry_date = entry['date']
        entry_price = entry['price']
        entry_iv = entry['iv']
        
        # Calculate strikes
        otm_pct = 0.10 + (0.20 - self.params['target_delta']) * 0.5
        short_strike = round(entry_price * (1 - otm_pct), 0)
        long_strike = round(short_strike - (entry_price * self.params['spread_width_pct']), 0)
        spread_width = short_strike - long_strike
        
        if spread_width <= 0:
            return None
        
        # GPU engine state
        gpu = None
        greeks = None
        
        # Estimate entry credit using GPU if available
        use_gpu = self.params.get('use_gpu', True)
        if use_gpu:
            try:
                from src.analysis.gpu_engine import get_gpu_engine
                from src.analysis.greeks_engine import GreeksEngine
                gpu = get_gpu_engine()
                greeks = GreeksEngine()
                
                # Use Black-Scholes via GPU for entry credit
                # Mid-price estimate
                T = GreeksEngine.dte_to_years(entry_dte)
                entry_credit = greeks.probability_of_profit(
                    entry_price, short_strike, 0, T, entry_iv, use_gpu=False
                ) / 100 * spread_width * 0.4 # Heuristic fallback
                
                # Better: use batch greeks to price the options
                prices = gpu.batch_greeks(entry_price, [short_strike, long_strike], T, [entry_iv, entry_iv])
                # Note: gpu_engine needs a 'price' method, but for now we'll stick to a better heuristic
                # or add price to gpu_engine if needed. 
                # For now let's use a refined heuristic since we are in a tight loop.
                entry_credit = min(spread_width * 0.35, spread_width * 0.25 * (entry_iv / 0.40))
            except Exception:
                entry_credit = min(spread_width * 0.35, spread_width * 0.25 * (entry_iv / 0.40))
        else:
            entry_credit = min(spread_width * 0.35, spread_width * 0.25 * (entry_iv / 0.40))
        
        # Position sizing
        num_contracts = self.calculate_position_size(spread_width, entry_price)
        
        # Find exit
        profit_target = self.params['profit_target']
        time_exit_dte = self.params['time_exit_dte']
        
        entry_idx = df[df['Date'] == entry_date].index
        if len(entry_idx) == 0:
            return None
        entry_idx = entry_idx[0]
        
        exit_date = None
        exit_debit = None
        exit_reason = None
        
        for days_held in range(1, entry_dte + 1):
            if entry_idx + days_held >= len(df):
                break
            
            future_row = df.iloc[entry_idx + days_held]
            future_price = future_row['Close']
            future_date = future_row['Date']
            future_iv = future_row['IV_Proxy']
            dte_remaining = entry_dte - days_held
            
            # Use GPU for real-time spread valuation during backtest
            if use_gpu and gpu and dte_remaining > 0:
                try:
                    T_rem = dte_remaining / 365.0
                    # Simplified valuation for speed: use delta/theta decay
                    # Price = Entry Credit - (Theta * days) + (Delta * price_change)
                    price_change = future_price - entry_price
                    # We can use the GPU to get the precise Delta/Theta at each step
                    g_res = gpu.batch_greeks(future_price, [short_strike], T_rem, [future_iv])
                    delta = g_res['delta'][0]
                    theta = g_res['theta'][0]
                    
                    # Estimate value based on delta/theta from entry
                    # This is much faster than full pricing in a backtest loop
                    current_value = max(0, min(spread_width, entry_credit - (delta * price_change)))
                except Exception:
                    time_decay_factor = (entry_dte - days_held) / entry_dte
                    current_value = entry_credit * time_decay_factor * 0.7
            else:
                time_decay_factor = (entry_dte - days_held) / entry_dte
                current_value = entry_credit * time_decay_factor * 0.7
            
            # Check 50% profit target
            if current_value <= entry_credit * profit_target:
                exit_date = future_date
                exit_debit = current_value
                exit_reason = 'profit_target'
                break
            
            # Check time exit
            if dte_remaining <= time_exit_dte:
                exit_date = future_date
                exit_debit = current_value
                exit_reason = 'time_exit'
                break
        
        # Expiration if no early exit
        if exit_date is None:
            if entry_idx + entry_dte < len(df):
                exp_row = df.iloc[entry_idx + entry_dte]
            else:
                exp_row = df.iloc[-1]
            
            exp_price = exp_row['Close']
            exit_date = exp_row['Date']
            
            if exp_price > short_strike:
                exit_debit = 0
                exit_reason = 'expired_win'
            elif exp_price < long_strike:
                exit_debit = spread_width
                exit_reason = 'expired_loss'
            else:
                exit_debit = short_strike - exp_price
                exit_reason = 'expired_loss'
        
        # Calculate P&L
        pnl_per_contract = (entry_credit - exit_debit) * 100
        total_pnl = pnl_per_contract * num_contracts
        account_return_pct = (total_pnl / self.account_size) * 100
        
        return BacktestTrade(
            entry_date=str(entry_date),
            exit_date=str(exit_date),
            underlying_price=entry_price,
            short_strike=short_strike,
            long_strike=long_strike,
            spread_width=spread_width,
            entry_credit=round(entry_credit, 2),
            exit_debit=round(exit_debit, 2),
            pnl_per_contract=round(pnl_per_contract, 2),
            num_contracts=num_contracts,
            total_pnl=round(total_pnl, 2),
            account_return_pct=round(account_return_pct, 2),
            iv_at_entry=round(entry_iv, 3),
            exit_reason=exit_reason,
        )
    
    def run_backtest(
        self, 
        start_date: str, 
        end_date: str
    ) -> BacktestResult:
        """Run full backtest."""
        logger.info(f"Running backtest: {start_date} to {end_date}")
        
        df = self.fetch_historical_data(start_date, end_date)
        signals = self.find_entry_signals(df)
        
        # Batch Calculate PoP/Touch for all signals using GPU
        if self.params.get('use_gpu', True):
            try:
                from src.analysis.gpu_engine import get_gpu_engine
                gpu = get_gpu_engine()
                logger.info(f"GPU batch calculating PoP for {len(signals)} candidate trades...")
                
                for signal in signals:
                    T = self.params['entry_dte'] / 365.0
                    # For backtesting, we use a slightly wider margin for strikes
                    short_strike = signal['price'] * 0.90
                    long_strike = signal['price'] * 0.85
                    credit = signal['price'] * 0.05
                    
                    # We'll calculate a single PoP/Touch per signal entry point
                    # This helps filter candidates by touch risk
                    res = gpu.monte_carlo_pop_heston(
                        S=signal['price'], short_strike=short_strike, long_strike=long_strike,
                        credit=credit, T=T, v0=signal['iv']**2, simulations=100_000 # Fast batch
                    )
                    signal['pop'] = res['pop']
                    signal['touch_risk'] = res['prob_of_touch_stop']
            except Exception as e:
                logger.warning(f"Batch PoP calculation failed: {e}")
        
        trades = []
        last_exit_date = None
        
        for signal in signals:
            if last_exit_date and signal['date'] <= last_exit_date:
                continue
            
            # Filter by Heston PoP/Touch if available
            if 'pop' in signal and signal['pop'] < 65:
                continue
            if 'touch_risk' in signal and signal['touch_risk'] > 30:
                continue
                
            trade = self.simulate_trade(signal, df)
            if trade:
                trades.append(trade)
                last_exit_date = datetime.strptime(trade.exit_date, "%Y-%m-%d").date()
        
        # Build result
        result = BacktestResult(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            account_size=self.account_size,
            trades=trades,
        )
        
        if trades:
            result.total_trades = len(trades)
            result.winners = sum(1 for t in trades if t.total_pnl > 0)
            result.losers = sum(1 for t in trades if t.total_pnl <= 0)
            result.win_rate = (result.winners / result.total_trades) * 100
            result.total_pnl = sum(t.total_pnl for t in trades)
            result.total_return_pct = (result.total_pnl / self.account_size) * 100
            result.avg_return_pct = result.total_return_pct / result.total_trades
            result.best_trade_pct = max(t.account_return_pct for t in trades)
            result.worst_trade_pct = min(t.account_return_pct for t in trades)
            
            # Max drawdown %
            cumulative = 0
            peak = 0
            max_dd = 0
            for trade in trades:
                cumulative += trade.total_pnl
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown_pct = (max_dd / self.account_size) * 100
            
            # P&L by IV bucket
            iv_buckets = {'40-50%': [], '50-60%': [], '60-70%': [], '70%+': []}
            for trade in trades:
                iv_pct = trade.iv_at_entry * 100
                if iv_pct < 50:
                    iv_buckets['40-50%'].append(trade)
                elif iv_pct < 60:
                    iv_buckets['50-60%'].append(trade)
                elif iv_pct < 70:
                    iv_buckets['60-70%'].append(trade)
                else:
                    iv_buckets['70%+'].append(trade)
            
            for bucket, bucket_trades in iv_buckets.items():
                if bucket_trades:
                    total = sum(t.total_pnl for t in bucket_trades)
                    avg_ret = sum(t.account_return_pct for t in bucket_trades) / len(bucket_trades)
                    result.pnl_by_iv[bucket] = {
                        'count': len(bucket_trades),
                        'total_pnl': total,
                        'total_return_pct': (total / self.account_size) * 100,
                        'avg_return_pct': avg_ret,
                    }
        
        logger.info(f"Backtest: {result.total_trades} trades, "
                    f"{result.total_return_pct:.1f}% return")
        
        return result
    
    def format_report(self, result: BacktestResult) -> str:
        """Format backtest results as readable report."""
        lines = [
            "=" * 60,
            "📈 STRATEGY BACKTEST REPORT",
            "=" * 60,
            "",
            f"Symbol: {result.symbol}",
            f"Period: {result.start_date} to {result.end_date}",
            f"Account Size: ${result.account_size:,.0f}",
            "",
            "PARAMETERS",
            "-" * 40,
            f"  IV Threshold: {self.params['iv_threshold']*100:.0f}%",
            f"  Drop Trigger: {self.params['price_drop_trigger']*100:.1f}%",
            f"  Target Delta: {self.params['target_delta']}",
            f"  Profit Target: {self.params['profit_target']*100:.0f}%",
            "",
            "PERFORMANCE",
            "-" * 40,
            f"  Total Trades: {result.total_trades}",
            f"  Winners: {result.winners} | Losers: {result.losers}",
            f"  Win Rate: {result.win_rate:.0f}%",
            "",
            "RETURNS (% of Account)",
            "-" * 40,
            f"  Total Return: {result.total_return_pct:+.1f}%",
            f"  Avg per Trade: {result.avg_return_pct:+.2f}%",
            f"  Best Trade: {result.best_trade_pct:+.2f}%",
            f"  Worst Trade: {result.worst_trade_pct:+.2f}%",
            f"  Max Drawdown: {result.max_drawdown_pct:.2f}%",
            "",
            f"  Total P&L: ${result.total_pnl:,.2f}",
            "",
        ]
        
        if result.pnl_by_iv:
            lines.extend([
                "BY IV BUCKET",
                "-" * 40,
            ])
            for bucket, stats in result.pnl_by_iv.items():
                lines.append(
                    f"  {bucket}: {stats['count']} trades, "
                    f"{stats['total_return_pct']:+.1f}% total, "
                    f"{stats['avg_return_pct']:+.2f}%/trade"
                )
            lines.append("")
        
        # Exit reasons
        exit_counts = {}
        for trade in result.trades:
            exit_counts[trade.exit_reason] = exit_counts.get(trade.exit_reason, 0) + 1
        
        if exit_counts:
            lines.extend([
                "EXIT REASONS",
                "-" * 40,
            ])
            for reason, count in exit_counts.items():
                lines.append(f"  {reason}: {count}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def run_backtest_cli():
    """Run backtest from command line."""
    print("=" * 60)
    print("📈 IBIT PUT SPREAD STRATEGY BACKTEST")
    print("    Using BITO as proxy for IBIT")
    print("=" * 60)
    
    # $5,000 account, looser criteria
    params = {
        'iv_threshold': 0.40,         # LOWERED from 0.50
        'price_drop_trigger': -0.02,  # LOWERED from -0.03
        'target_delta': 0.18,
        'spread_width_pct': 0.05,
        'profit_target': 0.50,
        'time_exit_dte': 5,
        'entry_dte': 14,
    }
    
    backtester = StrategyBacktester(
        symbol="BITO", 
        params=params,
        account_size=5000.0,
    )
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Account: $5,000")
    print("Fetching data and running simulation...\n")
    
    result = backtester.run_backtest(start_date, end_date)
    print(backtester.format_report(result))
    
    # Sample trades with % returns
    if result.trades:
        print("\n📋 SAMPLE TRADES:")
        print("-" * 60)
        for trade in result.trades[:10]:
            win = "✅" if trade.total_pnl > 0 else "❌"
            print(f"  {win} {trade.entry_date}: "
                  f"${trade.short_strike:.0f}/${trade.long_strike:.0f} x{trade.num_contracts} "
                  f"→ {trade.account_return_pct:+.2f}% ({trade.exit_reason})")
        
        # Monthly breakdown
        print("\n📅 MONTHLY RETURNS:")
        print("-" * 60)
        monthly = {}
        for trade in result.trades:
            month = trade.entry_date[:7]
            monthly[month] = monthly.get(month, 0) + trade.account_return_pct
        
        for month, ret in sorted(monthly.items()):
            bar = "█" * int(abs(ret) * 2) if abs(ret) > 0.5 else "▪"
            print(f"  {month}: {ret:+6.2f}% {bar}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_backtest_cli()
