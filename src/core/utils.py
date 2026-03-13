"""
Argus Utility Functions
=======================

Helper functions for statistical calculations, formatting, and common operations.
"""

import math
from typing import List, Optional, Union
from datetime import datetime, timezone


def calculate_z_score(value: float, values: List[float]) -> float:
    """
    Calculate z-score of a value relative to a list of values.
    
    Args:
        value: Current value to calculate z-score for
        values: Historical values to compare against
        
    Returns:
        Z-score (number of standard deviations from mean)
    """
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    std = calculate_std(values)
    
    if std == 0:
        return 0.0
    
    return (value - mean) / std


def calculate_std(values: List[float]) -> float:
    """
    Calculate standard deviation of a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Standard deviation
    """
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def calculate_mean(values: List[float]) -> float:
    """Calculate mean of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_volatility(
    prices: List[float],
    period_minutes: int = 60,
    annualize: bool = True
) -> float:
    """
    Calculate volatility from price series.
    
    Args:
        prices: List of prices (chronological order)
        period_minutes: Time period per price point
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility as percentage (e.g., 50.0 for 50%)
    """
    if len(prices) < 2:
        return 0.0
    
    # Calculate log returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            log_return = math.log(prices[i] / prices[i-1])
            returns.append(log_return)
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate standard deviation of returns
    std = calculate_std(returns)
    
    if annualize:
        # Annualize based on period
        periods_per_year = (365 * 24 * 60) / period_minutes
        std = std * math.sqrt(periods_per_year)
    
    # Convert to percentage
    return std * 100


def calculate_edge_after_costs(
    raw_edge_bps: float,
    slippage_bps: float = 5,
    fee_bps: float = 5,
    round_trips: int = 1
) -> float:
    """
    Calculate net edge after trading costs.
    
    Args:
        raw_edge_bps: Raw edge in basis points
        slippage_bps: Expected slippage per trade
        fee_bps: Taker fee per trade
        round_trips: Number of round trips (entry + exit = 1)
        
    Returns:
        Net edge in basis points
    """
    total_costs = (slippage_bps + fee_bps * 2) * round_trips
    return raw_edge_bps - total_costs


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: float = 365
) -> float:
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = calculate_mean(returns)
    std_return = calculate_std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_std = std_return * math.sqrt(periods_per_year)
    
    return (annual_return - risk_free_rate) / annual_std


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: List of equity values over time
        
    Returns:
        Maximum drawdown as percentage (e.g., -15.0 for 15% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        
        drawdown = (value - peak) / peak * 100
        if drawdown < max_dd:
            max_dd = drawdown
    
    return max_dd


def exponential_moving_average(
    values: List[float],
    period: int
) -> List[float]:
    """
    Calculate exponential moving average.
    
    Args:
        values: Input values
        period: EMA period
        
    Returns:
        List of EMA values (same length as input)
    """
    if not values:
        return []
    
    if period <= 0:
        return values.copy()
    
    multiplier = 2 / (period + 1)
    ema = [values[0]]
    
    for i in range(1, len(values)):
        ema_value = (values[i] * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_value)
    
    return ema


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Top number
        denominator: Bottom number
        default: Value to return if division is impossible
        
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2, symbol: str = "$") -> str:
    """Format value as currency string."""
    if value >= 0:
        return f"{symbol}{value:,.{decimals}f}"
    else:
        return f"-{symbol}{abs(value):,.{decimals}f}"


def format_bps(value: float) -> str:
    """Format value in basis points."""
    return f"{value:.1f} bps"


def format_large_number(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.2f}K"
    else:
        return f"{value:.2f}"


def parse_symbol(symbol: str) -> dict:
    """
    Parse a trading symbol into components.
    
    Examples:
        'BTC/USDT:USDT' -> {'base': 'BTC', 'quote': 'USDT', 'settle': 'USDT', 'is_perp': True}
        'BTC/USDT' -> {'base': 'BTC', 'quote': 'USDT', 'settle': None, 'is_perp': False}
    """
    result = {
        'base': None,
        'quote': None,
        'settle': None,
        'is_perp': False,
        'original': symbol
    }
    
    if ':' in symbol:
        main_part, settle = symbol.split(':')
        result['settle'] = settle
        result['is_perp'] = True
    else:
        main_part = symbol
    
    if '/' in main_part:
        base, quote = main_part.split('/')
        result['base'] = base
        result['quote'] = quote
    
    return result


def timestamp_now() -> str:
    """Get current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def timestamp_ago(minutes: int = 0, hours: int = 0, days: int = 0) -> str:
    """Get timestamp from some time ago."""
    from datetime import timedelta
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes, hours=hours, days=days)
    return dt.isoformat()


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def round_to_tick(price: float, tick_size: float) -> float:
    """Round price to nearest tick size."""
    if tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size


def calculate_position_size(
    capital: float,
    risk_percent: float,
    stop_loss_percent: float,
    max_position_percent: float = 100.0
) -> float:
    """
    Calculate position size based on risk.
    
    Args:
        capital: Total capital
        risk_percent: Percentage of capital to risk
        stop_loss_percent: Stop loss as percentage from entry
        max_position_percent: Maximum position as percentage of capital
        
    Returns:
        Position size in dollars
    """
    if stop_loss_percent <= 0:
        return 0.0
    
    # Calculate position size where risk_percent loss = stop_loss_percent move
    position = capital * (risk_percent / stop_loss_percent)
    
    # Cap at maximum
    max_position = capital * (max_position_percent / 100)
    return min(position, max_position)
