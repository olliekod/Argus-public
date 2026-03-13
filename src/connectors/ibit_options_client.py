"""
Crypto ETF Options Client
=========================

Fetches options chain data for crypto ETFs (IBIT, BITO, etc.) using yfinance.
Provides IV Rank, available strikes, and chain data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
from zoneinfo import ZoneInfo
import time

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    raise ImportError("Required: pip install yfinance pandas numpy")

logger = logging.getLogger(__name__)


class IBITOptionsClient:
    """
    Client for fetching options data from Yahoo Finance.
    
    Note: Data has 15-20 minute delay, which is acceptable for
    multi-day put spread strategies.
    """
    
    CACHE_TTL_SECONDS = 900  # 15 minutes
    
    def __init__(self, symbol: str = "IBIT"):
        """Initialize the options client for a given symbol."""
        self.symbol = symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        self._cache: Dict[str, Tuple[float, any]] = {}
        self._iv_history: List[float] = []
        self.last_poll_ts: Optional[float] = None
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures: int = 0
        self.request_count: int = 0
        self.error_count: int = 0
        self.last_latency_ms: Optional[float] = None
        self.avg_latency_ms: Optional[float] = None
        logger.info(f"{self.symbol} Options Client initialized")

    def _record_request(self, success: bool, latency_ms: float, error: Optional[str] = None) -> None:
        self.request_count += 1
        self.last_poll_ts = time.time()
        self.last_latency_ms = latency_ms
        self.avg_latency_ms = (
            latency_ms if self.avg_latency_ms is None
            else (latency_ms * 0.2) + (self.avg_latency_ms * 0.8)
        )
        if success:
            self.last_success_ts = time.time()
            self.consecutive_failures = 0
            self.last_error = None
        else:
            self.error_count += 1
            self.consecutive_failures += 1
            self.last_error = error
    
    def _get_cached(self, key: str) -> Optional[any]:
        """Get cached value if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self.CACHE_TTL_SECONDS:
                return value
        return None
    
    def _set_cache(self, key: str, value: any) -> None:
        """Set cache value with current timestamp."""
        self._cache[key] = (time.time(), value)
    
    def get_current_price(self) -> float:
        """Get current IBIT price."""
        cached = self._get_cached("price")
        if cached:
            return cached

        start = time.perf_counter()
        try:
            info = self.ticker.info
            price = info.get('regularMarketPrice') or info.get('previousClose', 0)
            self._set_cache("price", price)
            self._record_request(True, (time.perf_counter() - start) * 1000)
            return float(price)
        except Exception as e:
            self._record_request(False, (time.perf_counter() - start) * 1000, str(e))
            logger.error(f"Error fetching IBIT price: {e}")
            return 0.0
    
    def get_available_expirations(self) -> List[str]:
        """
        Get all available option expiration dates.
        
        Returns:
            List of expiration dates as strings (YYYY-MM-DD)
        """
        cached = self._get_cached("expirations")
        if cached:
            return cached
        
        start = time.perf_counter()
        try:
            expirations = list(self.ticker.options)
            self._set_cache("expirations", expirations)
            self._record_request(True, (time.perf_counter() - start) * 1000)
            logger.debug(f"Found {len(expirations)} IBIT expirations")
            return expirations
        except Exception as e:
            self._record_request(False, (time.perf_counter() - start) * 1000, str(e))
            logger.error(f"Error fetching expirations: {e}")
            return []
    
    def get_expirations_in_range(
        self, 
        min_dte: int = 7, 
        max_dte: int = 14
    ) -> List[Tuple[str, int]]:
        """
        Get expirations within DTE range.
        
        Args:
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            List of (expiration_date, dte) tuples
        """
        eastern = ZoneInfo("America/New_York")
        today = datetime.now(eastern).date()
        expirations = self.get_available_expirations()
        
        result = []
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if min_dte <= dte <= max_dte:
                    result.append((exp_str, dte))
            except ValueError:
                continue
        
        # Sort by DTE
        result.sort(key=lambda x: x[1])
        return result
    
    def get_options_chain(self, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get full options chain for an expiration.
        
        Args:
            expiration: Expiration date string (YYYY-MM-DD)
            
        Returns:
            Tuple of (calls_df, puts_df)
        """
        cache_key = f"chain_{expiration}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        start = time.perf_counter()
        try:
            chain = self.ticker.option_chain(expiration)
            self._set_cache(cache_key, (chain.calls, chain.puts))
            self._record_request(True, (time.perf_counter() - start) * 1000)
            logger.debug(f"Fetched chain for {expiration}: {len(chain.puts)} puts")
            return chain.calls, chain.puts
        except Exception as e:
            self._record_request(False, (time.perf_counter() - start) * 1000, str(e))
            logger.error(f"Error fetching chain for {expiration}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_puts_for_spread(
        self,
        expiration: str,
        target_delta: float = 0.18,
        spread_width: float = 2.0
    ) -> Optional[Dict]:
        """
        Find optimal put spread strikes.
        
        Args:
            expiration: Target expiration
            target_delta: Target delta for short put (~0.15-0.20)
            spread_width: Distance between strikes in dollars
            
        Returns:
            Dict with short_strike, long_strike, and chain data
        """
        _, puts = self.get_options_chain(expiration)
        if puts.empty:
            return None
        
        current_price = self.get_current_price()
        if not current_price:
            return None
        
        # Target OTM percentage based on delta
        # Delta ~0.18 is roughly 10-12% OTM
        otm_pct = 0.10 + (0.20 - target_delta) * 0.5  # Rough conversion
        target_strike = current_price * (1 - otm_pct)
        
        # Filter for quality quotes (no NaN, non-zero bid)
        valid_puts = puts.dropna(subset=['bid', 'ask', 'strike'])
        valid_puts = valid_puts[(valid_puts['bid'] > 0) & (valid_puts['ask'] > 0)]
        
        if valid_puts.empty:
            logger.warning(f"No valid put quotes found for {expiration} (all NaN or zero bid)")
            return None

        # Find closest strike to target
        puts_sorted = valid_puts.copy()
        puts_sorted['distance'] = abs(puts_sorted['strike'] - target_strike)
        puts_sorted = puts_sorted.sort_values('distance')
        
        short_put = puts_sorted.iloc[0]
        short_strike = float(short_put['strike'])
        
        # Find long strike (spread_width below short)
        long_target = short_strike - spread_width
        long_candidates = valid_puts[valid_puts['strike'] <= long_target].sort_values('strike', ascending=False)
        
        if long_candidates.empty:
            # Use next available lower strike
            lower_strikes = valid_puts[valid_puts['strike'] < short_strike].sort_values('strike', ascending=False)
            if lower_strikes.empty:
                return None
            long_put = lower_strikes.iloc[0]
        else:
            long_put = long_candidates.iloc[0]
        
        long_strike = float(long_put['strike'])
        actual_width = short_strike - long_strike
        
        # Calculate spread economics
        short_bid = float(short_put.get('bid', 0) or 0)
        short_ask = float(short_put.get('ask', 0) or 0)
        long_bid = float(long_put.get('bid', 0) or 0)
        long_ask = float(long_put.get('ask', 0) or 0)
        
        # Credit = Sell short (at bid) - Buy long (at ask)
        # Use mid prices for estimate
        short_mid = (short_bid + short_ask) / 2 if short_ask else short_bid
        long_mid = (long_bid + long_ask) / 2 if long_ask else long_bid
        
        net_credit = short_mid - long_mid
        max_risk = actual_width - net_credit
        
        # Guard against NaN/Inf in calculation
        import math
        if math.isnan(net_credit) or math.isnan(max_risk):
            logger.warning(f"NaN economics for {self.symbol} spread: credit={net_credit}, risk={max_risk}")
            return None
        
        # Get IV for short put
        short_iv = float(short_put.get('impliedVolatility', 0) or 0)
        
        return {
            'expiration': expiration,
            'short_strike': short_strike,
            'long_strike': long_strike,
            'spread_width': actual_width,
            'short_bid': short_bid,
            'short_ask': short_ask,
            'long_bid': long_bid,
            'long_ask': long_ask,
            'net_credit': round(net_credit, 2),
            'max_risk': round(max_risk, 2),
            'short_iv': short_iv,
            'short_volume': int(short_put.get('volume', 0) or 0),
            'short_oi': int(short_put.get('openInterest', 0) or 0),
            'underlying_price': current_price,
        }
    
    def get_atm_iv(self) -> float:
        """
        Get at-the-money implied volatility.
        
        Returns:
            ATM IV as decimal (e.g., 0.45 = 45%)
        """
        expirations = self.get_expirations_in_range(7, 30)
        if not expirations:
            expirations = self.get_expirations_in_range(1, 60)
        
        if not expirations:
            return 0.0
        
        # Use nearest expiration
        exp = expirations[0][0]
        _, puts = self.get_options_chain(exp)
        
        if puts.empty:
            return 0.0
        
        current_price = self.get_current_price()
        if not current_price:
            return 0.0
        
        # Find ATM put
        puts['distance'] = abs(puts['strike'] - current_price)
        atm_put = puts.sort_values('distance').iloc[0]
        
        iv = float(atm_put.get('impliedVolatility', 0) or 0)
        
        # Store for IV history
        self._iv_history.append(iv)
        if len(self._iv_history) > 252:  # Keep ~1 year
            self._iv_history = self._iv_history[-252:]
        
        return iv
    
    def get_iv_rank(self, current_iv: Optional[float] = None) -> float:
        """
        Calculate IV Rank (percentile over 52 weeks).
        
        Note: Since we don't have 52 weeks of history immediately,
        we approximate using available data or return 50% as neutral.
        
        Args:
            current_iv: Current IV (will fetch if not provided)
            
        Returns:
            IV Rank as percentage (0-100)
        """
        if current_iv is None:
            current_iv = self.get_atm_iv()
        
        if not current_iv:
            return 50.0  # Neutral if unknown
        
        # Try to get historical IV from ticker
        try:
            hist = self.ticker.history(period="1y")
            if not hist.empty and len(hist) > 20:
                # Use price volatility as proxy for IV range
                returns = hist['Close'].pct_change().dropna()
                realized_vol = returns.std() * np.sqrt(252)
                
                # Rough IV range estimation
                iv_low = realized_vol * 0.8
                iv_high = realized_vol * 1.5
                
                if iv_high > iv_low and not math.isnan(realized_vol):
                    iv_rank = ((current_iv - iv_low) / (iv_high - iv_low)) * 100
                    if math.isnan(iv_rank):
                        return 50.0
                    return max(0, min(100, iv_rank))
        except Exception as e:
            logger.debug(f"Could not calculate historical IV rank: {e}")
        
        # Fallback: Use known IBIT IV range (from research)
        # Low: 0.33, High: 0.62
        IV_LOW = 0.33
        IV_HIGH = 0.62
        
        if current_iv <= IV_LOW:
            return 0.0
        elif current_iv >= IV_HIGH:
            return 100.0
        else:
            return ((current_iv - IV_LOW) / (IV_HIGH - IV_LOW)) * 100
    
    def get_market_status(self) -> Dict:
        """
        Get current market status and IBIT overview.

        Returns:
            Dict with price, IV, IV rank, and market hours status
        """
        eastern = ZoneInfo("America/New_York")
        now = datetime.now(eastern)

        # Check if market is open (9:30 AM - 4:00 PM ET, weekdays)
        is_weekday = now.weekday() < 5
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_hours = is_weekday and market_open <= now <= market_close
        
        current_price = self.get_current_price()
        current_iv = self.get_atm_iv()
        iv_rank = self.get_iv_rank(current_iv)
        
        return {
            'symbol': self.symbol,
            'price': current_price,
            'iv': current_iv,
            'iv_pct': round(current_iv * 100, 1) if current_iv else 0,
            'iv_rank': round(iv_rank, 1),
            'is_market_hours': is_market_hours,
            'timestamp': now.isoformat(),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Return health for dashboard."""
        now = time.time()
        age = (now - self.last_success_ts) if self.last_success_ts else None
        if self.consecutive_failures > 0:
            status = "degraded"
        elif self.last_success_ts:
            status = "ok"
        else:
            status = "unknown"

        from ..core.status import build_status

        return build_status(
            name="ibit_options",
            type="batch",
            status=status,
            last_success_ts=self.last_success_ts,
            last_error=self.last_error,
            consecutive_failures=self.consecutive_failures,
            request_count=self.request_count,
            error_count=self.error_count,
            avg_latency_ms=round(self.avg_latency_ms, 2) if self.avg_latency_ms is not None else None,
            last_latency_ms=round(self.last_latency_ms, 2) if self.last_latency_ms is not None else None,
            last_poll_ts=self.last_poll_ts,
            age_seconds=round(age, 1) if age is not None else None,
            extras={
                "symbol": self.symbol,
                "cache_keys": len(self._cache),
            },
        )
    
    def get_bid_ask_quality(self, expiration: str, strike: float) -> Dict:
        """
        Get bid-ask spread quality for a specific option.
        
        Args:
            expiration: Expiration date string
            strike: Strike price
            
        Returns:
            Dict with bid, ask, spread, spread_pct, and quality rating
        """
        _, puts = self.get_options_chain(expiration)
        if puts.empty:
            return {'quality': 'unknown', 'spread_pct': None}
        
        # Find the option at this strike
        option = puts[puts['strike'] == strike]
        if option.empty:
            # Find closest strike
            puts['distance'] = abs(puts['strike'] - strike)
            option = puts.sort_values('distance').iloc[[0]]
        
        row = option.iloc[0]
        bid = float(row.get('bid', 0) or 0)
        ask = float(row.get('ask', 0) or 0)
        
        if bid <= 0 or ask <= 0:
            return {
                'bid': bid,
                'ask': ask,
                'spread': None,
                'spread_pct': None,
                'quality': 'no_quote'
            }
        
        spread = ask - bid
        mid = (bid + ask) / 2
        spread_pct = (spread / mid) * 100
        
        # Quality rating
        if spread_pct <= 5:
            quality = 'excellent'  # <5% spread
        elif spread_pct <= 10:
            quality = 'good'       # 5-10% spread
        elif spread_pct <= 20:
            quality = 'fair'       # 10-20% spread
        else:
            quality = 'poor'       # >20% spread
        
        return {
            'strike': strike,
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'spread': round(spread, 2),
            'spread_pct': round(spread_pct, 1),
            'quality': quality,
            'volume': int(row.get('volume', 0) or 0),
            'open_interest': int(row.get('openInterest', 0) or 0),
        }
    
    def get_spread_fill_estimate(
        self, 
        expiration: str, 
        short_strike: float, 
        long_strike: float
    ) -> Dict:
        """
        Estimate realistic fill for a put spread considering bid-ask.
        
        Args:
            expiration: Expiration date
            short_strike: Short put strike
            long_strike: Long put strike
            
        Returns:
            Dict with theoretical, realistic, and worst-case fills
        """
        short_quality = self.get_bid_ask_quality(expiration, short_strike)
        long_quality = self.get_bid_ask_quality(expiration, long_strike)
        
        if short_quality['quality'] == 'unknown' or long_quality['quality'] == 'unknown':
            return {'status': 'no_data'}
        
        # Theoretical (mid prices)
        theoretical = short_quality.get('mid', 0) - long_quality.get('mid', 0)
        
        # Realistic (sell short at mid-5%, buy long at mid+5%)
        realistic = (short_quality.get('mid', 0) * 0.95) - (long_quality.get('mid', 0) * 1.05)
        
        # Worst case (sell at bid, buy at ask)
        worst = short_quality.get('bid', 0) - long_quality.get('ask', 0)
        
        return {
            'status': 'ok',
            'theoretical_credit': round(theoretical, 3),
            'realistic_credit': round(realistic, 3),
            'worst_case_credit': round(worst, 3),
            'short_quality': short_quality['quality'],
            'long_quality': long_quality['quality'],
            'slippage_estimate_pct': round((theoretical - realistic) / theoretical * 100, 1) if theoretical > 0 else 0,
        }


# Test function
async def test_client():
    """Test the IBIT options client."""
    client = IBITOptionsClient()
    
    print("=" * 60)
    print("IBIT OPTIONS CLIENT TEST")
    print("=" * 60)
    
    # Market status
    status = client.get_market_status()
    print(f"\nMarket Status:")
    print(f"  Price: ${status['price']:.2f}")
    print(f"  IV: {status['iv_pct']:.1f}%")
    print(f"  IV Rank: {status['iv_rank']:.1f}%")
    print(f"  Market Open: {status['is_market_hours']}")
    
    # Expirations
    exps = client.get_expirations_in_range(7, 21)
    print(f"\nExpirations (7-21 DTE):")
    for exp, dte in exps[:5]:
        print(f"  {exp} ({dte} days)")
    
    # Put spread
    if exps:
        spread = client.get_puts_for_spread(exps[0][0])
        if spread:
            print(f"\nSuggested Put Spread ({exps[0][0]}):")
            print(f"  Short: ${spread['short_strike']:.0f} Put")
            print(f"  Long:  ${spread['long_strike']:.0f} Put")
            print(f"  Credit: ${spread['net_credit']:.2f}")
            print(f"  Max Risk: ${spread['max_risk']:.2f}")
            print(f"  Short IV: {spread['short_iv']*100:.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_client())
