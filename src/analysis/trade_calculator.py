# Created by Oliver Meihls

# Generates precise ETF put spread trade recommendations.
# Includes dynamic position sizing based on Probability of Profit.

import logging
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from ..connectors.ibit_options_client import IBITOptionsClient
from ..analysis.greeks_engine import GreeksEngine

logger = logging.getLogger(__name__)


@dataclass
class TradeRecommendation:
    # Complete trade recommendation.
    # Underlying info
    symbol: str
    underlying_price: float
    
    # Market conditions
    iv_rank: float
    btc_change_24h: float  # Regime proxy change
    symbol_change_24h: float
    
    # Trade structure
    expiration: str
    dte: int
    short_strike: float
    long_strike: float
    spread_width: float
    
    # Economics
    net_credit: float
    max_risk: float
    break_even: float
    risk_reward_ratio: float
    
    # Greeks
    net_delta: float
    net_theta: float
    net_vega: float
    
    # Probabilities
    probability_of_profit: float
    probability_of_touch_stop: float
    
    # Position sizing
    account_size: float
    position_size_pct: float
    num_contracts: int
    capital_at_risk: float
    
    # Exit plan
    profit_target: float  # Close at this credit (50% of max)
    time_exit_dte: int    # Close at this DTE
    
    # Warnings
    warnings: list
    
    # Timestamp
    generated_at: str
    
    # Metadata
    proxy_greeks: Optional[Dict[str, float]] = field(default=None)


class TradeCalculator:
    # Generates complete trade recommendations for ETF put spreads.
    #
    # Uses dynamic position sizing based on Probability of Profit:
    # - 80%+ PoP: 10% of account
    # - 70-80% PoP: 7% of account
    # - 60-70% PoP: 5% of account
    # - 50-60% PoP: 3% of account
    # - <50% PoP: NO TRADE
    
    # Position sizing parameters
    MIN_POSITION_SIZE = 0.03   # 3% minimum
    MAX_POSITION_SIZE = 0.10   # 10% maximum
    
    # PoP-based sizing tiers
    SIZING_TIERS = [
        (80, 0.10),  # 80%+ PoP -> 10%
        (70, 0.07),  # 70-80% PoP -> 7%
        (60, 0.05),  # 60-70% PoP -> 5%
        (50, 0.03),  # 50-60% PoP -> 3%
    ]
    
    # Trade parameters
    DEFAULT_SPREAD_WIDTH = 2.0
    HIGH_IV_SPREAD_WIDTH = 3.0
    HIGH_IV_THRESHOLD = 70  # IV Rank above this -> widen spread
    
    TARGET_DELTA = 0.18  # ~82% PoP target
    
    MIN_DTE = 7
    MAX_DTE = 14
    TIME_EXIT_DTE = 5  # Close if not profitable by this DTE
    
    PROFIT_TARGET_PCT = 0.50  # Close at 50% of max profit
    
    MIN_IV_RANK = 50  # Don't trade below this
    
    def __init__(
        self, 
        account_size: float = 3000.0, 
        symbol: str = "IBIT",
        iv_consensus: Optional[Any] = None
    ):
        # Initialize trade calculator.
        #
        # Args:
        # account_size: Account size in dollars (default $3,000)
        # symbol: Symbol to trade (IBIT, BITO, etc.)
        # iv_consensus: Shared IVConsensusEngine for real-time IV/Greeks
        self.account_size = account_size
        self.symbol = symbol.upper()
        self.options_client = IBITOptionsClient(symbol=symbol)
        self.greeks_engine = GreeksEngine()
        self.iv_consensus = iv_consensus
        self._last_spread_warning_ts: float = 0.0
        
        logger.info(f"{symbol} Trade Calculator initialized (account: ${account_size:,.0f}, consensus: {iv_consensus is not None})")
    
    def get_position_size_pct(self, pop: float) -> float:
        # Get position size percentage based on Probability of Profit.
        #
        # Args:
        # pop: Probability of Profit (0-100)
        #
        # Returns:
        # Position size as decimal (e.g., 0.07 for 7%)
        for threshold, size in self.SIZING_TIERS:
            if pop >= threshold:
                return size
        
        # Below 50% PoP
        return 0.0
    
    def calculate_num_contracts(
        self, 
        max_risk_per_contract: float, 
        position_size_pct: float
    ) -> int:
        # Calculate number of contracts based on position sizing.
        #
        # Args:
        # max_risk_per_contract: Max loss per contract in dollars
        # position_size_pct: Portfolio % to allocate
        #
        # Returns:
        # Number of contracts (minimum 1 if trade is valid)
        if position_size_pct <= 0:
            return 0
        
        import math
        if math.isnan(max_risk_per_contract) or max_risk_per_contract <= 0:
            return 0
        
        max_capital_at_risk = self.account_size * position_size_pct
        
        # Each contract represents 100 shares
        risk_per_contract = max_risk_per_contract * 100
        
        num_contracts = int(max_capital_at_risk / risk_per_contract)
        
        # Guard against zero or negative contracts
        return max(0, num_contracts)
    
    def generate_recommendation(
        self,
        btc_change_24h: float = 0.0,
        symbol_change_24h: float = 0.0,
        force: bool = False,
        proxy_greeks: Optional[Dict[str, float]] = None,
    ) -> Optional[TradeRecommendation]:
        # Generate a complete trade recommendation.
        #
        # Args:
        # btc_change_24h: Sentiment/Regime proxy change (e.g. BTC for crypto)
        # symbol_change_24h: Target ETF 24h price change
        # force: If True, generate even if IV Rank is low
        #
        # Returns:
        # TradeRecommendation or None if conditions not met
        warnings = []
        
        # Get market status
        status = self.options_client.get_market_status()
        current_iv = status['iv']
        iv_rank = status['iv_rank']
        underlying_price = status['price']

        iv_source = "yahoo_delayed"
        
        # Override with consensus if available
        if self.iv_consensus:
            # We use a very recent timestamp for consensus lookup
            now_ms = int(time.time() * 1000)
            # Try to get ATM consensus for the nearest relevant expiration
            # Note: We don't have the expiration yet, so we'll do a second pass after selecting expiration
            # or just use the ticker-level IV rank proxy from consensus if implemented.
            # For now, we'll fetch the expirations first to know what to ask consensus for.
            pass

        # Get available expirations
        expirations = self.options_client.get_expirations_in_range(
            self.MIN_DTE, self.MAX_DTE
        )
        
        if not expirations:
            # Try wider range
            expirations = self.options_client.get_expirations_in_range(5, 21)
            if expirations:
                warnings.append("⚠️ Using expiration outside ideal 7-14 DTE range")
        
        if not expirations:
            logger.warning("No suitable expirations found")
            return None
        
        # Use first (nearest) expiration
        expiration, dte = expirations[0]
        
        # RE-EVALUATE IV WITH CONSENSUS FOR SPECIFIC EXPIRATION
        if self.iv_consensus:
             from datetime import datetime, timezone
             exp_dt = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
             exp_ms = int(exp_dt.timestamp() * 1000)
             
             res = self.iv_consensus.get_atm_consensus(
                 underlying=self.symbol,
                 option_type="PUT",
                 expiration_ms=exp_ms,
                 as_of_ms=int(time.time() * 1000)
             )
             if res.consensus_iv:
                 current_iv = res.consensus_iv
                 iv_source = f"consensus_{res.iv_source_used}"
                 # Recalculate IV Rank if we have enough data in consensus hist, 
                 # otherwise Yahoo rank is a better proxy than nothing.
                 # For now, keep Yahoo's IV Rank as the 'regime' but use consensus IV for pricing.

        # Check IV Rank threshold
        if not force and iv_rank < self.MIN_IV_RANK:
            logger.info(f"IV Rank {iv_rank:.1f}% below threshold {self.MIN_IV_RANK}%")
            return None
        
        if iv_rank < self.MIN_IV_RANK:
            warnings.append(f"⚠️ IV Rank {iv_rank:.1f}% below ideal threshold")
        
        # Check market hours
        if not status['is_market_hours']:
            warnings.append("⚠️ Market is closed - prices may be stale")
        
        # Determine spread width based on IV
        spread_width = self.HIGH_IV_SPREAD_WIDTH if iv_rank >= self.HIGH_IV_THRESHOLD else self.DEFAULT_SPREAD_WIDTH
        
        # Get spread data
        spread_data = self.options_client.get_puts_for_spread(
            expiration,
            target_delta=self.TARGET_DELTA,
            spread_width=spread_width,
        )
        
        if not spread_data:
            now = time.time()
            if now - self._last_spread_warning_ts > 60:
                logger.warning("Could not find suitable spread")
                self._last_spread_warning_ts = now
            return None
        
        short_strike = spread_data['short_strike']
        long_strike = spread_data['long_strike']
        actual_width = spread_data['spread_width']
        net_credit = spread_data['net_credit']
        max_risk = spread_data['max_risk']
        short_iv = spread_data['short_iv'] or current_iv
        
        # Calculate break-even and risk/reward
        break_even = short_strike - net_credit
        risk_reward = net_credit / max_risk if max_risk > 0 else 0
        
        # Calculate Greeks
        T = GreeksEngine.dte_to_years(dte)
        spread_greeks = self.greeks_engine.calculate_spread_greeks(
            underlying_price, short_strike, long_strike, T, short_iv
        )
        
        # Calculate Probability of Profit
        # Use GPU Monte Carlo with Heston stochastic volatility
        pop_data = self.greeks_engine.probability_of_profit(
            underlying_price, 
            short_strike, 
            net_credit, 
            T, 
            short_iv,
            long_strike=long_strike,
            use_gpu=True,
            use_heston=True
        )
        pop = pop_data['pop']
        prob_touch_stop = pop_data.get('touch_stop_stop', pop_data.get('prob_of_touch_stop', 0.0))
        
        # Determine position size
        position_size_pct = self.get_position_size_pct(pop)
        
        if position_size_pct <= 0:
            logger.info(f"PoP {pop:.1f}% too low for trade")
            warnings.append(f"⚠️ PoP {pop:.1f}% below 50% threshold - consider skipping")
            position_size_pct = self.MIN_POSITION_SIZE  # Allow override
        
        # Guard against NaN values in critical economics
        if any(math.isnan(v) for v in [net_credit, max_risk, break_even, pop, prob_touch_stop]):
            logger.error(f"Cannot generate recommendation for {self.symbol}: NaN values detected")
            return None
        
        # Calculate contracts
        num_contracts = self.calculate_num_contracts(max_risk, position_size_pct)
        capital_at_risk = num_contracts * max_risk * 100
        
        # Profit target (50% of credit)
        profit_target = net_credit * self.PROFIT_TARGET_PCT
        
        return TradeRecommendation(
            symbol=self.symbol,
            underlying_price=underlying_price,
            iv_rank=iv_rank,
            btc_change_24h=btc_change_24h,
            symbol_change_24h=symbol_change_24h,
            expiration=expiration,
            dte=dte,
            short_strike=short_strike,
            long_strike=long_strike,
            spread_width=actual_width,
            net_credit=net_credit,
            max_risk=max_risk,
            break_even=break_even,
            risk_reward_ratio=risk_reward,
            net_delta=spread_greeks.net_delta,
            net_theta=spread_greeks.net_theta,
            net_vega=spread_greeks.net_vega,
            proxy_greeks=proxy_greeks,
            probability_of_profit=pop,
            probability_of_touch_stop=prob_touch_stop,
            account_size=self.account_size,
            position_size_pct=position_size_pct,
            num_contracts=num_contracts,
            capital_at_risk=capital_at_risk,
            profit_target=profit_target,
            time_exit_dte=self.TIME_EXIT_DTE,
            warnings=warnings,
            generated_at=datetime.now().isoformat(),
        )
    
    def format_telegram_alert(self, rec: TradeRecommendation) -> str:
        # Format trade recommendation for Telegram.
        #
        # EXTREMELY ACTIONABLE - tells user EXACTLY what to do.
        #
        # Args:
        # rec: TradeRecommendation object
        #
        # Returns:
        # Formatted message string with step-by-step instructions
        # Determine if market is likely open (rough check)
        from datetime import datetime
        now = datetime.now()
        is_weekday = now.weekday() < 5
        hour = now.hour
        market_likely_open = is_weekday and 9 <= hour < 16
        
        market_status = "🟢 MARKET OPEN" if market_likely_open else "🔴 MARKET CLOSED"
        
        # Build the message
        lines = [
            f"🎯 *{rec.symbol} PUT SPREAD SIGNAL*",
            "",
            f"*{market_status}*",
            "",
        ]
        
        # Warnings at top (if any)
        if rec.warnings:
            for w in rec.warnings:
                lines.append(w)
            lines.append("")
        
        # Conditions summary
        lines.extend([
            "*📊 Market Conditions:*",
            f"• Regime Proxy: {rec.btc_change_24h*100:+.1f}% (24h)",
            f"• {rec.symbol}: ${rec.underlying_price:.2f} ({rec.symbol_change_24h*100:+.1f}%)",
            f"• IV Rank: {rec.iv_rank:.0f}% (elevated = good)",
            f"• PoP (Heston): {rec.probability_of_profit:.1f}%",
            f"• Risk of Touch (Stop): {rec.probability_of_touch_stop:.1f}%",
            "",
        ])
        
        # THE EXACT TRADE - step by step
        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━",
            "*📝 EXACT STEPS (Robinhood):*",
            "",
            f"*1.* Open Robinhood app",
            f"*2.* Search: `{rec.symbol}`",
            f"*3.* Tap *Trade* → *Trade Options*",
            f"*4.* Select expiration: *{rec.expiration}*",
            f"*5.* Find *PUT* options:",
            f"     SELL ${rec.short_strike:.0f} Put",
            f"     BUY ${rec.long_strike:.0f} Put",
            f"*6.* Set order type: *Limit*",
            f"*7.* Set limit credit: *${rec.net_credit:.2f}* (or better)",
            f"*8.* Set quantity: *{rec.num_contracts}* contract(s)",
            f"*9.* Review & Submit",
            "━━━━━━━━━━━━━━━━━━━━━",
            "",
        ])
        
        # Economics summary
        lines.extend([
            "*💰 Trade Economics:*",
            f"• Credit received: ~${rec.net_credit * rec.num_contracts * 100:.0f}",
            f"• Max risk: ${rec.capital_at_risk:.0f}",
            f"• Break-even: ${rec.break_even:.2f}",
            f"• Risk/Reward: {rec.risk_reward_ratio:.1f}:1",
            "",
        ])
        
        # Exit plan
        lines.extend([
            "*🚪 Exit Plan:*",
            f"• Take profit at: ${rec.profit_target:.2f} debit (50% profit)",
            f"• Time exit: Close at {rec.time_exit_dte} DTE",
            f"• Max loss: ${rec.capital_at_risk:.0f} (if {rec.symbol} < ${rec.long_strike:.0f})",
            "",
        ])
        
        # Checklist
        lines.extend([
            "*⚠️ CHECKLIST (before clicking Submit):*",
            f"☐ Market is OPEN (9:30 AM - 4:00 PM ET)",
            f"☐ Order shows CREDIT (not debit)",
            f"☐ You have < 3 open positions",
            f"☐ No FOMC/CPI in next 48h",
            f"☐ Strike prices match exactly",
            "",
            f"_Generated: {rec.generated_at[:19]}_",
        ])
        
        return "\n".join(lines)
    
    def _get_gpu_status(self) -> str:
        # Get GPU engine status string.
        try:
            from src.analysis.gpu_engine import get_gpu_engine
            engine = get_gpu_engine()
            device = engine.device_name
            sims = engine.stats.total_simulations
            speed = engine.stats.simulations_per_second
            
            return f"{device} | {sims:,} total sims | {speed:,.0f} sims/sec"
        except Exception:
            return "GPU Engine: Offline"
            
    def format_console_output(self, rec: TradeRecommendation) -> str:
        # Format for console/logging output.
        return f"""
{'='*60}
{rec.symbol} PUT SPREAD RECOMMENDATION
{'='*60}

CONDITIONS
  Proxy:      {rec.btc_change_24h*100:+.1f}%
  {rec.symbol}:       ${rec.underlying_price:.2f} ({rec.symbol_change_24h*100:+.1f}%)
  IV Rank:    {rec.iv_rank:.0f}%
  PoP (Heston): {rec.probability_of_profit:.1f}%
  Touch Risk: {rec.probability_of_touch_stop:.1f}%

GPU STATS
  {self._get_gpu_status()}

TRADE
  SELL:       {rec.expiration} ${rec.short_strike:.0f} Put
  BUY:        {rec.expiration} ${rec.long_strike:.0f} Put
  Width:      ${rec.spread_width:.0f}
  DTE:        {rec.dte} days

ECONOMICS
  Credit:     ${rec.net_credit:.2f}
  Max Risk:   ${rec.max_risk:.2f}
  Break-even: ${rec.break_even:.2f}
  R:R:        {rec.risk_reward_ratio:.1f}:1

GREEKS
  Delta:      {rec.net_delta:.3f}
  Theta:      ${rec.net_theta*100:.2f}/day
  Vega:       ${rec.net_vega*100:.2f}/1% IV

POSITION
  Account:    ${rec.account_size:,.0f}
  Size:       {rec.position_size_pct*100:.0f}%
  Contracts:  {rec.num_contracts}
  At Risk:    ${rec.capital_at_risk:.0f}

EXIT PLAN
  Take profit at: ${rec.profit_target:.2f} (50%)
  Time exit:      {rec.time_exit_dte} DTE

{'='*60}
"""


# Test function
def test_calculator():
    # Test the trade calculator.
    print("=" * 60)
    print("TRADE CALCULATOR TEST")
    print("=" * 60)
    
    calc = TradeCalculator(account_size=3000)
    
    # Force recommendation even if IV is low (for testing)
    rec = calc.generate_recommendation(
        btc_change_24h=-0.05,
        symbol_change_24h=-0.04,
        force=True,
    )
    
    if rec:
        print(calc.format_console_output(rec))
        print("\n--- TELEGRAM FORMAT ---\n")
        print(calc.format_telegram_alert(rec))
    else:
        print("No recommendation generated (check IV Rank threshold)")


if __name__ == "__main__":
    test_calculator()
