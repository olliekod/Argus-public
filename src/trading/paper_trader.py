# Created by Oliver Meihls

# Paper Trader
#
# A single paper trading instance with specific parameters.
# Evaluates signals and logs trades to database.

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import uuid

from .collector_guard import guard_collector_mode

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    # Supported strategy types.
    BULL_PUT = "bull_put"           # Bullish/sideways
    BEAR_CALL = "bear_call"         # Bearish/sideways
    IRON_CONDOR = "iron_condor"     # Sideways/neutral
    STRADDLE_SELL = "straddle_sell" # High IV sideways


@dataclass
class TraderConfig:
    # Configuration for a single paper trader.
    trader_id: str
    strategy_type: StrategyType
    
    # Entry thresholds
    iv_min: float = 55.0           # Minimum IV to enter
    iv_max: float = 100.0          # Maximum IV to enter
    warmth_min: int = 5            # Minimum conditions warmth
    pop_min: float = 65.0          # Minimum probability of profit
    
    # Position params
    dte_target: int = 45           # Target days to expiration
    dte_min: int = 30              # Minimum DTE
    dte_max: int = 60              # Maximum DTE
    
    # Risk params
    gap_tolerance_pct: float = 12.0    # Max gap risk to accept
    max_position_size: int = 4          # Max open positions
    
    # Exit strategies - basic
    profit_target_pct: float = 50.0    # Take profit at 50% of credit
    stop_loss_pct: float = 200.0       # Stop at 200% of credit (2x loss)
    
    # Exit strategies - advanced (None = disabled)
    trailing_stop_pct: Optional[float] = None    # Lock in profits (e.g., 25%)
    dte_exit: int = 7                             # Force exit at this DTE (gamma risk)
    time_exit_days: Optional[int] = None         # Force exit after X days
    iv_exit_drop_pct: Optional[float] = None     # Exit if IV drops by X%
    
    # Market regime filter ('bull', 'bear', 'neutral', 'any')
    regime: str = 'any'
    session_filter: str = 'any'  # 'morning', 'midday', 'afternoon', 'any'
    
    # Budget / Position Sizing (Option 1)
    position_size_pct: float = 5.0     # % of account to risk
    max_risk_dollars: float = 1000.0   # Hard dollar limit
    
    def to_dict(self) -> Dict:
        # Convert to dictionary for JSON storage.
        return {
            'trader_id': self.trader_id,
            'strategy_type': self.strategy_type.value,
            'iv_min': self.iv_min,
            'iv_max': self.iv_max,
            'warmth_min': self.warmth_min,
            'pop_min': self.pop_min,
            'dte_target': self.dte_target,
            'dte_min': self.dte_min,
            'dte_max': self.dte_max,
            'gap_tolerance_pct': self.gap_tolerance_pct,
            'max_position_size': self.max_position_size,
            'profit_target_pct': self.profit_target_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'regime': self.regime,
            'session_filter': self.session_filter,
            'position_size_pct': self.position_size_pct,
            'max_risk_dollars': self.max_risk_dollars,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TraderConfig':
        # Create from dictionary.
        data = data.copy()
        data['strategy_type'] = StrategyType(data['strategy_type'])
        return cls(**data)


@dataclass
class PaperTrade:
    # A single paper trade.
    id: str
    trader_id: str
    strategy_type: str
    symbol: str

    timestamp: str
    strikes: str                   # e.g., "$48/$44"
    expiry: Optional[str]          # e.g., "2026-02-21"

    entry_credit: float            # Credit received
    contracts: int

    status: str = "open"           # open, closed, expired
    close_timestamp: Optional[str] = None
    close_price: Optional[float] = None
    realized_pnl: Optional[float] = None

    market_conditions: Optional[Dict] = None

    # ─── Research Provenance ─────────────────────────────────────────────
    strategy_id: Optional[str] = None   # e.g., "OPTIONS_SPREAD_V1" — links to SignalEvent.strategy_id
    case_id: Optional[str] = None       # Pantheon research case ID for return attribution
    
    def to_dict(self) -> Dict:
        # Convert to dictionary.
        return {
            'id': self.id,
            'trader_id': self.trader_id,
            'strategy_type': self.strategy_type,
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'strikes': self.strikes,
            'expiry': self.expiry,
            'entry_credit': self.entry_credit,
            'contracts': self.contracts,
            'status': self.status,
            'close_timestamp': self.close_timestamp,
            'close_price': self.close_price,
            'realized_pnl': self.realized_pnl,
            'market_conditions': self.market_conditions,
            'strategy_id': self.strategy_id,
            'case_id': self.case_id,
        }


class PaperTrader:
    # A single paper trading instance.
    #
    # Evaluates market conditions against its parameters and
    # logs trades when conditions match.
    
    def __init__(self, config: TraderConfig, db=None):
        # Initialize paper trader.
        #
        # Args:
        # config: Trader configuration
        # db: Database instance for logging trades
        self.config = config
        self.db = db
        self.open_positions: List[PaperTrade] = []
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
        }
        
        logger.debug(f"PaperTrader {config.trader_id} initialized: {config.strategy_type.value}")
    
    def should_enter(
        self,
        symbol: str,
        current_iv: float,
        warmth_score: int,
        dte: int,
        pop: float,
        gap_risk_pct: float,
        market_direction: str,  # 'bullish', 'bearish', 'neutral'
        current_time: Optional[datetime] = None,
    ) -> bool:
        # Evaluate whether to enter a trade.
        #
        # Args:
        # symbol: Ticker symbol
        # current_iv: Current implied volatility
        # warmth_score: Conditions warmth (1-10)
        # dte: Days to expiration
        # pop: Probability of profit
        # gap_risk_pct: Current gap risk percentage
        # market_direction: Market bias
        # current_time: Current time (UTC or Eastern)
        #
        # Returns:
        # True if should enter trade
        # 1. Check session filter (time-of-day) using Eastern Time
        session = self.config.session_filter
        if session != 'any':
            dt = current_time or datetime.now(timezone.utc)
            # P1 fix: Convert to Eastern Time for session checks
            eastern = ZoneInfo("America/New_York")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_et = dt.astimezone(eastern)
            hour = dt_et.hour + (dt_et.minute / 60)

            # Sessions (Eastern Time):
            # morning: 9:30 AM - 11:30 AM (9.5 - 11.5)
            # midday: 11:30 AM - 2:00 PM (11.5 - 14.0)
            # afternoon: 2:00 PM - 4:00 PM (14.0 - 16.0)

            if session == 'morning' and not (9.5 <= hour <= 11.5):
                return False
            elif session == 'midday' and not (11.5 < hour <= 14.0):
                return False
            elif session == 'afternoon' and not (14.0 < hour <= 16.0):
                return False
        
        # 2. Check IV bounds
        if not (self.config.iv_min <= current_iv <= self.config.iv_max):
            return False
        
        # 3. Check warmth
        if warmth_score < self.config.warmth_min:
            return False
        
        # 4. Check DTE
        if not (self.config.dte_min <= dte <= self.config.dte_max):
            return False
        
        # 5. Check PoP
        if pop < self.config.pop_min:
            return False
        
        # 6. Check gap risk
        if gap_risk_pct > self.config.gap_tolerance_pct:
            return False
        
        # 7. Check regime compatibility
        regime = self.config.regime
        if regime != 'any':
            direction_to_regime = {
                'bullish': 'bull',
                'bearish': 'bear',
                'neutral': 'neutral',
            }
            current_regime = direction_to_regime.get(market_direction, 'neutral')
            if regime != current_regime:
                return False
        
        # 8. Check strategy alignment with market direction
        strategy = self.config.strategy_type
        
        if strategy == StrategyType.BULL_PUT:
            if market_direction == 'bearish':
                return False
        
        elif strategy == StrategyType.BEAR_CALL:
            if market_direction == 'bullish':
                return False
        
        elif strategy == StrategyType.STRADDLE_SELL:
            if current_iv < 60:
                return False
        
        # 9. Check if we already have max positions
        if len(self.open_positions) >= self.config.max_position_size:
            return False
        
        return True
    
    def enter_trade(
        self,
        symbol: str,
        strikes: str,
        expiry: Optional[str],
        entry_credit: float,
        contracts: int,
        market_conditions: Dict,
        strategy_id: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> Optional[PaperTrade]:
        # Execute a paper trade entry.
        #
        # Returns None (instead of raising) if the trade fails final validation,
        # so the farm loop can skip it gracefully.
        #
        # Args:
        # symbol: Ticker symbol
        # strikes: Strike prices (e.g., "$48/$44")
        # expiry: Expiration date
        # entry_credit: Credit received per contract
        # contracts: Number of contracts
        # market_conditions: Snapshot of conditions at entry
        # strategy_id: Research strategy identifier for return attribution
        # case_id: Pantheon research case ID for return attribution
        #
        # Returns:
        # Created paper trade, or None if invalid
        guard_collector_mode()

        # ── Final entry-point validation ──────────────────────────────
        if entry_credit is None or entry_credit <= 0:
            logger.debug(
                f"[{self.config.trader_id}] REJECTED entry: "
                f"invalid entry_credit={entry_credit!r} for {symbol}"
            )
            return None

        if contracts is None or contracts <= 0:
            logger.debug(
                f"[{self.config.trader_id}] REJECTED entry: "
                f"invalid contracts={contracts!r} for {symbol}"
            )
            return None

        normalized_expiry = self._normalize_expiry(expiry)
        if normalized_expiry is None:
            logger.debug(
                f"[{self.config.trader_id}] REJECTED entry: "
                f"could not normalize expiry={expiry!r} for {symbol}"
            )
            return None

        if not strikes or strikes in ('N/A', 'NA'):
            logger.debug(
                f"[{self.config.trader_id}] REJECTED entry: "
                f"invalid strikes={strikes!r} for {symbol}"
            )
            return None

        trade = PaperTrade(
            id=str(uuid.uuid4()),
            trader_id=self.config.trader_id,
            strategy_type=self.config.strategy_type.value,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            strikes=strikes,
            expiry=normalized_expiry,
            entry_credit=entry_credit,
            contracts=contracts,
            market_conditions=market_conditions,
            strategy_id=strategy_id,
            case_id=case_id,
        )

        self.open_positions.append(trade)
        self.stats['total_trades'] += 1

        logger.debug(
            f"[{self.config.trader_id}] ENTRY: {symbol} {strikes} "
            f"exp {normalized_expiry} @ ${entry_credit:.2f} x{contracts}"
        )

        return trade

    @staticmethod
    def _normalize_expiry(expiry: Optional[str]) -> Optional[str]:
        # Normalize expiry to YYYY-MM-DD or return None if unparseable/past.
        if expiry is None:
            return None
        normalized = str(expiry).strip()
        if not normalized or normalized.upper() in {"N/A", "NA", "NONE", "NULL"}:
            return None
        try:
            exp_dt = datetime.strptime(normalized[:10], "%Y-%m-%d")
            if exp_dt.date() < date.today():
                logger.warning("Expiry date %s is in the past, rejecting", normalized[:10])
                return None
            return normalized[:10]
        except (ValueError, TypeError):
            return None
    
    def check_exits(
        self, 
        current_prices: Dict[str, float],
        current_iv: Optional[float] = None,
        current_date: Optional[datetime] = None,
    ) -> List[PaperTrade]:
        # Check open positions for exit conditions.
        #
        # Supports multiple exit strategies:
        # - Profit target: Close when profit reaches X%
        # - Stop loss: Close when loss reaches X%
        # - Trailing stop: Lock in profits when position was up X%
        # - DTE exit: Force close at X days to expiration
        # - Time exit: Force close after X days in trade
        # - IV exit: Close if IV drops by X%
        #
        # Args:
        # current_prices: Dict mapping trade_id to current spread price
        # current_iv: Current IV (for IV exit strategy)
        # current_date: Current date (for DTE/time exits, default: now)
        #
        # Returns:
        # List of trades that were closed
        closed = []
        now = current_date or datetime.now(timezone.utc)
        
        for trade in self.open_positions[:]:  # Copy to allow modification
            if trade.status != "open":
                continue
            if trade.id not in current_prices:
                continue

            current_price = current_prices[trade.id]
            entry_credit = trade.entry_credit

            # Guard: if entry_credit is zero/None the position is invalid.
            # Close it immediately rather than dividing by zero.
            if not entry_credit or entry_credit <= 0:
                logger.warning(
                    f"[{self.config.trader_id}] INVALID trade {trade.id}: "
                    f"entry_credit={entry_credit!r}, symbol={trade.symbol}, "
                    f"strikes={trade.strikes}. Closing as invalid_entry_zero_credit."
                )
                trade.status = 'closed'
                trade.close_timestamp = now.isoformat()
                trade.close_price = current_price
                trade.realized_pnl = 0.0
                if trade.market_conditions is None:
                    trade.market_conditions = {}
                trade.market_conditions['close_reason'] = 'invalid_entry_zero_credit'
                self.open_positions.remove(trade)
                closed.append(trade)
                continue

            # Calculate P&L per contract
            pnl_per = (entry_credit - current_price) * 100
            pnl_pct = (pnl_per / (entry_credit * 100)) * 100

            should_close = False
            close_reason = ""
            
            # 1. Profit target
            if current_price <= entry_credit * (1 - self.config.profit_target_pct / 100):
                should_close = True
                close_reason = "PROFIT_TARGET"
            
            # 2. Stop loss
            elif current_price >= entry_credit * (1 + self.config.stop_loss_pct / 100):
                should_close = True
                close_reason = "STOP_LOSS"
            
            # 3. Trailing stop (if enabled and position is in profit)
            elif self.config.trailing_stop_pct is not None and pnl_pct > 0:
                # Track high watermark (stored in market_conditions)
                if trade.market_conditions is None:
                    trade.market_conditions = {}
                
                high_watermark = trade.market_conditions.get('high_pnl_pct', 0)
                if pnl_pct > high_watermark:
                    trade.market_conditions['high_pnl_pct'] = pnl_pct
                    high_watermark = pnl_pct
                
                # Trigger if we've dropped from high watermark by trailing_stop_pct
                if high_watermark >= self.config.trailing_stop_pct:
                    # We were up enough to activate trailing stop
                    if pnl_pct <= high_watermark - self.config.trailing_stop_pct:
                        should_close = True
                        close_reason = f"TRAILING_STOP (peak {high_watermark:.0f}%)"
            
            # 4. DTE exit (force close before gamma risk)
            if not should_close and self.config.dte_exit:
                try:
                    expiry = datetime.fromisoformat(trade.expiry)
                    dte = (expiry - now).days
                    if dte <= self.config.dte_exit:
                        should_close = True
                        close_reason = f"DTE_EXIT ({dte} DTE)"
                except (ValueError, TypeError):
                    pass  # Skip if expiry parsing fails
            
            # 5. Time exit (force close after X days)
            if not should_close and self.config.time_exit_days:
                try:
                    entry_time = datetime.fromisoformat(trade.timestamp)
                    days_in_trade = (now - entry_time).days
                    if days_in_trade >= self.config.time_exit_days:
                        should_close = True
                        close_reason = f"TIME_EXIT ({days_in_trade} days)"
                except (ValueError, TypeError):
                    pass
            
            # 6. IV exit (if IV dropped significantly)
            if not should_close and self.config.iv_exit_drop_pct and current_iv:
                entry_iv = trade.market_conditions.get('iv') if trade.market_conditions else None
                if entry_iv:
                    iv_drop = ((entry_iv - current_iv) / entry_iv) * 100
                    if iv_drop >= self.config.iv_exit_drop_pct:
                        should_close = True
                        close_reason = f"IV_EXIT (dropped {iv_drop:.0f}%)"
            
            if should_close:
                trade.status = 'closed'
                trade.close_timestamp = now.isoformat()
                trade.close_price = current_price
                trade.realized_pnl = pnl_per * trade.contracts
                
                # Store close reason
                if trade.market_conditions is None:
                    trade.market_conditions = {}
                trade.market_conditions['close_reason'] = close_reason
                
                self.open_positions.remove(trade)
                closed.append(trade)
                
                # Update stats
                if trade.realized_pnl > 0:
                    self.stats['wins'] += 1
                else:
                    self.stats['losses'] += 1
                self.stats['total_pnl'] += trade.realized_pnl
                
                logger.debug(f"[{self.config.trader_id}] {close_reason}: P&L ${trade.realized_pnl:.2f}")
        
        return closed
    
    def expire_positions(self, expiry_date: str, final_prices: Dict[str, float]) -> List[PaperTrade]:
        # Handle position expiration.
        #
        # Args:
        # expiry_date: Date to check for expiration
        # final_prices: Final prices for expired trades
        #
        # Returns:
        # List of expired trades
        expired = []
        
        for trade in self.open_positions[:]:
            if trade.expiry == expiry_date:
                trade.status = 'expired'
                trade.close_timestamp = datetime.now(timezone.utc).isoformat()
                
                # Get final price (0 if expired worthless = full profit)
                final_price = final_prices.get(trade.id, 0)
                trade.close_price = final_price
                trade.realized_pnl = (trade.entry_credit - final_price) * 100 * trade.contracts
                
                self.open_positions.remove(trade)
                expired.append(trade)
                
                # Update stats
                if trade.realized_pnl > 0:
                    self.stats['wins'] += 1
                else:
                    self.stats['losses'] += 1
                self.stats['total_pnl'] += trade.realized_pnl
                
                logger.debug(
                    f"[{self.config.trader_id}] EXPIRED: {trade.symbol} "
                    f"P&L: ${trade.realized_pnl:.2f}"
                )
        
        return expired
    
    def get_positions_summary(self) -> List[Dict]:
        # Get summary of open positions.
        return [t.to_dict() for t in self.open_positions]
    
    def get_pnl_summary(self, current_prices: Optional[Dict[str, float]] = None) -> Dict:
        # Get P&L summary.
        #
        # Args:
        # current_prices: Optional dict mapping trade_id to current spread price
        # for mark-to-market open P&L calculation.
        if current_prices:
            open_pnl = sum(
                (t.entry_credit - current_prices.get(t.id, t.entry_credit)) * 100 * t.contracts
                for t in self.open_positions
            )
        else:
            # Without current prices, open_pnl is unknown — report 0 (not credit received)
            open_pnl = 0.0
        win_rate = (self.stats['wins'] / self.stats['total_trades'] * 100
                    if self.stats['total_trades'] > 0 else 0)

        return {
            'trader_id': self.config.trader_id,
            'strategy': self.config.strategy_type.value,
            'total_trades': self.stats['total_trades'],
            'wins': self.stats['wins'],
            'losses': self.stats['losses'],
            'win_rate': win_rate,
            'realized_pnl': self.stats['total_pnl'],
            'open_positions': len(self.open_positions),
            'open_pnl': open_pnl,
        }


# Test function
async def test_paper_trader():
    # Test paper trader functionality.
    print("Paper Trader Test")
    print("=" * 40)
    
    config = TraderConfig(
        trader_id="PT-TEST",
        strategy_type=StrategyType.BULL_PUT,
        iv_min=50,
        warmth_min=5,
        pop_min=60,
    )
    
    trader = PaperTrader(config)
    
    # Test entry evaluation
    should_enter = trader.should_enter(
        symbol="IBIT",
        current_iv=58,
        warmth_score=6,
        dte=45,
        pop=68,
        gap_risk_pct=5,
        market_direction='neutral',
    )
    print(f"Should enter (good conditions): {should_enter}")
    assert should_enter == True
    
    # Test with bad conditions
    should_enter = trader.should_enter(
        symbol="IBIT",
        current_iv=40,  # Too low
        warmth_score=6,
        dte=45,
        pop=68,
        gap_risk_pct=5,
        market_direction='neutral',
    )
    print(f"Should enter (low IV): {should_enter}")
    assert should_enter == False
    
    # Test trade entry
    trade = trader.enter_trade(
        symbol="IBIT",
        strikes="$48/$44",
        expiry="2026-02-21",
        entry_credit=0.42,
        contracts=2,
        market_conditions={'iv': 58, 'warmth': 6},
    )
    print(f"Trade entered: {trade.id}")
    
    # Test P&L summary
    summary = trader.get_pnl_summary()
    print(f"P&L Summary: {summary}")
    
    print("\n✅ All paper trader tests passed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_paper_trader())
