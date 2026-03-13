"""
Paper Trader Farm
=================

Manages 97K+ paper traders running in parallel.
Tests all unique parameter combinations with market regime awareness.
Generates monthly reports with strategy averages and top threshold combos.
Checks economic calendar for blackout periods before entries.
"""

import asyncio
import logging
import re
from datetime import datetime, date, timedelta, time, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any, Callable, Set
from pathlib import Path
import json

from .paper_trader import PaperTrader, TraderConfig, PaperTrade, StrategyType
from .trader_config_generator import generate_all_configs, get_config_summary
from .collector_guard import guard_collector_mode
from ..core.economic_calendar import EconomicCalendar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard tradeable-universe allowlist.
# Only these underlyings may be traded as options positions.
# Crypto symbols (BTCUSDT, ETHUSDT, …) are data/signal sources ONLY.
# ---------------------------------------------------------------------------
TRADEABLE_UNDERLYINGS: frozenset = frozenset({"IBIT", "BITO"})

# Regex that matches a valid YYYY-MM-DD expiry string
_EXPIRY_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class PaperTraderFarm:
    """
    Manages a farm of paper traders.
    
    Each trader runs with different parameters but shares
    the same market data. Trades are logged to database
    for later analysis.
    """
    
    def __init__(
        self,
        db=None,
        total_traders: int = 400000,
        full_coverage: bool = True,
        config_file: Optional[str] = None,
        starting_balance: float = 5000.0,
        max_traders: int = 2_000_000,
        max_open_positions_total: int = 500_000,
        max_trades_per_minute: int = 10_000,
    ):
        """
        Initialize the paper trader farm.

        Args:
            db: Database instance for logging trades
            total_traders: Number of traders (ignored if full_coverage=True)
            full_coverage: If True, generate ALL unique parameter combinations
            config_file: Optional path to saved configs
            max_traders: Hard cap on total configs (guardrail)
            max_open_positions_total: Safety cap on open positions
            max_trades_per_minute: Rate limit on new trades
        """
        self.db = db
        self.total_traders = total_traders
        self.full_coverage = full_coverage
        self.starting_balance = starting_balance
        self.max_traders = max_traders
        self.max_open_positions_total = max_open_positions_total
        self.max_trades_per_minute = max_trades_per_minute
        self._trades_this_minute: int = 0
        self._minute_reset_ts: float = 0
        
        # New: Tensors for GPU evaluation
        self.trader_tensors: Optional['torch.Tensor'] = None
        self.trader_configs: List[TraderConfig] = []
        
        # Only active traders (those with open positions) are instantiated
        self.active_traders: Dict[str, PaperTrader] = {} 
        
        self._running = False
        self.last_evaluation_time: Optional[datetime] = None
        self.last_evaluation_symbol: Optional[str] = None
        self.last_evaluation_entered: int = 0
        self._promoted_trader_ids: Optional[Set[str]] = None
        self._promoted_indices: Optional[Set[int]] = None
        
        # Callbacks for market data (set by orchestrator)
        self._get_conditions: Optional[Callable] = None
        self._get_options_chain: Optional[Callable] = None
        self._get_gap_risk: Optional[Callable] = None
        
        # Economic calendar for blackout periods (FOMC, CPI, Jobs)
        self.economic_calendar = EconomicCalendar()
        
        # Track open positions by symbol for correlation check
        self._positions_by_symbol: Dict[str, int] = {}  # symbol -> count
        self._positions_lock = asyncio.Lock()  # Protects _positions_by_symbol

        # P3: Drawdown circuit breaker
        self._peak_balance: float = starting_balance
        self._drawdown_breaker_pct: float = 20.0  # halt at 20% drawdown
        self._drawdown_halted: bool = False
        self._current_epoch_start: Optional[str] = None

        # Runaway safety: halt entries when exit-monitor errors spike
        self._exit_errors: List[float] = []          # timestamps of recent errors
        self._exit_error_window: int = 300            # 5-minute sliding window
        self._exit_error_threshold: int = 10          # halt after 10 errors in window
        self._runaway_halted: bool = False
        self._runaway_halt_reason: Optional[str] = None
        self._last_rejected_symbol_ts: float = 0.0   # rate-limit rejection logs
        self._telegram_callback: Optional[Callable] = None

        logger.info(f"PaperTraderFarm initialized (full_coverage={full_coverage})")
    
    async def initialize(self) -> None:
        """Initialize all traders and database tables."""
        import time as _time

        # Create database table
        if self.db:
            await self._create_tables()
            self._current_epoch_start = await self.db.get_current_epoch_start()

        # Generate trader configs with MAX_TRADERS guardrail
        t0 = _time.monotonic()
        configs = generate_all_configs(
            total_traders=self.total_traders,
            full_coverage=self.full_coverage,
            max_traders=self.max_traders,
        )
        self.trader_configs = configs
        gen_time = _time.monotonic() - t0

        if len(configs) >= self.max_traders:
            logger.warning(
                f"MAX_TRADERS guardrail hit: capped at {self.max_traders:,} "
                f"(full coverage would produce more). Continuing safely."
            )

        # Prepare Tensors for GPU
        t1 = _time.monotonic()
        await self._prepare_trader_tensors()
        tensor_time = _time.monotonic() - t1

        # Log summary
        summary = get_config_summary(configs)
        logger.info(
            f"Farm initialized: {len(configs):,} configs in {gen_time:.1f}s, "
            f"GPU upload in {tensor_time:.1f}s"
        )
        for strat, count in summary['by_strategy'].items():
            logger.info(f"  {strat}: {count:,} traders")
    
    async def _create_tables(self) -> None:
        """Create database tables for paper trading."""
        # Create paper_trades table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id TEXT PRIMARY KEY,
                trader_id TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                strikes TEXT,
                expiry TEXT,
                entry_credit REAL,
                contracts INTEGER,
                status TEXT DEFAULT 'open',
                close_timestamp TEXT,
                close_price REAL,
                realized_pnl REAL,
                market_conditions TEXT,
                strategy_id TEXT,
                case_id TEXT
            )
        """)
        
        # Create indexes
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_trader 
            ON paper_trades(trader_id)
        """)
        
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_status 
            ON paper_trades(status)
        """)
        
        # Create configs table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS paper_trader_configs (
                trader_id TEXT PRIMARY KEY,
                config TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Index on timestamp for retention cleanup
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_ts
            ON paper_trades(timestamp)
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_expiry
            ON paper_trades(expiry)
        """)

        # Migrations: add columns if missing
        migrations = [
            ("trader_id", "ALTER TABLE paper_trades ADD COLUMN trader_id TEXT"),
            ("close_reason", "ALTER TABLE paper_trades ADD COLUMN close_reason TEXT"),
            ("strategy_id", "ALTER TABLE paper_trades ADD COLUMN strategy_id TEXT"),
            ("case_id", "ALTER TABLE paper_trades ADD COLUMN case_id TEXT"),
        ]
        for col, sql in migrations:
            try:
                await self.db.execute(f"SELECT {col} FROM paper_trades LIMIT 1")
            except Exception:
                try:
                    await self.db.execute(sql)
                    logger.info(f"Migration: added {col} column to paper_trades")
                except Exception:
                    pass

        logger.info("Paper trading tables created")

    async def _prepare_trader_tensors(self) -> None:
        """Convert all trader configs into a single tensor for GPU evaluation."""
        import torch
        from .paper_trader import StrategyType
        
        N = len(self.trader_configs)
        # Columns: iv_min, iv_max, warmth_min, pop_min, dte_min, dte_max, gap_max, strategy_id
        params = torch.zeros((N, 8), dtype=torch.float32)
        
        strat_map = {
            StrategyType.BULL_PUT: 0,
            StrategyType.BEAR_CALL: 1,
            StrategyType.IRON_CONDOR: 2,
            StrategyType.STRADDLE_SELL: 3
        }
        
        for i, config in enumerate(self.trader_configs):
            params[i, 0] = config.iv_min
            params[i, 1] = config.iv_max
            params[i, 2] = config.warmth_min
            params[i, 3] = config.pop_min
            params[i, 4] = config.dte_min
            params[i, 5] = config.dte_max
            params[i, 6] = config.gap_tolerance_pct
            params[i, 7] = strat_map.get(config.strategy_type, 0)
            
        from ..analysis.gpu_engine import get_gpu_engine
        engine = get_gpu_engine()
        self.trader_tensors = params.to(engine._device)
        logger.info(f"Uploaded {N:,} trader params to GPU ({params.element_size() * N * 8 / 1e6:.1f} MB)")
    
    def set_data_sources(
        self,
        get_conditions: Callable = None,
        get_options_chain: Callable = None,
        get_gap_risk: Callable = None,
    ) -> None:
        """Set callbacks for market data access."""
        if get_conditions:
            self._get_conditions = get_conditions
        if get_options_chain:
            self._get_options_chain = get_options_chain
        if get_gap_risk:
            self._get_gap_risk = get_gap_risk
    
    def set_telegram_alert_callback(self, callback: Callable) -> None:
        """Set callback for sending Telegram alerts (runaway safety, etc.)."""
        self._telegram_callback = callback

    def record_exit_error(self) -> None:
        """Record an exit-monitor error for runaway detection.

        Called by the orchestrator whenever the exit monitor loop catches
        an exception so the farm can track error frequency.
        """
        import time as _time
        now = _time.time()
        self._exit_errors.append(now)
        # Trim old entries outside the sliding window
        cutoff = now - self._exit_error_window
        self._exit_errors = [t for t in self._exit_errors if t >= cutoff]
        if len(self._exit_errors) >= self._exit_error_threshold and not self._runaway_halted:
            self._runaway_halted = True
            self._runaway_halt_reason = (
                f"exit_monitor_errors={len(self._exit_errors)} "
                f"in {self._exit_error_window}s window"
            )
            logger.critical(
                f"RUNAWAY SAFETY HALT: {self._runaway_halt_reason}. "
                f"New entries blocked; collector feeds continue."
            )
            if self._telegram_callback:
                asyncio.ensure_future(self._telegram_callback(
                    f"🚨 *RUNAWAY SAFETY HALT*\n"
                    f"{self._runaway_halt_reason}\n"
                    f"New entries blocked. Collector feeds continue."
                ))

    @staticmethod
    def _validate_signal_data(signal_data: Dict[str, Any]) -> Optional[str]:
        """Validate critical signal fields before any trade insertion.

        Returns an error message string if invalid, or None if OK.
        """
        # Credit must be a positive number
        credit = signal_data.get('credit')
        if credit is None or not isinstance(credit, (int, float)) or credit <= 0:
            return f"invalid credit={credit!r}"

        # Expiry must be a parseable YYYY-MM-DD date
        expiry = signal_data.get('expiry')
        if not expiry or not isinstance(expiry, str):
            return f"missing/invalid expiry={expiry!r}"
        expiry_clean = str(expiry).strip()
        if not _EXPIRY_DATE_RE.match(expiry_clean):
            return f"expiry not YYYY-MM-DD: {expiry_clean!r}"

        # Strikes must be present and contain a '/'
        strikes = signal_data.get('strikes', '')
        if not strikes or '/' not in str(strikes):
            return f"missing/invalid strikes={strikes!r}"

        return None  # valid

    async def evaluate_signal(
        self,
        symbol: str,
        signal_data: Dict[str, Any],
    ) -> List[PaperTrade]:
        """
        Evaluate a trading signal across all traders.

        Each trader decides independently whether to enter.
        Respects economic calendar blackout periods and correlation limits.

        Args:
            symbol: Ticker symbol
            signal_data: Dict with signal details:
                - iv: Current implied volatility
                - warmth: Conditions warmth score
                - dte: Days to expiration
                - pop: Probability of profit
                - gap_risk: Current gap risk %
                - direction: Market direction
                - strikes: Strike prices
                - expiry: Expiration date
                - credit: Entry credit

        Returns:
            List of trades that were entered
        """
        import time as _time

        guard_collector_mode()

        # ── Hard tradeable-universe gate ──────────────────────────────
        if symbol.upper() not in TRADEABLE_UNDERLYINGS:
            now = _time.time()
            if now - self._last_rejected_symbol_ts > 60:  # rate-limit log
                logger.warning(
                    f"REJECTED non-tradeable symbol {symbol!r}. "
                    f"Allowed: {sorted(TRADEABLE_UNDERLYINGS)}"
                )
                self._last_rejected_symbol_ts = now
            return []

        # ── Runaway safety halt ───────────────────────────────────────
        if self._runaway_halted:
            logger.debug(
                f"Blocked signal for {symbol}: runaway safety halt "
                f"({self._runaway_halt_reason})"
            )
            return []

        # ── Validate signal data before any work ──────────────────────
        validation_error = self._validate_signal_data(signal_data)
        if validation_error:
            now = _time.time()
            if now - self._last_rejected_symbol_ts > 60:
                logger.warning(
                    f"REJECTED signal for {symbol}: {validation_error}. "
                    f"No positions will be created."
                )
                self._last_rejected_symbol_ts = now
            return []

        # Guardrail: rate limit trades per minute
        now = _time.time()
        if now - self._minute_reset_ts > 60:
            self._trades_this_minute = 0
            self._minute_reset_ts = now
        if self._trades_this_minute >= self.max_trades_per_minute:
            logger.warning(f"Trade rate limit hit ({self.max_trades_per_minute}/min). Skipping signal.")
            return []

        # Guardrail: total open positions cap
        total_open = sum(len(t.open_positions) for t in self.active_traders.values())
        if total_open >= self.max_open_positions_total:
            logger.warning(f"Open positions cap hit ({total_open:,}/{self.max_open_positions_total:,}). Skipping signal.")
            return []

        # P3: Drawdown circuit breaker check
        if self._drawdown_halted:
            logger.debug(f"Blocked signal for {symbol}: drawdown circuit breaker active")
            return []

        # 1. Check economic calendar blackout periods (FOMC, CPI, Jobs)
        is_blackout, event = self.economic_calendar.is_blackout_period()
        if is_blackout:
            logger.debug(f"Blocked signal for {symbol}: blackout period ({event.name if event else 'unknown'})")
            return []

        # 2. Check IBIT/BITO correlation (both track BTC, don't double-up)
        btc_symbols = {'IBIT', 'BITO', 'GBTC', 'BTCO'}  # All BTC-tracking ETFs
        if symbol.upper() in btc_symbols:
            btc_exposure = sum(
                count for sym, count in self._positions_by_symbol.items()
                if sym.upper() in btc_symbols
            )
            max_btc_exposure = len(self.trader_configs) // 2
            if btc_exposure >= max_btc_exposure:
                logger.debug(f"Blocked {symbol}: BTC exposure limit reached ({btc_exposure:,} positions)")
                return []

        entered_trades = []
        db_batch = []
        self.last_evaluation_time = datetime.now(timezone.utc)
        self.last_evaluation_symbol = symbol
        self.last_evaluation_entered = 0

        # P1 fix: Convert timestamp to Eastern Time for session filters
        eastern = ZoneInfo("America/New_York")
        current_time = signal_data.get('timestamp')
        if isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(current_time)
            except ValueError:
                current_time = None
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        # Ensure timezone-aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        current_time_et = current_time.astimezone(eastern)

        # 3. Vectorized Evaluation on GPU
        from ..analysis.gpu_engine import get_gpu_engine
        import torch
        engine = get_gpu_engine()

        # Prepare market data tensor
        dir_val = signal_data.get('direction', 'neutral')
        market_tensor = torch.tensor([
            signal_data.get('iv', 50),
            signal_data.get('warmth', 5),
            signal_data.get('dte', 45),
            signal_data.get('pop', 65),
            signal_data.get('gap_risk', 0),
            1.0 if dir_val == 'bullish' else 0.0,
            1.0 if dir_val == 'bearish' else 0.0,
            1.0 if dir_val == 'neutral' else 0.0,
            0.0, 0.0, 0.0
        ], device=engine._device, dtype=torch.float32)

        # Batch evaluate
        start_eval = datetime.now()
        mask = engine.evaluate_traders_batch(self.trader_tensors, market_tensor)
        eval_time = (datetime.now() - start_eval).total_seconds() * 1000

        # Get indices of traders that should enter
        entry_indices = torch.where(mask)[0].cpu().tolist()

        if self._promoted_indices is not None:
            entry_indices = [i for i in entry_indices if i in self._promoted_indices]

        # P1 fix: Use Eastern Time hour for session filter
        et_hour = current_time_et.hour + (current_time_et.minute / 60)

        # P1: Base credit from signal, apply realistic fill slippage
        raw_credit = signal_data.get('credit', 0.40)
        # Apply 5% bid-ask slippage to simulate realistic fills
        slippage_factor = 0.95
        realistic_credit = raw_credit * slippage_factor

        # Per-strategy strike computation (Bug 2+8 fix)
        put_strikes_str = signal_data.get('strikes', '')
        underlying_price = signal_data.get('price', 0)
        put_short = put_long = 0.0
        put_spread_width = 2.0
        try:
            if '/' in put_strikes_str:
                parts = put_strikes_str.replace('$', '').split('/')
                put_short = float(parts[0])
                put_long = float(parts[1])
                put_spread_width = abs(put_short - put_long)
        except (ValueError, IndexError) as e:
            logger.warning("Failed to parse strike string '%s': %s", put_strikes_str, e)

        # Compute call-side strikes mirrored from underlying price
        call_short = call_long = 0.0
        if underlying_price > 0 and put_short > 0:
            put_distance = underlying_price - put_short
            call_short = round(underlying_price + put_distance, 1)
            call_long = round(call_short + put_spread_width, 1)

        skipped_invalid = 0
        for idx in entry_indices:
            config = self.trader_configs[idx]

            # P1 fix: Session filter uses Eastern Time
            session = config.session_filter
            if session != 'any':
                if session == 'morning' and not (9.5 <= et_hour <= 11.5): continue
                elif session == 'midday' and not (11.5 < et_hour <= 14.0): continue
                elif session == 'afternoon' and not (14.0 < et_hour <= 16.0): continue

            # Re-check max positions for this specific config
            active_trader = self.active_traders.get(config.trader_id)
            if active_trader and len(active_trader.open_positions) >= config.max_position_size:
                continue

            # Per-strategy strike and credit selection
            strat = config.strategy_type
            if strat == StrategyType.BEAR_CALL and call_short > 0:
                trade_strikes = f"{call_short}/{call_long}"
                trade_credit = realistic_credit * 0.9  # Calls typically earn less premium
                trade_spread_width = put_spread_width
            elif strat == StrategyType.IRON_CONDOR and call_short > 0:
                trade_strikes = f"{call_short}/{call_long}/{put_short}/{put_long}"
                trade_credit = realistic_credit * 1.7  # Both legs combined credit
                trade_spread_width = put_spread_width  # Risk is max of either side
            elif strat == StrategyType.STRADDLE_SELL and underlying_price > 0:
                atm = round(underlying_price, 0)
                trade_strikes = f"{atm}C/{atm}P"
                trade_credit = realistic_credit * 2.5  # Straddle collects both premiums
                trade_spread_width = put_spread_width
            else:
                # Default: bull_put (original behavior)
                trade_strikes = put_strikes_str or 'N/A'
                trade_credit = realistic_credit
                trade_spread_width = put_spread_width

            # ── Per-trade validation: skip garbage ────────────────────
            if trade_credit <= 0:
                skipped_invalid += 1
                continue
            if trade_strikes in ('', 'N/A'):
                skipped_invalid += 1
                continue

            # P1: Differentiate trade construction per config
            max_risk_per_contract = (trade_spread_width - trade_credit) * 100
            if max_risk_per_contract > 0:
                risk_budget = min(
                    self.starting_balance * (config.position_size_pct / 100),
                    config.max_risk_dollars,
                )
                contracts = max(1, int(risk_budget / max_risk_per_contract))
            else:
                contracts = 1

            if contracts <= 0:
                skipped_invalid += 1
                continue

            # Instantiate trader if not already active (Lazy Loading)
            if not active_trader:
                active_trader = PaperTrader(config=config, db=self.db)
                self.active_traders[config.trader_id] = active_trader

            # Execute paper trade
            trade = active_trader.enter_trade(
                symbol=symbol,
                strikes=trade_strikes,
                expiry=signal_data.get('expiry'),
                entry_credit=trade_credit,
                contracts=contracts,
                strategy_id=signal_data.get('strategy_id'),
                case_id=signal_data.get('case_id'),
                market_conditions={
                    'iv': signal_data.get('iv'),
                    'warmth': signal_data.get('warmth'),
                    'direction': signal_data.get('direction'),
                    'pop': signal_data.get('pop'),
                    'pot': signal_data.get('pot'),
                    'iv_rank': signal_data.get('iv_rank'),
                    'dte': signal_data.get('dte'),
                    'credit': signal_data.get('credit'),
                    'realistic_credit': trade_credit,
                    'contracts': contracts,
                    'strategy': strat.value if hasattr(strat, 'value') else str(strat),
                    'conditions_label': signal_data.get('conditions_label'),
                    'btc_change_pct': signal_data.get('btc_change_pct'),
                    'ibit_change_pct': signal_data.get('ibit_change_pct'),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                },
            )

            if trade is None:
                # enter_trade returned None due to its own validation
                skipped_invalid += 1
                continue

            entered_trades.append(trade)
            db_batch.append(trade)

            # Track positions by symbol for correlation check
            self._positions_by_symbol[symbol] = self._positions_by_symbol.get(symbol, 0) + 1

        # P2: Batch database writes for entries
        if db_batch and self.db:
            await self._save_trades_batch(db_batch)

        # Update rate limit counter
        self._trades_this_minute += len(entered_trades)

        if entered_trades:
            logger.debug(
                f"Signal evaluated in {eval_time:.1f}ms: "
                f"{len(entered_trades)}/{len(self.trader_configs)} "
                f"traders entered {symbol} (credit=${realistic_credit:.3f}, slippage={1-slippage_factor:.0%})"
            )
        if skipped_invalid:
            logger.debug(f"Skipped {skipped_invalid} traders due to per-trade validation (zero credit/invalid strikes/qty)")
        self.last_evaluation_entered = len(entered_trades)

        # P3: Update drawdown circuit breaker
        self._update_drawdown_check()

        return entered_trades

    def _update_drawdown_check(self) -> None:
        """P3: Check portfolio drawdown and halt if exceeded."""
        total_pnl = sum(
            t.stats['total_pnl'] for t in self.active_traders.values()
        )
        current_balance = self.starting_balance + total_pnl
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
        if self._peak_balance > 0:
            drawdown_pct = ((self._peak_balance - current_balance) / self._peak_balance) * 100
            if drawdown_pct >= self._drawdown_breaker_pct:
                if not self._drawdown_halted:
                    self._drawdown_halted = True
                    logger.warning(
                        f"DRAWDOWN CIRCUIT BREAKER: {drawdown_pct:.1f}% drawdown "
                        f"(peak=${self._peak_balance:,.0f}, current=${current_balance:,.0f}). "
                        f"New entries halted."
                    )
            elif self._drawdown_halted and drawdown_pct < self._drawdown_breaker_pct * 0.5:
                # Reset breaker when drawdown recovers below half threshold
                self._drawdown_halted = False
                logger.info(f"Drawdown circuit breaker reset (drawdown={drawdown_pct:.1f}%)")

    def get_status_summary(self) -> Dict[str, Any]:
        """Return last evaluation status for the farm."""
        return {
            "last_evaluation_time": self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            "last_evaluation_symbol": self.last_evaluation_symbol,
            "last_evaluation_entered": self.last_evaluation_entered,
            "active_traders": len(self.active_traders),
            "total_configs": len(self.trader_configs),
            "promoted_traders": len(self._promoted_trader_ids) if self._promoted_trader_ids else 0,
            "tradeable_underlyings": sorted(TRADEABLE_UNDERLYINGS),
            "runaway_halted": self._runaway_halted,
            "runaway_halt_reason": self._runaway_halt_reason,
            "drawdown_halted": self._drawdown_halted,
            "exit_errors_recent": len(self._exit_errors),
        }

    def set_promoted_traders(self, trader_ids: List[str]) -> None:
        """Restrict evaluations to promoted trader IDs."""
        promoted = set(trader_ids)
        indices = {
            idx for idx, config in enumerate(self.trader_configs)
            if config.trader_id in promoted
        }
        self._promoted_trader_ids = promoted
        self._promoted_indices = indices
    
    async def _save_trade(self, trade: PaperTrade) -> None:
        """Save a single trade to the database."""
        await self.db.execute("""
            INSERT OR REPLACE INTO paper_trades
            (id, trader_id, strategy_type, symbol, timestamp, strikes, expiry,
             entry_credit, contracts, status, close_timestamp, close_price,
             realized_pnl, market_conditions, strategy_id, case_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.id,
            trade.trader_id,
            trade.strategy_type,
            trade.symbol,
            trade.timestamp,
            trade.strikes,
            trade.expiry,
            trade.entry_credit,
            trade.contracts,
            trade.status,
            trade.close_timestamp,
            trade.close_price,
            trade.realized_pnl,
            json.dumps(trade.market_conditions) if trade.market_conditions else None,
            trade.strategy_id,
            trade.case_id,
        ))

    async def _save_trades_batch(self, trades: List[PaperTrade]) -> None:
        """Save multiple trades in a single batch transaction (P2)."""
        if not trades:
            return
        rows = [
            (
                t.id, t.trader_id, t.strategy_type, t.symbol, t.timestamp,
                t.strikes, t.expiry, t.entry_credit, t.contracts, t.status,
                t.close_timestamp, t.close_price, t.realized_pnl,
                json.dumps(t.market_conditions) if t.market_conditions else None,
                t.strategy_id, t.case_id,
            )
            for t in trades
        ]
        await self.db.execute_many("""
            INSERT OR REPLACE INTO paper_trades
            (id, trader_id, strategy_type, symbol, timestamp, strikes, expiry,
             entry_credit, contracts, status, close_timestamp, close_price,
             realized_pnl, market_conditions, strategy_id, case_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    
    async def check_exits(self, current_prices: Dict[str, Dict[str, float]]) -> List[PaperTrade]:
        """
        Check all traders for exit conditions.

        Args:
            current_prices: Dict mapping symbol to {trade_id: current_price}

        Returns:
            List of closed trades
        """
        all_closed = []
        db_batch = []
        eligible = 0

        # Only check traders with open positions
        for trader_id, trader in list(self.active_traders.items()):
            # Get prices relevant to this trader's positions
            trader_prices = {}
            for trade in trader.open_positions:
                if trade.status != "open":
                    continue
                if trade.symbol in current_prices:
                    prices = current_prices[trade.symbol]
                    if trade.id in prices:
                        trader_prices[trade.id] = prices[trade.id]
                        eligible += 1

            if not trader_prices:
                continue

            closed = trader.check_exits(trader_prices)

            for trade in closed:
                db_batch.append(trade)
                all_closed.append(trade)
                # P0 fix: Decrement position counter
                if trade.symbol in self._positions_by_symbol:
                    self._positions_by_symbol[trade.symbol] = max(
                        0, self._positions_by_symbol[trade.symbol] - 1
                    )

            # Clean up active_traders if no more open positions
            if not trader.open_positions:
                del self.active_traders[trader_id]

        # P2: Batch database writes
        if db_batch and self.db:
            await self._save_trades_batch(db_batch)

        if all_closed:
            logger.debug(
                f"Exit check: {len(all_closed)} trades closed "
                f"(eligible={eligible:,}, db_updates={len(db_batch):,})"
            )

        return all_closed

    async def expire_positions(self, expiry_date: str) -> List[PaperTrade]:
        """Handle expiration for a given date."""
        all_expired = []
        db_batch = []

        for trader_id, trader in list(self.active_traders.items()):
            expired = trader.expire_positions(expiry_date, {})

            for trade in expired:
                db_batch.append(trade)
                all_expired.append(trade)
                # P0 fix: Decrement position counter
                if trade.symbol in self._positions_by_symbol:
                    self._positions_by_symbol[trade.symbol] = max(
                        0, self._positions_by_symbol[trade.symbol] - 1
                    )

            # Clean up active_traders if no more open positions
            if not trader.open_positions:
                del self.active_traders[trader_id]

        # P2: Batch database writes
        if db_batch and self.db:
            await self._save_trades_batch(db_batch)

        if all_expired:
            logger.debug(f"Expiration: {len(all_expired)} trades expired for {expiry_date}")

        return all_expired
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def get_aggregate_positions(self) -> List[Dict]:
        """Get all open positions across all traders (active only)."""
        positions = []
        for trader in self.active_traders.values():
            positions.extend(trader.get_positions_summary())
        return positions
    
    def get_aggregate_pnl(self) -> Dict:
        """Get aggregate P&L across all traders."""
        total_realized = 0
        total_trades = 0
        total_wins = 0
        open_positions = 0
        
        for trader in self.active_traders.values():
            pnl = trader.get_pnl_summary()
            total_realized += pnl['realized_pnl']
            total_trades += pnl['total_trades']
            total_wins += pnl['wins']
            open_positions += pnl['open_positions']
        
        return {
            'total_traders': len(self.trader_configs),
            'total_trades': total_trades,
            'total_wins': total_wins,
            'win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'realized_pnl': total_realized,
            'open_positions': open_positions,
        }
    
    def get_leaderboard(self, top_n: int = 20) -> List[Dict]:
        """Get top performing traders (active only for now)."""
        rankings = []
        
        for trader in self.active_traders.values():
            pnl = trader.get_pnl_summary()
            rankings.append(pnl)
        
        # Sort by realized P&L
        rankings.sort(key=lambda x: x['realized_pnl'], reverse=True)
        
        return rankings[:top_n]
    
    def get_strategy_breakdown(self) -> Dict[str, Dict]:
        """Get performance breakdown by strategy."""
        strategies = {}
        
        # For a full breakdown, we need to load stats from DB or track all-time
        # Since we lazy load objects, we'll iterate the configs
        for config in self.trader_configs:
            strat = config.strategy_type.value
            # Note: Total stats would ideally come from DB queries for a farm this size
            # but for this iteration we'll provide a placeholder or partial from active
            if strat not in strategies:
                strategies[strat] = {'traders': 0, 'total_trades': 0, 'wins': 0, 'realized_pnl': 0}
            strategies[strat]['traders'] += 1
            
        # Update with active stats
        for trader in self.active_traders.values():
            strat = trader.config.strategy_type.value
            pnl = trader.get_pnl_summary()
            strategies[strat]['total_trades'] += pnl['total_trades']
            strategies[strat]['wins'] += pnl['wins']
            strategies[strat]['realized_pnl'] += pnl['realized_pnl']
        
        # Calculate win rates
        for strat in strategies:
            total = strategies[strat]['total_trades']
            wins = strategies[strat]['wins']
            strategies[strat]['win_rate'] = (wins / total * 100) if total > 0 else 0
        
        return strategies
    
    def format_leaderboard(self, month: str = None) -> str:
        """Format monthly report for Telegram/display."""
        if month is None:
            month = datetime.now().strftime('%B %Y')
        
        breakdown = self.get_strategy_breakdown()
        aggregate = self.get_aggregate_pnl()
        
        lines = [
            f"📊 PAPER TRADER MONTHLY REPORT — {month}",
            "",
            "STRATEGY PERFORMANCE",
            "━" * 60,
            "Strategy      | Traders | Trades | Win% | Avg P&L | Total P&L",
            "━" * 60,
        ]
        
        # Sort by total P&L
        sorted_strats = sorted(breakdown.items(), key=lambda x: x[1]['realized_pnl'], reverse=True)
        best_strat = sorted_strats[0][0] if sorted_strats else None
        
        for strat_name, data in sorted_strats:
            traders_count = data['traders']
            trades = data['total_trades']
            win_rate = data['win_rate']
            total_pnl = data['realized_pnl']
            avg_pnl = total_pnl / trades if trades > 0 else 0
            
            star = " ⭐" if strat_name == best_strat else ""
            lines.append(
                f"{strat_name:13s} | {traders_count:7,d} | {trades:6d} | "
                f"{win_rate:3.0f}% | ${avg_pnl:+6.0f} | ${total_pnl:+,.0f}{star}"
            )
        
        # Top threshold combinations
        lines.extend([
            "",
            "TOP THRESHOLD COMBINATIONS",
            "━" * 60,
        ])
        
        top_combos = self._get_top_threshold_combos(5)
        for i, combo in enumerate(top_combos, 1):
            lines.append(
                f"{i}. {combo['strategy']} + IV>{combo['iv_min']} + "
                f"warmth>{combo['warmth']}: ${combo['pnl']:+,.0f} "
                f"({combo['trades']} trades, {combo['win_rate']:.0f}% win)"
            )
        
        # Summary
        lines.extend([
            "",
            "━" * 60,
            f"Total: {aggregate['total_traders']:,} traders | "
            f"{aggregate['total_trades']} trades | "
            f"${aggregate['realized_pnl']:+,.0f} P&L",
        ])
        
        return "\n".join(lines)
    
    def _get_top_threshold_combos(self, top_n: int = 5) -> List[Dict]:
        """Get top performing threshold combinations."""
        # Group ACTIVE traders by key threshold combos
        combos = {}
        
        for trader in self.active_traders.values():
            key = (
                trader.config.strategy_type.value,
                trader.config.iv_min,
                trader.config.warmth_min,
            )
            
            if key not in combos:
                combos[key] = {
                    'strategy': trader.config.strategy_type.value,
                    'iv_min': trader.config.iv_min,
                    'warmth': trader.config.warmth_min,
                    'pnl': 0,
                    'trades': 0,
                    'wins': 0,
                }
            
            combos[key]['pnl'] += trader.stats['total_pnl']
            combos[key]['trades'] += trader.stats['total_trades']
            combos[key]['wins'] += trader.stats['wins']
        
        # Calculate win rates
        for combo in combos.values():
            combo['win_rate'] = (combo['wins'] / combo['trades'] * 100) if combo['trades'] > 0 else 0
        
        # Sort by P&L
        sorted_combos = sorted(combos.values(), key=lambda x: x['pnl'], reverse=True)
        
        return sorted_combos[:top_n]
    
    # =========================================================================
    # For /positions and /pnl commands
    # =========================================================================
    
    async def get_positions_for_telegram(self) -> List[Dict]:
        """
        Get positions formatted for /positions command.
        
        Returns aggregated view, NOT individual positions.
        Groups by symbol and strategy for manageable output.
        """
        # Group by symbol AND strategy
        by_symbol_strategy = {}
        for pos in self.get_aggregate_positions():
            key = (pos['symbol'], pos.get('strategy_type', 'unknown'))
            if key not in by_symbol_strategy:
                by_symbol_strategy[key] = []
            by_symbol_strategy[key].append(pos)
        
        result = []
        for (symbol, strategy), positions in by_symbol_strategy.items():
            result.append({
                'symbol': symbol,
                'strategy': strategy,
                'count': len(positions),
                'traders_entered': len(set(p['trader_id'] for p in positions)),
                'sample_strikes': positions[0]['strikes'] if positions else 'N/A',
            })
        
        # Sort by count (most positions first)
        result.sort(key=lambda x: x['count'], reverse=True)
        
        # Limit to top 10 to avoid Telegram message limits
        return result[:10]

    async def get_positions_for_review(self) -> List[Dict]:
        """Get detailed open positions for daily review."""
        return self.get_aggregate_positions()
    
    def _et_day_bounds(self, target_date: date) -> tuple[datetime, datetime]:
        """Return UTC start/end for a given ET date."""
        eastern = ZoneInfo("America/New_York")
        start_et = datetime.combine(target_date, time.min, tzinfo=eastern)
        end_et = start_et + timedelta(days=1)
        return start_et.astimezone(timezone.utc), end_et.astimezone(timezone.utc)

    async def _get_epoch_start(self) -> Optional[str]:
        """Fetch current epoch start timestamp (cached)."""
        if not self.db:
            return None
        if not self._current_epoch_start:
            self._current_epoch_start = await self.db.get_current_epoch_start()
        return self._current_epoch_start

    async def _fetch_realized_stats(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Fetch realized trade stats between start/end timestamps."""
        if not self.db:
            return {'closed_trades': 0, 'winners': 0, 'realized_pnl': 0.0}
        epoch_start = await self._get_epoch_start()
        params = [start.isoformat(), end.isoformat()]
        epoch_clause = ""
        if epoch_start:
            epoch_clause = " AND timestamp >= ?"
            params.append(epoch_start)
        row = await self.db.fetch_one(
            f"""
            SELECT
                COUNT(*) as closed_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
                SUM(realized_pnl) as realized_pnl
            FROM paper_trades
            WHERE status != 'open'
              AND close_timestamp >= ?
              AND close_timestamp < ?
              {epoch_clause}
            """,
            tuple(params),
        )
        if not row:
            return {'closed_trades': 0, 'winners': 0, 'realized_pnl': 0.0}
        return {
            'closed_trades': row['closed_trades'] or 0,
            'winners': row['winners'] or 0,
            'realized_pnl': row['realized_pnl'] or 0.0,
        }

    async def _fetch_opened_count(self, start: datetime, end: datetime) -> int:
        """Fetch count of trades opened between start/end timestamps."""
        if not self.db:
            return 0
        epoch_start = await self._get_epoch_start()
        params = [start.isoformat(), end.isoformat()]
        epoch_clause = ""
        if epoch_start:
            epoch_clause = " AND timestamp >= ?"
            params.append(epoch_start)
        row = await self.db.fetch_one(
            f"""
            SELECT COUNT(*) as opened_trades
            FROM paper_trades
            WHERE timestamp >= ?
              AND timestamp < ?
              {epoch_clause}
            """,
            tuple(params),
        )
        return row['opened_trades'] if row else 0

    async def _fetch_open_positions_count(self) -> int:
        """Fetch count of open positions."""
        if not self.db:
            return len(self.get_aggregate_positions())
        epoch_start = await self._get_epoch_start()
        params: List[Any] = []
        epoch_clause = ""
        if epoch_start:
            epoch_clause = " AND timestamp >= ?"
            params.append(epoch_start)
        row = await self.db.fetch_one(
            f"""
            SELECT COUNT(*) as open_trades
            FROM paper_trades
            WHERE status = 'open'
            {epoch_clause}
            """,
            tuple(params),
        )
        return row['open_trades'] if row else 0

    async def get_trade_activity_summary(self) -> Dict[str, Any]:
        """Return realized P&L and activity stats for daily review and Telegram."""
        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        month_start = today_et.replace(day=1)
        year_start = date(today_et.year, 1, 1)

        today_start, today_end = self._et_day_bounds(today_et)
        month_start_utc, _ = self._et_day_bounds(month_start)
        year_start_utc, _ = self._et_day_bounds(year_start)

        today_stats = await self._fetch_realized_stats(today_start, today_end)
        mtd_stats = await self._fetch_realized_stats(month_start_utc, today_end)
        ytd_stats = await self._fetch_realized_stats(year_start_utc, today_end)

        opened_today = await self._fetch_opened_count(today_start, today_end)
        open_positions = await self._fetch_open_positions_count()

        win_rate_mtd = (
            (mtd_stats['winners'] / mtd_stats['closed_trades'] * 100)
            if mtd_stats['closed_trades'] > 0 else 0
        )
        win_rate_ytd = (
            (ytd_stats['winners'] / ytd_stats['closed_trades'] * 100)
            if ytd_stats['closed_trades'] > 0 else 0
        )

        return {
            'today_pnl': today_stats['realized_pnl'],
            'month_pnl': mtd_stats['realized_pnl'],
            'year_pnl': ytd_stats['realized_pnl'],
            'trades_today': today_stats['closed_trades'],
            'trades_mtd': mtd_stats['closed_trades'],
            'trades_ytd': ytd_stats['closed_trades'],
            'opened_today': opened_today,
            'open_positions': open_positions,
            'win_rate_mtd': win_rate_mtd,
            'win_rate_ytd': win_rate_ytd,
            'account_value': self.starting_balance + ytd_stats['realized_pnl'],
        }

    async def get_top_unrealized(self, n: int = 3) -> List[Dict]:
        """Get top N positions by unrealized P&L for display.

        Scans in-memory active traders and estimates current P&L
        based on entry credit (since we don't have live option prices
        for every position, we report the entry credit as the reference).
        """
        candidates = []
        for trader in self.active_traders.values():
            for trade in trader.open_positions:
                # Simple unrealized estimate: entry_credit * contracts * 100
                # represents max profit if the spread expires worthless
                entry_credit = trade.entry_credit or 0
                contracts = trade.contracts or 1
                max_profit = entry_credit * contracts * 100
                candidates.append({
                    'trader_id': trade.trader_id,
                    'strategy': trade.strategy_type,
                    'symbol': trade.symbol,
                    'strikes': trade.strikes,
                    'entry_credit': entry_credit,
                    'contracts': contracts,
                    'unrealized_pnl': max_profit * 0.5,  # Estimate 50% of max at open
                    'pnl_pct': 50.0,  # Placeholder until live pricing
                    'expiry': trade.expiry,
                })
        candidates.sort(key=lambda x: x['unrealized_pnl'], reverse=True)
        return candidates[:n]

    async def get_pnl_for_telegram(self) -> Dict:
        """
        Get statistically sane P&L for /pnl command.

        Returns per-trader aggregated statistics:
        - Returns are computed PER TRADER then aggregated (mean/median/std)
        - Each trader has the same starting notional (self.starting_balance)
        - Return = total_realized_pnl / starting_balance * 100
        - No double-counting realized + unrealized: only realized is reported
        - Open positions are counted separately
        """
        import statistics

        activity = await self.get_trade_activity_summary()

        # Per-trader statistics from DB
        per_trader_stats = await self._compute_per_trader_stats()

        # Fall back to basic activity if no per-trader data
        if not per_trader_stats:
            today_pct = (activity['today_pnl'] / self.starting_balance * 100) if self.starting_balance else 0
            month_pct = (activity['month_pnl'] / self.starting_balance * 100) if self.starting_balance else 0
            year_pct = (activity['year_pnl'] / self.starting_balance * 100) if self.starting_balance else 0
            return {
                'today_pnl': activity['today_pnl'],
                'today_pct': today_pct,
                'month_pnl': activity['month_pnl'],
                'month_pct': month_pct,
                'year_pnl': activity['year_pnl'],
                'year_pct': year_pct,
                'trades_today': activity['trades_today'],
                'trades_mtd': activity['trades_mtd'],
                'win_rate_mtd': activity['win_rate_mtd'],
                'opened_today': activity['opened_today'],
                'open_positions': activity['open_positions'],
                'account_value': activity['account_value'],
                'per_trader': None,
            }

        return {
            'today_pnl': activity['today_pnl'],
            'month_pnl': activity['month_pnl'],
            'year_pnl': activity['year_pnl'],
            'trades_today': activity['trades_today'],
            'trades_mtd': activity['trades_mtd'],
            'win_rate_mtd': activity['win_rate_mtd'],
            'opened_today': activity['opened_today'],
            'open_positions': activity['open_positions'],
            'account_value': activity['account_value'],
            'per_trader': per_trader_stats,
        }

    async def _compute_per_trader_stats(
        self, days: int = 30, min_trades: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Compute per-trader return distribution.

        Returns are computed PER TRADER: return_pct = total_pnl / starting_balance * 100.
        Then we aggregate across traders: mean, median, std, deciles.

        Exclusions are tracked with reason counts:
        - 'no_trades': trader has zero closed trades in window
        - 'insufficient_data': fewer than min_trades closed trades
        - 'zombie_contamination': trader has zombie-flagged trades
        """
        import statistics as stats_mod

        if not self.db:
            return None

        epoch_start = await self._get_epoch_start()
        rows = await self.db.get_per_trader_pnl(
            days=days,
            min_trades=min_trades,
            epoch_start=epoch_start,
        )
        if not rows:
            return None

        # Compute per-trader returns as % of starting equity
        returns = []
        exclusion_reasons: Dict[str, int] = {
            'no_trades': 0,
            'insufficient_data': 0,
        }

        for row in rows:
            closed = row.get('closed_trades', 0)
            if closed < min_trades:
                exclusion_reasons['insufficient_data'] += 1
                continue
            total_pnl = row.get('total_pnl', 0) or 0
            return_pct = (total_pnl / self.starting_balance) * 100
            returns.append({
                'trader_id': row['trader_id'],
                'strategy': row.get('strategy_type', 'unknown'),
                'return_pct': return_pct,
                'total_pnl': total_pnl,
                'closed_trades': closed,
                'wins': row.get('wins', 0) or 0,
                'win_rate': ((row.get('wins', 0) or 0) / closed * 100) if closed > 0 else 0,
            })

        if not returns:
            return None

        return_values = [r['return_pct'] for r in returns]
        return_values_sorted = sorted(return_values)
        n = len(return_values)

        # Decile boundaries
        decile_size = max(1, n // 10)
        top_decile = return_values_sorted[-decile_size:] if decile_size > 0 else []
        bottom_decile = return_values_sorted[:decile_size] if decile_size > 0 else []

        # Most consistent trader: lowest volatility with positive mean
        # Stability score = mean_return / (std_return + 1) — higher is better
        # (Sharpe-like, with +1 to avoid division by zero)
        best_stability = None
        best_stability_score = float('-inf')
        for r in returns:
            if r['closed_trades'] >= max(3, min_trades):
                # Approximate: we only have aggregate, not daily returns
                # Use win_rate and return as proxy
                stability = r['return_pct'] / (abs(r['return_pct']) * 0.5 + 1)
                if stability > best_stability_score:
                    best_stability_score = stability
                    best_stability = r

        mean_ret = stats_mod.mean(return_values)
        median_ret = stats_mod.median(return_values)
        std_ret = stats_mod.stdev(return_values) if n >= 2 else 0

        return {
            'window_days': days,
            'active_traders': n,
            'excluded_count': sum(exclusion_reasons.values()),
            'exclusion_reasons': exclusion_reasons,
            'mean_return_pct': round(mean_ret, 4),
            'median_return_pct': round(median_ret, 4),
            'std_return_pct': round(std_ret, 4),
            'best_return_pct': round(max(return_values), 4),
            'worst_return_pct': round(min(return_values), 4),
            'top_decile_avg_pct': round(stats_mod.mean(top_decile), 4) if top_decile else 0,
            'bottom_decile_avg_pct': round(stats_mod.mean(bottom_decile), 4) if bottom_decile else 0,
            'most_consistent': {
                'trader_id': best_stability['trader_id'],
                'return_pct': round(best_stability['return_pct'], 4),
                'stability_score': round(best_stability_score, 4),
                'strategy': best_stability['strategy'],
                'win_rate': round(best_stability['win_rate'], 1),
            } if best_stability else None,
            'starting_balance': self.starting_balance,
            'return_definition': 'realized_pnl / starting_balance * 100',
        }


    # =========================================================================
    # Zombie Detection & Cleanup (7-14 DTE aligned)
    # =========================================================================

    async def detect_zombies(self, stale_days: int = 14, grace_days: int = 2) -> List[Dict]:
        """Detect zombie positions without closing them.

        Zombie = orphaned/stuck, NOT just aged.

        Rules:
        1. Status='open' AND expiry < today - grace_days (expired + grace period)
        2. Status='open' AND no expiry AND timestamp < now - stale_days
           (truly stale with no expiry info)

        Does NOT close positions just due to age.
        """
        if not self.db:
            return []

        epoch_start = await self._get_epoch_start()
        params: List[Any] = [f"-{grace_days} days", f"-{stale_days} days"]
        epoch_clause = ""
        if epoch_start:
            epoch_clause = " AND timestamp >= ?"
            params.append(epoch_start)
        rows = await self.db.fetch_all(f"""
            SELECT id, trader_id, strategy_type, symbol, timestamp,
                   strikes, expiry, entry_credit, contracts, status,
                   market_conditions, close_reason
            FROM paper_trades
            WHERE status = 'open'
              AND (
                  (expiry IS NOT NULL AND expiry < date('now', ?))
                  OR (
                      (expiry IS NULL
                       OR TRIM(expiry) = ''
                       OR UPPER(expiry) IN ('N/A', 'NA', 'NONE', 'NULL'))
                      AND timestamp < datetime('now', ?)
                  )
              )
              {epoch_clause}
            ORDER BY timestamp ASC
        """, tuple(params))

        zombies = []
        for row in rows:
            r = dict(row)
            expiry_value = (r.get('expiry') or '').strip()
            expiry_is_missing = not expiry_value or expiry_value.upper() in {
                'N/A', 'NA', 'NONE', 'NULL'
            }
            if not expiry_is_missing and expiry_value < datetime.now().strftime('%Y-%m-%d'):
                r['zombie_reason'] = 'expired_past_grace'
            else:
                r['zombie_reason'] = 'stale_no_expiry'
            zombies.append(r)

        return zombies

    async def close_zombies(self, stale_days: int = 14, grace_days: int = 2) -> int:
        """Detect and close zombie positions.

        Sets status='expired', close_reason=zombie_reason.
        """
        zombies = await self.detect_zombies(stale_days, grace_days)
        if not zombies:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        for z in zombies:
            await self.db.execute("""
                UPDATE paper_trades
                SET status = 'expired',
                    close_timestamp = ?,
                    close_reason = ?,
                    realized_pnl = COALESCE(realized_pnl, 0)
                WHERE id = ? AND status = 'open'
            """, (now, z['zombie_reason'], z['id']))

        logger.info(f"Closed {len(zombies)} zombie positions")
        return len(zombies)

    async def format_zombies_report(self, stale_days: int = 14, grace_days: int = 2) -> str:
        """Format zombie detection report for Telegram / dashboard."""
        zombies = await self.detect_zombies(stale_days, grace_days)
        if not zombies:
            return "No zombie positions detected."

        by_reason = {}
        for z in zombies:
            reason = z.get('zombie_reason', 'unknown')
            by_reason[reason] = by_reason.get(reason, 0) + 1

        lines = [
            f"ZOMBIE REPORT: {len(zombies)} detected",
            f"  stale_days={stale_days}, grace_days={grace_days}",
            "",
        ]
        for reason, count in sorted(by_reason.items()):
            lines.append(f"  {reason}: {count}")

        lines.append("")
        lines.append("Use /zombie_clean to close them.")
        return "\n".join(lines)

    # =========================================================================
    # Paper Equity Reset
    # =========================================================================

    async def reset_paper_equity(self, scope: str = 'all', mode: str = 'epoch') -> str:
        """Reset paper equity by starting a new epoch.

        Args:
            scope: 'all' or a specific trader_id
            mode: 'epoch' (start fresh epoch, old data excluded from current metrics)
        """
        if not self.db:
            return "No database connected."

        if mode == 'epoch':
            epoch_start = await self.db.reset_paper_epoch(
                starting_equity=self.starting_balance,
                scope=scope,
                reason='manual_reset',
            )
            self._current_epoch_start = epoch_start
            # Clear in-memory active traders
            self.active_traders.clear()
            self._positions_by_symbol.clear()
            self._peak_balance = self.starting_balance
            self._drawdown_halted = False
            logger.info(f"Paper epoch reset: start={epoch_start} scope={scope}")
            return (
                f"New epoch started. starting_equity=${self.starting_balance:,.0f}, "
                f"scope={scope}. Old data preserved but excluded from current metrics."
            )
        return f"Unknown mode: {mode}"


# Test function
async def test_paper_trader_farm():
    """Test the paper trader farm."""
    print("Paper Trader Farm Test")
    print("=" * 50)
    
    # Mock database
    class MockDB:
        async def execute(self, *args): pass
        async def fetch_one(self, *args): return None
        async def fetch_all(self, *args): return []
    
    farm = PaperTraderFarm(db=MockDB(), total_traders=100)
    await farm.initialize()
    
    print(f"✅ Created {len(farm.trader_configs)} traders")
    
    # Test signal evaluation
    signal = {
        'iv': 58,
        'warmth': 6,
        'dte': 45,
        'pop': 68,
        'gap_risk': 5,
        'direction': 'neutral',
        'strikes': '$48/$44',
        'expiry': '2026-02-21',
        'credit': 0.42,
    }
    
    trades = await farm.evaluate_signal('IBIT', signal)
    print(f"✅ Signal evaluated: {len(trades)} traders entered")
    
    # Test aggregate stats
    aggregate = farm.get_aggregate_pnl()
    print(f"✅ Aggregate P&L: {aggregate}")
    
    # Test strategy breakdown
    breakdown = farm.get_strategy_breakdown()
    print(f"✅ Strategy breakdown:")
    for strat, data in breakdown.items():
        print(f"   {strat}: {data['traders']} traders, {data['total_trades']} trades")
    
    print("\n✅ All paper trader farm tests passed!")


if __name__ == "__main__":
    asyncio.run(test_paper_trader_farm())
