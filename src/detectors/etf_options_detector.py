# Created by Oliver Meihls

# Crypto ETF Options Opportunity Detector
#
# Detects opportunities to sell put spreads on crypto ETFs (IBIT, BITO, etc.).
# Integrates with TradeCalculator for precise recommendations.

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional
import logging

from .base_detector import BaseDetector
from ..core.events import BarEvent, Priority
from ..core.utils import calculate_z_score, calculate_mean
from ..core.economic_calendar import EconomicCalendar
from ..analysis.trade_calculator import TradeCalculator, TradeRecommendation
from ..analysis.greeks_engine import GreeksEngine
from ..analysis.paper_trader import PaperTrader
from ..connectors.ibit_options_client import IBITOptionsClient

logger = logging.getLogger(__name__)


class ETFOptionsDetector(BaseDetector):
    # Detector for ETF options opportunities (v2).
    #
    # Strategy: When volatility spikes and an ETF drops significantly,
    # sell put spreads to collect elevated premium.
    #
    # v2 Features:
    # - Real options chain data via yfinance
    # - Greeks calculation (Delta, Theta, Vega)
    # - Dynamic position sizing based on PoP
    # - Economic calendar blackout warnings
    # - IV Rank validation
    
    def __init__(self, config: Dict[str, Any], db, symbol: str = "SPY", iv_consensus: Optional[Any] = None):
        super().__init__(config, db)
        
        # Symbol this detector tracks
        self.symbol = symbol.upper()
        self.iv_consensus = iv_consensus
        
        # Volatility proxy symbol (e.g. BTC for IBIT/BITO, or self-symbol for others)
        _CRYPTO_ETF_SYMBOLS = {"IBIT", "BITO"}
        self.vol_proxy = config.get('vol_proxy', self.symbol if self.symbol not in _CRYPTO_ETF_SYMBOLS else "BTC")
        
        # Thresholds
        self.vol_iv_threshold = config.get('vol_iv_threshold', config.get('btc_iv_threshold', 50 if self.symbol not in _CRYPTO_ETF_SYMBOLS else 70))
        self.drop_threshold = config.get('drop_threshold', -3)
        self.combined_score_threshold = config.get('combined_score_threshold', 1.5)
        self.iv_rank_threshold = config.get('iv_rank_threshold', 50)
        
        # Alert cooldown
        self.cooldown_hours = config.get('cooldown_hours', 3)
        self._last_alert_time: Optional[datetime] = None
        
        # Data cache
        self._proxy_iv_history: List[float] = []
        self._current_proxy_iv: float = 0
        self._current_proxy_greeks: Optional[Dict[str, float]] = None
        self._current_symbol_data: Optional[Dict] = None
        self._symbol_price_history: List[Dict] = []
        
        # Initialize components
        self.account_size = config.get('account_size', 3000)
        self.trade_calculator = TradeCalculator(
            account_size=self.account_size, 
            symbol=self.symbol,
            iv_consensus=iv_consensus
        )
        self.options_client = IBITOptionsClient(symbol=self.symbol)
        self.economic_calendar = EconomicCalendar()
        
        # Paper trading - Farm integration
        self.paper_trading_enabled = config.get('paper_trading', True)
        self.paper_trader: Optional[PaperTrader] = None
        self.paper_trader_farm = None  # Set by orchestrator
        
        # Telegram callback for paper trade notifications
        self._telegram_callback = None
        
        self.logger.info(
            f"{self.symbol}Detector initialized (proxy: {self.vol_proxy}): "
            f"Vol IV >{self.vol_iv_threshold}%, "
            f"{self.symbol} drop >{abs(self.drop_threshold)}%, "
            f"IV Rank >{self.iv_rank_threshold}%, "
            f"consensus={'YES' if iv_consensus else 'NO'}"
        )
    
    def set_telegram_callback(self, callback) -> None:
        # Set callback for sending Telegram notifications.
        self._telegram_callback = callback
        
    def set_paper_trader_farm(self, farm) -> None:
        # Connect the larger scale paper trader farm.
        self.paper_trader_farm = farm
    
    def update_proxy_iv(self, iv: float) -> None:
        # Update volatility proxy IV data.
        self._current_proxy_iv = iv
        self._proxy_iv_history.append(iv)
        if len(self._proxy_iv_history) > 168:
            self._proxy_iv_history = self._proxy_iv_history[-168:]
    
    def update_symbol_data(self, data: Dict) -> None:
        # Update ETF price data.
        self._current_symbol_data = data
        self._symbol_price_history.append({
            'price': data.get('price', 0),
            'timestamp': data.get('timestamp'),
        })
        if len(self._symbol_price_history) > 720:
            self._symbol_price_history = self._symbol_price_history[-720:]

    def get_signal_checklist(self) -> Dict[str, Any]:
        # Return a checklist of conditions for an ETF signal.
        proxy_iv = self._current_proxy_iv or 0
        symbol_change = 0.0
        symbol_price = 0.0
        if self._current_symbol_data:
            symbol_change = self._current_symbol_data.get('price_change_pct', 0)
            symbol_price = self._current_symbol_data.get('price', 0)

        iv_elevated = proxy_iv >= self.vol_iv_threshold
        symbol_dropped = symbol_change <= self.drop_threshold
        iv_score = proxy_iv / self.vol_iv_threshold if self.vol_iv_threshold > 0 else 0
        drop_score = abs(symbol_change) / abs(self.drop_threshold) if symbol_change < 0 else 0
        combined_score = (iv_score + drop_score) / 2 if (self.vol_iv_threshold and self.drop_threshold) else 0

        # Cooldown only applies to live alert path (analyze()), NOT research farm
        cooldown_remaining = None
        cooldown_note = 'research_bypasses'
        if self._last_alert_time:
            elapsed = (datetime.now(timezone.utc) - self._last_alert_time).total_seconds() / 3600
            remaining = max(0.0, self.cooldown_hours - elapsed)
            if remaining > 0:
                cooldown_remaining = remaining

        iv_rank = None
        try:
            status = self.options_client.get_market_status()
            iv_rank = status.get('iv_rank')
        except Exception:
            iv_rank = None

        return {
            'symbol': self.symbol,
            'proxy': self.vol_proxy,
            'proxy_iv': proxy_iv,
            'vol_iv_threshold': self.vol_iv_threshold,
            'vol_iv_ok': iv_elevated,
            'price': symbol_price,
            'change_pct': symbol_change,
            'drop_threshold': self.drop_threshold,
            'drop_ok': symbol_dropped,
            'combined_score': combined_score,
            'combined_score_threshold': self.combined_score_threshold,
            'combined_score_ok': combined_score >= self.combined_score_threshold,
            'iv_rank': iv_rank,
            'iv_rank_threshold': self.iv_rank_threshold,
            'iv_rank_ok': iv_rank is not None and iv_rank >= self.iv_rank_threshold,
            'cooldown_remaining_hours': cooldown_remaining,
            'cooldown_note': cooldown_note,
            'has_proxy_iv': bool(self._current_proxy_iv),
            'has_symbol_data': bool(self._current_symbol_data),
        }

    def get_research_signal(
        self,
        conditions_score: int,
        conditions_label: str,
        regime_change_24h_pct: float,
        regime_change_5d_pct: float,
        timestamp: str,
    ) -> Optional[Dict[str, Any]]:
        # Build a research-mode signal without gating thresholds.
        if not self._current_proxy_iv or not self._current_symbol_data:
            return None

        regime_change_decimal = (regime_change_24h_pct or 0) / 100
        symbol_change_pct = self._current_symbol_data.get('price_change_pct', 0)
        symbol_change_decimal = symbol_change_pct / 100

        recommendation = self.trade_calculator.generate_recommendation(
            btc_change_24h=regime_change_decimal,
            symbol_change_24h=symbol_change_decimal,
            force=True,
        )
        if not recommendation:
            return None

        direction = 'neutral'
        if regime_change_24h_pct > 0.2 or regime_change_5d_pct > 4.0:
            direction = 'bullish'
        elif regime_change_24h_pct < -0.2 or regime_change_5d_pct < -4.0:
            direction = 'bearish'

        return {
            'symbol': self.symbol,
            'price': recommendation.underlying_price,
            'iv': self._current_proxy_iv,
            'warmth': conditions_score,
            'dte': recommendation.dte,
            'pop': recommendation.probability_of_profit,
            'pot': recommendation.probability_of_touch_stop,
            'gap_risk': 0,
            'direction': direction,
            'strikes': f"{recommendation.short_strike}/{recommendation.long_strike}",
            'expiry': recommendation.expiration,
            'credit': recommendation.net_credit,
            'iv_rank': recommendation.iv_rank,
            'conditions_label': conditions_label,
            'proxy_change_pct': regime_change_24h_pct,
            'proxy_change_5d_pct': regime_change_5d_pct,
            'symbol_change_pct': symbol_change_pct,
            'timestamp': timestamp,
        }
    
    # ── bar-driven path (event bus) ───────────────────────

    def on_bar(self, event: BarEvent) -> None:
        # Consume bars as context/features.
        if event.symbol == self.vol_proxy and event.source != 'yahoo':
            # Cryptos come as non-yahoo bars
            pass

        elif event.symbol == self.symbol:
            # Main symbol bar
            self.update_symbol_data({
                'price': event.close,
                'timestamp': str(event.timestamp),
                'price_change_pct': 0,  # not available from bar alone
            })

    def _check_market_hours(self) -> bool:
        # Check if US stock market is open (9:30 AM - 4:00 PM ET).
        eastern = ZoneInfo("America/New_York")
        now = datetime.now(eastern)
        is_weekday = now.weekday() < 5
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return is_weekday and market_open <= now <= market_close
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        # Analyze for ETF options opportunities.
        if not self.enabled:
            return None
        
        # Update data based on source
        source = market_data.get('source')
        if source == 'deribit' and self.vol_proxy == 'BTC':
            iv = market_data.get('atm_iv', 0)
            if iv > 0:
                self.update_proxy_iv(iv)
            
            greeks = market_data.get('greeks')
            if greeks:
                self._current_proxy_greeks = greeks
        elif source == 'yahoo':
            symbol = market_data.get('symbol')
            if symbol == self.symbol:
                self.update_symbol_data(market_data)
                # If using self-IV as proxy, update proxy IV too (common for non-crypto ETFs)
                if self.vol_proxy == self.symbol:
                    iv = market_data.get('iv', 0)
                    if iv > 0:
                        self.update_proxy_iv(iv)
            elif symbol == self.vol_proxy:
                iv = market_data.get('iv', 0)
                if iv > 0:
                    self.update_proxy_iv(iv)
        
        # Need both data points
        if not self._current_proxy_iv or not self._current_symbol_data:
            return None
        
        # Check cooldown
        if self._last_alert_time:
            elapsed = (datetime.now(timezone.utc) - self._last_alert_time).total_seconds() / 3600
            if elapsed < self.cooldown_hours:
                return None
        
        # === BASIC THRESHOLD CHECKS ===
        vol_iv = self._current_proxy_iv
        symbol_change = self._current_symbol_data.get('price_change_pct', 0)
        symbol_price = self._current_symbol_data.get('price', 0)
        
        iv_elevated = vol_iv >= self.vol_iv_threshold
        symbol_dropped = symbol_change <= self.drop_threshold
        
        # Combined score
        iv_score = vol_iv / self.vol_iv_threshold if self.vol_iv_threshold > 0 else 0
        drop_score = abs(symbol_change) / abs(self.drop_threshold) if symbol_change < 0 else 0
        combined_score = (iv_score + drop_score) / 2
        
        if combined_score < self.combined_score_threshold:
            return None
        
        if not (iv_elevated or symbol_dropped):
            return None
        
        # === IV RANK CHECK ===
        try:
            market_status = self.options_client.get_market_status()
            iv_rank = market_status.get('iv_rank', 50)
            
            if iv_rank < self.iv_rank_threshold:
                self.logger.debug(f"IV Rank {iv_rank:.1f}% below threshold {self.iv_rank_threshold}%")
                return None
        except Exception as e:
            self.logger.warning(f"Could not check IV Rank: {e}")
            iv_rank = 50  # Assume neutral
        
        # === ECONOMIC CALENDAR CHECK ===
        blackout_warning = None
        is_blackout, blackout_event = self.economic_calendar.is_blackout_period()
        if is_blackout:
            blackout_warning = self.economic_calendar.get_blackout_warning()
        
        # === GENERATE TRADE RECOMMENDATION ===
        try:
            # Note: trade_calculator uses 'btc_change_24h' as a legacy parameter name
            # but we pass the proxy change regardless.
            recommendation = self.trade_calculator.generate_recommendation(
                btc_change_24h=market_data.get('btc_change_24h', 0) if self.vol_proxy == 'BTC' else 0,
                symbol_change_24h=symbol_change / 100,
                force=True,
                proxy_greeks=self._current_proxy_greeks,
            )
            
            if not recommendation:
                self.logger.warning("Trade calculator could not generate recommendation")
                return None
            
            if blackout_warning:
                recommendation.warnings.append(blackout_warning)
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return None
        
        # === GPU SURFACE ANALYSIS ===
        skew_anomaly = None
        try:
            chain = self.options_client.get_chain(recommendation.expiration)
            if chain:
                from ..analysis.gpu_engine import get_gpu_engine
                gpu = get_gpu_engine()
                
                strikes = [p['strike'] for p in chain['puts']]
                sigmas = [p['iv'] for p in chain['puts']]
                
                T = GreeksEngine.dte_to_years(recommendation.dte)
                batch = gpu.batch_greeks(
                    symbol_price, strikes, T, sigmas, option_type='put'
                )
                
                for i in range(1, len(sigmas) - 1):
                    if sigmas[i] > sigmas[i-1] + 3 and sigmas[i] > sigmas[i+1] + 3:
                        skew_anomaly = f"IV Skew Anomaly at ${strikes[i]}"
                        recommendation.warnings.append(f"🎯 SKEW ANOMALY: ${strikes[i]} strike is overpriced (GPU found {sigmas[i]:.1f}% IV vs neighbors)")
                        break
        except Exception as e:
            self.logger.debug(f"GPU Surface analysis skipped: {e}")
            
        # === BUILD DETECTION ===
        iv_z_score = 0
        if len(self._proxy_iv_history) > 10:
            iv_z_score = calculate_z_score(vol_iv, self._proxy_iv_history)
        
        from dataclasses import asdict
        detection = self.create_detection(
            opportunity_type='etf_options',
            asset=self.symbol,
            exchange='robinhood',
            detection_data={
                'proxy': self.vol_proxy,
                'vol_iv': vol_iv,
                'vol_iv_z_score': iv_z_score,
                'price': symbol_price,
                'change_24h': symbol_change,
                'combined_score': combined_score,
                'iv_rank': iv_rank,
                'expiration': recommendation.expiration,
                'dte': recommendation.dte,
                'short_strike': recommendation.short_strike,
                'long_strike': recommendation.long_strike,
                'net_credit': recommendation.net_credit,
                'max_risk': recommendation.max_risk,
                'probability_of_profit': recommendation.probability_of_profit,
                'in_blackout': is_blackout,
                'recommendation': asdict(recommendation),
            },
            current_price=symbol_price,
            estimated_edge_bps=int(recommendation.net_credit / recommendation.max_risk * 100) if recommendation.max_risk > 0 else 100,
            alert_tier=1,
            notes=f"{self.symbol} put spread: ${recommendation.short_strike:.0f}/${recommendation.long_strike:.0f}, Credit: ${recommendation.net_credit:.2f}"
        )
        
        self._last_alert_time = datetime.now(timezone.utc)
        await self.log_detection(detection)
        
        # === AUTO-LOG PAPER TRADE ===
        if self.paper_trading_enabled:
            try:
                if self.paper_trader is None:
                    self.paper_trader = PaperTrader(self.db)
                
                paper_trade = await self.paper_trader.open_trade(
                    recommendation=recommendation,
                    btc_iv=vol_iv,
                )
                detection['detection_data']['paper_trade_id'] = paper_trade.id
                
                if self.paper_trader_farm:
                    farm_trades = await self.paper_trader_farm.evaluate_signal(
                        symbol=self.symbol,
                        signal_data={
                            'iv': vol_iv,
                            'warmth': combined_score,
                            'dte': recommendation.dte,
                            'pop': recommendation.probability_of_profit,
                            'strikes': f"{recommendation.short_strike}/{recommendation.long_strike}",
                            'expiry': recommendation.expiration,
                            'credit': recommendation.net_credit,
                        }
                    )
                    detection['detection_data']['farm_count'] = len(farm_trades)
                
                if self._telegram_callback:
                    paper_msg = f"📝 <b>PAPER TRADE: {self.symbol}</b>\n#{paper_trade.id} ${paper_trade.short_strike:.0f}/${paper_trade.long_strike:.0f} Spread"
                    if hasattr(self._telegram_callback, '__name__') and self._telegram_callback.__name__ == 'send_alert':
                        await self._telegram_callback(paper_msg.strip(), priority=3)
                    else:
                        await self._telegram_callback(paper_msg.strip())
            except Exception as e:
                self.logger.warning(f"Failed to log paper trade: {e}")
        
        return detection
    
    def format_telegram_alert(self, detection: Dict) -> str:
        # Format detection for Telegram.
        data = detection.get('detection_data', {})
        recommendation = data.get('recommendation')
        
        if recommendation:
            if isinstance(recommendation, dict):
                recommendation = TradeRecommendation(**recommendation)
            alert = self.trade_calculator.format_telegram_alert(recommendation)
            
            # Add farm confirmation
            farm_count = data.get('farm_count', 0)
            if farm_count > 0:
                alert += f"\n\n🚜 *FARM CHECK*: {farm_count:,} parallel traders followed signal"
            
            return alert
        
        # Fallback to basic format
        return f"""
🎯 *{self.symbol} SIGNAL*
Proxy IV: {data.get('vol_iv', 0):.0f}%
Price: ${data.get('price', 0):.2f} ({data.get('change_24h', 0):+.1f}%)
IV Rank: {data.get('iv_rank', 0):.0f}%
SELL: ${data.get('short_strike', 0):.0f} Put / BUY: ${data.get('long_strike', 0):.0f} Put
"""
    
    def calculate_edge(self, detection: Dict) -> float:
        # Calculate edge based on credit/risk ratio.
        data = detection.get('detection_data', {})
        credit = data.get('net_credit', 0)
        risk = data.get('max_risk', 1)
        return (credit / risk * 100) if risk > 0 else 100
    
    def get_current_conditions(self) -> Dict:
        # Get current market conditions.
        iv_rank = 50
        try:
            status = self.options_client.get_market_status()
            iv_rank = status.get('iv_rank', 50)
        except:
            pass
        
        return {
            'proxy_iv': self._current_proxy_iv,
            'proxy_avg': calculate_mean(self._proxy_iv_history) if self._proxy_iv_history else 0,
            'price': self._current_symbol_data.get('price', 0) if self._current_symbol_data else 0,
            'change_24h': self._current_symbol_data.get('price_change_pct', 0) if self._current_symbol_data else 0,
            'iv_rank': iv_rank,
            'market_open': self._check_market_hours(),
            'in_blackout': self.economic_calendar.is_blackout_period()[0],
        }
    
    def is_market_open(self) -> bool:
        # Check if stock market is currently open.
        return self._check_market_hours()
