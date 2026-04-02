# Created by Oliver Meihls

# Options IV Spike Detector
#
# Detects implied volatility spikes for manual options trading.
#
# NOTE: This detector now LOGS data only, does NOT alert.
# User cannot trade on Deribit. IBIT/BITO opportunities use
# the IBITDetector which has its own alerts.
#
# This detector feeds IV data to the IBIT detector.

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from .base_detector import BaseDetector
from ..core.utils import calculate_z_score, calculate_mean, calculate_std


class OptionsIVDetector(BaseDetector):
    # Detector for options implied volatility spikes.
    #
    # Strategy: When IV spikes >80% during panic, sell premium
    # via put spreads or other strategies.
    #
    # CHANGED: This detector now LOGS IV data but does NOT send alerts.
    # The IBIT detector uses this IV data to make IBIT/BITO recommendations.
    # User cannot trade on Deribit directly.
    #
    # Alert spam fix: Added 4-hour cooldown per currency.
    
    def __init__(self, config: Dict[str, Any], db):
        super().__init__(config, db)
        
        # Thresholds
        self.iv_threshold = config.get('threshold_percent', 80)
        self.z_score_threshold = config.get('z_score_threshold', 2.0)
        self.lookback_days = config.get('lookback_days', 30)
        
        # Option selection
        self.min_dte = config.get('min_days_to_expiry', 3)
        self.max_dte = config.get('max_days_to_expiry', 14)
        self.otm_percent = config.get('otm_percent', 10)
        
        # ALERT COOLDOWN - 3 hours minimum between alerts for same currency
        self.cooldown_hours = config.get('cooldown_hours', 3)
        self._last_alert_time: Dict[str, datetime] = {}
        
        # DISABLE DERIBIT ALERTS - user can't trade there
        # This detector now ONLY logs, no Telegram alerts
        self.alerts_enabled = config.get('alerts_enabled', False)  # Default OFF
        
        # IV history cache
        self._iv_history: Dict[str, List[float]] = {}
        
        self.logger.info(
            f"OptionsIVDetector initialized: IV threshold={self.iv_threshold}%, "
            f"cooldown={self.cooldown_hours}h, alerts_enabled={self.alerts_enabled}"
        )
    
    def _check_cooldown(self, currency: str) -> bool:
        # Check if we're still in cooldown for this currency.
        if currency not in self._last_alert_time:
            return False  # No cooldown
        
        elapsed = datetime.now(timezone.utc) - self._last_alert_time[currency]
        cooldown_delta = timedelta(hours=self.cooldown_hours)
        
        return elapsed < cooldown_delta
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        # Analyze ATM IV for spike opportunities.
        #
        # Args:
        # market_data: IV data from Deribit
        # - currency: BTC or ETH
        # - atm_iv: Current ATM implied volatility
        # - index_price: Current underlying price
        #
        # Returns:
        # Detection if IV spike found, with alert_tier=3 (log only, no Telegram)
        if not self.enabled:
            return None
        
        currency = market_data.get('currency', 'BTC')
        atm_iv = market_data.get('atm_iv')
        index_price = market_data.get('index_price')
        
        if atm_iv is None or atm_iv == 0:
            return None
        
        # Update IV history (always do this for data collection)
        await self._update_history(currency, atm_iv)
        
        history = self._iv_history.get(currency, [])
        
        # Need some history for comparison
        if len(history) < 10:
            return None
        
        # Check absolute threshold
        if atm_iv < self.iv_threshold:
            return None
        
        # Calculate z-score
        z_score = calculate_z_score(atm_iv, history)
        
        if z_score < self.z_score_threshold:
            return None
        
        # Check cooldown BEFORE creating detection
        if self._check_cooldown(currency):
            self.logger.debug(f"IV spike for {currency} suppressed (cooldown active)")
            return None
        
        # IV spike detected!
        mean_iv = calculate_mean(history)
        std_iv = calculate_std(history)
        
        # Calculate suggested strikes
        suggested_put_strike = index_price * (1 - self.otm_percent / 100)
        suggested_put_spread_width = index_price * 0.05  # 5% width
        
        # IMPORTANT: Set alert_tier to 3 (log only) since user can't trade Deribit
        # The IBIT detector will use this IV data to make IBIT/BITO recommendations
        alert_tier = 3 if not self.alerts_enabled else 1
        
        detection = self.create_detection(
            opportunity_type='options_iv',
            asset=currency,
            exchange='deribit',
            detection_data={
                'current_iv': atm_iv / 100,  # Store as decimal
                'mean_iv': mean_iv / 100,
                'std_iv': std_iv / 100,
                'z_score': z_score,
                'underlying_price': index_price,
                'suggested_put_strike': round(suggested_put_strike, -3),  # Round to 1000
                'suggested_spread_width': suggested_put_spread_width,
                # Flag to indicate this is data-only, not actionable
                'is_data_only': not self.alerts_enabled,
            },
            current_price=index_price,
            estimated_edge_bps=100,  # Arbitrary - manual trade
            alert_tier=alert_tier,  # Tier 3 = log only (no Telegram)
            notes=f"IV DATA: {currency} IV={atm_iv:.0f}% (avg: {mean_iv:.0f}%) - Data for IBIT/BITO decisions"
        )
        
        # Update cooldown
        self._last_alert_time[currency] = datetime.now(timezone.utc)
        
        await self.log_detection(detection)
        
        # Log that we detected but didn't alert
        self.logger.info(
            f"IV spike logged (no alert): {currency} {atm_iv:.1f}% "
            f"(z={z_score:.2f}) - alerts_enabled={self.alerts_enabled}"
        )
        
        return detection
    
    async def _update_history(self, currency: str, iv: float) -> None:
        # Update IV history cache.
        if currency not in self._iv_history:
            # Could load from database here
            self._iv_history[currency] = []
        
        self._iv_history[currency].append(iv)
        
        # Keep reasonable history (assuming ~1 update per minute)
        max_entries = self.lookback_days * 24 * 60
        if len(self._iv_history[currency]) > max_entries:
            self._iv_history[currency] = self._iv_history[currency][-max_entries:]
        
        # Store in database
        await self.db.insert_options_iv(
            asset=currency,
            expiry='ATM',  # Synthetic ATM entry
            strike=0,
            option_type='atm',
            iv=iv / 100,
        )
    
    def calculate_edge(self, detection: Dict) -> float:
        # Edge calculation not applicable for manual trades.
        return 100  # Placeholder
    
    def get_iv_history(self, currency: str) -> List[float]:
        # Get cached IV history.
        return self._iv_history.get(currency, [])
    
    def get_current_iv(self, currency: str) -> Optional[float]:
        # Get most recent IV value.
        history = self._iv_history.get(currency, [])
        return history[-1] if history else None
    
    def get_cooldown_status(self) -> Dict[str, str]:
        # Get cooldown status for each currency.
        status = {}
        now = datetime.now(timezone.utc)
        
        for currency, last_time in self._last_alert_time.items():
            elapsed = now - last_time
            remaining = timedelta(hours=self.cooldown_hours) - elapsed
            
            if remaining.total_seconds() > 0:
                status[currency] = f"cooldown ({remaining.seconds // 60}m remaining)"
            else:
                status[currency] = "ready"
        
        return status

