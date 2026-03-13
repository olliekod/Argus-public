"""
Signal Warmth Monitor
=====================

Shows how "warm" conditions are - forewarning before optimal trading conditions.

Temperature levels:
- COLD (0-25%):  Far from trade conditions
- COOL (25-50%): Getting interesting  
- WARM (50-75%): Close to trigger
- HOT (75-100%): Trade conditions met!

Runs continuously and sends Telegram alerts when temperature changes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class WarmthStatus:
    """Current warmth state."""
    temperature: int  # 0-100
    level: str        # COLD, COOL, WARM, HOT
    gates_passed: int
    total_gates: int
    details: Dict
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class WarmthMonitor:
    """
    Monitors conditions and reports "temperature" toward trade signals.
    
    Sends Telegram alerts when:
    - Temperature moves up a level (COLD→COOL, COOL→WARM, etc.)
    - HOT condition reached (trade signal!)
    - Temperature drops significantly
    """
    
    LEVELS = {
        (0, 25): 'COLD',
        (25, 50): 'COOL', 
        (50, 75): 'WARM',
        (75, 101): 'HOT',
    }
    
    def __init__(self, telegram_bot=None, config_path: str = "config/strategy_params.json"):
        """Initialize warmth monitor."""
        self.bot = telegram_bot
        self.config_path = Path(config_path)
        self._params = self._load_params()
        self._last_level: Optional[str] = None
        self._last_alert_time: Optional[datetime] = None
        
    def _load_params(self) -> Dict:
        """Load strategy parameters."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return config.get('paper', config.get('live', {}))
        
        return {
            'iv_threshold': 0.25,
            'price_drop_trigger': -0.005,
        }
    
    def calculate_warmth(
        self,
        iv_rank: float,
        price_change_pct: float,
        is_market_hours: bool,
        has_event_blackout: bool,
        fear_greed: int = 50,
    ) -> WarmthStatus:
        """
        Calculate current temperature.
        
        Each gate contributes to overall warmth:
        - Market hours: 25 points (binary)
        - IV rank: 0-25 points (scaled)
        - Price dip: 0-25 points (scaled)
        - No blackout: 15 points (binary)
        - Fear (not greed): 0-10 points (scaled)
        
        Total: 100 points max
        """
        points = 0
        details = {}
        gates_passed = 0
        
        # Gate 1: Market Hours (25 points, binary)
        if is_market_hours:
            points += 25
            gates_passed += 1
            details['market'] = {'status': 'OPEN', 'points': 25}
        else:
            details['market'] = {'status': 'CLOSED', 'points': 0}
        
        # Gate 2: IV Rank (0-25 points, scaled)
        iv_threshold = self._params.get('iv_threshold', 0.25) * 100
        if iv_rank >= iv_threshold:
            iv_points = 25
            gates_passed += 1
        else:
            # Partial credit: how close to threshold
            iv_points = int((iv_rank / iv_threshold) * 25)
        points += iv_points
        details['iv_rank'] = {
            'value': iv_rank,
            'threshold': iv_threshold,
            'points': iv_points,
            'passed': iv_rank >= iv_threshold,
        }
        
        # Gate 3: Price Dip (0-25 points, scaled)
        trigger = abs(self._params.get('price_drop_trigger', -0.005))
        actual_drop = abs(min(price_change_pct, 0))
        if actual_drop >= trigger:
            dip_points = 25
            gates_passed += 1
        else:
            # Partial credit: how close to trigger
            dip_points = int((actual_drop / trigger) * 25) if trigger > 0 else 0
        points += dip_points
        details['price_dip'] = {
            'change': price_change_pct,
            'trigger': -trigger,
            'points': dip_points,
            'passed': actual_drop >= trigger,
        }
        
        # Gate 4: No Blackout (15 points, binary)
        if not has_event_blackout:
            points += 15
            gates_passed += 1
            details['blackout'] = {'status': 'CLEAR', 'points': 15}
        else:
            details['blackout'] = {'status': 'BLOCKED', 'points': 0}
        
        # Gate 5: Fear/Greed (0-10 points, scaled)
        # Lower = more fear = better for puts
        if fear_greed <= 25:
            fg_points = 10
        elif fear_greed <= 45:
            fg_points = 7
        elif fear_greed <= 55:
            fg_points = 5
        elif fear_greed <= 75:
            fg_points = 2
        else:
            fg_points = 0
        points += fg_points
        details['sentiment'] = {
            'fear_greed': fear_greed,
            'points': fg_points,
        }
        
        # CRITICAL: If market is closed, cap temperature
        # Can't trade when market is closed, so never show HOT/WARM
        if not is_market_hours:
            points = min(points, 50)  # Cap at 50% = COOL max
        
        # Determine level
        level = 'COLD'
        for (low, high), lvl in self.LEVELS.items():
            if low <= points < high:
                level = lvl
                break
        
        return WarmthStatus(
            temperature=min(points, 100),
            level=level,
            gates_passed=gates_passed,
            total_gates=4,  # Market, IV, Dip, Blackout
            details=details,
        )
    
    async def check_and_alert(self, status: WarmthStatus) -> bool:
        """
        Check if we should send an alert.
        
        Alerts when:
        1. Level increases (COLD→COOL, etc.)
        2. HOT reached (trade signal!)
        3. Level drops by 2+ levels
        
        Returns True if alert was sent.
        """
        should_alert = False
        alert_type = None
        
        levels_order = ['COLD', 'COOL', 'WARM', 'HOT']
        current_idx = levels_order.index(status.level)
        
        if self._last_level:
            last_idx = levels_order.index(self._last_level)
            
            # Level increased
            if current_idx > last_idx:
                should_alert = True
                alert_type = 'warming'
            
            # HOT reached!
            if status.level == 'HOT' and self._last_level != 'HOT':
                should_alert = True
                alert_type = 'hot'
            
            # Significant drop
            if last_idx - current_idx >= 2:
                should_alert = True
                alert_type = 'cooling'
        
        else:
            # First check - alert if WARM or HOT
            if status.level in ['WARM', 'HOT']:
                should_alert = True
                alert_type = 'warming' if status.level == 'WARM' else 'hot'
        
        # Rate limit: max 1 alert per 15 minutes (except HOT)
        if should_alert and alert_type != 'hot':
            if self._last_alert_time:
                if datetime.now() - self._last_alert_time < timedelta(minutes=15):
                    should_alert = False
        
        if should_alert:
            await self._send_alert(status, alert_type)
            self._last_alert_time = datetime.now()
        
        self._last_level = status.level
        return should_alert
    
    async def _send_alert(self, status: WarmthStatus, alert_type: str) -> None:
        """Send Telegram alert via tiered system."""
        if not self.bot:
            return

        priority = 2
        if alert_type == 'hot':
            priority = 1
            msg = (
                f"🚨 <b>HOT SIGNAL: CONDITIONS MET</b>\n\n"
                f"🌡️ Temperature: <b>{status.temperature}%</b>\n"
                f"✅ Gates: {status.gates_passed}/{status.total_gates} passed\n\n"
                f"Check Robinhood/Alpaca for IBIT/BITO entries."
            )
        elif alert_type == 'warming':
            msg = (
                f"🌡️ <b>CONDITION WARMING: {status.level}</b>\n\n"
                f"Temperature: {status.temperature}%\n"
                f"Gates: {status.gates_passed}/{status.total_gates} passed\n"
                f"Conditions are approaching trigger levels."
            )
        else:  # cooling
            priority = 2
            msg = (
                f"🌊 <b>CONDITION COOLING</b>\n\n"
                f"Temperature: {status.temperature}% ({status.level})\n"
                f"Market conditions have drifted away."
            )
        
        # Use send_alert if available for tiering and deduplication
        if hasattr(self.bot, 'send_tiered_message'):
            await self.bot.send_tiered_message(msg.strip(), priority=priority, key=f"warmth_{alert_type}", rate_limit_mins=30)
        else:
            await self.bot.send_message(msg.strip())
    
    def format_status(self, status: WarmthStatus) -> str:
        """Format warmth status for display."""
        # Temperature bar
        filled = status.temperature // 10
        bar = '#' * filled + '.' * (10 - filled)
        
        # Level indicator
        level_emoji = {
            'COLD': '[___]',
            'COOL': '[__~]',
            'WARM': '[_~~]',
            'HOT': '[!!!]',
        }
        
        lines = [
            "=" * 40,
            "SIGNAL WARMTH MONITOR",
            "=" * 40,
            "",
            f"Temperature: [{bar}] {status.temperature}%",
            f"Level: {level_emoji.get(status.level, '[?]')} {status.level}",
            f"Gates: {status.gates_passed}/{status.total_gates} passed",
            "",
            "Breakdown:",
        ]
        
        for gate, data in status.details.items():
            passed = data.get('passed', data.get('status') in ['OPEN', 'CLEAR'])
            marker = "[+]" if passed else "[ ]"
            points = data.get('points', 0)
            lines.append(f"  {marker} {gate}: +{points} pts")
        
        lines.extend(["", "=" * 40])
        
        return "\n".join(lines)
    
    def format_telegram(self, status: WarmthStatus) -> str:
        """Format for Telegram notification."""
        level_emoji = {
            'COLD': '[___]',
            'COOL': '[__~]', 
            'WARM': '[_~~]',
            'HOT': '[!!!]',
        }
        
        filled = status.temperature // 10
        bar = '#' * filled + '.' * (10 - filled)
        
        msg = f"""
ARGUS STATUS

{level_emoji.get(status.level)} {status.level}
[{bar}] {status.temperature}%

Gates: {status.gates_passed}/{status.total_gates}
"""
        
        if status.level == 'HOT':
            msg += "\nCONDITIONS MET - CHECK FOR TRADE!"
        elif status.level == 'WARM':
            msg += "\nGetting close - stay alert."
        
        return msg.strip()


async def test_warmth():
    """Test warmth monitor."""
    monitor = WarmthMonitor()
    
    # Test scenarios
    scenarios = [
        {"name": "Cold (market closed)", "iv": 20, "price": 0.01, "mkt": False, "blackout": False, "fg": 50},
        {"name": "Cool (some conditions)", "iv": 30, "price": -0.002, "mkt": True, "blackout": False, "fg": 40},
        {"name": "Warm (close)", "iv": 50, "price": -0.004, "mkt": True, "blackout": False, "fg": 30},
        {"name": "Hot (all pass)", "iv": 65, "price": -0.015, "mkt": True, "blackout": False, "fg": 20},
    ]
    
    for s in scenarios:
        print()
        print(f"Scenario: {s['name']}")
        status = monitor.calculate_warmth(
            iv_rank=s['iv'],
            price_change_pct=s['price'],
            is_market_hours=s['mkt'],
            has_event_blackout=s['blackout'],
            fear_greed=s['fg'],
        )
        print(monitor.format_status(status))


if __name__ == "__main__":
    asyncio.run(test_warmth())
