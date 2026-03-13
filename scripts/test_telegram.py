"""
Test Telegram Messages
======================

Sends sample messages to verify Telegram is working.
Run: python scripts/test_telegram.py
"""

import pytest

pytest.skip("Manual integration script (not for automated pytest runs).", allow_module_level=True)

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts.telegram_bot import TelegramBot
from src.core.config import load_all_config


async def test_telegram():
    """Send sample Telegram messages."""
    print("=" * 50)
    print("TELEGRAM TEST")
    print("=" * 50)
    
    # Load config
    config = load_all_config("config")
    secrets = config.get('secrets', {})
    
    bot_token = secrets.get('telegram', {}).get('bot_token')
    chat_id = secrets.get('telegram', {}).get('chat_id')
    
    if not bot_token or not chat_id or bot_token.startswith('PASTE_'):
        print("[!] Telegram not configured in secrets.yaml")
        print("    Add your bot_token and chat_id")
        return
    
    print(f"Bot Token: {bot_token[:10]}...{bot_token[-5:]}")
    print(f"Chat ID: {chat_id}")
    print()
    
    # Create bot
    bot = TelegramBot(bot_token=bot_token, chat_id=chat_id)
    
    # Test connection
    print("Testing connection...")
    if not await bot.test_connection():
        print("[!] Connection failed")
        return
    
    print("[OK] Connected to Telegram!")
    print()
    
    # Send sample messages
    print("Sending sample messages...")
    print()
    
    # 1. System status
    print("  [1/4] System Status...")
    await bot.send_system_status('online', 'Test - Argus verification')
    await asyncio.sleep(1)
    
    # 2. Paper trade opened
    print("  [2/4] Paper Trade Opened...")
    paper_open_msg = """
📝 *PAPER TRADE OPENED*

Trade #TEST
$45/$40 Put Spread
Qty: 2 contracts
Credit: $0.85

IV Rank: 65%
PoP: 78%

_This is a TEST message._
"""
    await bot.send_message(paper_open_msg.strip(), parse_mode="Markdown")
    await asyncio.sleep(1)
    
    # 3. Paper trade closed
    print("  [3/4] Paper Trade Closed...")
    paper_close_msg = """
✅ *PAPER TRADE CLOSED*

Trade #TEST
$45/$40 Put Spread

Entry: $0.85
Exit: $0.42
Profit: +$86.00 (+50.6%)

Reason: profit target
Duration: 5 days

_This is a TEST message._
"""
    await bot.send_message(paper_close_msg.strip(), parse_mode="Markdown")
    await asyncio.sleep(1)
    
    # 4. IBIT Signal
    print("  [4/4] IBIT Signal...")
    ibit_signal = """
🎯 *IBIT OPTIONS OPPORTUNITY*

IBIT Price: $52.35 (-3.2%)
BTC IV: 75%
IV Rank: 68%

Suggested Trade:
SELL: $45 Put
BUY: $40 Put
Exp: 2 weeks

Credit: ~$0.85
Max Risk: $4.15
PoP: 78%

_Check Robinhood for current prices._
_This is a TEST message._
"""
    await bot.send_message(ibit_signal.strip(), parse_mode="Markdown")
    
    print()
    print("=" * 50)
    print("[OK] All test messages sent!")
    print("     Check your Telegram for 4 messages.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_telegram())
