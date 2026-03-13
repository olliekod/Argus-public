
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.alerts.telegram_bot import TelegramBot

async def test_telegram_method_names():
    # Mock dependencies
    bot = TelegramBot(bot_token="test_token", chat_id="test_chat")
    bot.send_message = AsyncMock(return_value=True)
    
    print("Testing send_tiered_message (generic signature)...")
    success = await bot.send_tiered_message("Test message", priority=2, key="test_key")
    assert success is True
    bot.send_message.assert_called()
    print("✓ send_tiered_message works.")

    print("Testing send_alert (structured signature)...")
    # Structured alert requires tier, alert_type, title, details
    success = await bot.send_alert(
        tier=1,
        alert_type="options_iv",
        title="Test Alert",
        details={"Price": "100"}
    )
    assert success is True
    print("✓ send_alert works.")

    print("All method name tests passed!")

if __name__ == "__main__":
    asyncio.run(test_telegram_method_names())
