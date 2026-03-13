import time
import warnings

import pytest

from src.alerts.telegram_bot import TelegramBot, _PollingRestart


@pytest.mark.asyncio
async def test_handle_polling_failure_waits_without_stop() -> None:
    bot = TelegramBot(bot_token="token", chat_id="chat")
    bot._compute_backoff = lambda attempt: 0.05
    start = time.monotonic()
    await bot._handle_polling_failure(RuntimeError("boom"), 1)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.04


@pytest.mark.asyncio
async def test_handle_polling_failure_returns_immediately_on_stop() -> None:
    bot = TelegramBot(bot_token="token", chat_id="chat")
    bot._compute_backoff = lambda attempt: 1.0
    bot._polling_stop_event.set()
    start = time.monotonic()
    await bot._handle_polling_failure(None, 1)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_wait_for_polling_signal_no_unawaited_warnings() -> None:
    bot = TelegramBot(bot_token="token", chat_id="chat")
    bot._polling_restart_event.set()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(_PollingRestart):
            await bot._wait_for_polling_signal()
