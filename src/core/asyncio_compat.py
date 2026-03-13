"""
Asyncio compatibility helpers.

Ensures a default event loop exists for Python 3.14+ on Windows where
asyncio.get_event_loop() no longer auto-creates a loop for the main thread.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine


def ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Ensure a current event loop exists for the main thread."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass

    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine safely from sync code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return asyncio.create_task(coro)
