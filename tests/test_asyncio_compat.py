import asyncio

from src.core.asyncio_compat import ensure_event_loop


def test_ensure_event_loop_creates_loop_when_missing():
    try:
        current_loop = asyncio.get_event_loop()
    except RuntimeError:
        current_loop = None

    asyncio.set_event_loop(None)
    new_loop = ensure_event_loop()
    assert new_loop is not None
    assert asyncio.get_event_loop() is new_loop

    if current_loop is not None:
        asyncio.set_event_loop(current_loop)
    else:
        asyncio.set_event_loop(None)
    if new_loop is not current_loop:
        new_loop.close()
