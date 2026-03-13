"""Minimal ib_insync connection test (Python 3.14+ event loop safe)."""

from ib_insync import IB, util

from src.core.asyncio_compat import ensure_event_loop, run_sync


async def main() -> None:
    ensure_event_loop()
    util.startLoop()
    ib = IB()
    ib.connect("127.0.0.1", 4002, clientId=42)
    print("Connected", ib.isConnected())
    print("ServerTime", ib.reqCurrentTime())
    ib.disconnect()


if __name__ == "__main__":
    run_sync(main())
