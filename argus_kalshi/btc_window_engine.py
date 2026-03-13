"""
Rolling 60-second ring-buffer for BTC mid-price (truth feed).

The CF Benchmarks BRTI settlement uses the 60-second simple average of
prices at 1-second resolution immediately prior to the settlement
timestamp.  This module maintains that window in real time.

Design
------
* **Ring buffer** of 60 slots, one per second.
* Each slot stores the *last* price observed during that integer second.
* When the wall-clock second advances, empty (un-ticked) slots are
  forward-filled from the most recent observed price so the average is
  always computed over a full 60-second span, matching BRTI methodology.
* ``BtcWindowState`` messages are published on ``btc.window_state`` after
  each tick or second rollover.

Determinism under irregular ticks
---------------------------------
* Multiple ticks within the same second: only the last one is kept.
* No tick for a given second: forward-fill from the previous second.
* Ticks arriving out of order (old timestamp): silently ignored.

Complexity
----------
* ``on_tick``: O(gap) where *gap* is the number of seconds since the
  last tick — bounded by 60 in the worst case.  Amortised O(1) under
  normal 1-second feed cadence.
* Sum/average retrieval: maintained incrementally — O(1).
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import List, Optional

from .bus import Bus
from .logging_utils import ComponentLogger
from .models import BtcMidPrice, BtcWindowState

log = ComponentLogger("window_engine")

_WINDOW_SIZE = 60  # seconds


class BtcWindowEngine:
    """Maintains a 60-second rolling average of per-asset mid prices."""

    def __init__(self, bus: Bus, truth_topic: str = "btc.mid_price", asset: str = "BTC") -> None:
        self._bus = bus
        self._truth_topic = truth_topic
        self._asset = asset.upper()

        # Ring buffer of 60 slots, indexed by epoch_second % 60.
        self._slots: List[float] = [math.nan] * _WINDOW_SIZE
        # The epoch second that the current head slot represents.
        self._head_second: int = 0
        self._count: int = 0        # number of valid (non-NaN) slots

        self._running_sum: float = 0.0
        self._last_price: float = math.nan
        self._initialised: bool = False

        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        q = await self._bus.subscribe(self._truth_topic)
        self._task = asyncio.create_task(self._consume(q))

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _consume(self, q: asyncio.Queue) -> None:
        """Read BtcMidPrice messages from the bus and process them."""
        try:
            while self._running:
                msg: BtcMidPrice = await q.get()
                if getattr(msg, "asset", self._asset).upper() != self._asset:
                    continue
                await self.on_tick(msg.price, msg.timestamp)
        except asyncio.CancelledError:
            pass

    # -- core logic ----------------------------------------------------------

    async def on_tick(self, price: float, timestamp: float) -> None:
        """Process a single price tick at the given epoch timestamp.

        This is the main entry point — also usable directly in tests
        without the bus.
        """
        tick_second = int(timestamp)

        if not self._initialised:
            self._initialise(price, tick_second)
            await self._publish(timestamp)
            return

        # Ignore ticks older than the oldest second in the window.
        oldest_in_window = self._head_second - _WINDOW_SIZE + 1
        if tick_second < oldest_in_window:
            return

        if tick_second == self._head_second:
            # Same second as head — overwrite with latest price.
            idx = tick_second % _WINDOW_SIZE
            old_val = self._slots[idx]
            self._slots[idx] = price
            if not math.isnan(old_val):
                self._running_sum += price - old_val
            else:
                self._running_sum += price
                self._count += 1
        elif tick_second > self._head_second:
            # Advance the window forward.
            gap = tick_second - self._head_second
            self._advance(gap, price)
        else:
            # tick_second < head_second but within window — late update.
            idx = tick_second % _WINDOW_SIZE
            old_val = self._slots[idx]
            self._slots[idx] = price
            if not math.isnan(old_val):
                self._running_sum += price - old_val
            else:
                self._running_sum += price
                self._count += 1

        self._last_price = price
        await self._publish(timestamp)

    def _initialise(self, price: float, tick_second: int) -> None:
        """Set up the ring buffer from the first tick."""
        self._slots = [math.nan] * _WINDOW_SIZE
        idx = tick_second % _WINDOW_SIZE
        self._slots[idx] = price
        self._head_second = tick_second
        self._count = 1
        self._running_sum = price
        self._last_price = price
        self._initialised = True

    def _advance(self, gap: int, new_price: float) -> None:
        """Advance the ring buffer by *gap* seconds, forward-filling."""
        fill_price = self._last_price

        if gap >= _WINDOW_SIZE:
            # Full reset: fill entire buffer with last known price,
            # then overwrite the final slot with the new tick.
            self._slots = [fill_price] * _WINDOW_SIZE
            self._running_sum = fill_price * _WINDOW_SIZE
            self._count = _WINDOW_SIZE
            self._head_second += gap
            idx = self._head_second % _WINDOW_SIZE
            self._running_sum += new_price - self._slots[idx]
            self._slots[idx] = new_price
            return

        for step in range(1, gap + 1):
            sec = self._head_second + step
            idx = sec % _WINDOW_SIZE
            old_val = self._slots[idx]

            if step < gap:
                write_price = fill_price
            else:
                write_price = new_price

            if math.isnan(old_val):
                self._running_sum += write_price
                self._count += 1
            else:
                self._running_sum += write_price - old_val

            self._slots[idx] = write_price

        self._head_second += gap

    # -- state queries -------------------------------------------------------

    @property
    def avg(self) -> float:
        if self._count == 0:
            return math.nan
        return self._running_sum / self._count

    @property
    def sum(self) -> float:
        return self._running_sum

    @property
    def count(self) -> int:
        return self._count

    @property
    def initialised(self) -> bool:
        return self._initialised

    def get_values(self) -> List[float]:
        """Return the current window values in chronological order."""
        if not self._initialised:
            return []
        result: List[float] = []
        # Walk from oldest to newest second in the window.
        oldest = self._head_second - _WINDOW_SIZE + 1
        for sec in range(oldest, self._head_second + 1):
            idx = sec % _WINDOW_SIZE
            val = self._slots[idx]
            if not math.isnan(val):
                result.append(val)
        return result

    # -- publish -------------------------------------------------------------

    async def _publish(self, timestamp: float) -> None:
        state = BtcWindowState(
            last_60_sum=self._running_sum,
            last_60_avg=self.avg,
            count=self._count,
            timestamp=timestamp,
            asset=self._asset,
        )
        await self._bus.publish(f"{self._asset.lower()}.window_state", state)
