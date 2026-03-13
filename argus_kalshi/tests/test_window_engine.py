"""
Tests for btc_window_engine.BtcWindowEngine.

Covers:
  - Deterministic ring buffer behavior
  - Irregular tick arrival (gaps, bursts)
  - Second bucket rollover
  - Correct running sum and average
  - Forward-fill behavior
"""

from __future__ import annotations

import asyncio
import math
import pytest

from argus_kalshi.btc_window_engine import BtcWindowEngine
from argus_kalshi.bus import Bus


@pytest.fixture
def bus() -> Bus:
    return Bus()


@pytest.fixture
def engine(bus: Bus) -> BtcWindowEngine:
    return BtcWindowEngine(bus, truth_topic="btc.mid_price")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

async def _tick(engine: BtcWindowEngine, price: float, ts: float) -> None:
    """Send one tick directly (bypassing the bus for deterministic testing)."""
    await engine.on_tick(price, ts)


# ---------------------------------------------------------------------------
#  Basic functionality
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_tick(engine: BtcWindowEngine) -> None:
    """A single tick initialises the buffer with count=1."""
    await _tick(engine, 50000.0, 1000.0)
    assert engine.initialised
    assert engine.count == 1
    assert engine.avg == pytest.approx(50000.0)
    assert engine.sum == pytest.approx(50000.0)


@pytest.mark.asyncio
async def test_two_consecutive_ticks(engine: BtcWindowEngine) -> None:
    """Two ticks one second apart → count=2, correct average."""
    await _tick(engine, 100.0, 1000.0)
    await _tick(engine, 200.0, 1001.0)
    assert engine.count == 2
    assert engine.sum == pytest.approx(300.0)
    assert engine.avg == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_same_second_overwrites(engine: BtcWindowEngine) -> None:
    """Multiple ticks in the same second keep only the last price."""
    await _tick(engine, 100.0, 1000.0)
    await _tick(engine, 200.0, 1000.5)   # same integer second
    await _tick(engine, 300.0, 1000.9)   # same integer second
    assert engine.count == 1
    assert engine.avg == pytest.approx(300.0)


# ---------------------------------------------------------------------------
#  Forward-fill and gap handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gap_forward_fill(engine: BtcWindowEngine) -> None:
    """A 3-second gap forward-fills with the last known price."""
    await _tick(engine, 100.0, 1000.0)
    # Jump 3 seconds ahead.
    await _tick(engine, 110.0, 1003.0)
    # Slots: s0=100, s1=100(fill), s2=100(fill), s3=110
    assert engine.count == 4
    assert engine.sum == pytest.approx(100 + 100 + 100 + 110)
    assert engine.avg == pytest.approx((100 * 3 + 110) / 4)


@pytest.mark.asyncio
async def test_full_window_fill(engine: BtcWindowEngine) -> None:
    """After 60 ticks one second apart, the window is fully populated."""
    for i in range(60):
        await _tick(engine, 100.0 + i, 1000.0 + i)
    assert engine.count == 60
    expected_sum = sum(100.0 + i for i in range(60))
    assert engine.sum == pytest.approx(expected_sum)
    assert engine.avg == pytest.approx(expected_sum / 60)


@pytest.mark.asyncio
async def test_rollover_evicts_oldest(engine: BtcWindowEngine) -> None:
    """Tick 61 evicts the first slot."""
    for i in range(61):
        await _tick(engine, 100.0 + i, 1000.0 + i)

    assert engine.count == 60
    # The oldest value (100.0) should be gone; newest is 160.0.
    expected_sum = sum(100.0 + i for i in range(1, 61))
    assert engine.sum == pytest.approx(expected_sum)


@pytest.mark.asyncio
async def test_large_gap_resets_buffer(engine: BtcWindowEngine) -> None:
    """A gap ≥ 60 seconds effectively resets the buffer."""
    await _tick(engine, 100.0, 1000.0)
    await _tick(engine, 200.0, 1100.0)  # 100s gap ≥ 60
    # The entire buffer should be filled with forward-fill (100.0) then
    # the head slot overwritten with 200.0.
    assert engine.count == 60
    assert engine.avg == pytest.approx((100.0 * 59 + 200.0) / 60)


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_old_tick_ignored(engine: BtcWindowEngine) -> None:
    """A tick with a timestamp older than the window is ignored."""
    await _tick(engine, 100.0, 1060.0)  # initialise at second 1060
    # Now send a very old tick.
    await _tick(engine, 999.0, 900.0)   # 160s in the past
    assert engine.count == 1
    assert engine.avg == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_irregular_tick_cadence(engine: BtcWindowEngine) -> None:
    """Mixed gaps and bursts produce correct averages."""
    # t=0: 100
    await _tick(engine, 100.0, 1000.0)
    # t=1: 101
    await _tick(engine, 101.0, 1001.0)
    # t=5: 105 (3-second gap, forward-fill with 101)
    await _tick(engine, 105.0, 1005.0)
    # t=5.5: 106 (same second, overwrite)
    await _tick(engine, 106.0, 1005.5)

    # Expected slots: [100, 101, 101, 101, 101, 106] = 6 slots
    assert engine.count == 6
    expected = 100 + 101 + 101 + 101 + 101 + 106
    assert engine.sum == pytest.approx(expected)
    assert engine.avg == pytest.approx(expected / 6)


# ---------------------------------------------------------------------------
#  Bus integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publishes_window_state(bus: Bus) -> None:
    """Engine publishes BtcWindowState messages on the bus."""
    engine = BtcWindowEngine(bus, truth_topic="btc.mid_price")
    q = await bus.subscribe("btc.window_state")

    await _tick(engine, 50000.0, 1000.0)

    msg = q.get_nowait()
    assert msg.last_60_avg == pytest.approx(50000.0)
    assert msg.count == 1


@pytest.mark.asyncio
async def test_get_values_chronological(engine: BtcWindowEngine) -> None:
    """get_values() returns prices in chronological order."""
    for i in range(5):
        await _tick(engine, float(i + 1), 1000.0 + i)
    vals = engine.get_values()
    assert vals == [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.asyncio
async def test_constant_price_average(engine: BtcWindowEngine) -> None:
    """Constant price over 60 seconds → average equals the price."""
    for i in range(60):
        await _tick(engine, 42000.0, 1000.0 + i)
    assert engine.avg == pytest.approx(42000.0)
    assert engine.count == 60


@pytest.mark.asyncio
async def test_running_sum_consistency(engine: BtcWindowEngine) -> None:
    """Running sum matches manual sum of get_values()."""
    import random
    random.seed(42)
    base = 50000.0
    for i in range(120):
        price = base + random.uniform(-100, 100)
        await _tick(engine, price, 1000.0 + i)

    vals = engine.get_values()
    assert engine.sum == pytest.approx(sum(vals), rel=1e-6)
    assert engine.avg == pytest.approx(sum(vals) / len(vals), rel=1e-6)
