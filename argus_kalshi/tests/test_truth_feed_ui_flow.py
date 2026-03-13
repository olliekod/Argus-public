"""
Hypothesis test: under event-loop load, StateAggregator (or UI consumer) is starved
and does not receive btc/eth/sol price updates in time.

Problem: User reports BTC/ETH/SOL header prices update "once every three minutes"
despite fast Coinbase/OKX feed. Probs 0, OB 0, Ask/Edge stale.

Hypothesis: The consumer (StateAggregator when UI is separate) competes with many
other tasks on the same event loop; it gets too few time slices to drain the
price queues, so _prices lag.

Test design:
- Publish a sequence of BtcMidPrice messages at ~5 Hz with known prices (last = 2400).
- Run StateAggregator in the same loop.
- (A) Without load: assert aggregator._prices["BTC"] == 2400 after run → baseline.
- (B) With load: many tasks simulating farm activity; assert aggregator._prices["BTC"] == 2400.
  If (B) fails (aggregator has 0 or old value), hypothesis is confirmed (consumer starved).
  After fix (drain prices first / give aggregator priority), (B) should pass.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from argus_kalshi.bus import Bus
from argus_kalshi.models import (
    BtcMidPrice,
    FairProbability,
    MarketMetadata,
    OrderbookState,
)
from argus_kalshi.ipc import StateAggregator


# Duration for truth feed to publish and aggregator to run (seconds).
_RUN_S = 3.0
# Publish interval (seconds); ~5 Hz.
_TICK_INTERVAL = 0.2
# Last price we publish (so we can assert aggregator saw it).
_LAST_PRICE = 2400.0
# Number of "load" tasks that simulate farm (competing for event loop).
_LOAD_TASK_COUNT = 400


async def _truth_feed_task(bus: Bus, stop_evt: asyncio.Event) -> float:
    """Publish BtcMidPrice to btc.mid_price every _TICK_INTERVAL; return last price published."""
    price = 1000.0
    last_published = price
    while not stop_evt.is_set():
        await bus.publish("btc.mid_price", BtcMidPrice(price=price, timestamp=time.time(), source="test", asset="BTC"))
        last_published = price
        price += 100.0
        if price > _LAST_PRICE:
            price = _LAST_PRICE
        await asyncio.sleep(_TICK_INTERVAL)
    return last_published


async def _load_task(stop_evt: asyncio.Event) -> None:
    """Simulate work competing for the event loop (like many farm bots / dispatcher)."""
    n = 0
    while not stop_evt.is_set():
        for _ in range(50):
            await asyncio.sleep(0)
        n += 1
        if n % 10 == 0:
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_truth_feed_to_aggregator_baseline_no_load():
    """
    Baseline: with no competing load, StateAggregator should receive every price tick
    and end with _prices["BTC"] == last published value.
    """
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id=None)
    await agg.start()

    stop = asyncio.Event()
    feed_task = asyncio.create_task(_truth_feed_task(bus, stop))
    try:
        await asyncio.sleep(_RUN_S)
    finally:
        stop.set()
        await feed_task

    agg.stop()
    # Give loop one more pass to let aggregator drain.
    await asyncio.sleep(0.1)

    assert agg._prices["BTC"] == _LAST_PRICE, (
        f"Baseline (no load): expected aggregator to have latest price {_LAST_PRICE}, got {agg._prices['BTC']}"
    )


@pytest.mark.asyncio
async def test_truth_feed_to_aggregator_under_load_starved():
    """
    Hypothesis test: under event-loop load (many competing tasks), StateAggregator
    may be starved and not receive the latest price.

    If this test FAILS (aggregator has old or zero price), the hypothesis is
    confirmed: consumer is starved. A fix (drain prices first, or give aggregator
    more priority) should make this test PASS (aggregator gets _LAST_PRICE).
    """
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id=None)
    await agg.start()

    stop = asyncio.Event()
    feed_task = asyncio.create_task(_truth_feed_task(bus, stop))
    load_tasks = [asyncio.create_task(_load_task(stop)) for _ in range(_LOAD_TASK_COUNT)]

    try:
        await asyncio.sleep(_RUN_S)
    finally:
        stop.set()
        await feed_task
        await asyncio.gather(*load_tasks)

    agg.stop()
    await asyncio.sleep(0.1)

    # Hypothesis: under load, aggregator might not reach _LAST_PRICE (starved).
    # We assert that it MUST reach _LAST_PRICE; if the implementation is correct
    # (prices drained with sufficient priority), this passes.
    assert agg._prices["BTC"] == _LAST_PRICE, (
        f"Under load ({_LOAD_TASK_COUNT} tasks): expected aggregator to have latest price {_LAST_PRICE}, "
        f"got {agg._prices['BTC']}. Hypothesis CONFIRMED: UI consumer is starved by event-loop load."
    )


@pytest.mark.asyncio
async def test_truth_feed_to_aggregator_flooded_ob_prob_starves_prices():
    """
    Hypothesis: when OB and prob queues are flooded (many tickers, like production),
    the aggregator drains ob/prob/meta/outcomes/fills/orders before prices in each
    iteration and may not drain btc.mid_price often enough, so _prices["BTC"] lags.

    We flood kalshi.orderbook and kalshi.fair_prob with many messages, then publish
    a short sequence of prices. If aggregator ends with old or zero BTC price,
    hypothesis confirmed.
    """
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id=None)
    await agg.start()

    # Flood OB and prob (simulate many markets): 150 tickers × 2 = 300 ob + 300 prob.
    n_tickers = 150
    for i in range(n_tickers):
        t = f"TICKER-{i:04d}"
        await bus.publish(
            "kalshi.orderbook",
            OrderbookState(
                market_ticker=t,
                best_yes_bid_cents=45,
                best_no_bid_cents=55,
                implied_yes_ask_cents=55,
                implied_no_ask_cents=45,
                seq=1,
                valid=True,
            ),
        )
        await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=t, p_yes=0.5 + (i % 10) / 100.0))

    stop = asyncio.Event()
    feed_task = asyncio.create_task(_truth_feed_task(bus, stop))
    try:
        await asyncio.sleep(_RUN_S)
    finally:
        stop.set()
        await feed_task

    agg.stop()
    await asyncio.sleep(0.2)

    assert agg._prices["BTC"] == _LAST_PRICE, (
        f"With flooded OB/prob ({n_tickers} tickers): expected aggregator to have latest price {_LAST_PRICE}, "
        f"got {agg._prices['BTC']}. Hypothesis CONFIRMED: aggregator drains ob/prob before prices and starves."
    )


def _sync_work_chunk(n: int) -> None:
    """Simulate dispatcher doing N sync evaluations (no I/O)."""
    x = 0
    for _ in range(n):
        x += 1
    return None


@pytest.mark.asyncio
async def test_truth_feed_to_aggregator_dispatcher_style_load():
    """
    Hypothesis: one task that does large chunks of sync work (like FarmDispatcher
    calling evaluate_sync on 64 bots) then yields once — so the event loop is
    blocked for a few ms and the aggregator runs rarely. Under that load, price
    updates may not be consumed in time.

    Simulate: one "dispatcher" task that in a loop does 5000 sync steps then
    await asyncio.sleep(0), for the whole run. Plus truth feed publishing.
    """
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id=None)
    await agg.start()

    stop = asyncio.Event()

    async def dispatcher_style_load() -> None:
        while not stop.is_set():
            _sync_work_chunk(100_000)  # Hold event loop for many ms (like 64 bot evals × many ticks)
            await asyncio.sleep(0)

    feed_task = asyncio.create_task(_truth_feed_task(bus, stop))
    dispatcher_task = asyncio.create_task(dispatcher_style_load())

    try:
        await asyncio.sleep(_RUN_S)
    finally:
        stop.set()
        await feed_task
        await dispatcher_task

    agg.stop()
    await asyncio.sleep(0.1)

    assert agg._prices["BTC"] == _LAST_PRICE, (
        "Under dispatcher-style load (sync chunks + yield): expected aggregator to have latest price "
        f"{_LAST_PRICE}, got {agg._prices['BTC']}. Hypothesis CONFIRMED: aggregator starved when loop is busy."
    )


@pytest.mark.asyncio
async def test_drain_order_matters_prices_before_ob_prob():
    """
    Prove that drain order matters: if we have limited iterations per cycle and
    we drain ob/prob first, we may never reach the price queue when those are flooded.

    This test uses the real bus and a minimal "consumer" that drains in BAD order
    (ob/prob first, then price) with a cap of 10 iterations. We flood ob, then
    publish one price. After one cycle, bad-order consumer has not seen the price;
    good-order (price first) would have seen it.
    """
    bus = Bus()
    # Subscribe and flood ob
    q_ob = await bus.subscribe("kalshi.orderbook")
    q_btc = await bus.subscribe("btc.mid_price")
    for i in range(25):
        await bus.publish(
            "kalshi.orderbook",
            OrderbookState(
                market_ticker=f"T{i}",
                best_yes_bid_cents=50,
                best_no_bid_cents=50,
                implied_yes_ask_cents=50,
                implied_no_ask_cents=50,
                seq=1,
                valid=True,
            ),
        )
    # One price tick
    await bus.publish("btc.mid_price", BtcMidPrice(price=9999.0, timestamp=time.time(), source="test", asset="BTC"))

    # BAD order: drain ob first (25 msgs), then btc. With 10 iterations we drain 10 ob, 0 btc.
    prices_seen = []

    def bad_order_drain(max_iter: int) -> None:
        for _ in range(max_iter):
            if not q_ob.empty():
                q_ob.get_nowait()
            elif not q_btc.empty():
                msg = q_btc.get_nowait()
                if hasattr(msg, "price"):
                    prices_seen.append(msg.price)

    bad_order_drain(10)
    assert 9999.0 not in prices_seen, "Bad order: with 10 iters we should not have reached btc yet (ob had 25 msgs)."

    # GOOD order: drain btc first, then ob. One iteration gets the price.
    prices_seen_good = []

    def good_order_drain(max_iter: int) -> None:
        for _ in range(max_iter):
            if not q_btc.empty():
                msg = q_btc.get_nowait()
                if hasattr(msg, "price"):
                    prices_seen_good.append(msg.price)
            if not q_ob.empty():
                q_ob.get_nowait()

    good_order_drain(10)
    assert 9999.0 in prices_seen_good, "Good order: draining btc first should have got the price."


# --- Issue 3: Probs 0 / OB 0 ---


@pytest.mark.asyncio
async def test_aggregator_prob_and_ob_populate_states_and_snapshot():
    """After publishing FairProbability and OrderbookState, aggregator _states and get_snapshot() have them."""
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id=None)
    await agg.start()

    ticker = "KXBTC-15M-TEST"
    await bus.publish(
        "kalshi.orderbook",
        OrderbookState(
            market_ticker=ticker,
            best_yes_bid_cents=45,
            best_no_bid_cents=55,
            implied_yes_ask_cents=55,
            implied_no_ask_cents=45,
            seq=1,
            valid=True,
        ),
    )
    await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=ticker, p_yes=0.62))

    for _ in range(20):
        await asyncio.sleep(0.01)
        if ticker in agg._states and agg._states[ticker].get("ob_valid") and agg._states[ticker].get("p_yes") != 0.5:
            break

    agg.stop()
    await asyncio.sleep(0.05)

    assert ticker in agg._states
    assert agg._states[ticker]["ob_valid"] is True
    assert agg._states[ticker]["yes_ask"] == 55
    assert agg._states[ticker]["no_ask"] == 45
    assert agg._states[ticker]["p_yes"] == 0.62

    snap = agg.get_snapshot()
    assert ticker in snap["states"]
    assert snap["states"][ticker]["ob_valid"] is True
    assert snap["states"][ticker]["p_yes"] == 0.62


@pytest.mark.asyncio
async def test_remote_ui_counts_derived_from_snapshot():
    """Snapshot with prob/ob data yields non-zero Probs and OB counts when applied to UI."""
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id=None)
    await agg.start()

    for i in range(3):
        t = f"T{i}"
        await bus.publish(
            "kalshi.orderbook",
            OrderbookState(
                market_ticker=t,
                best_yes_bid_cents=50,
                best_no_bid_cents=50,
                implied_yes_ask_cents=50,
                implied_no_ask_cents=50,
                seq=1,
                valid=True,
            ),
        )
        await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=t, p_yes=0.5 + i * 0.1))

    await asyncio.sleep(0.1)
    agg.stop()
    snap = agg.get_snapshot()
    states = snap.get("states") or {}
    prob_count = sum(1 for s in states.values() if s.get("p_yes", 0.5) != 0.5)
    ob_count = sum(1 for s in states.values() if s.get("ob_valid"))
    assert prob_count >= 2
    assert ob_count >= 2


# --- Issue 4: Ask and Edge ---


@pytest.mark.asyncio
async def test_aggregator_ask_edge_from_orderbook_and_fair_prob():
    """OrderbookState + FairProbability → state has yes_ask, no_ask, p_yes; Ask/Edge derivable."""
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id=None)
    await agg.start()

    ticker = "KXBTC-TEST"
    await bus.publish(
        "kalshi.orderbook",
        OrderbookState(
            market_ticker=ticker,
            best_yes_bid_cents=40,
            best_no_bid_cents=58,
            implied_yes_ask_cents=58,
            implied_no_ask_cents=42,
            seq=1,
            valid=True,
        ),
    )
    await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=ticker, p_yes=0.55))

    await asyncio.sleep(0.1)
    agg.stop()

    assert ticker in agg._states
    s = agg._states[ticker]
    assert s["yes_ask"] == 58
    assert s["no_ask"] == 42
    assert s["p_yes"] == 0.55
    assert s["ob_valid"] is True
    # Edge YES = p_yes - yes_ask/100 = 0.55 - 0.58 = -0.03; Edge NO = (1-p_yes) - no_ask/100 = 0.45 - 0.42 = 0.03
    best_edge_yes = 0.55 - (s["yes_ask"] / 100.0)
    best_edge_no = (1 - 0.55) - (s["no_ask"] / 100.0)
    assert best_edge_no > best_edge_yes


# --- Issue 5: 15m market row ---


def _make_meta_15m(ticker: str, asset: str = "BTC", exp_seconds_from_now: float = 600.0) -> MarketMetadata:
    from datetime import datetime, timezone, timedelta
    exp = datetime.now(timezone.utc) + timedelta(seconds=exp_seconds_from_now)
    return MarketMetadata(
        market_ticker=ticker,
        strike_price=68_000.0,
        strike_floor=None,
        strike_cap=None,
        settlement_time_iso=exp.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        last_trade_time_iso=exp.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        asset=asset,
        is_range=False,
        window_minutes=15,
    )


@pytest.mark.asyncio
async def test_aggregator_metadata_15min_populates_state_and_best_per_type():
    """MarketMetadata for 15min BTC ticker → aggregator state has window_min 15; UI _best_per_type would have entry."""
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id=None)
    await agg.start()

    ticker = "KXBTC-15M-001"
    meta = _make_meta_15m(ticker)
    await bus.publish("kalshi.market_metadata", meta)

    await asyncio.sleep(0.1)
    agg.stop()
    snap = agg.get_snapshot()

    assert ticker in snap.get("metadata", {})
    assert ticker in snap.get("states", {})
    st = snap["states"][ticker]
    assert st.get("window_min") == 15
    assert st.get("is_range") is False


@pytest.mark.asyncio
async def test_snapshot_with_15min_market_makes_row_not_empty():
    """Apply snapshot with 15min BTC ticker to TerminalVisualizer; _best_per_type has BTC 15min."""
    from argus_kalshi.terminal_ui import TerminalVisualizer

    bus = Bus()
    vis = TerminalVisualizer(bus, metadata={}, dry_run=True, primary_bot_id=None, leaderboard_only=False)
    ticker = "KXBTC-15M-001"
    meta = _make_meta_15m(ticker)
    meta_dict = {
        "market_ticker": ticker,
        "strike_price": 68_000.0,
        "strike_floor": None,
        "strike_cap": None,
        "settlement_time_iso": meta.settlement_time_iso,
        "last_trade_time_iso": meta.last_trade_time_iso,
        "asset": "BTC",
        "is_range": False,
        "window_minutes": 15,
    }
    snapshot = {
        "primary_bot_id": None,
        "prices": {"BTC": 68_000.0, "ETH": 2000.0, "SOL": 86.0},
        "states": {
            ticker: {
                "ticker": ticker,
                "asset": "BTC",
                "p_yes": 0.5,
                "p_yes_hist": [0.5],
                "yes_ask": 50,
                "no_ask": 50,
                "yes_bid": 45,
                "no_bid": 55,
                "ob_valid": True,
                "ob_had_valid": True,
                "exp_ts": time.time() + 600.0,
                "window_min": 15,
                "is_range": False,
                "strike": 68_000.0,
            }
        },
        "metadata": {ticker: meta_dict},
    }
    vis.update_from_snapshot(snapshot)
    best = vis._best_per_type()
    assert best["BTC"]["15min"] is not None
    assert best["BTC"]["15min"].ticker == ticker


# --- Live feed simulation: IPC drain_for_snapshot under load (separate UI) ---
#
# Why drain_for_snapshot() fixes the slow UI: With 7488 bots, the event loop is dominated
# by FarmDispatcher (evaluate_sync × many bots per tick). The StateAggregator task gets
# few time slices, so when ipc_server_loop sends a snapshot every 0.2s it was sending
# stale _prices and _states. Now the IPC loop calls drain_for_snapshot() before each
# get_snapshot(), so it pulls control (ws/rtt/balance), prices (btc/eth/sol), and a
# bounded amount of ob/prob from the same queues. The UI gets fresh data every 0.2s
# regardless of whether the aggregator task ran.

_LIVE_RUN_S = 2.0
_LIVE_TICK_INTERVAL = 0.05   # 20 Hz prices → 40 ticks in 2s
_LIVE_IPC_INTERVAL = 0.2     # match IPC_SNAPSHOT_INTERVAL
# Reach target in 2s: 40 ticks, start 1000, so increment (5000-1000)/40 = 100
_LIVE_LAST_PRICE = 5000.0
_LIVE_N_TICKERS = 80         # flood ob/prob


async def _live_feed_prices(bus: Bus, stop: asyncio.Event) -> float:
    """Publish btc/eth/sol at 20 Hz; return last BTC price published."""
    price = 1000.0
    last = price
    while not stop.is_set():
        await bus.publish(
            "btc.mid_price",
            BtcMidPrice(price=price, timestamp=time.time(), source="test", asset="BTC"),
        )
        await bus.publish(
            "eth.mid_price",
            BtcMidPrice(price=price * 0.03, timestamp=time.time(), source="test", asset="ETH"),
        )
        await bus.publish(
            "sol.mid_price",
            BtcMidPrice(price=price * 0.012, timestamp=time.time(), source="test", asset="SOL"),
        )
        last = price
        price += (_LIVE_LAST_PRICE - 1000.0) / 40.0  # reach _LIVE_LAST_PRICE in 40 ticks (2s)
        if price >= _LIVE_LAST_PRICE:
            price = _LIVE_LAST_PRICE
        await asyncio.sleep(_LIVE_TICK_INTERVAL)
    return last


async def _live_feed_ob_prob(bus: Bus, stop: asyncio.Event) -> None:
    """Flood ob + prob for many tickers so aggregator task would be busy draining them."""
    i = 0
    while not stop.is_set():
        for j in range(_LIVE_N_TICKERS):
            t = f"T{i}-{j}"
            await bus.publish(
                "kalshi.orderbook",
                OrderbookState(
                    market_ticker=t,
                    best_yes_bid_cents=45,
                    best_no_bid_cents=55,
                    implied_yes_ask_cents=55,
                    implied_no_ask_cents=45,
                    seq=1 + i,
                    valid=True,
                ),
            )
            await bus.publish("kalshi.fair_prob", FairProbability(market_ticker=t, p_yes=0.5 + (j % 10) / 100.0))
        i += 1
        await asyncio.sleep(0.08)


def _sync_work_chunk_live(n: int) -> None:
    """Heavy sync work to starve aggregator (simulate farm dispatcher)."""
    x = 0
    for _ in range(n):
        x += 1
    return None


@pytest.mark.asyncio
async def test_ipc_drain_under_live_feed_snapshots_fresh_and_fast() -> None:
    """
    Simulate separate UI: live feed (prices 20 Hz + ob/prob flood), heavy dispatcher load,
    and only the IPC send loop running every 0.2s calling drain_for_snapshot() then get_snapshot().

    Asserts:
    - At least one snapshot has the latest published BTC price (drain_for_snapshot works).
    - We get at least 5 snapshots in 2s (IPC loop runs ~every 0.2s).
    - Snapshots see non-zero ob/prob counts (drain_for_snapshot pulls ob/prob too).

    With -s, prints timing and counts so you can see "speed of literally everything".
    """
    bus = Bus()
    agg = StateAggregator(bus, primary_bot_id=None)
    await agg.start()
    # Stop aggregator task so only IPC loop drains (simulates separate UI where aggregator is starved).
    if agg._task:
        agg._task.cancel()
        try:
            await agg._task
        except asyncio.CancelledError:
            pass
    agg._task = None

    stop = asyncio.Event()
    snapshots: list[dict] = []
    snapshot_times: list[float] = []

    async def ipc_loop_sim() -> None:
        """Simulate ipc_server_loop: every 0.2s drain then snapshot."""
        while not stop.is_set():
            agg.drain_for_snapshot(max_ob=40, max_prob=40)
            snap = agg.get_snapshot()
            snapshots.append(snap)
            snapshot_times.append(time.monotonic())
            await asyncio.sleep(_LIVE_IPC_INTERVAL)

    async def dispatcher_load() -> None:
        while not stop.is_set():
            _sync_work_chunk_live(80_000)
            await asyncio.sleep(0)

    feed_prices = asyncio.create_task(_live_feed_prices(bus, stop))
    feed_ob_prob = asyncio.create_task(_live_feed_ob_prob(bus, stop))
    ipc_task = asyncio.create_task(ipc_loop_sim())
    dispatcher = asyncio.create_task(dispatcher_load())

    try:
        await asyncio.sleep(_LIVE_RUN_S)
    finally:
        stop.set()
        await feed_prices
        await feed_ob_prob
        await ipc_task
        await dispatcher

    agg.stop()
    await asyncio.sleep(0.05)

    # --- Assertions ---
    assert len(snapshots) >= 5, (
        f"Expected at least 5 snapshots in {_LIVE_RUN_S}s (IPC every {_LIVE_IPC_INTERVAL}s), got {len(snapshots)}"
    )
    btc_in_snapshots = [s["prices"].get("BTC") for s in snapshots]
    max_btc = max(btc_in_snapshots)
    assert max_btc >= _LIVE_LAST_PRICE * 0.85, (
        f"IPC drain_for_snapshot should deliver fresh prices; max BTC in snapshots={max_btc}, target={_LIVE_LAST_PRICE}; "
        f"seen: {btc_in_snapshots[:15]}..."
    )
    # At least one snapshot should have some ob/prob state (we drain 40 ob + 40 prob per cycle)
    states_counts = [len(s.get("states") or {}) for s in snapshots]
    ob_valid_counts = [
        sum(1 for st in (s.get("states") or {}).values() if st.get("ob_valid"))
        for s in snapshots
    ]
    assert max(states_counts) >= 1 and max(ob_valid_counts) >= 1, (
        f"Expected at least one snapshot with ob/prob state; max states={max(states_counts)}, max ob_valid={max(ob_valid_counts)}"
    )

    # Report speed (visible with pytest -s)
    print()
    print("=== Live feed simulation (IPC drain_for_snapshot under load) ===")
    print(f"  Run: {_LIVE_RUN_S}s  |  Price tick: {_LIVE_TICK_INTERVAL}s (~{1/_LIVE_TICK_INTERVAL:.0f} Hz)  |  IPC interval: {_LIVE_IPC_INTERVAL}s")
    print(f"  Snapshots received: {len(snapshots)}  (expected ~{int(_LIVE_RUN_S / _LIVE_IPC_INTERVAL)})")
    max_btc = max(btc_in_snapshots)
    print(f"  BTC in snapshots: min={min(btc_in_snapshots):.0f} max={max_btc:.0f}  (target {_LIVE_LAST_PRICE})")
    if len(snapshot_times) >= 2:
        intervals = [snapshot_times[i + 1] - snapshot_times[i] for i in range(len(snapshot_times) - 1)]
        print(f"  Snapshot intervals (s): min={min(intervals):.3f} max={max(intervals):.3f} avg={sum(intervals)/len(intervals):.3f}")
    print(f"  States per snapshot: min={min(states_counts)} max={max(states_counts)}")
    print(f"  OB-valid per snapshot: min={min(ob_valid_counts)} max={max(ob_valid_counts)}")
    print("  => IPC drain_for_snapshot() keeps snapshots fresh under load.")
    print()
