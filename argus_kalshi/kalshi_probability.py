"""
Settlement probability model for Kalshi BTC strike contracts.

The contract resolves YES if the 60-second simple average of CF Benchmarks
BRTI immediately prior to settlement is **at or above** the strike price K.

Model
-----
We model the BTC mid-price as geometric Brownian motion:

    dS / S = σ dW

(zero drift over the short horizons relevant here — sub-minute to a few
minutes).

The 60-second arithmetic average A = (1/60) Σ S_t is approximated by
treating the integrand as a continuous integral of GBM, which — for
small σ√T — has a distribution well-approximated by a normal.

**Case 1: time_to_settle > 60 s**
    The entire averaging window is in the future.  We approximate:

        E[A] ≈ S_now
        Var[A] ≈ S_now² · σ² · T_avg / 3

    where T_avg = 60 s expressed in the same units as σ (seconds).

    P(YES) = Φ((E[A] - K) / √Var[A])

**Case 2: time_to_settle ≤ 60 s (inside the window)**
    Some seconds have already been observed.  Let:
        n_obs = 60 - τ    (observed seconds, already locked in)
        S_obs = Σ observed prices
        τ      = remaining seconds
    Then:
        A = (S_obs + Σ_{remaining} S_t) / 60
    We need  Σ_{remaining} ≥ 60·K - S_obs  =: R
    Required per-second mean: m_req = R / τ

    P(YES) = Φ((S_now - m_req) / (S_now · σ · √(τ/3)))

    This is derived from the same integral-of-GBM normal approximation
    applied only to the remaining τ seconds.

Volatility estimator
--------------------
Use the last 120 seconds of log-returns from the truth feed.  Robust to
missing seconds (we only use available pairs).

Deviation from spec
-------------------
* The spec's formula for the inside-window case computes ``m_req`` then
  evaluates a diffusion CDF.  We refine this slightly: rather than using
  ``m_req`` as if it were a single-price level, we properly compute the
  probability that the *average* of the remaining τ prices exceeds
  ``m_req``, accounting for the variance reduction from averaging.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .logging_utils import ComponentLogger
from .models import BtcWindowState, FairProbability, MarketMetadata

log = ComponentLogger("probability")

# ---------------------------------------------------------------------------
#  Standard normal CDF (no scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the Abramowitz & Stegun approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
#  Volatility estimator
# ---------------------------------------------------------------------------

def estimate_volatility(prices: List[float], dt: float = 1.0) -> float:
    """Annualised-style σ from log-returns, but in per-second units.

    Returns σ such that the variance of a 1-second log-return is σ².
    Uses only consecutive non-NaN price pairs.
    """
    if len(prices) < 2:
        return 0.0

    log_returns: List[float] = []
    for i in range(1, len(prices)):
        if prices[i] > 0 and prices[i - 1] > 0:
            log_returns.append(math.log(prices[i] / prices[i - 1]))

    n = len(log_returns)
    if n < 2:
        return 0.0

    mean = sum(log_returns) / n
    var = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    return math.sqrt(var / dt)




def estimate_excess_kurtosis(prices: List[float]) -> float:
    """Estimate excess kurtosis of log-returns for heavy-tail adjustment."""
    if len(prices) < 5:
        return 0.0

    log_returns: List[float] = []
    for i in range(1, len(prices)):
        if prices[i] > 0 and prices[i - 1] > 0:
            log_returns.append(math.log(prices[i] / prices[i - 1]))

    n = len(log_returns)
    if n < 4:
        return 0.0

    mean = sum(log_returns) / n
    m2 = sum((r - mean) ** 2 for r in log_returns) / n
    if m2 <= 0:
        return 0.0
    m4 = sum((r - mean) ** 4 for r in log_returns) / n
    kurtosis = m4 / (m2 * m2)
    return max(0.0, kurtosis - 3.0)


# ---------------------------------------------------------------------------
#  HAR-J (Heterogeneous Autoregressive with Jumps) volatility estimator
# ---------------------------------------------------------------------------

# Jump detection threshold: a return is classified as a jump if its
# absolute value exceeds this many standard deviations of the sample.
_JUMP_THRESHOLD_SIGMA = 3.0


def estimate_volatility_harj(
    prices: List[float],
    dt: float = 1.0,
) -> float:
    """Jump-aware short-horizon volatility estimator (HAR-J style).

    1.  Compute log-returns from *prices*.
    2.  Estimate the continuous-component σ by excluding returns that
        exceed ``_JUMP_THRESHOLD_SIGMA`` standard deviations (jumps).
    3.  If jumps were detected, inflate σ using the Bipower Variation
        ratio so the model accounts for the higher realised variance
        without being dominated by outliers.

    Falls back to :func:`estimate_volatility` when there are too few
    data points for jump detection (< 10 prices).

    Returns σ in per-second units (same convention as
    :func:`estimate_volatility`).
    """
    if len(prices) < 10:
        return estimate_volatility(prices, dt)

    log_returns: List[float] = []
    for i in range(1, len(prices)):
        if prices[i] > 0 and prices[i - 1] > 0:
            log_returns.append(math.log(prices[i] / prices[i - 1]))

    n = len(log_returns)
    if n < 5:
        return estimate_volatility(prices, dt)

    # Step 1: initial σ from all returns.
    mean = sum(log_returns) / n
    var_all = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    sigma_all = math.sqrt(var_all) if var_all > 0 else 0.0

    if sigma_all == 0.0:
        return 0.0

    # Step 2: identify jumps and compute continuous-component σ.
    threshold = _JUMP_THRESHOLD_SIGMA * sigma_all
    continuous: List[float] = []
    jump_count = 0
    for r in log_returns:
        if abs(r - mean) > threshold:
            jump_count += 1
        else:
            continuous.append(r)

    if len(continuous) < 3:
        # Almost all returns are jumps — fall back to full σ.
        return math.sqrt(var_all / dt)

    mean_c = sum(continuous) / len(continuous)
    var_c = sum((r - mean_c) ** 2 for r in continuous) / (len(continuous) - 1)
    sigma_c = math.sqrt(var_c) if var_c > 0 else 0.0

    # Step 3: if jumps were present, inflate σ_c by the ratio of total
    # to continuous realised variance so that the model's diffusion
    # coefficient reflects the *actual* experienced variance, not just
    # the smooth part.  Capped at 2× to avoid blow-up from a single
    # outlier tick.
    if jump_count > 0 and sigma_c > 0:
        ratio = min(math.sqrt(var_all / var_c), 2.0)
        sigma_c *= ratio

    return sigma_c / math.sqrt(dt) if dt != 1.0 else sigma_c


def estimate_momentum(prices: List[float], window: int = 30) -> float:
    """Estimate short-term per-second log-drift via OLS slope of log-prices.

    Uses only the last ``window`` prices from ``prices``.  Returns drift
    in per-second units (same scale as sigma from estimate_volatility*).
    Positive → uptrend, negative → downtrend, 0.0 → flat or insufficient data.
    """
    if len(prices) < 3:
        return 0.0
    tail = prices[-min(window, len(prices)):]
    n = len(tail)
    if n < 3:
        return 0.0
    log_p: List[float] = []
    for p in tail:
        if p > 0:
            log_p.append(math.log(p))
    n = len(log_p)
    if n < 3:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(log_p) / n
    num = sum((i - x_mean) * (log_p[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
#  Core probability computation
# ---------------------------------------------------------------------------

# Volatility bounds — prevent degenerate model behaviour.
_SIGMA_FLOOR = 1e-10    # below this, model is effectively deterministic
_SIGMA_CEILING = 0.05   # per-second; ~0.05 ≈ 300% annualised — beyond this
                        # the normal approximation of integrated GBM breaks down


def compute_probability(
    strike: float,
    current_price: float,
    sigma: float,
    time_to_settle_s: float,
    window_state: Optional[BtcWindowState] = None,
    tail_scale: float = 1.0,
    drift: float = 0.0,          # per-second log-drift
) -> float:
    """Return P(YES) — probability the 60-s average ≥ strike.

    The return value is always clamped to [0.0, 1.0].

    Parameters
    ----------
    strike : float
        Contract strike price.
    current_price : float
        Latest BTC mid price.
    sigma : float
        Per-second volatility of log-returns.
    time_to_settle_s : float
        Seconds until settlement.
    window_state : BtcWindowState, optional
        Current 60-second window state (needed when inside the window).
    """
    if current_price <= 0 or sigma < 0:
        return 0.5  # no information

    # Clamp sigma to prevent degenerate behaviour.
    sigma = max(sigma, _SIGMA_FLOOR)
    sigma = min(sigma, _SIGMA_CEILING)
    sigma *= max(1.0, tail_scale)

    window = 60.0

    if time_to_settle_s <= 0:
        # Settlement already happened — use observed average.
        if window_state and window_state.count > 0:
            return 1.0 if window_state.last_60_avg >= strike else 0.0
        return 0.5

    if time_to_settle_s > window:
        # --- Case 1: entire window is in the future ---
        # The average of GBM over [T-60, T] starting from S_now.
        # E[A] ≈ S_now  (zero drift assumption).
        # Var[A] ≈ S_now² · σ² · (T - 40)
        # (This is a simplified approximation for the variance of the 60s
        # average at time T, where T is seconds until settlement).
        # Apply drift over the interval centred on the averaging window.
        # Drift is per-second log-drift; horizon ≈ tts - 30 (midpoint of 60s window).
        drift_horizon = max(0.0, time_to_settle_s - 30.0)
        e_a = current_price * math.exp(drift * drift_horizon)
        var_a = (current_price ** 2) * (sigma ** 2) * (time_to_settle_s - 40.0)
        std_a = math.sqrt(var_a) if var_a > 0 else 1e-12
        z = (e_a - strike) / std_a
        return _clamp_prob(_norm_cdf(z))

    else:
        # --- Case 2: inside the 60-second window ---
        tau = min(time_to_settle_s, window)  # remaining seconds
        n_obs = int(window - tau)

        # Sum of observed prices.
        if window_state and window_state.count > 0:
            s_obs = window_state.last_60_sum
            n_actual_obs = window_state.count
        else:
            s_obs = 0.0
            n_actual_obs = 0

        # Required total from remaining seconds.
        required_total = window * strike - s_obs
        if tau <= 0:
            return 1.0 if s_obs / max(n_actual_obs, 1) >= strike else 0.0

        m_req = required_total / tau  # required per-second average

        # For very small τ (< 1 second), the variance from averaging
        # shrinks to nearly zero.  The probability becomes nearly
        # deterministic: if current_price > m_req → ~1, else → ~0.
        # We still compute via the normal CDF which handles this
        # gracefully (z → ±∞).
        e_future = current_price * math.exp(drift * tau / 2.0)
        var_future = (current_price ** 2) * (sigma ** 2) * tau / 3.0
        std_future = math.sqrt(var_future) if var_future > 0 else 1e-12
        z = (e_future - m_req) / std_future
        return _clamp_prob(_norm_cdf(z))


def _clamp_prob(p: float) -> float:
    """Ensure probability is in [0.0, 1.0]."""
    if math.isnan(p) or math.isinf(p):
        return 0.5
    return max(0.0, min(1.0, p))


# ---------------------------------------------------------------------------
#  Sync recompute for executor (off main event loop)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RecomputeSnapshot:
    """Immutable snapshot of engine state for running recompute in a thread."""
    now: float
    markets: Dict[str, MarketMetadata]
    last_price_by_asset: Dict[str, float]
    last_window_by_asset: Dict[str, BtcWindowState]
    sigma_by_asset: Dict[str, float]
    tail_scale_by_asset: Dict[str, float]
    momentum_by_asset: Dict[str, float]


def _recompute_chunk_sync(
    snapshot: _RecomputeSnapshot, tickers: List[str]
) -> List[FairProbability]:
    """Compute FairProbability for a subset of tickers. Runs in thread; no bus."""
    from datetime import datetime, timezone

    now = snapshot.now
    results: List[FairProbability] = []
    for ticker in tickers:
        meta = snapshot.markets.get(ticker)
        if meta is None:
            continue
        try:
            settle_dt = datetime.fromisoformat(
                meta.settlement_time_iso.replace("Z", "+00:00")
            )
            settle_epoch = settle_dt.timestamp()
        except (ValueError, AttributeError):
            continue

        tts = settle_epoch - now
        if tts < -120:
            continue

        price = snapshot.last_price_by_asset.get(meta.asset, 0.0)
        if price <= 0:
            continue

        asset_sigma = snapshot.sigma_by_asset.get(meta.asset, 0.0)
        asset_tail_scale = snapshot.tail_scale_by_asset.get(meta.asset, 1.0)
        asset_drift = snapshot.momentum_by_asset.get(meta.asset, 0.0)
        ws = snapshot.last_window_by_asset.get(meta.asset)

        if meta.is_range and meta.strike_floor is not None and meta.strike_cap is not None:
            p_ge_floor = compute_probability(
                strike=meta.strike_floor,
                current_price=price,
                sigma=asset_sigma,
                time_to_settle_s=tts,
                window_state=ws,
                tail_scale=asset_tail_scale,
                drift=asset_drift,
            )
            p_ge_cap = compute_probability(
                strike=meta.strike_cap,
                current_price=price,
                sigma=asset_sigma,
                time_to_settle_s=tts,
                window_state=ws,
                tail_scale=asset_tail_scale,
                drift=asset_drift,
            )
            p_yes = max(0.0, min(1.0, p_ge_floor - p_ge_cap))
        else:
            # For directional contracts with no reference price (strike=0), use
            # current price as effective strike: P(end > current) updates as price moves.
            effective_strike = meta.strike_price if meta.strike_price > 0 else price
            p_yes = compute_probability(
                strike=effective_strike,
                current_price=price,
                sigma=asset_sigma,
                time_to_settle_s=tts,
                window_state=ws,
                tail_scale=asset_tail_scale,
                drift=asset_drift,
            )

        fp = FairProbability(
            market_ticker=ticker,
            p_yes=p_yes,
            drift=asset_drift,
            model_inputs={
                "strike": meta.strike_price,
                "current_price": price,
                "sigma": asset_sigma,
                "time_to_settle_s": round(tts, 2),
                "window_avg": ws.last_60_avg if ws else 0.0,
                "window_count": ws.count if ws else 0,
            },
        )
        results.append(fp)
    return results


def _recompute_all_sync(snapshot: _RecomputeSnapshot) -> List[FairProbability]:
    """Compute FairProbability for all near-money markets. Runs in thread; no bus."""
    return _recompute_chunk_sync(snapshot, list(snapshot.markets.keys()))


# ---------------------------------------------------------------------------
#  Bus-integrated probability engine
# ---------------------------------------------------------------------------

class ProbabilityEngine:
    """Subscribes to window state + orderbook, publishes FairProbability."""

    def __init__(
        self,
        bus: "Bus",  # noqa: F821
        markets: Dict[str, MarketMetadata],
        vol_lookback_prices: Optional[List[float]] = None,
        urgent_move_pct: float = 0.005,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self._bus = bus
        self._markets = markets
        # Per-asset vol deques: keyed by asset (e.g. "BTC", "ETH", "SOL").
        # The old single self._vol_prices deque received prices from all three
        # assets, causing log-returns like log(2019/67511) = -3.51 to inflate
        # sigma massively (the bug this change fixes).
        self._vol_prices_by_asset: Dict[str, Deque[float]] = {}
        self._tail_scale_by_asset: Dict[str, float] = {}
        self._sigma_by_asset: Dict[str, float] = {}
        self._momentum_by_asset: Dict[str, float] = {}
        # vol_lookback_prices is kept for backward-compat but silently ignored;
        # it was only used to seed the old single deque and no callers pass it
        # in production.
        _ = vol_lookback_prices  # noqa: F841

        self._last_window_by_asset: Dict[str, BtcWindowState] = {}
        self._last_price_by_asset: Dict[str, float] = {}
        self._tasks: List[asyncio.Task] = []
        self._recompute_dirty: bool = False
        self._recompute_interval: float = 0.25  # seconds — max 4 recomputations/sec

        # Urgent-move threshold: fraction of price change that triggers an
        # immediate recompute, bypassing the 250ms throttle.  Lets us catch
        # flash crashes before other bots re-price the Kalshi order book.
        self._urgent_move_pct: float = urgent_move_pct

        # Optional thread-pool executor: when set, _recompute_all runs in a
        # worker thread so the event loop stays responsive (avoids ping spikes).
        self._executor: Optional[ThreadPoolExecutor] = executor
        self._first_fair_prob_logged: bool = False
        
    def _ingest_price(self, asset: str, price: float) -> None:
        """Update per-asset vol deque and recompute sigma. Called from _consume_prices."""
        if asset not in self._vol_prices_by_asset:
            self._vol_prices_by_asset[asset] = deque(maxlen=120)
        self._vol_prices_by_asset[asset].append(price)
        vol_prices = list(self._vol_prices_by_asset[asset])
        self._sigma_by_asset[asset] = estimate_volatility_harj(vol_prices)
        excess_kurtosis = estimate_excess_kurtosis(vol_prices)
        self._tail_scale_by_asset[asset] = min(4.0, math.sqrt((3.0 + excess_kurtosis) / 3.0))
        self._momentum_by_asset[asset] = estimate_momentum(vol_prices, window=30)

    async def update_markets(self, added: Dict[str, MarketMetadata], removed: List[str]) -> None:
        """Dynamically update tracked markets during a session."""
        for ticker in removed:
            self._markets.pop(ticker, None)
        for ticker, meta in added.items():
            self._markets[ticker] = meta
        log.info(f"Probability engine updated targets: +{len(added)} / -{len(removed)}")

    async def start(self) -> None:
        topics_price = ["btc.mid_price", "eth.mid_price", "sol.mid_price"]
        topics_window = ["btc.window_state", "eth.window_state", "sol.window_state"]
        self._tasks = []
        for t in topics_price:
            q = await self._bus.subscribe(t)
            self._tasks.append(asyncio.create_task(self._consume_prices(q, t)))
        for t in topics_window:
            q = await self._bus.subscribe(t)
            self._tasks.append(asyncio.create_task(self._consume_window(q, t)))
        # Throttled recomputation loop — replaces per-tick _recompute_all calls
        self._tasks.append(asyncio.create_task(self._recompute_loop()))

    def stop(self) -> None:
        """Cancel all background consumer tasks."""
        for task in self._tasks:
            task.cancel()

    async def _consume_prices(self, q: asyncio.Queue, topic: str) -> None:
        try:
            while True:
                msg = await q.get()
                asset = getattr(msg, "asset", topic.split(".")[0].upper()).upper()

                # Track the previous price so we can detect violent moves.
                prev_price = self._last_price_by_asset.get(asset, 0.0)
                self._last_price_by_asset[asset] = msg.price

                self._ingest_price(asset, msg.price)

                # Urgent path: if price moved >= urgent_move_pct in a single
                # tick, bypass the 250ms throttle and recompute immediately.
                # This captures flash crashes / violent pumps before Kalshi's
                # market makers can update their quotes.
                if prev_price > 0:
                    move_pct = abs(msg.price - prev_price) / prev_price
                    if move_pct >= self._urgent_move_pct:
                        log.info(
                            f"Urgent recompute triggered: {asset} moved "
                            f"{move_pct:.2%} (${prev_price:,.0f} → ${msg.price:,.0f})"
                        )
                        self._recompute_dirty = False  # cancel pending throttled run
                        await self._recompute_all()
                        continue  # skip setting dirty flag — just ran full recompute

                self._recompute_dirty = True
        except asyncio.CancelledError:
            pass

    async def _consume_window(self, q: asyncio.Queue, topic: str) -> None:
        try:
            while True:
                msg: BtcWindowState = await q.get()
                asset = getattr(msg, "asset", topic.split(".")[0].upper()).upper()
                self._last_window_by_asset[asset] = msg
                self._recompute_dirty = True
        except asyncio.CancelledError:
            pass

    async def _recompute_loop(self) -> None:
        """Throttled recomputation — runs at most every 250ms instead of on every tick."""
        try:
            while True:
                await asyncio.sleep(self._recompute_interval)
                if self._recompute_dirty:
                    self._recompute_dirty = False
                    await self._recompute_all()
        except asyncio.CancelledError:
            pass

    async def _recompute_all(self) -> None:
        """Recompute P(YES) for tracked markets and publish.

        When an executor is configured, runs the heavy math in a worker thread
        and only publishes on the event loop, keeping the loop responsive.
        """
        if self._executor is not None:
            # Offload CPU-bound work to worker threads; parallelize by market chunks.
            import time as _time
            loop = asyncio.get_running_loop()
            snapshot = _RecomputeSnapshot(
                now=_time.time(),
                markets=dict(self._markets),
                last_price_by_asset=dict(self._last_price_by_asset),
                last_window_by_asset=dict(self._last_window_by_asset),
                sigma_by_asset=dict(self._sigma_by_asset),
                tail_scale_by_asset=dict(self._tail_scale_by_asset),
                momentum_by_asset=dict(self._momentum_by_asset),
            )
            tickers = list(snapshot.markets.keys())
            n_workers = min(32, max(1, (os.cpu_count() or 4) * 2))
            if not tickers:
                return
            # Split tickers into n_workers chunks for parallel recompute.
            chunk_size = max(1, (len(tickers) + n_workers - 1) // n_workers)
            chunks = [
                tickers[i : i + chunk_size]
                for i in range(0, len(tickers), chunk_size)
            ]
            chunk_results = await asyncio.gather(
                *[
                    loop.run_in_executor(
                        self._executor,
                        _recompute_chunk_sync,
                        snapshot,
                        chunk,
                    )
                    for chunk in chunks
                ],
                return_exceptions=True,
            )
            results: List[FairProbability] = []
            for r in chunk_results:
                if isinstance(r, Exception):
                    log.warning("Recompute chunk failed: %s", r)
                    continue
                results.extend(r)
            if results and not self._first_fair_prob_logged:
                log.info(
                    "First fair_prob published",
                    data={"count": len(results), "sample_ticker": results[0].market_ticker},
                )
                self._first_fair_prob_logged = True
            for fp in results:
                await self._bus.publish(f"kalshi.fair_prob.{fp.market_ticker}", fp)
                await self._bus.publish("kalshi.fair_prob", fp)
            return

        # Inline path (no executor): original behavior for tests / backward compat.
        import time as _time
        from datetime import datetime, timezone

        now = _time.time()
        count = 0

        for ticker, meta in list(self._markets.items()):
            try:
                settle_dt = datetime.fromisoformat(
                    meta.settlement_time_iso.replace("Z", "+00:00")
                )
                settle_epoch = settle_dt.timestamp()
            except (ValueError, AttributeError):
                continue

            tts = settle_epoch - now
            if tts < -120:
                continue  # long past settlement

            price = self._last_price_by_asset.get(meta.asset, 0.0)
            if price <= 0:
                continue

            asset_sigma = self._sigma_by_asset.get(meta.asset, 0.0)
            asset_tail_scale = self._tail_scale_by_asset.get(meta.asset, 1.0)
            asset_drift = self._momentum_by_asset.get(meta.asset, 0.0)

            if meta.is_range and meta.strike_floor is not None and meta.strike_cap is not None:
                p_ge_floor = compute_probability(
                    strike=meta.strike_floor,
                    current_price=price,
                    sigma=asset_sigma,
                    time_to_settle_s=tts,
                    window_state=self._last_window_by_asset.get(meta.asset),
                    tail_scale=asset_tail_scale,
                    drift=asset_drift,
                )
                p_ge_cap = compute_probability(
                    strike=meta.strike_cap,
                    current_price=price,
                    sigma=asset_sigma,
                    time_to_settle_s=tts,
                    window_state=self._last_window_by_asset.get(meta.asset),
                    tail_scale=asset_tail_scale,
                    drift=asset_drift,
                )
                p_yes = max(0.0, min(1.0, p_ge_floor - p_ge_cap))
            else:
                # For directional contracts with no reference price (strike=0), use
                # current price as effective strike: P(end > current) updates as price moves.
                effective_strike = meta.strike_price if meta.strike_price > 0 else price
                p_yes = compute_probability(
                    strike=effective_strike,
                    current_price=price,
                    sigma=asset_sigma,
                    time_to_settle_s=tts,
                    window_state=self._last_window_by_asset.get(meta.asset),
                    tail_scale=asset_tail_scale,
                    drift=asset_drift,
                )

            ws = self._last_window_by_asset.get(meta.asset)
            fp = FairProbability(
                market_ticker=ticker,
                p_yes=p_yes,
                drift=asset_drift,
                model_inputs={
                    "strike": meta.strike_price,
                    "current_price": price,
                    "sigma": asset_sigma,
                    "time_to_settle_s": round(tts, 2),
                    "window_avg": ws.last_60_avg if ws else 0.0,
                    "window_count": ws.count if ws else 0,
                },
            )
            if not self._first_fair_prob_logged:
                log.info(
                    "First fair_prob published",
                    data={"market_ticker": ticker, "p_yes": round(p_yes, 4)},
                )
                self._first_fair_prob_logged = True
            await self._bus.publish(f"kalshi.fair_prob.{ticker}", fp)
            await self._bus.publish("kalshi.fair_prob", fp)

            count += 1
            if count % 25 == 0:
                await asyncio.sleep(0)
