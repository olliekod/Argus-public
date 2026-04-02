# Created by Oliver Meihls

# MispricingAnalyzer — structural diagnostics for Kalshi BTC contract scalping.
#
# Tracks the following measurements independently of terminal_ui state:
#
# Distance-from-strike buckets  (|S-K|/K):  ≤0.05%, 0.05-0.1%, 0.1-0.25%, >0.25%
# Time-to-settlement buckets:                <60s, 1-3min, 3-10min, >10min
# Realized volatility (60s / 120s):          BTC price range % over each window
# Expected contract move per 0.1% BTC move:  OLS slope of p_yes vs price (30s window)
# Move / spread ratio:                        expected_move / spread
# Repricing lag distribution:                median, p95, mean, n (near-money only)
# Strike-crossing events (30-min window):    count, avg edge at cross, median lag after
# Edge persistence durations (near-money):   avg, p95, current streak
#
# This module is diagnostic only. It never places orders and never calls the bus.
# Call update() once per frame; read public attributes for display.
#
# Interpretation guide
# DIST buckets   — edge concentrates near-the-money; >.25% should show low/zero edge
# TIME buckets   — near expiry (<60s) often shows high edge but wide spread and fast repricing
# RVol           — low vol → small moves → edge evaporates quickly after repricing
# Move/spread    — >1× means a 0.1% BTC move shifts p_yes by more than the spread (good)
# Lag p95        — upper tail of catch-up time; longer = more time to act after each move
# Crossings      — high edge at strike-cross moments is the most predictable scalp window
# Persistence    — avg>0.5s with p95>2s suggests edges are tradable before repricing

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

# ── Config ────────────────────────────────────────────────────────────────────
EDGE_THRESHOLD_CENTS  = 2          # min edge (¢) to count as opportunity
MOVE_THRESHOLD_FRAC   = 0.0005     # BTC move ≥ 0.05% triggers lag timer (fraction)
CATCHUP_EPSILON_CENTS = 2          # edge ≤ this → Kalshi has "caught up"
WINDOW_MINUTES        = 30         # rolling look-back for all stats
_STATS_HZ             = 1.0        # recompute at most once per second

# Distance-from-strike buckets (fractions of strike)
DIST_BUCKETS: List[Tuple[float, float]] = [
    (0.0,     0.0005),          # ≤ 0.05 %
    (0.0005,  0.001),           # 0.05 – 0.1 %
    (0.001,   0.0025),          # 0.1 – 0.25 %
    (0.0025,  float("inf")),    # > 0.25 %
]
DIST_LABELS: List[str] = ["≤.05%", ".05-.1%", ".1-.25%", ">.25%"]

# Time-to-settlement buckets (seconds remaining in current window)
TIME_BUCKETS: List[Tuple[float, float]] = [
    (0,   60),                  # < 60 s
    (60,  180),                 # 1 – 3 min
    (180, 600),                 # 3 – 10 min
    (600, float("inf")),        # > 10 min
]
TIME_LABELS: List[str] = ["<60s", "1-3m", "3-10m", ">10m"]

_MAXLEN_BUCKET = 3600           # per-bucket deque depth  (~6 min at 10 fps, trimmed on read)
_MAXLEN_PRICES = 1200           # 120 s at 10 fps for rvol
_MAXLEN_EVENTS = 300            # lag / cross / persist event deques


# ── Internal helpers ──────────────────────────────────────────────────────────

class _BucketStats:
    # Per-bucket rolling store of (ts, max_edge_cents, spread_cents) samples.

    __slots__ = ("_s",)

    def __init__(self) -> None:
        self._s: deque = deque(maxlen=_MAXLEN_BUCKET)

    def push(self, ts: float, max_edge: float, spread: float) -> None:
        self._s.append((ts, max_edge, spread))

    def stats(self, cutoff: float) -> Optional[Dict]:
        # Return stats dict for samples with ts >= cutoff, or None if no data.
        valid = [(e, s) for ts, e, s in self._s if ts >= cutoff]
        if not valid:
            return None
        edges   = [e for e, _ in valid]
        spreads = [s for _, s in valid]
        s_e     = sorted(edges)
        n       = len(s_e)
        return {
            "avg_edge":   sum(edges) / n,
            "p95_edge":   s_e[min(int(n * 0.95), n - 1)],
            "avg_spread": sum(spreads) / n,
            "n":          n,
        }


# ── Public API ────────────────────────────────────────────────────────────────

class MispricingAnalyzer:
    # Diagnostic-only structural analysis for Kalshi BTC contract scalping viability.
    #
    # Call update() once per frame. Read public attributes (rvol_60s_pct, lag_median_s,
    # etc.) or call dist_stats(cutoff) / time_stats(cutoff) for display.

    def __init__(self) -> None:
        # ── Bucket trackers ───────────────────────────────────────────────────
        self._dist: List[_BucketStats] = [_BucketStats() for _ in DIST_BUCKETS]
        self._time: List[_BucketStats] = [_BucketStats() for _ in TIME_BUCKETS]

        # ── Realized volatility ───────────────────────────────────────────────
        self._price_hist: deque = deque(maxlen=_MAXLEN_PRICES)   # (ts, price) — 120 s for rvol
        self._price_3s:   deque = deque(maxlen=30)               # (ts, price) — 3 s for move detect
        self.rvol_60s_pct:  Optional[float] = None   # BTC range % over last 60 s
        self.rvol_120s_pct: Optional[float] = None   # BTC range % over last 120 s

        # ── Expected contract move per 0.1% BTC (OLS slope, 30 s window) ─────
        self._dp_hist: deque = deque(maxlen=300)     # (ts, btc_price, p_yes)
        self.expected_move_c:   Optional[float] = None   # cents p_yes shifts per 0.1% BTC
        self.move_spread_ratio: Optional[float] = None   # expected_move / spread

        # ── Repricing lag distribution (near-money, confirmed catch-ups) ──────
        self._catchup_times: deque = deque(maxlen=_MAXLEN_EVENTS)
        self._move_event: Optional[Dict] = None           # {t0}
        self.lag_median_s: Optional[float] = None
        self.lag_p95_s:    Optional[float] = None
        self.lag_mean_s:   Optional[float] = None
        self.lag_n:        int = 0

        # ── Strike-crossing events ────────────────────────────────────────────
        self._prev_side: Optional[int] = None             # +1 above strike, -1 below
        self._cross_events:    deque = deque(maxlen=_MAXLEN_EVENTS)  # (ts, edge@cross)
        self._cross_lag_times: deque = deque(maxlen=_MAXLEN_EVENTS)
        self._cross_event: Optional[Dict] = None          # {t0, edge_at_cross}
        self.cross_count_30m:    int = 0
        self.cross_avg_edge_c:   Optional[float] = None
        self.cross_lag_median_s: Optional[float] = None

        # ── Edge persistence durations (near-money) ───────────────────────────
        self._persist_start: Optional[float] = None
        self._persist_durations: deque = deque(maxlen=_MAXLEN_EVENTS)
        self.persist_current_s: float = 0.0
        self.persist_avg_s:     Optional[float] = None
        self.persist_p95_s:     Optional[float] = None

        # ── Stat refresh throttle ─────────────────────────────────────────────
        self._last_stats_ts: float = 0.0
        self._last_spread:   float = 0.0   # cached for move_spread_ratio

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _range_pct(self, now: float, window_s: float) -> Optional[float]:
        # BTC price range as % of mid over the last *window_s* seconds.
        cut = now - window_s
        pts = [p for ts, p in self._price_hist if ts >= cut and p > 0]
        if len(pts) < 2:
            return None
        avg = sum(pts) / len(pts)
        return (max(pts) - min(pts)) / avg * 100.0

    # ── Snapshot accessors ────────────────────────────────────────────────────

    def dist_stats(self, cutoff: float) -> List[Optional[Dict]]:
        # Per-bucket stats dicts for the DIST_BUCKETS, filtered to ts >= cutoff.
        return [b.stats(cutoff) for b in self._dist]

    def time_stats(self, cutoff: float) -> List[Optional[Dict]]:
        # Per-bucket stats dicts for the TIME_BUCKETS, filtered to ts >= cutoff.
        return [b.stats(cutoff) for b in self._time]

    # ── Main frame update ─────────────────────────────────────────────────────

    def update(
        self,
        now:          float,
        price:        float,   # BTC mid price
        p_yes:        float,   # fair probability (0–1)
        strike:       float,   # contract reference/strike price
        t_rem_s:      float,   # seconds remaining in current 15-min window
        edge_yes:     float,   # pre-computed edge YES cents
        edge_no:      float,   # pre-computed edge NO cents
        spread:       float,   # pre-computed round-trip spread cents
        near_money:   bool,    # BTC within NEAR_MONEY_FRAC of strike
    ) -> None:
        # Called once per frame. All work is O(1) per call; heavy stats at 1 Hz.
        max_edge = max(edge_yes, edge_no)
        self._last_spread = spread

        # Always: price history (for rvol + move detection) and p_yes history (for slope)
        if price > 0:
            self._price_hist.append((now, price))
            self._price_3s.append((now, price))
        if price > 0 and 0.0 < p_yes < 1.0:
            self._dp_hist.append((now, price, p_yes))

        # ── Distance-from-strike bucket ───────────────────────────────────────
        if strike > 0 and price > 0:
            dist_frac = abs(price - strike) / strike
            for i, (lo, hi) in enumerate(DIST_BUCKETS):
                if lo <= dist_frac < hi:
                    self._dist[i].push(now, max_edge, spread)
                    break

        # ── Time-to-settlement bucket ─────────────────────────────────────────
        if t_rem_s >= 0:
            for i, (lo, hi) in enumerate(TIME_BUCKETS):
                if lo <= t_rem_s < hi:
                    self._time[i].push(now, max_edge, spread)
                    break

        # ── Strike-crossing detection (all frames, not gated) ─────────────────
        if strike > 0 and price > 0:
            cur_side = 1 if price >= strike else -1
            if self._prev_side is not None and cur_side != self._prev_side:
                # BTC just crossed the strike
                self._cross_events.append((now, max_edge))
                self._cross_event = {"t0": now, "edge_at_cross": max_edge}
            if self._cross_event is not None:
                elapsed = now - self._cross_event["t0"]
                if max_edge <= CATCHUP_EPSILON_CENTS:
                    self._cross_lag_times.append(elapsed)
                    self._cross_event = None
                elif elapsed > 60.0:
                    self._cross_event = None   # abandon stale
            self._prev_side = cur_side

        # ── Near-money gate: lag, edge persistence ────────────────────────────
        if near_money:
            # BTC move detection (compare against 3-second-old price)
            if price > 0 and self._price_3s:
                oldest_ts, oldest_price = self._price_3s[0]
                if (now - oldest_ts) <= 3.0 and oldest_price > 0:
                    frac = abs(price - oldest_price) / oldest_price
                    if frac >= MOVE_THRESHOLD_FRAC and self._move_event is None:
                        self._move_event = {"t0": now}

            # Catch-up check
            if self._move_event is not None:
                elapsed = now - self._move_event["t0"]
                if max_edge <= CATCHUP_EPSILON_CENTS:
                    self._catchup_times.append(elapsed)
                    self.lag_n += 1
                    self._move_event = None
                elif elapsed > 30.0:
                    self._move_event = None   # abandon

            # Edge persistence streak
            if max_edge >= EDGE_THRESHOLD_CENTS:
                if self._persist_start is None:
                    self._persist_start = now
                self.persist_current_s = now - self._persist_start
            else:
                if self._persist_start is not None:
                    self._persist_durations.append(now - self._persist_start)
                    self._persist_start = None
                self.persist_current_s = 0.0

        else:
            # Exiting near-money zone: close open trackers cleanly
            if self._persist_start is not None:
                self._persist_durations.append(now - self._persist_start)
                self._persist_start = None
            self.persist_current_s = 0.0
            if self._move_event is not None and (now - self._move_event["t0"]) > 30.0:
                self._move_event = None

        # ── Rolling stats (1 Hz) ──────────────────────────────────────────────
        if now - self._last_stats_ts < (1.0 / _STATS_HZ):
            return
        self._last_stats_ts = now

        # Realized volatility: price range as % of mid over each window
        self.rvol_60s_pct  = self._range_pct(now, 60.0)
        self.rvol_120s_pct = self._range_pct(now, 120.0)

        # Expected contract move per 0.1% BTC (OLS slope of p_yes on btc_price, 30 s)
        pts_30 = [(p, py) for ts, p, py in self._dp_hist if ts >= now - 30 and p > 0]
        self.expected_move_c   = None
        self.move_spread_ratio = None
        if len(pts_30) >= 5:
            prices_v = [p for p, _ in pts_30]
            pyes_v   = [py for _, py in pts_30]
            mp  = sum(prices_v) / len(prices_v)
            mpy = sum(pyes_v)   / len(pyes_v)
            cov  = sum((prices_v[i] - mp) * (pyes_v[i] - mpy) for i in range(len(pts_30)))
            varp = sum((p - mp) ** 2 for p in prices_v)
            if varp > 0:
                slope = cov / varp             # dp_yes per $1 BTC
                # cents p_yes moves when BTC moves 0.1% of its current price
                self.expected_move_c = abs(slope * mp * 0.001 * 100)
                if self.expected_move_c > 0 and self._last_spread > 0:
                    self.move_spread_ratio = self.expected_move_c / self._last_spread

        # Repricing lag distribution
        ct = sorted(self._catchup_times)
        if ct:
            n = len(ct)
            self.lag_median_s = ct[n // 2]
            self.lag_p95_s    = ct[min(int(n * 0.95), n - 1)]
            self.lag_mean_s   = sum(ct) / n

        # Strike-crossing stats (30-min window)
        cutoff_30m = now - 30 * 60
        recent = [(ts, e) for ts, e in self._cross_events if ts >= cutoff_30m]
        self.cross_count_30m = len(recent)
        self.cross_avg_edge_c = (
            sum(e for _, e in recent) / len(recent) if recent else None
        )
        clt = sorted(self._cross_lag_times)
        self.cross_lag_median_s = clt[len(clt) // 2] if clt else None

        # Edge persistence stats
        pd = sorted(self._persist_durations)
        if pd:
            n = len(pd)
            self.persist_avg_s = sum(pd) / n
            self.persist_p95_s = pd[min(int(n * 0.95), n - 1)]
        else:
            self.persist_avg_s = None
            self.persist_p95_s = None
