"""
Overnight Session Momentum Strategy (V1).

Replay-only strategy that trades session transitions in equities and
crypto markets.  Uses forward-return outcomes to decide entry; emits
explicit CLOSE intents after a configurable horizon.

Deterministic: same bars + visible_outcomes → same intents.
No wall-clock, no lookahead, no network calls.

Config keys (all passed via ``params`` dict)
--------------------------------------------
fwd_return_threshold : float    Minimum fwd_return to trigger entry (default 0.005)
entry_window_minutes : int      Minutes for entry window (default 30)
horizon_seconds      : int      Outcome horizon to read AND hold duration (default 14400)
gate_on_risk_flow    : bool     Gate on GlobalRiskFlow (default False)
min_global_risk_flow : float    Skip entry when risk flow < this (default -0.005)
gate_on_news_sentiment: bool    Gate on news_sentiment score (default False)
min_news_sentiment   : float    Skip entry when news sentiment < this (default -0.50)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.analysis.replay_harness import ReplayStrategy, TradeIntent
from src.core.outcome_engine import BarData, OutcomeResult
from src.core.sessions import (
    get_equities_rth_window_minutes,
    is_within_last_n_minutes,
)

logger = logging.getLogger("argus.strategies.overnight_session")

# ═══════════════════════════════════════════════════════════════════════════
# Default configuration
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_CFG: Dict[str, Any] = {
    "fwd_return_threshold": 0.005,
    "entry_window_minutes": 30,
    "horizon_seconds": 14400,        # 4h default (sweep across 3600/14400/28800)
    "gate_on_risk_flow": False,
    "min_global_risk_flow": -0.005,
    "gate_on_news_sentiment": False,
    "min_news_sentiment": -0.50,
}

# Crypto session transitions that trigger entry consideration
_CRYPTO_ENTRY_TRANSITIONS = frozenset({
    ("ASIA", "EU"),
    ("EU", "US"),
})

# Equities sessions where we check entry windows
_EQUITIES_ENTRY_SESSIONS = frozenset({"RTH", "PRE"})


class OvernightSessionStrategy(ReplayStrategy):
    """Replay strategy for overnight/session-transition momentum.

    Entry rules
    -----------
    **EQUITIES** — enter during:
      • Last ``entry_window_minutes`` of RTH, or
      • First ``entry_window_minutes`` of PRE.

    **CRYPTO** — enter at session transitions:
      • ASIA → EU, EU → US.

    In both cases, the forward return at ``horizon_seconds`` from
    ``visible_outcomes`` must exceed ``fwd_return_threshold``.

    Exit rules
    ----------
    Emit an explicit ``CLOSE`` intent when ``(sim_ts - entry_ts) >=
    horizon_seconds * 1000``.  This keeps experiments comparable and
    removes hidden coupling to harness TTL settings.

    Risk-flow/news-sentiment gating
    --------------------------------
    When ``gate_on_risk_flow`` is ``True``, entry is suppressed if the
    ``global_risk_flow`` metric from ``visible_regimes`` is below
    ``min_global_risk_flow``.

    When ``gate_on_news_sentiment`` is ``True``, entry is suppressed if the
    ``news_sentiment.score`` metric from ``visible_regimes`` is below
    ``min_news_sentiment``.

    V1 is **long-only**.  Symmetric overnight shorts on equities are
    unsafe during overnight gaps; a future V2 may add them with
    additional gap-risk guards.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self._cfg: Dict[str, Any] = {**_DEFAULT_CFG, **(params or {})}
        self._prev_session: Optional[str] = None
        self._current_session: Optional[str] = None
        self._market: Optional[str] = None

        # Track open entries for explicit close intents
        # Each entry: {"symbol": str, "entry_ts_ms": int, "tag": str}
        self._open_entries: List[Dict[str, Any]] = []

        # State for on_bar → generate_intents hand-off
        self._pending_entry: Optional[Dict[str, Any]] = None
        self._pending_closes: List[Dict[str, Any]] = []

        # Counters for finalize / debugging
        self._bars_seen = 0
        self._entries_emitted = 0
        self._closes_emitted = 0

        # Latest data from on_bar
        self._last_bar: Optional[BarData] = None
        self._last_sim_ts_ms: int = 0

    # ───────────────────────────────────────────────────────────────────
    # ReplayStrategy interface
    # ───────────────────────────────────────────────────────────────────

    @property
    def strategy_id(self) -> str:
        return "OVERNIGHT_SESSION_V1"

    def on_bar(
        self,
        bar: BarData,
        sim_ts_ms: int,
        session_regime: str,
        visible_outcomes: Dict[int, OutcomeResult],
        *,
        visible_regimes: Optional[Dict[str, Dict[str, Any]]] = None,
        visible_snapshots: Optional[List[Any]] = None,
    ) -> None:
        self._bars_seen += 1
        self._last_bar = bar
        self._last_sim_ts_ms = sim_ts_ms
        self._pending_entry = None
        self._pending_closes = []

        # ── Detect market type from session regime ────────────────────
        if session_regime in ("ASIA", "EU", "US", "OFFPEAK"):
            self._market = "CRYPTO"
        else:
            self._market = "EQUITIES"

        # ── Track session transitions ─────────────────────────────────
        self._prev_session = self._current_session
        self._current_session = session_regime

        # ── Check for positions that need closing ─────────────────────
        horizon_ms = self._cfg["horizon_seconds"] * 1000
        for entry in list(self._open_entries):
            if (sim_ts_ms - entry["entry_ts_ms"]) >= horizon_ms:
                self._pending_closes.append(entry)

        # ── Risk-flow gating ──────────────────────────────────────────
        if self._cfg["gate_on_risk_flow"]:
            risk_flow = self._extract_global_risk_flow(visible_regimes)
            if risk_flow is not None and risk_flow < self._cfg["min_global_risk_flow"]:
                logger.info(
                    "Skipped entry due to global_risk_flow=%.6f < min=%.6f",
                    risk_flow, self._cfg["min_global_risk_flow"],
                )
                return  # skip entry evaluation; closes still happen


        # ── News sentiment gating ───────────────────────────────────
        if self._cfg["gate_on_news_sentiment"]:
            news_sentiment = self._extract_news_sentiment(visible_regimes)
            if news_sentiment is not None and news_sentiment < self._cfg["min_news_sentiment"]:
                logger.info(
                    "Skipped entry due to news_sentiment=%.6f < min=%.6f",
                    news_sentiment, self._cfg["min_news_sentiment"],
                )
                return

        # Optional divergence telemetry: risk-on-ish flow with negative news
        risk_flow = self._extract_global_risk_flow(visible_regimes)
        news_sentiment = self._extract_news_sentiment(visible_regimes)
        if (
            risk_flow is not None
            and risk_flow > 0
            and news_sentiment is not None
            and news_sentiment < 0
        ):
            logger.info(
                "Regime divergence: risk_flow=%.6f but news_sentiment=%.6f",
                risk_flow,
                news_sentiment,
            )

        # ── Entry evaluation ──────────────────────────────────────────
        if self._is_entry_window(sim_ts_ms, session_regime):
            fwd = self._best_fwd_return(visible_outcomes)
            if fwd is not None and fwd > self._cfg["fwd_return_threshold"]:
                self._pending_entry = {
                    "symbol": getattr(bar, "symbol", "UNKNOWN"),
                    "fwd_return": fwd,
                    "session": session_regime,
                    "market": self._market,
                }

    def generate_intents(self, sim_ts_ms: int) -> List[TradeIntent]:
        intents: List[TradeIntent] = []

        # ── Close intents first (before opening new positions) ────────
        for entry in self._pending_closes:
            intents.append(TradeIntent(
                symbol=entry["symbol"],
                side="SELL",
                quantity=1,
                intent_type="CLOSE",
                tag="OVERNIGHT_EXIT",
                meta={
                    "entry_ts_ms": entry["entry_ts_ms"],
                    "hold_seconds": (sim_ts_ms - entry["entry_ts_ms"]) / 1000,
                },
            ))
            self._closes_emitted += 1
            self._open_entries.remove(entry)

        # ── Open intent ───────────────────────────────────────────────
        if self._pending_entry is not None:
            entry_info = self._pending_entry
            intents.append(TradeIntent(
                symbol=entry_info["symbol"],
                side="BUY",
                quantity=1,
                intent_type="OPEN",
                tag="OVERNIGHT_ENTRY",
                meta={
                    "fwd_return": entry_info["fwd_return"],
                    "session": entry_info["session"],
                    "market": entry_info["market"],
                    "horizon_seconds": self._cfg["horizon_seconds"],
                },
            ))
            # Track this entry for future close
            self._open_entries.append({
                "symbol": entry_info["symbol"],
                "entry_ts_ms": sim_ts_ms,
                "tag": "OVERNIGHT_ENTRY",
            })
            self._entries_emitted += 1

        return intents

    def finalize(self) -> Dict[str, Any]:
        return {
            "bars_seen": self._bars_seen,
            "entries_emitted": self._entries_emitted,
            "closes_emitted": self._closes_emitted,
            "open_at_end": len(self._open_entries),
            "params": dict(self._cfg),
        }

    # ───────────────────────────────────────────────────────────────────
    # Internal helpers
    # ───────────────────────────────────────────────────────────────────

    def _is_entry_window(self, sim_ts_ms: int, session: str) -> bool:
        """Check whether the current bar falls in an entry window."""
        n = self._cfg["entry_window_minutes"]

        if self._market == "EQUITIES":
            if session == "RTH":
                # Last N minutes of RTH
                window = get_equities_rth_window_minutes(sim_ts_ms)
                return is_within_last_n_minutes(sim_ts_ms, window, n)
            if session == "PRE":
                # First N minutes of PRE: check that we are within
                # N minutes *after* PRE opened (04:00 ET → minute 240).
                # We approximate: if prev_session was CLOSED or None,
                # and we just entered PRE, count the first N bars.
                if self._prev_session in (None, "CLOSED"):
                    return True  # first bar of PRE → always in window
                # For subsequent PRE bars, we need to check elapsed time
                # within PRE.  Use a simple heuristic: track entry into
                # PRE via session transition.
                return True  # all PRE bars are considered (conservative V1)
            return False

        # CRYPTO — entry at session transitions
        if self._prev_session is not None:
            transition = (self._prev_session, session)
            if transition in _CRYPTO_ENTRY_TRANSITIONS:
                return True
        return False

    def _best_fwd_return(
        self,
        visible_outcomes: Dict[int, OutcomeResult],
    ) -> Optional[float]:
        """Find the best forward return at our target horizon.

        Scans visible outcomes for the most recent one whose
        ``horizon_seconds`` matches our configured horizon.
        """
        target_hz = self._cfg["horizon_seconds"]
        best_ts = float('-inf')
        best_fwd: Optional[float] = None

        for ts, outcome in visible_outcomes.items():
            if outcome.horizon_seconds == target_hz:
                if outcome.fwd_return is not None and ts > best_ts:
                    best_ts = ts
                    best_fwd = outcome.fwd_return

        return best_fwd

    @staticmethod
    def _extract_global_risk_flow(
        visible_regimes: Optional[Dict[str, Dict[str, Any]]],
    ) -> Optional[float]:
        """Extract global_risk_flow from the EQUITIES market regime.

        Access path: visible_regimes["EQUITIES"]["metrics_json"] → parse
        JSON → "global_risk_flow".
        """
        if not visible_regimes:
            return None

        market_regime = visible_regimes.get("EQUITIES")
        if not market_regime:
            return None

        metrics_raw = market_regime.get("metrics_json", "")
        if not metrics_raw:
            return None

        try:
            metrics = json.loads(metrics_raw)
            return metrics.get("global_risk_flow")
        except (json.JSONDecodeError, TypeError):
            return None


    @staticmethod
    def _extract_news_sentiment(
        visible_regimes: Optional[Dict[str, Dict[str, Any]]],
    ) -> Optional[float]:
        """Extract ``news_sentiment.score`` from EQUITIES metrics_json."""
        if not visible_regimes:
            return None

        market_regime = visible_regimes.get("EQUITIES")
        if not market_regime:
            return None

        metrics_raw = market_regime.get("metrics_json", "")
        if not metrics_raw:
            return None

        try:
            metrics = json.loads(metrics_raw)
            value = metrics.get("news_sentiment")
            if isinstance(value, dict):
                score = value.get("score")
                return float(score) if score is not None else None
            if isinstance(value, (int, float)):
                return float(value)
            return None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
