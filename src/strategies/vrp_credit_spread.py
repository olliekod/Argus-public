# Created by Oliver Meihls

# VRP Credit Spread Strategy (Reference Strategy #2)
#
# Generates credit spread signals when Volatility Risk Premium (VRP) is high,
# conditioned on market and symbol regimes.
#
# Signal Logic:
# 1.  Monitor IV (from Greeks/Snapshots) and RV (Realized Volatility).
# 2.  Compute VRP = IV - RV.
# 3.  If VRP > Threshold AND Regime == BULLISH/NEUTRAL AND Volatility is STABLE:
# - Emit intent to SELL Put Spread.
#
# IV Source Selection (data-source policy):
# - PRIMARY: Tastytrade snapshots (``atm_iv`` / surface fields).
# - FALLBACK: Derived IV computed from Tastytrade bid/ask quotes
# when ``atm_iv`` is missing.
# - Alpaca is bars and outcomes only; it is never used for IV or options data.

import logging
import math
from typing import Dict, List, Any, Optional

from src.analysis.replay_harness import ReplayStrategy, TradeIntent
from src.core.outcome_engine import BarData, OutcomeResult
from src.core.regimes import VolRegime, TrendRegime

logger = logging.getLogger("argus.strategies.vrp_credit_spread")

# One-time IV diagnosis log (avoid spam)
_iv_diagnosis_logged: bool = False

# Provider names (must match data_sources policy values)
_TASTYTRADE = "tastytrade"
_ALPACA = "alpaca"


def _derive_iv_from_quotes(snapshot: Any) -> Optional[float]:
    # Attempt to derive ATM implied volatility from bid/ask quotes.
    #
    # This is a simplified Newton-Raphson IV solver used as a fallback
    # when the snapshot's ``atm_iv`` field is absent.  It uses the
    # midpoint of the closest-to-ATM put as a proxy.
    #
    # Returns None if derivation is not possible.
    try:
        import json as _json

        quotes_raw = None
        if hasattr(snapshot, "quotes_json"):
            quotes_raw = snapshot.quotes_json
        elif isinstance(snapshot, dict):
            quotes_raw = snapshot.get("quotes_json")

        if not quotes_raw:
            return None

        quotes = _json.loads(quotes_raw) if isinstance(quotes_raw, str) else quotes_raw
        puts = quotes.get("puts", [])
        if not puts:
            return None

        underlying = None
        if hasattr(snapshot, "underlying_price"):
            underlying = snapshot.underlying_price
        elif isinstance(snapshot, dict):
            underlying = snapshot.get("underlying_price")
        if not underlying or float(underlying or 0) <= 0:
            underlying = quotes.get("underlying_price")

        try:
            underlying = float(underlying)
        except (TypeError, ValueError):
            return None
        if not underlying or underlying <= 0:
            return None

        # Find closest-to-ATM put with usable price (prefer iv, then bid/ask, then mid/last)
        best = None
        best_dist = float("inf")
        for p in puts:
            try:
                strike = p.get("strike")
                if strike is None:
                    continue
                strike = float(strike)
            except (TypeError, ValueError):
                continue
            dist = abs(strike - underlying)
            if dist >= best_dist:
                continue
            # Prefer put with iv > 0 (direct from provider)
            put_iv = p.get("iv")
            if put_iv is not None:
                try:
                    iv_f = float(put_iv)
                    if iv_f > 0 and 0.005 <= iv_f <= 5.0:
                        best_dist = dist
                        best = p
                        continue
                except (TypeError, ValueError):
                    pass

            bid = p.get("bid")
            ask = p.get("ask")
            if bid is not None and ask is not None:
                try:
                    bid_f, ask_f = float(bid), float(ask)
                    if bid_f > 0 and ask_f > 0:
                        best_dist = dist
                        best = p
                        continue
                except (TypeError, ValueError):
                    pass

        if best is None:
            return None

        # Use iv if present, else bid/ask mid, else mid/last for approximation
        put_iv = best.get("iv")
        if put_iv is not None:
            try:
                iv_f = float(put_iv)
                if 0.005 <= iv_f <= 5.0:
                    return round(iv_f, 6)
            except (TypeError, ValueError):
                pass

        bid, ask = best.get("bid"), best.get("ask")
        if bid is not None and ask is not None:
            try:
                mid = (float(bid) + float(ask)) / 2.0
            except (TypeError, ValueError):
                mid = None
        else:
            mid = None
        if mid is None or mid <= 0:
            mid = best.get("mid") or best.get("last")
            try:
                mid = float(mid) if mid is not None else None
            except (TypeError, ValueError):
                mid = None
        if mid is None or mid <= 0:
            return None
        strike = float(best["strike"])
        # Rough Brenner-Subrahmanyam approximation:  IV ≈ mid / (0.4 * S * sqrt(T))
        # Assume ~14 DTE (0.038 years) if not provided.
        dte_years = best.get("dte_years", 14 / 365.0)
        try:
            dte_years = float(dte_years)
        except (TypeError, ValueError):
            dte_years = 14 / 365.0
        if dte_years <= 0:
            dte_years = 14 / 365.0
        iv_approx = mid / (0.4 * underlying * math.sqrt(dte_years))
        if 0.005 <= iv_approx <= 5.0:
            return round(iv_approx, 6)
        return None
    except Exception:
        return None


def _select_iv_from_snapshots(visible_snapshots: List[Any]) -> Optional[float]:
    # Select the best IV value from visible snapshots.
    #
    # Selection order:
    # 1. Latest Tastytrade snapshot with non-null ``atm_iv``.
    # 2. Derived IV from latest Tastytrade snapshot's bid/ask quotes.
    # 3. Latest snapshot from any other allowed provider with ``atm_iv`` (replay fallback).
    # 4. None — logs one-line reason and strategy skips deterministically.
    #
    # Alpaca is never used for IV (bars/outcomes only).
    if not visible_snapshots:
        return None

    def _provider(snap: Any) -> str:
        raw = ""
        if hasattr(snap, "source"):
            raw = getattr(snap, "source", "")
        elif hasattr(snap, "provider"):
            raw = getattr(snap, "provider", "")
        elif isinstance(snap, dict):
            raw = snap.get("provider", snap.get("source", ""))
        return (raw or "").lower().strip()

    def _atm_iv(snap: Any) -> Optional[float]:
        if hasattr(snap, "atm_iv"):
            return snap.atm_iv
        if isinstance(snap, dict):
            return snap.get("atm_iv")
        return None

    def _recv_ts(snap: Any) -> int:
        if hasattr(snap, "recv_ts_ms"):
            return getattr(snap, "recv_ts_ms", 0)
        if isinstance(snap, dict):
            return int(snap.get("recv_ts_ms") or snap.get("timestamp_ms") or 0)
        return 0

    tt_snaps = [s for s in visible_snapshots if _provider(s) == _TASTYTRADE]
    tt_snap_count = len(tt_snaps)

    # Pass 1: Use the most recent Tastytrade snapshot that has atm_iv > 0 (so we trade when IV exists)
    with_iv = [(s, _atm_iv(s)) for s in tt_snaps if _atm_iv(s) is not None and _atm_iv(s) > 0]
    if with_iv:
        best = max(with_iv, key=lambda x: _recv_ts(x[0]))
        return best[1]

    # Pass 2: Derived IV from Tastytrade quotes (newest first)
    for snap in sorted(tt_snaps, key=_recv_ts, reverse=True):
        iv = _derive_iv_from_quotes(snap)
        if iv is not None:
            logger.debug("Using derived IV %.4f from Tastytrade quotes", iv)
            return iv

    # Pass 3: Any allowed provider with atm_iv (replay fallback). Alpaca is never used for IV.
    # Exclude Alpaca: bars/outcomes only, never IV or options
    non_alpaca = [s for s in visible_snapshots if _provider(s) != _ALPACA]
    any_with_iv = [(s, _atm_iv(s)) for s in non_alpaca if _atm_iv(s) is not None and _atm_iv(s) > 0]
    if any_with_iv:
        best = max(any_with_iv, key=lambda x: _recv_ts(x[0]))
        logger.debug("Using IV %.4f from %s (replay fallback)", best[1], _provider(best[0]))
        return best[1]

    # When many visible but still no IV, log once to confirm pack→object atm_iv propagation
    if len(visible_snapshots) >= 200:
        sample_ivs = [_atm_iv(s) for s in visible_snapshots[:10]]
        logger.warning(
            "VRP IV: %d visible but 0 with IV; sample atm_iv from objects: %s",
            len(visible_snapshots), sample_ivs,
        )

    # Log one-line reason for IV unavailability (and one-time INFO for diagnosis)
    if tt_snap_count == 0:
        logger.debug("IV unavailable: no tastytrade snapshots in %d visible snapshots", len(visible_snapshots))
    else:
        logger.debug("IV unavailable: %d tastytrade snapshots lack atm_iv and bid/ask quotes", tt_snap_count)
    # One-time WARNING to diagnose why IV was not found (visible vs provider vs atm_iv)
    global _iv_diagnosis_logged
    if not _iv_diagnosis_logged and (tt_snap_count == 0 or not with_iv):
        _iv_diagnosis_logged = True
        providers_seen = {_provider(s) for s in visible_snapshots} if visible_snapshots else set()
        n_with_iv = sum(1 for s in visible_snapshots if _atm_iv(s) is not None and _atm_iv(s) > 0) if visible_snapshots else 0
        sample_recv = [_recv_ts(s) for s in visible_snapshots[:3]] if visible_snapshots else []
        logger.warning(
            "VRP IV diagnosis: visible=%d providers=%s tt_snaps=%d with_iv=%d sample_recv_ts_ms=%s (no IV selected)",
            len(visible_snapshots) if visible_snapshots else 0, providers_seen, tt_snap_count, n_with_iv, sample_recv,
        )

    return None


def _select_atm_put_quote(visible_snapshots: List[Any], underlying_price: float) -> Optional[Dict[str, Any]]:
    # Return the best available ATM put quote dict from visible snapshots.
    #
    # Used to attach real option pricing to a TradeIntent so that the
    # execution model fills at actual option prices rather than equity OHLC.
    #
    # Returns a dict with keys: bid, ask, bid_size, ask_size, quote_ts_ms, recv_ts_ms
    # or None if no usable quote is found.
    if not visible_snapshots or underlying_price <= 0:
        return None

    import json as _json

    def _recv_ts(snap: Any) -> int:
        if hasattr(snap, "recv_ts_ms"):
            return getattr(snap, "recv_ts_ms", 0)
        if isinstance(snap, dict):
            return int(snap.get("recv_ts_ms") or snap.get("timestamp_ms") or 0)
        return 0

    def _quotes_raw(snap: Any) -> Optional[str]:
        if hasattr(snap, "quotes_json"):
            return snap.quotes_json
        if isinstance(snap, dict):
            return snap.get("quotes_json")
        return None

    # Prefer the most recent snapshot that has quotes
    for snap in sorted(visible_snapshots, key=_recv_ts, reverse=True):
        raw = _quotes_raw(snap)
        if not raw:
            continue
        try:
            data = _json.loads(raw) if isinstance(raw, str) else raw
            puts = data.get("puts", [])
            if not puts:
                continue
            # Find closest-to-ATM put with a non-zero bid AND ask
            best_put = None
            best_dist = float("inf")
            for p in puts:
                try:
                    strike = float(p.get("strike", 0))
                    bid = float(p.get("bid") or 0)
                    ask = float(p.get("ask") or 0)
                    if bid <= 0 or ask <= 0:
                        continue
                    dist = abs(strike - underlying_price)
                    if dist < best_dist:
                        best_dist = dist
                        best_put = p
                except (TypeError, ValueError):
                    continue
            if best_put is not None:
                recv = _recv_ts(snap)
                return {
                    "bid": float(best_put.get("bid", 0)),
                    "ask": float(best_put.get("ask", 0)),
                    "bid_size": int(best_put.get("bid_size") or 0),
                    "ask_size": int(best_put.get("ask_size") or 0),
                    "quote_ts_ms": recv,
                    "recv_ts_ms": recv,
                }
        except Exception:
            continue
    return None


class VRPCreditSpreadStrategy(ReplayStrategy):
    # Ref strategy for VRP put spread selling.
    #
    # Uses Tastytrade snapshots as the authoritative IV source.
    # Falls back to derived IV (Brenner-Subrahmanyam approximation)
    # from Tastytrade bid/ask when ``atm_iv`` is absent.
    # Alpaca is never used for IV (bars and outcomes only).
    #
    # Parameters
    # thresholds : dict, optional
    # Strategy parameters dict.  Accepts a ``symbol`` key to specify
    # the underlying (default ``"IBIT"``).  The symbol is used for
    # both regime-scope lookup and TradeIntent symbol.

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self._thresholds = thresholds or {
            "min_vrp": 0.05,       # 5 points of IV over RV
            "max_vol_regime": "VOL_NORMAL",
            "avoid_trend": "TREND_DOWN",
        }
        # Derive the symbol from params; default to IBIT (the primary use-case).
        # Accepts both `symbol` and `underlying` keys for convenience.
        self.symbol: str = (
            self._thresholds.get("symbol")
            or self._thresholds.get("underlying")
            or "IBIT"
        )
        self.last_close = 0.0
        self.last_iv = None
        self.last_iv_source = None  # "tastytrade", "derived", or other allowed provider
        self.last_rv = None
        self._last_visible_snapshots: List[Any] = []
        self._logged_no_iv = False
        self._logged_no_rv = False
        self._logged_vrp_or_regime = False
        self._logged_gating = False
        self.open_position = False

    @property
    def strategy_id(self) -> str:
        return "VRP_PUT_SPREAD_V1"

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
        self.last_close = bar.close
        self.visible_regimes = visible_regimes or {}
        self._last_visible_snapshots = visible_snapshots or []

        # Extract IV using provider-aware selection (Alpaca never used for IV)
        iv = _select_iv_from_snapshots(visible_snapshots or [])
        if iv is not None:
            self.last_iv = iv
            if not getattr(self, "_logged_iv_ok", False):
                self._logged_iv_ok = True
                logger.warning("VRP: first IV selected %.4f (visible_snapshots=%d)", iv, len(visible_snapshots or []))

        # Extract realized_vol from latest outcome (if available)
        for ts in sorted(visible_outcomes.keys(), reverse=True):
            o = visible_outcomes[ts]
            if o.realized_vol is not None:
                self.last_rv = o.realized_vol
                break

    def generate_intents(self, sim_ts_ms: int) -> List[TradeIntent]:
        intents = []

        # 1. Check if we have data — skip deterministically if IV or RV unavailable
        if self.last_iv is None:
            if not self._logged_no_iv:
                logger.warning("VRPCreditSpreadStrategy: skipping trade — no IV available (provider atm_iv absent and derived IV failed)")
                self._logged_no_iv = True
            return []

        if self.last_rv is None:
            if not self._logged_no_rv:
                logger.warning("VRPCreditSpreadStrategy: skipping trade — no RV available (waiting for outcome engine results)")
                self._logged_no_rv = True
            return []

        # 2. Compute VRP
        vrp = self.last_iv - self.last_rv

        # 3. Check Regimes — look up by the actual strategy symbol first,
        #    fall back to the broad EQUITIES market regime.
        symbol_regime = (
            self.visible_regimes.get(self.symbol)
            or self.visible_regimes.get("EQUITIES")
            or {}
        )
        vol = symbol_regime.get("vol_regime", "UNKNOWN")
        trend = symbol_regime.get("trend_regime", "UNKNOWN")

        # 4. Get best available ATM put quote (for realistic fill pricing)
        atm_quote = _select_atm_put_quote(self._last_visible_snapshots, self.last_close)

        # 5. Gating Logic
        if self.open_position:
            # Check exit conditions
            if vrp <= 0.0 or vol == "VOL_SPIKE" or trend == "TREND_DOWN":
                intent_meta: Dict[str, Any] = {
                    "vrp": vrp,
                    "iv": self.last_iv,
                    "rv": self.last_rv,
                    "vol": vol,
                    "trend": trend,
                }
                if atm_quote is not None:
                    intent_meta["quote"] = atm_quote
                intents.append(TradeIntent(
                    symbol=self.symbol,
                    side="BUY",
                    quantity=1,
                    intent_type="CLOSE",
                    tag="VRP_EDGE_EXIT",
                    meta=intent_meta,
                ))
        else:
            # Check entry conditions
            if vrp > self._thresholds.get("min_vrp", 0.05):
                if vol != "VOL_SPIKE" and trend != "TREND_DOWN":
                    intent_meta = {
                        "vrp": vrp,
                        "iv": self.last_iv,
                        "rv": self.last_rv,
                        "vol": vol,
                        "trend": trend,
                    }
                    if atm_quote is not None:
                        intent_meta["quote"] = atm_quote
                    intents.append(TradeIntent(
                        symbol=self.symbol,
                        side="SELL",
                        quantity=1,
                        intent_type="OPEN",
                        tag="VRP_EDGE",
                        meta=intent_meta,
                    ))

        if self.last_iv is not None and not intents and not self._logged_gating:
            self._logged_gating = True
            min_vrp = self._thresholds.get("min_vrp", 0.05)
            logger.warning(
                "VRP gating: vrp=%.4f (need>%.2f) vol=%s trend=%s rv=%s",
                vrp, min_vrp, vol, trend, self.last_rv,
            )

        return intents

    def on_fill(self, intent: TradeIntent, fill: Any) -> None:
        if intent.intent_type == "OPEN":
            self.open_position = True
        elif intent.intent_type == "CLOSE":
            self.open_position = False

    def finalize(self) -> Dict[str, Any]:
        return {
            "last_iv": self.last_iv,
            "last_rv": self.last_rv,
        }
