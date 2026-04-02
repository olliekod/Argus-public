# Created by Oliver Meihls

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.iv_consensus")


@dataclass(frozen=True, slots=True)
class ContractKey:
    underlying: str
    expiration_ms: int
    option_type: str
    strike: float


@dataclass(frozen=True, slots=True)
class ATMKey:
    underlying: str
    option_type: str
    expiration_ms: int


@dataclass(frozen=True, slots=True)
class SourceObservation:
    source: str
    recv_ts_ms: int
    iv: Optional[float]
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None


@dataclass(frozen=True, slots=True)
class IVConsensusConfig:
    contract_freshness_ms: int = 15_000
    atm_freshness_ms: int = 15_000
    ring_size: int = 8
    policy: str = "prefer_public"  # prefer_dxlink | prefer_public | winner_based | blended

    winner_abs_threshold: float = 0.02
    winner_rel_threshold: float = 0.10

    warn_abs_threshold: float = 0.02
    warn_rel_threshold: float = 0.10
    bad_abs_threshold: float = 0.05
    bad_rel_threshold: float = 0.25

    source_weights: Dict[str, float] = field(default_factory=lambda: {"dxlink": 0.3, "public": 0.7})
    recency_half_life_ms: int = 7_500


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    consensus_iv: Optional[float]
    consensus_greeks: Dict[str, Optional[float]]
    iv_source_used: str
    iv_public: Optional[float]
    iv_dxlink: Optional[float]
    discrepancy_abs: Optional[float]
    discrepancy_rel: Optional[float]
    iv_quality: str
    freshness_ms: Optional[int]
    cache_hit: Dict[str, bool]


_OCC_PATTERN = re.compile(r"^\.?([A-Z]+)(\d{6})([CP])(\d+(?:\.\d+)?)$")


def normalize_iv(iv: Optional[float]) -> Optional[float]:
    # Normalize IV to decimal (e.g. 0.22 = 22%). Rejects None/NaN/<=0 or >10.
    #
    # If the raw value is > 1.5, it is interpreted as percent (e.g. 47.0 → 0.47).
    # Logged at DEBUG as normalize_iv_percent_interpreted so we don't flood WARNING.
    if iv is None:
        return None
    try:
        x = float(iv)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x <= 0:
        return None
    if x > 10.0:
        return None
    if x > 1.5:
        maybe = x / 100.0
        if maybe <= 1.5:
            logger.debug(
                "normalize_iv_percent_interpreted raw_iv=%.4f normalized=%.4f (feed sent percent, converted to decimal)",
                x, maybe,
            )
            return maybe
    return x


def parse_occ_like_symbol(event_symbol: str) -> Optional[ContractKey]:
    m = _OCC_PATTERN.match((event_symbol or "").strip())
    if not m:
        return None
    underlying, yymmdd, cp, strike = m.groups()
    from datetime import datetime, timezone

    expiration_ms = int(datetime.strptime(yymmdd, "%y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    return ContractKey(underlying=underlying, expiration_ms=expiration_ms, option_type="PUT" if cp == "P" else "CALL", strike=float(strike))


class IVConsensusEngine:
    def __init__(self, config: Optional[IVConsensusConfig] = None) -> None:
        self.cfg = config or IVConsensusConfig()
        self._contract_obs: Dict[ContractKey, Dict[str, SourceObservation]] = defaultdict(dict)
        self._atm_obs: Dict[ATMKey, Dict[str, SourceObservation]] = defaultdict(dict)
        self._last_update_ms: int = 0
        self._last_source_update_ms: Dict[str, int] = {"public": 0, "dxlink": 0}
        self._contract_hist: Dict[ContractKey, Dict[str, Deque[SourceObservation]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.cfg.ring_size)))
        self._atm_hist: Dict[ATMKey, Dict[str, Deque[SourceObservation]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.cfg.ring_size)))
        self._disc_abs: Deque[float] = deque(maxlen=10_000)
        self._disc_rel: Deque[float] = deque(maxlen=10_000)
        self._disc_total = 0
        self._disc_warn = 0
        self._disc_bad = 0

    def observe_dxlink_greeks(self, event: Any, recv_ts_ms: Optional[int] = None) -> None:
        event_symbol = getattr(event, "event_symbol", None)
        key = parse_occ_like_symbol(event_symbol or "")
        if key is None:
            return
        ts = int(recv_ts_ms if recv_ts_ms is not None else (getattr(event, "receipt_time", None) or 0))
        if ts <= 0:
            return
        obs = SourceObservation(
            source="dxlink",
            recv_ts_ms=ts,
            iv=normalize_iv(getattr(event, "volatility", None)),
            delta=getattr(event, "delta", None),
            gamma=getattr(event, "gamma", None),
            theta=getattr(event, "theta", None),
            vega=getattr(event, "vega", None),
            rho=getattr(event, "rho", None),
        )
        if obs.iv is None:
            return
        self._record_contract_obs(key, obs)
        self._record_atm_obs(ATMKey(key.underlying, key.option_type, key.expiration_ms), obs)

    def observe_public_snapshot(self, snapshot: Any, recv_ts_ms: Optional[int] = None) -> None:
        ts = int(recv_ts_ms if recv_ts_ms is not None else getattr(snapshot, "recv_ts_ms", 0))
        if ts <= 0:
            return
        symbol = getattr(snapshot, "symbol", "")
        expiration_ms = int(getattr(snapshot, "expiration_ms", 0) or 0)
        if not symbol or expiration_ms <= 0:
            return

        atm_iv = normalize_iv(getattr(snapshot, "atm_iv", None))
        if atm_iv is not None:
            for opt_type in ("PUT", "CALL"):
                atm_key = ATMKey(symbol, opt_type, expiration_ms)
                obs = SourceObservation(source="public", recv_ts_ms=ts, iv=atm_iv)
                self._record_atm_obs(atm_key, obs)

        for side_name, opt_type in (("puts", "PUT"), ("calls", "CALL")):
            for q in getattr(snapshot, side_name, ()):
                strike = float(getattr(q, "strike", 0.0) or 0.0)
                if strike <= 0:
                    continue
                iv = normalize_iv(getattr(q, "iv", None))
                if iv is None:
                    continue
                key = ContractKey(symbol, expiration_ms, opt_type, strike)
                obs = SourceObservation(
                    source="public",
                    recv_ts_ms=ts,
                    iv=iv,
                    delta=getattr(q, "delta", None),
                    gamma=getattr(q, "gamma", None),
                    theta=getattr(q, "theta", None),
                    vega=getattr(q, "vega", None),
                )
                self._record_contract_obs(key, obs)

    def get_contract_consensus(self, key: ContractKey, as_of_ms: int, *, policy: Optional[str] = None) -> ConsensusResult:
        obs = self._contract_obs.get(key, {})
        return self._consensus(obs.get("public"), obs.get("dxlink"), as_of_ms, freshness_ms=self.cfg.contract_freshness_ms, policy=(policy or self.cfg.policy), context={"scope": "contract", "underlying": key.underlying, "option_type": key.option_type, "expiry": key.expiration_ms, "strike": key.strike})

    def get_atm_consensus(self, underlying: str, option_type: str, expiration_ms: int, as_of_ms: int, *, policy: Optional[str] = None) -> ConsensusResult:
        key = ATMKey(underlying, option_type.upper(), expiration_ms)
        obs = self._atm_obs.get(key, {})
        return self._consensus(obs.get("public"), obs.get("dxlink"), as_of_ms, freshness_ms=self.cfg.atm_freshness_ms, policy=(policy or self.cfg.policy), context={"scope": "atm", "underlying": underlying, "option_type": option_type.upper(), "expiry": expiration_ms})

    def get_discrepancy_rollup(self) -> Dict[str, Any]:
        n = self._disc_total
        abs_vals = sorted(self._disc_abs)
        rel_vals = sorted(self._disc_rel)

        def _pct(vals: List[float], q: float) -> Optional[float]:
            if not vals:
                return None
            idx = int((len(vals) - 1) * q)
            return vals[idx]

        return {
            "count": n,
            "warn_count": self._disc_warn,
            "bad_count": self._disc_bad,
            "pct_warn": (self._disc_warn / n * 100.0) if n else 0.0,
            "pct_bad": (self._disc_bad / n * 100.0) if n else 0.0,
            "abs_p50": _pct(abs_vals, 0.5),
            "abs_p90": _pct(abs_vals, 0.9),
            "abs_p99": _pct(abs_vals, 0.99),
            "rel_p50": _pct(rel_vals, 0.5),
            "rel_p90": _pct(rel_vals, 0.9),
            "rel_p99": _pct(rel_vals, 0.99),
        }

    def _record_contract_obs(self, key: ContractKey, obs: SourceObservation) -> None:
        prev = self._contract_obs[key].get(obs.source)
        if prev is None or obs.recv_ts_ms >= prev.recv_ts_ms:
            self._contract_obs[key][obs.source] = obs
            self._contract_hist[key][obs.source].append(obs)

    def _record_atm_obs(self, key: ATMKey, obs: SourceObservation) -> None:
        prev = self._atm_obs[key].get(obs.source)
        if prev is None or obs.recv_ts_ms >= prev.recv_ts_ms:
            self._atm_obs[key][obs.source] = obs
            self._atm_hist[key][obs.source].append(obs)
            
            # Update activity tracking
            ts = obs.recv_ts_ms
            if ts > self._last_update_ms:
                self._last_update_ms = ts
            if ts > self._last_source_update_ms.get(obs.source, 0):
                self._last_source_update_ms[obs.source] = ts

    @property
    def last_update_ms(self) -> int:
        # Midnight-UTC epoch ms of the most recent observation.
        return self._last_update_ms

    @property
    def last_source_update_ms(self) -> Dict[str, int]:
        # Epoch ms of the most recent observation per source.
        return dict(self._last_source_update_ms)

    @property
    def size(self) -> int:
        # Total number of unique contract observations.
        return len(self._contract_obs)

    def _usable(self, obs: Optional[SourceObservation], as_of_ms: int, freshness_ms: int) -> bool:
        return bool(obs is not None and obs.iv is not None and obs.recv_ts_ms <= as_of_ms and (as_of_ms - obs.recv_ts_ms) <= freshness_ms)

    def _consensus(self, public: Optional[SourceObservation], dxlink: Optional[SourceObservation], as_of_ms: int, *, freshness_ms: int, policy: str, context: Dict[str, Any]) -> ConsensusResult:
        public_ok = self._usable(public, as_of_ms, freshness_ms)
        dx_ok = self._usable(dxlink, as_of_ms, freshness_ms)
        iv_public = public.iv if public_ok and public else None
        iv_dx = dxlink.iv if dx_ok and dxlink else None

        discrepancy_abs = discrepancy_rel = None
        quality = "none"
        if iv_public is not None and iv_dx is not None:
            eps = 1e-9
            discrepancy_abs = abs(iv_public - iv_dx)
            discrepancy_rel = discrepancy_abs / max(iv_dx, eps)
            self._disc_total += 1
            self._disc_abs.append(discrepancy_abs)
            self._disc_rel.append(discrepancy_rel)
            if discrepancy_abs > self.cfg.bad_abs_threshold or discrepancy_rel > self.cfg.bad_rel_threshold:
                quality = "bad"
                self._disc_bad += 1
            elif discrepancy_abs > self.cfg.warn_abs_threshold or discrepancy_rel > self.cfg.warn_rel_threshold:
                quality = "warn"
                self._disc_warn += 1
            else:
                quality = "ok"

        iv = None
        source = "none"
        if policy == "prefer_dxlink":
            if iv_dx is not None:
                iv, source = iv_dx, "dxlink"
            elif iv_public is not None:
                iv, source = iv_public, "public"
        elif policy == "prefer_public":
            if iv_public is not None:
                iv, source = iv_public, "public"
            elif iv_dx is not None:
                iv, source = iv_dx, "dxlink"
        elif policy == "winner_based":
            if iv_public is not None:
                iv, source = iv_public, "public"
            if iv_public is not None and iv_dx is not None and discrepancy_abs is not None and discrepancy_rel is not None:
                if discrepancy_abs > self.cfg.winner_abs_threshold or discrepancy_rel > self.cfg.winner_rel_threshold:
                    iv, source = iv_dx, "dxlink"
            elif iv is None and iv_dx is not None:
                iv, source = iv_dx, "dxlink"
        elif policy == "blended":
            parts: List[Tuple[float, SourceObservation]] = []
            if iv_dx is not None and dxlink:
                parts.append((self.cfg.source_weights.get("dxlink", 0.5), dxlink))
            if iv_public is not None and public:
                parts.append((self.cfg.source_weights.get("public", 0.5), public))
            if parts:
                weighted = []
                for base_w, o in parts:
                    age = max(as_of_ms - o.recv_ts_ms, 0)
                    rec = 0.5 ** (age / max(self.cfg.recency_half_life_ms, 1))
                    weighted.append((base_w * rec, o.iv or 0.0))
                denom = sum(w for w, _ in weighted)
                if denom > 0:
                    iv = sum(w * val for w, val in weighted) / denom
                    source = "blended"
        else:
            raise ValueError(f"Unknown IV consensus policy: {policy}")

        greeks = {}
        if source == "dxlink" and dxlink:
            greeks = {"delta": dxlink.delta, "gamma": dxlink.gamma, "theta": dxlink.theta, "vega": dxlink.vega, "rho": dxlink.rho}
        elif source == "public" and public:
            greeks = {"delta": public.delta, "gamma": public.gamma, "theta": public.theta, "vega": public.vega, "rho": public.rho}
        elif source == "blended":
            greeks = {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}

        freshest_ts = max([o.recv_ts_ms for o in (public, dxlink) if o is not None], default=None)
        freshness_value = (as_of_ms - freshest_ts) if freshest_ts is not None else None

        if discrepancy_abs is not None and discrepancy_rel is not None and quality in {"warn", "bad"}:
            payload = {
                "timestamp": as_of_ms,
                **context,
                "iv_public": iv_public,
                "iv_dxlink": iv_dx,
                "abs_diff": discrepancy_abs,
                "rel_diff": discrepancy_rel,
                "selected_source": source,
                "policy": policy,
                "freshness_ms": freshness_value,
                "quality": quality,
            }
            logger.warning("iv_discrepancy %s", json.dumps(payload, sort_keys=True))

        return ConsensusResult(
            consensus_iv=iv,
            consensus_greeks=greeks,
            iv_source_used=source,
            iv_public=iv_public,
            iv_dxlink=iv_dx,
            discrepancy_abs=discrepancy_abs,
            discrepancy_rel=discrepancy_rel,
            iv_quality=quality,
            freshness_ms=freshness_value,
            cache_hit={"public": public_ok, "dxlink": dx_ok},
        )
