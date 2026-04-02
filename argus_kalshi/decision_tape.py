# Created by Oliver Meihls

from __future__ import annotations

import json
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .context_policy import build_context_key, momentum_bucket
from .decision_context import _strike_distance_bucket
from .simulation import assign_family

TAPE_VERSION = 1
_SENTINEL = object()

# High-volume structural rejections that are not parameter-sensitive.
# Recording these dominates tape size without improving offline evaluation quality.
# too_early_to_expiry:           ~18M/30s (pure time gate)
# scalp_entry_price_out_of_range: ~18M/30s (fixed price bounds, not swept)
# arb_*:                         ~20M/30s (arb market structure)
# truth_or_ws_stale:             ~10M/30s (environment noise)
# *_family_filtered:             high-volume, family is already in ctx_key
_SKIP_REJECT_REASONS: frozenset[str] = frozenset({
    "too_early_to_expiry",
    "range_too_far_to_expiry",
    "scalp_entry_price_out_of_range",
    "arb_yes_entry_price_out_of_range",
    "arb_no_entry_price_out_of_range",
    "arb_no_cross",
    "hold_family_filtered",
    "scalp_family_filtered",
    "truth_or_ws_stale",
})


def _normalize_side(side: Optional[str]) -> Optional[str]:
    if side is None:
        return None
    text = str(side).strip().lower()
    return text if text in {"yes", "no"} else None


def _coerce_price(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _compute_ctx_key(prepared: Any, side: Optional[str], edge: float) -> Optional[str]:
    norm_side = _normalize_side(side)
    if norm_side is None:
        return None
    family = assign_family(
        ticker=str(getattr(prepared, "ticker", "")),
        asset=str(getattr(prepared, "asset", "")),
        window_minutes=int(getattr(prepared, "window_minutes", 0) or 0),
        is_range=bool(getattr(prepared, "is_range", False)),
    )
    limit_cents = (
        _coerce_price(getattr(prepared, "yes_ask_cents", 0))
        if norm_side == "yes"
        else _coerce_price(getattr(prepared, "no_ask_cents", 0))
    )
    strike_distance_pct = getattr(prepared, "strike_distance_pct", None)
    strike_bucket = _strike_distance_bucket(strike_distance_pct, [0.005, 0.01, 0.02, 0.04])
    near_money = bool(strike_distance_pct is not None and float(strike_distance_pct) <= 0.08)
    price_bucket = "lt_40"
    if limit_cents >= 78:
        price_bucket = "ge_78"
    elif limit_cents >= 70:
        price_bucket = "70_78"
    elif limit_cents >= 55:
        price_bucket = "55_70"
    elif limit_cents >= 40:
        price_bucket = "40_55"
    edge_bucket = "lt_0.05"
    if edge >= 0.20:
        edge_bucket = "ge_0.20"
    elif edge >= 0.10:
        edge_bucket = "0.10_0.20"
    elif edge >= 0.05:
        edge_bucket = "0.05_0.10"
    return build_context_key(
        family=family,
        side=norm_side,
        edge_bucket=edge_bucket,
        price_bucket=price_bucket,
        strike_distance_bucket=strike_bucket,
        near_money=near_money,
        momentum=momentum_bucket(float(getattr(prepared, "momentum_drift", 0.0) or 0.0)),
    )


def build_tape_record(
    prepared: Any,
    source: str,
    decision: str,
    side: Optional[str],
    edge: float,
    reject_reason: Optional[str],
    params: Optional[Mapping[str, Any]],
    ctx_key: Optional[str],
    sampled_rejection: bool = False,
    quantity_contracts: Optional[int] = None,
) -> Dict[str, Any]:
    settlement_epoch = float(
        getattr(
            prepared,
            "settlement_epoch",
            (float(getattr(prepared, "now_wall", 0.0) or 0.0) + float(getattr(prepared, "time_to_settle_s", 0.0) or 0.0)),
        )
    )
    family = assign_family(
        ticker=str(getattr(prepared, "ticker", "")),
        asset=str(getattr(prepared, "asset", "")),
        window_minutes=int(getattr(prepared, "window_minutes", 0) or 0),
        is_range=bool(getattr(prepared, "is_range", False)),
    )
    normalized_side = _normalize_side(side)
    record: Dict[str, Any] = {
        "tape_v": TAPE_VERSION,
        "ts": float(getattr(prepared, "now_wall", time.time()) or time.time()),
        "ticker": str(getattr(prepared, "ticker", "")),
        "source": str(source),
        "decision": str(decision),
        "p_yes": float(getattr(prepared, "p_yes", 0.0) or 0.0),
        "yes_ask_cents": _coerce_price(getattr(prepared, "yes_ask_cents", 0)),
        "no_ask_cents": _coerce_price(getattr(prepared, "no_ask_cents", 0)),
        "yes_bid_cents": _coerce_price(getattr(prepared, "yes_bid_cents", 0)),
        "no_bid_cents": _coerce_price(getattr(prepared, "no_bid_cents", 0)),
        "ev_yes": float(getattr(prepared, "ev_yes", 0.0) or 0.0),
        "ev_no": float(getattr(prepared, "ev_no", 0.0) or 0.0),
        "time_to_settle_s": float(getattr(prepared, "time_to_settle_s", 0.0) or 0.0),
        "momentum_drift": float(getattr(prepared, "momentum_drift", 0.0) or 0.0),
        "trade_flow": float(getattr(prepared, "trade_flow", 0.0) or 0.0),
        "obi": float(getattr(prepared, "obi", 0.0) or 0.0),
        "depth_pressure": float(getattr(prepared, "depth_pressure", 0.0) or 0.0),
        "delta_flow_yes": float(getattr(prepared, "delta_flow_yes", 0.0) or 0.0),
        "delta_flow_no": float(getattr(prepared, "delta_flow_no", 0.0) or 0.0),
        "asset": str(getattr(prepared, "asset", "") or "").upper(),
        "family": family,
        "window_minutes": int(getattr(prepared, "window_minutes", 0) or 0),
        "is_range": bool(getattr(prepared, "is_range", False)),
        "settlement_epoch": settlement_epoch if settlement_epoch > 0 else 0.0,
        "side": normalized_side,
        "edge": float(edge or 0.0),
        "reject_reason": str(reject_reason) if reject_reason else None,
        "params": dict(params) if params is not None else None,
        "ctx_key": ctx_key or _compute_ctx_key(prepared, normalized_side, float(edge or 0.0)),
        "sampled_rejection": bool(sampled_rejection),
        "strike_distance_pct": getattr(prepared, "strike_distance_pct", None),
    }
    if quantity_contracts is not None:
        record["quantity_contracts"] = int(quantity_contracts)
    return record


class DecisionTapeWriter:
    # Background JSONL writer for decision-tape records.

    def __init__(
        self,
        path: str,
        signal_sample_rate: float = 1.0,
        rejection_sample_rate: float = 0.10,
        queue_maxsize: int = 50000,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._signal_sample_rate = max(0.0, min(1.0, float(signal_sample_rate)))
        self._rejection_sample_rate = max(0.0, min(1.0, float(rejection_sample_rate)))
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max(1, int(queue_maxsize)))
        self._file = self._path.open("a", encoding="utf-8")
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain_loop, name="decision_tape_writer", daemon=True)
        self._thread.start()

    def _sample(self, rate: float) -> bool:
        return rate >= 1.0 or (rate > 0.0 and random.random() <= rate)

    def write_signal(self, record: Dict[str, Any]) -> None:
        if not self._sample(self._signal_sample_rate):
            return
        with self._lock:
            if self._closed:
                return
        self._queue.put(record)

    def write_rejection(self, record: Dict[str, Any]) -> None:
        if record.get("reject_reason") in _SKIP_REJECT_REASONS:
            return
        if not self._sample(self._rejection_sample_rate):
            return
        with self._lock:
            if self._closed:
                return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            return

    def flush_and_close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=10.0)
        try:
            self._file.flush()
        finally:
            self._file.close()

    def _drain_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    break
                self._file.write(json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n")
            finally:
                self._queue.task_done()
        self._file.flush()
