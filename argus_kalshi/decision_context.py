# Created by Oliver Meihls

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import MarketMetadata, OrderbookState, TradeSignal


def _edge_bucket(edge: float) -> str:
    if edge < 0.05:
        return "lt_0.05"
    if edge < 0.10:
        return "0.05_0.10"
    if edge < 0.20:
        return "0.10_0.20"
    return "ge_0.20"


def _price_bucket(price_cents: int) -> str:
    if price_cents < 40:
        return "lt_40"
    if price_cents < 55:
        return "40_55"
    if price_cents < 70:
        return "55_70"
    if price_cents < 78:
        return "70_78"
    return "ge_78"


def _tts_bucket(tts_s: Optional[float]) -> str:
    if tts_s is None:
        return "unknown"
    if tts_s < 60:
        return "lt_1m"
    if tts_s < 180:
        return "1_3m"
    if tts_s < 600:
        return "3_10m"
    if tts_s < 1800:
        return "10_30m"
    return "ge_30m"


def _strike_distance_pct(
    spot: float,
    metadata: Optional[MarketMetadata],
) -> Optional[float]:
    # Compute abs(spot - strike) / spot for binary, or midpoint for range.
    if metadata is None or spot <= 0:
        return None
    if metadata.is_range and metadata.strike_floor is not None and metadata.strike_cap is not None:
        midpoint = (metadata.strike_floor + metadata.strike_cap) / 2.0
        return abs(spot - midpoint) / spot
    if metadata.strike_price > 0:
        return abs(spot - metadata.strike_price) / spot
    return None


def _strike_distance_bucket(
    sd_pct: Optional[float],
    edges: Optional[List[float]] = None,
) -> str:
    # Bucket strike distance by configurable edges. Default: [0.5%, 1%, 2%, 5%].
    if sd_pct is None:
        return "na"
    if edges is None:
        edges = [0.005, 0.01, 0.02, 0.05]
    for i, edge in enumerate(edges):
        if sd_pct < edge:
            if i == 0:
                return f"lt_{edge}"
            return f"{edges[i-1]}_{edge}"
    return f"ge_{edges[-1]}"


def _spread_bucket(spread_cents: Optional[int]) -> str:
    if spread_cents is None:
        return "na"
    if spread_cents <= 1:
        return "tight"
    if spread_cents <= 3:
        return "normal"
    if spread_cents <= 6:
        return "wide"
    return "very_wide"


def _liq_bucket(depth_contracts: Optional[int]) -> str:
    if depth_contracts is None:
        return "na"
    if depth_contracts >= 50:
        return "deep"
    if depth_contracts >= 20:
        return "normal"
    if depth_contracts >= 5:
        return "thin"
    return "dry"


def build_decision_context(
    signal: TradeSignal,
    *,
    family: str,
    source: str,
    profile_name: str,
    now_ts: float,
    orderbook: Optional[OrderbookState] = None,
    metadata: Optional[MarketMetadata] = None,
    spot_price: float = 0.0,
    near_money_pct: float = 0.08,
    strike_distance_bucket_edges: Optional[List[float]] = None,
    drift: float = 0.0,
    flow: float = 0.0,
) -> Dict[str, Any]:
    # Build a compact decision context payload for paper logs.
    #
    # Design goals:
    # - small: compact keys, rounded numeric values
    # - stable: explicit version field for future migrations
    # - safe: avoid large nested payloads
    spread_cents: Optional[int] = None
    depth_contracts: Optional[int] = None
    obi: Optional[float] = None
    micro_price: Optional[float] = None
    if orderbook and orderbook.valid:
        spread_cents = max(0, int(orderbook.implied_yes_ask_cents + orderbook.implied_no_ask_cents - 100))
        depth_centicx = orderbook.best_no_depth if signal.side == "yes" else orderbook.best_yes_depth
        depth_contracts = max(0, depth_centicx // 100)
        obi = round(float(orderbook.obi), 4)
        micro_price = round(float(orderbook.micro_price_cents), 2)

    tts_s: Optional[float] = None
    if metadata is not None:
        try:
            settlement_iso = (metadata.settlement_time_iso or "").replace("Z", "+00:00")
            from datetime import datetime

            settle_ts = datetime.fromisoformat(settlement_iso).timestamp()
            tts_s = max(0.0, settle_ts - now_ts)
        except Exception:
            tts_s = None

    # Strike distance and near-money classification
    sd_pct = _strike_distance_pct(spot_price, metadata)
    sd_bucket = _strike_distance_bucket(sd_pct, strike_distance_bucket_edges)
    near_money = sd_pct is not None and sd_pct <= near_money_pct

    side_code = "y" if signal.side == "yes" else "n"
    style_code = "p" if signal.order_style == "passive" else "a"
    source_code = "scalp" if source == "mispricing_scalp" else ("strategy" if source else "unknown")

    ctx: Dict[str, Any] = {
        "v": 2,
        "fam": family,
        "sd": side_code,
        "src": source_code,
        "sty": style_code,
        "scn": profile_name,
        "qty": int(signal.quantity_contracts),
        "px": int(signal.limit_price_cents),
        "py": round(float(signal.p_yes), 4),
        "edge": round(float(signal.edge), 4),
        "eb": _edge_bucket(float(signal.edge)),
        "pb": _price_bucket(int(signal.limit_price_cents)),
        "tts": int(tts_s) if tts_s is not None else None,
        "tb": _tts_bucket(tts_s),
        "sp": spread_cents,
        "d": depth_contracts,
        "obi": obi,
        "mp": micro_price,
        # New v2 fields
        "sdp": round(sd_pct, 4) if sd_pct is not None else None,
        "sdb": sd_bucket,
        "nm": near_money,
        "spb": _spread_bucket(spread_cents),
        "lb": _liq_bucket(depth_contracts),
        # Directional signals (Task 7: momentum drift + trade flow)
        "drift": round(drift, 6) if drift != 0.0 else None,
        "flow": round(flow, 4) if flow != 0.0 else None,
        # Momentum regime at entry time — stored so apply_promotion can include
        # it in the context key. Direction-agnostic: "up"/"dn"/"flat".
        "mb": ("up" if drift > 1e-4 else ("dn" if drift < -1e-4 else "flat")),
    }

    # Strip null-ish optional fields for compact JSONL footprint.
    return {k: v for k, v in ctx.items() if v is not None}
