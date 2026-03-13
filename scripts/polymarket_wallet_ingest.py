#!/usr/bin/env python3
"""
Ingest Polymarket wallet data for strategy analysis.

This script pulls read-only wallet data from Polymarket Data API endpoints,
stores immutable raw payloads, and populates normalized analytics tables.

Usage:
  python scripts/polymarket_wallet_ingest.py --wallet 0xabc...
  python scripts/polymarket_wallet_ingest.py --wallet 0xabc --wallet 0xdef --max-pages 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DATA_API_BASE = "https://data-api.polymarket.com"
DEFAULT_DB = "data/polymarket_wallets.db"
DEFAULT_ENDPOINTS = ("trades", "activity", "positions", "closed-positions", "value")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pick(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _best_timestamp_str(obj: Dict[str, Any]) -> str:
    ts = _pick(
        obj,
        "timestamp",
        "time",
        "createdAt",
        "created_at",
        "updatedAt",
        "updated_at",
        "lastUpdatedAt",
        "last_updated_at",
    )
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            return _utc_now_iso()
    if isinstance(ts, str) and ts.strip():
        return ts
    return _utc_now_iso()


def _fetch_json(url: str, timeout_s: float = 25.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            # Data API often rejects generic clients without a browser-like UA.
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Origin": "https://polymarket.com",
            "Referer": "https://polymarket.com/",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # nosec B310
        raw = resp.read()
    return json.loads(raw)


def _paginate_endpoint(
    wallet: str,
    endpoint: str,
    *,
    max_pages: int,
    page_size: int,
    sleep_ms: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    cursor: Optional[str] = None
    # Different endpoints use different wallet keys/pagination styles.
    if endpoint in ("positions", "closed-positions", "value"):
        wallet_keys = ("user", "proxyWallet")
    elif endpoint == "trades":
        wallet_keys = ("user", "proxyWallet", "maker", "taker", "address")
    elif endpoint == "activity":
        wallet_keys = ("user", "proxyWallet", "wallet", "address")
    else:
        wallet_keys = ("user", "proxyWallet")

    pagination_styles = ("offset", "cursor", "none", "minimal")
    active_style = pagination_styles[0]

    def _extract_rows_and_cursor(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)], None
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                next_cursor = _pick(
                    payload,
                    "next_cursor",
                    "nextCursor",
                    "cursor",
                    "next",
                    default=None,
                )
                return [r for r in payload.get("data", []) if isinstance(r, dict)], (
                    str(next_cursor) if next_cursor else None
                )
            return [payload], None
        return [], None

    for _ in range(max_pages):
        payload = None
        last_exc: Optional[Exception] = None
        for wkey in wallet_keys:
            for style in pagination_styles:
                params: Dict[str, Any] = {wkey: wallet}
                if style == "offset":
                    params["limit"] = page_size
                    params["offset"] = offset
                elif style == "cursor" and cursor:
                    params["limit"] = page_size
                    params["cursor"] = cursor
                elif style == "none":
                    params["limit"] = page_size
                url = f"{DATA_API_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
                try:
                    payload = _fetch_json(url)
                    active_style = style
                    last_exc = None
                    break
                except urllib.error.HTTPError as exc:
                    # 4xx likely means wrong query shape for this endpoint.
                    last_exc = exc
                    continue
            if payload is not None:
                break
        if payload is None:
            if last_exc is not None:
                raise last_exc
            break

        page_rows, next_cursor = _extract_rows_and_cursor(payload)
        if not page_rows:
            break
        rows.extend(page_rows)
        if len(page_rows) < page_size:
            break
        if active_style == "offset":
            offset += page_size
        elif active_style == "cursor":
            if not next_cursor:
                break
            cursor = next_cursor
        else:
            # No reliable pagination key for this endpoint shape.
            break
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
    return rows


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS poly_ingest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            wallets_json TEXT NOT NULL,
            endpoints_json TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS poly_wallet_endpoint_raw (
            wallet TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            event_ts TEXT,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (wallet, endpoint, row_hash)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS poly_wallet_trades (
            wallet TEXT NOT NULL,
            trade_uid TEXT NOT NULL,
            ts TEXT NOT NULL,
            market_slug TEXT,
            market_id TEXT,
            token_id TEXT,
            side TEXT,
            price REAL,
            size REAL,
            notional_usd REAL,
            role TEXT,
            tx_hash TEXT,
            order_id TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (wallet, trade_uid)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS poly_wallet_activity (
            wallet TEXT NOT NULL,
            activity_uid TEXT NOT NULL,
            ts TEXT NOT NULL,
            activity_type TEXT,
            market_slug TEXT,
            token_id TEXT,
            side TEXT,
            price REAL,
            size REAL,
            notional_usd REAL,
            tx_hash TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (wallet, activity_uid)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS poly_wallet_positions (
            wallet TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL,
            market_slug TEXT,
            market_id TEXT,
            token_id TEXT,
            outcome TEXT,
            avg_price REAL,
            size REAL,
            notional_usd REAL,
            unrealized_pnl REAL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (wallet, snapshot_ts, market_id, token_id, outcome)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS poly_wallet_closed_positions (
            wallet TEXT NOT NULL,
            closed_uid TEXT NOT NULL,
            ts TEXT NOT NULL,
            market_slug TEXT,
            market_id TEXT,
            token_id TEXT,
            outcome TEXT,
            size REAL,
            avg_entry REAL,
            avg_exit REAL,
            realized_pnl REAL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (wallet, closed_uid)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS poly_wallet_value (
            wallet TEXT NOT NULL,
            ts TEXT NOT NULL,
            portfolio_value REAL,
            cash_value REAL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (wallet, ts)
        )
        """
    )
    conn.commit()


def _trade_uid(row: Dict[str, Any], row_hash: str) -> str:
    return str(
        _pick(
            row,
            "id",
            "tradeID",
            "tradeId",
            "trade_id",
            "matchID",
            "matchId",
            "txHash",
            "transactionHash",
            default=row_hash,
        )
    )


def _activity_uid(row: Dict[str, Any], row_hash: str) -> str:
    return str(
        _pick(
            row,
            "id",
            "activityID",
            "activityId",
            "activity_id",
            "txHash",
            "transactionHash",
            default=row_hash,
        )
    )


def _closed_uid(row: Dict[str, Any], row_hash: str) -> str:
    return str(_pick(row, "id", "positionId", "position_id", default=row_hash))


def _extract_market_slug(row: Dict[str, Any]) -> Optional[str]:
    return _pick(row, "marketSlug", "market_slug", "slug", "questionSlug", "question_slug")


def _extract_market_id(row: Dict[str, Any]) -> Optional[str]:
    return _pick(row, "market", "marketId", "market_id", "conditionId", "condition_id")


def _extract_token_id(row: Dict[str, Any]) -> Optional[str]:
    return _pick(row, "asset", "tokenID", "tokenId", "token_id")


def _upsert_normalized(conn: sqlite3.Connection, wallet: str, endpoint: str, rows: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
    for row_hash, row in rows:
        ts = _best_timestamp_str(row)
        raw_json = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if endpoint == "trades":
            side = str(_pick(row, "side", "takerSide", "makerSide", default="")).lower() or None
            price = _to_float(_pick(row, "price", "tradePrice", "trade_price", default=None), default=0.0)
            size = _to_float(_pick(row, "size", "quantity", "amount", "shares", default=None), default=0.0)
            notional = _to_float(_pick(row, "notional", "value", "amountUsd", "amount_usd", default=price * size))
            tx_hash = _pick(row, "txHash", "transactionHash")
            order_id = _pick(row, "orderID", "orderId", "order_id")
            maker = str(_pick(row, "makerAddress", "maker", default="")).lower()
            taker = str(_pick(row, "takerAddress", "taker", default="")).lower()
            role = None
            wl = wallet.lower()
            if wl and wl == maker:
                role = "maker"
            elif wl and wl == taker:
                role = "taker"
            conn.execute(
                """
                INSERT OR REPLACE INTO poly_wallet_trades (
                    wallet, trade_uid, ts, market_slug, market_id, token_id, side, price, size,
                    notional_usd, role, tx_hash, order_id, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wallet,
                    _trade_uid(row, row_hash),
                    ts,
                    _extract_market_slug(row),
                    _extract_market_id(row),
                    _extract_token_id(row),
                    side,
                    price,
                    size,
                    notional,
                    role,
                    tx_hash,
                    order_id,
                    raw_json,
                ),
            )
        elif endpoint == "activity":
            side = str(_pick(row, "side", default="")).lower() or None
            price = _to_float(_pick(row, "price", default=None), default=0.0)
            size = _to_float(_pick(row, "size", "quantity", default=None), default=0.0)
            notional = _to_float(_pick(row, "notional", "amountUsd", "amount_usd", default=price * size))
            conn.execute(
                """
                INSERT OR REPLACE INTO poly_wallet_activity (
                    wallet, activity_uid, ts, activity_type, market_slug, token_id, side, price,
                    size, notional_usd, tx_hash, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wallet,
                    _activity_uid(row, row_hash),
                    ts,
                    _pick(row, "type", "activityType", "activity_type"),
                    _extract_market_slug(row),
                    _extract_token_id(row),
                    side,
                    price,
                    size,
                    notional,
                    _pick(row, "txHash", "transactionHash"),
                    raw_json,
                ),
            )
        elif endpoint == "positions":
            conn.execute(
                """
                INSERT OR REPLACE INTO poly_wallet_positions (
                    wallet, snapshot_ts, market_slug, market_id, token_id, outcome, avg_price, size,
                    notional_usd, unrealized_pnl, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wallet,
                    ts,
                    _extract_market_slug(row),
                    _extract_market_id(row),
                    _extract_token_id(row),
                    _pick(row, "outcome", "position", "name"),
                    _to_float(_pick(row, "avgPrice", "avg_price", "averagePrice")),
                    _to_float(_pick(row, "size", "quantity", "amount")),
                    _to_float(_pick(row, "value", "notional", "amountUsd", "amount_usd")),
                    _to_float(_pick(row, "unrealizedPnl", "unrealized_pnl")),
                    raw_json,
                ),
            )
        elif endpoint == "closed-positions":
            conn.execute(
                """
                INSERT OR REPLACE INTO poly_wallet_closed_positions (
                    wallet, closed_uid, ts, market_slug, market_id, token_id, outcome, size, avg_entry,
                    avg_exit, realized_pnl, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wallet,
                    _closed_uid(row, row_hash),
                    ts,
                    _extract_market_slug(row),
                    _extract_market_id(row),
                    _extract_token_id(row),
                    _pick(row, "outcome", "position", "name"),
                    _to_float(_pick(row, "size", "quantity", "amount")),
                    _to_float(_pick(row, "avgEntry", "avg_entry", "entryPrice")),
                    _to_float(_pick(row, "avgExit", "avg_exit", "exitPrice")),
                    _to_float(_pick(row, "realizedPnl", "realized_pnl", "pnl")),
                    raw_json,
                ),
            )
        elif endpoint == "value":
            conn.execute(
                """
                INSERT OR REPLACE INTO poly_wallet_value (
                    wallet, ts, portfolio_value, cash_value, raw_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    wallet,
                    ts,
                    _to_float(_pick(row, "portfolioValue", "portfolio_value", "value")),
                    _to_float(_pick(row, "cashValue", "cash_value", "cash")),
                    raw_json,
                ),
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Polymarket wallet data for strategy analysis.")
    ap.add_argument("--wallet", action="append", required=True, help="Wallet address (repeatable).")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite output path (default: {DEFAULT_DB})")
    ap.add_argument(
        "--endpoints",
        default=",".join(DEFAULT_ENDPOINTS),
        help=f"Comma-separated endpoints (default: {','.join(DEFAULT_ENDPOINTS)})",
    )
    ap.add_argument("--max-pages", type=int, default=100, help="Max pages per endpoint")
    ap.add_argument("--page-size", type=int, default=200, help="Rows per page")
    ap.add_argument("--sleep-ms", type=int, default=50, help="Pause between pages (ms)")
    args = ap.parse_args()

    wallets = sorted({w.strip().lower() for w in args.wallet if w.strip()})
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        _init_db(conn)
        run_id = conn.execute(
            """
            INSERT INTO poly_ingest_runs (started_at, wallets_json, endpoints_json, status)
            VALUES (?, ?, ?, ?)
            """,
            (_utc_now_iso(), json.dumps(wallets), json.dumps(endpoints), "running"),
        ).lastrowid
        conn.commit()
        total_raw = 0
        endpoint_errors: List[Dict[str, str]] = []
        for wallet in wallets:
            for endpoint in endpoints:
                try:
                    rows = _paginate_endpoint(
                        wallet,
                        endpoint,
                        max_pages=args.max_pages,
                        page_size=args.page_size,
                        sleep_ms=args.sleep_ms,
                    )
                except Exception as exc:
                    endpoint_errors.append(
                        {"wallet": wallet, "endpoint": endpoint, "error": str(exc)[:300]}
                    )
                    print(f"[ingest] wallet={wallet} endpoint={endpoint} ERROR={exc}")
                    continue
                raw_rows: List[Tuple[str, Dict[str, Any]]] = []
                for row in rows:
                    row_hash = _json_hash(row)
                    raw_rows.append((row_hash, row))
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO poly_wallet_endpoint_raw (
                            wallet, endpoint, row_hash, ingested_at, event_ts, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            wallet,
                            endpoint,
                            row_hash,
                            _utc_now_iso(),
                            _best_timestamp_str(row),
                            json.dumps(row, sort_keys=True, separators=(",", ":")),
                        ),
                    )
                _upsert_normalized(conn, wallet, endpoint, raw_rows)
                conn.commit()
                total_raw += len(raw_rows)
                print(f"[ingest] wallet={wallet} endpoint={endpoint} rows={len(raw_rows)}")
        conn.execute(
            "UPDATE poly_ingest_runs SET finished_at = ?, status = ? WHERE id = ?",
            (_utc_now_iso(), "ok_with_errors" if endpoint_errors else "ok", run_id),
        )
        conn.commit()
        if endpoint_errors:
            print("[ingest] completed with endpoint errors:")
            for err in endpoint_errors:
                print(
                    f"  wallet={err['wallet']} endpoint={err['endpoint']} error={err['error']}"
                )
        print(f"[ingest] done wallets={len(wallets)} endpoints={len(endpoints)} raw_rows={total_raw} db={db_path}")
    except Exception as exc:
        try:
            conn.execute(
                "UPDATE poly_ingest_runs SET finished_at = ?, status = ?, error = ? WHERE id = (SELECT MAX(id) FROM poly_ingest_runs)",
                (_utc_now_iso(), "error", str(exc)[:1000]),
            )
            conn.commit()
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
