"""
Daily Replay Pack Tool
=======================

Slices market_bars, bar_outcomes, regimes, and option chain snapshots
from the database for a specific symbol and time window, and saves them
to a JSON file for deterministic offline replay.

Supports:
- Single-symbol mode: ``--symbol SPY``
- Universe mode: ``--universe`` (loads all liquid ETF symbols)

Provider defaults are read from the ``data_sources`` policy in
``config/config.yaml``.  No ``--provider`` flag is required for
normal usage.  Advanced overrides (``--bars-provider``,
``--options-snapshot-provider``) are available but rarely needed.

Option chain snapshots are included when available.  Symbols without
options data simply produce packs with an empty ``snapshots`` list.
"""

import asyncio
import json
import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.database import Database
from src.core.data_sources import get_data_source_policy, DataSourcePolicy
from src.core.liquid_etf_universe import get_liquid_etf_universe

logger = logging.getLogger("argus.replay_pack")
from src.core.global_risk_flow import (
    compute_global_risk_flow,
    ASIA_SYMBOLS,
    EUROPE_SYMBOLS,
    FX_RISK_SYMBOL,
)


def get_news_sentiment_for_replay(sim_ts_ms: int) -> Dict[str, Any]:
    """Deterministic replay helper for news sentiment.

    Current phase uses a constant stub payload. This function is the
    extension point for future historical sentiment lookup keyed by
    ``sim_ts_ms``.
    """
    _ = sim_ts_ms
    return {"score": 0.0, "label": "stub", "n_headlines": 0}



def _atm_iv_from_quotes_json(quotes_json: str, underlying_price: float) -> Optional[float]:
    """Fill ATM put IV from quotes_json when snapshot has no atm_iv.

    Provider IV is always preferred; derived IV is used only when provider
    did not supply it. Order: (1) top-level atm_iv (from connector/provider),
    (2) ATM put's iv field (provider on the quote), (3) derived from bid/ask
    via GreeksEngine (Black-Scholes) when neither is present.

    When underlying_price is 0 or missing, tries to use underlying_price from
    the parsed quotes_json (serialized chain includes it).
    """
    if not quotes_json:
        logger.debug("atm_iv_from_quotes: empty quotes_json")
        return None
    try:
        data = json.loads(quotes_json)
        if not isinstance(data, dict):
            logger.debug("atm_iv_from_quotes: parsed json not a dict")
            return None
        # Resolve underlying_price: use arg, else from serialized chain
        underlying = float(underlying_price or 0)
        if underlying <= 0:
            raw = data.get("underlying_price")
            if raw is not None:
                try:
                    underlying = float(raw)
                except (TypeError, ValueError):
                    pass
        if underlying <= 0:
            logger.debug("atm_iv_from_quotes: underlying_price missing or 0 (arg=%s)", underlying_price)
            return None
        # Top-level atm_iv from serialized OptionChainSnapshotEvent
        top = data.get("atm_iv")
        if top is not None and top != "" and float(top) > 0:
            return float(top)
        puts = data.get("puts") or []
        if not puts:
            logger.debug("atm_iv_from_quotes: no puts in chain")
            return None
        timestamp_ms = int(data.get("timestamp_ms") or 0)
        expiration_ms = int(data.get("expiration_ms") or 0)
        T_years = 0.0
        if timestamp_ms and expiration_ms and expiration_ms > timestamp_ms:
            T_years = (expiration_ms - timestamp_ms) / (1000.0 * 365.25 * 24 * 3600)
        # Find ATM put (strike closest to underlying)
        best_put: Optional[Dict[str, Any]] = None
        best_dist = float("inf")
        for q in puts:
            if not isinstance(q, dict):
                continue
            strike = q.get("strike")
            if strike is None:
                continue
            try:
                s = float(strike)
                dist = abs(s - underlying)
                if dist < best_dist:
                    best_dist = dist
                    best_put = q
            except (TypeError, ValueError):
                continue
        if not best_put:
            logger.debug("atm_iv_from_quotes: no valid put with strike (puts=%d)", len(puts))
            return None
        # Prefer provider iv on the ATM put
        iv = best_put.get("iv")
        if iv is not None and iv != "" and float(iv) > 0:
            return float(iv)
        # Build list of puts with usable price, sorted by distance to ATM
        others_with_price = []
        for q in puts:
            if not isinstance(q, dict) or q is best_put:
                continue
            strike = q.get("strike")
            if strike is None:
                continue
            try:
                s = float(strike)
            except (TypeError, ValueError):
                continue
            try:
                iv_ok = q.get("iv") and float(q.get("iv") or 0) > 0
                mid_ok = float(q.get("mid") or 0) > 0
                last_ok = float(q.get("last") or 0) > 0
                bid_ok = float(q.get("bid") or 0) > 0
                ask_ok = float(q.get("ask") or 0) > 0
                has_price = iv_ok or mid_ok or last_ok or bid_ok or ask_ok
            except (TypeError, ValueError):
                has_price = False
            if has_price:
                others_with_price.append((abs(s - underlying), q))
        others_with_price.sort(key=lambda x: x[0])
        candidates: List[Dict[str, Any]] = [best_put] + [q for _, q in others_with_price]
        for put in candidates:
            iv = put.get("iv")
            if iv is not None and iv != "" and float(iv) > 0:
                return float(iv)
            K = float(put.get("strike", 0))
            if K <= 0 or T_years <= 0:
                continue
            bid, ask = put.get("bid"), put.get("ask")
            mid_from_bid_ask = None
            if bid is not None and ask is not None and (float(bid or 0) > 0 or float(ask or 0) > 0):
                mid_from_bid_ask = (float(bid) + float(ask)) / 2.0
            mid_or_last = mid_from_bid_ask
            if (mid_or_last is None or mid_or_last <= 0) and put.get("mid"):
                try:
                    mid_or_last = float(put["mid"])
                except (TypeError, ValueError):
                    pass
            if (mid_or_last is None or mid_or_last <= 0) and put.get("last"):
                try:
                    mid_or_last = float(put["last"])
                except (TypeError, ValueError):
                    pass
            if mid_or_last and mid_or_last > 0:
                try:
                    from src.analysis.greeks_engine import GreeksEngine
                    engine = GreeksEngine()
                    kwargs = {}
                    if bid is not None and ask is not None and float(bid or 0) > 0:
                        kwargs["bid"] = float(bid)
                        kwargs["ask"] = float(ask)
                    iv_val, _ = engine.implied_volatility(
                        mid_or_last, underlying, K, T_years, "put", **kwargs
                    )
                    if iv_val and iv_val > 0:
                        return iv_val
                except Exception as e:
                    logger.debug("atm_iv_from_quotes: GreeksEngine failed for put K=%s: %s", put.get("strike"), e)
                    continue
        logger.debug(
            "atm_iv_from_quotes: no usable iv/bid_ask/mid on any put; puts=%d best_put_keys=%s",
            len(puts), list(best_put.keys()) if best_put else [],
        )
        return None
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug("atm_iv_from_quotes: parse/type error: %s", e)
        return None


def _bar_timestamp_to_ms(ts: Any) -> int:
    """Convert bar timestamp (ISO str or number) to milliseconds (UTC)."""
    if ts is None:
        return 0
    if isinstance(ts, (int, float)):
        return int(ts * 1000) if ts < 1e12 else int(ts)
    try:
        s = str(ts)
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        try:
            return int(float(ts) * 1000)
        except Exception:
            return 0


async def _fetch_snapshots(
    db: Database,
    symbol: str,
    start_ms: int,
    end_ms: int,
    provider_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load option chain snapshots for a symbol and date range.

    By default includes all providers (Alpaca + Tastytrade) for cross-validation.
    Pass provider_filter to restrict to one provider (e.g. "alpaca" or "tastytrade").

    Returns a list of snapshot dicts ordered chronologically by
    ``recv_ts_ms`` (falling back to ``timestamp_ms`` for legacy rows).

    Each dict includes:
    - timestamp_ms, recv_ts_ms, provider, underlying_price
    - atm_iv (if available)
    - quotes_json payload
    """
    raw = await db.get_option_chain_snapshots(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    snapshots: List[Dict[str, Any]] = []
    for row in raw:
        if provider_filter is not None and row.get("provider") != provider_filter:
            continue
        recv_ts = row.get("recv_ts_ms")
        if recv_ts is None:
            recv_ts = row.get("timestamp_ms", 0)
        atm_iv = row.get("atm_iv")
        if atm_iv is None or (isinstance(atm_iv, (int, float)) and atm_iv <= 0):
            underlying = float(row.get("underlying_price") or 0)
            atm_iv = _atm_iv_from_quotes_json(row.get("quotes_json", "") or "", underlying)

        snapshots.append({
            "timestamp_ms": row.get("timestamp_ms", 0),
            "recv_ts_ms": recv_ts,
            "provider": row.get("provider", ""),
            "underlying_price": row.get("underlying_price", 0.0),
            "atm_iv": atm_iv,
            "quotes_json": row.get("quotes_json", ""),
            "symbol": row.get("symbol", symbol),
            "n_strikes": row.get("n_strikes", 0),
        })

    # Sort by recv_ts_ms for strict chronological ordering
    snapshots.sort(key=lambda s: s["recv_ts_ms"])
    return snapshots




def _merge_snapshots_primary_with_fallback(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
    gap_ms: int,
) -> tuple[List[Dict[str, Any]], int]:
    """Prefer primary snapshots and fill long primary gaps from secondary."""
    if not primary:
        merged = sorted(secondary, key=lambda s: s["recv_ts_ms"])
        return merged, len(merged)

    merged = list(primary)
    inserted = 0
    secondary_sorted = sorted(secondary, key=lambda s: s["recv_ts_ms"])

    # Fill gaps before first and after last primary snapshot.
    first_primary_ts = primary[0]["recv_ts_ms"]
    last_primary_ts = primary[-1]["recv_ts_ms"]
    for s in secondary_sorted:
        ts = s["recv_ts_ms"]
        if ts < first_primary_ts and (first_primary_ts - ts) >= gap_ms:
            merged.append(s)
            inserted += 1
        elif ts > last_primary_ts and (ts - last_primary_ts) >= gap_ms:
            merged.append(s)
            inserted += 1

    # Fill interior primary gaps.
    for idx in range(len(primary) - 1):
        left = primary[idx]["recv_ts_ms"]
        right = primary[idx + 1]["recv_ts_ms"]
        if (right - left) < gap_ms:
            continue
        candidates = [
            s for s in secondary_sorted
            if left < s["recv_ts_ms"] < right
        ]
        if not candidates:
            continue
        merged.extend(candidates)
        inserted += len(candidates)

    merged.sort(key=lambda s: s["recv_ts_ms"])
    return merged, inserted

async def _fetch_snapshots_multi(
    db: Database,
    symbol: str,
    start_ms: int,
    end_ms: int,
    providers: List[str],
) -> List[Dict[str, Any]]:
    """Fetch snapshots for multiple providers and merge chronologically."""
    all_snaps: List[Dict[str, Any]] = []
    for prov in providers:
        snaps = await _fetch_snapshots(db, symbol, start_ms, end_ms, provider_filter=prov)
        all_snaps.extend(snaps)
    all_snaps.sort(key=lambda s: s["recv_ts_ms"])
    return all_snaps


async def create_replay_pack(
    symbol: str,
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD
    output_path: str,
    provider: Optional[str] = None,
    bar_duration: int = 60,
    db_path: str = "data/argus.db",
    snapshot_provider: Optional[str] = None,
    include_secondary_options: bool = False,
    options_snapshot_fallback: bool = False,
    options_snapshot_gap_minutes: int = 3,
    policy: Optional[DataSourcePolicy] = None,
    benchmark_symbol: str = "BTC-INDEX",
) -> Dict[str, Any]:
    """Create a replay pack for a single symbol.

    Provider defaults come from the data-source policy unless
    explicitly overridden via *provider* or *snapshot_provider*.

    Returns the pack dict (also written to *output_path*).
    """
    if policy is None:
        policy = get_data_source_policy()

    # Resolve effective providers from policy + overrides
    bars_provider = provider if provider is not None else policy.bars_provider
    effective_snapshot_provider = (
        snapshot_provider if snapshot_provider is not None
        else policy.options_snapshot_provider
    )

    db = Database(db_path)
    await db.connect()

    try:
        # 1. Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) + (24 * 3600 * 1000) - 1

        print(f"Packing data for {symbol} (bars={bars_provider}, snapshots={effective_snapshot_provider}) from {start_date} to {end_date}...")

        # 2. Fetch Bars
        bars_raw = await db.get_bars_for_outcome_computation(
            source=bars_provider,
            symbol=symbol,
            bar_duration=bar_duration,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        print(f"  Fetched {len(bars_raw)} bars.")

        # 3. Fetch Outcomes (always from bars_primary)
        outcomes = await db.get_bar_outcomes(
            provider=bars_provider,
            symbol=symbol,
            bar_duration_seconds=bar_duration,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        print(f"  Fetched {len(outcomes)} outcomes.")

        # 3.1 Fetch Benchmark Metrics
        benchmark_metrics = await db.get_metrics(
            symbol=benchmark_symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        print(f"  Fetched {len(benchmark_metrics)} benchmark metrics for {benchmark_symbol}.")

        # If no bars or outcomes, hint which provider/source has data in the DB
        if len(bars_raw) == 0 or len(outcomes) == 0:
            bar_inv = await db.get_bar_inventory()
            outcome_inv = await db.get_outcome_inventory()
            bar_sources = [r for r in bar_inv if r.get("symbol") == symbol]
            outcome_providers = [r for r in outcome_inv if r.get("symbol") == symbol]
            if bar_sources and len(bars_raw) == 0:
                sources_str = ", ".join(f"{r['source']} ({r.get('bar_count', 0)} bars)" for r in bar_sources)
                print(f"  Hint: No bars for provider={provider!r}. Bars in DB for {symbol}: {sources_str}. Try --provider <source>.")
            if outcome_providers and len(outcomes) == 0:
                prov_str = ", ".join(f"{r['provider']}" for r in outcome_providers)
                print(f"  Hint: No outcomes for provider={bars_provider!r}. Outcomes in DB for {symbol}: provider(s) {prov_str}.")
                print(f"  Likely cause: no outcomes computed for {start_date}. Run backfill first:")
                print(f"    python -m src.outcomes backfill --provider {bars_provider} --symbol {symbol} --bar {bar_duration} --start {start_date} --end {end_date}")

        # 4. Fetch Regimes (Symbol and Market)
        market = "EQUITIES"
        symbol_regimes = await db.get_regimes(scope=symbol, start_ms=start_ms, end_ms=end_ms)
        market_regimes = await db.get_regimes(scope=market, start_ms=start_ms, end_ms=end_ms)
        all_regimes = symbol_regimes + market_regimes
        print(f"  Fetched {len(all_regimes)} regimes ({len(symbol_regimes)} symbol, {len(market_regimes)} market).")

        # 5. Fetch Option Chain Snapshots
        fallback_used = False
        fallback_filled_count = 0
        fallback_gap_minutes = options_snapshot_gap_minutes
        if options_snapshot_fallback:
            secondary_providers = [
                p for p in policy.options_snapshots_secondary
                if p != effective_snapshot_provider
            ]
            primary_snaps = await _fetch_snapshots(
                db, symbol, start_ms, end_ms, provider_filter=effective_snapshot_provider
            )
            secondary_snaps = await _fetch_snapshots_multi(
                db, symbol, start_ms, end_ms, secondary_providers
            ) if secondary_providers else []
            snapshots, fallback_filled_count = _merge_snapshots_primary_with_fallback(
                primary=primary_snaps,
                secondary=secondary_snaps,
                gap_ms=max(1, options_snapshot_gap_minutes) * 60_000,
            )
            fallback_used = fallback_filled_count > 0
            print(
                f"  Fetched {len(primary_snaps)} primary snapshots and {len(secondary_snaps)} secondary snapshots; "
                f"fallback filled {fallback_filled_count} entries (gap>={options_snapshot_gap_minutes}m)."
            )
        elif include_secondary_options:
            snap_providers = policy.snapshot_providers(include_secondary=True)
            # Override primary if user specified a custom snapshot provider
            if snapshot_provider is not None:
                snap_providers = [snapshot_provider] + [
                    p for p in policy.options_snapshots_secondary
                    if p != snapshot_provider
                ]
            snapshots = await _fetch_snapshots_multi(
                db, symbol, start_ms, end_ms, snap_providers
            )
            print(f"  Fetched {len(snapshots)} option chain snapshots (providers: {snap_providers}).")
        else:
            snapshots = await _fetch_snapshots(
                db, symbol, start_ms, end_ms, provider_filter=effective_snapshot_provider
            )
            print(f"  Fetched {len(snapshots)} option chain snapshots (provider: {effective_snapshot_provider}).")

        # 6. Build Pack
        secondary_included = include_secondary_options
        pack: Dict[str, Any] = {
            "metadata": {
                "symbol": symbol,
                "provider": bars_provider,
                "bars_provider": bars_provider,
                "options_snapshot_provider": effective_snapshot_provider,
                "secondary_options_included": secondary_included,
                "options_snapshot_fallback_enabled": options_snapshot_fallback,
                "options_snapshot_fallback_used": fallback_used if options_snapshot_fallback else False,
                "options_snapshot_fallback_gap_minutes": fallback_gap_minutes if options_snapshot_fallback else None,
                "options_snapshot_fallback_filled": fallback_filled_count if options_snapshot_fallback else 0,
                "bar_duration": bar_duration,
                "start_date": start_date,
                "end_date": end_date,
                "packed_at": datetime.now(timezone.utc).isoformat(),
                "bar_count": len(bars_raw),
                "outcome_count": len(outcomes),
                "regime_count": len(all_regimes),
                "snapshot_count": len(snapshots),
            },
            "bars": [
                {**b, "timestamp_ms": _bar_timestamp_to_ms(b.get("timestamp"))}
                for b in bars_raw
            ],
            "outcomes": outcomes,
            "regimes": all_regimes,
            "snapshots": snapshots,
            "benchmark_metrics": benchmark_metrics,
            "benchmark_symbol": benchmark_symbol,
        }

        # 6b. Inject external metrics into Regimes
        # regimes in DB have metrics_json: str. We parse, inject, and re-serialize.
        # This ensures the replay-pack consumer (RegimeDetector or Strategy)
        # sees the metric as if it were emitted by the live system.
        all_daily_symbols = list(ASIA_SYMBOLS) + list(EUROPE_SYMBOLS) + [FX_RISK_SYMBOL]
        av_bars_by_sym = await db.get_bars_daily_for_risk_flow(
            source="alphavantage",
            symbols=all_daily_symbols,
            end_ms=end_ms,
            lookback_days=365,
        )

        injected_count = 0
        for regime in all_regimes:
            # regimes from DB are dicts with 'timestamp' (ISO string) or 'timestamp_ms' if already processed
            regime_ts_ms = regime.get("timestamp_ms")
            if regime_ts_ms is None:
                regime_ts_ms = _bar_timestamp_to_ms(regime.get("timestamp"))
            
            # Parse metrics_json once; skip invalid JSON rows.
            m_json = regime.get("metrics_json", "{}")
            try:
                metrics = json.loads(m_json)
            except json.JSONDecodeError:
                continue

            changed = False

            risk_flow = compute_global_risk_flow(av_bars_by_sym, regime_ts_ms)
            if risk_flow is not None:
                new_risk = round(risk_flow, 8)
                if metrics.get("global_risk_flow") != new_risk:
                    metrics["global_risk_flow"] = new_risk
                    changed = True

            news_sentiment = get_news_sentiment_for_replay(regime_ts_ms)
            if metrics.get("news_sentiment") != news_sentiment:
                metrics["news_sentiment"] = news_sentiment
                changed = True

            if changed:
                # Local reserialization with sort_keys for determinism.
                regime["metrics_json"] = json.dumps(metrics, sort_keys=True)
                injected_count += 1
        
        if injected_count > 0:
            print(f"  Injected external metrics into {injected_count} regimes.")

        # 7. Write
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(pack, f, indent=2)

        print(f"  Pack saved to {output_path}")
        return pack

    finally:
        await db.close()


async def create_universe_packs(
    start_date: str,
    end_date: str,
    output_dir: str = "data/packs",
    provider: Optional[str] = None,
    bar_duration: int = 60,
    db_path: str = "data/argus.db",
    symbols: Optional[List[str]] = None,
    snapshot_provider: Optional[str] = None,
    include_secondary_options: bool = False,
    options_snapshot_fallback: bool = False,
    options_snapshot_gap_minutes: int = 3,
    policy: Optional[DataSourcePolicy] = None,
) -> List[str]:
    """Create replay packs for every symbol in the liquid ETF universe.

    Provider defaults come from the data-source policy.

    Returns a list of output file paths that were written.
    """
    if symbols is None:
        symbols = get_liquid_etf_universe()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for sym in symbols:
        out_path = os.path.join(output_dir, f"{sym}_{start_date}_{end_date}.json")
        try:
            await create_replay_pack(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                output_path=out_path,
                provider=provider,
                bar_duration=bar_duration,
                db_path=db_path,
                snapshot_provider=snapshot_provider,
                include_secondary_options=include_secondary_options,
                options_snapshot_fallback=options_snapshot_fallback,
                options_snapshot_gap_minutes=options_snapshot_gap_minutes,
                policy=policy,
            )
            written.append(out_path)
        except Exception as exc:
            print(f"  WARNING: Failed to pack {sym}: {exc}")

    print(f"\nUniverse packing complete: {len(written)}/{len(symbols)} symbols.")
    return written


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Replay Pack(s) from the Argus database.  "
                    "Defaults follow the data_sources policy in config/config.yaml.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", help="Single symbol to pack (e.g. SPY)")
    group.add_argument(
        "--universe",
        action="store_true",
        default=False,
        help="Pack all symbols in the liquid ETF universe",
    )

    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (single-symbol mode) or directory (universe mode). "
             "Defaults to data/packs/",
    )

    # ── Provider overrides (advanced) ─────────────────────────────────
    parser.add_argument(
        "--provider",
        default=None,
        help="(Legacy) Alias for --bars-provider.  Still accepted for backward compatibility.",
    )
    parser.add_argument(
        "--bars-provider",
        default=None,
        help="Override bars/outcomes provider (default: data_sources.bars_primary from config).",
    )
    parser.add_argument(
        "--options-snapshot-provider",
        default=None,
        help="Override primary options snapshot provider (default: data_sources.options_snapshots_primary from config).",
    )
    parser.add_argument(
        "--snapshot-provider",
        default=None,
        help="(Legacy) Alias for --options-snapshot-provider.",
    )
    parser.add_argument(
        "--include-secondary-options",
        action="store_true",
        default=False,
        help="Also include secondary options snapshots alongside primary.",
    )
    parser.add_argument(
        "--options-snapshot-fallback",
        action="store_true",
        default=False,
        help="Prefer primary options snapshots and fill multi-minute gaps with secondary provider snapshots.",
    )
    parser.add_argument(
        "--options-snapshot-gap-minutes",
        type=int,
        default=3,
        help="Gap threshold in minutes for --options-snapshot-fallback (default: 3).",
    )
    parser.add_argument("--db", default="data/argus.db", help="Path to argus.db")
    parser.add_argument(
        "--benchmark",
        default="BTC-INDEX",
        help="Benchmark symbol to include metrics for (default: BTC-INDEX)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve provider overrides: new flags take precedence over legacy
    bars_prov = args.bars_provider or args.provider  # None = use policy default
    snap_prov = args.options_snapshot_provider or args.snapshot_provider  # None = use policy default

    if args.universe:
        out_dir = args.out or "data/packs"
        asyncio.run(create_universe_packs(
            start_date=args.start,
            end_date=args.end,
            output_dir=out_dir,
            provider=bars_prov,
            db_path=args.db,
            snapshot_provider=snap_prov,
            include_secondary_options=args.include_secondary_options,
            options_snapshot_fallback=args.options_snapshot_fallback,
            options_snapshot_gap_minutes=args.options_snapshot_gap_minutes,
        ))
    else:
        default_file = f"data/packs/{args.symbol}_{args.start}_{args.end}.json"
        if args.out is None:
            out_path = default_file
        else:
            p = Path(args.out)
            if p.suffix != ".json" or p.exists() and p.is_dir():
                # Treat as output directory: write SYMBOL_start_end.json inside it
                out_path = str(Path(args.out).resolve() / f"{args.symbol}_{args.start}_{args.end}.json")
            else:
                out_path = args.out
        asyncio.run(create_replay_pack(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            output_path=out_path,
            provider=bars_prov,
            db_path=args.db,
            snapshot_provider=snap_prov,
            include_secondary_options=args.include_secondary_options,
            options_snapshot_fallback=args.options_snapshot_fallback,
            options_snapshot_gap_minutes=args.options_snapshot_gap_minutes,
            benchmark_symbol=args.benchmark,
        ))


if __name__ == "__main__":
    main()
