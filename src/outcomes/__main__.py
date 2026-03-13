"""
Argus Outcomes CLI
==================

Command-line interface for outcome computation, coverage reporting,
uptime diagnostics, and gap analysis.

Usage::

    python -m src.outcomes list              # bar inventory + coverage %
    python -m src.outcomes list-outcomes      # outcome inventory by status
    python -m src.outcomes heartbeats         # heartbeat components, counts, DB path
    python -m src.outcomes health             # collector health: last bar age per source/symbol
    python -m src.outcomes gaps --provider bybit --symbol BTCUSDT  # top gaps
    python -m src.outcomes uptime --start 2026-02-01 --end 2026-02-09
    python -m src.outcomes diagnose --provider bybit --symbol BTCUSDT --start 2026-02-05 --end 2026-02-09
    python -m src.outcomes discover --bases BTC,ETH,SOL  # discover Bybit instruments
    python -m src.outcomes backfill-bars --symbol BTCUSDT [--start ...] [--end ...]
    python -m src.outcomes backfill --provider bybit --symbol BTCUSDT --bar 60 --start ...
    python -m src.outcomes backfill-all --start ... --end ...
    python -m src.outcomes coverage [--provider X] [--symbol Y]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.core.database import Database
from src.core.outcome_engine import OutcomeEngine
from src.core.coverage import (
    analyze_bar_continuity,
    compute_uptime,
    diagnose_coverage,
    is_likely_equity,
    GapInfo,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_date(s: str) -> int:
    """Parse date string to epoch milliseconds (UTC)."""
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)


def _format_duration(total_seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if total_seconds <= 0:
        return "0s"
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    years, days = divmod(days, 365)
    months, days = divmod(days, 30)
    parts: list[str] = []
    if years:
        parts.append(f"{years}y")
    if months:
        parts.append(f"{months}mo")
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs:
        parts.append(f"{secs}s")
    return " ".join(parts) if parts else "0s"


def _iso_from_ts_str(ts_str: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.isoformat()
    except Exception:
        return ts_str


def _iso_from_ms(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ms)


def _bar_dur_label(seconds: int) -> str:
    labels = {60: "1m", 300: "5m", 900: "15m", 3600: "1h",
              14400: "4h", 86400: "1d", 604800: "1w"}
    return labels.get(seconds, f"{seconds}s")


def _span_seconds_from_ts_strs(min_ts: str, max_ts: str) -> int:
    try:
        t1 = datetime.fromisoformat(min_ts.replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(max_ts.replace("Z", "+00:00"))
        return max(0, int((t2 - t1).total_seconds()))
    except Exception:
        return 0


def _pct(num: float, den: float) -> str:
    if den <= 0:
        return "N/A"
    return f"{num / den * 100:.1f}%"


async def _get_bar_keys(db: Database) -> List[tuple]:
    inv = await db.get_bar_inventory()
    return [(r["source"], r["symbol"], r["bar_duration"]) for r in inv]


def _suggest_keys(requested_provider: Optional[str],
                  requested_symbol: Optional[str],
                  keys: List[tuple]) -> None:
    if not keys:
        print("\n  (market_bars table is empty - no data ingested yet)")
        return
    print("\n  Available keys in market_bars:")
    seen = set()
    for src, sym, dur in keys:
        tag = f"    {src} / {sym}  ({_bar_dur_label(dur)})"
        if tag not in seen:
            print(tag)
            seen.add(tag)
    if requested_provider or requested_symbol:
        providers = {k[0] for k in keys}
        symbols = {k[1] for k in keys}
        if requested_provider and requested_provider not in providers:
            close = [p for p in providers
                     if requested_provider.lower() in p.lower()
                     or p.lower() in requested_provider.lower()]
            if close:
                print(f"  Hint: provider '{requested_provider}' not found; "
                      f"did you mean: {', '.join(close)}?")
        if requested_symbol and requested_symbol not in symbols:
            close = [s for s in symbols
                     if requested_symbol.replace("/", "").replace(":", "").upper()
                     in s.replace("/", "").replace(":", "").upper()
                     or s.replace("/", "").replace(":", "").upper()
                     in requested_symbol.replace("/", "").replace(":", "").upper()]
            if close:
                print(f"  Hint: symbol '{requested_symbol}' not found; "
                      f"did you mean: {', '.join(close)}?")


def _print_gap_table(gaps: List[GapInfo], indent: str = "    ") -> None:
    """Print a formatted table of gaps."""
    if not gaps:
        print(f"{indent}(no significant gaps)")
        return
    print(f"{indent}{'#':<4} {'Gap':>10}  {'From':<20} {'To':<20}")
    print(f"{indent}{'-'*56}")
    for i, g in enumerate(gaps, 1):
        print(f"{indent}{i:<4} {_format_duration(g.gap_seconds):>10}  "
              f"{_iso_from_ms(g.start_ms):<20} {_iso_from_ms(g.end_ms):<20}")


async def _open_db(args) -> Database:
    config = load_config(args.config)
    db_path = getattr(args, "db", None) or config.get("database", {}).get("path", "data/argus.db")
    db = Database(db_path)
    await db.connect()
    return db


# ═══════════════════════════════════════════════════════════════════════════════
#  Commands
# ═══════════════════════════════════════════════════════════════════════════════


async def _cmd_list(args):
    """List all bar keys with continuity diagnostics."""
    db = await _open_db(args)
    try:
        inventory = await db.get_bar_inventory()
        if not inventory:
            print("No bars in market_bars.")
            return

        W = 110
        print("=" * W)
        print("  MARKET BARS INVENTORY")
        print("=" * W)
        print(f"  {'Provider':<10} {'Symbol':<14} {'Bar':<4} "
              f"{'Count':>8} {'Expected':>8} {'Cov%':>6} "
              f"{'MaxGap':>8} {'Gaps5m+':>7}  {'Span':<16} {'Note'}")
        print("-" * W)

        for row in inventory:
            src = row["source"]
            sym = row["symbol"]
            dur = row["bar_duration"]
            cnt = row["bar_count"]
            min_ts = row["min_ts"] or ""
            max_ts = row["max_ts"] or ""
            span_s = _span_seconds_from_ts_strs(min_ts, max_ts) if min_ts and max_ts else 0
            expected = (span_s // dur) + 1 if span_s > 0 and dur > 0 else cnt
            cov = cnt / expected * 100 if expected > 0 else 0.0
            equity = is_likely_equity(src, sym)
            note = "equity" if equity else ""

            # For max_gap we'd need full timestamp list — use a lightweight query
            # Instead show N/A and point to `gaps` command
            max_gap_str = "--"
            gap5m_str = "--"

            print(f"  {src:<10} {sym:<14} {_bar_dur_label(dur):<4} "
                  f"{cnt:>8,} {expected:>8,} {cov:>5.1f}% "
                  f"{max_gap_str:>8} {gap5m_str:>7}  "
                  f"{_format_duration(span_s):<16} {note}")

        print("-" * W)
        print("  Cov% = count / expected_bars (expected = span/bar_dur + 1)")
        print("  For detailed gaps: python -m src.outcomes gaps --provider X --symbol Y")
        print("=" * W)
    finally:
        await db.close()


async def _cmd_list_outcomes(args):
    """List all outcome keys in bar_outcomes."""
    db = await _open_db(args)
    try:
        inventory = await db.get_outcome_inventory()
        if not inventory:
            print("No outcomes in bar_outcomes.")
            return

        W = 105
        print("=" * W)
        print("  BAR OUTCOMES INVENTORY")
        print("=" * W)
        print(f"  {'Provider':<10} {'Symbol':<14} {'Bar':<4} {'Horizon':<8} "
              f"{'Total':>8} {'OK':>8} {'INC':>6} {'GAP':>6}  "
              f"{'Min Timestamp':<20} {'Span'}")
        print("-" * W)

        for row in inventory:
            prov = row["provider"]
            sym = row["symbol"]
            dur = row["bar_duration_seconds"]
            hor = row["horizon_seconds"]
            total = row["total"]
            ok = row["ok_count"]
            inc = row["incomplete_count"]
            gap = row["gap_count"]
            min_ms = row["min_ts_ms"]
            max_ms = row["max_ts_ms"]
            span = max(0, (max_ms - min_ms) // 1000) if min_ms and max_ms else 0

            print(f"  {prov:<10} {sym:<14} {_bar_dur_label(dur):<4} "
                  f"{_bar_dur_label(hor):<8} "
                  f"{total:>8,} {ok:>8,} {inc:>6,} {gap:>6,}  "
                  f"{_iso_from_ms(min_ms):<20} {_format_duration(span)}")

        print("=" * W)
    finally:
        await db.close()


async def _cmd_gaps(args):
    """Show top N timestamp gaps for a provider/symbol/bar."""
    db = await _open_db(args)
    try:
        bar_keys = await _get_bar_keys(db)
        if (args.provider, args.symbol, args.bar) not in bar_keys:
            print(f"No bars for {args.provider}/{args.symbol} "
                  f"({_bar_dur_label(args.bar)})")
            _suggest_keys(args.provider, args.symbol, bar_keys)
            return

        # Get all bar timestamps
        inv = await db.get_bar_inventory()
        row = next(r for r in inv
                   if r["source"] == args.provider
                   and r["symbol"] == args.symbol
                   and r["bar_duration"] == args.bar)
        min_ts_str = row["min_ts"]
        max_ts_str = row["max_ts"]
        min_dt = datetime.fromisoformat(min_ts_str.replace("Z", "+00:00"))
        max_dt = datetime.fromisoformat(max_ts_str.replace("Z", "+00:00"))
        start_ms = int(min_dt.timestamp() * 1000)
        end_ms = int(max_dt.timestamp() * 1000)

        bar_ts = await db.get_bar_timestamps(
            args.provider, args.symbol, args.bar, start_ms, end_ms)

        result = analyze_bar_continuity(
            bar_ts, args.bar,
            gap_threshold_seconds=args.threshold,
            top_n=args.top,
        )

        equity = is_likely_equity(args.provider, args.symbol)

        W = 72
        print("=" * W)
        print(f"  GAP ANALYSIS: {args.provider}/{args.symbol} "
              f"({_bar_dur_label(args.bar)})")
        print("=" * W)
        print(f"  Bars:          {result.bar_count:,}")
        print(f"  Span:          {_format_duration(result.span_seconds)}")
        print(f"  Expected:      {result.expected_bars:,}")
        print(f"  Coverage:      {result.coverage_pct:.1f}%")
        print(f"  Max gap:       {_format_duration(result.max_gap_seconds)}")
        print(f"  Gaps >= {_format_duration(args.threshold)}: "
              f"{result.gap_count_above_threshold}")
        if equity:
            print(f"  Note:          Equity/ETF - overnight/weekend gaps expected")
        print()
        print(f"  Top {len(result.top_gaps)} gaps:")
        _print_gap_table(result.top_gaps, indent="    ")
        print("=" * W)

    finally:
        await db.close()


async def _cmd_uptime(args):
    """Show system uptime computed from heartbeat records."""
    db = await _open_db(args)
    try:
        start_ms = _parse_date(args.start)
        end_ms = _parse_date(args.end) + 86400 * 1000 - 1
        component = args.component

        heartbeats = await db.get_heartbeats(component, start_ms, end_ms)
        threshold_ms = args.threshold * 1000

        result = compute_uptime(heartbeats, start_ms, end_ms, threshold_ms)

        W = 64
        print("=" * W)
        print(f"  ARGUS UPTIME REPORT ({component})")
        print("=" * W)
        print(f"  Range:           {args.start} to {args.end}")
        print(f"  Wall span:       {_format_duration(result.wall_span_s)} "
              f"({result.wall_span_s:,}s)")
        print(f"  Heartbeats:      {result.heartbeat_count:,}")
        if result.median_cadence_ms > 0:
            print(f"  Inferred cadence:{result.median_cadence_ms / 1000:.1f}s "
                  f"(median delta between heartbeats)")
        print(f"  Gap threshold:   {args.threshold}s")
        print()
        print(f"  Uptime:          {_format_duration(result.uptime_s)} "
              f"({result.uptime_s:,}s)")
        print(f"  Downtime:        {_format_duration(result.downtime_s)} "
              f"({result.downtime_s:,}s)")
        if result.wall_span_s > 0:
            up_pct = result.uptime_s / result.wall_span_s * 100
            print(f"  Uptime %:        {up_pct:.1f}%")
            print(f"  Downtime %:      {100 - up_pct:.1f}%")

        if result.off_intervals:
            # Sort by duration desc for "top 5 longest" display
            sorted_off = sorted(result.off_intervals,
                                key=lambda iv: iv[1] - iv[0], reverse=True)
            print()
            print(f"  Downtime intervals: {len(result.off_intervals)}")
            print(f"  Top {min(5, len(sorted_off))} longest:")
            for i, (s, e) in enumerate(sorted_off[:5], 1):
                dur_s = (e - s) // 1000
                print(f"    {i:>3}. {_iso_from_ms(s)} -> {_iso_from_ms(e)}  "
                      f"({_format_duration(dur_s)})")

            if len(result.off_intervals) > 5:
                print()
                print(f"  All OFF intervals (chronological, "
                      f"{len(result.off_intervals)} total):")
                for i, (s, e) in enumerate(result.off_intervals[:20], 1):
                    dur_s = (e - s) // 1000
                    print(f"    {i:>3}. {_iso_from_ms(s)} -> {_iso_from_ms(e)}  "
                          f"({_format_duration(dur_s)})")
                if len(result.off_intervals) > 20:
                    print(f"    ... and {len(result.off_intervals) - 20} more")
        elif result.heartbeat_count >= 2:
            print(f"\n  Downtime intervals: 0")
            print("  No OFF intervals detected - continuous uptime.")

        print("=" * W)
    finally:
        await db.close()


async def _cmd_diagnose(args):
    """Uptime-aware bar coverage diagnosis."""
    db = await _open_db(args)
    try:
        bar_keys = await _get_bar_keys(db)
        if (args.provider, args.symbol, args.bar) not in bar_keys:
            print(f"No bars for {args.provider}/{args.symbol} "
                  f"({_bar_dur_label(args.bar)})")
            _suggest_keys(args.provider, args.symbol, bar_keys)
            return

        start_ms = _parse_date(args.start)
        end_ms = _parse_date(args.end) + 86400 * 1000 - 1

        # Get heartbeats
        heartbeats = await db.get_heartbeats(
            args.component, start_ms, end_ms)
        threshold_ms = args.hb_threshold * 1000
        uptime = compute_uptime(heartbeats, start_ms, end_ms, threshold_ms)

        # Get bar timestamps
        bar_ts = await db.get_bar_timestamps(
            args.provider, args.symbol, args.bar, start_ms, end_ms)

        equity = is_likely_equity(args.provider, args.symbol)
        result = diagnose_coverage(
            bar_ts, args.bar, uptime, start_ms, end_ms,
            is_equity=equity,
        )

        W = 72
        print("=" * W)
        print(f"  COVERAGE DIAGNOSIS: {args.provider}/{args.symbol} "
              f"({_bar_dur_label(args.bar)})")
        print("=" * W)

        # --- Heartbeat info ---
        print(f"\n  Heartbeats")
        print(f"  {'Count:':<30} {uptime.heartbeat_count:,}")
        if uptime.median_cadence_ms > 0:
            print(f"  {'Inferred cadence:':<30} "
                  f"{uptime.median_cadence_ms / 1000:.1f}s")
        print(f"  {'Gap threshold:':<30} {args.hb_threshold}s")
        print(f"  {'Downtime intervals:':<30} "
              f"{len(uptime.off_intervals)}")

        # --- Time ---
        print(f"\n  Time")
        print(f"  {'Wall span:':<30} {_format_duration(result.wall_span_s)} "
              f"({result.wall_span_s:,}s)")
        print(f"  {'Uptime:':<30} {_format_duration(result.uptime_s)} "
              f"({result.uptime_s:,}s) [{result.uptime_pct}%]")
        print(f"  {'Downtime:':<30} {_format_duration(result.downtime_s)} "
              f"({result.downtime_s:,}s) [{100 - result.uptime_pct:.1f}%]")

        if uptime.off_intervals:
            sorted_off = sorted(uptime.off_intervals,
                                key=lambda iv: iv[1] - iv[0], reverse=True)
            print(f"\n  Top {min(5, len(sorted_off))} longest downtime gaps:")
            for i, (s, e) in enumerate(sorted_off[:5], 1):
                dur_s = (e - s) // 1000
                print(f"    {i:>3}. {_iso_from_ms(s)} -> {_iso_from_ms(e)}  "
                      f"({_format_duration(dur_s)})")

        # --- Bars ---
        print(f"\n  Bars (total)")
        print(f"  {'Total bars in range:':<30} {result.bars_total:,}")
        print(f"  {'Bars during uptime:':<30} {result.bars_during_uptime:,}")
        print(f"  {'Bars during downtime:':<30} "
              f"{result.bars_during_downtime:,}")

        print(f"\n  Coverage during uptime (actionable)")
        print(f"  {'Expected during uptime:':<30} "
              f"{result.expected_bars_during_uptime:,}")
        print(f"  {'Observed during uptime:':<30} "
              f"{result.bars_during_uptime:,}")
        print(f"  {'Uptime coverage:':<30} {result.uptime_coverage_pct}%")

        print(f"\n  Missing bars breakdown")
        print(f"  {'Missing during uptime:':<30} "
              f"{result.missing_during_uptime:,}"
              f"  <- ACTIONABLE (feed/ingestion issue)")
        print(f"  {'Missing during downtime:':<30} "
              f"{result.missing_during_downtime:,}"
              f"  <- expected (Argus was OFF)")

        if result.is_equity and result.equity_session_note:
            print(f"\n  Note: {result.equity_session_note}")

        if result.top_uptime_gaps:
            print(f"\n  Top gaps during uptime (feed issues):")
            _print_gap_table(result.top_uptime_gaps)

        if not heartbeats:
            print(f"\n  WARNING: No heartbeats found for '{args.component}' "
                  f"in this range.")
            print(f"  Uptime tracking requires Argus to be running with "
                  f"the orchestrator heartbeat enabled.")
            print(f"  Without heartbeats, all time is counted as downtime.")

        print("=" * W)
    finally:
        await db.close()


async def _cmd_heartbeats(args):
    """List heartbeat components, counts, and timestamp ranges."""
    config = load_config(args.config)
    db_path = config.get("database", {}).get("path", "data/argus.db")
    db = Database(db_path)
    await db.connect()
    try:
        inventory = await db.get_heartbeat_inventory()

        W = 72
        print("=" * W)
        print("  SYSTEM HEARTBEAT INVENTORY")
        print("=" * W)
        print(f"  DB path: {db_path}")
        print()

        if not inventory:
            print("  No heartbeats recorded yet.")
            print("  Heartbeats are written automatically when Argus is running.")
            print("=" * W)
            return

        print(f"  {'Component':<20} {'Count':>8}  "
              f"{'Min Timestamp':<20} {'Max Timestamp':<20} {'Span'}")
        print("-" * W)

        for row in inventory:
            comp = row["component"]
            cnt = row["count"]
            min_ms = row["min_ts_ms"]
            max_ms = row["max_ts_ms"]
            span_s = max(0, (max_ms - min_ms) // 1000) if min_ms and max_ms else 0

            print(f"  {comp:<20} {cnt:>8,}  "
                  f"{_iso_from_ms(min_ms):<20} {_iso_from_ms(max_ms):<20} "
                  f"{_format_duration(span_s)}")

        print("=" * W)
    finally:
        await db.close()


async def _cmd_backfill(args):
    """Backfill outcomes for a date range."""
    config = load_config(args.config)
    outcomes_config = config.get("outcomes", {})
    db = await _open_db(args)
    try:
        engine = OutcomeEngine(db, outcomes_config)
        start_ms = _parse_date(args.start)
        end_ms = _parse_date(args.end) + 86400 * 1000 - 1

        bar_keys = await _get_bar_keys(db)
        if (args.provider, args.symbol, args.bar) not in bar_keys:
            print(f"WARNING: No bars found for "
                  f"provider='{args.provider}' symbol='{args.symbol}' "
                  f"bar_duration={args.bar}s")
            _suggest_keys(args.provider, args.symbol, bar_keys)
            return

        print(f"Backfilling outcomes:")
        print(f"  Provider: {args.provider}")
        print(f"  Symbol:   {args.symbol}")
        print(f"  Bar:      {args.bar}s ({_bar_dur_label(args.bar)})")
        print(f"  Range:    {args.start} to {args.end}")
        print(f"  Version:  {engine.outcome_version}")
        print()

        bars, outcomes, upserted = await engine.compute_outcomes_for_range(
            provider=args.provider, symbol=args.symbol,
            bar_duration_seconds=args.bar,
            start_ms=start_ms, end_ms=end_ms)

        print(f"Results:")
        print(f"  Bars processed:    {bars:,}")
        print(f"  Outcomes computed:  {outcomes:,}")
        print(f"  Rows upserted:     {upserted:,}")
    finally:
        await db.close()


async def _cmd_backfill_all(args):
    """Backfill outcomes for all provider/symbol combos in market_bars."""
    config = load_config(args.config)
    outcomes_config = config.get("outcomes", {})
    if not outcomes_config.get("enabled", True):
        print("Outcomes are disabled in config (outcomes.enabled=false)")
        return

    db = await _open_db(args)
    try:
        engine = OutcomeEngine(db, outcomes_config)
        start_ms = _parse_date(args.start)
        end_ms = _parse_date(args.end) + 86400 * 1000 - 1

        inventory = await db.get_bar_inventory()
        if not inventory:
            print("No bars in market_bars - nothing to backfill.")
            return

        total_bars = total_outcomes = total_upserted = skipped = 0
        for row in inventory:
            provider = row["source"]
            symbol = row["symbol"]
            bar_duration = row["bar_duration"]
            if bar_duration not in engine.horizons_by_bar:
                skipped += 1
                continue

            horizons = engine.horizons_by_bar[bar_duration]
            print(f"Processing {provider}/{symbol} "
                  f"({_bar_dur_label(bar_duration)}) "
                  f"-> horizons: {[_bar_dur_label(h) for h in horizons]} ...")

            bars, outcomes, upserted = await engine.compute_outcomes_for_range(
                provider=provider, symbol=symbol,
                bar_duration_seconds=bar_duration,
                start_ms=start_ms, end_ms=end_ms)
            total_bars += bars
            total_outcomes += outcomes
            total_upserted += upserted
            print(f"  -> {bars:,} bars, {outcomes:,} outcomes, "
                  f"{upserted:,} upserted")

        print()
        if skipped:
            print(f"Skipped {skipped} bar-duration(s) with no configured horizons")
        print(f"Total: {total_bars:,} bars, {total_outcomes:,} outcomes, "
              f"{total_upserted:,} upserted")
    finally:
        await db.close()


async def _cmd_health(args):
    """Show collector health: last bar time per provider/symbol."""
    db = await _open_db(args)
    try:
        rows = await db.get_bar_health()
        if not rows:
            print("No bars in market_bars.")
            return

        W = 90
        print("=" * W)
        print("  COLLECTOR HEALTH (last bar per source/symbol)")
        print("=" * W)
        print(f"  {'Source':<10} {'Symbol':<14} {'Bar':<4} "
              f"{'Count':>8}  {'Last Bar':>20}  {'Age'}")
        print("-" * W)

        for row in rows:
            src = row["source"]
            sym = row["symbol"]
            dur = row["bar_duration"]
            cnt = row["bar_count"]
            last_ts = row.get("last_ts", "")
            age_s = row.get("last_ts_age_s", -1)
            age_str = _format_duration(age_s) if age_s >= 0 else "N/A"
            stale = " ** STALE" if age_s > 300 else ""

            print(f"  {src:<10} {sym:<14} {_bar_dur_label(dur):<4} "
                  f"{cnt:>8,}  {_iso_from_ts_str(last_ts):>20}  "
                  f"{age_str}{stale}")

        print("=" * W)
    finally:
        await db.close()


async def _cmd_discover(args):
    """Discover tradeable Bybit perpetual instruments."""
    from src.connectors.bybit_rest import BybitRestClient

    config = load_config(args.config)
    bybit_cfg = config.get("exchanges", {}).get("bybit", {})
    testnet = bybit_cfg.get("testnet", False)
    from src.connectors.bybit_rest import BYBIT_BASE_URL, BYBIT_TESTNET_URL
    base = BYBIT_TESTNET_URL if testnet else BYBIT_BASE_URL

    client = BybitRestClient(base_url=base)
    try:
        base_coins = None
        if args.bases:
            base_coins = [b.strip().upper() for b in args.bases.split(",")]

        instruments = await client.discover_perpetuals(
            quote_coin=args.quote or "USDT",
            base_coins=base_coins,
        )

        W = 72
        print("=" * W)
        print(f"  BYBIT PERPETUAL INSTRUMENTS ({len(instruments)} found)")
        print("=" * W)
        print(f"  {'Symbol':<16} {'Base':<8} {'Quote':<6} "
              f"{'Status':<12} {'Contract Type'}")
        print("-" * W)

        for inst in sorted(instruments, key=lambda i: i.symbol):
            print(f"  {inst.symbol:<16} {inst.base_coin:<8} "
                  f"{inst.quote_coin:<6} {inst.status:<12} "
                  f"{inst.contract_type}")

        print("=" * W)
    finally:
        await client.close()


async def _cmd_backfill_bars(args):
    """Backfill missing 1m bars from Bybit REST kline API."""
    from src.connectors.bybit_rest import BybitRestClient, klines_to_bar_rows

    config = load_config(args.config)
    bybit_cfg = config.get("exchanges", {}).get("bybit", {})
    testnet = bybit_cfg.get("testnet", False)
    from src.connectors.bybit_rest import BYBIT_BASE_URL, BYBIT_TESTNET_URL
    base = BYBIT_TESTNET_URL if testnet else BYBIT_BASE_URL

    db = await _open_db(args)
    client = BybitRestClient(base_url=base)

    try:
        symbol = args.symbol

        # Determine range
        if args.start:
            start_ms = _parse_date(args.start)
        else:
            # Auto: start from last bar + 1 minute
            last_ms = await db.get_last_bar_timestamp_ms("bybit", symbol, 60)
            if last_ms:
                start_ms = last_ms + 60_000
                print(f"Auto-start: resuming from last bar + 1m "
                      f"({_iso_from_ms(start_ms)})")
            else:
                print(f"No existing bars for bybit/{symbol}. "
                      f"Use --start to specify start date.")
                return

        if args.end:
            end_ms = _parse_date(args.end) + 86400 * 1000 - 1
        else:
            # Auto: up to now
            import time
            end_ms = int(time.time() * 1000)
            print(f"Auto-end: using current time ({_iso_from_ms(end_ms)})")

        if start_ms >= end_ms:
            print("Start >= end, nothing to backfill.")
            return

        span_s = (end_ms - start_ms) // 1000
        expected = span_s // 60
        print(f"\nBackfilling bybit/{symbol} bars:")
        print(f"  Range:     {_iso_from_ms(start_ms)} to {_iso_from_ms(end_ms)}")
        print(f"  Span:      {_format_duration(span_s)}")
        print(f"  Expected:  ~{expected:,} bars")
        print()

        def _progress(fetched, chunks):
            print(f"  ... fetched {fetched:,} klines ({chunks} API calls)",
                  end="\r")

        klines = await client.backfill_klines(
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            interval="1",
            category="linear",
            progress_callback=_progress,
        )

        print(f"\n  Received {len(klines):,} klines from Bybit API")

        if not klines:
            print("  No klines returned. Nothing to insert.")
            return

        rows = klines_to_bar_rows(klines, source="bybit",
                                  symbol=symbol, bar_duration=60)
        inserted = await db.upsert_bars_backfill(rows)

        print(f"\n  Results:")
        print(f"    Klines received:  {len(klines):,}")
        print(f"    New bars inserted:{inserted:,}")
        print(f"    Already existed:  {len(klines) - inserted:,}")

    finally:
        await client.close()
        await db.close()


async def _cmd_coverage(args):
    """Show bar and outcome coverage stats."""
    db = await _open_db(args)
    try:
        W = 64
        print("=" * W)
        print("  ARGUS DATA COVERAGE REPORT")
        print("=" * W)

        bar_stats = await db.get_bar_coverage_stats(
            source=args.provider, symbol=args.symbol)

        print("\n  MARKET BARS")
        print("-" * W)
        total_bars = bar_stats.get("total_bars", 0)
        if total_bars:
            min_ts = bar_stats.get("min_ts")
            max_ts = bar_stats.get("max_ts")
            span_s = _span_seconds_from_ts_strs(min_ts, max_ts) if min_ts and max_ts else 0
            if args.provider:
                print(f"  Provider filter: {args.provider}")
            if args.symbol:
                print(f"  Symbol filter:   {args.symbol}")
            print(f"  Total bars:      {total_bars:,}")
            if min_ts:
                print(f"  Min timestamp:   {_iso_from_ts_str(min_ts)}")
            if max_ts:
                print(f"  Max timestamp:   {_iso_from_ts_str(max_ts)}")
            if span_s:
                print(f"  Span:            {_format_duration(span_s)} "
                      f"({span_s:,}s)")
        else:
            print("  No bars found for this filter.")
            bar_keys = await _get_bar_keys(db)
            _suggest_keys(args.provider, args.symbol, bar_keys)

        outcome_stats = await db.get_outcome_coverage_stats(
            provider=args.provider, symbol=args.symbol)

        print("\n  BAR OUTCOMES")
        print("-" * W)
        total_out = outcome_stats.get("total_outcomes", 0)
        if total_out:
            ok = outcome_stats.get("ok_count", 0)
            inc = outcome_stats.get("incomplete_count", 0)
            gap = outcome_stats.get("gap_count", 0)
            min_ms = outcome_stats.get("min_ts_ms")
            max_ms = outcome_stats.get("max_ts_ms")
            span_s = max(0, (max_ms - min_ms) // 1000) if min_ms and max_ms else 0

            print(f"  Total outcomes:  {total_out:,}")
            print(f"  Status breakdown:")
            print(f"    OK:            {ok:,}  ({_pct(ok, total_out)})")
            print(f"    INCOMPLETE:    {inc:,}  ({_pct(inc, total_out)})")
            print(f"    GAP:           {gap:,}  ({_pct(gap, total_out)})")
            if min_ms:
                print(f"  Min timestamp:   {_iso_from_ms(min_ms)}")
            if max_ms:
                print(f"  Max timestamp:   {_iso_from_ms(max_ms)}")
            if span_s:
                print(f"  Span:            {_format_duration(span_s)} "
                      f"({span_s:,}s)")
        else:
            print("  No outcomes found for this filter.")
            if total_bars:
                print("  (Run `backfill` or `backfill-all` to compute outcomes.)")

        print()
        print("=" * W)
    finally:
        await db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Research Loop (Hades Port)
# ═══════════════════════════════════════════════════════════════════════════════


async def _cmd_research_loop(args):
    """Run the Hades strategy research loop.

    Wraps ``scripts/strategy_research_loop.py`` logic as a first-class CLI command.
    Supports --case-id to tag runs with a Pantheon research case.
    """
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    _logger = _logging.getLogger("argus.research_loop")

    # Import the research loop machinery
    try:
        from src.analysis.research_loop_config import (
            ConfigValidationError,
            load_research_loop_config,
        )
    except ImportError as exc:
        print(f"ERROR: Could not import research loop config: {exc}")
        print("Ensure src/analysis/research_loop_config.py exists.")
        sys.exit(1)

    config_path = getattr(args, "config_rl", None) or "config/research_loop.yaml"
    case_id = getattr(args, "case_id", None)
    dry_run = getattr(args, "dry_run", False)
    once = getattr(args, "once", False)

    try:
        config = load_research_loop_config(config_path)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    except ConfigValidationError as exc:
        print(f"ERROR: Invalid config: {exc}")
        sys.exit(1)

    if case_id:
        _logger.info("Research loop tagged with case_id=%s", case_id)

    # Import the cycle runner from the scripts module
    try:
        from scripts.strategy_research_loop import run_cycle
    except ImportError:
        # Fallback: add repo root to path and retry
        _repo = Path(__file__).resolve().parent.parent.parent
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        from scripts.strategy_research_loop import run_cycle

    if once or dry_run:
        try:
            run_cycle(config, dry_run=dry_run, case_id=case_id)
        except Exception:
            _logger.exception("Research cycle failed.")
            sys.exit(1)
    else:
        import time as _time

        _logger.info(
            "Starting daemon mode (interval=%.1fh).",
            config.loop.interval_hours,
        )
        while True:
            try:
                run_cycle(config, case_id=case_id)
            except Exception:
                _logger.exception("Research cycle failed — will retry next interval.")
            _logger.info(
                "Sleeping %.1fh until next cycle...",
                config.loop.interval_hours,
            )
            _time.sleep(config.loop.interval_hours * 3600)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src.outcomes",
        description="Argus Outcome Engine CLI - outcomes, coverage, "
                    "uptime, gap analysis",
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--db", default=None,
        help="Override database path (overrides database.path from config)",
    )
    sub = parser.add_subparsers(dest="command")

    # ── list ──
    sub.add_parser("list",
                    help="List bar keys with coverage diagnostics")

    # ── list-outcomes ──
    sub.add_parser("list-outcomes",
                    help="List outcome keys with status counts")

    # ── heartbeats ──
    sub.add_parser("heartbeats",
                    help="List heartbeat components, counts, and DB path")

    # ── gaps ──
    gp = sub.add_parser("gaps",
                         help="Show top N timestamp gaps for a bar key")
    gp.add_argument("--provider", required=True)
    gp.add_argument("--symbol", required=True)
    gp.add_argument("--bar", type=int, default=60)
    gp.add_argument("--top", type=int, default=10,
                    help="Number of largest gaps to show (default: 10)")
    gp.add_argument("--threshold", type=int, default=300,
                    help="Gap threshold in seconds (default: 300 = 5min)")

    # ── uptime ──
    up = sub.add_parser("uptime",
                         help="Show system uptime from heartbeat records")
    up.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    up.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    up.add_argument("--component", default="orchestrator",
                    help="Heartbeat component name (default: orchestrator)")
    up.add_argument("--threshold", type=int, default=120,
                    help="Gap threshold in seconds to declare OFF (default: 120)")

    # ── diagnose ──
    dx = sub.add_parser("diagnose",
                         help="Uptime-aware bar coverage diagnosis")
    dx.add_argument("--provider", required=True)
    dx.add_argument("--symbol", required=True)
    dx.add_argument("--bar", type=int, default=60)
    dx.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    dx.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    dx.add_argument("--component", default="orchestrator",
                    help="Heartbeat component (default: orchestrator)")
    dx.add_argument("--hb-threshold", type=int, default=120,
                    help="Heartbeat gap threshold in seconds (default: 120)")

    # ── backfill ──
    bf = sub.add_parser("backfill",
                         help="Backfill outcomes for a date range")
    bf.add_argument("--provider", required=True)
    bf.add_argument("--symbol", required=True)
    bf.add_argument("--bar", type=int, default=60)
    bf.add_argument("--start", required=True)
    bf.add_argument("--end", required=True)

    # ── backfill-all ──
    bfa = sub.add_parser("backfill-all",
                          help="Backfill outcomes for ALL bar keys")
    bfa.add_argument("--start", required=True)
    bfa.add_argument("--end", required=True)

    # ── coverage ──
    cov = sub.add_parser("coverage",
                          help="Show bar and outcome coverage stats")
    cov.add_argument("--provider", default=None)
    cov.add_argument("--symbol", default=None)

    # ── health ──
    sub.add_parser("health",
                    help="Show collector health (last bar time per source/symbol)")

    # ── discover ──
    disc = sub.add_parser("discover",
                           help="Discover tradeable Bybit perpetual instruments")
    disc.add_argument("--bases", default=None,
                      help="Comma-separated base coins to filter "
                           "(e.g. BTC,ETH,SOL). Omit for all.")
    disc.add_argument("--quote", default="USDT",
                      help="Quote coin filter (default: USDT)")

    # ── backfill-bars ──
    bfb = sub.add_parser("backfill-bars",
                          help="Backfill missing 1m bars from Bybit REST API")
    bfb.add_argument("--symbol", required=True,
                     help="Bybit symbol (e.g. BTCUSDT)")
    bfb.add_argument("--start", default=None,
                     help="Start date (YYYY-MM-DD). Omit = auto from last bar.")
    bfb.add_argument("--end", default=None,
                     help="End date (YYYY-MM-DD). Omit = now.")

    # ── research-loop ──
    rl = sub.add_parser("research-loop",
                         help="Run the Hades strategy research loop "
                              "(outcomes -> packs -> experiments -> evaluation)")
    rl.add_argument("--research-config", dest="config_rl",
                    default="config/research_loop.yaml",
                    help="Path to research loop YAML config "
                         "(default: config/research_loop.yaml)")
    rl.add_argument("--once", action="store_true", default=False,
                    help="Run a single cycle and exit (default: daemon mode)")
    rl.add_argument("--dry-run", action="store_true", default=False,
                    help="Validate config and log steps without executing")
    rl.add_argument("--case-id", default=None,
                    help="Pantheon case ID to associate with this research run")

    args = parser.parse_args()

    dispatch = {
        "list": _cmd_list,
        "list-outcomes": _cmd_list_outcomes,
        "heartbeats": _cmd_heartbeats,
        "gaps": _cmd_gaps,
        "uptime": _cmd_uptime,
        "diagnose": _cmd_diagnose,
        "backfill": _cmd_backfill,
        "backfill-all": _cmd_backfill_all,
        "coverage": _cmd_coverage,
        "health": _cmd_health,
        "discover": _cmd_discover,
        "backfill-bars": _cmd_backfill_bars,
        "research-loop": _cmd_research_loop,
    }

    handler = dispatch.get(args.command)
    if handler:
        asyncio.run(handler(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
