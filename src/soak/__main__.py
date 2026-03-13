"""
Argus Soak Summary CLI
======================

Usage::

    # Fetch from running dashboard (default http://127.0.0.1:8777)
    python -m argus.soak

    # Custom host/port
    python -m argus.soak --url http://localhost:9000

    # JSON output (for piping)
    python -m argus.soak --json

    # Export tape (if tape recorder is enabled in running instance)
    python -m argus.soak export-tape --minutes 5 --output tape.jsonl

    # Replay a tape and compare bars for determinism
    python -m argus.soak replay --tape tape.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error


def _fetch_soak(url: str) -> dict:
    """Fetch the soak summary from the running dashboard."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach Argus dashboard at {url}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print("  Is Argus running with the dashboard enabled?", file=sys.stderr)
        sys.exit(1)


def _health_icon(status: str) -> str:
    if status in ("ok", "connected"):
        return "[OK]"
    if status in ("warn", "degraded"):
        return "[!!]"
    if status in ("alert", "down", "error"):
        return "[XX]"
    return "[??]"


def _print_summary(data: dict) -> None:
    """Pretty-print the soak summary to stdout."""
    print("=" * 60)
    print("  ARGUS SOAK SUMMARY")
    print("=" * 60)

    # Guards overview
    guards = data.get("guards", {})
    overall = guards.get("overall_health", "unknown")
    print(f"\n  Overall Health: {_health_icon(overall)} {overall.upper()}")

    per_guard = guards.get("per_guard", {})
    if per_guard:
        for g, status in sorted(per_guard.items()):
            msg = guards.get("messages", {}).get(g, "")
            print(f"    {_health_icon(status)} {g}: {status}" + (f" — {msg}" if msg else ""))

    # Components
    components = data.get("components", {})
    if components:
        print(f"\n{'─' * 60}")
        print("  COMPONENTS")
        for name, info in sorted(components.items()):
            if isinstance(info, dict):
                status = info.get("status", "unknown")
                age = info.get("staleness", {}).get("age_seconds")
                age_str = f" (age: {age:.0f}s)" if age is not None else ""
                print(f"    {_health_icon(status)} {name}: {status}{age_str}")
            else:
                print(f"    [??] {name}: {info}")

    # EventBus
    event_bus = data.get("event_bus", {})
    if event_bus:
        print(f"\n{'─' * 60}")
        print("  EVENT BUS")
        for topic, stats in sorted(event_bus.items()):
            if isinstance(stats, dict):
                pub = stats.get("events_published", 0)
                drop = stats.get("dropped_events", 0)
                depth = stats.get("queue_depth", 0)
                drop_str = f" DROPPED={drop}" if drop else ""
                print(f"    {topic}: pub={pub:,} depth={depth}{drop_str}")

    # Persistence
    pt = data.get("persistence_telemetry", {})
    if pt:
        print(f"\n{'─' * 60}")
        print("  PERSISTENCE")
        print(f"    write_queue_depth: {pt.get('write_queue_depth', '?')}")
        print(f"    bar_buffer_size: {pt.get('bar_buffer_size', '?')}")
        print(f"    bars_writes_total: {pt.get('bars_writes_total', '?')}")
        print(f"    bars_dropped_total: {pt.get('bars_dropped_total', 0)}")
        print(f"    bar_flush_failures: {pt.get('bar_flush_failures', 0)}")
        print(f"    bar_flush_retries: {pt.get('bar_flush_retries', 0)}")
        dropped = []
        for k in ("signals_dropped_total", "metrics_dropped_total", "heartbeats_dropped_total"):
            v = pt.get(k, 0)
            if v:
                dropped.append(f"{k}={v}")
        if dropped:
            print(f"    drops: {', '.join(dropped)}")
        lag = pt.get("persist_lag_ema_ms")
        if lag is not None:
            print(f"    persist_lag_ema: {lag:.0f}ms")
        crypto_lag = pt.get("persist_lag_crypto_ema_ms")
        if crypto_lag is not None:
            print(f"    persist_lag_crypto_ema: {crypto_lag:.0f}ms")
        deribit_lag = pt.get("persist_lag_deribit_ema_ms")
        if deribit_lag is not None:
            print(f"    persist_lag_deribit_ema: {deribit_lag:.0f}ms")
        equities_lag = pt.get("persist_lag_equities_ema_ms")
        if equities_lag is not None:
            print(f"    persist_lag_equities_ema: {equities_lag:.0f}ms")
        counters = []
        for key in (
            "source_ts_future_clamped_total",
            "source_ts_stale_ignored_total",
            "source_ts_units_discarded_total",
            "source_ts_missing_total",
        ):
            value = pt.get(key, 0)
            if value:
                counters.append(f"{key}={value}")
        if counters:
            print(f"    source_ts: {', '.join(counters)}")
        # Spool overflow status
        if pt.get("spool_active"):
            pending = pt.get("spool_bars_pending", 0)
            size_mb = pt.get("spool_file_size", 0) / (1024 * 1024)
            total = pt.get("bars_spooled_total", 0)
            errors = pt.get("spool_write_errors", 0)
            print(f"    [!!] SPOOL ACTIVE: {pending} bars pending ({size_mb:.1f}MB)")
            print(f"         bars_spooled_total: {total}  write_errors: {errors}")

    # Data integrity
    di = data.get("data_integrity", {})
    if di:
        print(f"\n{'─' * 60}")
        print("  DATA INTEGRITY")
        print(f"    bars_emitted_total: {di.get('bars_emitted_total', 0)}")
        print(f"    quotes_rejected_total: {di.get('quotes_rejected_total', 0)}")
        print(
            "    quotes_rejected_invalid_price_total: "
            f"{di.get('quotes_rejected_invalid_price_total', 0)}"
        )
        rejected = di.get("quotes_rejected_by_symbol", {})
        if rejected:
            print(f"      by_symbol: {rejected}")
        invalid_price = di.get("quotes_rejected_invalid_price_by_symbol", {})
        if invalid_price:
            print(f"      invalid_price_by_symbol: {invalid_price}")
        print(f"    bybit_invalid_quotes_total: {di.get('bybit_invalid_quotes_total', 0)}")
        invalid_reasons = di.get("invalid_quotes_by_reason", {})
        if invalid_reasons:
            print(f"      invalid_by_reason: {invalid_reasons}")
        invalid_symbols = di.get("invalid_quotes_by_symbol", {})
        if invalid_symbols:
            print(f"      invalid_by_symbol: {invalid_symbols}")
        print(f"    late_ticks_dropped_total: {di.get('late_ticks_dropped_total', 0)}")
        print(f"    bar_invariant_violations: {di.get('bar_invariant_violations', 0)}")
        # Rolling bars/minute
        bpm = di.get("bars_per_minute_by_symbol", {})
        if bpm:
            p50 = di.get("bars_per_minute_p50", 0)
            p95 = di.get("bars_per_minute_p95", 0)
            print(f"    bars/min (5m window): p50={p50:.2f}  p95={p95:.2f}")
            for sym, rate in sorted(bpm.items()):
                print(f"      {sym}: {rate:.2f}/min")

    # Resources
    res = data.get("resources", {})
    if res:
        print(f"\n{'─' * 60}")
        print("  RESOURCES")
        proc = res.get("process", {})
        if proc.get("rss_mb") is not None:
            print(f"    RSS: {proc['rss_mb']}MB  CPU: {proc.get('cpu_percent', '?')}%  FDs: {proc.get('open_fds', '?')}")
        storage = res.get("storage", {})
        if storage.get("disk_free_gb") is not None:
            print(f"    Disk free: {storage['disk_free_gb']}GB")
        if storage.get("db_size_mb") is not None:
            print(f"    DB: {storage['db_size_mb']}MB  WAL: {storage.get('wal_size_mb', '?')}MB")
        log = res.get("log_entropy", {})
        if log:
            print(f"    Errors total: {log.get('errors_total', 0)} (last hour: {log.get('errors_last_hour', 0)})")
            print(f"    Warns total: {log.get('warns_total', 0)} (last hour: {log.get('warns_last_hour', 0)})")
            top = log.get("top_errors_last_hour", [])
            if top:
                print("    Top errors (last hour):")
                for e in top[:5]:
                    print(f"      [{e['count']}x] {e['message'][:80]}")

    # Tape
    tape = data.get("tape", {})
    if tape:
        print(f"\n{'─' * 60}")
        print("  TAPE RECORDER")
        print(f"    enabled: {tape.get('enabled', False)}")
        if tape.get("enabled"):
            print(f"    size: {tape.get('tape_size', 0)}/{tape.get('maxlen', 0)}")
            print(f"    captured: {tape.get('events_captured', 0)}  evicted: {tape.get('events_evicted', 0)}")

    print(f"\n{'=' * 60}")


def _cmd_replay(args):
    """Replay a tape file twice and compare bars."""
    from .tape import TapeRecorder, _to_ms

    tape = TapeRecorder.load_tape(args.tape)
    print(f"Loaded {len(tape)} events from {args.tape}")

    print("Run 1...")
    bars1 = TapeRecorder.replay_tape(tape)
    print(f"  → {len(bars1)} bars")

    print("Run 2...")
    bars2 = TapeRecorder.replay_tape(tape)
    print(f"  → {len(bars2)} bars")

    if len(bars1) != len(bars2):
        print(f"FAIL: bar count mismatch ({len(bars1)} vs {len(bars2)})")
        sys.exit(1)

    def _bar_key(bar):
        return (
            bar.source or "unknown",
            bar.symbol,
            getattr(bar, "bar_duration", 60),
            _to_ms(bar.timestamp),
        )

    def _bar_payload(bar):
        return {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "tick_count": bar.tick_count,
            "n_ticks": bar.n_ticks,
            "first_source_ts": _to_ms(bar.first_source_ts) if bar.first_source_ts else 0,
            "last_source_ts": _to_ms(bar.last_source_ts) if bar.last_source_ts else 0,
            "late_ticks_dropped": bar.late_ticks_dropped,
            "close_reason": bar.close_reason,
            "source_ts": _to_ms(bar.source_ts) if bar.source_ts else 0,
            "repaired": bar.repaired,
        }

    bars1_map = {_bar_key(bar): _bar_payload(bar) for bar in bars1}
    bars2_map = {_bar_key(bar): _bar_payload(bar) for bar in bars2}

    keys1 = set(bars1_map)
    keys2 = set(bars2_map)
    mismatches = 0

    if keys1 != keys2:
        missing_in_2 = sorted(keys1 - keys2)
        missing_in_1 = sorted(keys2 - keys1)
        if missing_in_2:
            key = missing_in_2[0]
            print(f"MISMATCH key missing in Run2: {key}")
            print(f"  Run1 payload: {json.dumps(bars1_map[key], sort_keys=True)}")
            print("  Run2 payload: <missing>")
        else:
            key = missing_in_1[0]
            print(f"MISMATCH key missing in Run1: {key}")
            print("  Run1 payload: <missing>")
            print(f"  Run2 payload: {json.dumps(bars2_map[key], sort_keys=True)}")
        mismatches = len(missing_in_2) + len(missing_in_1)
    else:
        for key in sorted(keys1):
            if bars1_map[key] != bars2_map[key]:
                print(f"MISMATCH at key: {key}")
                print(f"  Run1 payload: {json.dumps(bars1_map[key], sort_keys=True)}")
                print(f"  Run2 payload: {json.dumps(bars2_map[key], sort_keys=True)}")
                mismatches += 1
                break

    if mismatches:
        print(f"\nFAIL: {mismatches} bar mismatches")
        sys.exit(1)
    else:
        print(f"\nPASS: {len(bars1)} bars are bit-identical across both runs")


def _cmd_export_tape(args):
    """Fetch tape data from running instance and save to JSONL."""
    url = f"{args.url.rsplit('/', 1)[0]}/tape/export"
    if args.minutes:
        url += f"?minutes={args.minutes}"

    print(f"Fetching tape from {url} ...")
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach Argus tape endpoint at {url}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(f"ERROR: Unexpected response from server: {data}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing {len(data)} events to {args.output} ...")
    with open(args.output, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src.soak",
        description="Argus Soak Summary — single-glance system health",
    )
    sub = parser.add_subparsers(dest="command")

    # Default: fetch and display summary
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8777/debug/soak",
        help="Dashboard soak endpoint URL",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted text",
    )

    # Replay sub-command
    replay_p = sub.add_parser("replay", help="Replay a tape and verify determinism")
    replay_p.add_argument("--tape", required=True, help="Path to JSONL tape file")

    # Export sub-command
    export_p = sub.add_parser("export-tape", help="Export tape from running instance")
    export_p.add_argument("--minutes", type=int, help="Only export last N minutes")
    export_p.add_argument("--output", required=True, help="Output JSONL path")

    args = parser.parse_args()

    if args.command == "replay":
        _cmd_replay(args)
        return
    elif args.command == "export-tape":
        _cmd_export_tape(args)
        return

    # Default: fetch summary
    data = _fetch_soak(args.url)
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        _print_summary(data)


if __name__ == "__main__":
    main()
