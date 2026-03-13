"""
Soak Summary Builder
====================

Aggregates telemetry from all components into a single JSON payload
for the ``/debug/soak`` endpoint and ``python -m argus.soak`` CLI.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


def build_soak_summary(
    *,
    bus=None,
    bar_builder=None,
    persistence=None,
    feature_builder=None,
    regime_detector=None,
    resource_monitor=None,
    guardian=None,
    tape_recorder=None,
    providers: Optional[Dict[str, Any]] = None,
    detectors: Optional[Dict[str, Any]] = None,
    polymarket_gamma=None,
    polymarket_clob=None,
    polymarket_watchlist=None,
    bybit_ws=None,
) -> Dict[str, Any]:
    """Build a single-payload soak summary.

    All parameters are optional — missing components produce
    ``None`` entries so the summary is always valid JSON.
    """
    now = time.time()
    summary: Dict[str, Any] = {"timestamp": now}

    # 1) Component health (uptime, heartbeat age, health_status)
    components: Dict[str, Any] = {}

    if bar_builder:
        try:
            components["bar_builder"] = bar_builder.get_status()
        except Exception as e:
            components["bar_builder"] = {"error": str(e)}

    if persistence:
        try:
            components["persistence"] = persistence.get_status()
        except Exception as e:
            components["persistence"] = {"error": str(e)}

    if feature_builder:
        try:
            components["feature_builder"] = feature_builder.get_status()
        except Exception as e:
            components["feature_builder"] = {"error": str(e)}

    if regime_detector:
        try:
            components["regime_detector"] = regime_detector.get_status()
        except Exception as e:
            components["regime_detector"] = {"error": str(e)}

    summary["components"] = components

    if providers is not None:
        summary["providers"] = providers
    if detectors is not None:
        summary["detectors"] = detectors

    # 2) EventBus telemetry
    if bus:
        try:
            summary["event_bus"] = bus.get_status_summary()
        except Exception as e:
            summary["event_bus"] = {"error": str(e)}

    # 3) Persistence telemetry (extracted from component status)
    if persistence:
        try:
            ps = persistence.get_status()
            summary["persistence_telemetry"] = {
                "write_queue_depth": ps.get("extras", {}).get("write_queue_depth"),
                "bar_buffer_size": ps.get("extras", {}).get("bar_buffer_size"),
                "bars_writes_total": ps.get("extras", {}).get("bars_writes_total"),
                "signals_dropped_total": ps.get("extras", {}).get("signals_dropped_total"),
                "metrics_dropped_total": ps.get("extras", {}).get("metrics_dropped_total"),
                "heartbeats_dropped_total": ps.get("extras", {}).get("heartbeats_dropped_total"),
                "bars_dropped_total": getattr(persistence, "_bars_dropped_total", 0),
                "bar_flush_failures": ps.get("counters", {}).get("error_count", 0),
                "bar_flush_retries": getattr(persistence, "_bar_retry_count", 0),
                "persist_lag_ms": ps.get("extras", {}).get("persist_lag_ms"),
                "persist_lag_ema_ms": ps.get("extras", {}).get("persist_lag_ema_ms"),
                "persist_lag_crypto_ema_ms": ps.get("extras", {}).get("persist_lag_crypto_ema_ms"),
                "persist_lag_deribit_ema_ms": ps.get("extras", {}).get("persist_lag_deribit_ema_ms"),
                "persist_lag_equities_ema_ms": ps.get("extras", {}).get("persist_lag_equities_ema_ms"),
                "source_ts_future_clamped_total": ps.get("extras", {}).get(
                    "source_ts_future_clamped_total", 0
                ),
                "source_ts_stale_ignored_total": ps.get("extras", {}).get(
                    "source_ts_stale_ignored_total", 0
                ),
                "source_ts_units_discarded_total": ps.get("extras", {}).get(
                    "source_ts_units_discarded_total", 0
                ),
                "source_ts_missing_total": ps.get("extras", {}).get(
                    "source_ts_missing_total", 0
                ),
                # Spool overflow metrics
                "spool_active": ps.get("extras", {}).get("spool_active", False),
                "spool_bars_pending": ps.get("extras", {}).get("spool_bars_pending", 0),
                "spool_file_size": ps.get("extras", {}).get("spool_file_size", 0),
                "spool_max_bytes": ps.get("extras", {}).get("spool_max_bytes", 0),
                "bars_spooled_total": ps.get("extras", {}).get("bars_spooled_total", 0),
                "spool_write_errors": ps.get("extras", {}).get("spool_write_errors", 0),
                # Safe-pause metrics
                "ingestion_paused": ps.get("extras", {}).get("ingestion_paused", False),
                "bars_rejected_paused": ps.get("extras", {}).get("bars_rejected_paused", 0),
                "pause_entered_ts": ps.get("extras", {}).get("pause_entered_ts"),
            }
        except Exception as e:
            summary["persistence_telemetry"] = {"error": str(e)}

    # 4) Data integrity counters
    if bar_builder:
        try:
            bs = bar_builder.get_status()
            extras = bs.get("extras", {})
            summary["data_integrity"] = {
                "quotes_rejected_total": extras.get("quotes_rejected_total", 0),
                "quotes_rejected_by_symbol": extras.get("quotes_rejected_by_symbol", {}),
                "quotes_rejected_invalid_price_total": extras.get(
                    "quotes_rejected_invalid_price_total", 0
                ),
                "quotes_rejected_invalid_price_by_symbol": extras.get(
                    "quotes_rejected_invalid_price_by_symbol", {}
                ),
                "late_ticks_dropped_total": extras.get("late_ticks_dropped_total", 0),
                "late_ticks_dropped_by_symbol": extras.get("late_ticks_dropped_by_symbol", {}),
                "bars_emitted_total": extras.get("bars_emitted_total", 0),
                "bars_emitted_by_symbol": extras.get("bars_emitted_by_symbol", {}),
                "bars_emitted_recent_by_symbol": extras.get("bars_emitted_recent_by_symbol", {}),
                "bar_invariant_violations": extras.get("bar_invariant_violations", 0),
                # Rolling bars/minute (5-minute window)
                "bars_per_minute_by_symbol": extras.get("bars_per_minute_by_symbol", {}),
                "bars_per_minute_p50": extras.get("bars_per_minute_p50", 0.0),
                "bars_per_minute_p95": extras.get("bars_per_minute_p95", 0.0),
            }
        except Exception as e:
            summary["data_integrity"] = {"error": str(e)}
    if bybit_ws:
        try:
            bybit_status = bybit_ws.get_health_status()
            bybit_extras = bybit_status.get("extras", {})
            data_integrity = summary.setdefault("data_integrity", {})
            data_integrity.update(
                {
                    "bybit_invalid_quotes_total": bybit_extras.get("quotes_invalid_total", 0),
                    "invalid_quotes_by_reason": _top_n(
                        bybit_extras.get("quotes_invalid_by_reason", {}),
                    ),
                    "invalid_quotes_by_symbol": _top_n(
                        bybit_extras.get("quotes_invalid_by_symbol", {}),
                    ),
                }
            )
        except Exception as e:
            data_integrity = summary.setdefault("data_integrity", {})
            data_integrity["bybit_invalid_quotes_error"] = str(e)

    # 5) Polymarket telemetry
    poly: Dict[str, Any] = {}
    if polymarket_gamma:
        try:
            poly["gamma"] = polymarket_gamma.get_health_status()
        except Exception as e:
            poly["gamma"] = {"error": str(e)}
    if polymarket_clob:
        try:
            poly["clob"] = polymarket_clob.get_health_status()
        except Exception as e:
            poly["clob"] = {"error": str(e)}
    if polymarket_watchlist:
        try:
            if hasattr(polymarket_watchlist, "get_status"):
                poly["watchlist"] = polymarket_watchlist.get_status()
            else:
                poly["watchlist"] = {"status": "running"}
        except Exception as e:
            poly["watchlist"] = {"error": str(e)}
    if poly:
        summary["polymarket"] = poly

    # 6) Process-level resources
    if resource_monitor:
        try:
            summary["resources"] = resource_monitor.get_full_snapshot()
        except Exception as e:
            summary["resources"] = {"error": str(e)}

    # 7) Soak guards health
    if guardian:
        try:
            summary["guards"] = {
                "overall_health": guardian.get_overall_health(),
                "per_guard": guardian.health_status,
                "messages": guardian.guard_messages,
            }
        except Exception as e:
            summary["guards"] = {"error": str(e)}

    # 8) Tape recorder status
    if tape_recorder:
        try:
            summary["tape"] = tape_recorder.get_status()
        except Exception as e:
            summary["tape"] = {"error": str(e)}

    return summary


def _top_n(counter: Dict[str, int], n: int = 10) -> Dict[str, int]:
    if not counter:
        return {}
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    return {k: v for k, v in items[:n]}
