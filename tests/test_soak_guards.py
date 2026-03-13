"""
Tests for SoakGuardian threshold guard logic.

Run with:  python -m pytest tests/test_soak_guards.py -v
"""

import time
from src.soak.guards import SoakGuardian, ALERT, WARN


def _make_bus_stats(**overrides):
    """Build minimal bus stats dict."""
    base = {
        "market.quotes": {
            "events_published": 1000,
            "dropped_events": 0,
            "queue_depth": 10,
        },
    }
    for topic, vals in overrides.items():
        if topic not in base:
            base[topic] = {}
        base[topic].update(vals)
    return base


def _make_bb_status(**extras_overrides):
    extras = {
        "quotes_rejected_total": 0,
        "quotes_rejected_by_symbol": {},
        "late_ticks_dropped_total": 0,
        "late_ticks_dropped_by_symbol": {},
        "bars_emitted_total": 100,
        "bars_emitted_by_symbol": {"BTC/USDT:USDT": 100},
        "last_bar_ts_by_symbol_epoch": {},
        "bar_invariant_violations": 0,
        "uptime_s": 600,
    }
    extras.update(extras_overrides)
    return {"extras": extras}


def _make_persist_status(**extras_overrides):
    extras = {
        "write_queue_depth": 10,
        "bar_buffer_size": 0,
        "persist_lag_ema_ms": None,
        "persist_lag_crypto_ema_ms": None,
        "persist_lag_deribit_ema_ms": None,
        "persist_lag_equities_ema_ms": None,
    }
    extras.update(extras_overrides)
    return {
        "extras": extras,
        "counters": {"error_count": 0},
        "last_error": None,
    }


def _make_resource(**overrides):
    base = {
        "process": {"rss_mb": 200, "cpu_percent": 5, "open_fds": 50, "psutil_available": True},
        "storage": {"disk_free_gb": 50.0, "db_size_mb": 100, "wal_size_mb": 10},
        "log_entropy": {"errors_total": 0, "warns_total": 0, "errors_last_hour": 0, "warns_last_hour": 0, "top_errors_last_hour": []},
    }
    for k, v in overrides.items():
        if k in base:
            base[k].update(v)
    return base


class TestQuoteDropsGuard:
    def test_no_drops_is_ok(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert g.health_status.get("quote_drops") == "ok"
        assert not alerts

    def test_new_drops_triggers_alert(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        # First eval — baseline
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        # Second eval — drops increased
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(**{"market.quotes": {"dropped_events": 5}}),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert any(a["guard"] == "quote_drops" for a in alerts)
        assert g.health_status["quote_drops"] == "alert"


class TestRejectedQuotesGuard:
    def test_low_rate_is_ok(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(quotes_rejected_total=1, uptime_s=600),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert g.health_status.get("rejected_quotes") == "ok"

    def test_high_rate_triggers_alert(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0, "rejected_quotes_per_min_threshold": 5})
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(
                quotes_rejected_total=100, uptime_s=60,
                quotes_rejected_by_symbol={"BTC": 100},
            ),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert any(a["guard"] == "rejected_quotes" for a in alerts)


class TestHeartbeatStalenessGuard:
    def test_fresh_heartbeats_ok(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        now = time.time()
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
            component_heartbeats={"bar_builder": now - 30, "persistence": now - 20},
        )
        assert g.health_status.get("heartbeat_missing") == "ok"

    def test_stale_heartbeat_alerts(self):
        g = SoakGuardian(config={
            "alert_cooldown_s": 0,
            "heartbeat_missing_intervals": 2,
            "heartbeat_interval_s": 60,
        })
        now = time.time()
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
            component_heartbeats={"bar_builder": now - 200},  # stale
        )
        assert any(a["guard"] == "heartbeat_missing" for a in alerts)


class TestDiskFatigueGuard:
    def test_healthy_disk_ok(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert g.health_status.get("disk_fatigue") == "ok"

    def test_low_disk_warns(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0, "disk_free_warn_gb": 5.0})
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(storage={"disk_free_gb": 3.0, "wal_size_mb": 10}),
        )
        assert any(a["guard"] == "disk_fatigue" for a in alerts)

    def test_large_wal_alerts(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0, "wal_size_alert_mb": 2048})
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(storage={"disk_free_gb": 50.0, "wal_size_mb": 3000}),
        )
        assert any("wal" in a["guard"] for a in alerts)


class TestLogFloodGuard:
    def test_normal_log_rate_ok(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(log_entropy={"errors_last_hour": 10}),
        )
        assert g.health_status.get("log_flood") == "ok"

    def test_flood_triggers_alert(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0, "log_error_flood_threshold_per_min": 50})
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(log_entropy={"errors_last_hour": 5000}),
        )
        assert any(a["guard"] == "log_flood" for a in alerts)


class TestPersistLagGuard:
    def test_crypto_lag_drives_alerts(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0, "persist_lag_sustained_s": 0})
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(persist_lag_crypto_ema_ms=20_000),
            resource_snapshot=_make_resource(),
        )
        assert any(a["guard"] == "persist_lag" for a in alerts)

    def test_deribit_lag_does_not_trigger_crypto_guard(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0, "persist_lag_sustained_s": 0})
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(
                persist_lag_crypto_ema_ms=1000,
                persist_lag_deribit_ema_ms=25_000,
                persist_lag_ema_ms=25_000,
            ),
            resource_snapshot=_make_resource(),
        )
        assert not any(a["guard"] == "persist_lag" for a in alerts)

    def test_deribit_enabled_without_use_crypto_only(self):
        g = SoakGuardian(config={
            "alert_cooldown_s": 0,
            "persist_lag_sustained_s": 0,
            "persist_lag": {"deribit_enabled": True},
        })
        assert g._cfg["persist_lag_use_crypto_only"] is True
        assert g._cfg["persist_lag_deribit_enabled"] is True

    def test_use_crypto_only_true_preserves_deribit_flag(self):
        g = SoakGuardian(config={
            "alert_cooldown_s": 0,
            "persist_lag_sustained_s": 0,
            "persist_lag": {"use_crypto_only": True, "deribit_enabled": True},
        })
        assert g._cfg["persist_lag_use_crypto_only"] is True
        assert g._cfg["persist_lag_deribit_enabled"] is True
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(
                persist_lag_crypto_ema_ms=500,
                persist_lag_deribit_ema_ms=50_000,
            ),
            resource_snapshot=_make_resource(),
        )
        assert any(a["guard"] == "persist_lag_deribit" for a in alerts)

    def test_use_crypto_only_defaults_true(self):
        g = SoakGuardian(config={
            "alert_cooldown_s": 0,
            "persist_lag": {},
        })
        assert g._cfg["persist_lag_use_crypto_only"] is True

class TestBarLivenessGuard:
    def test_active_bars_ok(self):
        g = SoakGuardian(config={
            "alert_cooldown_s": 0,
            "bar_liveness_symbols": ["BTC/USDT:USDT"],
            "bar_liveness_timeout_s": 180,
        })
        now = time.time()
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(
                last_bar_ts_by_symbol_epoch={"BTC/USDT:USDT": now - 60}
            ),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert g.health_status.get("bar_liveness") == "ok"

    def test_stale_bars_alert(self):
        g = SoakGuardian(config={
            "alert_cooldown_s": 0,
            "bar_liveness_symbols": ["BTC/USDT:USDT"],
            "bar_liveness_timeout_s": 180,
        })
        now = time.time()
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(
                last_bar_ts_by_symbol_epoch={"BTC/USDT:USDT": now - 300}
            ),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert any(a["guard"] == "bar_liveness" for a in alerts)


class TestBarsDroppedGuard:
    def test_no_bars_dropped_ok(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert g.health_status.get("bars_dropped") == "ok"

    def test_bars_dropped_alerts(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        g._bars_dropped_count = 1
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert any(a["guard"] == "bars_dropped" for a in alerts)
        assert g.health_status["bars_dropped"] == "alert"


class TestRateLimiting:
    def test_alerts_are_rate_limited(self):
        g = SoakGuardian(config={"alert_cooldown_s": 3600})
        # First eval — alert fires
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(storage={"disk_free_gb": 1.0, "wal_size_mb": 10}),
        )
        # Second eval — same condition, but rate limited
        alerts = g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(storage={"disk_free_gb": 1.0, "wal_size_mb": 10}),
        )
        # Still marked as alert, but no new alerts returned
        assert g.health_status.get("disk_fatigue") == "alert"
        assert not any(a["guard"] == "disk_fatigue" for a in alerts)


class TestOverallHealth:
    def test_all_ok(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert g.get_overall_health() == "ok"

    def test_worst_case_propagates(self):
        g = SoakGuardian(config={"alert_cooldown_s": 0})
        g._bars_dropped_count = 1
        g.evaluate(
            bus_stats=_make_bus_stats(),
            bar_builder_status=_make_bb_status(),
            persistence_status=_make_persist_status(),
            resource_snapshot=_make_resource(),
        )
        assert g.get_overall_health() == "alert"
