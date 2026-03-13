import time

from src.core.status_tracker import ActivityStatusTracker


def test_status_snapshot_includes_provider_and_detector_activity():
    now = time.time()
    tracker = ActivityStatusTracker(
        provider_names=["bybit"],
        detector_names=["IBITDetector"],
        boot_ts=now - 300,
    )
    tracker.record_provider_event(
        "bybit", event_ts=now - 1, source_ts=now - 2, kind="quote"
    )
    tracker.record_detector_event("IBITDetector", event_ts=now - 2, kind="metric")
    tracker.record_detector_signal("IBITDetector", event_ts=now - 1)

    providers = tracker.get_provider_statuses(now=now)
    detectors = tracker.get_detector_statuses(now=now)

    assert "bybit" in providers
    assert providers["bybit"]["health"] == "ok"
    assert "IBITDetector" in detectors
    assert detectors["IBITDetector"]["health"] == "ok"
    assert detectors["IBITDetector"]["counters"]["signals_total"] == 1


def test_unknown_resolves_after_activity():
    now = time.time()
    tracker = ActivityStatusTracker(
        provider_names=["bybit"],
        detector_names=["IBITDetector"],
        boot_ts=now - 10,
    )

    providers = tracker.get_provider_statuses(now=now)
    assert providers["bybit"]["health"] == "unknown"

    tracker.record_provider_event("bybit", event_ts=now, source_ts=now, kind="quote")
    providers = tracker.get_provider_statuses(now=now)
    assert providers["bybit"]["health"] == "ok"


def test_boot_grace_period_controls_unknown():
    boot_ts = time.time()
    tracker = ActivityStatusTracker(
        provider_names=["bybit"],
        detector_names=["IBITDetector"],
        boot_ts=boot_ts,
        boot_grace_s=120,
    )

    providers = tracker.get_provider_statuses(now=boot_ts + 60)
    assert providers["bybit"]["health"] == "unknown"

    providers = tracker.get_provider_statuses(now=boot_ts + 200)
    assert providers["bybit"]["health"] == "alert"
