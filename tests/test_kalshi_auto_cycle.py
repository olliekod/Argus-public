import tempfile
from pathlib import Path

from scripts import kalshi_auto_cycle


def test_temporary_farm_cycle_offset_updates_and_restores():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        settings_path = Path(handle.name)
        original = "argus_kalshi:\n  farm:\n    seed: 0\n"
        settings_path.write_text(original, encoding="utf-8")
    try:
        with kalshi_auto_cycle.temporary_farm_cycle_offset(str(settings_path), 3, dry_run=False):
            inside = settings_path.read_text(encoding="utf-8")
            assert "farm_cycle_offset: 3" in inside
            assert "seed: 0" in inside

        assert settings_path.read_text(encoding="utf-8") == original
    finally:
        settings_path.unlink(missing_ok=True)


def test_run_cycle_runs_promoter_on_final_cycle(monkeypatch):
    calls = []

    monkeypatch.setattr(kalshi_auto_cycle, "archive_paper_trades", lambda dry_run: None)
    monkeypatch.setattr(kalshi_auto_cycle, "run_backtest_eval", lambda dry_run, skip_cross_val=False: None)
    monkeypatch.setattr(kalshi_auto_cycle, "log", lambda msg: None)

    def _fake_run_script(script: str, *args: str, dry_run: bool = False) -> int:
        calls.append((script, args, dry_run))
        return 0

    monkeypatch.setattr(kalshi_auto_cycle, "run_script", _fake_run_script)

    kalshi_auto_cycle.run_cycle(
        cycle_num=3,
        total_cycles=3,
        settings="config/kalshi_family_adaptive.yaml",
        duration_hours=0.0,
        dry_run=False,
        skip_farm=True,
        skip_backtest=True,
        promote_every=10,
    )

    assert any(script == "kalshi_auto_promote.py" for script, _args, _dry in calls)


def test_run_cycle_skips_promoter_when_not_scheduled(monkeypatch):
    calls = []

    monkeypatch.setattr(kalshi_auto_cycle, "archive_paper_trades", lambda dry_run: None)
    monkeypatch.setattr(kalshi_auto_cycle, "run_backtest_eval", lambda dry_run, skip_cross_val=False: None)
    monkeypatch.setattr(kalshi_auto_cycle, "log", lambda msg: None)

    def _fake_run_script(script: str, *args: str, dry_run: bool = False) -> int:
        calls.append((script, args, dry_run))
        return 0

    monkeypatch.setattr(kalshi_auto_cycle, "run_script", _fake_run_script)

    kalshi_auto_cycle.run_cycle(
        cycle_num=1,
        total_cycles=3,
        settings="config/kalshi_family_adaptive.yaml",
        duration_hours=0.0,
        dry_run=False,
        skip_farm=True,
        skip_backtest=True,
        promote_every=3,
    )

    assert all(script != "kalshi_auto_promote.py" for script, _args, _dry in calls)
