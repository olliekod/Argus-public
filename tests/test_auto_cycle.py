# Created by Oliver Meihls

# Tests for scripts/kalshi_auto_cycle.py — automated 8-hour run loop.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import scripts.kalshi_auto_cycle as ac  # noqa: E402


def test_help_flag_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "scripts/kalshi_auto_cycle.py", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert "--cycles" in result.stdout
    assert "--settings" in result.stdout
    assert "--dry-run" in result.stdout


def test_archive_dry_run_does_not_create_file(tmp_path, monkeypatch):
    # Dry run must not copy the file.
    monkeypatch.setattr(ac, "ROOT", tmp_path)
    src = tmp_path / "logs" / "paper_trades.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text('{"order_id":"x1"}\n')

    result = ac.archive_paper_trades(dry_run=True)
    assert result is not None
    assert not result.exists()  # dry run — nothing written


def test_archive_real_copies_file(tmp_path, monkeypatch):
    # Real archive must copy file into training_data/.
    monkeypatch.setattr(ac, "ROOT", tmp_path)
    src = tmp_path / "logs" / "paper_trades.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text('{"order_id":"x2"}\n')

    result = ac.archive_paper_trades(dry_run=False)
    assert result is not None
    assert result.exists()
    assert result.parent.name == "training_data"
    assert result.read_text() == '{"order_id":"x2"}\n'


def test_archive_skips_empty_file(tmp_path, monkeypatch):
    # Empty paper_trades.jsonl must return None (no archive created).
    monkeypatch.setattr(ac, "ROOT", tmp_path)
    src = tmp_path / "logs" / "paper_trades.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text("")

    result = ac.archive_paper_trades(dry_run=False)
    assert result is None


def test_archive_skips_missing_file(tmp_path, monkeypatch):
    # Missing paper_trades.jsonl must return None gracefully.
    monkeypatch.setattr(ac, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True)

    result = ac.archive_paper_trades(dry_run=False)
    assert result is None


def test_dry_run_cycle_calls_all_scripts(tmp_path, monkeypatch):
    # A full dry-run cycle (skip_farm=True, duration=0) must invoke
    # kalshi_run_pack, kalshi_apply_promotion, and reset_kalshi_paper
    # — all with dry_run=True — and archive the paper trades file.
    monkeypatch.setattr(ac, "ROOT", tmp_path)

    # Create a non-empty paper_trades.jsonl so archive is triggered
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "paper_trades.jsonl").write_text('{"order_id":"x1"}\n')

    called: list[tuple[str, bool]] = []

    def fake_run_script(script: str, *args: str, dry_run: bool = False) -> int:
        called.append((script, dry_run))
        return 0

    monkeypatch.setattr(ac, "run_script", fake_run_script)

    ac.run_cycle(
        cycle_num=1,
        total_cycles=1,
        settings="config/kalshi_family_adaptive.yaml",
        duration_hours=0.0,
        dry_run=True,
        skip_farm=True,
    )

    script_names = [c[0] for c in called]
    assert "kalshi_run_pack.py" in script_names
    assert "kalshi_apply_promotion.py" in script_names
    assert "reset_kalshi_paper.py" in script_names
    # All invocations must honour dry_run=True
    assert all(c[1] for c in called), f"Not all scripts got dry_run=True: {called}"
