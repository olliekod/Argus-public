"""
Tests for scripts/kalshi_aggregate_training.py — merge run archives into classifier dataset.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import scripts.kalshi_aggregate_training as agg  # noqa: E402


def test_help_flag():
    result = subprocess.run(
        [sys.executable, "scripts/kalshi_aggregate_training.py", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert "--input-dir" in result.stdout
    assert "--output-dir" in result.stdout


def test_aggregation_deduplicates_and_produces_csv(tmp_path):
    """Two run files with one overlapping order_id → deduplicated output."""
    run_dir = tmp_path / "training_data"
    run_dir.mkdir()

    records = [
        {
            "order_id": "a1",
            "fill_price_cents": 55,
            "outcome": "win",
            "family": "BTC 15m",
            "side": "yes",
            "edge_at_entry": 0.12,
            "tts_at_entry": 600,
            "drift_at_entry": 0.00003,
            "flow_at_entry": 0.2,
            "obi_at_entry": 0.1,
            "strike_distance_pct": 0.01,
            "near_money": False,
        },
        {
            "order_id": "a2",
            "fill_price_cents": 40,
            "outcome": "loss",
            "family": "ETH 15m",
            "side": "no",
            "edge_at_entry": 0.08,
            "tts_at_entry": 300,
            "drift_at_entry": -0.00001,
            "flow_at_entry": -0.3,
            "obi_at_entry": -0.2,
            "strike_distance_pct": 0.005,
            "near_money": True,
        },
    ]

    # run_1: both records
    (run_dir / "run_20260308_010000.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    # run_2: duplicate of a1 — should be dropped
    (run_dir / "run_20260308_090000.jsonl").write_text(
        json.dumps(records[0]) + "\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    summary = agg.aggregate(input_dir=run_dir, output_dir=out_dir)

    assert summary["total_files"] == 2
    assert summary["unique_records"] == 2  # deduplicated

    # Verify CSV
    csv_files = list(out_dir.glob("labeled_*.csv"))
    assert len(csv_files) == 1
    rows = list(csv.DictReader(csv_files[0].open()))
    assert len(rows) == 2

    # Verify JSONL
    jsonl_files = list(out_dir.glob("aggregate_*.jsonl"))
    assert len(jsonl_files) == 1
    lines = [l for l in jsonl_files[0].read_text().strip().splitlines() if l.strip()]
    assert len(lines) == 2


def test_empty_input_dir_returns_zero(tmp_path):
    """Empty input directory must return gracefully with zero counts."""
    run_dir = tmp_path / "training_data"
    run_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    summary = agg.aggregate(input_dir=run_dir, output_dir=out_dir)
    assert summary["total_files"] == 0
    assert summary["unique_records"] == 0
    # No output files created
    assert not list(out_dir.glob("labeled_*.csv"))
    assert not list(out_dir.glob("aggregate_*.jsonl"))


def test_records_without_order_id_are_included(tmp_path):
    """Records lacking order_id must still be included (not silently dropped)."""
    run_dir = tmp_path / "training_data"
    run_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    recs = [
        {"fill_price_cents": 50, "outcome": "win", "family": "BTC 15m"},
        {"fill_price_cents": 45, "outcome": "loss", "family": "ETH 60m"},
    ]
    (run_dir / "run_20260308_000000.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n"
    )

    summary = agg.aggregate(input_dir=run_dir, output_dir=out_dir)
    assert summary["unique_records"] == 2
