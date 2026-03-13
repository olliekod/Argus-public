from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import scripts.kalshi_apply_promotion as promo  # noqa: E402


def _write_run(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _settlement(
    bot_id: str,
    market_ticker: str,
    timestamp: float,
    pnl_usd: float,
    won: bool,
) -> dict:
    return {
        "type": "settlement",
        "bot_id": bot_id,
        "market_ticker": market_ticker,
        "timestamp": timestamp,
        "pnl_usd": pnl_usd,
        "won": won,
        "family": "BTC 15m",
        "side": "yes",
        "entry_price_cents": 50,
        "settlement_method": "settled",
        "source": "test",
        "quantity_contracts": 1,
    }


def _make_case_dir() -> Path:
    base = ROOT / ".tmp_testdata"
    base.mkdir(exist_ok=True)
    case_dir = base / f"case_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=False)
    return case_dir


def test_lifetime_bootstrap_and_rerun_dedup(monkeypatch):
    case_dir = _make_case_dir()
    training_dir = case_dir / "training_data"
    training_dir.mkdir()
    lifetime_path = case_dir / "kalshi_lifetime_performance.json"

    _write_run(
        training_dir / "run_20260308_010000.jsonl",
        [
            _settlement("Alpha", "MKT1", 1.0, 25.0, True),
            _settlement("Alpha", "MKT2", 2.0, 25.0, True),
            _settlement("Alpha", "MKT2", 2.0, 25.0, True),
        ],
    )
    _write_run(
        training_dir / "run_20260309_010000.jsonl",
        [
            _settlement("Beta", "MKT3", 3.0, 15.0, True),
            _settlement("Beta", "MKT4", 4.0, 15.0, True),
        ],
    )

    monkeypatch.setattr(
        promo,
        "_load_bot_params_lookup",
        lambda _: {
            "Alpha": {"min_entry_cents": 10, "max_entry_cents": 20},
            "Beta": {"min_entry_cents": 90, "max_entry_cents": 99},
        },
    )

    current_cycle = {
        "winner_zone": {"min_entry_cents": {"min": 90, "max": 90, "mean": 90.0}},
        "top_bots": [
            {
                "bot_id": "Beta",
                "fills": 2,
                "total_pnl": 30.0,
                "avg_pnl": 15.0,
                "win_rate": 1.0,
                "params": {"min_entry_cents": 90, "max_entry_cents": 99},
            }
        ],
        "bottom_bots": [],
    }

    try:
        first = promo._update_lifetime_performance(
            training_dir=training_dir,
            lifetime_path=lifetime_path,
            current_bot_performance=current_cycle,
            dwarf_names_file="unused.txt",
            top_n=1,
            min_fills=2,
        )

        assert first["total_cycles"] == 2
        assert first["total_settlements"] == 4
        assert first["top_bots"][0]["bot_id"] == "Alpha"
        assert first["winner_zone"]["min_entry_cents"]["mean"] == 10.0
        assert first["bots"]["Alpha"]["fills"] == 2
        assert sorted(first["processed_run_files"]) == [
            "run_20260308_010000.jsonl",
            "run_20260309_010000.jsonl",
        ]

        second = promo._update_lifetime_performance(
            training_dir=training_dir,
            lifetime_path=lifetime_path,
            current_bot_performance=current_cycle,
            dwarf_names_file="unused.txt",
            top_n=1,
            min_fills=2,
        )

        assert second["total_cycles"] == 2
        assert second["total_settlements"] == 4
        assert second["bots"]["Alpha"]["total_pnl"] == 50.0
        assert second["_summary"]["processed_now"] == []
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_lifetime_winner_zone_uses_lifetime_top_bot_not_current_cycle(monkeypatch):
    case_dir = _make_case_dir()
    training_dir = case_dir / "training_data"
    training_dir.mkdir()
    lifetime_path = case_dir / "kalshi_lifetime_performance.json"

    _write_run(
        training_dir / "run_20260310_010000.jsonl",
        [
            _settlement("Alpha", "MKT1", 1.0, 20.0, True),
            _settlement("Alpha", "MKT2", 2.0, 20.0, True),
            _settlement("Alpha", "MKT3", 3.0, 20.0, True),
            _settlement("Beta", "MKT4", 4.0, 50.0, True),
        ],
    )

    monkeypatch.setattr(
        promo,
        "_load_bot_params_lookup",
        lambda _: {
            "Alpha": {"min_entry_cents": 11, "max_entry_cents": 22},
            "Beta": {"min_entry_cents": 88, "max_entry_cents": 99},
        },
    )

    current_cycle = {
        "winner_zone": {"min_entry_cents": {"min": 88, "max": 88, "mean": 88.0}},
        "top_bots": [
            {
                "bot_id": "Beta",
                "fills": 1,
                "total_pnl": 50.0,
                "avg_pnl": 50.0,
                "win_rate": 1.0,
                "params": {"min_entry_cents": 88, "max_entry_cents": 99},
            }
        ],
        "bottom_bots": [],
    }

    try:
        lifetime = promo._update_lifetime_performance(
            training_dir=training_dir,
            lifetime_path=lifetime_path,
            current_bot_performance=current_cycle,
            dwarf_names_file="unused.txt",
            top_n=1,
            min_fills=2,
        )

        assert lifetime["top_bots"][0]["bot_id"] == "Alpha"
        assert lifetime["winner_zone"] == {
            "min_entry_cents": {"min": 11.0, "max": 11.0, "mean": 11.0},
            "max_entry_cents": {"min": 22.0, "max": 22.0, "mean": 22.0},
        }
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_lifetime_accumulates_fills_and_cycles_across_three_runs(monkeypatch):
    case_dir = _make_case_dir()
    training_dir = case_dir / "training_data"
    training_dir.mkdir()
    lifetime_path = case_dir / "kalshi_lifetime_performance.json"

    _write_run(
        training_dir / "run_20260308_010000.jsonl",
        [
            _settlement("Gamma", "MKT1", 1.0, 1.0, True),
            _settlement("Gamma", "MKT2", 2.0, -2.0, False),
        ],
    )
    _write_run(
        training_dir / "run_20260309_010000.jsonl",
        [
            _settlement("Gamma", "MKT3", 3.0, 3.0, True),
            _settlement("Gamma", "MKT4", 4.0, 4.0, True),
        ],
    )
    _write_run(
        training_dir / "run_20260310_010000.jsonl",
        [
            _settlement("Gamma", "MKT5", 5.0, 5.0, True),
        ],
    )

    monkeypatch.setattr(
        promo,
        "_load_bot_params_lookup",
        lambda _: {"Gamma": {"min_entry_cents": 33, "max_entry_cents": 44}},
    )

    try:
        lifetime = promo._update_lifetime_performance(
            training_dir=training_dir,
            lifetime_path=lifetime_path,
            current_bot_performance={"winner_zone": {}, "top_bots": [], "bottom_bots": []},
            dwarf_names_file="unused.txt",
            top_n=5,
            min_fills=1,
        )

        gamma = lifetime["bots"]["Gamma"]
        assert gamma["fills"] == 5
        assert gamma["wins"] == 4
        assert gamma["cycles_active"] == 3
        assert gamma["last_seen_cycle"] == "20260310_010000"
        assert gamma["total_pnl"] == 11.0
        assert gamma["avg_pnl"] == 2.2
        assert gamma["win_rate"] == 0.8
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
