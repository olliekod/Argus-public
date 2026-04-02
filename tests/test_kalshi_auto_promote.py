# Created by Oliver Meihls

import json
import os
import shutil
import uuid
from pathlib import Path

from scripts.kalshi_auto_promote import DEFAULT_THRESHOLDS, run_promotion


def _write_run_file(path: Path, bot_id: str, cycle_idx: int, *, wins: int, losses: int) -> None:
    rows = []
    timestamp = 1_700_000_000 + (cycle_idx * 10_000)
    markets = ["KXBTC15M-TEST-A", "KXETH15M-TEST-B", "KXBTCH-TEST-C", "KXETHD-TEST-D"]
    for i in range(wins + losses):
        won = i < wins
        market = markets[i % len(markets)]
        rows.append(
            {
                "type": "settlement",
                "bot_id": bot_id,
                "market_ticker": market,
                "ticker": market,
                "timestamp": timestamp + i,
                "won": won,
                "pnl_usd": 1.0 if won else -0.4,
                "quantity_contracts": 1,
                "side": "yes" if i % 2 == 0 else "no",
                "source": "hold_to_expiry",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_lifetime(path: Path, bot_id: str) -> None:
    payload = {
        "generated_at": "2026-03-12T00:00:00+00:00",
        "total_cycles": 5,
        "total_settlements": 2000,
        "processed_run_files": [f"run_2026030{i}_000000.jsonl" for i in range(1, 6)],
        "bots": {
            bot_id: {
                "total_pnl": 920.0,
                "fills": 2000,
                "wins": 1300,
                "avg_pnl": 0.46,
                "win_rate": 0.65,
                "cycles_active": 5,
                "last_seen_cycle": "20260305_000000",
                "params": {
                    "min_entry_cents": 30,
                    "max_entry_cents": 70,
                    "min_edge_threshold": 0.09,
                    "persistence_window_ms": 180,
                    "scalp_min_edge_cents": 6,
                    "scalp_min_profit_cents": 6,
                },
            },
            "bad_bot": {
                "total_pnl": -10.0,
                "fills": 20,
                "wins": 8,
                "avg_pnl": -0.5,
                "win_rate": 0.4,
                "cycles_active": 1,
                "last_seen_cycle": "20260305_000000",
                "params": {},
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_settings(path: Path, promoted_bot_id: str = "") -> None:
    path.write_text(
        "\n".join(
            [
                f'promoted_bot_id: "{promoted_bot_id}"',
                "",
                "argus_kalshi:",
                "  farm:",
                "    base:",
                "      bankroll_usd: 5000.0",
                "      hold_min_divergence_threshold: 0.04",
                "      min_entry_cents: 25",
                "      max_entry_cents: 75",
                "      min_edge_threshold: 0.08",
                "      persistence_window_ms: 120",
                "      scalp_min_edge_cents: 6",
                "      scalp_min_profit_cents: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _workspace_temp_dir() -> Path:
    path = Path.cwd() / "_test_artifacts" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_promotion_writes_promoted_bot_artifacts():
    tmp_path = _workspace_temp_dir()
    try:
        config_dir = tmp_path / "config"
        logs_dir = tmp_path / "logs"
        training_dir = logs_dir / "training_data"
        config_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True)

        for cycle_idx in range(5):
            _write_run_file(
                training_dir / f"run_2026030{cycle_idx + 1}_000000.jsonl",
                "winner_bot",
                cycle_idx,
                wins=260,
                losses=140,
            )

        lifetime_path = config_dir / "kalshi_lifetime_performance.json"
        settings_path = config_dir / "kalshi_family_adaptive.yaml"
        promoted_json_path = config_dir / "kalshi_promoted_bot.json"
        history_path = logs_dir / "promotion_history.jsonl"
        _write_lifetime(lifetime_path, "winner_bot")
        _write_settings(settings_path)

        rc = run_promotion(
            lifetime_path=lifetime_path,
            tape_dir=training_dir,
            settings_path=settings_path,
            promoted_json_path=promoted_json_path,
            history_path=history_path,
            thresholds=dict(DEFAULT_THRESHOLDS),
        )

        assert rc == 0
        assert 'promoted_bot_id: "winner_bot"' in settings_path.read_text(encoding="utf-8")
        promoted = json.loads(promoted_json_path.read_text(encoding="utf-8"))
        assert promoted["bot_id"] == "winner_bot"
        assert promoted["backtest_stress_pnl"] is None
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert history[-1]["action"] == "promoted"
        assert history[-1]["promoted_bot_id"] == "winner_bot"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_promotion_no_change_for_same_bot_without_meaningful_score_delta():
    tmp_path = _workspace_temp_dir()
    try:
        config_dir = tmp_path / "config"
        logs_dir = tmp_path / "logs"
        training_dir = logs_dir / "training_data"
        config_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True)

        for cycle_idx in range(5):
            _write_run_file(
                training_dir / f"run_2026030{cycle_idx + 1}_000000.jsonl",
                "winner_bot",
                cycle_idx,
                wins=260,
                losses=140,
            )

        lifetime_path = config_dir / "kalshi_lifetime_performance.json"
        settings_path = config_dir / "kalshi_family_adaptive.yaml"
        promoted_json_path = config_dir / "kalshi_promoted_bot.json"
        history_path = logs_dir / "promotion_history.jsonl"
        _write_lifetime(lifetime_path, "winner_bot")
        _write_settings(settings_path, promoted_bot_id="winner_bot")

        first_rc = run_promotion(
            lifetime_path=lifetime_path,
            tape_dir=training_dir,
            settings_path=settings_path,
            promoted_json_path=promoted_json_path,
            history_path=history_path,
            thresholds=dict(DEFAULT_THRESHOLDS),
        )
        assert first_rc == 0

        rc = run_promotion(
            lifetime_path=lifetime_path,
            tape_dir=training_dir,
            settings_path=settings_path,
            promoted_json_path=promoted_json_path,
            history_path=history_path,
            thresholds=dict(DEFAULT_THRESHOLDS),
        )

        assert rc == 0
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert history[-1]["action"] == "no_change"
        assert history[-1]["promoted_bot_id"] == "winner_bot"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
