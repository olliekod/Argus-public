import json

from argus_kalshi.runner import (
    _load_bot_stats_from_jsonl,
    _load_running_bankroll,
    _load_ui_stats_from_jsonl,
)


def test_jsonl_stats_are_scoped_by_bot_id(tmp_path):
    log_path = tmp_path / "paper_trades.jsonl"
    rows = [
        {"type": "settlement", "bot_id": "default", "pnl_usd": 10.0, "won": True, "timestamp": 1000},
        {"type": "settlement", "bot_id": "farm_001", "pnl_usd": -3.0, "won": False, "timestamp": 1001},
        {"type": "paper_fill", "bot_id": "default", "timestamp": 1002},
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    pnl, wins, total, first_ts = _load_ui_stats_from_jsonl(str(log_path), bot_id="default")
    assert pnl == 10.0
    assert wins == 1
    assert total == 1
    assert first_ts == 1000


def test_running_bankroll_is_scoped_by_bot_id(tmp_path):
    log_path = tmp_path / "paper_trades.jsonl"
    rows = [
        {"type": "settlement", "bot_id": "default", "pnl_usd": 8.0},
        {"type": "settlement", "bot_id": "farm_001", "pnl_usd": 12.0},
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    bankroll = _load_running_bankroll(5000.0, str(log_path), bot_id="default")
    assert bankroll == 5008.0


def test_load_bot_stats_from_jsonl_includes_fills_and_primary_bot_id(tmp_path):
    """Fills are restored from paper_fill records; primary_* values follow primary_bot_id."""
    log_path = tmp_path / "paper_trades.jsonl"
    rows = [
        {"type": "paper_fill", "bot_id": "farm_001", "timestamp": 1000},
        {"type": "paper_fill", "bot_id": "farm_001", "timestamp": 1001},
        {"type": "paper_fill", "bot_id": "farm_002", "timestamp": 1002},
        {"type": "settlement", "bot_id": "farm_001", "pnl_usd": 5.0, "won": True, "timestamp": 1003},
        {"type": "settlement", "bot_id": "farm_002", "pnl_usd": -2.0, "won": False, "timestamp": 1004},
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    bot_stats, primary_pnl, primary_wins, primary_losses, _, _, _, _, first_ts = _load_bot_stats_from_jsonl(
        str(log_path), primary_bot_id="farm_001"
    )
    assert bot_stats["farm_001"]["fills"] == 2
    assert bot_stats["farm_001"]["trade_count"] == 1
    assert bot_stats["farm_002"]["fills"] == 1
    assert bot_stats["farm_002"]["trade_count"] == 1
    assert primary_pnl == 5.0
    assert primary_wins == 1
    assert primary_losses == 0
    assert first_ts == 1003
