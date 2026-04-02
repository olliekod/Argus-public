# Created by Oliver Meihls

from __future__ import annotations

import io
import json
import os
import shutil
import uuid
from contextlib import redirect_stdout

from argus_kalshi.bus import Bus
from argus_kalshi.terminal_ui import TerminalVisualizer, calculate_alpha_score


def _render_output(vis: TerminalVisualizer, monkeypatch) -> str:
    monkeypatch.setattr("shutil.get_terminal_size", lambda fallback=(90, 40): os.terminal_size((120, 45)))
    out = io.StringIO()
    with redirect_stdout(out):
        vis._render()
    return out.getvalue()


def _workspace_temp_dir() -> str:
    path = os.path.join(os.getcwd(), "_test_artifacts", uuid.uuid4().hex)
    os.makedirs(path, exist_ok=True)
    return path


def test_leaderboard_includes_bot_with_order_activity(monkeypatch) -> None:
    vis = TerminalVisualizer(Bus(), dry_run=True)
    vis._frame = 10
    vis._ws_connected = True
    vis._bot_stats = {
        "farm_001": {"pnl": 0.0, "wins": 0, "losses": 0, "fills": 0, "orders": 1, "last_active": 0.0}
    }

    output = _render_output(vis, monkeypatch)

    assert "farm_001" in output
    assert "Awaiting initial bot activity" not in output


def test_render_fps_uses_runtime_session_start(monkeypatch) -> None:
    vis = TerminalVisualizer(Bus(), dry_run=True)
    vis._frame = 40
    vis._start_time = 1000.0           # historical uptime anchor
    vis._runtime_start_time = 1018.0   # actual process/runtime anchor

    monkeypatch.setattr("time.time", lambda: 1020.0)
    output = _render_output(vis, monkeypatch)

    # 40 frames / 2s runtime = 20 fps (rounded to integer in UI; ANSI codes may sit between "fps " and number)
    assert "fps" in output and "20" in output


def test_calculate_alpha_score_penalizes_small_samples(monkeypatch) -> None:
    monkeypatch.setattr("time.time", lambda: 2000.0)
    base = {
        "pnl": 120.0,
        "wins": 8,
        "losses": 2,
        "trade_count": 10,
        "gross_profit": 180.0,
        "gross_loss": 60.0,
        "max_drawdown": 25.0,
        "last_active": 1999.0,
    }
    small = dict(base, wins=3, losses=1, trade_count=4)

    large_score = calculate_alpha_score(base)
    small_score = calculate_alpha_score(small)

    assert large_score > small_score


def test_leaderboard_sorts_by_alpha_score(monkeypatch) -> None:
    vis = TerminalVisualizer(Bus(), dry_run=True)
    vis._frame = 10
    vis._ws_connected = True
    vis._bot_stats = {
        "farm_high_pnl_risky": {
            "pnl": 220.0,
            "wins": 6,
            "losses": 6,
            "fills": 30,
            "orders": 50,
            "trade_count": 12,
            "gross_profit": 260.0,
            "gross_loss": 40.0,
            "peak_pnl": 260.0,
            "max_drawdown": 180.0,
            "last_active": 100.0,
        },
        "farm_balanced": {
            "pnl": 180.0,
            "wins": 8,
            "losses": 2,
            "fills": 25,
            "orders": 40,
            "trade_count": 10,
            "gross_profit": 240.0,
            "gross_loss": 60.0,
            "peak_pnl": 190.0,
            "max_drawdown": 20.0,
            "last_active": 100.0,
        },
    }

    monkeypatch.setattr("time.time", lambda: 100.0)
    output = _render_output(vis, monkeypatch)

    assert output.index("farm_balanced") < output.index("farm_high_pnl_risky")


def test_unified_mode_no_primary_bot_shows_full_layout_and_null_promoted(monkeypatch) -> None:
    # With no primary bot, full layout is shown: MARKETS, STATS, ORDERS, HISTORY, leaderboard.
    vis = TerminalVisualizer(Bus(), dry_run=True, primary_bot_id=None, leaderboard_only=False)
    vis._frame = 10
    vis._ws_connected = True
    vis._bot_stats = {
        "farm_001": {
            "pnl": 5.0,
            "wins": 1,
            "losses": 0,
            "fills": 3,
            "orders": 5,
            "trade_count": 1,
            "gross_profit": 5.0,
            "gross_loss": 0.0,
            "peak_pnl": 5.0,
            "max_drawdown": 0.0,
            "last_active": 0.0,
        }
    }

    output = _render_output(vis, monkeypatch)

    assert "Bot:" in output and ("—" in output or "NULL" in output)
    assert "MARKETS" in output
    assert "STATS" in output
    assert "ORDERS" in output
    assert "HISTORY" in output
    assert "No promoted bot session yet" in output or "No primary bot" in output
    assert "lifetime fills required" not in output
    assert "farm_001" in output
    assert "Leaderboard-only farm mode enabled" not in output


def test_event_bot_id_does_not_default_missing_to_default() -> None:
    vis = TerminalVisualizer(Bus(), dry_run=True)

    class _Evt:
        pass

    evt = _Evt()
    assert vis._event_bot_id(evt) is None


def test_promoted_bot_section_shows_star_and_drawdown_warning(monkeypatch) -> None:
    tmp = _workspace_temp_dir()
    try:
        monkeypatch.chdir(tmp)
        cfg_dir = os.path.join(tmp, "config")
        os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(cfg_dir, "kalshi_promoted_bot.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "bot_id": "farm_001",
                    "promotion_timestamp": "2026-03-12T12:00:00+00:00",
                    "robustness_score": 61.25,
                    "lifetime_stats": {
                        "fills": 2400,
                        "total_pnl": 312.4,
                        "win_rate": 0.58,
                    },
                    "params": {
                        "min_entry_cents": 30,
                        "max_entry_cents": 70,
                    },
                },
                handle,
            )
        with open(os.path.join(cfg_dir, "kalshi_lifetime_performance.json"), "w", encoding="utf-8") as handle:
            json.dump({"bots": {"farm_001": {"fills": 2400}}}, handle)

        vis = TerminalVisualizer(Bus(), dry_run=True, primary_bot_id="farm_001")
        vis._frame = 10
        vis._ws_connected = True
        vis._session_pnl = 0.0
        vis._promoted_session_peak_pnl = 250.0
        vis._bot_stats = {
            "farm_001": {
                "pnl": 12.0,
                "wins": 4,
                "losses": 1,
                "fills": 10,
                "fills_e": 6,
                "fills_s": 4,
                "fills_a": 0,
                "orders": 2,
                "trade_count": 5,
                "gross_profit": 15.0,
                "gross_loss": 3.0,
                "peak_pnl": 12.0,
                "max_drawdown": 2.0,
                "last_active": 100.0,
                "start_equity": 5000.0,
            }
        }

        output = _render_output(vis, monkeypatch)

        assert "PROMOTED BOT" in output
        assert "[★] farm_001" in output
        assert "WARN drawdown" in output
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
