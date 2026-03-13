from __future__ import annotations

import json
import time
from pathlib import Path

from argus_kalshi.decision_tape import DecisionTapeWriter
from argus_kalshi.offline_sim import (
    SimParams,
    cross_validate,
    diff_strategies,
    evaluate_grid,
    load_tape,
    rank_results,
    simulate,
    summarize_cross_validation,
)
from argus_kalshi.settlement_index import SettlementIndex, SettlementRecord


def _hold_signal(
    ticker: str,
    *,
    ts: float,
    p_yes: float,
    yes_ask: int,
    no_ask: int,
    yes_bid: int | None = None,
    no_bid: int | None = None,
    family: str = "BTC 15m",
    qty: int = 1,
) -> dict:
    return {
        "tape_v": 1,
        "ts": ts,
        "ticker": ticker,
        "source": "mispricing_hold",
        "decision": "signal",
        "asset": "BTC",
        "family": family,
        "window_minutes": 15,
        "is_range": False,
        "p_yes": p_yes,
        "yes_ask_cents": yes_ask,
        "no_ask_cents": no_ask,
        "yes_bid_cents": yes_bid if yes_bid is not None else max(1, yes_ask - 1),
        "no_bid_cents": no_bid if no_bid is not None else max(1, no_ask - 1),
        "ev_yes": p_yes - (yes_ask / 100.0),
        "ev_no": (1.0 - p_yes) - (no_ask / 100.0),
        "time_to_settle_s": 120.0,
        "momentum_drift": 0.0,
        "trade_flow": 0.0,
        "obi": 0.0,
        "depth_pressure": 0.0,
        "delta_flow_yes": 0.0,
        "delta_flow_no": 0.0,
        "quantity_contracts": qty,
        "ctx_key": f"{family}|yes|0.10_0.20|40_55|na|far|flat",
    }


def _work_tmp(name: str) -> Path:
    path = Path("tmp") / "pytest_backtesting" / f"{name}_{time.time_ns()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_tape_writer_roundtrip():
    path = _work_tmp("roundtrip") / "tape.jsonl"
    writer = DecisionTapeWriter(path=str(path))
    for i in range(10):
        writer.write_signal({"tape_v": 1, "ts": 1000.0 + i, "decision": "signal", "ticker": f"T{i}"})
    writer.flush_and_close()
    records = load_tape(str(path))
    assert len(records) == 10
    assert records[0]["ts"] == 1000.0
    assert records[-1]["ts"] == 1009.0


def test_settlement_index_binary_outcome():
    idx = SettlementIndex()
    idx.add_record(
        SettlementRecord(
            ticker="T",
            side="yes",
            won=True,
            entry_price_cents=40,
            exit_price_cents=100,
            pnl_usd=0.58,
            fees_usd=0.02,
            settlement_method="contract_expiry",
            timestamp=1.0,
            source="mispricing_hold",
        )
    )
    idx.add_record(
        SettlementRecord(
            ticker="T",
            side="no",
            won=False,
            entry_price_cents=60,
            exit_price_cents=0,
            pnl_usd=-0.62,
            fees_usd=0.02,
            settlement_method="contract_expiry",
            timestamp=1.0,
            source="mispricing_hold",
        )
    )
    assert idx.get_binary_outcome("T") is True


def test_simulate_known_outcome():
    tape = [
        _hold_signal("T1", ts=1.0, p_yes=0.60, yes_ask=40, no_ask=60),
        _hold_signal("T2", ts=2.0, p_yes=0.80, yes_ask=60, no_ask=40),
        _hold_signal("T3", ts=3.0, p_yes=0.20, yes_ask=80, no_ask=35),
    ]
    idx = SettlementIndex()
    idx._binary_outcomes = {"T1": True, "T2": True, "T3": True}
    params = SimParams(
        min_edge_threshold=0.08,
        min_entry_cents=0,
        max_entry_cents=100,
        no_avoid_above_cents=100,
        yes_avoid_min_cents=0,
        yes_avoid_max_cents=0,
    )
    result = simulate(tape, idx, params, scenario="best")
    assert result.fills_count == 3
    assert result.wins == 2
    assert result.losses == 1
    assert abs(result.total_pnl_usd - 0.587) < 0.02


def test_cross_validator_detects_overfit():
    tmp_dir = _work_tmp("crossval")
    run0 = tmp_dir / "run0.jsonl"
    run1 = tmp_dir / "run1.jsonl"
    run0.write_text(
        "\n".join(
                json.dumps(row)
                for row in [
                    _hold_signal("R0_WIN", ts=1.0, p_yes=0.70, yes_ask=40, no_ask=60),
                    _hold_signal("R0_LOSS", ts=2.0, p_yes=0.67, yes_ask=59, no_ask=41),
                ]
            ),
        encoding="utf-8",
    )
    run1.write_text(
        "\n".join(
                json.dumps(row)
                for row in [
                    _hold_signal("R1_LOSS", ts=1.0, p_yes=0.70, yes_ask=40, no_ask=60),
                    _hold_signal("R1_WIN", ts=2.0, p_yes=0.60, yes_ask=52, no_ask=48),
                ]
            ),
        encoding="utf-8",
    )
    idx = SettlementIndex()
    idx._binary_outcomes = {
        "R0_WIN": True,
        "R0_LOSS": False,
        "R1_LOSS": False,
        "R1_WIN": True,
    }
    params_a = SimParams(
        min_edge_threshold=0.20,
        min_entry_cents=0,
        max_entry_cents=100,
        no_avoid_above_cents=100,
        yes_avoid_min_cents=0,
        yes_avoid_max_cents=0,
    )
    params_b = SimParams(
        min_edge_threshold=0.05,
        min_entry_cents=0,
        max_entry_cents=100,
        no_avoid_above_cents=100,
        yes_avoid_min_cents=0,
        yes_avoid_max_cents=0,
    )
    folds = cross_validate([str(run0), str(run1)], idx, [params_a, params_b], scenario="best", min_fills_train=1)
    summary = summarize_cross_validation(folds)
    assert summary["overfit_verdict"] == "severe"


def test_grid_evaluator_ranks_correctly():
    tape_path = _work_tmp("grid") / "tape.jsonl"
    tape_path.write_text(
        "\n".join(
                json.dumps(row)
                for row in [
                    _hold_signal("G1", ts=1.0, p_yes=0.75, yes_ask=40, no_ask=60),
                    _hold_signal("G2", ts=2.0, p_yes=0.97, yes_ask=90, no_ask=10),
                ]
            ),
        encoding="utf-8",
    )
    idx = SettlementIndex()
    idx._binary_outcomes = {"G1": True, "G2": False}
    high_edge = SimParams(
        min_edge_threshold=0.20,
        min_entry_cents=0,
        max_entry_cents=100,
        no_avoid_above_cents=100,
        yes_avoid_min_cents=0,
        yes_avoid_max_cents=0,
    )
    low_edge = SimParams(
        min_edge_threshold=0.05,
        min_entry_cents=0,
        max_entry_cents=100,
        no_avoid_above_cents=100,
        yes_avoid_min_cents=0,
        yes_avoid_max_cents=0,
    )
    results = evaluate_grid([str(tape_path)], idx, [high_edge, low_edge], scenario="best")
    ranked = rank_results(results, min_fills=1)
    assert ranked[0]["params"]["min_edge_threshold"] == 0.2


def test_strategy_diff_reports_delta():
    tape_path = _work_tmp("diff") / "diff.jsonl"
    tape_path.write_text(
        "\n".join(
                json.dumps(row)
                for row in [
                    _hold_signal("D1", ts=1.0, p_yes=0.75, yes_ask=40, no_ask=60),
                    _hold_signal("D2", ts=2.0, p_yes=0.98, yes_ask=90, no_ask=10),
                ]
            ),
        encoding="utf-8",
    )
    idx = SettlementIndex()
    idx._binary_outcomes = {"D1": True, "D2": False}
    old = SimParams(
        min_edge_threshold=0.20,
        min_entry_cents=0,
        max_entry_cents=100,
        no_avoid_above_cents=100,
        yes_avoid_min_cents=0,
        yes_avoid_max_cents=0,
        hold_tail_penalty_start_cents=100,
        hold_min_net_edge_cents=0,
        hold_entry_cost_buffer_cents=0,
        hold_min_divergence_threshold=0.0,
    )
    new = SimParams(
        min_edge_threshold=0.05,
        min_entry_cents=0,
        max_entry_cents=100,
        no_avoid_above_cents=100,
        yes_avoid_min_cents=0,
        yes_avoid_max_cents=0,
        hold_tail_penalty_start_cents=100,
        hold_min_net_edge_cents=0,
        hold_entry_cost_buffer_cents=0,
        hold_min_divergence_threshold=0.0,
    )
    diff = diff_strategies([str(tape_path)], idx, old, new, scenario="best", min_fills_for_verdict=1)
    assert diff.delta_fills_count == 1
    assert diff.verdict == "degraded"
