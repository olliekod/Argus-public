"""
Tests for Strategy Evaluator
==============================

Verifies:
- Deterministic ranking
- Metric edge-case handling (no trades, no losses, zero variance, NaN)
- Robustness penalty computation
- Regime-conditioned scoring
- Composite score correctness
- Walk-forward stability penalty
- Full evaluator pipeline (load → evaluate → rank)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.analysis.strategy_evaluator import (
    DEFAULT_WEIGHTS,
    StrategyEvaluator,
    _safe_float,
    compute_composite_score,
    compute_drawdown_penalty,
    compute_regime_dependency_penalty,
    compute_regime_scores,
    compute_reject_penalty,
    compute_robustness_penalty,
    compute_walk_forward_penalty,
    extract_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: synthetic experiment artifacts
# ═══════════════════════════════════════════════════════════════════════════

def _make_experiment(
    strategy_id: str = "TEST_STRATEGY",
    run_id: str = "abc123",
    strategy_class: str = "TestStrategy",
    strategy_params: Dict[str, Any] = None,
    total_pnl: float = 500.0,
    total_return_pct: float = 5.0,
    sharpe: float = 1.5,
    max_drawdown: float = 200.0,
    max_drawdown_pct: float = 2.0,
    expectancy: float = 10.0,
    profit_factor: float = 2.0,
    win_rate: float = 60.0,
    total_trades: int = 50,
    winners: int = 30,
    losers: int = 20,
    fills: int = 50,
    rejects: int = 5,
    fill_rate: float = 0.91,
    starting_cash: float = 10000.0,
    bars_replayed: int = 1000,
    regime_breakdown: Dict[str, Any] = None,
) -> Dict[str, Any]:
    return {
        "manifest": {
            "run_id": run_id,
            "strategy_class": strategy_class,
            "strategy_params": strategy_params or {},
            "replay_packs": [{"path": "test.json", "hash": "abc"}],
            "environment": {"git_commit": "test123", "timestamp": "20240101_000000"},
        },
        "result": {
            "strategy_id": strategy_id,
            "bars_replayed": bars_replayed,
            "outcomes_used": 500,
            "regimes_loaded": 100,
            "portfolio": {
                "starting_cash": starting_cash,
                "total_realized_pnl": total_pnl,
                "total_return_pct": total_return_pct,
                "sharpe_annualized_proxy": sharpe,
                "max_drawdown": max_drawdown,
                "max_drawdown_pct": max_drawdown_pct,
                "expectancy": expectancy,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "winners": winners,
                "losers": losers,
                "regime_breakdown": regime_breakdown or {},
            },
            "execution": {
                "fills": fills,
                "rejects": rejects,
                "fill_rate": fill_rate,
                "total_commission": 65.0,
                "total_slippage": 12.5,
            },
            "sessions": {"RTH": 800, "ETH": 200},
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _safe_float
# ═══════════════════════════════════════════════════════════════════════════

class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_string_number(self):
        assert _safe_float("2.5") == 2.5

    def test_none(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, 99.0) == 99.0

    def test_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf(self):
        assert _safe_float(float("inf")) == 0.0
        assert _safe_float(float("-inf")) == 0.0

    def test_invalid_string(self):
        assert _safe_float("not_a_number") == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: extract_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractMetrics:
    def test_basic_extraction(self):
        exp = _make_experiment(total_pnl=1000.0, sharpe=2.5)
        m = extract_metrics(exp)
        assert m["total_pnl"] == 1000.0
        assert m["sharpe"] == 2.5
        assert m["strategy_id"] == "TEST_STRATEGY"
        assert m["run_id"] == "abc123"

    def test_missing_fields_use_defaults(self):
        exp = {"result": {}, "manifest": {}}
        m = extract_metrics(exp)
        assert m["total_pnl"] == 0.0
        assert m["sharpe"] == 0.0
        assert m["total_trades"] == 0
        assert m["strategy_id"] == "UNKNOWN"

    def test_empty_experiment(self):
        m = extract_metrics({})
        assert m["total_pnl"] == 0.0

    def test_nan_values_handled(self):
        exp = _make_experiment(sharpe=float("nan"))
        m = extract_metrics(exp)
        assert m["sharpe"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Penalty computations
# ═══════════════════════════════════════════════════════════════════════════

class TestRejectPenalty:
    def test_high_fill_rate_no_penalty(self):
        assert compute_reject_penalty({"fill_rate": 0.95}) == 0.0
        assert compute_reject_penalty({"fill_rate": 0.80}) == 0.0

    def test_low_fill_rate_penalty(self):
        p = compute_reject_penalty({"fill_rate": 0.4})
        assert 0.0 < p < 1.0

    def test_zero_fill_rate_max_penalty(self):
        assert compute_reject_penalty({"fill_rate": 0.0}) == 1.0

    def test_missing_fill_rate(self):
        assert compute_reject_penalty({}) == 0.0


class TestDrawdownPenalty:
    def test_small_drawdown_no_penalty(self):
        assert compute_drawdown_penalty({"max_drawdown_pct": 3.0}) == 0.0

    def test_moderate_drawdown(self):
        p = compute_drawdown_penalty({"max_drawdown_pct": 25.0})
        assert 0.0 < p < 1.0

    def test_severe_drawdown_max_penalty(self):
        assert compute_drawdown_penalty({"max_drawdown_pct": 50.0}) == 1.0
        assert compute_drawdown_penalty({"max_drawdown_pct": 75.0}) == 1.0

    def test_zero_drawdown(self):
        assert compute_drawdown_penalty({"max_drawdown_pct": 0.0}) == 0.0


class TestRegimeDependencyPenalty:
    def test_balanced_regimes(self):
        breakdown = {
            "regime:SPY": {"pnl": 100.0, "bars": 50},
            "regime:EQUITIES": {"pnl": 80.0, "bars": 50},
            "session:RTH": {"pnl": 90.0, "bars": 60},
        }
        p = compute_regime_dependency_penalty({"regime_breakdown": breakdown})
        assert p == 0.0

    def test_concentrated_regime(self):
        breakdown = {
            "regime:SPY": {"pnl": 950.0, "bars": 50},
            "regime:EQUITIES": {"pnl": 10.0, "bars": 50},
            "session:RTH": {"pnl": 5.0, "bars": 60},
        }
        p = compute_regime_dependency_penalty({"regime_breakdown": breakdown})
        assert p >= 0.5

    def test_empty_breakdown(self):
        assert compute_regime_dependency_penalty({"regime_breakdown": {}}) == 0.0
        assert compute_regime_dependency_penalty({}) == 0.0


class TestRobustnessPenalty:
    def test_single_run_no_penalty(self):
        metrics = [{"run_id": "a", "strategy_class": "S", "total_pnl": 100}]
        assert compute_robustness_penalty(metrics, "a") == 0.0

    def test_consistent_pnl_low_penalty(self):
        metrics = [
            {"run_id": "a", "strategy_class": "S", "total_pnl": 100},
            {"run_id": "b", "strategy_class": "S", "total_pnl": 110},
            {"run_id": "c", "strategy_class": "S", "total_pnl": 105},
        ]
        p = compute_robustness_penalty(metrics, "a")
        assert p == 0.0  # Very low CV

    def test_fragile_high_penalty(self):
        metrics = [
            {"run_id": "a", "strategy_class": "S", "total_pnl": 1000},
            {"run_id": "b", "strategy_class": "S", "total_pnl": -500},
            {"run_id": "c", "strategy_class": "S", "total_pnl": 10},
        ]
        p = compute_robustness_penalty(metrics, "a")
        assert p > 0.0

    def test_different_strategies_isolated(self):
        metrics = [
            {"run_id": "a", "strategy_class": "A", "total_pnl": 100},
            {"run_id": "b", "strategy_class": "B", "total_pnl": -500},
        ]
        # Strategy A has only one run → 0 penalty
        assert compute_robustness_penalty(metrics, "a") == 0.0

    def test_unknown_target(self):
        assert compute_robustness_penalty([], "nonexistent") == 0.0

    def test_zero_mean_pnl(self):
        metrics = [
            {"run_id": "a", "strategy_class": "S", "total_pnl": 100},
            {"run_id": "b", "strategy_class": "S", "total_pnl": -100},
        ]
        p = compute_robustness_penalty(metrics, "a")
        assert p == 0.5  # Zero mean → fragile


class TestWalkForwardPenalty:
    def test_consistent_positive(self):
        params = {"threshold": 0.5}
        metrics = [
            {"run_id": f"w{i}", "strategy_class": "S", "strategy_params": params, "total_pnl": 50 + i * 10}
            for i in range(5)
        ]
        p = compute_walk_forward_penalty(metrics, "w0")
        assert p == 0.0

    def test_inconsistent_sign(self):
        params = {"threshold": 0.5}
        metrics = [
            {"run_id": "w0", "strategy_class": "S", "strategy_params": params, "total_pnl": 100},
            {"run_id": "w1", "strategy_class": "S", "strategy_params": params, "total_pnl": -50},
            {"run_id": "w2", "strategy_class": "S", "strategy_params": params, "total_pnl": 30},
            {"run_id": "w3", "strategy_class": "S", "strategy_params": params, "total_pnl": -80},
        ]
        p = compute_walk_forward_penalty(metrics, "w0")
        assert p > 0.0

    def test_single_window_no_penalty(self):
        metrics = [
            {"run_id": "w0", "strategy_class": "S", "strategy_params": {}, "total_pnl": 100},
        ]
        p = compute_walk_forward_penalty(metrics, "w0")
        assert p == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Regime scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestRegimeScores:
    def test_basic_regime_scores(self):
        metrics = {
            "regime_breakdown": {
                "regime:SPY": {"pnl": 200.0, "bars": 100},
                "session:RTH": {"pnl": 150.0, "bars": 300},
            }
        }
        scores = compute_regime_scores(metrics)
        assert "regime:SPY" in scores
        assert scores["regime:SPY"]["pnl"] == 200.0
        assert scores["regime:SPY"]["bars"] == 100
        assert scores["regime:SPY"]["pnl_per_bar"] == pytest.approx(2.0)

    def test_zero_bars(self):
        metrics = {
            "regime_breakdown": {
                "regime:SPY": {"pnl": 0.0, "bars": 0},
            }
        }
        scores = compute_regime_scores(metrics)
        assert scores["regime:SPY"]["pnl_per_bar"] == 0.0

    def test_empty_breakdown(self):
        assert compute_regime_scores({"regime_breakdown": {}}) == {}
        assert compute_regime_scores({}) == {}


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Composite scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestCompositeScore:
    def test_single_experiment_gets_mid_score(self):
        """A single experiment should get norm 0.5 for return and sharpe."""
        m = extract_metrics(_make_experiment())
        score = compute_composite_score(m, [m])
        assert score["components"]["return_norm"] == 0.5
        assert score["components"]["sharpe_norm"] == 0.5

    def test_better_strategy_scores_higher(self):
        exp_good = _make_experiment(
            run_id="good", total_return_pct=10.0, sharpe=3.0,
            max_drawdown_pct=2.0, fill_rate=0.95,
        )
        exp_bad = _make_experiment(
            run_id="bad", total_return_pct=-5.0, sharpe=-0.5,
            max_drawdown_pct=40.0, fill_rate=0.3,
        )
        m_good = extract_metrics(exp_good)
        m_bad = extract_metrics(exp_bad)
        all_m = [m_good, m_bad]

        score_good = compute_composite_score(m_good, all_m)
        score_bad = compute_composite_score(m_bad, all_m)

        assert score_good["composite_score"] > score_bad["composite_score"]

    def test_penalties_reduce_score(self):
        exp_clean = _make_experiment(
            run_id="clean", max_drawdown_pct=2.0, fill_rate=0.95,
        )
        exp_penalized = _make_experiment(
            run_id="penalized", max_drawdown_pct=40.0, fill_rate=0.2,
        )
        m_clean = extract_metrics(exp_clean)
        m_pen = extract_metrics(exp_penalized)
        # Use same metrics list so return/sharpe normalization is comparable
        all_m = [m_clean, m_pen]

        score_clean = compute_composite_score(m_clean, all_m)
        score_pen = compute_composite_score(m_pen, all_m)

        assert score_clean["composite_score"] > score_pen["composite_score"]


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Deterministic ranking
# ═══════════════════════════════════════════════════════════════════════════

class TestDeterministicRanking:
    def test_ranking_is_deterministic(self):
        """Running evaluate() twice on the same data produces identical rankings."""
        exps = [
            _make_experiment(run_id="r1", strategy_id="S1", total_return_pct=5.0, sharpe=1.5),
            _make_experiment(run_id="r2", strategy_id="S2", total_return_pct=8.0, sharpe=2.0),
            _make_experiment(run_id="r3", strategy_id="S3", total_return_pct=3.0, sharpe=0.5),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, exp in enumerate(exps):
                with open(os.path.join(tmpdir, f"exp_{i}.json"), "w") as f:
                    json.dump(exp, f)

            ev1 = StrategyEvaluator(input_dir=tmpdir)
            ev1.load_experiments()
            r1 = ev1.evaluate()

            ev2 = StrategyEvaluator(input_dir=tmpdir)
            ev2.load_experiments()
            r2 = ev2.evaluate()

        assert len(r1) == len(r2) == 3
        for a, b in zip(r1, r2):
            assert a["rank"] == b["rank"]
            assert a["run_id"] == b["run_id"]
            assert a["composite_score"] == b["composite_score"]

    def test_best_strategy_ranks_first(self):
        exps = [
            _make_experiment(run_id="bad", strategy_id="BAD", total_return_pct=-10.0, sharpe=-1.0, max_drawdown_pct=45.0),
            _make_experiment(run_id="good", strategy_id="GOOD", total_return_pct=15.0, sharpe=3.0, max_drawdown_pct=3.0),
            _make_experiment(run_id="mid", strategy_id="MID", total_return_pct=5.0, sharpe=1.0, max_drawdown_pct=10.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, exp in enumerate(exps):
                with open(os.path.join(tmpdir, f"exp_{i}.json"), "w") as f:
                    json.dump(exp, f)

            ev = StrategyEvaluator(input_dir=tmpdir)
            ev.load_experiments()
            rankings = ev.evaluate()

        assert rankings[0]["strategy_id"] == "GOOD"
        assert rankings[0]["rank"] == 1
        assert rankings[-1]["strategy_id"] == "BAD"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_no_trades_experiment(self):
        exp = _make_experiment(
            total_trades=0, winners=0, losers=0,
            total_pnl=0.0, total_return_pct=0.0, sharpe=0.0,
            expectancy=0.0, profit_factor=0.0, win_rate=0.0,
            fills=0, rejects=0,
        )
        m = extract_metrics(exp)
        assert m["total_trades"] == 0
        score = compute_composite_score(m, [m])
        assert not math.isnan(score["composite_score"])

    def test_no_losses_experiment(self):
        exp = _make_experiment(
            total_trades=10, winners=10, losers=0,
            profit_factor=99.9, win_rate=100.0,
        )
        m = extract_metrics(exp)
        assert m["winners"] == 10
        assert m["losers"] == 0
        score = compute_composite_score(m, [m])
        assert not math.isnan(score["composite_score"])

    def test_all_losses_experiment(self):
        exp = _make_experiment(
            total_trades=10, winners=0, losers=10,
            total_pnl=-500.0, total_return_pct=-5.0,
            profit_factor=0.0, win_rate=0.0, sharpe=-2.0,
        )
        m = extract_metrics(exp)
        score = compute_composite_score(m, [m])
        assert not math.isnan(score["composite_score"])

    def test_nan_in_experiment_json(self):
        """NaN values in JSON should not crash the evaluator."""
        exp = _make_experiment()
        exp["result"]["portfolio"]["sharpe_annualized_proxy"] = None
        m = extract_metrics(exp)
        assert m["sharpe"] == 0.0

    def test_missing_manifest(self):
        exp = {"result": {"portfolio": {}, "execution": {}}}
        m = extract_metrics(exp)
        assert m["run_id"] == ""

    def test_single_experiment(self):
        """Evaluator works with just one experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = _make_experiment()
            with open(os.path.join(tmpdir, "exp.json"), "w") as f:
                json.dump(exp, f)

            ev = StrategyEvaluator(input_dir=tmpdir)
            ev.load_experiments()
            rankings = ev.evaluate()

        assert len(rankings) == 1
        assert rankings[0]["rank"] == 1

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = StrategyEvaluator(input_dir=tmpdir)
            count = ev.load_experiments()
            assert count == 0

    def test_invalid_json_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write invalid JSON
            with open(os.path.join(tmpdir, "bad.json"), "w") as f:
                f.write("not json")
            # Write valid JSON without result key
            with open(os.path.join(tmpdir, "no_result.json"), "w") as f:
                json.dump({"foo": "bar"}, f)
            # Write valid experiment
            with open(os.path.join(tmpdir, "good.json"), "w") as f:
                json.dump(_make_experiment(), f)

            ev = StrategyEvaluator(input_dir=tmpdir)
            count = ev.load_experiments()
            assert count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_load_evaluate_save(self):
        exps = [
            _make_experiment(run_id="r1", strategy_id="S1", total_return_pct=5.0),
            _make_experiment(run_id="r2", strategy_id="S2", total_return_pct=10.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            os.makedirs(input_dir)
            for i, exp in enumerate(exps):
                with open(os.path.join(input_dir, f"exp_{i}.json"), "w") as f:
                    json.dump(exp, f)

            output_path = os.path.join(tmpdir, "rankings.json")
            ev = StrategyEvaluator(input_dir=input_dir, output_dir=tmpdir)
            ev.load_experiments()
            ev.evaluate()
            saved = ev.save_rankings(output_path=output_path)

            assert os.path.exists(saved)
            with open(saved) as f:
                data = json.load(f)

            assert data["experiment_count"] == 2
            assert len(data["rankings"]) == 2
            assert data["rankings"][0]["rank"] == 1
            assert data["rankings"][1]["rank"] == 2

    def test_console_summary(self):
        exp = _make_experiment()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "exp.json"), "w") as f:
                json.dump(exp, f)

            ev = StrategyEvaluator(input_dir=tmpdir)
            ev.load_experiments()
            ev.evaluate()
            summary = ev.print_summary()

        assert "STRATEGY RANKINGS" in summary
        assert "TEST_STRATEGY" in summary

    def test_manifest_references_included(self):
        exp = _make_experiment()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "exp.json"), "w") as f:
                json.dump(exp, f)

            ev = StrategyEvaluator(input_dir=tmpdir)
            ev.load_experiments()
            rankings = ev.evaluate()

        assert "manifest_ref" in rankings[0]
        assert rankings[0]["manifest_ref"]["run_id"] == "abc123"
