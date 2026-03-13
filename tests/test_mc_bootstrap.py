import json

from src.analysis.mc_bootstrap import evaluate_mc_kill, run_mc_paths
from src.analysis.strategy_evaluator import StrategyEvaluator


def test_mc_paths_deterministic_with_seed():
    trade_pnls = [50.0, -20.0, 40.0, -10.0, 15.0, -5.0]
    a = run_mc_paths(trade_pnls, starting_cash=10_000, n_paths=400, method="bootstrap", random_seed=7)
    b = run_mc_paths(trade_pnls, starting_cash=10_000, n_paths=400, method="bootstrap", random_seed=7)
    assert a == b


def test_mc_kill_bad_list_high_ruin_or_low_positive():
    trade_pnls = [20.0, 15.0, -5000.0, 10.0, -40.0]
    summary = run_mc_paths(trade_pnls, starting_cash=10_000, n_paths=1000, method="bootstrap", random_seed=3)
    decision = evaluate_mc_kill(
        summary,
        {
            "mc_ruin_prob_max": 0.10,
            "mc_fraction_positive_min": 0.50,
        },
    )
    assert decision["killed"]
    reasons = {r["reason"] for r in decision["reasons"]}
    assert "mc_fraction_positive" in reasons or "mc_ruin_prob" in reasons


def test_mc_kill_good_list_survives():
    trade_pnls = [15.0, 10.0, 20.0, 8.0, 11.0, 14.0]
    summary = run_mc_paths(trade_pnls, starting_cash=10_000, n_paths=500, method="bootstrap", random_seed=11)
    decision = evaluate_mc_kill(
        summary,
        {
            "mc_median_return_min": 0.001,
            "mc_fraction_positive_min": 0.8,
            "mc_ruin_prob_max": 0.01,
        },
    )
    assert not decision["killed"]


def test_block_bootstrap_differs_from_iid_for_clustered_series():
    clustered = [25.0] * 15 + [-30.0] * 15 + [20.0] * 15 + [-35.0] * 15
    iid = run_mc_paths(clustered, starting_cash=10_000, n_paths=500, method="iid", random_seed=5)
    block = run_mc_paths(
        clustered,
        starting_cash=10_000,
        n_paths=500,
        method="bootstrap",
        block_size=10,
        random_seed=5,
    )
    assert iid["p95_max_drawdown"] != block["p95_max_drawdown"]


def test_evaluator_includes_mc_kill_reason(tmp_path):
    exp = {
        "manifest": {
            "run_id": "abc123",
            "strategy_class": "DummyStrategy",
            "strategy_params": {"a": 1},
            "mc_bootstrap": {
                "enabled": True,
                "metrics": {
                    "median_return": -0.05,
                    "p95_max_drawdown": 0.80,
                    "ruin_probability": 0.25,
                    "fraction_positive": 0.1,
                },
                "killed": True,
                "reasons": [
                    {"reason": "mc_ruin_prob", "value": 0.25, "threshold": 0.05},
                ],
            },
        },
        "result": {
            "strategy_id": "DUMMY",
            "bars_replayed": 10,
            "outcomes_used": 0,
            "regimes_loaded": 0,
            "execution": {"fills": 1, "rejects": 0, "fill_rate": 1.0},
            "sessions": {"RTH": 10},
            "portfolio": {
                "starting_cash": 10000,
                "total_realized_pnl": -100,
                "total_return_pct": -1,
                "sharpe_annualized_proxy": -0.1,
                "max_drawdown": 80,
                "max_drawdown_pct": 0.8,
                "expectancy": -10,
                "profit_factor": 0.5,
                "win_rate": 10,
                "total_trades": 10,
                "winners": 1,
                "losers": 9,
                "regime_breakdown": {},
            },
        },
    }
    p = tmp_path / "e1.json"
    p.write_text(json.dumps(exp))

    evaluator = StrategyEvaluator(
        input_dir=str(tmp_path),
        kill_thresholds={
            "mc_fraction_positive_min": 0.2,
            "mc_ruin_prob_max": 0.1,
        },
    )
    assert evaluator.load_experiments() == 1
    evaluator.evaluate()
    reasons = {r["reason"] for r in evaluator.killed}
    assert "mc_ruin_prob" in reasons
