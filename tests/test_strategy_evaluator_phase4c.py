import json

from src.analysis.strategy_evaluator import (
    StrategyEvaluator,
    compute_regime_sensitivity_score,
)
from src.analysis.regime_stress import map_bars_to_regime_keys
from src.core.outcome_engine import BarData


def _bar(ts: int) -> BarData:
    b = BarData(timestamp_ms=ts, open=100, high=101, low=99, close=100, volume=1)
    object.__setattr__(b, "symbol", "SPY")
    return b


def test_regime_sensitivity_score_edge_cases():
    assert compute_regime_sensitivity_score({"regime_breakdown": {}}) == 0.5

    single = {"regime_breakdown": {"regime:SPY:VOL_LOW": {"pnl": 10.0, "bars": 5}}}
    assert compute_regime_sensitivity_score(single) == 0.0


def test_regime_sensitivity_score_balanced_vs_fragile():
    balanced = {
        "regime_breakdown": {
            "regime:SPY:VOL_LOW": {"pnl": 10.0, "bars": 10},
            "regime:SPY:VOL_HIGH": {"pnl": 11.0, "bars": 10},
        }
    }
    fragile = {
        "regime_breakdown": {
            "regime:SPY:VOL_LOW": {"pnl": 50.0, "bars": 10},
            "regime:SPY:VOL_HIGH": {"pnl": 1.0, "bars": 10},
        }
    }
    assert compute_regime_sensitivity_score(balanced) > compute_regime_sensitivity_score(fragile)


def test_kill_list_output(tmp_path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()

    # Same strategy_class with large pnl dispersion => robustness penalty high.
    base = {
        "manifest": {
            "strategy_class": "DummyStrategy",
            "strategy_params": {"x": 1},
            "execution_config": "DEFAULT",
            "replay_packs": [],
            "data_sources": {},
            "environment": {},
        },
        "result": {
            "strategy_id": "DUMMY",
            "bars_replayed": 10,
            "outcomes_used": 0,
            "regimes_loaded": 1,
            "execution": {"fills": 1, "rejects": 0, "fill_rate": 1.0},
            "sessions": {"RTH": 10},
            "portfolio": {
                "starting_cash": 10000,
                "total_return_pct": 0,
                "sharpe_annualized_proxy": 0,
                "max_drawdown": 0,
                "max_drawdown_pct": 1,
                "expectancy": 0,
                "profit_factor": 1,
                "win_rate": 50,
                "total_trades": 1,
                "winners": 1,
                "losers": 0,
                "regime_breakdown": {"regime:SPY:VOL_LOW": {"pnl": 100.0, "bars": 10}},
            },
        },
    }

    for idx, pnl in enumerate([1000.0, -500.0]):
        data = json.loads(json.dumps(base))
        data["manifest"]["run_id"] = f"run{idx}"
        data["result"]["portfolio"]["total_realized_pnl"] = pnl
        data["result"]["portfolio"]["total_return_pct"] = pnl / 100
        with open(exp_dir / f"run{idx}.json", "w") as f:
            json.dump(data, f)

    evaluator = StrategyEvaluator(input_dir=str(exp_dir))
    assert evaluator.load_experiments() == 2
    evaluator.evaluate()
    assert len(evaluator.killed) >= 2
    reasons = {entry["reason"] for entry in evaluator.killed}
    assert "robustness_penalty" in reasons


def test_map_bars_to_regime_keys():
    bars = [_bar(1000), _bar(2000), _bar(3000)]
    regimes = [
        {"timestamp_ms": 1000, "scope": "SPY", "vol_regime": "VOL_LOW", "trend_regime": "TREND_UP"},
        {"timestamp_ms": 2500, "scope": "SPY", "vol_regime": "VOL_HIGH", "trend_regime": "TREND_DOWN"},
    ]
    mapping = map_bars_to_regime_keys(bars, regimes, include_session=False)
    assert "regime:SPY:VOL_LOW" in mapping[1000]
    assert "regime:SPY:VOL_LOW" in mapping[2000]
    assert "regime:SPY:VOL_HIGH" in mapping[3000]
