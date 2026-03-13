import pytest
import json
from pathlib import Path
from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig
from src.analysis.replay_harness import ReplayStrategy, TradeIntent
from src.core.outcome_engine import BarData

class MockStrategy(ReplayStrategy):
    def __init__(self, params):
        self.params = params
        self.bars_seen = 0

    @property
    def strategy_id(self) -> str:
        return f"MOCK_{self.params.get('id', 0)}"

    def on_bar(self, bar, sim_ts_ms, session, visible_outcomes, **kwargs):
        self.bars_seen += 1

    def generate_intents(self, sim_ts_ms):
        return []

    def finalize(self):
        return {"bars_seen": self.bars_seen, "params": self.params}

def test_experiment_runner_basic(tmp_path):
    # 1. Create a mock replay pack
    pack_path = tmp_path / "test_pack.json"
    pack = {
        "bars": [
            {"timestamp_ms": 1000, "open": 100, "high": 101, "low": 99, "close": 100, "symbol": "SPY"},
            {"timestamp_ms": 2000, "open": 100, "high": 101, "low": 99, "close": 100, "symbol": "SPY"},
        ],
        "outcomes": [],
        "regimes": []
    }
    with open(pack_path, "w") as f:
        json.dump(pack, f)

    runner = ExperimentRunner(output_dir=str(tmp_path / "logs"))
    config = ExperimentConfig(
        strategy_class=MockStrategy,
        strategy_params={"id": 1},
        replay_pack_paths=[str(pack_path)],
        starting_cash=10000.0
    )

    result = runner.run(config)
    assert result.bars_replayed == 2
    assert result.strategy_state["bars_seen"] == 2
    assert result.strategy_state["params"]["id"] == 1

def test_parameter_sweep(tmp_path):
    pack_path = tmp_path / "test_pack.json"
    pack = {"bars": [{"timestamp_ms": 1000, "open": 100, "high": 101, "low": 99, "close": 100, "symbol": "SPY"}], "outcomes": [], "regimes": []}
    with open(pack_path, "w") as f:
        json.dump(pack, f)

    runner = ExperimentRunner(output_dir=str(tmp_path / "logs"))
    base_config = ExperimentConfig(
        strategy_class=MockStrategy,
        replay_pack_paths=[str(pack_path)]
    )
    
    param_grid = {"id": [1, 2, 3]}
    results = runner.run_parameter_grid(MockStrategy, base_config, param_grid)
    
    assert len(results) == 3
    ids = [r.strategy_state["params"]["id"] for r in results]
    assert sorted(ids) == [1, 2, 3]


def test_parameter_sweep_merges_base_params(tmp_path):
    """Verify sweep grid entries are merged with base strategy_params (P1 badge fix)."""
    pack_path = tmp_path / "test_pack.json"
    pack = {"bars": [{"timestamp_ms": 1000, "open": 100, "high": 101, "low": 99, "close": 100, "symbol": "SPY"}], "outcomes": [], "regimes": []}
    with open(pack_path, "w") as f:
        json.dump(pack, f)

    runner = ExperimentRunner(output_dir=str(tmp_path / "logs"))
    base_config = ExperimentConfig(
        strategy_class=MockStrategy,
        strategy_params={"base_key": "fixed", "id": 0},  # id=0 gets overridden by sweep
        replay_pack_paths=[str(pack_path)],
    )

    # Sweep varies only id; base_key should be inherited in every run
    param_grid = {"id": [1, 2]}
    results = runner.run_parameter_grid(MockStrategy, base_config, param_grid)

    assert len(results) == 2
    for r in results:
        assert r.strategy_state["params"]["base_key"] == "fixed"
    ids = [r.strategy_state["params"]["id"] for r in results]
    assert sorted(ids) == [1, 2]

def test_walk_forward_splitting():
    runner = ExperimentRunner()
    # 5 days of data
    day_ms = 24 * 3600 * 1000
    bars = [BarData(timestamp_ms=i * day_ms, open=100, high=101, low=99, close=100, volume=0) for i in range(10)]
    
    # Train 2 days, Test 1 day
    windows = list(runner.split_walk_forward(bars, train_days=2, test_days=1))
    
    # Window 1: Train [0, 1], Test [2]
    # Window 2: Train [1, 2], Test [3]
    # ...
    assert len(windows) > 0
    train_bars, test_bars = windows[0]
    assert len(train_bars) == 2
    assert len(test_bars) == 1
    assert train_bars[0].timestamp_ms == 0
    assert test_bars[0].timestamp_ms == 2 * day_ms

if __name__ == "__main__":
    pytest.main([__file__])

def test_experiment_runner_persists_mc_bootstrap(tmp_path):
    pack_path = tmp_path / "test_pack_mc.json"
    pack = {
        "bars": [
            {"timestamp_ms": 1000, "open": 100, "high": 101, "low": 99, "close": 100, "symbol": "SPY"},
            {"timestamp_ms": 2000, "open": 100, "high": 101, "low": 99, "close": 100, "symbol": "SPY"},
        ],
        "outcomes": [],
        "regimes": [],
    }
    with open(pack_path, "w") as f:
        json.dump(pack, f)

    runner = ExperimentRunner(output_dir=str(tmp_path / "logs"))
    config = ExperimentConfig(
        strategy_class=MockStrategy,
        strategy_params={"id": 2},
        replay_pack_paths=[str(pack_path)],
        starting_cash=10000.0,
        mc_bootstrap_enabled=True,
        mc_paths=100,
        mc_method="bootstrap",
        mc_random_seed=9,
        mc_kill_thresholds={"mc_fraction_positive_min": 0.0},
    )
    runner.run(config)

    artifacts = list((tmp_path / "logs").glob("*.json"))
    assert artifacts
    data = json.loads(artifacts[0].read_text())
    assert "mc_bootstrap" in data["manifest"]
    assert "metrics" in data["manifest"]["mc_bootstrap"]
