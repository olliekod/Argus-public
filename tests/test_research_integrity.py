import pytest
import json
import hashlib
from datetime import datetime, timezone
from src.analysis.replay_harness import VirtualPortfolio, PortfolioSnapshot
from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig, ReplayStrategy
from src.core.outcome_engine import BarData

def test_sharpe_proxy_math():
    """Verify Sharpe formula using a simple known return sequence."""
    port = VirtualPortfolio(starting_cash=1000.0)
    
    # Create a 10% gain over 100 bars. 
    # Returns = 0.001 per bar roughly.
    for i in range(101):
        # Linearly increasing equity
        equity = 1000.0 + (i * 1.0)
        port._equity_curve.append(PortfolioSnapshot(
            ts_ms=i * 60000,
            equity=equity,
            cash=equity,
            open_positions=0,
            unrealized_pnl=0.0,
            realized_pnl=i*1.0
        ))
    
    summary = port.summary()
    # Sharpe should be high because there's zero variance in returns (all 0.001 approx)
    # Actually stdev will be 0 if all returns are identical.
    # Let's add some jitter
    port._equity_curve = []
    import statistics
    returns = [0.001, -0.0005, 0.002, -0.001, 0.0015] * 20 # 100 returns
    equity = 1000.0
    port._equity_curve.append(PortfolioSnapshot(0, equity, equity, 0, 0, 0))
    for i, r in enumerate(returns):
        equity *= (1.0 + r)
        port._equity_curve.append(PortfolioSnapshot((i+1)*60000, equity, equity, 0, 0, 0))
    
    summary = port.summary()
    sharpe = summary["sharpe_annualized_proxy"]
    assert sharpe != 0.0
    
    # Verify annualization constant usage
    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    import math
    expected_sharpe = round((mean_r / std_r) * math.sqrt(252 * 6.5 * 60), 2)
    assert sharpe == expected_sharpe

class ManifestStrategy(ReplayStrategy):
    def __init__(self, params): self.params = params
    @property
    def strategy_id(self) -> str: return "MANIFEST_TEST"
    def on_bar(self, *args, **kwargs): pass
    def generate_intents(self, *args, **kwargs): return []
    def finalize(self): return {}

def test_manifest_integrity(tmp_path):
    """Verify that every run produces a rich manifest."""
    runner = ExperimentRunner(output_dir=str(tmp_path))
    pack_path = tmp_path / "integrity_pack.json"
    pack_data = {"bars": [{"timestamp_ms": 1000, "open": 100, "high": 101, "low": 99, "close": 100, "symbol": "SPY"}], "outcomes": [], "regimes": []}
    with open(pack_path, "w") as f: json.dump(pack_data, f)
    
    config = ExperimentConfig(
        strategy_class=ManifestStrategy,
        strategy_params={"alpha": 0.5, "beta": 0.1},
        replay_pack_paths=[str(pack_path)],
        tag="INTEGRITY"
    )
    
    result = runner.run(config)
    
    # Find artifact
    files = list(tmp_path.glob("*.json"))
    manifest_file = [f for f in files if f.name != "integrity_pack.json"][0]
    
    with open(manifest_file, "r") as f:
        data = json.load(f)
        
    assert "manifest" in data
    m = data["manifest"]
    assert "run_id" in m
    assert "strategy_params" in m
    assert m["strategy_params"] == {"alpha": 0.5, "beta": 0.1}
    assert "replay_packs" in m
    assert len(m["replay_packs"]) == 1
    assert "hash" in m["replay_packs"][0]
    assert m["replay_packs"][0]["hash"] != "ERROR"
    assert "environment" in m
    assert "git_commit" in m["environment"]

def test_walk_forward_day_alignment():
    """Verify that walk-forward splits align with the first bar of the day."""
    runner = ExperimentRunner()
    
    # 3 days, 10 bars each
    day_ms = 24 * 3600 * 1000
    bars = []
    for d in range(3):
        ts = int(datetime(2024, 1, 1 + d, 9, 30, tzinfo=timezone.utc).timestamp() * 1000)
        for b in range(10):
            bars.append(BarData(timestamp_ms=ts + (b * 60000), open=100, high=101, low=99, close=100, volume=0))
            
    # Train 1 day, Test 1 day. 
    # Should get 2 windows: (Day 1, Day 2) and (Day 2, Day 3)
    windows = list(runner.split_walk_forward(bars, train_days=1, test_days=1))
    assert len(windows) == 2
    
    # Verify alignment: First bar of test window should be Day 2 Start
    day2_start = int(datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert windows[0][1][0].timestamp_ms == day2_start
    
    day3_start = int(datetime(2024, 1, 3, 9, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert windows[1][1][0].timestamp_ms == day3_start

if __name__ == "__main__":
    pytest.main([__file__])
