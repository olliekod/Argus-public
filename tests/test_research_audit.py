import pytest
import json
from datetime import datetime, timedelta, timezone
from src.analysis.replay_harness import (
    ReplayHarness, ReplayStrategy, TradeIntent, ReplayConfig, VirtualPortfolio
)
from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig
from src.core.outcome_engine import BarData

class SimpleTrader(ReplayStrategy):
    def __init__(self, params):
        self.params = params
        self.traded = False

    @property
    def strategy_id(self) -> str: return "AUDIT_TRADER"

    def on_bar(self, bar, sim_ts_ms, session, visible_outcomes, **kwargs):
        pass

    def generate_intents(self, sim_ts_ms):
        if not self.traded:
            self.traded = True
            return [TradeIntent(symbol="SPY", side="BUY", quantity=1, intent_type="OPEN")]
        return []

    def finalize(self): return {}

def test_pnl_commission_accuracy():
    """Verify that realized PnL is net of both entry and exit commissions."""
    port = VirtualPortfolio(starting_cash=1000.0)
    
    # Entry: Buy 1 @ 100, Comm = 1.0
    pos = port.open_position("SPY", "LONG", 1, 100.0, 1000, commission=1.0, multiplier=1)
    
    # Exit: Sell 1 @ 110, Comm = 2.0
    # Gross PnL = 10.0
    # Net PnL = 10.0 - 1.0 - 2.0 = 7.0
    net_pnl = port.close_position(pos, 110.0, 2000, commission=2.0, multiplier=1)
    
    assert net_pnl == 7.0
    assert port.total_commission == 3.0
    summary = port.summary()
    assert summary["total_realized_pnl"] == 7.0
    assert summary["expectancy"] == 7.0

def test_experiment_determinism(tmp_path):
    """Verify that identical runs produce identical IDs."""
    runner = ExperimentRunner(output_dir=str(tmp_path))
    pack_path = tmp_path / "pack.json"
    with open(pack_path, "w") as f:
        json.dump({"bars": [{"timestamp_ms": 1000, "open": 100, "high": 101, "low": 99, "close": 100, "symbol": "SPY"}], "outcomes": [], "regimes": []}, f)
        
    config = ExperimentConfig(
        strategy_class=SimpleTrader,
        strategy_params={"val": 42},
        replay_pack_paths=[str(pack_path)]
    )
    
    # Run twice
    res1 = runner.run(config)
    res2 = runner.run(config)
    
    # Check that filenames and IDs match
    # pack.json is also in tmp_path, so there should be 2 total .json files if the result overwrote correctly
    files = list(tmp_path.glob("*.json"))
    result_files = [f for f in files if f.name != "pack.json"]
    assert len(result_files) == 1
    assert res1.portfolio_summary["total_realized_pnl"] == res2.portfolio_summary["total_realized_pnl"]

def test_robust_walk_forward_weekend_gaps():
    """Verify that unique trading days are counted correctly regardless of weekend gaps."""
    runner = ExperimentRunner()
    
    # Friday 2024-01-05
    friday_ts = int(datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
    # Monday 2024-01-08
    monday_ts = int(datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
    # Tuesday 2024-01-09
    tuesday_ts = int(datetime(2024, 1, 9, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
    
    bars = [
        BarData(timestamp_ms=friday_ts, open=100, high=101, low=99, close=100, volume=0),
        BarData(timestamp_ms=monday_ts, open=101, high=102, low=100, close=101, volume=0),
        BarData(timestamp_ms=tuesday_ts, open=102, high=103, low=101, close=102, volume=0),
    ]
    
    # Train 1 day, Test 1 day. Should yield 2 windows (Fri-Mon, Mon-Tue)
    windows = list(runner.split_walk_forward(bars, train_days=1, test_days=1))
    assert len(windows) == 2
    
    # Window 0: Train=Friday, Test=Monday
    assert windows[0][0][0].timestamp_ms == friday_ts
    assert windows[0][1][0].timestamp_ms == monday_ts
    
    # Window 1: Train=Monday, Test=Tuesday
    assert windows[1][0][0].timestamp_ms == monday_ts
    assert windows[1][1][0].timestamp_ms == tuesday_ts

def test_sharpe_zero_movement():
    """Verify that flat equity doesn't crash Sharpe calculation."""
    port = VirtualPortfolio(starting_cash=1000.0)
    # 5 bars of flat equity
    for i in range(5):
        port.mark_to_market({}, 1000 * i)
    
    summary = port.summary()
    assert summary["sharpe_annualized_proxy"] == 0.0
    assert summary["profit_factor"] == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
