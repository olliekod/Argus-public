from __future__ import annotations

from src.analysis.execution_model import ExecutionModel
from src.analysis.replay_harness import MarketDataSnapshot, ReplayConfig, ReplayHarness
from src.core.outcome_engine import BarData
from src.strategies.vrp_credit_spread import VRPCreditSpreadStrategy


def test_vrp_replay_emits_non_zero_trades_when_iv_exceeds_rv():
    start_ms = 1_700_000_000_000
    bars = [
        BarData(timestamp_ms=start_ms, open=100.0, high=101.0, low=99.5, close=100.5, volume=1_000.0),
        BarData(timestamp_ms=start_ms + 60_000, open=100.5, high=101.2, low=100.0, close=101.0, volume=1_100.0),
    ]

    outcomes = [
        {
            "provider": "alpaca",
            "symbol": "SPY",
            "timestamp_ms": start_ms,
            "window_end_ms": start_ms + 60_000,
            "bar_duration_seconds": 60,
            "realized_vol": 0.10,
            "status": "OK",
        }
    ]

    regimes = [
        {
            "scope": "SPY",
            "timestamp_ms": start_ms,
            "vol_regime": "VOL_NORMAL",
            "trend_regime": "TREND_UP",
        }
    ]

    snapshots = [
        MarketDataSnapshot(
            symbol="SPY",
            recv_ts_ms=start_ms + 50_000,
            underlying_price=450.0,
            atm_iv=0.30,
            source="tastytrade",
        )
    ]

    strategy = VRPCreditSpreadStrategy(thresholds={"min_vrp": 0.05})
    harness = ReplayHarness(
        bars=bars,
        outcomes=outcomes,
        strategy=strategy,
        execution_model=ExecutionModel(),
        config=ReplayConfig(starting_cash=10_000.0, bar_duration_seconds=60),
        snapshots=snapshots,
        regimes=regimes,
    )

    result = harness.run()

    assert result.execution_summary["fills"] > 0
    assert result.execution_summary["rejects"] == 0
