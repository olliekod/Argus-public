import pytest
from typing import List, Dict, Any
from src.analysis.replay_harness import (
    ReplayHarness, ReplayStrategy, TradeIntent, ReplayConfig, VirtualPortfolio
)
from src.core.outcome_engine import BarData

class SimpleMomentumStrategy(ReplayStrategy):
    @property
    def strategy_id(self) -> str:
        return "TEST_MOMENTUM"

    def on_bar(self, bar: BarData, sim_ts_ms: int, session_regime: str, 
               visible_outcomes: Dict[int, Any], **kwargs) -> None:
        self.last_close = bar.close
        # Just track regimes for verification
        self.visible_regimes = kwargs.get("visible_regimes", {})

    def generate_intents(self, sim_ts_ms: int) -> List[TradeIntent]:
        # Buy on every bar to see equity move
        return [TradeIntent(symbol="SPY", side="BUY", quantity=1, intent_type="OPEN")]

    def finalize(self) -> Dict[str, Any]:
        return {}

class MockExecutionModel:
    def __init__(self):
        class Ledger:
            def summary(self): return {"fills": 1, "rejects": 0}
            @property
            def fills_count(self): return 1
            @property
            def rejects_count(self): return 0
        self.ledger = Ledger()

    def reset(self):
        """Reset ledger — no-op for mock since state is stateless."""
        pass

    def attempt_fill(self, quote, side, quantity, sim_ts_ms, multiplier=100):
        from src.analysis.execution_model import FillResult
        return FillResult(filled=True, fill_price=quote.ask, quantity=quantity, side=side, commission=1.0)

def test_regime_conditioned_reporting():
    # 1. Setup mock data (2024-01-03 10:00 AM ET = 15:00 UTC = 1704294000 s)
    start_ts = 1704294000 * 1000
    
    def create_bar_with_symbol(ts, close):
        b = BarData(timestamp_ms=ts, open=close, high=close, low=close, close=close, volume=1000)
        # Monkey-patch symbol since BarData is frozen=True dataclass
        object.__setattr__(b, 'symbol', 'SPY')
        return b

    bars = [
        create_bar_with_symbol(start_ts, 100),
        create_bar_with_symbol(start_ts + 60000, 103),
        create_bar_with_symbol(start_ts + 120000, 104),
        create_bar_with_symbol(start_ts + 180000, 101),
    ]
    
    # 2. Setup mock regimes
    regimes = [
        {"timestamp_ms": start_ts + 0, "scope": "SPY", "vol_regime": "VOL_LOW", "trend_regime": "TREND_UP"},
        {"timestamp_ms": start_ts + 180000, "scope": "SPY", "vol_regime": "VOL_HIGH", "trend_regime": "TREND_DOWN"},
    ]
    
    strategy = SimpleMomentumStrategy()
    exec_model = MockExecutionModel()
    
    harness = ReplayHarness(
        bars=bars,
        outcomes=[],
        strategy=strategy,
        execution_model=exec_model,
        regimes=regimes,
        config=ReplayConfig(starting_cash=10000, multiplier=1) # Use 1 for simplicity
    )
    
    # 3. Run replay
    result = harness.run()
    summary = result.summary()
    
    # Debug: print equity curve
    print("\nEquity Curve:")
    for i, s in enumerate(harness.portfolio.equity_curve):
        print(f"Bar {i}: equity={s.equity} cash={s.cash} pos={s.open_positions} regimes={s.regimes}")

    # 4. Verify breakdown
    breakdown = summary["portfolio"]["regime_breakdown"]
    
    # We should have entries for session, regime:SPY (vol), and regime:SPY_trend
    assert "session:RTH" in breakdown
    assert "regime:SPY:VOL_LOW" in breakdown
    assert "regime:SPY:VOL_HIGH" in breakdown
    assert "regime:SPY_trend:TREND_UP" in breakdown
    assert "regime:SPY_trend:TREND_DOWN" in breakdown
    
    # Bars count verification:
    # Bar 1 (1000+60): SPY=VOL_LOW (at 1000)
    # Bar 2 (2000+60): SPY=VOL_LOW
    # Bar 3 (3000+60): SPY=VOL_LOW
    # Bar 4 (4000+60): SPY=VOL_HIGH (at 4000)
    
    assert breakdown["regime:SPY:VOL_LOW"]["bars"] == 2
    assert breakdown["regime:SPY:VOL_HIGH"]["bars"] == 2
    
    # VOL_LOW PnL = -101 (bar 0) + -101 (bar 1) = -202.0
    # VOL_HIGH PnL = -103 (bar 2) + -111 (bar 3) = -214.0
    
    assert breakdown["regime:SPY:VOL_LOW"]["pnl"] == -202.0
    assert breakdown["regime:SPY:VOL_HIGH"]["pnl"] == -214.0

if __name__ == "__main__":
    test_regime_conditioned_reporting()
