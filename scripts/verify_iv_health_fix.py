# Created by Oliver Meihls


import sys
import os
import time
from dataclasses import dataclass

# Add src to path
sys.path.append(os.path.abspath("src"))

from core.iv_consensus import IVConsensusEngine, IVConsensusConfig, SourceObservation, ContractKey, ATMKey

def test_iv_consensus_activity_tracking():
    print("Testing IVConsensusEngine activity tracking...")
    engine = IVConsensusEngine()
    
    # 1. Initial state
    assert engine.last_update_ms == 0
    assert engine.last_source_update_ms["public"] == 0
    assert engine.last_source_update_ms["dxlink"] == 0
    print("OK: Initial state is zeroed.")

    # 2. Add public observation
    ts_public = int(time.time() * 1000)
    # Mocking record_atm_obs call via a public helper or direct observation
    from core.iv_consensus import SourceObservation
    obs_public = SourceObservation(source="public", recv_ts_ms=ts_public, iv=0.45)
    
    # We need to use the public methods to trigger the logic
    # observe_public_snapshot is a good entry point
    @dataclass
    class MockQuote:
        strike: float
        iv: float
        delta: float = None
        gamma: float = None
        theta: float = None
        vega: float = None

    @dataclass
    class MockSnapshot:
        symbol: str
        expiration_ms: int
        recv_ts_ms: int
        atm_iv: float
        puts: list
        calls: list

    snapshot = MockSnapshot(
        symbol="SPY",
        expiration_ms=1742515200000,
        recv_ts_ms=ts_public,
        atm_iv=0.22,
        puts=[MockQuote(strike=500.0, iv=0.25)],
        calls=[]
    )
    
    engine.observe_public_snapshot(snapshot)
    
    assert engine.last_update_ms == ts_public
    assert engine.last_source_update_ms["public"] == ts_public
    assert engine.last_source_update_ms["dxlink"] == 0
    print(f"OK: Public activity tracked (ts={ts_public})")

    # 3. Add dxlink observation (more recent)
    ts_dxlink = ts_public + 1000
    
    @dataclass
    class MockDXEvent:
        event_symbol: str = ".SPY250321P500"
        volatility: float = 0.26
        receipt_time: int = ts_dxlink

    engine.observe_dxlink_greeks(MockDXEvent())
    
    assert engine.last_update_ms == ts_dxlink
    assert engine.last_source_update_ms["public"] == ts_public
    assert engine.last_source_update_ms["dxlink"] == ts_dxlink
    print(f"OK: DXLink activity tracked (ts={ts_dxlink})")

    # 4. Add older observation (should not move last_update_ms forward but might update source?)
    # The logic is: if ts > self._last_source_update_ms[source]: update
    ts_old = ts_public - 5000
    obs_old = MockSnapshot(
        symbol="SPY",
        expiration_ms=1742515200000,
        recv_ts_ms=ts_old,
        atm_iv=0.21,
        puts=[],
        calls=[]
    )
    engine.observe_public_snapshot(obs_old)
    
    assert engine.last_update_ms == ts_dxlink
    assert engine.last_source_update_ms["public"] == ts_public
    print("OK: Older observation did not regress last_update_ms.")

    print("\nSUCCESS: IVConsensusEngine activity tracking verified.")

if __name__ == "__main__":
    try:
        test_iv_consensus_activity_tracking()
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
