# Created by Oliver Meihls


import sys
import time
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.signals import SignalEvent, signal_to_dict, dict_to_signal

def test_signal_attribution():
    sig = SignalEvent(
        timestamp_ms=int(time.time() * 1000),
        symbol="SPY",
        timeframe=60,
        strategy_id="TEST_STRAT",
        config_hash="abc",
        signal_type="ENTRY",
        direction="LONG",
        case_id="TEST_CASE_123"
    )
    d = signal_to_dict(sig)
    print(f"Serialized case_id: {d.get('case_id')}")
    
    sig2 = dict_to_signal(d)
    print(f"Deserialized case_id: {sig2.case_id}")
    
    assert sig2.case_id == "TEST_CASE_123"
    print("Signal attribution verification PASSED")

if __name__ == "__main__":
    test_signal_attribution()
