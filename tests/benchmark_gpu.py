import time
import torch
import math
from src.analysis.gpu_engine import get_gpu_engine

def benchmark_heston():
    print("Argus Phase 1: GPU Heston Benchmark")
    print("=" * 50)
    
    engine = get_gpu_engine()
    print(f"Device: {engine.device_name}")
    
    # Test case: IBIT at $42, 14 DTE, 65% IV
    S = 42.0
    K_short = 38.0
    K_long = 34.0
    credit = 0.80
    T = 14 / 365
    v0 = 0.65**2
    
    print(f"\nEvaluating Put Spread: ${K_short}/${K_long} (Credit: ${credit})")
    print(f"Market: S=${S}, T={T*365:.1f} days, IV={math.sqrt(v0)*100:.0f}%")
    
    # Run Heston
    start = time.perf_counter()
    result = engine.monte_carlo_pop_heston(
        S=S, short_strike=K_short, long_strike=K_long,
        credit=credit, T=T, v0=v0, simulations=1_000_000
    )
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"\n✅ Heston Result (1M paths):")
    print(f"  PoP (Terminal):    {result['pop']:.2f}%")
    print(f"  Touch Risk (Stop): {result['prob_of_touch_stop']:.2f}%")
    print(f"  Execution Time:    {elapsed:.2f}ms")
    
    # benchmark Batch Evaluation
    print("\n📊 Batch Evaluation Benchmark (400,000 Traders)")
    N = 400_000
    trader_params = torch.rand((N, 8), device=engine._device) # Mock params
    market_data = torch.tensor([65.0, 7.0, 14.0, 75.0, 5.0, 1, 0, 0, 1, 0, 0], 
                              device=engine._device)
    
    start = time.perf_counter()
    mask = engine.evaluate_traders_batch(trader_params, market_data)
    elapsed_batch = (time.perf_counter() - start) * 1000
    
    count = mask.sum().item()
    print(f"  Evaluated {N:,} traders in {elapsed_batch:.2f}ms")
    print(f"  Found {count:,} matches")
    print(f"  Speed: {N / (elapsed_batch/1000):,.0f} traders/sec")

if __name__ == "__main__":
    benchmark_heston()
