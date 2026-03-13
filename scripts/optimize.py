"""
Ultra Optimize Script
=====================

Maximum thoroughness optimization for powerful hardware.
Uses Monte Carlo simulation and walk-forward validation.

Run: python scripts/optimize.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.ultra_optimizer import UltraOptimizer


def main():
    """Run ultra optimization."""
    print()
    print("=" * 70)
    print("ULTRA STRATEGY OPTIMIZATION")
    print("=" * 70)
    
    optimizer = UltraOptimizer(symbol="BITO", num_threads=8)
    
    total_combos = optimizer._count_combinations()
    
    print()
    print("This is a THOROUGH optimization with:")
    print(f"  * {total_combos} parameter combinations")
    print("  * Full history: Oct 2021 to yesterday")
    print("  * Walk-forward validation (70% train, 30% test)")
    print("  * Monte Carlo simulation (500 randomized runs)")
    print()
    print("Estimated time: 2-5 minutes")
    print()
    
    report = optimizer.optimize(
        min_trades=30,
        monte_carlo_runs=500,
        show_progress=True,
    )
    
    print()
    print(optimizer.format_report(report))
    
    # Save report
    report_path = Path("data/optimization_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(optimizer.format_report(report))
    
    print(f"\nReport saved: {report_path}")
    
    if report.should_apply:
        print()
        print("=" * 70)
        print("[+] To apply these parameters to PAPER TRADING:")
        print("    python scripts/apply_params.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
