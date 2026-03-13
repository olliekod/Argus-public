"""
Strategy Health Check
=====================

Runs quarterly health check with smart recommendations.
Run: python scripts/strategy_health.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.strategy_monitor import StrategyMonitor


def main():
    """Run strategy health check."""
    print()
    monitor = StrategyMonitor(symbol="BITO")
    print(monitor.generate_health_report())
    
    print("\n💡 Run this quarterly to validate strategy parameters.")
    print("   The monitor uses RISK-ADJUSTED metrics (Sharpe ratio),")
    print("   not just raw P&L, to make recommendations.")


if __name__ == "__main__":
    main()
