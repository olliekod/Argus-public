"""
Test IBIT Options System
========================

Tests the new Phase 6 components:
- IBIT Options Client (yfinance)
- Greeks Engine (Black-Scholes)
- Trade Calculator (recommendations)
- Economic Calendar (blackouts)
"""

import pytest

pytest.skip("Manual integration script (not for automated pytest runs).", allow_module_level=True)

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_options_client():
    """Test IBIT options client."""
    print("\n" + "="*60)
    print("OPTIONS CLIENT TEST")
    print("="*60)
    
    from src.connectors.ibit_options_client import IBITOptionsClient
    
    # Test both IBIT and BITO
    for symbol in ["IBIT", "BITO"]:
        print(f"\n--- {symbol} ---")
        client = IBITOptionsClient(symbol=symbol)
    
    # Market status
    status = client.get_market_status()
    print(f"\nMarket Status:")
    print(f"  Symbol: {status['symbol']}")
    print(f"  Price: ${status['price']:.2f}")
    print(f"  IV: {status['iv_pct']:.1f}%")
    print(f"  IV Rank: {status['iv_rank']:.1f}%")
    print(f"  Market Open: {status['is_market_hours']}")
    
    # Expirations
    exps = client.get_expirations_in_range(7, 21)
    print(f"\nExpirations (7-21 DTE): {len(exps)} found")
    for exp, dte in exps[:3]:
        print(f"  {exp} ({dte} days)")
    
    # Put spread
    if exps:
        spread = client.get_puts_for_spread(exps[0][0])
        if spread:
            print(f"\nSuggested Put Spread ({exps[0][0]}):")
            print(f"  Short: ${spread['short_strike']:.0f} Put")
            print(f"  Long:  ${spread['long_strike']:.0f} Put")
            print(f"  Credit: ${spread['net_credit']:.2f}")
            print(f"  Max Risk: ${spread['max_risk']:.2f}")
        else:
            print("\n  Could not find suitable spread")
    
    return True


def test_greeks_engine():
    """Test Greeks engine."""
    print("\n" + "="*60)
    print("GREEKS ENGINE TEST")
    print("="*60)
    
    from src.analysis.greeks_engine import GreeksEngine
    
    engine = GreeksEngine()
    
    # Example: IBIT at $42, selling $38 put, 10 DTE, 50% IV
    S = 42.0
    K = 38.0
    T = 10 / 365
    sigma = 0.50
    
    print(f"\nExample: IBIT ${S}, ${K} Put, 10 DTE, {sigma*100:.0f}% IV")
    
    greeks = engine.calculate_all_greeks(S, K, T, sigma, 'put')
    print(f"\nSingle Put Greeks:")
    print(f"  Delta: {greeks.delta:.4f}")
    print(f"  Gamma: {greeks.gamma:.4f}")
    print(f"  Theta: ${greeks.theta:.4f}/day")
    print(f"  Vega:  ${greeks.vega:.4f}/1% IV")
    
    # Spread Greeks
    spread_greeks = engine.calculate_spread_greeks(S, 38.0, 36.0, T, sigma)
    print(f"\nPut Spread Greeks (Sell $38, Buy $36):")
    print(f"  Net Delta: {spread_greeks.net_delta:.4f}")
    print(f"  Net Theta: ${spread_greeks.net_theta:.4f}/day (+ is good)")
    print(f"  Net Vega:  ${spread_greeks.net_vega:.4f}/1% IV (- is good)")
    
    # PoP
    credit = 0.50
    pop = engine.probability_of_profit(S, 38.0, credit, T, sigma)
    print(f"\nProbability of Profit: {pop:.1f}%")
    
    # Expected move
    low, high = engine.expected_move(S, sigma, T)
    print(f"Expected Move (1 SD): ${low:.2f} - ${high:.2f}")
    
    return True


def test_trade_calculator():
    """Test trade calculator."""
    print("\n" + "="*60)
    print("TRADE CALCULATOR TEST")
    print("="*60)
    
    from src.analysis.trade_calculator import TradeCalculator
    
    calc = TradeCalculator(account_size=3000)
    
    # Force recommendation (ignore IV threshold for testing)
    rec = calc.generate_recommendation(
        btc_change_24h=-0.05,
        ibit_change_24h=-0.04,
        force=True,
    )
    
    if rec:
        print(f"\nTrade Recommendation Generated:")
        print(f"  Expiration: {rec.expiration} ({rec.dte} DTE)")
        print(f"  Short Strike: ${rec.short_strike:.0f}")
        print(f"  Long Strike: ${rec.long_strike:.0f}")
        print(f"  Net Credit: ${rec.net_credit:.2f}")
        print(f"  Max Risk: ${rec.max_risk:.2f}")
        print(f"  PoP: {rec.probability_of_profit:.0f}%")
        print(f"  Position Size: {rec.position_size_pct*100:.0f}%")
        print(f"  Contracts: {rec.num_contracts}")
        print(f"  Capital at Risk: ${rec.capital_at_risk:.0f}")
        
        if rec.warnings:
            print(f"\n  Warnings:")
            for w in rec.warnings:
                print(f"    {w}")
        
        print("\n--- TELEGRAM FORMAT ---")
        print(calc.format_telegram_alert(rec))
    else:
        print("\n  No recommendation generated")
    
    return True


def test_economic_calendar():
    """Test economic calendar."""
    print("\n" + "="*60)
    print("ECONOMIC CALENDAR TEST")
    print("="*60)
    
    from src.core.economic_calendar import EconomicCalendar
    
    cal = EconomicCalendar()
    
    # Get all events
    events = cal.get_all_events()
    print(f"\n2026 Events: {len(events)} total")
    
    # Next 5 events
    print("\nUpcoming Events:")
    for event in events[:5]:
        print(f"  {event.date.strftime('%Y-%m-%d %H:%M')} - {event.name} ({event.risk.value})")
    
    # Next event
    next_event = cal.get_next_event()
    if next_event:
        print(f"\nNext Event: {next_event.name} on {next_event.date.strftime('%Y-%m-%d')}")
    
    # Blackout check
    is_blackout, event = cal.is_blackout_period()
    print(f"\nCurrently in blackout: {is_blackout}")
    if event:
        print(f"  Caused by: {event.name}")
    
    # Warning
    warning = cal.get_blackout_warning()
    if warning:
        print(f"\nWarning: {warning}")
    else:
        print("\nNo blackout warning - safe to trade")
    
    # Weekly summary
    print(f"\n{cal.format_weekly_summary()}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ARGUS IBIT OPTIONS SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Options Client", test_options_client),
        ("Greeks Engine", test_greeks_engine),
        ("Trade Calculator", test_trade_calculator),
        ("Economic Calendar", test_economic_calendar),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, result in results:
        status = "✅" if result == "PASS" else "❌"
        print(f"  {status} {name}: {result}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
