"""
Apply Optimized Parameters
==========================

Applies the optimized parameters from the last optimization run.
By default, applies to PAPER TRADING first.

Run: python scripts/apply_params.py [--paper|--live]
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.production_optimizer import ProductionOptimizer


def load_last_report() -> dict:
    """Load the last optimization report."""
    report_path = Path("data/optimization_report.txt")
    if not report_path.exists():
        return None
    
    # Parse the report to extract params
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Extract parameters from report text
    params = {}
    lines = content.split('\n')
    
    in_params = False
    for line in lines:
        if 'OPTIMAL PARAMETERS' in line:
            in_params = True
            continue
        if in_params and '---' in line:
            continue
        if in_params and line.strip() == '':
            in_params = False
            continue
        
        if in_params and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                key = parts[0].strip().lower().replace(' ', '_')
                value = parts[1].strip().replace('%', '')
                
                try:
                    if 'threshold' in key:
                        params['iv_threshold'] = float(value) / 100
                    elif 'trigger' in key:
                        params['price_drop_trigger'] = float(value) / 100
                    elif 'delta' in key:
                        params['target_delta'] = float(value)
                    elif 'profit' in key:
                        params['profit_target'] = float(value) / 100
                    elif 'width' in key:
                        params['spread_width_pct'] = float(value) / 100
                except ValueError:
                    continue
    
    return params if params else None


def apply_to_paper(params: dict) -> None:
    """Apply parameters to thresholds.yaml (the actual config file)."""
    import yaml
    
    config_path = Path("config/thresholds.yaml")
    
    if not config_path.exists():
        print("❌ thresholds.yaml not found")
        return
    
    # Read current config
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update IBIT section values
    # IV threshold (stored as %, e.g., 25 means 25%)
    iv_pct = int(params.get('iv_threshold', 0.25) * 100)
    content = content.replace(
        f"btc_iv_threshold: {_find_current_value(content, 'btc_iv_threshold')}",
        f"btc_iv_threshold: {iv_pct}"
    )
    
    # Drop threshold (stored as %, e.g., -0.5 means -0.5%)
    drop_pct = params.get('price_drop_trigger', -0.005) * 100
    current_drop = _find_current_value(content, 'ibit_drop_threshold')
    if current_drop:
        content = content.replace(
            f"ibit_drop_threshold: {current_drop}",
            f"ibit_drop_threshold: {drop_pct}"
        )
    
    # Profit target (stored as decimal, e.g., 0.70)
    profit = params.get('profit_target', 0.70)
    current_profit = _find_current_value(content, 'profit_target')
    if current_profit:
        content = content.replace(
            f"profit_target: {current_profit}",
            f"profit_target: {profit}"
        )
    
    # Write back
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {config_path}")


def _find_current_value(content: str, key: str) -> str:
    """Find current value for a key in YAML content."""
    import re
    match = re.search(rf'{key}:\s*([-\d.]+)', content)
    if match:
        return match.group(1)
    return None


def apply_to_live(params: dict) -> None:
    """Apply parameters to live trading configuration."""
    config_path = Path("config/strategy_params.json")
    
    if not config_path.exists():
        print("❌ No config file found. Run optimization first.")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if paper tested first
    if not config.get('paper'):
        print("❌ No paper trading params found.")
        print("   Please run paper trading first before going live.")
        return
    
    # Update live params
    config['live'] = {
        **params,
        'time_exit_dte': 5,
        'entry_dte': 14,
        'updated_at': datetime.now().isoformat(),
        'source': 'promotion_from_paper',
    }
    
    config['history'].append({
        'params': params,
        'target': 'live',
        'timestamp': datetime.now().isoformat(),
    })
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Saved LIVE params to {config_path}")


def main():
    """Apply optimized parameters."""
    # Parse args
    mode = 'paper'
    if '--live' in sys.argv:
        mode = 'live'
    
    print()
    print("=" * 60)
    print("📋 APPLY OPTIMIZED PARAMETERS")
    print("=" * 60)
    
    # Load last report
    params = load_last_report()
    
    if not params:
        print()
        print("❌ No optimization report found.")
        print("   Run: python scripts/optimize.py")
        print()
        return
    
    print()
    print("Parameters to apply:")
    print("-" * 40)
    print(f"  IV Threshold:  {params.get('iv_threshold', 0.4)*100:.0f}%")
    print(f"  Drop Trigger:  {params.get('price_drop_trigger', -0.02)*100:.1f}%")
    print(f"  Target Delta:  {params.get('target_delta', 0.18)}")
    print(f"  Profit Target: {params.get('profit_target', 0.5)*100:.0f}%")
    print(f"  Spread Width:  {params.get('spread_width_pct', 0.05)*100:.0f}%")
    print()
    
    if mode == 'paper':
        print("Applying to: PAPER TRADING")
        print()
        confirm = input("Proceed? (y/n): ").lower().strip()
        
        if confirm == 'y':
            apply_to_paper(params)
            print()
            print("=" * 60)
            print("✅ Parameters applied to PAPER TRADING")
            print()
            print("The paper trading system will now use these parameters.")
            print("Monitor performance, then promote to live when ready:")
            print("   python scripts/apply_params.py --live")
            print("=" * 60)
        else:
            print("Cancelled.")
    
    elif mode == 'live':
        print("⚠️  Applying to: LIVE TRADING")
        print()
        print("Are you sure? This will affect real trades.")
        confirm = input("Type 'APPLY' to confirm: ").strip()
        
        if confirm == 'APPLY':
            apply_to_live(params)
            print()
            print("=" * 60)
            print("✅ Parameters applied to LIVE TRADING")
            print("=" * 60)
        else:
            print("Cancelled.")


if __name__ == "__main__":
    main()
