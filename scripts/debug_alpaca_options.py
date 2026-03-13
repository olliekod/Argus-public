"""Debug Alpaca options: expirations + snapshot build. Run from repo root: python scripts/debug_alpaca_options.py"""

import asyncio
import logging
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.connectors.alpaca_options import AlpacaOptionsConnector, AlpacaOptionsConfig

# Load secrets (relative to repo root)
with open(REPO_ROOT / "config" / "secrets.yaml") as f:
    secrets = yaml.safe_load(f)

api_key = secrets.get('alpaca', {}).get('api_key')
api_secret = secrets.get('alpaca', {}).get('api_secret')

logging.basicConfig(level=logging.INFO)

async def test_alpaca():
    if not api_key or not api_secret:
        print("Error: Alpaca API keys missing in secrets.yaml")
        return

    connector = AlpacaOptionsConnector(AlpacaOptionsConfig(
        api_key=api_key,
        api_secret=api_secret,
        cache_ttl_seconds=0
    ))
    
    print(f"Testing expirations for SPY...")
    # Add raw request debug
    session = await connector._get_session()
    url = f"{connector._config.base_url}/v1beta1/options/snapshots/SPY"
    async with session.get(url, params={"feed": "indicative", "limit": 10}) as resp:
        print(f"Raw Status: {resp.status}")
        raw_json = await resp.json()
        print(f"Raw Response Keys: {list(raw_json.keys())}")
        if "snapshots" in raw_json:
            print(f"Snapshots count: {len(raw_json['snapshots'])}")
            if len(raw_json['snapshots']) > 0:
                print(f"First snapshot key: {list(raw_json['snapshots'].keys())[0]}")
        else:
            print(f"Full Response: {raw_json}")

    exps = await connector.get_expirations("SPY")
    print(f"Found {len(exps)} expirations for SPY.")
    if exps:
        print(f"First 5: {exps[:5]}")
        
        target_exp = exps[0]
        print(f"Building snapshot for {target_exp}...")
        snapshot = await connector.build_chain_snapshot("SPY", target_exp)
        if snapshot:
            print(f"SUCCESS: Snapshot built with {snapshot.n_strikes} strikes.")
            print(f"Underlying price: {snapshot.underlying_price}")
        else:
            print("FAILED: Snapshot was None.")
    else:
        print("FAILED: No expirations found.")

    await connector.close()

if __name__ == "__main__":
    asyncio.run(test_alpaca())
