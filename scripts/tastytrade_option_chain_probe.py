"""Probe Tastytrade nested option chains.

Usage:
  python scripts/tastytrade_option_chain_probe.py --underlying IBIT
  python scripts/tastytrade_option_chain_probe.py --symbol SPY
"""

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import ConfigurationError, load_config, load_secrets
from src.core.options_normalize import normalize_tastytrade_nested_chain
from src.connectors.tastytrade_rest import TastytradeError


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


def _sample_symbols(contracts: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    def _valid(symbol: str | None) -> bool:
        return bool(symbol) and str(symbol).lower() != "n/a"

    call_symbol = next(
        (
            contract.get("option_symbol")
            for contract in contracts
            if contract.get("right") == "C" and _valid(contract.get("option_symbol"))
        ),
        None,
    )
    put_symbol = next(
        (
            contract.get("option_symbol")
            for contract in contracts
            if contract.get("right") == "P" and _valid(contract.get("option_symbol"))
        ),
        None,
    )
    return call_symbol, put_symbol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--underlying", default="IBIT", help="Underlying symbol (e.g. IBIT)")
    parser.add_argument("--symbol", dest="underlying", help="Alias for --underlying")
    args = parser.parse_args()

    try:
        config = load_config()
        secrets = load_secrets()
    except ConfigurationError as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        from scripts.tastytrade_health_audit import get_tastytrade_rest_client
        client = get_tastytrade_rest_client(config, secrets)
    except Exception as e:
        print(f"Tastytrade credentials missing or auth failed: {e}")
        return 1

    try:
        chain = client.get_nested_option_chains(args.underlying)
    except TastytradeError as exc:
        client.close()
        print(f"Tastytrade request failed: {exc}")
        return 1

    normalized = normalize_tastytrade_nested_chain(chain)
    expirations = sorted({item["expiry"] for item in normalized if item.get("expiry")})
    strikes = {item["strike"] for item in normalized if item.get("strike") is not None}

    print(f"Underlying: {args.underlying}")
    print(f"Expirations count: {len(expirations)}")
    print(f"Strikes count: {len(strikes)}")

    call_symbol, put_symbol = _sample_symbols(normalized)
    print(f"Sample call symbol: {call_symbol or 'n/a'}")
    print(f"Sample put symbol: {put_symbol or 'n/a'}")

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
