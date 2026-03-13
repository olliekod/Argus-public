import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import ConfigurationError, load_config, load_secrets
from src.connectors.tastytrade_rest import TastytradeError


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


def main() -> None:
    try:
        config = load_config()
        secrets = load_secrets()
    except ConfigurationError as exc:
        print(f"Config error: {exc}")
        return

    try:
        from scripts.tastytrade_health_audit import get_tastytrade_rest_client
        client = get_tastytrade_rest_client(config, secrets)
    except Exception as e:
        print(f"Tastytrade credentials missing or auth failed: {e}")
        return

    try:
        accounts = client.get_accounts() or []
    except TastytradeError as exc:
        client.close()
        print(f"Tastytrade auth failed: {exc}")
        sys.exit(1)

    masked_accounts = []
    for account in accounts:
        account_id = (
            account.get("account-number")
            or account.get("account_number")
            or account.get("id")
            or ""
        )
        if account_id:
            masked_accounts.append(f"****{str(account_id)[-4:]}")
        else:
            masked_accounts.append("****")
    print(f"Accounts returned: {len(accounts)}")
    if masked_accounts:
        print(f"Account IDs: {', '.join(masked_accounts)}")

    for symbol in ("IBIT", "BITO"):
        chain = client.get_option_chain(symbol)
        print(f"Chain {symbol}: {bool(chain)}")

    client.close()


if __name__ == "__main__":
    main()
