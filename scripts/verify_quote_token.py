import sys
import os
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_secrets
from src.connectors.tastytrade_rest import TastytradeRestClient

def main():
    secrets = load_secrets()
    # Check for oauth refresh token
    oauth_cfg = secrets.get("tastytrade_oauth2", {})
    if not oauth_cfg.get("refresh_token"):
        print("No refresh_token found in secrets.yaml. Run bootstrap first.")
        return 1

    print("Refreshing OAuth access token...")
    from src.connectors.tastytrade_oauth import TastytradeOAuthClient
    oauth_client = TastytradeOAuthClient(
        client_id=oauth_cfg["client_id"],
        client_secret=oauth_cfg["client_secret"],
        refresh_token=oauth_cfg["refresh_token"]
    )
    token_result = oauth_client.refresh_access_token()
    access_token = token_result.access_token

    print("Fetching API Quote Token...")
    try:
        url = "https://api.tastytrade.com/api-quote-tokens"
        headers = {"Authorization": f"Bearer {access_token}"}
        import requests
        resp = requests.get(url, headers=headers)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print("Successfully acquired Quote Token!")
            print(f"Streamer URL: {data.get('data', {}).get('dxlink-url')}")
            return 0
        else:
            print(f"Error: {resp.text}")
            return 1
    except Exception as e:
        print(f"Exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
