import asyncio
import json
import logging
import sys
from pathlib import Path
import websockets

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Setup logging to both console and file
LOG_FILE = REPO_ROOT / "logs" / "dxlink_debug.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

async def verify_streaming():
    from src.core.config import load_secrets
    from src.connectors.tastytrade_rest import TastytradeRestClient
    from src.connectors.tastytrade_oauth import TastytradeOAuthClient

    secrets = load_secrets()
    oauth_cfg = secrets.get("tastytrade_oauth2", {})
    
    logger.info("Refreshing access token...")
    oauth_client = TastytradeOAuthClient(
        client_id=oauth_cfg["client_id"],
        client_secret=oauth_cfg["client_secret"],
        refresh_token=oauth_cfg["refresh_token"]
    )
    token_result = oauth_client.refresh_access_token()
    access_token = token_result.access_token

    import requests
    resp = requests.get(
        "https://api.tastytrade.com/api-quote-tokens",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    quote_data = resp.json().get("data", {})
    dxlink_url = quote_data.get("dxlink-url")
    dxlink_token = quote_data.get("token")

    # Pick an option symbol
    rest_client = TastytradeRestClient(
        username=secrets.get("tastytrade", {}).get("username"),
        password=secrets.get("tastytrade", {}).get("password")
    )
    rest_client._token = f"Bearer {access_token}"
    rest_client._session.headers["Authorization"] = f"Bearer {access_token}"
    chain_data = rest_client.get_nested_option_chains("IBIT")
    chains_envelope = chain_data.get("data", {})
    if isinstance(chains_envelope, dict) and "items" in chains_envelope:
        chains = chains_envelope["items"]
    elif isinstance(chains_envelope, dict) and "expirations" in chains_envelope:
        chains = [chains_envelope]
    else:
        chains = []
    
    option_symbol = None
    if chains:
        # Get expirations from the first chain
        expirations = chains[0].get("expirations", [])
        if expirations:
            # Get strikes from the first expiration
            strikes = expirations[0].get("strikes", [])
            for strike in strikes:
                if isinstance(strike, dict):
                    option_symbol = strike.get("call-streamer-symbol")
                    if option_symbol: break
    
    if not option_symbol:
        logger.error("No option symbol found.")
        return

    logger.info(f"Connecting to {dxlink_url}...")
    async with websockets.connect(dxlink_url) as ws:
        async def send(msg):
            logger.info(f"OUT: {json.dumps(msg)}")
            await ws.send(json.dumps(msg))

        # 1. SETUP
        await send({
            "type": "SETUP",
            "channel": 0,
            "keepaliveTimeout": 60,
            "acceptAggregationPeriod": 0.1,
            "acceptDataFormat": "json",
            "acceptEventFlavor": "instrument-id",
            "version": "0.1"
        })
        
        # 2. WAIT FOR SETUP_RESPONSE or AUTH_STATE (UNAUTHORIZED)
        waiting = True
        while waiting:
            resp = json.loads(await ws.recv())
            logger.info(f"IN: {resp}")
            st = resp.get("state") # Top-level state
            
            if resp.get("type") == "AUTH_STATE" and st == "UNAUTHORIZED":
                # 3. AUTH
                await send({
                    "type": "AUTH",
                    "channel": 0,
                    "token": dxlink_token
                })
            elif resp.get("type") == "AUTH_STATE" and st == "AUTHORIZED":
                # 4. CHANNEL_REQUEST
                await send({
                    "type": "CHANNEL_REQUEST",
                    "channel": 1,
                    "service": "FEED",
                    "parameters": {"contract": "AUTO"}
                })
            elif resp.get("type") == "CHANNEL_OPENED":
                channel_id = resp.get("channel")
                logger.info(f"Channel {channel_id} opened.")
                # 5. FEED_SETUP
                await send({
                    "type": "FEED_SETUP",
                    "channel": channel_id,
                    "acceptEventTypes": ["Quote", "Greeks", "Summary", "Profile"]
                })
                # 6. FEED_SUBSCRIPTION
                await send({
                    "type": "FEED_SUBSCRIPTION",
                    "channel": channel_id,
                    "add": [
                        {"type": "Quote", "symbol": "IBIT"},
                        {"type": "Quote", "symbol": option_symbol}
                    ]
                })
                waiting = False
            elif resp.get("type") == "ERROR":
                logger.error(f"Handshake error: {resp}")
                return

        logger.info("Waiting for data (20 seconds)...")
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < 20:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if msg.get("type") == "FEED_DATA":
                    logger.info(f"DATA: {msg}")
                elif msg.get("type") == "KEEPALIVE":
                    await send({"type": "KEEPALIVE", "channel": 0})
                else:
                    logger.info(f"IN: {msg}")
            except asyncio.TimeoutError:
                continue

if __name__ == "__main__":
    asyncio.run(verify_streaming())
