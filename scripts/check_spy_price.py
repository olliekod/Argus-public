import asyncio
from src.connectors.alpaca_client import AlpacaClient
from src.utils.config import load_config, load_secrets

async def check_spy():
    config = load_config()
    secrets = load_secrets()
    client = AlpacaClient(config, secrets)
    
    # Get latest quote
    quote = client.get_latest_quote("SPY")
    print(f"Alpaca SPY Quote: {quote}")

if __name__ == "__main__":
    asyncio.run(check_spy())
