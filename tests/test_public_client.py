import pytest

from src.connectors.public_client import PublicAPIClient, PublicAPIConfig, PublicAPIError


@pytest.mark.asyncio
async def test_public_client_requires_api_secret():
    with pytest.raises(PublicAPIError):
        PublicAPIClient(PublicAPIConfig(api_secret=""))


@pytest.mark.asyncio
async def test_get_option_greeks_limit_enforced():
    client = PublicAPIClient(PublicAPIConfig(api_secret="token", account_id="acct"))
    symbols = [f"SPY250321P{i:08d}" for i in range(251)]
    with pytest.raises(PublicAPIError):
        await client.get_option_greeks(symbols)


@pytest.mark.asyncio
async def test_get_option_greeks_uses_account_and_parses_rows(monkeypatch):
    client = PublicAPIClient(PublicAPIConfig(api_secret="token", account_id="acct-1"))

    calls = []

    async def fake_request(method, path, *, params=None, retry_401=True):
        calls.append((method, path, params))
        return {
            "greeks": [
                {
                    "symbol": "SPY250321P00590000",
                    "greeks": {
                        "impliedVolatility": 0.27,
                        "delta": -0.5,
                    },
                }
            ]
        }

    monkeypatch.setattr(client, "_request", fake_request)
    rows = await client.get_option_greeks(["SPY250321P00590000"])
    assert rows
    assert rows[0]["symbol"] == "SPY250321P00590000"
    assert calls[0][1] == "/userapigateway/option-details/acct-1/greeks"


@pytest.mark.asyncio
async def test_get_account_id_uses_config():
    client = PublicAPIClient(PublicAPIConfig(api_secret="token", account_id="acct-from-config"))
    assert await client.get_account_id() == "acct-from-config"
    assert await client.get_account_id() == "acct-from-config"


@pytest.mark.asyncio
async def test_get_account_id_raises_when_missing():
    client = PublicAPIClient(PublicAPIConfig(api_secret="token", account_id=""))
    with pytest.raises(PublicAPIError, match="account_id is required"):
        await client.get_account_id()


@pytest.mark.asyncio
async def test_public_rate_limiter_waits_when_budget_exhausted(monkeypatch):
    client = PublicAPIClient(PublicAPIConfig(api_secret="token", account_id="acct", rate_limit_rps=2))

    ticks = [0.0, 0.1, 0.2, 0.2, 1.2]
    sleeps = []

    def fake_monotonic():
        if ticks:
            return ticks.pop(0)
        return 1.2

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("src.connectors.public_client.time.monotonic", fake_monotonic)
    monkeypatch.setattr("src.connectors.public_client.asyncio.sleep", fake_sleep)

    await client._acquire_rate_limit()
    await client._acquire_rate_limit()
    await client._acquire_rate_limit()

    assert sleeps
    assert sleeps[0] == pytest.approx(0.8, abs=1e-6)
