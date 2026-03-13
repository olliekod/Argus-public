import json
from datetime import datetime, timezone

from src.connectors.tastytrade_rest import (
    RetryConfig,
    TastytradeRestClient,
    parse_rfc3339_nano,
)


class DummyResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def request(self, method, url, params=None, json=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "json": json,
                "timeout": timeout,
            }
        )
        return self.responses.pop(0)

    def close(self):
        return None


def test_base_url_selection():
    client = TastytradeRestClient(
        "user",
        "pass",
        environment="sandbox",
    )
    assert "cert" in client.base_url


def test_login_sets_auth_header():
    session = DummySession(
        [DummyResponse(200, {"data": {"session-token": "token-123"}})]
    )
    client = TastytradeRestClient(
        "user",
        "pass",
        session=session,
    )
    token = client.login()
    assert token == "token-123"
    assert session.headers["Authorization"] == "token-123"


def test_retry_on_server_error(monkeypatch):
    responses = [
        DummyResponse(500, {"error": "bad"}),
        DummyResponse(200, {"data": {"items": [1]}}),
    ]
    session = DummySession(responses)
    client = TastytradeRestClient(
        "user",
        "pass",
        retries=RetryConfig(max_attempts=2, backoff_seconds=0, backoff_multiplier=1),
        session=session,
    )
    client._token = "token"
    session.headers["Authorization"] = "token"
    data = client.get_accounts()
    assert data == [1]


def test_list_nested_option_chains_url_and_params():
    responses = [
        DummyResponse(200, {"data": {"session-token": "token-123"}}),
        DummyResponse(200, {"data": {"expirations": []}}),
    ]
    session = DummySession(responses)
    client = TastytradeRestClient(
        "user",
        "pass",
        session=session,
    )
    client.login()
    client.list_nested_option_chains("IBIT", include_weeklies="true")
    assert session.calls[-1]["url"].endswith("/option-chains/IBIT/nested")
    assert session.calls[-1]["params"] == {"include_weeklies": "true"}


def test_parse_rfc3339_nano():
    parsed = parse_rfc3339_nano("2024-01-02T03:04:05.123456789Z")
    assert parsed == datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
