import json

from src.connectors.tastytrade_rest import TastytradeRestClient


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
                "headers": dict(self.headers),
            }
        )
        return self.responses.pop(0)

    def close(self):
        return None


def test_get_nested_option_chains_path_and_auth_header():
    responses = [
        DummyResponse(200, {"data": {"session-token": "token-abc"}}),
        DummyResponse(200, {"data": {"expirations": []}}),
    ]
    session = DummySession(responses)
    client = TastytradeRestClient("user", "pass", session=session)

    client.login()
    payload = client.get_nested_option_chains("IBIT")

    last_call = session.calls[-1]
    assert last_call["url"].endswith("/option-chains/IBIT/nested")
    assert last_call["headers"].get("Authorization") == "token-abc"
    assert payload == {"data": {"expirations": []}}


def test_get_nested_option_chains_error_excerpt():
    responses = [
        DummyResponse(200, {"data": {"session-token": "token-abc"}}),
        DummyResponse(404, {"error": "X" * 600}),
    ]
    session = DummySession(responses)
    client = TastytradeRestClient("user", "pass", session=session)
    client.login()

    try:
        client.get_nested_option_chains("IBIT")
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected error not raised")

    assert "HTTP 404" in message
    assert "X" * 400 in message
