from src.connectors.deribit_client import DeribitClient


def test_deribit_coerce_result_list_from_dict():
    payload = {"result": [{"instrument_name": "BTC-TEST"}]}
    items = DeribitClient._coerce_result_list(payload, "book_summary")
    assert isinstance(items, list)
    assert items[0]["instrument_name"] == "BTC-TEST"


def test_deribit_coerce_result_list_from_list():
    payload = [{"instrument_name": "BTC-TEST"}]
    items = DeribitClient._coerce_result_list(payload, "book_summary")
    assert isinstance(items, list)
    assert items[0]["instrument_name"] == "BTC-TEST"


def test_deribit_extract_source_ts_handles_list_payload():
    source_ts, reason, label, raw = DeribitClient._extract_source_ts([{"timestamp": 123}])
    assert source_ts is None
    assert reason == "missing"
    assert label is None
    assert raw is None
