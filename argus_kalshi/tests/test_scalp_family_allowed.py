# Created by Oliver Meihls

from argus_kalshi.mispricing_scalper import _scalp_family_allowed


def test_scalp_family_allowed_btc_and_eth_15m_60m_only():
    assert _scalp_family_allowed("BTC", 15, False) is True
    assert _scalp_family_allowed("BTC", 60, False) is True
    assert _scalp_family_allowed("ETH", 15, False) is True
    assert _scalp_family_allowed("ETH", 60, False) is True

    assert _scalp_family_allowed("BTC", 5, False) is False
    assert _scalp_family_allowed("ETH", 5, False) is False
    assert _scalp_family_allowed("BTC", 15, True) is False
    assert _scalp_family_allowed("SOL", 15, False) is False
