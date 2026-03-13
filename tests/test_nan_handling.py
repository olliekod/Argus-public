import pytest
import math
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.connectors.ibit_options_client import IBITOptionsClient
from src.analysis.trade_calculator import TradeCalculator

def test_trade_calculator_nan_risk():
    calc = TradeCalculator(account_size=3000)
    # Test calculate_num_contracts with NaN
    assert calc.calculate_num_contracts(float('nan'), 0.05) == 0
    # Test with Inf
    assert calc.calculate_num_contracts(float('inf'), 0.05) == 0
    # Test with 0
    assert calc.calculate_num_contracts(0.0, 0.05) == 0

@patch('src.connectors.ibit_options_client.yf.Ticker')
def test_ibit_options_client_nan_filter(mock_ticker):
    client = IBITOptionsClient(symbol="IBIT")
    
    # Mock price
    client.ticker.info = {'regularMarketPrice': 40.0}
    
    # Mock options chain with some NaN values
    mock_puts = pd.DataFrame({
        'strike': [28.0, 30.0, 31.0, 32.0],
        'bid': [0.1, 0.5, np.nan, 0.7],
        'ask': [0.2, 0.6, 0.8, np.nan],
        'impliedVolatility': [0.4, 0.4, 0.4, 0.4],
        'volume': [10, 10, 10, 10],
        'openInterest': [100, 100, 100, 100]
    })
    
    with patch.object(client, 'get_options_chain', return_value=(pd.DataFrame(), mock_puts)):
        spread = client.get_puts_for_spread("2026-12-19", target_delta=0.18, spread_width=2.0)
        
        # Should only consider the first strike because others have NaN
        assert spread is not None
        assert spread['short_strike'] == 30.0
        assert not math.isnan(spread['net_credit'])

def test_trade_calculator_generate_recommendation_nan_guard():
    calc = TradeCalculator(account_size=3000)
    
    # Mock options client to return NaN net_credit
    calc.options_client.get_market_status = MagicMock(return_value={
        'iv_rank': 60, 'iv': 0.5, 'price': 40.0, 'is_market_hours': True
    })
    calc.options_client.get_expirations_in_range = MagicMock(return_value=[("2026-12-19", 10)])
    
    calc.options_client.get_puts_for_spread = MagicMock(return_value={
        'short_strike': 35.0,
        'long_strike': 33.0,
        'spread_width': 2.0,
        'net_credit': float('nan'),
        'max_risk': 1.5,
        'short_iv': 0.5
    })
    
    rec = calc.generate_recommendation(force=True)
    assert rec is None

if __name__ == "__main__":
    pytest.main([__file__])
