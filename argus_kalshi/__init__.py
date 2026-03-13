"""
Argus Kalshi — BTC strike contract trading module.

Connects to the Kalshi exchange via REST + WebSocket, maintains a live orderbook,
computes settlement probabilities against the CF Benchmarks BRTI 60-second
simple average, and executes trades when expected value exceeds threshold.
"""

__version__ = "0.1.0"
