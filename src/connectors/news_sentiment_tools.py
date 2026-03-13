"""
Delphi-registered news & sentiment tools for Pantheon agents.
=============================================================

Exposes ``get_market_sentiment`` as a Delphi tool so that research
agents (Prometheus, Ares) can query live market sentiment during
the debate protocol.

Uses the ``@tool`` decorator from :mod:`src.agent.delphi` for
automatic discovery via ``DelphiToolRegistry.discover_tools()``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from src.agent.delphi import RiskLevel, tool

logger = logging.getLogger("argus.connectors.news_sentiment_tools")

# Module-level client singleton (initialised lazily by the tool function).
_client: Optional[Any] = None
_config: Optional[Dict[str, Any]] = None


def configure_sentiment_client(config: Dict[str, Any]) -> None:
    """Set the config used to lazily initialise the sentiment client.

    Call this once during application startup before any tool invocation.
    """
    global _config
    _config = config


def _get_client() -> Any:
    """Lazy-init the NewsSentimentClient singleton."""
    global _client
    if _client is None:
        from src.connectors.news_sentiment_client import NewsSentimentClient
        cfg = _config or {}
        _client = NewsSentimentClient(cfg)
    return _client


@tool(
    name="get_market_sentiment",
    description=(
        "Fetch current market sentiment for a ticker symbol. "
        "Returns a sentiment score (-1.0 to 1.0), a label "
        "(bullish/bearish/neutral), and 3-5 key headline bullet points. "
        "Use this during research to ground strategy proposals in "
        "current market conditions."
    ),
    risk_level=RiskLevel.READ_ONLY,
    parameters_schema={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": (
                    "Ticker symbol to check sentiment for (e.g., 'BTCUSDT', "
                    "'IBIT', 'SPY'). Leave empty for aggregate market sentiment."
                ),
            },
        },
        "required": ["ticker"],
    },
    estimated_cost=0.0,
)
async def get_market_sentiment(ticker: str = "") -> Dict[str, Any]:
    """Delphi tool: fetch and score market sentiment for a given ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol or empty string for aggregate market sentiment.

    Returns
    -------
    dict
        ``{"score": float, "label": str, "n_headlines": int,
        "ticker": str, "ticker_mentions": int, "bullets": list,
        "timestamp_utc": str}``
    """
    client = _get_client()
    return await client.get_market_sentiment(ticker=ticker)
