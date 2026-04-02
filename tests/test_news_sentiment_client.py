# Created by Oliver Meihls

# Unit tests for NewsSentimentClient ticker filtering and scoring.

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.connectors.news_sentiment_client import NewsSentimentClient


@pytest.mark.asyncio
async def test_sentiment_client_ticker_filtering():
    # Verify that headlines are correctly filtered and scored for a specific ticker.
    config = {
        "news_sentiment": {
            "enabled": True,
            "feeds": ["https://example.com/rss"],
            "max_headlines": 50,
        }
    }
    client = NewsSentimentClient(config)

    # Mock the fetcher
    client._fetcher.fetch_headlines = AsyncMock(return_value=[
        {
            "title": "Bitcoin surge as BTC growth gain",
            "summary": "Positive momentum and strength.",
            "published": None,
            "source": "rss",
        },
        {
            "title": "Ethereum improved and resilient",
            "summary": "ETH outperform and benefit.",
            "published": None,
            "source": "rss",
        },
        {
            "title": "Market stable today",
            "summary": "Nothing much happening.",
            "published": None,
            "source": "rss",
        }
    ])

    # Test filtering for BTCUSDT
    result = await client.get_market_sentiment(ticker="BTCUSDT")
    assert result["ticker"] == "BTCUSDT"
    assert result["ticker_mentions"] == 1
    assert "Bitcoin surge" in result["bullets"][0]
    assert result["score"] > 0

    # Test filtering for ETHUSDT
    result = await client.get_market_sentiment(ticker="ETHUSDT")
    assert result["ticker"] == "ETHUSDT"
    assert result["ticker_mentions"] == 1
    assert "Ethereum improved" in result["bullets"][0]

    # Test aggregate MARKET sentiment
    result = await client.get_market_sentiment(ticker="")
    assert result["ticker"] == "MARKET"
    assert result["n_headlines"] == 3
    assert result["ticker_mentions"] == 3

    await client.close()


@pytest.mark.asyncio
async def test_sentiment_client_fallback_logic():
    # Verify that aggregate sentiment is returned if no ticker headlines are found.
    config = {
        "news_sentiment": {
            "enabled": True,
            "feeds": ["https://example.com/rss"],
        }
    }
    client = NewsSentimentClient(config)

    client._fetcher.fetch_headlines = AsyncMock(return_value=[
        {
            "title": "Strong market growth",
            "summary": "Economy is doing well.",
            "published": None,
            "source": "rss",
        }
    ])

    # Search for an unrelated ticker
    result = await client.get_market_sentiment(ticker="UNRELATED_TICKER")
    assert result["ticker"] == "UNRELATED_TICKER"
    assert result["ticker_mentions"] == 0
    assert result["score"] > 0  # Should fall back to market aggregate
    assert len(result["bullets"]) == 1

    await client.close()


@pytest.mark.asyncio
async def test_sentiment_client_disabled_stub():
    # Verify that a neutral stub is returned when disabled.
    client = NewsSentimentClient({"news_sentiment": {"enabled": False}})
    result = await client.get_market_sentiment(ticker="BTCUSDT")
    assert result["score"] == 0.0
    assert result["label"] == "neutral"
    assert result["ticker"] == "BTCUSDT"
    assert result["n_headlines"] == 0
