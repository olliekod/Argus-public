import pytest
from unittest.mock import MagicMock

from src.core.events import ExternalMetricEvent, TOPIC_EXTERNAL_METRICS
from src.core.lexicon_scorer import LexiconScorer
from src.core.news_sentiment_updater import NewsSentimentUpdater


def test_lexicon_scorer_basic():
    scorer = LexiconScorer()
    score, counts = scorer.score_text("Strong growth but weak outlook and risk")
    assert counts["pos_count"] >= 2
    assert counts["neg_count"] >= 2
    assert -1.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_news_sentiment_updater_with_mocked_fetcher():
    bus = MagicMock()
    updater = NewsSentimentUpdater(
        bus=bus,
        config={
            "news_sentiment": {
                "enabled": True,
                "feeds": ["https://example.com/rss"],
                "max_headlines": 50,
                "lexicon": "loughran_mcdonald",
            }
        },
    )

    async def fake_fetch_headlines(limit: int = 50):
        return [
            {
                "title": "Markets surge on strong earnings",
                "summary": "profit growth improves outlook",
                "published": None,
                "source": "rss",
            },
            {
                "title": "Stocks drop amid recession risk",
                "summary": "loss concerns and weak demand",
                "published": None,
                "source": "rss",
            },
        ]

    updater._fetcher.fetch_headlines = fake_fetch_headlines  # type: ignore[attr-defined]

    payload = await updater.update()

    assert payload is not None
    assert -1.0 <= payload["score"] <= 1.0
    assert payload["n_headlines"] == 2
    assert payload["label"] in {"bullish", "neutral", "bearish"}

    bus.publish.assert_called()
    topic, event = bus.publish.call_args[0]
    assert topic == TOPIC_EXTERNAL_METRICS
    assert isinstance(event, ExternalMetricEvent)
    assert event.key == "news_sentiment"
    assert event.value == payload

    await updater.close()
