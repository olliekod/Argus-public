"""News sentiment external metric updater.

Builds a sentiment score from fetched headlines and a finance lexicon, then
publishes ``ExternalMetricEvent(key="news_sentiment", value=...)``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .bus import EventBus
from .events import ExternalMetricEvent, TOPIC_EXTERNAL_METRICS
from .headline_fetcher import HeadlineFetcher
from .lexicon_scorer import LexiconScorer

logger = logging.getLogger("argus.news_sentiment_updater")

_DEFAULT_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
]


def format_news_sentiment_telegram(payload: Optional[Dict[str, Any]]) -> str:
    """Format a concise news sentiment summary line for Telegram."""
    if not payload or payload.get("label") == "stub":
        return "⚪ News: (stub/unavailable)"

    score = float(payload.get("score", 0.0) or 0.0)
    label = str(payload.get("label") or "?")
    n_headlines = int(payload.get("n_headlines", 0) or 0)
    emoji = "🟢" if label == "bullish" else "🔴" if label == "bearish" else "⚪"
    return f"{emoji} News: {label} ({score:+.2f}) | {n_headlines} headlines"


class NewsSentimentUpdater:
    """Collect and publish a ``news_sentiment`` external metric."""

    def __init__(self, bus: EventBus, config: Dict[str, Any]) -> None:
        self._bus = bus

        ns_cfg = config.get("news_sentiment") or {}
        self._enabled: bool = bool(ns_cfg.get("enabled", False))
        self._max_headlines = max(1, int(ns_cfg.get("max_headlines", 50) or 50))

        feeds_cfg = ns_cfg.get("feeds", _DEFAULT_FEEDS)
        feeds: List[str] = [str(feed).strip() for feed in (feeds_cfg or []) if str(feed).strip()]
        # Use default feeds only when key is missing; explicit feeds=[] means no fetches
        if not feeds and "feeds" not in ns_cfg:
            feeds = list(_DEFAULT_FEEDS)

        self._fetcher = HeadlineFetcher(
            feeds=feeds,
            newsapi_key=ns_cfg.get("newsapi_key"),
            max_headlines=self._max_headlines,
        )
        self._scorer = LexiconScorer(
            lexicon=str(ns_cfg.get("lexicon") or "loughran_mcdonald"),
            lexicon_path=ns_cfg.get("lexicon_path"),
        )
        self._last_payload: Optional[Dict[str, Any]] = None

        if self._enabled:
            logger.info("NewsSentimentUpdater enabled (source=headlines_lexicon)")
        else:
            logger.info("NewsSentimentUpdater disabled (news_sentiment.enabled=false)")

    async def close(self) -> None:
        await self._fetcher.close()

    def get_last_payload(self) -> Optional[Dict[str, Any]]:
        """Return the last computed/published payload, if available."""
        return self._last_payload

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score >= 0.10:
            return "bullish"
        if score <= -0.10:
            return "bearish"
        return "neutral"

    @staticmethod
    def _stub_payload() -> Dict[str, Any]:
        return {"score": 0.0, "label": "stub", "n_headlines": 0}

    def _publish(self, payload: Dict[str, Any], now_ms: int) -> None:
        self._bus.publish(
            TOPIC_EXTERNAL_METRICS,
            ExternalMetricEvent(key="news_sentiment", value=payload, timestamp_ms=now_ms),
        )

    async def update(self) -> Optional[Dict[str, Any]]:
        """Fetch, score, aggregate, and publish news sentiment payload."""
        now_ms = int(time.time() * 1000)

        if not self._enabled:
            payload = self._stub_payload()
            self._publish(payload, now_ms)
            self._last_payload = payload
            return payload

        try:
            headlines = await self._fetcher.fetch_headlines(limit=self._max_headlines)
        except Exception as exc:
            logger.warning("NewsSentiment headline fetch failed: %s", exc)
            payload = self._stub_payload()
            self._publish(payload, now_ms)
            self._last_payload = payload
            return payload

        if not headlines:
            payload = self._stub_payload()
            self._publish(payload, now_ms)
            self._last_payload = payload
            logger.info("NewsSentiment updated: score=0.0000 label=stub n_headlines=0 sources=none")
            return payload

        scores: List[float] = []
        sources: set[str] = set()
        for item in headlines:
            text = f"{item.get('title', '')} {item.get('summary', '')}".strip()
            score, _counts = self._scorer.score_text(text)
            scores.append(max(-1.0, min(1.0, float(score))))
            source = str(item.get("source") or "unknown").strip()
            if source:
                sources.add(source)

        if not scores:
            payload = self._stub_payload()
            self._publish(payload, now_ms)
            self._last_payload = payload
            return payload

        avg_score = sum(scores) / len(scores)
        payload = {
            "score": round(max(-1.0, min(1.0, avg_score)), 6),
            "label": self._label_from_score(avg_score),
            "n_headlines": len(scores),
        }
        self._publish(payload, now_ms)
        self._last_payload = payload
        logger.info(
            "NewsSentiment updated: score=%.4f label=%s n_headlines=%d sources=%s",
            payload["score"],
            payload["label"],
            payload["n_headlines"],
            ",".join(sorted(sources)) if sources else "none",
        )
        return payload
