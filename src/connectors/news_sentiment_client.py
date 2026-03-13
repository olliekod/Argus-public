"""
News & Sentiment Client for Pantheon Research Agents
=====================================================

Provides a unified interface for fetching news headlines and computing
sentiment scores.  Designed to be consumed by Delphi-registered tools
so that Pantheon agents (especially Ares and Prometheus) can ground
their analysis in current market sentiment.

Integrates with the existing :class:`~src.core.headline_fetcher.HeadlineFetcher`
and :class:`~src.core.lexicon_scorer.LexiconScorer` infrastructure.

The ``get_market_sentiment`` entry point returns a structured dict with:
- ``score``: Aggregate sentiment (-1.0 to 1.0)
- ``label``: Human-readable label (bullish/bearish/neutral)
- ``bullets``: 3-5 key headline summaries
- ``n_headlines``: Number of headlines analyzed
- ``ticker_mentions``: Count of ticker-specific mentions
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.headline_fetcher import HeadlineFetcher
from src.core.lexicon_scorer import LexiconScorer

logger = logging.getLogger("argus.connectors.news_sentiment_client")

# Common ticker aliases for matching
_TICKER_ALIASES: Dict[str, List[str]] = {
    "BTCUSDT": ["bitcoin", "btc", "crypto"],
    "ETHUSDT": ["ethereum", "eth", "ether"],
    "IBIT": ["ibit", "bitcoin etf", "blackrock bitcoin"],
    "SPY": ["s&p 500", "s&p500", "sp500", "spy"],
    "QQQ": ["nasdaq", "qqq", "tech stocks"],
    "DIA": ["dow jones", "dow", "djia", "dia"],
    "GLD": ["gold", "gld"],
    "TLT": ["treasury", "bonds", "tlt"],
    "XLE": ["energy", "oil", "xle"],
    "XLF": ["financials", "banks", "xlf"],
    "XLK": ["technology", "tech", "xlk"],
    "SMH": ["semiconductors", "chips", "smh"],
    "IWM": ["russell", "small cap", "iwm"],
}


def _label_from_score(score: float) -> str:
    """Convert numeric score to sentiment label."""
    if score >= 0.10:
        return "bullish"
    if score <= -0.10:
        return "bearish"
    return "neutral"


def _matches_ticker(text: str, ticker: str) -> bool:
    """Check if text mentions a ticker or its aliases."""
    lower = text.lower()

    # Direct ticker match (case-insensitive, word boundary)
    if re.search(rf"\b{re.escape(ticker.lower())}\b", lower):
        return True

    # Alias matching
    aliases = _TICKER_ALIASES.get(ticker.upper(), [])
    for alias in aliases:
        if alias in lower:
            return True

    return False


def _pick_top_bullets(
    headlines: List[Dict[str, Any]],
    scores: List[float],
    limit: int = 5,
) -> List[str]:
    """Select the most sentiment-significant headlines as bullet points.

    Picks the headlines with the highest absolute sentiment score.
    """
    if not headlines or not scores:
        return []

    paired = list(zip(headlines, scores))
    paired.sort(key=lambda x: abs(x[1]), reverse=True)

    bullets: List[str] = []
    for headline, score in paired[:limit]:
        title = str(headline.get("title", "")).strip()
        if not title:
            continue
        direction = "+" if score > 0 else "-" if score < 0 else "~"
        bullets.append(f"[{direction}] {title}")

    return bullets


class NewsSentimentClient:
    """Fetch news and compute sentiment for Pantheon research agents.

    Parameters
    ----------
    config : dict
        Full Argus config dict (reads ``news_sentiment`` section).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        ns_cfg = config.get("news_sentiment") or {}
        self._enabled: bool = bool(ns_cfg.get("enabled", False))

        feeds = [
            str(f).strip()
            for f in (ns_cfg.get("feeds") or [])
            if str(f).strip()
        ]
        self._max_headlines = max(1, int(ns_cfg.get("max_headlines", 50) or 50))

        self._fetcher = HeadlineFetcher(
            feeds=feeds,
            newsapi_key=ns_cfg.get("newsapi_key"),
            max_headlines=self._max_headlines,
        )
        self._scorer = LexiconScorer(
            lexicon=str(ns_cfg.get("lexicon") or "loughran_mcdonald"),
            lexicon_path=ns_cfg.get("lexicon_path"),
        )

        # Cache
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_ts: float = 0.0
        self._cache_ttl: float = float(ns_cfg.get("interval_seconds", 3600))

        if self._enabled:
            logger.info("NewsSentimentClient initialized (enabled)")
        else:
            logger.info("NewsSentimentClient initialized (disabled)")

    async def close(self) -> None:
        """Clean up HTTP sessions."""
        await self._fetcher.close()

    async def get_market_sentiment(self, ticker: str = "") -> Dict[str, Any]:
        """Compute market sentiment, optionally filtered to a specific ticker.

        Parameters
        ----------
        ticker : str
            If non-empty, filter headlines to those mentioning this ticker
            and compute ticker-specific sentiment.  If empty, return
            aggregate market sentiment from all headlines.

        Returns
        -------
        dict
            {
                "score": float,        # -1.0 to 1.0
                "label": str,          # "bullish" / "bearish" / "neutral"
                "n_headlines": int,
                "ticker": str,
                "ticker_mentions": int,
                "bullets": List[str],  # 3-5 key headlines
                "timestamp_utc": str,
            }
        """
        if not self._enabled:
            return self._stub(ticker)

        try:
            headlines = await self._fetcher.fetch_headlines(
                limit=self._max_headlines,
            )
        except Exception as exc:
            logger.warning("Headline fetch failed: %s", exc)
            return self._stub(ticker)

        if not headlines:
            return self._stub(ticker)

        # Score all headlines
        all_scores: List[float] = []
        all_headlines: List[Dict[str, Any]] = []
        for item in headlines:
            text = f"{item.get('title', '')} {item.get('summary', '')}".strip()
            score, _counts = self._scorer.score_text(text)
            clamped = max(-1.0, min(1.0, float(score)))
            all_scores.append(clamped)
            all_headlines.append(item)

        # Optionally filter to ticker
        if ticker:
            filtered_headlines: List[Dict[str, Any]] = []
            filtered_scores: List[float] = []
            for hl, sc in zip(all_headlines, all_scores):
                text = f"{hl.get('title', '')} {hl.get('summary', '')}".strip()
                if _matches_ticker(text, ticker):
                    filtered_headlines.append(hl)
                    filtered_scores.append(sc)

            ticker_mentions = len(filtered_headlines)
            if filtered_scores:
                avg_score = sum(filtered_scores) / len(filtered_scores)
                bullets = _pick_top_bullets(filtered_headlines, filtered_scores, 5)
            else:
                # No ticker-specific headlines; fall back to aggregate
                avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
                bullets = _pick_top_bullets(all_headlines, all_scores, 5)
        else:
            ticker_mentions = len(all_headlines)
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            bullets = _pick_top_bullets(all_headlines, all_scores, 5)

        clamped_score = max(-1.0, min(1.0, avg_score))
        return {
            "score": round(clamped_score, 4),
            "label": _label_from_score(clamped_score),
            "n_headlines": len(all_headlines),
            "ticker": ticker or "MARKET",
            "ticker_mentions": ticker_mentions,
            "bullets": bullets,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _stub(ticker: str) -> Dict[str, Any]:
        """Return a neutral stub when sentiment data is unavailable."""
        return {
            "score": 0.0,
            "label": "neutral",
            "n_headlines": 0,
            "ticker": ticker or "MARKET",
            "ticker_mentions": 0,
            "bullets": [],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
