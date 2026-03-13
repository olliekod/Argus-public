"""Headline ingestion for news sentiment.

Fetches headlines from RSS feeds (required) and optionally NewsAPI when a key is
configured. Output is deterministic and JSON-serialisable.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("argus.headline_fetcher")


class HeadlineFetcher:
    """Fetch and normalize headlines from configured providers."""

    def __init__(
        self,
        feeds: List[str],
        newsapi_key: Optional[str],
        max_headlines: int,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._feeds = [str(feed).strip() for feed in (feeds or []) if str(feed).strip()]
        self._newsapi_key = self._resolve_env_token(newsapi_key)
        self._max_headlines = max(1, int(max_headlines or 50))
        self._session = session
        self._owns_session = session is None
        self._throttle_lock = asyncio.Lock()
        self._min_fetch_interval_s = 60.0
        self._last_fetch_ts = 0.0

    @staticmethod
    def _resolve_env_token(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        token = str(value)
        if token.startswith("${") and token.endswith("}"):
            env_var = token[2:-1]
            token = os.getenv(env_var, "")
        token = token.strip()
        return token or None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _throttle(self) -> None:
        async with self._throttle_lock:
            now = time.monotonic()
            elapsed = now - self._last_fetch_ts
            if elapsed < self._min_fetch_interval_s:
                await asyncio.sleep(self._min_fetch_interval_s - elapsed)
            self._last_fetch_ts = time.monotonic()

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", title.lower())).strip()

    @staticmethod
    def _parse_published(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_with_feedparser(content: str) -> Optional[List[Dict[str, Any]]]:
        try:
            module = importlib.import_module("feedparser")
        except ModuleNotFoundError:
            return None

        parsed = module.parse(content)
        out: List[Dict[str, Any]] = []
        for entry in parsed.entries:
            title = str(entry.get("title") or "").strip()
            if not title:
                continue
            out.append(
                {
                    "title": title,
                    "summary": str(entry.get("summary") or entry.get("description") or "").strip(),
                    "published": HeadlineFetcher._parse_published(entry.get("published") or entry.get("updated")),
                }
            )
        return out

    @staticmethod
    def _parse_with_etree(content: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return out

        items = root.findall(".//item") + root.findall(".//{*}entry")
        for item in items:
            title_node = item.find("title") or item.find("{*}title")
            title = (title_node.text or "").strip() if title_node is not None else ""
            if not title:
                continue
            summary_node = (
                item.find("description")
                or item.find("summary")
                or item.find("{*}summary")
                or item.find("{*}description")
            )
            pub_node = (
                item.find("pubDate")
                or item.find("published")
                or item.find("updated")
                or item.find("{*}published")
                or item.find("{*}updated")
            )
            out.append(
                {
                    "title": title,
                    "summary": (summary_node.text or "").strip() if summary_node is not None else "",
                    "published": HeadlineFetcher._parse_published(pub_node.text if pub_node is not None else None),
                }
            )
        return out

    async def _fetch_rss_headlines(self) -> List[Dict[str, Any]]:
        headlines: List[Dict[str, Any]] = []
        if not self._feeds:
            return headlines

        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(total=20)
        for feed_url in self._feeds:
            try:
                async with session.get(feed_url, timeout=timeout) as resp:
                    if resp.status != 200:
                        logger.warning("RSS fetch failed status=%d feed=%s", resp.status, feed_url)
                        continue
                    content = await resp.text()

                parsed = self._parse_with_feedparser(content)
                entries = parsed if parsed is not None else self._parse_with_etree(content)
                for entry in entries:
                    headlines.append(
                        {
                            "title": entry["title"],
                            "summary": entry["summary"],
                            "published": entry["published"],
                            "source": str(feed_url),
                        }
                    )
            except Exception as exc:
                logger.warning("RSS feed fetch error feed=%s err=%s", feed_url, exc)
                continue

        return headlines

    async def _fetch_newsapi_headlines(self) -> List[Dict[str, Any]]:
        if not self._newsapi_key:
            return []
        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(total=20)
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "category": "business",
            "language": "en",
            "pageSize": str(self._max_headlines),
            "apiKey": self._newsapi_key,
        }

        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("NewsAPI fetch failed status=%d body=%s", resp.status, text[:180])
                    return []
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("NewsAPI fetch error: %s", exc)
            return []

        results: List[Dict[str, Any]] = []
        for article in data.get("articles") or []:
            if not isinstance(article, dict):
                continue
            title = str(article.get("title") or "").strip()
            if not title:
                continue
            published = self._parse_published(article.get("publishedAt"))
            source = article.get("source") or {}
            source_name = source.get("name") if isinstance(source, dict) else None
            results.append(
                {
                    "title": title,
                    "summary": str(article.get("description") or "").strip(),
                    "published": published,
                    "source": str(source_name or "newsapi"),
                }
            )
        return results

    async def fetch_headlines(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return deduplicated headlines."""
        await self._throttle()

        cap = min(max(1, int(limit or self._max_headlines)), self._max_headlines)
        rss = await self._fetch_rss_headlines()
        newsapi = await self._fetch_newsapi_headlines()

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in rss + newsapi:
            title = str(item.get("title") or "").strip()
            norm = self._normalize_title(title)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(item)
            if len(deduped) >= cap:
                break

        return deduped
