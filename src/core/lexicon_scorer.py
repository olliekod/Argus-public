"""Lexicon-based sentiment scoring for finance headlines."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger("argus.lexicon_scorer")

_FALLBACK_POSITIVE = {
    "gain", "gains", "growth", "beat", "beats", "bullish", "rally", "upgrade", "surge", "strong",
}
_FALLBACK_NEGATIVE = {
    "loss", "losses", "decline", "miss", "misses", "bearish", "drop", "downgrade", "weak", "risk",
}


class LexiconScorer:
    """Score text using positive/negative word lists."""

    def __init__(self, lexicon: str = "loughran_mcdonald", lexicon_path: Optional[str] = None):
        self._lexicon = lexicon
        self._positive, self._negative = self._load_lexicon(lexicon, lexicon_path)

    @staticmethod
    def _load_words(path: Path) -> Set[str]:
        words: Set[str] = set()
        if not path.exists():
            return words
        for line in path.read_text(encoding="utf-8").splitlines():
            token = line.strip().lower()
            if token and not token.startswith("#"):
                words.add(token)
        return words

    def _load_lexicon(self, lexicon: str, lexicon_path: Optional[str]) -> Tuple[Set[str], Set[str]]:
        if lexicon_path:
            root = Path(lexicon_path)
            pos = self._load_words(root / "loughran_positive.txt")
            neg = self._load_words(root / "loughran_negative.txt")
            if pos and neg:
                return pos, neg
            logger.warning("Lexicon path missing expected files, falling back: %s", lexicon_path)

        repo_root = Path(__file__).resolve().parents[2]
        default_pos = repo_root / "data" / "loughran_positive.txt"
        default_neg = repo_root / "data" / "loughran_negative.txt"
        pos = self._load_words(default_pos)
        neg = self._load_words(default_neg)
        if pos and neg:
            return pos, neg

        logger.warning("Using built-in minimal sentiment lexicon for %s", lexicon)
        return set(_FALLBACK_POSITIVE), set(_FALLBACK_NEGATIVE)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [tok for tok in re.split(r"[^a-zA-Z]+", text.lower()) if tok]

    def score_text(self, text: str) -> tuple[float, Dict[str, int]]:
        """Returns ``(score, counts)`` where counts has pos/neg/word counts."""
        tokens = self._tokenize(text or "")
        pos_count = sum(1 for token in tokens if token in self._positive)
        neg_count = sum(1 for token in tokens if token in self._negative)
        score = (pos_count - neg_count) / (pos_count + neg_count + 1)
        return score, {
            "pos_count": pos_count,
            "neg_count": neg_count,
            "word_count": len(tokens),
        }
