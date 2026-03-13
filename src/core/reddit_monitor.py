"""
Reddit Sentiment Monitor
========================

Scrapes crypto subreddits for sentiment analysis.
Uses strict bot filtering: account >= 90 days, karma >= 500.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)

# Try to import PRAW
try:
    import praw
    from praw.models import Submission, Comment
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("PRAW not installed. Run: pip install praw")


@dataclass
class RedditSentiment:
    """Aggregated Reddit sentiment snapshot."""
    timestamp: str
    posts_analyzed: int
    comments_analyzed: int
    users_filtered_out: int  # Transparency on bot filtering
    
    bullish_mentions: int
    bearish_mentions: int
    neutral_mentions: int
    
    sentiment_score: float  # -100 to +100
    confidence: float       # 0 to 1, based on sample size
    
    top_tickers: List[Tuple[str, int]] = field(default_factory=list)
    trending_terms: List[Tuple[str, int]] = field(default_factory=list)
    
    subreddits_scanned: List[str] = field(default_factory=list)


class RedditMonitor:
    """
    Reddit sentiment monitor with strict bot filtering.
    
    Monitors:
    - r/Bitcoin
    - r/CryptoCurrency
    - r/wallstreetbets (crypto mentions only)
    
    Strict filtering:
    - Account age >= 90 days
    - Total karma >= 500
    """
    
    # Subreddits to monitor
    SUBREDDITS = [
        'Bitcoin',
        'CryptoCurrency',
        'wallstreetbets',
    ]
    
    # Keywords for sentiment analysis (lowercase)
    BULLISH_KEYWORDS = [
        'bullish', 'moon', 'buy', 'buying', 'long', 'calls',
        'hodl', 'hold', 'dip', 'btfd', 'undervalued', 'accumulate',
        'breakout', 'rally', 'pump', 'green', 'gains', 'rocket',
        'diamond hands', 'ath', 'all time high', 'to the moon',
    ]
    
    BEARISH_KEYWORDS = [
        'bearish', 'sell', 'selling', 'short', 'puts', 'dump',
        'crash', 'bubble', 'overvalued', 'scam', 'rug', 'dead',
        'rip', 'red', 'loss', 'panic', 'fear', 'capitulation',
        'paper hands', 'exit', 'top', 'correction', 'plunge',
    ]
    
    # Tickers to track
    TICKERS = ['BTC', 'ETH', 'IBIT', 'BITO', 'SOL', 'XRP', 'DOGE']
    
    # Bot filtering thresholds (STRICT)
    MIN_ACCOUNT_AGE_DAYS = 90
    MIN_KARMA = 500
    
    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        user_agent: str = "Argus/1.0",
    ):
        """
        Initialize Reddit monitor.
        
        Args:
            client_id: Reddit app client ID
            client_secret: Reddit app client secret
            user_agent: User agent string for Reddit API
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        
        self._reddit = None
        self._cache: Optional[RedditSentiment] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)
        
        # Statistics
        self._total_scanned = 0
        self._total_filtered = 0
        
        if PRAW_AVAILABLE and client_id and client_secret:
            self._init_reddit()
        
        logger.info(
            f"RedditMonitor initialized: "
            f"min_age={self.MIN_ACCOUNT_AGE_DAYS}d, min_karma={self.MIN_KARMA}"
        )
    
    def _init_reddit(self) -> None:
        """Initialize PRAW Reddit instance."""
        try:
            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
            logger.info("PRAW Reddit client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PRAW: {e}")
            self._reddit = None
    
    def is_valid_user(self, author) -> Tuple[bool, str]:
        """
        Check if user passes strict bot filtering.
        
        Criteria:
        - Account age >= 90 days
        - Total karma >= 500
        - Not AutoModerator or [deleted]
        
        Args:
            author: PRAW Redditor object
            
        Returns:
            (is_valid, rejection_reason)
        """
        if author is None:
            return False, "deleted_user"
        
        try:
            name = author.name
        except AttributeError:
            return False, "no_name"
        
        # Filter known bots
        if name.lower() in ['automoderator', '[deleted]', 'autotldr']:
            return False, "known_bot"
        
        try:
            # Check account age
            created_utc = datetime.fromtimestamp(author.created_utc, tz=timezone.utc)
            account_age_days = (datetime.now(timezone.utc) - created_utc).days
            
            if account_age_days < self.MIN_ACCOUNT_AGE_DAYS:
                return False, f"account_too_new_{account_age_days}d"
            
            # Check karma
            total_karma = author.comment_karma + author.link_karma
            
            if total_karma < self.MIN_KARMA:
                return False, f"low_karma_{total_karma}"
            
            return True, "passed"
            
        except Exception as e:
            return False, f"error_{str(e)[:20]}"
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for sentiment and tickers.
        
        Args:
            text: Post title, body, or comment text
            
        Returns:
            Dict with sentiment scores and detected tickers
        """
        text_lower = text.lower()
        
        # Count keyword matches
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        # Detect tickers (case-sensitive for accuracy)
        text_upper = text.upper()
        tickers_found = [t for t in self.TICKERS if f' {t} ' in f' {text_upper} ']
        
        return {
            'bullish': bullish_count,
            'bearish': bearish_count,
            'tickers': tickers_found,
        }
    
    async def fetch_sentiment(self, force_refresh: bool = False) -> Optional[RedditSentiment]:
        """
        Fetch current sentiment from Reddit.
        
        Uses caching to respect rate limits.
        
        Args:
            force_refresh: Bypass cache
            
        Returns:
            RedditSentiment snapshot or None
        """
        # Check cache
        if not force_refresh and self._cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_ttl:
                return self._cache
        
        if not self._reddit:
            logger.warning("Reddit client not initialized")
            return None
        
        # Run synchronous PRAW calls in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._fetch_sync)
        
        if result:
            self._cache = result
            self._cache_time = datetime.now()
        
        return result
    
    def _fetch_sync(self) -> Optional[RedditSentiment]:
        """Synchronous Reddit fetching (PRAW is not async)."""
        posts_analyzed = 0
        comments_analyzed = 0
        users_filtered = 0
        
        total_bullish = 0
        total_bearish = 0
        total_neutral = 0
        
        ticker_counts = Counter()
        term_counts = Counter()
        
        try:
            for subreddit_name in self.SUBREDDITS:
                try:
                    subreddit = self._reddit.subreddit(subreddit_name)
                    
                    # Get hot posts (last 24 hours-ish)
                    for post in subreddit.hot(limit=50):
                        # Skip stickied posts
                        if post.stickied:
                            continue
                        
                        # For WSB, only crypto-related posts
                        if subreddit_name == 'wallstreetbets':
                            title_upper = post.title.upper()
                            if not any(t in title_upper for t in ['BTC', 'BITCOIN', 'CRYPTO', 'IBIT', 'BITO']):
                                continue
                        
                        # Check author validity
                        is_valid, reason = self.is_valid_user(post.author)
                        if not is_valid:
                            users_filtered += 1
                            continue
                        
                        # Analyze post
                        full_text = f"{post.title} {post.selftext or ''}"
                        analysis = self.analyze_text(full_text)
                        
                        # Weight by score (upvotes)
                        weight = min(post.score / 100, 10)  # Cap at 10x
                        weight = max(weight, 1)  # Min 1x
                        
                        total_bullish += analysis['bullish'] * weight
                        total_bearish += analysis['bearish'] * weight
                        
                        for ticker in analysis['tickers']:
                            ticker_counts[ticker] += 1
                        
                        posts_analyzed += 1
                        
                        # Sample top comments (expensive, limit)
                        post.comments.replace_more(limit=0)  # Don't fetch more
                        for comment in post.comments[:10]:
                            is_valid, reason = self.is_valid_user(comment.author)
                            if not is_valid:
                                users_filtered += 1
                                continue
                            
                            comment_analysis = self.analyze_text(comment.body)
                            total_bullish += comment_analysis['bullish']
                            total_bearish += comment_analysis['bearish']
                            comments_analyzed += 1
                            
                except Exception as e:
                    logger.warning(f"Error fetching r/{subreddit_name}: {e}")
                    continue
            
            # Calculate sentiment score (-100 to +100)
            total_mentions = total_bullish + total_bearish
            if total_mentions > 0:
                sentiment_score = ((total_bullish - total_bearish) / total_mentions) * 100
            else:
                sentiment_score = 0
            
            # Neutral = no clear signal
            total_neutral = max(0, posts_analyzed - int(total_bullish) - int(total_bearish))
            
            # Confidence based on sample size
            confidence = min(posts_analyzed / 100, 1.0)
            
            # Update stats
            self._total_scanned += posts_analyzed + comments_analyzed
            self._total_filtered += users_filtered
            
            sentiment = RedditSentiment(
                timestamp=datetime.now(timezone.utc).isoformat(),
                posts_analyzed=posts_analyzed,
                comments_analyzed=comments_analyzed,
                users_filtered_out=users_filtered,
                bullish_mentions=int(total_bullish),
                bearish_mentions=int(total_bearish),
                neutral_mentions=total_neutral,
                sentiment_score=sentiment_score,
                confidence=confidence,
                top_tickers=ticker_counts.most_common(5),
                trending_terms=[],  # Could add more analysis
                subreddits_scanned=self.SUBREDDITS,
            )
            
            logger.info(
                f"Reddit sentiment: {sentiment_score:+.1f}, "
                f"analyzed {posts_analyzed} posts, filtered {users_filtered} users"
            )
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Reddit fetch error: {e}")
            return None
    
    def compare_to_position(
        self, 
        sentiment: RedditSentiment,
        open_trades: List = None,
    ) -> str:
        """
        Compare retail sentiment to our current position.
        
        Args:
            sentiment: Current Reddit sentiment
            open_trades: List of open paper trades (if any)
            
        Returns:
            Insight message for Telegram
        """
        open_trades = open_trades or []
        
        # Determine our bias
        our_bias = "neutral"
        if open_trades:
            # We're selling puts = bullish bias
            our_bias = "bullish"
        
        score = sentiment.sentiment_score
        
        if our_bias == "bullish":
            if score < -50:
                return "✅ Contrarian: You're bullish, retail is fearful — historically good"
            elif score < -20:
                return "📊 Retail sentiment is negative, you're positioned against the crowd"
            elif score > 70:
                return "⚠️ Crowded trade: You AND retail are bullish — exercise caution"
            elif score > 40:
                return "📊 Retail sentiment aligns with your position"
            else:
                return "📊 Retail sentiment is mixed"
        else:
            if score < -50:
                return "📉 Extreme fear in retail — potential opportunity for bullish trades"
            elif score > 70:
                return "📈 Extreme greed in retail — potential contrarian short opportunity"
            else:
                return f"📊 Retail sentiment: {score:+.0f}/100"
    
    def format_telegram(self, sentiment: RedditSentiment) -> str:
        """Format sentiment for Telegram notification."""
        # Emoji based on score
        if sentiment.sentiment_score > 50:
            emoji = "🟢"
            label = "BULLISH"
        elif sentiment.sentiment_score > 20:
            emoji = "🟡"
            label = "SLIGHTLY BULLISH"
        elif sentiment.sentiment_score > -20:
            emoji = "⚪"
            label = "NEUTRAL"
        elif sentiment.sentiment_score > -50:
            emoji = "🟠"
            label = "SLIGHTLY BEARISH"
        else:
            emoji = "🔴"
            label = "BEARISH"
        
        # Top tickers
        tickers_str = ", ".join([f"{t}({c})" for t, c in sentiment.top_tickers[:3]]) or "None"
        
        return f"""📱 REDDIT SENTIMENT UPDATE

{emoji} Overall: {label} ({sentiment.sentiment_score:+.0f}/100)

📊 Analysis:
• Posts scanned: {sentiment.posts_analyzed}
• Comments: {sentiment.comments_analyzed}
• Bots filtered: {sentiment.users_filtered_out}

📈 Breakdown:
• Bullish mentions: {sentiment.bullish_mentions}
• Bearish mentions: {sentiment.bearish_mentions}

🏷️ Top tickers: {tickers_str}

Confidence: {sentiment.confidence:.0%}
Subreddits: {', '.join(sentiment.subreddits_scanned)}"""
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        return {
            'total_scanned': self._total_scanned,
            'total_filtered': self._total_filtered,
            'filter_rate': self._total_filtered / max(self._total_scanned, 1),
            'cache_valid': self._cache is not None,
            'praw_available': PRAW_AVAILABLE,
            'reddit_connected': self._reddit is not None,
        }


async def test_reddit_monitor():
    """Test the Reddit monitor (requires API keys)."""
    print("Reddit Monitor Test")
    print("=" * 40)
    
    if not PRAW_AVAILABLE:
        print("PRAW not installed. Run: pip install praw")
        return
    
    # These would come from secrets.yaml
    # For testing, you can hardcode temporarily
    monitor = RedditMonitor(
        client_id=None,  # Add your client_id
        client_secret=None,  # Add your client_secret
        user_agent="Argus/1.0 Test",
    )
    
    if monitor._reddit:
        print("Fetching sentiment...")
        sentiment = await monitor.fetch_sentiment()
        
        if sentiment:
            print(monitor.format_telegram(sentiment))
            print("\nStats:", monitor.get_stats())
        else:
            print("Failed to fetch sentiment")
    else:
        print("Reddit client not initialized (need API keys)")
        
        # Test text analysis without API
        print("\nTesting text analysis:")
        test_texts = [
            "BTC to the moon! This is so bullish, diamond hands!",
            "Market is going to crash, sell everything, we're in a bubble",
            "IBIT looking interesting, might buy some",
        ]
        
        for text in test_texts:
            result = monitor.analyze_text(text)
            print(f"  '{text[:40]}...'")
            print(f"    Bullish: {result['bullish']}, Bearish: {result['bearish']}, Tickers: {result['tickers']}")


if __name__ == "__main__":
    asyncio.run(test_reddit_monitor())
