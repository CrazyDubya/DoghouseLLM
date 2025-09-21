import asyncio
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime

import httpx

from packages.shared_types.models import ModerationResult

logger = logging.getLogger(__name__)


class ContentModerator:
    """Content moderation system with multiple filters"""

    def __init__(self):
        self.openai_client = None
        self.custom_filters = CustomContentFilters()
        self.violation_cache = {}

    async def initialize(self):
        """Initialize the content moderator"""
        # Initialize OpenAI client if API key is available
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.AsyncOpenAI(api_key=api_key)
                logger.info("OpenAI moderation enabled")
            else:
                logger.warning("OpenAI API key not found, using custom filters only")
        except ImportError:
            logger.warning("OpenAI library not installed, using custom filters only")

    def is_healthy(self) -> bool:
        """Check if content moderator is healthy"""
        return True

    async def moderate_text(self, text: str, context: Optional[Dict] = None) -> ModerationResult:
        """Moderate text content through multiple filters"""
        try:
            # Stage 1: Quick custom filters
            custom_result = await self.custom_filters.check(text)
            if not custom_result.allowed:
                return custom_result

            # Stage 2: OpenAI moderation (if available)
            if self.openai_client:
                openai_result = await self._check_openai_moderation(text)
                if not openai_result.allowed:
                    return openai_result

            # Stage 3: Context-aware checks
            if context:
                context_result = await self._check_context(text, context)
                if not context_result.allowed:
                    return context_result

            # All checks passed
            return ModerationResult(
                allowed=True,
                content=text,
                score=0.0
            )

        except Exception as e:
            logger.error(f"Error in content moderation: {e}")
            # Fail open for non-critical errors
            return ModerationResult(
                allowed=True,
                content=text,
                score=0.0,
                reason="Moderation service error"
            )

    async def _check_openai_moderation(self, text: str) -> ModerationResult:
        """Check content using OpenAI moderation API"""
        try:
            response = await self.openai_client.moderations.create(input=text)
            result = response.results[0]

            if result.flagged:
                categories = [cat for cat, flagged in result.categories.dict().items() if flagged]
                return ModerationResult(
                    allowed=False,
                    reason="OpenAI moderation flagged",
                    categories=categories,
                    score=max(result.category_scores.dict().values())
                )

            return ModerationResult(
                allowed=True,
                score=max(result.category_scores.dict().values())
            )

        except Exception as e:
            logger.error(f"OpenAI moderation error: {e}")
            # Continue with other checks if OpenAI fails
            return ModerationResult(allowed=True, score=0.0)

    async def _check_context(self, text: str, context: Dict) -> ModerationResult:
        """Perform context-aware moderation checks"""
        # Check for spam (repeated messages)
        if context.get("agent_id"):
            agent_id = context["agent_id"]
            if await self._is_spam(text, agent_id):
                return ModerationResult(
                    allowed=False,
                    reason="Spam detected",
                    score=1.0
                )

        # Check for inappropriate location-based content
        if context.get("location"):
            location_result = await self._check_location_appropriateness(text, context["location"])
            if not location_result.allowed:
                return location_result

        return ModerationResult(allowed=True, score=0.0)

    async def _is_spam(self, text: str, agent_id: str) -> bool:
        """Check if message is spam (repeated content)"""
        cache_key = f"recent_messages:{agent_id}"

        # Get recent messages from cache
        recent_messages = self.violation_cache.get(cache_key, [])

        # Check for exact duplicates
        if text in recent_messages:
            return True

        # Check for similar messages (simple implementation)
        for msg in recent_messages:
            if self._similarity_score(text, msg) > 0.8:
                return True

        # Add to cache (keep last 10 messages)
        recent_messages.append(text)
        if len(recent_messages) > 10:
            recent_messages.pop(0)

        self.violation_cache[cache_key] = recent_messages
        return False

    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate simple similarity score between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    async def _check_location_appropriateness(self, text: str, location: str) -> ModerationResult:
        """Check if content is appropriate for the location"""
        # Simple implementation - could be expanded with location-specific rules
        inappropriate_in_public = [
            "private", "personal", "secret", "confidential"
        ]

        if "public" in location.lower() or "plaza" in location.lower():
            for word in inappropriate_in_public:
                if word in text.lower():
                    return ModerationResult(
                        allowed=False,
                        reason=f"Inappropriate content for public location",
                        score=0.7
                    )

        return ModerationResult(allowed=True, score=0.0)

    async def moderate_agent_profile(self, profile: Dict) -> ModerationResult:
        """Moderate agent profile data"""
        # Combine all text fields for moderation
        text_content = []

        if "name" in profile:
            text_content.append(profile["name"])

        if "occupation" in profile:
            text_content.append(profile["occupation"])

        if "background" in profile:
            for value in profile["background"].values():
                if isinstance(value, str):
                    text_content.append(value)

        combined_text = " ".join(text_content)
        return await self.moderate_text(combined_text)

    async def get_moderation_stats(self) -> Dict[str, any]:
        """Get moderation statistics"""
        return {
            "total_checks": 0,  # Would track in production
            "violations_detected": 0,
            "openai_enabled": self.openai_client is not None,
            "custom_filters_enabled": True
        }


class CustomContentFilters:
    """Custom content filtering rules"""

    def __init__(self):
        # Blocked patterns (regex)
        self.blocked_patterns = [
            r'(?i)\b(password|secret|key|token)\s*[:=]\s*\S+',  # Credentials
            r'(?i)\b\d{3}-?\d{2}-?\d{4}\b',  # SSN pattern
            r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email
            r'(?i)\b(?:\d{4}[\s-]?){3}\d{4}\b',  # Credit card
            r'(?i)\b(kill|murder|bomb|terrorist|weapon)\b',  # Violence
        ]

        # Blocked words/phrases
        self.blocked_words = [
            "hate", "racist", "sexist", "abuse", "harass",
            "spam", "scam", "fraud", "illegal", "drugs"
        ]

        # Blocked domains (if URLs are detected)
        self.blocked_domains = [
            "malicious-site.com",
            "phishing-domain.net",
            "spam-website.org"
        ]

    async def check(self, text: str) -> ModerationResult:
        """Run custom content filters"""
        text_lower = text.lower()

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, text):
                return ModerationResult(
                    allowed=False,
                    reason="Sensitive information detected",
                    score=1.0
                )

        # Check blocked words
        for word in self.blocked_words:
            if word in text_lower:
                return ModerationResult(
                    allowed=False,
                    reason=f"Inappropriate language: {word}",
                    score=0.9
                )

        # Check for URLs with blocked domains
        url_pattern = r'https?://(?:[-\w.])+(?:\.[a-zA-Z]{2,4})+(?:/?|[/?]\S+)'
        urls = re.findall(url_pattern, text)

        for url in urls:
            for domain in self.blocked_domains:
                if domain in url:
                    return ModerationResult(
                        allowed=False,
                        reason=f"Blocked domain: {domain}",
                        score=1.0
                    )

        # Check for excessive capitalization (potential spam)
        if len(text) > 10 and sum(c.isupper() for c in text) / len(text) > 0.7:
            return ModerationResult(
                allowed=False,
                reason="Excessive capitalization",
                score=0.6
            )

        # Check for repeated characters (potential spam)
        if re.search(r'(.)\1{5,}', text):
            return ModerationResult(
                allowed=False,
                reason="Repeated characters",
                score=0.6
            )

        # All checks passed
        return ModerationResult(allowed=True, score=0.0)

    def add_blocked_word(self, word: str):
        """Add a word to the blocked list"""
        if word not in self.blocked_words:
            self.blocked_words.append(word.lower())

    def remove_blocked_word(self, word: str):
        """Remove a word from the blocked list"""
        if word.lower() in self.blocked_words:
            self.blocked_words.remove(word.lower())

    def add_blocked_pattern(self, pattern: str):
        """Add a regex pattern to the blocked list"""
        if pattern not in self.blocked_patterns:
            self.blocked_patterns.append(pattern)


class ViolationTracker:
    """Track moderation violations for users and agents"""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.violations = {}  # In-memory fallback

    async def record_violation(
        self,
        user_id: str,
        agent_id: Optional[str],
        violation_type: str,
        content: str,
        severity: str = "medium"
    ):
        """Record a content violation"""
        violation = {
            "user_id": user_id,
            "agent_id": agent_id,
            "violation_type": violation_type,
            "content": content[:200],  # Truncate for storage
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Store in Redis if available
        if self.redis:
            key = f"violations:{user_id}"
            await self.redis.lpush(key, str(violation))
            await self.redis.ltrim(key, 0, 99)  # Keep last 100 violations
            await self.redis.expire(key, 86400 * 30)  # 30 days

        # Store in memory as fallback
        if user_id not in self.violations:
            self.violations[user_id] = []
        self.violations[user_id].append(violation)

        logger.warning(f"Recorded violation for user {user_id}: {violation_type}")

    async def get_violation_count(self, user_id: str, hours: int = 24) -> int:
        """Get violation count for a user in the specified time window"""
        if self.redis:
            key = f"violations:{user_id}"
            violations = await self.redis.lrange(key, 0, -1)
            # Would parse timestamps and filter by time window
            return len(violations)
        else:
            # Fallback to in-memory
            user_violations = self.violations.get(user_id, [])
            return len(user_violations)

    async def should_escalate(self, user_id: str) -> bool:
        """Determine if violations should trigger escalated response"""
        recent_violations = await self.get_violation_count(user_id, hours=24)

        # Escalate if more than 5 violations in 24 hours
        return recent_violations > 5