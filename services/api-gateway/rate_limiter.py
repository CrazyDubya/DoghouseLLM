import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based rate limiter with sliding window"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.limits = {
            "free": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "burst_allowance": 10
            },
            "standard": {
                "requests_per_minute": 120,
                "requests_per_hour": 5000,
                "burst_allowance": 20
            },
            "premium": {
                "requests_per_minute": 300,
                "requests_per_hour": 20000,
                "burst_allowance": 50
            },
            "enterprise": {
                "requests_per_minute": 1000,
                "requests_per_hour": 100000,
                "burst_allowance": 100
            }
        }

    async def initialize(self):
        """Initialize the rate limiter"""
        logger.info("Rate limiter initialized")

    def is_healthy(self) -> bool:
        """Check if rate limiter is healthy"""
        return True

    async def check_limit(self, user_id: str, tier: str = "free") -> bool:
        """Check if user has exceeded rate limits"""
        if tier not in self.limits:
            tier = "free"

        limits = self.limits[tier]

        # Check minute limit (sliding window)
        minute_key = f"rate_limit:minute:{user_id}"
        minute_allowed = await self._check_sliding_window(
            minute_key,
            limits["requests_per_minute"],
            60
        )

        if not minute_allowed:
            raise RateLimitExceeded("Requests per minute limit exceeded")

        # Check hour limit (sliding window)
        hour_key = f"rate_limit:hour:{user_id}"
        hour_allowed = await self._check_sliding_window(
            hour_key,
            limits["requests_per_hour"],
            3600
        )

        if not hour_allowed:
            raise RateLimitExceeded("Requests per hour limit exceeded")

        # Check burst allowance
        burst_allowed = await self._check_burst_limit(
            user_id,
            limits["burst_allowance"]
        )

        if not burst_allowed:
            raise RateLimitExceeded("Burst limit exceeded")

        return True

    async def _check_sliding_window(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> bool:
        """Check sliding window rate limit"""
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds

        # Use Redis sorted set for sliding window
        pipe = self.redis.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current entries
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(now): now})

        # Set expiration
        pipe.expire(key, window_seconds + 10)

        results = await pipe.execute()
        current_count = results[1]

        return current_count < limit

    async def _check_burst_limit(self, user_id: str, burst_limit: int) -> bool:
        """Check burst limit (requests in last 10 seconds)"""
        burst_key = f"rate_limit:burst:{user_id}"
        now = datetime.utcnow().timestamp()
        burst_window_start = now - 10  # 10 seconds

        pipe = self.redis.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(burst_key, 0, burst_window_start)

        # Count current entries
        pipe.zcard(burst_key)

        # Add current request
        pipe.zadd(burst_key, {str(now): now})

        # Set expiration
        pipe.expire(burst_key, 20)

        results = await pipe.execute()
        current_count = results[1]

        return current_count < burst_limit

    async def get_remaining_quota(self, user_id: str, tier: str = "free") -> Dict[str, int]:
        """Get remaining quota for a user"""
        if tier not in self.limits:
            tier = "free"

        limits = self.limits[tier]
        now = datetime.utcnow().timestamp()

        # Check minute quota
        minute_key = f"rate_limit:minute:{user_id}"
        minute_count = await self.redis.zcount(minute_key, now - 60, now)
        minute_remaining = max(0, limits["requests_per_minute"] - minute_count)

        # Check hour quota
        hour_key = f"rate_limit:hour:{user_id}"
        hour_count = await self.redis.zcount(hour_key, now - 3600, now)
        hour_remaining = max(0, limits["requests_per_hour"] - hour_count)

        # Check burst quota
        burst_key = f"rate_limit:burst:{user_id}"
        burst_count = await self.redis.zcount(burst_key, now - 10, now)
        burst_remaining = max(0, limits["burst_allowance"] - burst_count)

        return {
            "minute_remaining": minute_remaining,
            "hour_remaining": hour_remaining,
            "burst_remaining": burst_remaining,
            "reset_time": int(now + 60)  # Next minute reset
        }

    async def reset_user_limits(self, user_id: str):
        """Reset all limits for a user (admin function)"""
        keys = [
            f"rate_limit:minute:{user_id}",
            f"rate_limit:hour:{user_id}",
            f"rate_limit:burst:{user_id}"
        ]

        for key in keys:
            await self.redis.delete(key)

        logger.info(f"Reset rate limits for user {user_id}")

    async def get_user_stats(self, user_id: str) -> Dict[str, any]:
        """Get rate limiting statistics for a user"""
        now = datetime.utcnow().timestamp()

        # Get current usage
        minute_key = f"rate_limit:minute:{user_id}"
        minute_count = await self.redis.zcount(minute_key, now - 60, now)

        hour_key = f"rate_limit:hour:{user_id}"
        hour_count = await self.redis.zcount(hour_key, now - 3600, now)

        burst_key = f"rate_limit:burst:{user_id}"
        burst_count = await self.redis.zcount(burst_key, now - 10, now)

        # Get request timestamps for analysis
        recent_requests = await self.redis.zrangebyscore(
            hour_key, now - 3600, now, withscores=True
        )

        return {
            "user_id": user_id,
            "current_usage": {
                "minute": minute_count,
                "hour": hour_count,
                "burst": burst_count
            },
            "recent_requests": len(recent_requests),
            "last_request": max([score for _, score in recent_requests]) if recent_requests else None
        }

    async def block_user(self, user_id: str, duration_seconds: int = 3600):
        """Temporarily block a user"""
        block_key = f"rate_limit:blocked:{user_id}"
        await self.redis.setex(block_key, duration_seconds, "blocked")
        logger.warning(f"Blocked user {user_id} for {duration_seconds} seconds")

    async def is_user_blocked(self, user_id: str) -> bool:
        """Check if user is blocked"""
        block_key = f"rate_limit:blocked:{user_id}"
        blocked = await self.redis.get(block_key)
        return blocked is not None

    async def get_global_stats(self) -> Dict[str, any]:
        """Get global rate limiting statistics"""
        # This is a simplified version - in production you'd want more detailed metrics
        try:
            # Count active rate limit keys
            keys = await self.redis.keys("rate_limit:*")
            active_users = len(set(key.split(":")[2] for key in keys if len(key.split(":")) > 2))

            # Count blocked users
            blocked_keys = await self.redis.keys("rate_limit:blocked:*")
            blocked_users = len(blocked_keys)

            return {
                "active_users": active_users,
                "blocked_users": blocked_users,
                "total_keys": len(keys)
            }

        except Exception as e:
            logger.error(f"Error getting global stats: {e}")
            return {
                "active_users": 0,
                "blocked_users": 0,
                "total_keys": 0
            }


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""

    def __init__(self, message: str, retry_after: int = 60):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)


class IPRateLimiter:
    """IP-based rate limiter for DDoS protection"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.ip_limits = {
            "requests_per_minute": 1000,
            "requests_per_hour": 10000
        }

    async def check_ip_limit(self, ip_address: str) -> bool:
        """Check if IP has exceeded limits"""
        # Check minute limit
        minute_key = f"ip_rate_limit:minute:{ip_address}"
        minute_allowed = await self._check_ip_window(
            minute_key,
            self.ip_limits["requests_per_minute"],
            60
        )

        if not minute_allowed:
            # Auto-block suspicious IPs
            await self._block_ip(ip_address, 300)  # 5 minutes
            raise RateLimitExceeded("IP rate limit exceeded")

        # Check hour limit
        hour_key = f"ip_rate_limit:hour:{ip_address}"
        hour_allowed = await self._check_ip_window(
            hour_key,
            self.ip_limits["requests_per_hour"],
            3600
        )

        if not hour_allowed:
            await self._block_ip(ip_address, 3600)  # 1 hour
            raise RateLimitExceeded("IP rate limit exceeded")

        return True

    async def _check_ip_window(self, key: str, limit: int, window_seconds: int) -> bool:
        """Check IP-based sliding window"""
        current_count = await self.redis.incr(key)

        if current_count == 1:
            await self.redis.expire(key, window_seconds)

        return current_count <= limit

    async def _block_ip(self, ip_address: str, duration_seconds: int):
        """Block an IP address"""
        block_key = f"ip_blocked:{ip_address}"
        await self.redis.setex(block_key, duration_seconds, "blocked")
        logger.warning(f"Blocked IP {ip_address} for {duration_seconds} seconds")

    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        block_key = f"ip_blocked:{ip_address}"
        blocked = await self.redis.get(block_key)
        return blocked is not None