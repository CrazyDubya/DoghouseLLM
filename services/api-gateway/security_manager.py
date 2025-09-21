import asyncio
import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional

import jwt
import redis.asyncio as redis
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class SecurityManager:
    """Centralized security management for authentication and authorization"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.jwt_secret = os.getenv("JWT_SECRET", self._generate_jwt_secret())
        self.jwt_algorithm = "HS256"
        self.token_expiry = 3600  # 1 hour

        # Initialize encryption
        self.encryption_key = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
        self.fernet = Fernet(self.encryption_key.encode())

    def _generate_jwt_secret(self) -> str:
        """Generate a random JWT secret"""
        return secrets.token_urlsafe(32)

    async def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return user information"""
        try:
            # Hash the API key for lookup
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Check if API key exists in Redis
            user_data = await self.redis.hgetall(f"api_key:{key_hash}")

            if not user_data:
                # For demo purposes, create a default user
                # In production, this would check a proper user database
                return await self._create_demo_user(api_key, key_hash)

            # Decode user data
            user_info = {}
            for key, value in user_data.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value
                user_info[key_str] = value_str

            # Check if API key is active
            if user_info.get("status") != "active":
                return None

            return user_info

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    async def _create_demo_user(self, api_key: str, key_hash: str) -> Dict:
        """Create a demo user for testing purposes"""
        user_id = f"demo_user_{secrets.token_hex(8)}"

        user_info = {
            "user_id": user_id,
            "api_key": key_hash,
            "tier": "standard",
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "email": f"{user_id}@demo.com"
        }

        # Store in Redis
        await self.redis.hset(f"api_key:{key_hash}", mapping=user_info)
        await self.redis.hset(f"user:{user_id}", mapping=user_info)

        logger.info(f"Created demo user {user_id}")
        return user_info

    async def generate_token(self, user_info: Dict) -> str:
        """Generate JWT token for authenticated user"""
        try:
            payload = {
                "user_id": user_info["user_id"],
                "tier": user_info.get("tier", "free"),
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry)
            }

            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

            # Store token in Redis for tracking
            await self.redis.setex(
                f"token:{user_info['user_id']}:{token}",
                self.token_expiry,
                "active"
            )

            return token

        except Exception as e:
            logger.error(f"Error generating token: {e}")
            raise

    async def verify_token(self, token: str) -> Dict:
        """Verify JWT token and return payload"""
        try:
            # Decode and verify token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # Check if token is revoked
            user_id = payload["user_id"]
            token_key = f"token:{user_id}:{token}"

            if not await self.redis.exists(token_key):
                raise jwt.InvalidTokenError("Token revoked or expired")

            return payload

        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token expired")
        except jwt.InvalidTokenError as e:
            raise jwt.InvalidTokenError(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            raise jwt.InvalidTokenError("Token verification failed")

    async def revoke_token(self, token: str, user_id: str):
        """Revoke a specific token"""
        try:
            token_key = f"token:{user_id}:{token}"
            await self.redis.delete(token_key)
            logger.info(f"Revoked token for user {user_id}")
        except Exception as e:
            logger.error(f"Error revoking token: {e}")

    async def revoke_all_user_tokens(self, user_id: str):
        """Revoke all tokens for a user"""
        try:
            pattern = f"token:{user_id}:*"
            keys = await self.redis.keys(pattern)

            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Revoked all tokens for user {user_id}")

        except Exception as e:
            logger.error(f"Error revoking user tokens: {e}")

    async def create_api_key(self, user_id: str, tier: str = "free") -> str:
        """Create a new API key for a user"""
        try:
            # Generate API key
            api_key = f"mac_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Store API key info
            key_info = {
                "user_id": user_id,
                "api_key": key_hash,
                "tier": tier,
                "status": "active",
                "created_at": datetime.utcnow().isoformat()
            }

            await self.redis.hset(f"api_key:{key_hash}", mapping=key_info)

            # Also store for user lookup
            await self.redis.sadd(f"user_api_keys:{user_id}", key_hash)

            logger.info(f"Created API key for user {user_id}")
            return api_key

        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            raise

    async def revoke_api_key(self, api_key: str):
        """Revoke an API key"""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Get user ID first
            key_info = await self.redis.hgetall(f"api_key:{key_hash}")
            if key_info:
                user_id = key_info.get(b"user_id", key_info.get("user_id"))
                if isinstance(user_id, bytes):
                    user_id = user_id.decode()

                # Remove from user's key set
                await self.redis.srem(f"user_api_keys:{user_id}", key_hash)

            # Delete API key
            await self.redis.delete(f"api_key:{key_hash}")

            # Revoke all tokens for this API key (if we tracked them)
            logger.info(f"Revoked API key {key_hash[:8]}...")

        except Exception as e:
            logger.error(f"Error revoking API key: {e}")

    async def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise

    async def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise

    async def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            import bcrypt
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode(), salt)
            return hashed.decode()
        except ImportError:
            # Fallback to simpler hashing if bcrypt not available
            import hashlib
            salt = secrets.token_hex(16)
            hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return f"{salt}:{hashed.hex()}"

    async def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            import bcrypt
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except ImportError:
            # Fallback verification
            import hashlib
            salt, stored_hash = hashed.split(':')
            computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return computed_hash.hex() == stored_hash

    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        details: Optional[Dict] = None,
        severity: str = "info"
    ):
        """Log security events for monitoring"""
        try:
            event = {
                "type": event_type,
                "user_id": user_id,
                "details": details or {},
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
                "ip_address": "unknown"  # Would be filled from request context
            }

            # Store in Redis for monitoring
            await self.redis.lpush("security_events", str(event))
            await self.redis.ltrim("security_events", 0, 999)  # Keep last 1000 events

            if severity in ["warning", "error", "critical"]:
                logger.warning(f"Security event: {event_type} - {details}")

        except Exception as e:
            logger.error(f"Error logging security event: {e}")

    async def check_user_permissions(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has permission to perform action on resource"""
        try:
            # Simple permission system - in production would be more sophisticated
            user_info = await self.redis.hgetall(f"user:{user_id}")

            if not user_info:
                return False

            # Check user status
            status = user_info.get(b"status", user_info.get("status", ""))
            if isinstance(status, bytes):
                status = status.decode()

            if status != "active":
                return False

            # Basic permission checks based on tier
            tier = user_info.get(b"tier", user_info.get("tier", "free"))
            if isinstance(tier, bytes):
                tier = tier.decode()

            # Define permissions by tier
            permissions = {
                "free": ["read", "create_agent"],
                "standard": ["read", "create_agent", "update_agent"],
                "premium": ["read", "create_agent", "update_agent", "delete_agent", "admin_read"],
                "enterprise": ["*"]  # All permissions
            }

            user_permissions = permissions.get(tier, ["read"])

            return "*" in user_permissions or action in user_permissions

        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
            return False

    async def get_security_stats(self) -> Dict:
        """Get security statistics"""
        try:
            # Count active sessions
            session_keys = await self.redis.keys("token:*")
            active_sessions = len(session_keys)

            # Count API keys
            api_keys = await self.redis.keys("api_key:*")
            total_api_keys = len(api_keys)

            # Get recent security events
            events = await self.redis.lrange("security_events", 0, 9)
            recent_events = len(events)

            return {
                "active_sessions": active_sessions,
                "total_api_keys": total_api_keys,
                "recent_security_events": recent_events,
                "encryption_enabled": True,
                "jwt_enabled": True
            }

        except Exception as e:
            logger.error(f"Error getting security stats: {e}")
            return {
                "active_sessions": 0,
                "total_api_keys": 0,
                "recent_security_events": 0,
                "encryption_enabled": True,
                "jwt_enabled": True
            }