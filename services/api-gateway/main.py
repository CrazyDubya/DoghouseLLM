import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import redis.asyncio as redis
import httpx
from pydantic import BaseModel
import jwt

from packages.shared_types.models import (
    Agent, AgentRegistration, Action, ApiResponse,
    HealthCheck, ModerationResult
)
from rate_limiter import RateLimiter
from content_moderation import ContentModerator
from security_manager import SecurityManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
redis_client: redis.Redis = None
rate_limiter: RateLimiter = None
content_moderator: ContentModerator = None
security_manager: SecurityManager = None
world_orchestrator_url: str = None
agent_scheduler_url: str = None

# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global redis_client, rate_limiter, content_moderator, security_manager
    global world_orchestrator_url, agent_scheduler_url

    # Initialize Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)

    # Initialize service URLs
    world_orchestrator_url = os.getenv("WORLD_ORCHESTRATOR_URL", "http://localhost:8001")
    agent_scheduler_url = os.getenv("AGENT_SCHEDULER_URL", "http://localhost:8002")

    # Initialize security components
    security_manager = SecurityManager(redis_client)
    rate_limiter = RateLimiter(redis_client)
    content_moderator = ContentModerator()

    await rate_limiter.initialize()
    await content_moderator.initialize()

    logger.info("API Gateway started successfully")

    yield

    # Cleanup
    await redis_client.close()
    logger.info("API Gateway stopped")


app = FastAPI(
    title="Multi-Agent City API Gateway",
    description="Secure API gateway for Multi-Agent City platform",
    version="1.0.0",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AuthRequest(BaseModel):
    api_key: str
    agent_id: Optional[str] = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate JWT token and return user info"""
    token = credentials.credentials

    try:
        # Verify JWT token
        payload = await security_manager.verify_token(token)
        return payload
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Rate limiting dependency
async def check_rate_limit(user: dict = Depends(get_current_user)):
    """Check rate limits for the current user"""
    user_id = user.get("user_id")
    tier = user.get("tier", "free")

    try:
        await rate_limiter.check_limit(user_id, tier)
    except Exception as e:
        logger.warning(f"Rate limit exceeded for user {user_id}: {e}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    services = {
        "redis": "operational" if await redis_client.ping() else "degraded",
        "rate_limiter": "operational" if rate_limiter.is_healthy() else "degraded",
        "content_moderator": "operational" if content_moderator.is_healthy() else "degraded"
    }

    # Check downstream services
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            world_response = await client.get(f"{world_orchestrator_url}/health")
            services["world_orchestrator"] = "operational" if world_response.status_code == 200 else "degraded"

            agent_response = await client.get(f"{agent_scheduler_url}/health")
            services["agent_scheduler"] = "operational" if agent_response.status_code == 200 else "degraded"
    except:
        services["world_orchestrator"] = "degraded"
        services["agent_scheduler"] = "degraded"

    return HealthCheck(
        status="healthy" if all(s == "operational" for s in services.values()) else "degraded",
        version="1.0.0",
        uptime=0,  # Calculate actual uptime
        services=services
    )


@app.post("/auth/token", response_model=AuthResponse)
async def authenticate(auth_request: AuthRequest):
    """Authenticate user and return JWT token"""
    try:
        # Validate API key
        user_info = await security_manager.validate_api_key(auth_request.api_key)

        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Generate JWT token
        token = await security_manager.generate_token(user_info)

        return AuthResponse(
            access_token=token,
            expires_in=3600
        )

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


@app.post("/api/v1/agents/register", response_model=ApiResponse)
async def register_agent(
    registration: AgentRegistration,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Register a new agent"""
    try:
        # Content moderation on agent profile
        moderation_result = await content_moderator.moderate_text(
            f"{registration.name} {registration.profile.dict()}"
        )

        if not moderation_result.allowed:
            raise HTTPException(
                status_code=422,
                detail=f"Content policy violation: {moderation_result.reason}"
            )

        # Create agent object
        agent = Agent(
            name=registration.name,
            user_id=user["user_id"],
            profile=registration.profile,
            state=create_initial_agent_state(),
            model_config=registration.model_config,
            external_endpoint=registration.external_endpoint
        )

        # Register with agent scheduler
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{agent_scheduler_url}/agents/{agent.id}/register",
                json=agent.dict()
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to register agent with scheduler"
                )

        return ApiResponse(
            success=True,
            data={
                "agent_id": str(agent.id),
                "name": agent.name,
                "status": "registered"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agents/{agent_id}/start", response_model=ApiResponse)
async def start_agent(
    agent_id: str,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Start an agent"""
    try:
        # Verify agent ownership
        await verify_agent_ownership(agent_id, user["user_id"])

        # Start agent via scheduler
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{agent_scheduler_url}/agents/{agent_id}/start")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to start agent"
                )

        return ApiResponse(success=True, data={"status": "started"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agents/{agent_id}/action", response_model=ApiResponse)
async def execute_action(
    agent_id: str,
    action: Action,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Execute an action for an agent"""
    try:
        # Verify agent ownership
        await verify_agent_ownership(agent_id, user["user_id"])

        # Content moderation on action
        if action.parameters.message:
            moderation_result = await content_moderator.moderate_text(
                action.parameters.message
            )

            if not moderation_result.allowed:
                raise HTTPException(
                    status_code=422,
                    detail=f"Content policy violation: {moderation_result.reason}"
                )

        # Execute action via scheduler
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{agent_scheduler_url}/agents/{agent_id}/action",
                json=action.dict()
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to execute action"
                )

            result = response.json()
            return ApiResponse(success=True, data=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing action for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/{agent_id}/observation")
async def get_agent_observation(
    agent_id: str,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Get current observation for an agent"""
    try:
        # Verify agent ownership
        await verify_agent_ownership(agent_id, user["user_id"])

        # Get observation from scheduler
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{agent_scheduler_url}/agents/{agent_id}/observation")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to get observation"
                )

            return response.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting observation for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/world/state")
async def get_world_state(
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Get current world state"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{world_orchestrator_url}/world/state")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to get world state"
                )

            return response.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting world state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/{agent_id}/memory/search")
async def search_memories(
    agent_id: str,
    query: str,
    limit: int = 10,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Search agent memories"""
    try:
        # Verify agent ownership
        await verify_agent_ownership(agent_id, user["user_id"])

        # Search memories via scheduler
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{agent_scheduler_url}/agents/{agent_id}/memory/search",
                params={"query": query, "limit": limit}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to search memories"
                )

            return response.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching memories for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/content/moderate", response_model=ModerationResult)
async def moderate_content(
    content: str,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Test content moderation (for development)"""
    try:
        result = await content_moderator.moderate_text(content)
        return result
    except Exception as e:
        logger.error(f"Error moderating content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def create_initial_agent_state():
    """Create initial state for a new agent"""
    from packages.shared_types.models import AgentState, AgentStatus, Location, Coordinates

    return AgentState(
        agent_id=None,  # Will be set by Agent model
        status=AgentStatus.PENDING,
        location=Location(
            district="Downtown",
            neighborhood="Central Plaza",
            coordinates=Coordinates(x=0, y=0, z=0),
            type="plaza"
        ),
        health={"energy": 100, "mood": "neutral", "stress": 0},
        resources={"currency": 1000},
        activity={"current": "idle", "duration": 0},
        relationships={}
    )


async def verify_agent_ownership(agent_id: str, user_id: str):
    """Verify that user owns the agent"""
    # In a real implementation, this would check the database
    # For now, we'll allow all operations for demo purposes
    pass


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )