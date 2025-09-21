import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import redis.asyncio as redis
import httpx

from packages.shared_types.models import (
    Agent, Action, ActionResult, Observation, ApiResponse, HealthCheck
)
from agent_runtime import AgentRuntime
from memory_system import MemorySystem
from reflection_engine import ReflectionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
agent_runtime: AgentRuntime = None
memory_system: MemorySystem = None
reflection_engine: ReflectionEngine = None
redis_client: redis.Redis = None
world_orchestrator_url: str = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global agent_runtime, memory_system, reflection_engine, redis_client, world_orchestrator_url

    # Initialize Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)

    # Initialize world orchestrator connection
    world_orchestrator_url = os.getenv("WORLD_ORCHESTRATOR_URL", "http://localhost:8001")

    # Initialize memory system
    memory_system = MemorySystem(redis_client)
    await memory_system.initialize()

    # Initialize reflection engine
    reflection_engine = ReflectionEngine(memory_system)

    # Initialize agent runtime
    agent_runtime = AgentRuntime(
        redis_client=redis_client,
        memory_system=memory_system,
        reflection_engine=reflection_engine,
        world_orchestrator_url=world_orchestrator_url
    )
    await agent_runtime.initialize()

    logger.info("Agent Scheduler started successfully")

    yield

    # Cleanup
    await redis_client.close()
    await agent_runtime.shutdown()
    logger.info("Agent Scheduler stopped")


app = FastAPI(
    title="Agent Scheduler",
    description="Agent execution and scheduling service for Multi-Agent City",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.utcnow() - agent_runtime.start_time).total_seconds()

    services = {
        "redis": "operational" if await redis_client.ping() else "degraded",
        "memory_system": "operational" if memory_system.is_healthy() else "degraded",
        "agent_runtime": "operational" if agent_runtime.is_running else "stopped"
    }

    return HealthCheck(
        status="healthy" if all(s == "operational" for s in services.values()) else "degraded",
        version="1.0.0",
        uptime=uptime,
        services=services
    )


@app.post("/agents/{agent_id}/register", response_model=ApiResponse)
async def register_agent(agent_id: str, agent: Agent):
    """Register a new agent"""
    try:
        await agent_runtime.register_agent(agent)
        return ApiResponse(success=True, data={"agent_id": agent_id})
    except Exception as e:
        logger.error(f"Error registering agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/start", response_model=ApiResponse)
async def start_agent(agent_id: str):
    """Start an agent"""
    try:
        await agent_runtime.start_agent(agent_id)
        return ApiResponse(success=True, data={"status": "started"})
    except Exception as e:
        logger.error(f"Error starting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/stop", response_model=ApiResponse)
async def stop_agent(agent_id: str):
    """Stop an agent"""
    try:
        await agent_runtime.stop_agent(agent_id)
        return ApiResponse(success=True, data={"status": "stopped"})
    except Exception as e:
        logger.error(f"Error stopping agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/action", response_model=ActionResult)
async def execute_action(agent_id: str, action: Action):
    """Execute an action for an agent"""
    try:
        result = await agent_runtime.execute_action(agent_id, action)
        return result
    except Exception as e:
        logger.error(f"Error executing action for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}/observation")
async def get_observation(agent_id: str):
    """Get current observation for an agent"""
    try:
        observation = await agent_runtime.get_observation(agent_id)
        return observation
    except Exception as e:
        logger.error(f"Error getting observation for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/agents/{agent_id}/stream")
async def agent_stream(websocket: WebSocket, agent_id: str):
    """WebSocket stream for agent observations"""
    await websocket.accept()

    try:
        # Subscribe agent to observation stream
        await agent_runtime.subscribe_agent_stream(agent_id, websocket)

        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            # Handle incoming action from external agent
            try:
                import json
                action_data = json.loads(data)
                if action_data.get("type") == "action":
                    action = Action(**action_data["action"])
                    await agent_runtime.execute_action(agent_id, action)
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")

    except WebSocketDisconnect:
        logger.info(f"Agent {agent_id} WebSocket disconnected")
        await agent_runtime.unsubscribe_agent_stream(agent_id)
    except Exception as e:
        logger.error(f"WebSocket error for agent {agent_id}: {e}")


@app.get("/agents/{agent_id}/memory/search")
async def search_memories(agent_id: str, query: str, limit: int = 10):
    """Search agent memories"""
    try:
        memories = await memory_system.search_memories(agent_id, query, limit)
        return {"memories": memories}
    except Exception as e:
        logger.error(f"Error searching memories for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/memory/store", response_model=ApiResponse)
async def store_memory(agent_id: str, memory_data: dict):
    """Store a memory for an agent"""
    try:
        from packages.shared_types.models import Memory
        memory = Memory(agent_id=agent_id, **memory_data)
        await memory_system.store_memory(memory)
        return ApiResponse(success=True, data={"memory_id": str(memory.id)})
    except Exception as e:
        logger.error(f"Error storing memory for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/reflect", response_model=ApiResponse)
async def trigger_reflection(agent_id: str):
    """Trigger reflection for an agent"""
    try:
        reflection = await reflection_engine.generate_reflection(agent_id)
        return ApiResponse(success=True, data={"reflection": reflection})
    except Exception as e:
        logger.error(f"Error generating reflection for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents", response_model=List[str])
async def list_agents():
    """List all registered agents"""
    try:
        agents = await agent_runtime.list_agents()
        return agents
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get agent scheduler metrics"""
    try:
        metrics = await agent_runtime.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )