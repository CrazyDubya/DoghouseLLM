import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import redis.asyncio as redis
import paho.mqtt.client as mqtt

from packages.shared_types.models import (
    WorldState, Event, Agent, Location, District,
    EventType, ApiResponse, HealthCheck
)
from world_engine import WorldEngine
from database import Database
from mqtt_client import MQTTClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
world_engine: WorldEngine = None
db: Database = None
mqtt_client: MQTTClient = None
redis_client: redis.Redis = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global world_engine, db, mqtt_client, redis_client

    # Initialize database
    database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/multiagent_city")
    db = Database(database_url)
    await db.initialize()

    # Initialize Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)

    # Initialize MQTT
    mqtt_host = os.getenv("MQTT_HOST", "localhost")
    mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
    mqtt_client = MQTTClient(mqtt_host, mqtt_port)
    await mqtt_client.connect()

    # Initialize World Engine
    world_engine = WorldEngine(db, redis_client, mqtt_client)
    await world_engine.initialize()

    # Start simulation loop
    simulation_task = asyncio.create_task(world_engine.run_simulation())

    logger.info("World Orchestrator started successfully")

    yield

    # Cleanup
    simulation_task.cancel()
    await mqtt_client.disconnect()
    await redis_client.close()
    logger.info("World Orchestrator stopped")


app = FastAPI(
    title="World Orchestrator",
    description="Core world simulation engine for Multi-Agent City",
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
    uptime = (datetime.utcnow() - world_engine.start_time).total_seconds()

    services = {
        "database": "operational" if await db.is_healthy() else "degraded",
        "redis": "operational" if await redis_client.ping() else "degraded",
        "mqtt": "operational" if mqtt_client.is_connected() else "degraded",
        "world_engine": "operational" if world_engine.is_running else "stopped"
    }

    return HealthCheck(
        status="healthy" if all(s == "operational" for s in services.values()) else "degraded",
        version="1.0.0",
        uptime=uptime,
        services=services
    )


@app.get("/world/state", response_model=WorldState)
async def get_world_state():
    """Get current world state"""
    try:
        state = await world_engine.get_world_state()
        return state
    except Exception as e:
        logger.error(f"Error getting world state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/world/agents", response_model=List[Agent])
async def get_active_agents():
    """Get all active agents in the world"""
    try:
        agents = await world_engine.get_active_agents()
        return agents
    except Exception as e:
        logger.error(f"Error getting active agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/world/districts", response_model=List[District])
async def get_districts():
    """Get all districts in the world"""
    try:
        districts = await world_engine.get_districts()
        return districts
    except Exception as e:
        logger.error(f"Error getting districts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/world/events", response_model=ApiResponse)
async def create_event(event: Event):
    """Create a new world event"""
    try:
        await world_engine.process_event(event)
        return ApiResponse(success=True, data={"event_id": str(event.id)})
    except Exception as e:
        logger.error(f"Error creating event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/spawn", response_model=ApiResponse)
async def spawn_agent(agent_id: str, location: Location):
    """Spawn an agent at a specific location"""
    try:
        await world_engine.spawn_agent(agent_id, location)
        return ApiResponse(success=True, data={"agent_id": agent_id, "location": location.dict()})
    except Exception as e:
        logger.error(f"Error spawning agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/move", response_model=ApiResponse)
async def move_agent(agent_id: str, location: Location):
    """Move an agent to a new location"""
    try:
        await world_engine.move_agent(agent_id, location)
        return ApiResponse(success=True, data={"agent_id": agent_id, "new_location": location.dict()})
    except Exception as e:
        logger.error(f"Error moving agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}/observation")
async def get_agent_observation(agent_id: str):
    """Get current observation for an agent"""
    try:
        observation = await world_engine.get_agent_observation(agent_id)
        return observation
    except Exception as e:
        logger.error(f"Error getting observation for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/world/tick", response_model=ApiResponse)
async def force_tick():
    """Force a world simulation tick (for testing)"""
    try:
        await world_engine.tick()
        return ApiResponse(success=True, data={"tick": world_engine.current_tick})
    except Exception as e:
        logger.error(f"Error forcing tick: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get world simulation metrics"""
    try:
        metrics = await world_engine.get_metrics()
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