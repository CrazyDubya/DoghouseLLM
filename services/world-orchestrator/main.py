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
from governance_system import ProposalType, VoteChoice
from uuid import UUID
from world_engine import WorldEngine
from database import Database
from mqtt_client import MQTTClient
from metrics import metrics_collector
from fastapi.responses import Response

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


@app.post("/economy/transaction")
async def create_transaction(
    sender_id: str,
    receiver_id: str,
    amount: float,
    transaction_type: str = "payment",
    metadata: dict = None
):
    """Create an economic transaction between agents"""
    try:
        if not world_engine.economy:
            raise HTTPException(status_code=503, detail="Economy system not available")

        transaction = await world_engine.economy.process_transaction(
            UUID(sender_id),
            UUID(receiver_id),
            amount,
            transaction_type,
            metadata or {}
        )

        if transaction:
            return ApiResponse(success=True, data={"transaction": transaction.dict()})
        else:
            raise HTTPException(status_code=400, detail="Transaction failed")

    except Exception as e:
        logger.error(f"Error processing transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/economy/balance/{agent_id}")
async def get_agent_balance(agent_id: str):
    """Get an agent's economic balance"""
    try:
        if not world_engine.economy:
            raise HTTPException(status_code=503, detail="Economy system not available")

        balance = await world_engine.economy.get_balance(UUID(agent_id))
        wealth = await world_engine.economy.calculate_agent_wealth(UUID(agent_id))

        return {
            "agent_id": agent_id,
            "balance": balance,
            "wealth": wealth,
            "currency": world_engine.economy.currency_name
        }

    except Exception as e:
        logger.error(f"Error getting agent balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/economy/metrics")
async def get_economy_metrics():
    """Get economic system metrics"""
    try:
        if not world_engine.economy:
            raise HTTPException(status_code=503, detail="Economy system not available")

        metrics = await world_engine.economy.get_economic_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Error getting economy metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get world simulation metrics in JSON format"""
    try:
        metrics = await world_engine.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/proposals")
async def create_proposal(
    proposer_id: str,
    proposal_type: str,
    title: str,
    description: str,
    metadata: dict = None,
    voting_duration_hours: int = 24
):
    """Create a new governance proposal"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        proposal = await world_engine.governance.create_proposal(
            proposer_id=UUID(proposer_id),
            proposal_type=ProposalType(proposal_type),
            title=title,
            description=description,
            metadata=metadata or {},
            voting_duration_hours=voting_duration_hours
        )

        if proposal:
            return ApiResponse(success=True, data={"proposal": proposal.to_dict()})
        else:
            raise HTTPException(status_code=400, detail="Could not create proposal")

    except Exception as e:
        logger.error(f"Error creating proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/proposals/{proposal_id}/start-voting")
async def start_voting(proposal_id: str):
    """Start voting on a proposal"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        success = await world_engine.governance.start_voting(UUID(proposal_id))

        if success:
            return ApiResponse(success=True, data={"status": "voting_started"})
        else:
            raise HTTPException(status_code=400, detail="Could not start voting")

    except Exception as e:
        logger.error(f"Error starting voting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/proposals/{proposal_id}/vote")
async def cast_vote(
    proposal_id: str,
    voter_id: str,
    vote: str,
    comment: str = None
):
    """Cast a vote on a proposal"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        success = await world_engine.governance.cast_vote(
            proposal_id=UUID(proposal_id),
            voter_id=UUID(voter_id),
            vote=VoteChoice(vote),
            comment=comment
        )

        if success:
            return ApiResponse(success=True, data={"status": "vote_cast"})
        else:
            raise HTTPException(status_code=400, detail="Could not cast vote")

    except Exception as e:
        logger.error(f"Error casting vote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/proposals")
async def get_active_proposals():
    """Get all active proposals"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        proposals = await world_engine.governance.get_active_proposals()
        return {
            "proposals": [p.to_dict() for p in proposals]
        }

    except Exception as e:
        logger.error(f"Error getting active proposals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/proposals/{proposal_id}")
async def get_proposal(proposal_id: str):
    """Get a specific proposal"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        proposal = await world_engine.governance.get_proposal(UUID(proposal_id))

        if proposal:
            return proposal.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Proposal not found")

    except Exception as e:
        logger.error(f"Error getting proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/proposals/{proposal_id}/results")
async def get_voting_results(proposal_id: str):
    """Get voting results for a proposal"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        results = await world_engine.governance.tally_votes(UUID(proposal_id))
        return results

    except Exception as e:
        logger.error(f"Error getting voting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/agents/{agent_id}/reputation")
async def get_agent_reputation(agent_id: str):
    """Get an agent's reputation score"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        reputation = await world_engine.governance.get_reputation(UUID(agent_id))
        return {
            "agent_id": agent_id,
            "reputation": reputation,
            "can_propose": reputation >= world_engine.governance.min_reputation_to_propose,
            "can_vote": reputation >= world_engine.governance.min_reputation_to_vote
        }

    except Exception as e:
        logger.error(f"Error getting agent reputation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/agents/{agent_id}/voting-history")
async def get_agent_voting_history(agent_id: str):
    """Get voting history for an agent"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        history = await world_engine.governance.get_agent_voting_history(UUID(agent_id))
        return {
            "agent_id": agent_id,
            "voting_history": history
        }

    except Exception as e:
        logger.error(f"Error getting voting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/council/{agent_id}/add")
async def add_council_member(agent_id: str):
    """Add an agent to the council (admin only)"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        await world_engine.governance.add_council_member(UUID(agent_id))
        return ApiResponse(success=True, data={"status": "added_to_council"})

    except Exception as e:
        logger.error(f"Error adding council member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/council/{agent_id}/remove")
async def remove_council_member(agent_id: str):
    """Remove an agent from the council (admin only)"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        await world_engine.governance.remove_council_member(UUID(agent_id))
        return ApiResponse(success=True, data={"status": "removed_from_council"})

    except Exception as e:
        logger.error(f"Error removing council member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/metrics")
async def get_governance_metrics():
    """Get governance system metrics"""
    try:
        if not world_engine.governance:
            raise HTTPException(status_code=503, detail="Governance system not available")

        metrics = await world_engine.governance.get_governance_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Error getting governance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    try:
        # Update metrics from current state
        world_state = await world_engine.get_world_state()
        metrics_collector.update_world_metrics(world_state.dict())

        # Update economy metrics if available
        if world_engine.economy:
            economy_metrics = await world_engine.economy.get_economic_metrics()
            metrics_collector.update_economy_metrics(economy_metrics)

        # Update property metrics if available
        if world_engine.property_manager:
            property_metrics = await world_engine.property_manager.get_property_metrics()
            metrics_collector.update_property_metrics(property_metrics)

        # Get Prometheus formatted metrics
        metrics_data = metrics_collector.get_metrics()

        return Response(content=metrics_data, media_type="text/plain")
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )