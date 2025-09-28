"""Web Dashboard for Multi-Agent City Platform"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
WORLD_ORCHESTRATOR_URL = os.getenv("WORLD_ORCHESTRATOR_URL", "http://world-orchestrator:8001")
AGENT_SCHEDULER_URL = os.getenv("AGENT_SCHEDULER_URL", "http://agent-scheduler:8002")
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")

app = FastAPI(
    title="Multi-Agent City Dashboard",
    description="Web dashboard for monitoring and controlling the multi-agent city",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connections for real-time updates
connected_clients = set()


@app.get("/")
async def root():
    """Serve the main dashboard page"""
    return FileResponse("templates/index.html")


@app.get("/world")
async def world_view():
    """Serve the world visualization page"""
    return FileResponse("templates/world.html")


@app.get("/agents")
async def agents_view():
    """Serve the agents management page"""
    return FileResponse("templates/agents.html")


@app.get("/economy")
async def economy_view():
    """Serve the economy dashboard page"""
    return FileResponse("templates/economy.html")


@app.get("/governance")
async def governance_view():
    """Serve the governance page"""
    return FileResponse("templates/governance.html")


@app.get("/interactions")
async def interactions_view():
    """Serve the interactions visualization page"""
    return FileResponse("templates/interactions.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.add(websocket)

    try:
        # Start sending periodic updates
        while True:
            # Fetch current state from services
            try:
                data = await fetch_dashboard_data()
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Error fetching dashboard data: {e}")

            await asyncio.sleep(1)  # Update every second

    except WebSocketDisconnect:
        connected_clients.remove(websocket)


async def fetch_dashboard_data():
    """Fetch current data from all services"""
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "world": {},
        "agents": {},
        "economy": {},
        "governance": {},
        "interactions": {}
    }

    async with httpx.AsyncClient() as client:
        try:
            # Fetch world state
            response = await client.get(f"{WORLD_ORCHESTRATOR_URL}/world/state")
            if response.status_code == 200:
                data["world"] = response.json()
        except Exception as e:
            logger.error(f"Error fetching world state: {e}")

        try:
            # Fetch active agents
            response = await client.get(f"{WORLD_ORCHESTRATOR_URL}/world/agents")
            if response.status_code == 200:
                data["agents"]["active"] = response.json()
        except Exception as e:
            logger.error(f"Error fetching agents: {e}")

        try:
            # Fetch economy metrics
            response = await client.get(f"{WORLD_ORCHESTRATOR_URL}/economy/metrics")
            if response.status_code == 200:
                data["economy"] = response.json()
        except Exception as e:
            logger.error(f"Error fetching economy metrics: {e}")

        try:
            # Fetch governance metrics
            response = await client.get(f"{WORLD_ORCHESTRATOR_URL}/governance/metrics")
            if response.status_code == 200:
                data["governance"] = response.json()

            # Fetch active proposals
            response = await client.get(f"{WORLD_ORCHESTRATOR_URL}/governance/proposals")
            if response.status_code == 200:
                data["governance"]["proposals"] = response.json().get("proposals", [])
        except Exception as e:
            logger.error(f"Error fetching governance data: {e}")

        try:
            # Fetch interaction metrics
            response = await client.get(f"{AGENT_SCHEDULER_URL}/interactions/metrics")
            if response.status_code == 200:
                data["interactions"] = response.json()
        except Exception as e:
            logger.error(f"Error fetching interaction metrics: {e}")

    return data


@app.get("/api/world/districts")
async def get_districts():
    """Proxy endpoint for districts"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{WORLD_ORCHESTRATOR_URL}/world/districts")
        return response.json()


@app.get("/api/agents/list")
async def get_agents_list():
    """Get list of all agents with details"""
    async with httpx.AsyncClient() as client:
        agents = []
        try:
            # Get agent list
            response = await client.get(f"{AGENT_SCHEDULER_URL}/agents")
            if response.status_code == 200:
                agent_ids = response.json()

                # Get details for each agent
                for agent_id in agent_ids[:20]:  # Limit to 20 for performance
                    try:
                        # Get balance
                        balance_response = await client.get(f"{WORLD_ORCHESTRATOR_URL}/economy/balance/{agent_id}")
                        balance_data = balance_response.json() if balance_response.status_code == 200 else {}

                        # Get reputation
                        rep_response = await client.get(f"{WORLD_ORCHESTRATOR_URL}/governance/agents/{agent_id}/reputation")
                        rep_data = rep_response.json() if rep_response.status_code == 200 else {}

                        agents.append({
                            "id": agent_id,
                            "balance": balance_data.get("balance", 0),
                            "reputation": rep_data.get("reputation", 0),
                            "status": "active"
                        })
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error getting agents list: {e}")

        return agents


@app.get("/api/economy/transactions/recent")
async def get_recent_transactions():
    """Get recent transactions"""
    # This would need to be implemented in the world orchestrator
    return []


@app.get("/api/interactions/active")
async def get_active_interactions():
    """Get all active interactions"""
    async with httpx.AsyncClient() as client:
        try:
            # This would need to aggregate active interactions
            return []
        except Exception as e:
            logger.error(f"Error getting active interactions: {e}")
            return []


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )