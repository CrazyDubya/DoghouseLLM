import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

import redis.asyncio as redis
import httpx
from fastapi import WebSocket

from packages.shared_types.models import (
    Agent, Action, ActionResult, Observation, AgentStatus
)

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Agent execution runtime and scheduler"""

    def __init__(self, redis_client, memory_system, reflection_engine, world_orchestrator_url):
        self.redis = redis_client
        self.memory_system = memory_system
        self.reflection_engine = reflection_engine
        self.world_orchestrator_url = world_orchestrator_url

        # Runtime state
        self.agents: Dict[str, Agent] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.agent_streams: Dict[str, WebSocket] = {}
        self.start_time = datetime.utcnow()
        self.is_running = False

        # Metrics
        self.metrics = {
            "total_agents": 0,
            "active_agents": 0,
            "actions_executed": 0,
            "avg_response_time": 0,
            "error_count": 0
        }

    async def initialize(self):
        """Initialize the agent runtime"""
        self.is_running = True

        # Start background tasks
        asyncio.create_task(self._metrics_updater())
        asyncio.create_task(self._agent_monitor())

        logger.info("Agent Runtime initialized successfully")

    async def register_agent(self, agent: Agent):
        """Register a new agent"""
        agent_id = str(agent.id)
        self.agents[agent_id] = agent

        # Store agent in Redis for persistence
        await self.redis.hset(
            f"agent:{agent_id}",
            mapping={
                "name": agent.name,
                "profile": json.dumps(agent.profile.dict()),
                "state": json.dumps(agent.state.dict()),
                "model_config": json.dumps(agent.model_config),
                "external_endpoint": agent.external_endpoint or "",
                "status": agent.state.status
            }
        )

        self.metrics["total_agents"] += 1
        logger.info(f"Registered agent {agent.name} ({agent_id})")

    async def start_agent(self, agent_id: str):
        """Start an agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        agent.state.status = AgentStatus.ACTIVE

        # Start agent execution loop
        if agent_id not in self.agent_tasks:
            task = asyncio.create_task(self._agent_execution_loop(agent_id))
            self.agent_tasks[agent_id] = task

        # Update status in Redis
        await self.redis.hset(f"agent:{agent_id}", "status", AgentStatus.ACTIVE)

        self.metrics["active_agents"] += 1
        logger.info(f"Started agent {agent.name} ({agent_id})")

    async def stop_agent(self, agent_id: str):
        """Stop an agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        agent.state.status = AgentStatus.PAUSED

        # Cancel agent execution task
        if agent_id in self.agent_tasks:
            self.agent_tasks[agent_id].cancel()
            del self.agent_tasks[agent_id]

        # Update status in Redis
        await self.redis.hset(f"agent:{agent_id}", "status", AgentStatus.PAUSED)

        self.metrics["active_agents"] = max(0, self.metrics["active_agents"] - 1)
        logger.info(f"Stopped agent {agent.name} ({agent_id})")

    async def _agent_execution_loop(self, agent_id: str):
        """Main execution loop for an agent"""
        try:
            while self.agents[agent_id].state.status == AgentStatus.ACTIVE:
                # Get current observation
                observation = await self.get_observation(agent_id)

                # Let agent decide on action
                action = await self._agent_decide_action(agent_id, observation)

                if action:
                    # Execute the action
                    await self.execute_action(agent_id, action)

                # Sleep before next iteration
                await asyncio.sleep(5)  # 5 second intervals

        except asyncio.CancelledError:
            logger.info(f"Agent {agent_id} execution loop cancelled")
        except Exception as e:
            logger.error(f"Error in agent {agent_id} execution loop: {e}")
            # Mark agent as offline
            if agent_id in self.agents:
                self.agents[agent_id].state.status = AgentStatus.OFFLINE

    async def _agent_decide_action(self, agent_id: str, observation: Observation) -> Optional[Action]:
        """Let agent decide on an action based on observation"""
        agent = self.agents[agent_id]

        try:
            if agent.external_endpoint:
                # External agent - send observation via webhook
                return await self._call_external_agent(agent, observation)
            else:
                # Hosted agent - use internal LLM
                return await self._call_hosted_agent(agent, observation)

        except Exception as e:
            logger.error(f"Error getting action from agent {agent_id}: {e}")
            self.metrics["error_count"] += 1
            return None

    async def _call_external_agent(self, agent: Agent, observation: Observation) -> Optional[Action]:
        """Call external agent via webhook"""
        if not agent.external_endpoint:
            return None

        payload = {
            "agent_id": str(agent.id),
            "observation": observation.dict(),
            "request_id": f"req_{datetime.utcnow().timestamp()}"
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    agent.external_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    data = response.json()
                    action_data = data.get("action")
                    if action_data:
                        return Action(
                            agent_id=agent.id,
                            **action_data
                        )

        except Exception as e:
            logger.error(f"Error calling external agent {agent.id}: {e}")

        return None

    async def _call_hosted_agent(self, agent: Agent, observation: Observation) -> Optional[Action]:
        """Call hosted agent using internal LLM"""
        # Simple rule-based behavior for demo
        # In production, this would use LangChain with the agent's model config

        from packages.shared_types.models import ActionType, ActionParameters

        # Get recent memories for context
        memories = await self.memory_system.get_recent_memories(str(agent.id), limit=5)

        # Simple decision logic based on observation
        if observation.audible_messages:
            # Respond to messages
            for message in observation.audible_messages:
                if message.speaker_id != agent.id:
                    return Action(
                        agent_id=agent.id,
                        type=ActionType.SPEAK,
                        parameters=ActionParameters(
                            message=f"Hello! I heard you say: {message.message}",
                            target=str(message.speaker_id)
                        ),
                        reasoning="Responding to audible message"
                    )

        # Random movement if no visible agents
        if len(observation.visible_agents) == 0:
            import random
            if random.random() < 0.3:  # 30% chance to move
                return Action(
                    agent_id=agent.id,
                    type=ActionType.MOVE,
                    parameters=ActionParameters(
                        destination=observation.location
                    ),
                    reasoning="Exploring the area"
                )

        # Think action as default
        return Action(
            agent_id=agent.id,
            type=ActionType.THINK,
            parameters=ActionParameters(),
            reasoning="Observing the environment"
        )

    async def execute_action(self, agent_id: str, action: Action) -> ActionResult:
        """Execute an action for an agent"""
        start_time = datetime.utcnow()

        try:
            # Validate action
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")

            # Store action as memory
            await self._store_action_memory(agent_id, action)

            # Execute action based on type
            if action.type == ActionType.SPEAK:
                result = await self._execute_speak_action(agent_id, action)
            elif action.type == ActionType.MOVE:
                result = await self._execute_move_action(agent_id, action)
            elif action.type == ActionType.INTERACT:
                result = await self._execute_interact_action(agent_id, action)
            elif action.type == ActionType.THINK:
                result = await self._execute_think_action(agent_id, action)
            else:
                result = ActionResult(
                    action_id=action.id,
                    status="unsupported",
                    effects={},
                    error=f"Action type {action.type} not supported"
                )

            # Update metrics
            self.metrics["actions_executed"] += 1
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_avg_response_time(response_time)

            return result

        except Exception as e:
            logger.error(f"Error executing action {action.id}: {e}")
            return ActionResult(
                action_id=action.id,
                status="error",
                effects={},
                error=str(e)
            )

    async def _execute_speak_action(self, agent_id: str, action: Action) -> ActionResult:
        """Execute a speak action"""
        agent = self.agents[agent_id]
        message = action.parameters.message
        target = action.parameters.target

        # Send message via world orchestrator
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.world_orchestrator_url}/world/events",
                json={
                    "type": "message",
                    "agent_id": agent_id,
                    "data": {
                        "sender_id": agent_id,
                        "content": message,
                        "target": target
                    },
                    "district": agent.state.location.district
                }
            )

        return ActionResult(
            action_id=action.id,
            status="completed",
            effects={"message_sent": True, "target": target}
        )

    async def _execute_move_action(self, agent_id: str, action: Action) -> ActionResult:
        """Execute a move action"""
        destination = action.parameters.destination

        if destination:
            # Send move request to world orchestrator
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.world_orchestrator_url}/agents/{agent_id}/move",
                    json=destination.dict()
                )

            return ActionResult(
                action_id=action.id,
                status="completed",
                effects={"new_location": destination.dict()}
            )

        return ActionResult(
            action_id=action.id,
            status="failed",
            effects={},
            error="No destination specified"
        )

    async def _execute_interact_action(self, agent_id: str, action: Action) -> ActionResult:
        """Execute an interact action"""
        # Simple interaction logic
        return ActionResult(
            action_id=action.id,
            status="completed",
            effects={"interaction": "completed"}
        )

    async def _execute_think_action(self, agent_id: str, action: Action) -> ActionResult:
        """Execute a think action"""
        # Trigger reflection if enough memories accumulated
        memory_count = await self.memory_system.get_memory_count(agent_id)

        if memory_count > 0 and memory_count % 20 == 0:  # Reflect every 20 memories
            await self.reflection_engine.generate_reflection(agent_id)

        return ActionResult(
            action_id=action.id,
            status="completed",
            effects={"thought": "processed"}
        )

    async def _store_action_memory(self, agent_id: str, action: Action):
        """Store action as a memory"""
        from packages.shared_types.models import Memory, MemoryType

        memory = Memory(
            agent_id=UUID(agent_id),
            type=MemoryType.OBSERVATION,
            content=f"I performed action: {action.type} - {action.reasoning}",
            importance=0.5
        )

        await self.memory_system.store_memory(memory)

    async def get_observation(self, agent_id: str) -> Observation:
        """Get current observation for an agent"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.world_orchestrator_url}/agents/{agent_id}/observation"
            )

            if response.status_code == 200:
                data = response.json()
                return Observation(**data)
            else:
                raise Exception(f"Failed to get observation: {response.status_code}")

    async def subscribe_agent_stream(self, agent_id: str, websocket: WebSocket):
        """Subscribe agent to observation stream"""
        self.agent_streams[agent_id] = websocket

    async def unsubscribe_agent_stream(self, agent_id: str):
        """Unsubscribe agent from observation stream"""
        if agent_id in self.agent_streams:
            del self.agent_streams[agent_id]

    async def _metrics_updater(self):
        """Background task to update metrics"""
        while self.is_running:
            try:
                # Update Redis metrics
                await self.redis.hset("metrics:agent_scheduler", mapping=self.metrics)
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")

    async def _agent_monitor(self):
        """Background task to monitor agent health"""
        while self.is_running:
            try:
                # Check for unresponsive agents
                for agent_id, agent in self.agents.items():
                    if agent.state.status == AgentStatus.ACTIVE:
                        # Implement health check logic
                        pass

                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in agent monitor: {e}")

    def _update_avg_response_time(self, response_time: float):
        """Update average response time metric"""
        current_avg = self.metrics["avg_response_time"]
        count = self.metrics["actions_executed"]

        if count == 1:
            self.metrics["avg_response_time"] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics["avg_response_time"] = alpha * response_time + (1 - alpha) * current_avg

    async def list_agents(self) -> List[str]:
        """List all registered agents"""
        return list(self.agents.keys())

    async def get_metrics(self) -> Dict:
        """Get current metrics"""
        return self.metrics.copy()

    async def shutdown(self):
        """Shutdown the agent runtime"""
        self.is_running = False

        # Cancel all agent tasks
        for task in self.agent_tasks.values():
            task.cancel()

        # Close all WebSocket connections
        for websocket in self.agent_streams.values():
            try:
                await websocket.close()
            except:
                pass

        logger.info("Agent Runtime shutdown complete")