import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from packages.shared_types.models import (
    WorldState, Event, Agent, Location, District, Observation,
    VisibleAgent, Environment, AudibleMessage, EventType,
    Coordinates, LocationType
)

logger = logging.getLogger(__name__)


class WorldEngine:
    """Core world simulation engine"""

    def __init__(self, database, redis_client, mqtt_client):
        self.db = database
        self.redis = redis_client
        self.mqtt = mqtt_client

        # Simulation state
        self.current_tick = 0
        self.start_time = datetime.utcnow()
        self.is_running = False
        self.tick_rate = 60  # seconds per tick

        # World data
        self.agents: Dict[UUID, Agent] = {}
        self.districts: Dict[str, District] = {}
        self.events_queue = asyncio.Queue()

        # Economic system
        self.economy = None

        # Property manager
        self.property_manager = None

        # Governance system
        self.governance = None

        # Metrics
        self.metrics = {
            "total_agents": 0,
            "active_agents": 0,
            "total_events": 0,
            "messages_per_second": 0,
            "avg_response_time": 0
        }

    async def initialize(self):
        """Initialize the world engine"""
        logger.info("Initializing World Engine...")

        # Initialize economy system
        from economy_system import EconomySystem
        self.economy = EconomySystem(self.db, self.redis)
        await self.economy.initialize()

        # Initialize property manager
        from property_manager import PropertyManager
        self.property_manager = PropertyManager(self.db, self.redis, self.economy)
        await self.property_manager.initialize()

        # Initialize governance system
        from governance_system import GovernanceSystem
        self.governance = GovernanceSystem(self.db, self.redis, self.economy)
        await self.governance.initialize()

        # Create default districts
        await self._create_default_districts()

        # Load existing agents
        await self._load_agents()

        # Set initial world state
        await self._set_initial_world_state()

        logger.info("World Engine initialized successfully")

    async def _create_default_districts(self):
        """Create default districts and neighborhoods"""
        districts_data = [
            {
                "name": "Downtown",
                "neighborhoods": ["Financial District", "Arts Quarter", "Government Center"],
                "population": 0,
                "governance": {"council_size": 5, "voting_threshold": 0.6},
                "economy": {"base_rent": 100, "business_tax": 0.1}
            },
            {
                "name": "Market Square",
                "neighborhoods": ["Central Market", "Food Court", "Artisan Row"],
                "population": 0,
                "governance": {"council_size": 3, "voting_threshold": 0.5},
                "economy": {"base_rent": 75, "business_tax": 0.08}
            },
            {
                "name": "Tech Hub",
                "neighborhoods": ["Innovation Campus", "Startup District", "Research Park"],
                "population": 0,
                "governance": {"council_size": 7, "voting_threshold": 0.7},
                "economy": {"base_rent": 150, "business_tax": 0.05}
            },
            {
                "name": "Residential",
                "neighborhoods": ["Green Hills", "Riverside", "Old Town"],
                "population": 0,
                "governance": {"council_size": 4, "voting_threshold": 0.55},
                "economy": {"base_rent": 50, "business_tax": 0.12}
            }
        ]

        for district_data in districts_data:
            district = District(**district_data)
            self.districts[district.name] = district
            await self.db.save_district(district)

    async def _load_agents(self):
        """Load existing agents from database"""
        agents = await self.db.get_all_agents()
        for agent in agents:
            self.agents[agent.id] = agent
        logger.info(f"Loaded {len(self.agents)} agents from database")

    async def _set_initial_world_state(self):
        """Set initial world state"""
        await self.redis.set("world:tick", 0)
        await self.redis.set("world:time", datetime.utcnow().isoformat())
        await self.redis.set("world:weather", "sunny")

    async def run_simulation(self):
        """Main simulation loop"""
        self.is_running = True
        logger.info("Starting world simulation loop...")

        while self.is_running:
            try:
                tick_start = datetime.utcnow()

                # Process simulation tick
                await self.tick()

                # Process events
                await self._process_events()

                # Update metrics
                await self._update_metrics()

                # Rate limiting
                elapsed = (datetime.utcnow() - tick_start).total_seconds()
                sleep_time = max(0, self.tick_rate - elapsed)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def tick(self):
        """Execute one simulation tick"""
        self.current_tick += 1

        # Update world time
        world_time = datetime.utcnow()
        await self.redis.set("world:tick", self.current_tick)
        await self.redis.set("world:time", world_time.isoformat())

        # Update environment
        await self._update_environment()

        # Broadcast tick event
        await self._broadcast_tick()

        logger.debug(f"Tick {self.current_tick} completed")

    async def _update_environment(self):
        """Update environmental conditions"""
        # Simple weather simulation
        current_hour = datetime.utcnow().hour
        if 6 <= current_hour < 18:
            weather = "sunny" if current_hour < 14 else "partly_cloudy"
        else:
            weather = "clear_night"

        await self.redis.set("world:weather", weather)

        # Update time of day
        if 6 <= current_hour < 12:
            time_of_day = "morning"
        elif 12 <= current_hour < 17:
            time_of_day = "afternoon"
        elif 17 <= current_hour < 20:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        await self.redis.set("world:time_of_day", time_of_day)

    async def _broadcast_tick(self):
        """Broadcast tick event to all agents"""
        tick_event = {
            "type": "world_tick",
            "tick": self.current_tick,
            "timestamp": datetime.utcnow().isoformat(),
            "world_state": await self.get_world_state()
        }

        await self.mqtt.publish("world/tick", json.dumps(tick_event))

    async def _process_events(self):
        """Process queued events"""
        processed = 0
        while not self.events_queue.empty() and processed < 100:  # Limit processing per tick
            try:
                event = await asyncio.wait_for(self.events_queue.get(), timeout=0.1)
                await self.process_event(event)
                processed += 1
            except asyncio.TimeoutError:
                break

    async def process_event(self, event: Event):
        """Process a single event"""
        try:
            if event.type == EventType.AGENT_SPAWN:
                await self._handle_agent_spawn(event)
            elif event.type == EventType.AGENT_ACTION:
                await self._handle_agent_action(event)
            elif event.type == EventType.MESSAGE:
                await self._handle_message(event)
            elif event.type == EventType.TRANSACTION:
                await self._handle_transaction(event)

            # Mark event as processed
            event.processed = True
            await self.db.save_event(event)

            # Update metrics
            self.metrics["total_events"] += 1

        except Exception as e:
            logger.error(f"Error processing event {event.id}: {e}")

    async def _handle_agent_spawn(self, event: Event):
        """Handle agent spawn event"""
        agent_id = UUID(event.data["agent_id"])
        location_data = event.data["location"]
        location = Location(**location_data)

        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.state.location = location
            agent.state.status = "active"

            # Create economic account for agent
            if self.economy:
                await self.economy.create_agent_account(agent_id)

            # Broadcast spawn event
            spawn_message = {
                "type": "agent_spawned",
                "agent_id": str(agent_id),
                "name": agent.name,
                "location": location.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }

            await self.mqtt.publish(f"district/{location.district}/spawn", json.dumps(spawn_message))

    async def _handle_agent_action(self, event: Event):
        """Handle agent action event"""
        # Implementation for processing agent actions
        pass

    async def _handle_message(self, event: Event):
        """Handle message event"""
        message_data = event.data
        sender_id = UUID(message_data["sender_id"])
        content = message_data["content"]
        target = message_data.get("target", "broadcast")

        # Determine message scope
        if target == "broadcast":
            # Broadcast to district
            if event.district:
                topic = f"district/{event.district}/broadcast"
            else:
                topic = "world/broadcast"
        else:
            # Direct message
            topic = f"agent/{target}/direct"

        message = {
            "type": "message",
            "sender_id": str(sender_id),
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.mqtt.publish(topic, json.dumps(message))

    async def _handle_transaction(self, event: Event):
        """Handle transaction event"""
        if not self.economy:
            logger.warning("Economy system not initialized")
            return

        transaction_data = event.data
        sender_id = UUID(transaction_data["sender_id"])
        receiver_id = UUID(transaction_data["receiver_id"])
        amount = float(transaction_data["amount"])
        transaction_type = transaction_data.get("type", "payment")
        metadata = transaction_data.get("metadata", {})

        # Process the transaction
        transaction = await self.economy.process_transaction(
            sender_id, receiver_id, amount, transaction_type, metadata
        )

        if transaction:
            # Broadcast transaction completed
            await self.mqtt.publish_world_event(
                "transaction_completed",
                {
                    "transaction_id": str(transaction.id),
                    "sender": str(sender_id),
                    "receiver": str(receiver_id),
                    "amount": amount
                }
            )
        else:
            logger.warning(f"Transaction failed: {sender_id} -> {receiver_id}, {amount}")

    async def spawn_agent(self, agent_id: str, location: Location):
        """Spawn an agent at a location"""
        event = Event(
            type=EventType.AGENT_SPAWN,
            agent_id=UUID(agent_id),
            data={
                "agent_id": agent_id,
                "location": location.dict()
            },
            district=location.district
        )
        await self.events_queue.put(event)

    async def move_agent(self, agent_id: str, location: Location):
        """Move an agent to a new location"""
        agent_uuid = UUID(agent_id)
        if agent_uuid in self.agents:
            old_location = self.agents[agent_uuid].state.location
            self.agents[agent_uuid].state.location = location

            # Broadcast movement
            move_message = {
                "type": "agent_moved",
                "agent_id": agent_id,
                "from": old_location.dict(),
                "to": location.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }

            await self.mqtt.publish(f"district/{location.district}/movement", json.dumps(move_message))

    async def get_agent_observation(self, agent_id: str) -> Observation:
        """Generate observation for an agent"""
        agent_uuid = UUID(agent_id)
        if agent_uuid not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent = self.agents[agent_uuid]
        location = agent.state.location

        # Get visible agents
        visible_agents = await self._get_visible_agents(agent_uuid, location)

        # Get audible messages
        audible_messages = await self._get_audible_messages(agent_uuid, location)

        # Get environment
        environment = await self._get_environment(location)

        observation = Observation(
            agent_id=agent_uuid,
            tick=self.current_tick,
            location=location,
            visible_agents=visible_agents,
            audible_messages=audible_messages,
            environment=environment,
            available_actions=["speak", "move", "interact", "think"]
        )

        return observation

    async def _get_visible_agents(self, agent_id: UUID, location: Location) -> List[VisibleAgent]:
        """Get agents visible to the given agent"""
        visible = []
        for other_id, other_agent in self.agents.items():
            if other_id == agent_id or other_agent.state.status != "active":
                continue

            # Check if in same general area
            if (other_agent.state.location.district == location.district and
                other_agent.state.location.neighborhood == location.neighborhood):

                # Calculate distance
                distance = self._calculate_distance(location.coordinates, other_agent.state.location.coordinates)

                if distance <= 50.0:  # Visibility range
                    visible.append(VisibleAgent(
                        agent_id=other_id,
                        name=other_agent.name,
                        distance=distance,
                        activity=other_agent.state.activity.get("current", "idle")
                    ))

        return visible

    async def _get_audible_messages(self, agent_id: UUID, location: Location) -> List[AudibleMessage]:
        """Get messages audible to the agent"""
        # This would typically fetch recent messages from the message bus
        # For now, return empty list
        return []

    async def _get_environment(self, location: Location) -> Environment:
        """Get environmental conditions for a location"""
        weather = await self.redis.get("world:weather") or "sunny"
        time_of_day = await self.redis.get("world:time_of_day") or "day"

        # Simple crowd level simulation
        current_hour = datetime.utcnow().hour
        if 8 <= current_hour <= 10 or 17 <= current_hour <= 19:
            crowd_level = "busy"
        elif 11 <= current_hour <= 16:
            crowd_level = "moderate"
        else:
            crowd_level = "quiet"

        return Environment(
            time_of_day=time_of_day,
            weather=weather.decode() if isinstance(weather, bytes) else weather,
            temperature=72.0,  # Default temperature
            crowd_level=crowd_level
        )

    def _calculate_distance(self, coord1: Coordinates, coord2: Coordinates) -> float:
        """Calculate distance between two coordinates"""
        return ((coord1.x - coord2.x) ** 2 + (coord1.y - coord2.y) ** 2) ** 0.5

    async def get_world_state(self) -> WorldState:
        """Get current world state"""
        active_agents = sum(1 for agent in self.agents.values() if agent.state.status == "active")
        weather = await self.redis.get("world:weather") or "sunny"

        # Get economic metrics
        economy_data = {"total_transactions": 0, "gdp": 0}
        if self.economy:
            economy_metrics = await self.economy.get_economic_metrics()
            economy_data = {
                "total_transactions": economy_metrics["total_transactions"],
                "gdp": economy_metrics["gdp"],
                "average_balance": economy_metrics["average_balance"],
                "total_supply": economy_metrics["total_supply"]
            }

        return WorldState(
            tick=self.current_tick,
            time=datetime.utcnow(),
            active_agents=active_agents,
            total_events=self.metrics["total_events"],
            weather=weather.decode() if isinstance(weather, bytes) else weather,
            economy=economy_data
        )

    async def get_active_agents(self) -> List[Agent]:
        """Get all active agents"""
        return [agent for agent in self.agents.values() if agent.state.status == "active"]

    async def get_districts(self) -> List[District]:
        """Get all districts"""
        return list(self.districts.values())

    async def _update_metrics(self):
        """Update system metrics"""
        self.metrics["total_agents"] = len(self.agents)
        self.metrics["active_agents"] = sum(1 for agent in self.agents.values() if agent.state.status == "active")

        # Store metrics in Redis for monitoring
        await self.redis.hset("metrics:world", mapping=self.metrics)

    async def get_metrics(self) -> Dict:
        """Get current metrics"""
        return self.metrics.copy()

    def stop(self):
        """Stop the simulation"""
        self.is_running = False