"""Agent interaction system for communication and collaboration"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4
from enum import Enum

from packages.shared_types.models import Agent, Memory, MemoryType

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of agent interactions"""
    CONVERSATION = "conversation"
    COLLABORATION = "collaboration"
    NEGOTIATION = "negotiation"
    TEACHING = "teaching"
    TRADING = "trading"
    CONFLICT = "conflict"
    GREETING = "greeting"
    FAREWELL = "farewell"


class MessageType(Enum):
    """Types of messages between agents"""
    TEXT = "text"
    EMOTION = "emotion"
    ACTION = "action"
    PROPOSAL = "proposal"
    RESPONSE = "response"
    QUESTION = "question"
    STATEMENT = "statement"


class Interaction:
    """Represents an interaction between agents"""

    def __init__(
        self,
        id: UUID,
        type: InteractionType,
        initiator_id: UUID,
        participants: List[UUID],
        location: Dict = None,
        context: Dict = None
    ):
        self.id = id
        self.type = type
        self.initiator_id = initiator_id
        self.participants = participants
        self.location = location or {}
        self.context = context or {}
        self.started_at = datetime.utcnow()
        self.ended_at = None
        self.messages: List[Dict] = []
        self.outcomes: Dict = {}
        self.is_active = True

    def add_message(self, sender_id: UUID, content: str, message_type: MessageType = MessageType.TEXT):
        """Add a message to the interaction"""
        self.messages.append({
            "id": str(uuid4()),
            "sender_id": str(sender_id),
            "content": content,
            "type": message_type.value,
            "timestamp": datetime.utcnow().isoformat()
        })

    def end_interaction(self, outcomes: Dict = None):
        """End the interaction"""
        self.ended_at = datetime.utcnow()
        self.is_active = False
        if outcomes:
            self.outcomes = outcomes

    def to_dict(self) -> Dict:
        """Convert interaction to dictionary"""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "initiator_id": str(self.initiator_id),
            "participants": [str(p) for p in self.participants],
            "location": self.location,
            "context": self.context,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": (self.ended_at - self.started_at).total_seconds() if self.ended_at else None,
            "message_count": len(self.messages),
            "is_active": self.is_active,
            "outcomes": self.outcomes
        }


class InteractionSystem:
    """System for managing agent interactions"""

    def __init__(self, memory_system, llm_integration, redis_client):
        self.memory_system = memory_system
        self.llm = llm_integration
        self.redis = redis_client

        # Active interactions
        self.active_interactions: Dict[UUID, Interaction] = {}

        # Agent interaction state
        self.agent_interactions: Dict[UUID, Set[UUID]] = {}  # agent_id -> set of interaction_ids

        # Interaction history
        self.interaction_history: List[Interaction] = []

        # Configuration
        self.max_concurrent_interactions = 3
        self.interaction_timeout = 600  # 10 minutes
        self.message_cooldown = 2  # seconds between messages

    async def initialize(self):
        """Initialize the interaction system"""
        logger.info("Initializing interaction system...")

        # Start background tasks
        asyncio.create_task(self._interaction_monitor())

        logger.info("Interaction system initialized")

    async def initiate_interaction(
        self,
        initiator_id: UUID,
        target_ids: List[UUID],
        interaction_type: InteractionType,
        initial_message: str = None,
        location: Dict = None,
        context: Dict = None
    ) -> Optional[Interaction]:
        """Initiate an interaction between agents"""
        try:
            # Check if initiator can start new interaction
            current_interactions = self.agent_interactions.get(initiator_id, set())
            if len(current_interactions) >= self.max_concurrent_interactions:
                logger.warning(f"Agent {initiator_id} has too many concurrent interactions")
                return None

            # Create interaction
            interaction = Interaction(
                id=uuid4(),
                type=interaction_type,
                initiator_id=initiator_id,
                participants=[initiator_id] + target_ids,
                location=location,
                context=context
            )

            # Add initial message if provided
            if initial_message:
                interaction.add_message(initiator_id, initial_message, MessageType.TEXT)

            # Register interaction for all participants
            for participant_id in interaction.participants:
                if participant_id not in self.agent_interactions:
                    self.agent_interactions[participant_id] = set()
                self.agent_interactions[participant_id].add(interaction.id)

            # Store active interaction
            self.active_interactions[interaction.id] = interaction

            # Save to Redis for persistence
            await self._save_interaction(interaction)

            # Create memories for participants
            await self._create_interaction_memory(interaction, "interaction_started")

            logger.info(f"Interaction {interaction.id} initiated between {len(interaction.participants)} agents")
            return interaction

        except Exception as e:
            logger.error(f"Error initiating interaction: {e}")
            return None

    async def send_message(
        self,
        interaction_id: UUID,
        sender_id: UUID,
        message: str,
        message_type: MessageType = MessageType.TEXT
    ) -> bool:
        """Send a message in an interaction"""
        try:
            interaction = self.active_interactions.get(interaction_id)

            if not interaction:
                logger.warning(f"Interaction {interaction_id} not found")
                return False

            if not interaction.is_active:
                logger.warning(f"Interaction {interaction_id} is not active")
                return False

            if sender_id not in interaction.participants:
                logger.warning(f"Agent {sender_id} is not a participant in interaction {interaction_id}")
                return False

            # Add message to interaction
            interaction.add_message(sender_id, message, message_type)

            # Save updated interaction
            await self._save_interaction(interaction)

            # Process message with LLM for other participants
            await self._process_message(interaction, sender_id, message)

            # Create memory for sender
            await self._create_message_memory(sender_id, interaction, message, "sent")

            return True

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    async def generate_response(
        self,
        agent_id: UUID,
        interaction_id: UUID,
        context_messages: List[Dict] = None
    ) -> Optional[str]:
        """Generate a response for an agent in an interaction"""
        try:
            interaction = self.active_interactions.get(interaction_id)

            if not interaction:
                return None

            # Get agent's recent memories
            memories = await self.memory_system.search_memories(
                str(agent_id),
                f"interaction with agents",
                limit=5
            )

            # Get recent messages for context
            recent_messages = interaction.messages[-5:] if len(interaction.messages) > 5 else interaction.messages

            # Format context for LLM
            context = {
                "interaction_type": interaction.type.value,
                "location": interaction.location,
                "participants": [str(p) for p in interaction.participants],
                "recent_messages": recent_messages,
                "memories": [m.content for m in memories] if memories else []
            }

            # Generate response using LLM
            prompt = self._build_response_prompt(agent_id, interaction, context)
            response = await self.llm.generate_text(prompt, context)

            if response:
                # Add response to interaction
                interaction.add_message(agent_id, response, MessageType.RESPONSE)
                await self._save_interaction(interaction)

                # Create memory for responder
                await self._create_message_memory(agent_id, interaction, response, "sent")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None

    async def end_interaction(
        self,
        interaction_id: UUID,
        ender_id: UUID,
        reason: str = None,
        outcomes: Dict = None
    ) -> bool:
        """End an interaction"""
        try:
            interaction = self.active_interactions.get(interaction_id)

            if not interaction:
                logger.warning(f"Interaction {interaction_id} not found")
                return False

            if not interaction.is_active:
                logger.warning(f"Interaction {interaction_id} is already ended")
                return False

            # End the interaction
            interaction.end_interaction(outcomes)

            # Remove from active interactions
            del self.active_interactions[interaction_id]

            # Remove from agent interactions
            for participant_id in interaction.participants:
                if participant_id in self.agent_interactions:
                    self.agent_interactions[participant_id].discard(interaction_id)

            # Add to history
            self.interaction_history.append(interaction)

            # Save final state
            await self._save_interaction(interaction)

            # Create memories for participants
            await self._create_interaction_memory(interaction, "interaction_ended", reason)

            logger.info(f"Interaction {interaction_id} ended by {ender_id}")
            return True

        except Exception as e:
            logger.error(f"Error ending interaction: {e}")
            return False

    async def get_agent_interactions(self, agent_id: UUID) -> List[Interaction]:
        """Get all active interactions for an agent"""
        interaction_ids = self.agent_interactions.get(agent_id, set())
        return [self.active_interactions[iid] for iid in interaction_ids if iid in self.active_interactions]

    async def get_interaction(self, interaction_id: UUID) -> Optional[Interaction]:
        """Get a specific interaction"""
        return self.active_interactions.get(interaction_id)

    async def get_interaction_history(self, agent_id: UUID = None, limit: int = 10) -> List[Dict]:
        """Get interaction history"""
        history = self.interaction_history

        if agent_id:
            history = [i for i in history if agent_id in i.participants]

        # Sort by most recent
        history.sort(key=lambda i: i.ended_at or i.started_at, reverse=True)

        return [i.to_dict() for i in history[:limit]]

    async def analyze_interaction(self, interaction_id: UUID) -> Dict:
        """Analyze an interaction for insights"""
        try:
            interaction = self.active_interactions.get(interaction_id)
            if not interaction:
                # Check history
                interaction = next((i for i in self.interaction_history if i.id == interaction_id), None)

            if not interaction:
                return {}

            # Analyze messages
            message_count = len(interaction.messages)
            participants_count = len(interaction.participants)

            # Count messages per participant
            message_distribution = {}
            for msg in interaction.messages:
                sender = msg["sender_id"]
                message_distribution[sender] = message_distribution.get(sender, 0) + 1

            # Analyze sentiment (simplified)
            positive_words = ["good", "great", "excellent", "happy", "agree", "yes", "thanks"]
            negative_words = ["bad", "terrible", "angry", "disagree", "no", "hate"]

            positive_count = 0
            negative_count = 0
            for msg in interaction.messages:
                content_lower = msg["content"].lower()
                positive_count += sum(1 for word in positive_words if word in content_lower)
                negative_count += sum(1 for word in negative_words if word in content_lower)

            sentiment = "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"

            # Calculate duration
            duration = None
            if interaction.ended_at:
                duration = (interaction.ended_at - interaction.started_at).total_seconds()

            analysis = {
                "interaction_id": str(interaction.id),
                "type": interaction.type.value,
                "participants_count": participants_count,
                "message_count": message_count,
                "message_distribution": message_distribution,
                "sentiment": sentiment,
                "positive_indicators": positive_count,
                "negative_indicators": negative_count,
                "duration_seconds": duration,
                "is_active": interaction.is_active,
                "outcomes": interaction.outcomes
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing interaction: {e}")
            return {}

    async def find_nearby_agents(self, agent_id: UUID, location: Dict, radius: float = 10.0) -> List[UUID]:
        """Find agents near a given location for potential interaction"""
        try:
            # This would integrate with the world engine to find nearby agents
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Error finding nearby agents: {e}")
            return []

    async def _process_message(self, interaction: Interaction, sender_id: UUID, message: str):
        """Process a message and notify other participants"""
        try:
            # Notify other participants
            for participant_id in interaction.participants:
                if participant_id != sender_id:
                    # Create memory for receiver
                    await self._create_message_memory(participant_id, interaction, message, "received", sender_id)

                    # Publish to MQTT for real-time updates
                    await self._publish_message_notification(participant_id, interaction, sender_id, message)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _create_interaction_memory(self, interaction: Interaction, event_type: str, detail: str = None):
        """Create memory for interaction events"""
        try:
            for participant_id in interaction.participants:
                content = f"{event_type}: {interaction.type.value} interaction"
                if detail:
                    content += f" - {detail}"

                memory = Memory(
                    agent_id=participant_id,
                    type=MemoryType.INTERACTION,
                    content=content,
                    importance=0.5,
                    participants=interaction.participants,
                    tags=[interaction.type.value, event_type]
                )

                await self.memory_system.store_memory(memory)

        except Exception as e:
            logger.error(f"Error creating interaction memory: {e}")

    async def _create_message_memory(
        self,
        agent_id: UUID,
        interaction: Interaction,
        message: str,
        direction: str,
        sender_id: UUID = None
    ):
        """Create memory for a message"""
        try:
            if direction == "sent":
                content = f"I said: {message}"
            else:
                sender_str = f"Agent {sender_id}" if sender_id else "Someone"
                content = f"{sender_str} said: {message}"

            memory = Memory(
                agent_id=agent_id,
                type=MemoryType.CONVERSATION,
                content=content,
                importance=0.4,
                participants=interaction.participants,
                tags=["message", direction, interaction.type.value]
            )

            await self.memory_system.store_memory(memory)

        except Exception as e:
            logger.error(f"Error creating message memory: {e}")

    async def _save_interaction(self, interaction: Interaction):
        """Save interaction to Redis"""
        try:
            key = f"interaction:{interaction.id}"
            data = {
                "id": str(interaction.id),
                "type": interaction.type.value,
                "initiator_id": str(interaction.initiator_id),
                "participants": json.dumps([str(p) for p in interaction.participants]),
                "location": json.dumps(interaction.location),
                "context": json.dumps(interaction.context),
                "started_at": interaction.started_at.isoformat(),
                "ended_at": interaction.ended_at.isoformat() if interaction.ended_at else "",
                "messages": json.dumps(interaction.messages),
                "is_active": str(interaction.is_active),
                "outcomes": json.dumps(interaction.outcomes)
            }

            await self.redis.hset(key, mapping=data)

            # Set expiry for completed interactions
            if not interaction.is_active:
                await self.redis.expire(key, 86400)  # Keep for 1 day

        except Exception as e:
            logger.error(f"Error saving interaction: {e}")

    async def _publish_message_notification(self, recipient_id: UUID, interaction: Interaction, sender_id: UUID, message: str):
        """Publish message notification via MQTT"""
        # This would integrate with MQTT client to send real-time notifications
        pass

    def _build_response_prompt(self, agent_id: UUID, interaction: Interaction, context: Dict) -> str:
        """Build a prompt for generating a response"""
        prompt = f"""You are an agent in a {interaction.type.value} interaction.

Recent conversation:
"""
        for msg in context["recent_messages"][-3:]:
            sender = "You" if msg["sender_id"] == str(agent_id) else f"Agent {msg['sender_id'][:8]}"
            prompt += f"{sender}: {msg['content']}\n"

        prompt += f"""
Based on the conversation and your role, generate an appropriate response.
Be concise and stay in character. Your response:"""

        return prompt

    async def _interaction_monitor(self):
        """Monitor interactions for timeouts and cleanup"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                now = datetime.utcnow()
                interactions_to_end = []

                for interaction_id, interaction in self.active_interactions.items():
                    # Check for timeout
                    duration = (now - interaction.started_at).total_seconds()
                    if duration > self.interaction_timeout:
                        interactions_to_end.append((interaction_id, "timeout"))

                    # Check for inactive interactions (no messages for 5 minutes)
                    if interaction.messages:
                        last_message_time = datetime.fromisoformat(interaction.messages[-1]["timestamp"])
                        if (now - last_message_time).total_seconds() > 300:
                            interactions_to_end.append((interaction_id, "inactivity"))

                # End timed out interactions
                for interaction_id, reason in interactions_to_end:
                    interaction = self.active_interactions.get(interaction_id)
                    if interaction:
                        await self.end_interaction(
                            interaction_id,
                            interaction.initiator_id,
                            reason=reason
                        )

            except Exception as e:
                logger.error(f"Error in interaction monitor: {e}")
                await asyncio.sleep(30)

    async def get_interaction_metrics(self) -> Dict:
        """Get interaction system metrics"""
        try:
            total_active = len(self.active_interactions)
            total_historical = len(self.interaction_history)

            # Count by type
            type_distribution = {}
            for interaction in self.active_interactions.values():
                type_distribution[interaction.type.value] = type_distribution.get(interaction.type.value, 0) + 1

            # Average duration of completed interactions
            durations = []
            for interaction in self.interaction_history:
                if interaction.ended_at:
                    duration = (interaction.ended_at - interaction.started_at).total_seconds()
                    durations.append(duration)

            avg_duration = sum(durations) / len(durations) if durations else 0

            metrics = {
                "total_active_interactions": total_active,
                "total_completed_interactions": total_historical,
                "interaction_type_distribution": type_distribution,
                "average_interaction_duration_seconds": avg_duration,
                "agents_in_interactions": len(self.agent_interactions)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting interaction metrics: {e}")
            return {}