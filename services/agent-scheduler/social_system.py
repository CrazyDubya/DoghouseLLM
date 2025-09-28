"""Social relationship tracking system for agents"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships between agents"""
    FRIEND = "friend"
    ACQUAINTANCE = "acquaintance"
    COLLEAGUE = "colleague"
    RIVAL = "rival"
    ROMANTIC = "romantic"
    FAMILY = "family"
    MENTOR = "mentor"
    MENTEE = "mentee"
    BUSINESS_PARTNER = "business_partner"
    CUSTOMER = "customer"
    VENDOR = "vendor"


class RelationshipStatus(Enum):
    """Status of a relationship"""
    PENDING = "pending"  # One-sided, not yet reciprocated
    ACTIVE = "active"    # Mutual relationship
    BROKEN = "broken"    # Ended relationship
    BLOCKED = "blocked"  # One agent blocked the other


class SocialEvent(Enum):
    """Types of social events"""
    FIRST_MEETING = "first_meeting"
    POSITIVE_INTERACTION = "positive_interaction"
    NEGATIVE_INTERACTION = "negative_interaction"
    COLLABORATION = "collaboration"
    CONFLICT = "conflict"
    GIFT_GIVEN = "gift_given"
    FAVOR_DONE = "favor_done"
    BETRAYAL = "betrayal"
    RECONCILIATION = "reconciliation"


class Relationship:
    """Represents a relationship between two agents"""

    def __init__(
        self,
        agent1_id: UUID,
        agent2_id: UUID,
        relationship_type: RelationshipType,
        initiated_by: UUID = None
    ):
        self.id = uuid4()
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.type = relationship_type
        self.status = RelationshipStatus.PENDING if initiated_by else RelationshipStatus.ACTIVE
        self.initiated_by = initiated_by
        self.strength = 0.5  # 0 to 1, represents bond strength
        self.sentiment = 0.0  # -1 (negative) to 1 (positive)
        self.trust = 0.5  # 0 to 1
        self.familiarity = 0.1  # 0 to 1, how well they know each other
        self.created_at = datetime.utcnow()
        self.last_interaction = datetime.utcnow()
        self.interaction_count = 0
        self.history: List[Dict] = []
        self.metadata: Dict = {}

    def update_from_interaction(self, event_type: SocialEvent, impact: float = 0.1):
        """Update relationship based on an interaction"""
        self.last_interaction = datetime.utcnow()
        self.interaction_count += 1

        # Update familiarity (always increases with interaction)
        self.familiarity = min(1.0, self.familiarity + 0.05)

        # Update based on event type
        if event_type == SocialEvent.POSITIVE_INTERACTION:
            self.sentiment = min(1.0, self.sentiment + impact)
            self.strength = min(1.0, self.strength + impact * 0.5)
            self.trust = min(1.0, self.trust + impact * 0.3)

        elif event_type == SocialEvent.NEGATIVE_INTERACTION:
            self.sentiment = max(-1.0, self.sentiment - impact)
            self.strength = max(0.0, self.strength - impact * 0.3)
            self.trust = max(0.0, self.trust - impact * 0.5)

        elif event_type == SocialEvent.COLLABORATION:
            self.sentiment = min(1.0, self.sentiment + impact * 1.5)
            self.strength = min(1.0, self.strength + impact)
            self.trust = min(1.0, self.trust + impact * 0.7)

        elif event_type == SocialEvent.CONFLICT:
            self.sentiment = max(-1.0, self.sentiment - impact * 1.5)
            self.trust = max(0.0, self.trust - impact * 0.8)

        elif event_type == SocialEvent.BETRAYAL:
            self.sentiment = max(-1.0, self.sentiment - impact * 2)
            self.trust = max(0.0, self.trust - impact * 2)
            self.strength = max(0.0, self.strength - impact)

        elif event_type == SocialEvent.GIFT_GIVEN:
            self.sentiment = min(1.0, self.sentiment + impact * 0.7)
            self.strength = min(1.0, self.strength + impact * 0.4)

        # Add to history
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type.value,
            "impact": impact,
            "sentiment_after": self.sentiment,
            "trust_after": self.trust
        })

    def get_relationship_quality(self) -> float:
        """Calculate overall relationship quality (0 to 1)"""
        return (self.strength * 0.3 + (self.sentiment + 1) / 2 * 0.3 +
                self.trust * 0.2 + self.familiarity * 0.2)

    def to_dict(self) -> Dict:
        """Convert relationship to dictionary"""
        return {
            "id": str(self.id),
            "agent1_id": str(self.agent1_id),
            "agent2_id": str(self.agent2_id),
            "type": self.type.value,
            "status": self.status.value,
            "strength": round(self.strength, 2),
            "sentiment": round(self.sentiment, 2),
            "trust": round(self.trust, 2),
            "familiarity": round(self.familiarity, 2),
            "quality": round(self.get_relationship_quality(), 2),
            "created_at": self.created_at.isoformat(),
            "last_interaction": self.last_interaction.isoformat(),
            "interaction_count": self.interaction_count,
            "initiated_by": str(self.initiated_by) if self.initiated_by else None
        }


class SocialNetwork:
    """Manages the social network of all agents"""

    def __init__(self, redis_client, memory_system):
        self.redis = redis_client
        self.memory_system = memory_system

        # Relationship storage: {(agent1_id, agent2_id): Relationship}
        self.relationships: Dict[Tuple[UUID, UUID], Relationship] = {}

        # Agent's relationships: {agent_id: Set[other_agent_ids]}
        self.agent_connections: Dict[UUID, Set[UUID]] = {}

        # Social groups: {group_id: Set[agent_ids]}
        self.social_groups: Dict[UUID, Set[UUID]] = {}

        # Agent reputation in social network
        self.social_reputation: Dict[UUID, float] = {}

        # Configuration
        self.max_relationships_per_agent = 150  # Dunbar's number
        self.relationship_decay_rate = 0.01  # Daily decay without interaction

    async def initialize(self):
        """Initialize the social network"""
        logger.info("Initializing social network...")

        # Load existing relationships from Redis
        await self._load_relationships()

        # Start background tasks
        asyncio.create_task(self._relationship_maintenance())

        logger.info("Social network initialized")

    async def _load_relationships(self):
        """Load relationships from Redis"""
        try:
            rel_keys = await self.redis.keys("relationship:*")

            for key in rel_keys:
                rel_data = await self.redis.hgetall(key)
                if rel_data:
                    relationship = await self._deserialize_relationship(rel_data)
                    if relationship:
                        key_tuple = self._get_relationship_key(
                            relationship.agent1_id,
                            relationship.agent2_id
                        )
                        self.relationships[key_tuple] = relationship

                        # Update agent connections
                        self._add_to_connections(relationship.agent1_id, relationship.agent2_id)

            logger.info(f"Loaded {len(self.relationships)} relationships")

        except Exception as e:
            logger.error(f"Error loading relationships: {e}")

    def _get_relationship_key(self, agent1_id: UUID, agent2_id: UUID) -> Tuple[UUID, UUID]:
        """Get a consistent key for a relationship regardless of order"""
        return (min(agent1_id, agent2_id), max(agent1_id, agent2_id))

    def _add_to_connections(self, agent1_id: UUID, agent2_id: UUID):
        """Add agents to each other's connection sets"""
        if agent1_id not in self.agent_connections:
            self.agent_connections[agent1_id] = set()
        if agent2_id not in self.agent_connections:
            self.agent_connections[agent2_id] = set()

        self.agent_connections[agent1_id].add(agent2_id)
        self.agent_connections[agent2_id].add(agent1_id)

    async def create_relationship(
        self,
        agent1_id: UUID,
        agent2_id: UUID,
        relationship_type: RelationshipType,
        initiated_by: UUID = None,
        initial_sentiment: float = 0.0
    ) -> Optional[Relationship]:
        """Create a new relationship between agents"""
        try:
            # Check if relationship already exists
            key = self._get_relationship_key(agent1_id, agent2_id)
            if key in self.relationships:
                logger.info(f"Relationship already exists between {agent1_id} and {agent2_id}")
                return self.relationships[key]

            # Check relationship limits
            agent1_count = len(self.agent_connections.get(agent1_id, set()))
            agent2_count = len(self.agent_connections.get(agent2_id, set()))

            if agent1_count >= self.max_relationships_per_agent:
                logger.warning(f"Agent {agent1_id} has reached relationship limit")
                return None

            if agent2_count >= self.max_relationships_per_agent:
                logger.warning(f"Agent {agent2_id} has reached relationship limit")
                return None

            # Create relationship
            relationship = Relationship(agent1_id, agent2_id, relationship_type, initiated_by)
            relationship.sentiment = initial_sentiment

            # Store relationship
            self.relationships[key] = relationship
            self._add_to_connections(agent1_id, agent2_id)

            # Save to Redis
            await self._save_relationship(relationship)

            # Create memory for both agents
            await self._create_relationship_memory(
                agent1_id,
                agent2_id,
                f"Formed {relationship_type.value} relationship",
                "relationship_formed"
            )

            logger.info(f"Created {relationship_type.value} relationship between {agent1_id} and {agent2_id}")
            return relationship

        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return None

    async def update_relationship(
        self,
        agent1_id: UUID,
        agent2_id: UUID,
        event_type: SocialEvent,
        impact: float = 0.1
    ) -> bool:
        """Update a relationship based on an interaction"""
        try:
            key = self._get_relationship_key(agent1_id, agent2_id)
            relationship = self.relationships.get(key)

            if not relationship:
                # Create acquaintance relationship if none exists
                relationship = await self.create_relationship(
                    agent1_id,
                    agent2_id,
                    RelationshipType.ACQUAINTANCE
                )

            if relationship:
                # Update relationship
                relationship.update_from_interaction(event_type, impact)

                # Save updated relationship
                await self._save_relationship(relationship)

                # Create memory of the interaction
                await self._create_relationship_memory(
                    agent1_id,
                    agent2_id,
                    f"{event_type.value} with sentiment {relationship.sentiment:.2f}",
                    event_type.value
                )

                # Update social reputation based on interaction
                await self._update_social_reputation(agent1_id, agent2_id, event_type)

                return True

            return False

        except Exception as e:
            logger.error(f"Error updating relationship: {e}")
            return False

    async def get_relationship(self, agent1_id: UUID, agent2_id: UUID) -> Optional[Relationship]:
        """Get relationship between two agents"""
        key = self._get_relationship_key(agent1_id, agent2_id)
        return self.relationships.get(key)

    async def get_agent_relationships(
        self,
        agent_id: UUID,
        relationship_type: RelationshipType = None,
        min_quality: float = 0.0
    ) -> List[Relationship]:
        """Get all relationships for an agent"""
        relationships = []

        for other_agent_id in self.agent_connections.get(agent_id, set()):
            key = self._get_relationship_key(agent_id, other_agent_id)
            relationship = self.relationships.get(key)

            if relationship:
                # Filter by type if specified
                if relationship_type and relationship.type != relationship_type:
                    continue

                # Filter by quality
                if relationship.get_relationship_quality() < min_quality:
                    continue

                relationships.append(relationship)

        # Sort by quality
        relationships.sort(key=lambda r: r.get_relationship_quality(), reverse=True)

        return relationships

    async def get_social_circle(self, agent_id: UUID, max_distance: int = 2) -> Set[UUID]:
        """Get agents within social distance (friend of friends, etc.)"""
        visited = set()
        to_visit = {agent_id}
        social_circle = set()
        current_distance = 0

        while to_visit and current_distance < max_distance:
            next_level = set()

            for current_agent in to_visit:
                if current_agent in visited:
                    continue

                visited.add(current_agent)

                if current_agent != agent_id:
                    social_circle.add(current_agent)

                # Add connections
                connections = self.agent_connections.get(current_agent, set())
                next_level.update(connections - visited)

            to_visit = next_level
            current_distance += 1

        return social_circle

    async def recommend_connections(self, agent_id: UUID, limit: int = 5) -> List[UUID]:
        """Recommend new connections for an agent"""
        try:
            current_connections = self.agent_connections.get(agent_id, set())
            social_circle = await self.get_social_circle(agent_id, max_distance=2)

            # Find potential connections (friends of friends)
            recommendations = []

            for candidate_id in social_circle:
                if candidate_id not in current_connections:
                    # Calculate affinity score
                    mutual_friends = 0
                    for friend_id in self.agent_connections.get(candidate_id, set()):
                        if friend_id in current_connections:
                            mutual_friends += 1

                    if mutual_friends > 0:
                        recommendations.append((candidate_id, mutual_friends))

            # Sort by number of mutual connections
            recommendations.sort(key=lambda x: x[1], reverse=True)

            return [r[0] for r in recommendations[:limit]]

        except Exception as e:
            logger.error(f"Error recommending connections: {e}")
            return []

    async def form_social_group(self, member_ids: List[UUID], group_name: str = None) -> UUID:
        """Form a social group"""
        try:
            group_id = uuid4()
            self.social_groups[group_id] = set(member_ids)

            # Create relationships between all members
            for i, member1 in enumerate(member_ids):
                for member2 in member_ids[i + 1:]:
                    await self.create_relationship(
                        member1,
                        member2,
                        RelationshipType.COLLEAGUE,
                        initial_sentiment=0.3
                    )

            # Save group to Redis
            await self.redis.hset(
                f"social_group:{group_id}",
                mapping={
                    "id": str(group_id),
                    "name": group_name or f"Group {group_id}",
                    "members": json.dumps([str(m) for m in member_ids]),
                    "created_at": datetime.utcnow().isoformat()
                }
            )

            logger.info(f"Formed social group {group_id} with {len(member_ids)} members")
            return group_id

        except Exception as e:
            logger.error(f"Error forming social group: {e}")
            return None

    async def get_social_reputation(self, agent_id: UUID) -> float:
        """Get agent's social reputation"""
        if agent_id not in self.social_reputation:
            # Calculate based on relationships
            relationships = await self.get_agent_relationships(agent_id)

            if not relationships:
                self.social_reputation[agent_id] = 0.5
            else:
                # Average sentiment from all relationships
                total_sentiment = sum(r.sentiment for r in relationships)
                avg_sentiment = total_sentiment / len(relationships)

                # Factor in number of positive relationships
                positive_relationships = sum(1 for r in relationships if r.sentiment > 0.3)
                relationship_bonus = min(0.3, positive_relationships * 0.02)

                self.social_reputation[agent_id] = min(1.0, 0.5 + avg_sentiment * 0.3 + relationship_bonus)

        return self.social_reputation[agent_id]

    async def _update_social_reputation(self, agent1_id: UUID, agent2_id: UUID, event_type: SocialEvent):
        """Update social reputation based on interaction"""
        try:
            # Positive events increase reputation
            if event_type in [SocialEvent.POSITIVE_INTERACTION, SocialEvent.COLLABORATION,
                             SocialEvent.GIFT_GIVEN, SocialEvent.FAVOR_DONE]:
                self.social_reputation[agent1_id] = min(1.0,
                    self.social_reputation.get(agent1_id, 0.5) + 0.01)
                self.social_reputation[agent2_id] = min(1.0,
                    self.social_reputation.get(agent2_id, 0.5) + 0.01)

            # Negative events decrease reputation
            elif event_type in [SocialEvent.CONFLICT, SocialEvent.BETRAYAL]:
                self.social_reputation[agent1_id] = max(0.0,
                    self.social_reputation.get(agent1_id, 0.5) - 0.02)
                self.social_reputation[agent2_id] = max(0.0,
                    self.social_reputation.get(agent2_id, 0.5) - 0.02)

        except Exception as e:
            logger.error(f"Error updating social reputation: {e}")

    async def _create_relationship_memory(
        self,
        agent1_id: UUID,
        agent2_id: UUID,
        content: str,
        event_type: str
    ):
        """Create memory of relationship event"""
        try:
            from packages.shared_types.models import Memory, MemoryType

            # Create memory for both agents
            for agent_id in [agent1_id, agent2_id]:
                other_agent = agent2_id if agent_id == agent1_id else agent1_id

                memory = Memory(
                    agent_id=agent_id,
                    type=MemoryType.SOCIAL,
                    content=f"Relationship with {other_agent}: {content}",
                    importance=0.5,
                    participants=[agent1_id, agent2_id],
                    tags=["relationship", event_type]
                )

                await self.memory_system.store_memory(memory)

        except Exception as e:
            logger.error(f"Error creating relationship memory: {e}")

    async def _save_relationship(self, relationship: Relationship):
        """Save relationship to Redis"""
        try:
            key = f"relationship:{relationship.id}"
            data = {
                "id": str(relationship.id),
                "agent1_id": str(relationship.agent1_id),
                "agent2_id": str(relationship.agent2_id),
                "type": relationship.type.value,
                "status": relationship.status.value,
                "strength": relationship.strength,
                "sentiment": relationship.sentiment,
                "trust": relationship.trust,
                "familiarity": relationship.familiarity,
                "created_at": relationship.created_at.isoformat(),
                "last_interaction": relationship.last_interaction.isoformat(),
                "interaction_count": relationship.interaction_count,
                "history": json.dumps(relationship.history[-10:])  # Keep last 10 events
            }

            await self.redis.hset(key, mapping=data)

        except Exception as e:
            logger.error(f"Error saving relationship: {e}")

    async def _deserialize_relationship(self, data: Dict) -> Optional[Relationship]:
        """Deserialize relationship from Redis"""
        try:
            # Handle bytes from Redis
            if isinstance(data.get(b'id'), bytes):
                data = {k.decode(): v.decode() for k, v in data.items()}

            relationship = Relationship(
                UUID(data["agent1_id"]),
                UUID(data["agent2_id"]),
                RelationshipType(data["type"])
            )

            relationship.id = UUID(data["id"])
            relationship.status = RelationshipStatus(data["status"])
            relationship.strength = float(data["strength"])
            relationship.sentiment = float(data["sentiment"])
            relationship.trust = float(data["trust"])
            relationship.familiarity = float(data["familiarity"])
            relationship.created_at = datetime.fromisoformat(data["created_at"])
            relationship.last_interaction = datetime.fromisoformat(data["last_interaction"])
            relationship.interaction_count = int(data["interaction_count"])

            if data.get("history"):
                relationship.history = json.loads(data["history"])

            return relationship

        except Exception as e:
            logger.error(f"Error deserializing relationship: {e}")
            return None

    async def _relationship_maintenance(self):
        """Background task for relationship decay and maintenance"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run hourly

                now = datetime.utcnow()

                for relationship in self.relationships.values():
                    # Apply decay if no recent interaction
                    time_since_interaction = (now - relationship.last_interaction).total_seconds() / 86400

                    if time_since_interaction > 1:  # More than 1 day
                        decay = self.relationship_decay_rate * time_since_interaction

                        # Decay strength and familiarity slowly
                        relationship.strength = max(0.0, relationship.strength - decay * 0.5)
                        relationship.familiarity = max(0.0, relationship.familiarity - decay * 0.2)

                        # Sentiment slowly returns to neutral
                        if relationship.sentiment > 0:
                            relationship.sentiment = max(0.0, relationship.sentiment - decay)
                        elif relationship.sentiment < 0:
                            relationship.sentiment = min(0.0, relationship.sentiment + decay)

                        await self._save_relationship(relationship)

            except Exception as e:
                logger.error(f"Error in relationship maintenance: {e}")
                await asyncio.sleep(60)

    async def get_network_metrics(self) -> Dict:
        """Get social network metrics"""
        try:
            total_relationships = len(self.relationships)
            active_relationships = sum(1 for r in self.relationships.values()
                                     if r.status == RelationshipStatus.ACTIVE)

            # Calculate average metrics
            if self.relationships:
                avg_strength = sum(r.strength for r in self.relationships.values()) / total_relationships
                avg_sentiment = sum(r.sentiment for r in self.relationships.values()) / total_relationships
                avg_trust = sum(r.trust for r in self.relationships.values()) / total_relationships
            else:
                avg_strength = avg_sentiment = avg_trust = 0

            # Count relationships by type
            type_distribution = {}
            for relationship in self.relationships.values():
                type_distribution[relationship.type.value] = type_distribution.get(relationship.type.value, 0) + 1

            metrics = {
                "total_relationships": total_relationships,
                "active_relationships": active_relationships,
                "total_agents_connected": len(self.agent_connections),
                "average_strength": round(avg_strength, 2),
                "average_sentiment": round(avg_sentiment, 2),
                "average_trust": round(avg_trust, 2),
                "social_groups": len(self.social_groups),
                "relationship_types": type_distribution
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting network metrics: {e}")
            return {}